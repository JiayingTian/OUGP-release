#!/usr/bin/env python3
"""Unified Twitch leave-one-domain-out main-table benchmark.

All methods use the same five-source/one-target fold, clean data, stratified
2:1:1 source splits, hidden dimension 128, full target graph deployment, and
four-seed reproducibility. Methods marked adapted use sparse local adapters so
the benchmark remains runnable on the large disjoint-union source graph.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(path))

from deployed_costs import deployment_flops_metrics, elementwise_weight_pruning_flops_metrics
from ougp.deployment import MaterializedGCNDeployment, deployment_mask_set
from ougp.model import sparse_gcn_mm, symmetric_norm
from ougp.twitch import TWITCH_DOMAINS, combine_twitch_domains, load_twitch_domain
from ougp.tta import ensemble_log_probabilities, propagate_predictions
from tta_energy_shift_probe import build_parser as build_tta_parser
from run_tta_smoke import accuracy, set_seed, train_source_variant
from method_efficiency import gated_controller, train_exp121_controller
from training_flops import source_training_flops, tta_training_flops


def load_twitch_source(root: Path, target_domain: str, split_seed: int):
    source_domains = [domain for domain in TWITCH_DOMAINS if domain != target_domain]
    source_graphs = [
        (domain, load_twitch_domain(root, domain, split_seed=split_seed))
        for domain in source_domains
    ]
    return combine_twitch_domains(source_graphs)


def load_twitch_target(root: Path, target_domain: str, split_seed: int):
    return load_twitch_domain(root, target_domain, split_seed=split_seed)


METHODS = (
    "dense",
    "ugs",
    "cgp",
    "unifews",
    "ace_glt",
    "dspar",
    "sgcn",
    "adaptivegcn",
    "lsp_p",
    "neuralsparse",
    "grasp",
    "gcnp",
    "ougp",
    "ougp_tta",
)
JOINT_WEIGHT_METHODS = {"ugs", "cgp", "unifews", "ace_glt"}
GRAPH_ONLY_METHODS = {"dspar", "sgcn", "adaptivegcn", "lsp_p", "neuralsparse"}
CHANNEL_METHODS = {"grasp", "gcnp"}


def add_self_loops(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(num_nodes, device=edge_index.device)
    loops = torch.stack([nodes, nodes], dim=0)
    return torch.cat([edge_index, loops], dim=1), torch.cat(
        [edge_weight, torch.ones(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)], dim=0
    )


def normalized_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    full_index, full_weight = add_self_loops(edge_index, edge_weight, num_nodes)
    return full_index, symmetric_norm(full_index, full_weight, num_nodes)


class FixedMaskedGCN(nn.Module):
    """Two-layer sparse GCN with fixed edge and element-wise weight masks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_index1: torch.Tensor,
        edge_weight1: torch.Tensor,
        edge_index2: torch.Tensor,
        edge_weight2: torch.Tensor,
        num_nodes: int,
        weight_mask1: torch.Tensor | None = None,
        weight_mask2: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        edge_index1, edge_weight1 = normalized_edges(edge_index1, edge_weight1, num_nodes)
        edge_index2, edge_weight2 = normalized_edges(edge_index2, edge_weight2, num_nodes)
        self.register_buffer("edge_index1", edge_index1)
        self.register_buffer("edge_weight1", edge_weight1)
        self.register_buffer("edge_index2", edge_index2)
        self.register_buffer("edge_weight2", edge_weight2)
        self.num_nodes = int(num_nodes)
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.register_buffer(
            "weight_mask1",
            torch.ones_like(self.lin1.weight) if weight_mask1 is None else weight_mask1.detach().float().clone(),
        )
        self.register_buffer(
            "weight_mask2",
            torch.ones_like(self.lin2.weight) if weight_mask2 is None else weight_mask2.detach().float().clone(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = sparse_gcn_mm(self.edge_index1, self.edge_weight1, x, self.num_nodes)
        h = F.relu(F.linear(h, self.lin1.weight * self.weight_mask1))
        h = sparse_gcn_mm(self.edge_index2, self.edge_weight2, h, self.num_nodes)
        return F.linear(h, self.lin2.weight * self.weight_mask2)


def train_fixed(
    graph,
    device: torch.device,
    edge_index: torch.Tensor,
    weight_mask1: torch.Tensor | None,
    weight_mask2: torch.Tensor | None,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[FixedMaskedGCN, float, float, int]:
    model = FixedMaskedGCN(
        graph.num_features,
        hidden_dim,
        graph.num_classes,
        edge_index,
        torch.ones(edge_index.size(1), device=device),
        edge_index,
        torch.ones(edge_index.size(1), device=device),
        graph.num_nodes,
        weight_mask1,
        weight_mask2,
    ).to(device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x, y = graph.x.to(device), graph.y.to(device)
    train_mask, val_mask, test_mask = (
        graph.train_mask.to(device),
        graph.val_mask.to(device),
        graph.test_mask.to(device),
    )
    best_val = -1.0
    best_test = 0.0
    best_epoch = -1
    best_state = None
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x)
            current_val = accuracy(logits, y, val_mask)
            if current_val > best_val:
                best_val = current_val
                best_test = accuracy(logits, y, test_mask)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Fixed GCN training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    return model, best_val, best_test, best_epoch


def copy_weights(source: FixedMaskedGCN, target: FixedMaskedGCN) -> None:
    with torch.no_grad():
        target.lin1.weight.copy_(source.lin1.weight)
        target.lin2.weight.copy_(source.lin2.weight)


def exact_topk(scores: torch.Tensor, keep_rate: float) -> torch.Tensor:
    flat = scores.detach().float().flatten()
    keep = max(1, min(flat.numel(), int(round(flat.numel() * float(keep_rate)))))
    indices = torch.topk(flat, keep, largest=True, sorted=False).indices
    result = torch.zeros_like(flat)
    result[indices] = 1.0
    return result.reshape_as(scores).to(dtype=scores.dtype)


def edge_scores(graph, method: str, device: torch.device) -> torch.Tensor:
    edge_index = graph.edge_index.to(device)
    src, dst = edge_index
    x = graph.x.to(device).float()
    similarity = F.cosine_similarity(x[src], x[dst], dim=-1, eps=1e-8).abs()
    degree = torch.bincount(src, minlength=graph.num_nodes).float().to(device)
    degree_score = (degree[src] + degree[dst]).log1p()
    if method in {"ugs", "dspar"}:
        return degree_score
    if method in {"cgp", "unifews", "sgcn", "adaptivegcn", "neuralsparse"}:
        return similarity
    if method == "lsp_p":
        labels = graph.y.to(device)
        known = graph.train_mask.to(device)[src] & graph.train_mask.to(device)[dst]
        agreement = (labels[src] == labels[dst]).float()
        return 0.75 * agreement + 0.25 * similarity
    if method == "ace_glt":
        return similarity * (1.0 + degree_score)
    return torch.ones(edge_index.size(1), device=device)


def gradient_weight_scores(model: FixedMaskedGCN, graph, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits = model(graph.x.to(device))
    loss = F.cross_entropy(logits[graph.train_mask.to(device)], graph.y.to(device)[graph.train_mask.to(device)])
    loss.backward()
    return (model.lin1.weight.detach().abs() * model.lin1.weight.grad.detach().abs(),
            model.lin2.weight.detach().abs() * model.lin2.weight.grad.detach().abs())


def weight_masks(model: FixedMaskedGCN, graph, method: str, keep_rate: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if method in {"grasp", "gcnp"}:
        score1, score2 = gradient_weight_scores(model, graph, device)
    else:
        score1, score2 = model.lin1.weight.detach().abs(), model.lin2.weight.detach().abs()
    if method in {"grasp", "gcnp"}:
        channel_score = score1.sum(dim=1) + score2.sum(dim=0)
        kept_channels = torch.topk(channel_score, max(1, int(round(channel_score.numel() * keep_rate)))).indices
        mask1 = torch.zeros_like(score1)
        mask2 = torch.zeros_like(score2)
        mask1[kept_channels, :] = 1.0
        mask2[:, kept_channels] = 1.0
        return mask1, mask2
    all_scores = torch.cat([score1.flatten(), score2.flatten()])
    hard = exact_topk(all_scores, keep_rate)
    first_size = score1.numel()
    return hard[:first_size].reshape_as(score1), hard[first_size:].reshape_as(score2)


def measure_latency(forward_fn, device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(max(0, int(warmup))):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000.0 / max(1, int(repeats))
    return {"target_latency_ms": float(elapsed), "latency_repeats": int(repeats)}


def target_cost(
    target,
    original_hidden_dim: int,
    deployed_hidden_dim: int,
    parameter_mask: torch.Tensor | None,
    method: str,
) -> dict[str, float | str]:
    non_loop_edges = int((target.edge_index[0] != target.edge_index[1]).sum().item())
    if parameter_mask is not None and method not in {"grasp", "gcnp"}:
        retained = int(parameter_mask.sum().item())
        total = int(parameter_mask.numel())
        return elementwise_weight_pruning_flops_metrics(
            num_nodes=target.num_nodes,
            input_dim=target.num_features,
            hidden_dim=original_hidden_dim,
            output_dim=target.num_classes,
            original_edge_count=non_loop_edges,
            deployed_edge_count=non_loop_edges,
            retained_parameter_count=retained,
            total_parameter_count=total,
            hidden_layers=1,
        )
    return deployment_flops_metrics(
        num_nodes=target.num_nodes,
        input_dim=target.num_features,
        original_hidden_dim=original_hidden_dim,
        deployed_hidden_dim=deployed_hidden_dim,
        output_dim=target.num_classes,
        original_edge_count=non_loop_edges,
        deployed_edge_count=non_loop_edges,
        hidden_layers=1,
        flops_reduction_source="full target graph with materialized baseline structure",
    )


def run_standard(
    args,
    source,
    target,
    method: str,
    device: torch.device,
    target_loader=None,
) -> dict[str, Any]:
    def ensure_target():
        nonlocal target
        if target is None:
            if target_loader is None:
                raise RuntimeError("Twitch target loader is required for target inference.")
            target = target_loader()
        return target

    keep_rate = 1.0 - float(args.graph_sparsity)
    source_graph_sparsity = 0.0
    pretrain_epochs = max(1, int(args.epochs) // 2)
    retrain_epochs = max(1, int(args.epochs) - pretrain_epochs)
    dense_model, dense_val, dense_test, dense_epoch = train_fixed(
        source,
        device,
        source.edge_index.to(device),
        None,
        None,
        args.hidden_dim,
        pretrain_epochs if method != "dense" else args.epochs,
        args.lr,
        args.weight_decay,
    )
    if method == "dense":
        target = ensure_target()
        target_model = FixedMaskedGCN(
            target.num_features,
            args.hidden_dim,
            target.num_classes,
            target.edge_index.to(device),
            torch.ones(target.edge_index.size(1), device=device),
            target.edge_index.to(device),
            torch.ones(target.edge_index.size(1), device=device),
            target.num_nodes,
        ).to(device)
        copy_weights(dense_model, target_model)
        source_model = dense_model
        source_graph_sparsity = 0.0
        parameter_mask = None
        selected_hidden_dim = args.hidden_dim
        source_val = dense_val
        best_epoch = dense_epoch
    else:
        edge_mask = exact_topk(edge_scores(source, method, device), keep_rate) if method not in CHANNEL_METHODS else torch.ones(source.edge_index.size(1), device=device)
        source_edge_index = source.edge_index.to(device)[:, edge_mask.bool()]
        if method in JOINT_WEIGHT_METHODS or method in CHANNEL_METHODS:
            mask1, mask2 = weight_masks(dense_model, source, method, keep_rate, device)
        else:
            mask1 = torch.ones_like(dense_model.lin1.weight)
            mask2 = torch.ones_like(dense_model.lin2.weight)
        if method in CHANNEL_METHODS:
            selected = mask1.sum(dim=1) > 0
            selected_hidden_dim = int(selected.sum().item())
            compact_source = FixedMaskedGCN(
                source.num_features,
                selected_hidden_dim,
                source.num_classes,
                source.edge_index.to(device),
                torch.ones(source.edge_index.size(1), device=device),
                source.edge_index.to(device),
                torch.ones(source.edge_index.size(1), device=device),
                source.num_nodes,
            ).to(device)
            with torch.no_grad():
                compact_source.lin1.weight.copy_(dense_model.lin1.weight[selected, :])
                compact_source.lin2.weight.copy_(dense_model.lin2.weight[:, selected])
            source_model, source_val, source_test, best_epoch = train_fixed(
                source,
                device,
                source.edge_index.to(device),
                None,
                None,
                selected_hidden_dim,
                retrain_epochs,
                args.lr,
                args.weight_decay,
                init_state=compact_source.state_dict(),
            )
            target = ensure_target()
            target_model = FixedMaskedGCN(
                target.num_features,
                selected_hidden_dim,
                target.num_classes,
                target.edge_index.to(device),
                torch.ones(target.edge_index.size(1), device=device),
                target.edge_index.to(device),
                torch.ones(target.edge_index.size(1), device=device),
                target.num_nodes,
            ).to(device)
            copy_weights(source_model, target_model)
            parameter_mask = mask1
        else:
            source_model, source_val, source_test, best_epoch = train_fixed(
                source,
                device,
                source_edge_index,
                mask1,
                mask2,
                args.hidden_dim,
                retrain_epochs,
                args.lr,
                args.weight_decay,
            )
            target = ensure_target()
            target_model = FixedMaskedGCN(
                target.num_features,
                args.hidden_dim,
                target.num_classes,
                target.edge_index.to(device),
                torch.ones(target.edge_index.size(1), device=device),
                target.edge_index.to(device),
                torch.ones(target.edge_index.size(1), device=device),
                target.num_nodes,
                mask1,
                mask2,
            ).to(device)
            copy_weights(source_model, target_model)
            selected_hidden_dim = args.hidden_dim
            parameter_mask = torch.cat([mask1.flatten(), mask2.flatten()])
            source_graph_sparsity = 1.0 - float(edge_mask.mean().item())
        target_model.eval()
        source_model.eval()
    source_x, source_y = source.x.to(device), source.y.to(device)
    target_x, target_y = target.x.to(device), target.y.to(device)
    with torch.no_grad():
        source_logits = source_model(source_x)
        target_logits = target_model(target_x)
    target_acc = accuracy(target_logits, target_y, target.test_mask.to(device))
    source_acc = accuracy(source_logits, source_y, source.test_mask.to(device))
    source_model.eval()
    target_model.eval()
    target_parameter_mask = None if method == "dense" else (
        torch.cat([source_model.weight_mask1.flatten(), source_model.weight_mask2.flatten()])
        if method not in CHANNEL_METHODS else source_model.weight_mask1
    )
    cost = target_cost(
        target,
        args.hidden_dim,
        selected_hidden_dim,
        target_parameter_mask,
        method,
    )
    latency = measure_latency(lambda: target_model(target_x), device, args.latency_warmup, args.latency_repeats)
    result = {
        "method": method,
        "method_label": {
            "dense": "Dense",
            "ugs": "UGS",
            "cgp": "CGP",
            "unifews": "UniFews",
            "ace_glt": "ACE-GLT",
            "dspar": "DSpar",
            "sgcn": "SGCN",
            "adaptivegcn": "AdaptiveGCN",
            "lsp_p": "LSP-P",
            "neuralsparse": "NeuralSparse",
            "grasp": "GraSP",
            "gcnp": "GCNP",
        }.get(method, method.upper()),
        "source_domains": list(args.source_domains),
        "target_domain": args.target_domain,
        "source_num_nodes": source.num_nodes,
        "source_num_edges": int(source.edge_index.size(1)),
        "target_num_nodes": target.num_nodes,
        "target_num_edges": int(target.edge_index.size(1)),
        "input_dim": source.num_features,
        "hidden_dim": args.hidden_dim,
        "materialized_hidden_dim": selected_hidden_dim,
        "source_test_accuracy": source_acc,
        "target_accuracy": target_acc,
        "source_best_val_accuracy": source_val,
        "source_best_epoch": best_epoch if method != "dense" else dense_epoch,
        "source_graph_sparsity": float(source_graph_sparsity),
        "target_graph_sparsity": 0.0,
        "target_parameter_sparsity": (
            0.0
            if method == "dense"
            else 1.0 - selected_hidden_dim / args.hidden_dim
            if method in CHANNEL_METHODS
            else 1.0
            - float(
                torch.cat(
                    [target_model.weight_mask1.flatten(), target_model.weight_mask2.flatten()]
                ).mean().item()
            )
        ),
        "deployment_mode": "full target graph",
        "implementation_status": "sparse benchmark adapter",
        "provenance": "project-local sparse implementation for Twitch LODO comparison",
        **latency,
        **cost,
    }
    return result


def build_184_args(args, out_dir: Path) -> argparse.Namespace:
    parser = build_tta_parser()
    tta = parser.parse_args([])
    values = {
        "experiment_name": args.experiment_name,
        "out_dir": str(out_dir),
        "dataset": "twitch_lodo",
        "data_root": str(args.data_root),
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_selection": "best_val",
        "warmup_epochs": args.warmup_epochs,
        "hidden_dim": args.hidden_dim,
        "backbone": "gcn",
        "num_gnn_layers": 2,
        "graph_sparsity": 0.30,
        "param_sparsity": 0.30,
        "param_pruning_granularity": "weight",
        "param_policy_granularity": "channel",
        "use_hidden_coupling": True,
        "hidden_coupling_mix_graph": 0.30,
        "hidden_coupling_mix_param": 0.30,
        "memory_rank": 8,
        "graph_memory_layout": "multi",
        "param_memory_layout": "multi",
        "graph_score_init": "topofeat",
        "param_score_init": "magnitude",
        "lr": 0.05,
        "weight_decay": 0.0005,
        "sparsity_lambda": 0.08,
        "training_mask_mode": "periodic_ste",
        "training_mask_hardening_mode": "quantile_binary",
        "mask_hardening_interval": 1,
        "hardening_mode": "quantile_binary",
        "deployment_graph_policy": "full",
        "tta_train_mask_ratio": 0.0,
        "tta_train_shift_scope": "train_mask",
        "tta_adaptation_nodes": "train_val",
        "tta_epochs": args.tta_epochs,
        "tta_log_every": 1,
        "tta_lr": 0.00001,
        "tta_loss_mode": "unsupervised",
        "tta_lambda_energy": 1.0,
        "tta_energy_negative_mask_ratio": 0.0,
        "tta_energy_clean_margin": 0.10,
        "tta_lambda_consistency": 0.10,
        "tta_consistency_mask_ratio": 0.0,
        "tta_lambda_anchor": 0.08,
        "tta_lambda_teacher": 0.8,
        "tta_lambda_state": 0.0,
        "tta_state_mode": "stateless",
        "tta_affine_mode": "direct",
        "tta_energy_reduction": "train_mask",
        "tta_consistency_reduction": "train_mask",
        "tta_gamma_a": 0.5,
        "tta_gamma_b": 0.5,
        "tta_lambda_entropy": 0.05,
        "tta_center_delta_a": True,
        "tta_normalize_delta_a": True,
        "tta_center_delta_b": True,
        "tta_normalize_delta_b": True,
        "tta_delta_normalization_floor": 0.0001,
        "tta_joint_energy_gate": True,
        "tta_joint_energy_gate_train_min": 0.0,
        "tta_gate_learn_mode": "node",
        "tta_test_mode": "energy_gated",
        "energy_gate_mode": "signed_z_tanh",
        "energy_gate_min": 0.0,
        "energy_gate_max": 1.0,
        "energy_gate_center_z": 0.0005,
        "energy_gate_temperature": 0.7,
        "energy_gate_polarity": 1.0,
        "tta_output_mlp": False,
        "tta_prediction_propagation": False,
        "profile_inference_repeats": 0,
        "profile_deployment_repeats": 0,
        "source_checkpoint_policy": "off",
        "source_checkpoint_key": "",
        "source_checkpoint_root": "",
    }
    for key, value in values.items():
        setattr(tta, key, value)
    tta.seeds = [args.seed]
    tta.mask_ratio = 0.0
    tta.original_num_nodes = 0
    tta.original_num_edges = 0
    return tta


def build_245_args(args, out_dir: Path) -> argparse.Namespace:
    parser = build_tta_parser()
    tta = parser.parse_args([])
    values = {
        "experiment_name": args.experiment_name,
        "out_dir": str(out_dir),
        "dataset": "twitch_lodo",
        "data_root": str(args.data_root),
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_selection": "best_val",
        "warmup_epochs": args.warmup_epochs,
        "hidden_dim": args.hidden_dim,
        "backbone": "gcn",
        "num_gnn_layers": 2,
        "graph_sparsity": 0.30,
        "param_sparsity": 0.30,
        "param_pruning_granularity": "weight",
        "param_policy_granularity": "channel",
        "use_hidden_coupling": True,
        "hidden_coupling_mix_graph": 0.30,
        "hidden_coupling_mix_param": 0.30,
        "memory_rank": 8,
        "graph_memory_layout": "multi",
        "param_memory_layout": "multi",
        "graph_score_init": "topofeat",
        "param_score_init": "magnitude",
        "lr": 0.05,
        "weight_decay": 0.0005,
        "sparsity_lambda": 0.08,
        "training_mask_mode": "periodic_ste",
        "training_mask_hardening_mode": "quantile_binary",
        "mask_hardening_interval": 1,
        "hardening_mode": "quantile_binary",
        "deployment_graph_policy": "full",
        "tta_train_mask_ratio": 0.0,
        "tta_train_shift_scope": "train_mask",
        "tta_adaptation_nodes": "train_val",
        "tta_epochs": 100,
        "tta_log_every": 10,
        "tta_checkpoint_selection": "best_val",
        "tta_strength_selection": "best_val",
        "tta_strength_candidates": [0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0],
        "tta_lr": 0.0005,
        "tta_weight_decay": 0.0,
        "tta_loss_mode": "supervised",
        "tta_lambda_logits": 1.0,
        "tta_lambda_energy": 0.10,
        "tta_energy_negative_mask_ratio": 0.0,
        "tta_energy_clean_margin": 0.10,
        "tta_lambda_consistency": 0.0,
        "tta_consistency_mask_ratio": 0.0,
        "tta_lambda_anchor": 0.0001,
        "tta_lambda_teacher": 0.20,
        "tta_lambda_entropy": 0.0,
        "tta_lambda_state": 0.0,
        "tta_state_mode": "stateless",
        "tta_affine_mode": "direct",
        "tta_gamma_a": 0.10,
        "tta_gamma_b": 0.05,
        "tta_logit_affine": False,
        "tta_logit_residual": False,
        "tta_output_mlp": True,
        "tta_output_mlp_rank": 8,
        "tta_gamma_output_mlp": 2.75,
        "tta_output_gate_mode": "signed",
        "tta_center_delta_a": True,
        "tta_normalize_delta_a": True,
        "tta_center_delta_b": True,
        "tta_normalize_delta_b": True,
        "tta_delta_normalization_floor": 0.0001,
        "tta_joint_energy_gate": True,
        "tta_joint_energy_gate_train_min": 0.0,
        "tta_gate_learn_mode": "mean",
        "tta_test_mode": "energy_gated",
        "energy_gate_mode": "compatibility_rbf",
        "energy_gate_min": 0.0,
        "energy_gate_max": 1.0,
        "energy_gate_center_z": 0.0005,
        "energy_gate_temperature": 0.3,
        "energy_gate_polarity": 1.0,
        "tta_target_reset_controller": False,
        "tta_target_adaptation_epochs": 0,
        "tta_target_gate_scale": 1.0,
        "tta_prediction_propagation": False,
        "tta_propagation_mode": "logspace_signed",
        "tta_propagation_alpha_candidates": [-0.15, 0.0, 0.025],
        "tta_propagation_step_candidates": [1],
        "tta_propagation_label_anchor": True,
        "tta_propagation_label_anchor_positive_only": True,
        "profile_inference_repeats": 0,
        "profile_deployment_repeats": 0,
        "source_checkpoint_policy": "off",
        "source_checkpoint_key": "",
        "source_checkpoint_root": "",
    }
    for key, value in values.items():
        setattr(tta, key, value)
    tta.seeds = [args.seed]
    tta.mask_ratio = 0.0
    tta.original_num_nodes = 0
    tta.original_num_edges = 0
    return tta


def apply_tta_prediction_propagation(
    logits: torch.Tensor,
    target,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    if not bool(getattr(args, "tta_prediction_propagation", False)):
        return logits, {
            "enabled": False,
            "mode": str(getattr(args, "tta_propagation_mode", "logspace_signed")),
            "alpha": 0.0,
            "steps": 1,
            "ensemble_topk": 0,
            "label_anchor": bool(getattr(args, "tta_propagation_label_anchor", False)),
            "label_anchor_positive_only": bool(
                getattr(args, "tta_propagation_label_anchor_positive_only", False)
            ),
            "relative_validation": False,
            "baseline_validation_accuracy": None,
            "best_validation_accuracy": None,
            "validation": [],
            "components": [],
        }

    alpha_candidates = [
        float(value)
        for value in getattr(args, "tta_propagation_alpha_candidates", [-0.15, 0.0, 0.025])
    ]
    step_candidates = [
        int(value)
        for value in getattr(args, "tta_propagation_step_candidates", [1])
    ]
    ensemble_topk = int(getattr(args, "tta_propagation_ensemble_topk", 1))
    if ensemble_topk <= 0:
        raise ValueError("tta_propagation_ensemble_topk must be positive.")
    mode = str(getattr(args, "tta_propagation_mode", "logspace_signed"))
    label_anchor = bool(getattr(args, "tta_propagation_label_anchor", False))
    anchor_positive_only = bool(
        getattr(args, "tta_propagation_label_anchor_positive_only", False)
    )
    relative_validation = bool(getattr(args, "tta_propagation_relative_validation", False))
    y = target.y.to(device)
    val_mask = target.val_mask.to(device)
    train_mask = target.train_mask.to(device)
    selection_mask = val_mask
    selection_anchor_mask = train_mask
    final_anchor_mask = train_mask | val_mask
    baseline_value = accuracy(logits, y, selection_mask)
    validation_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for alpha in alpha_candidates:
        for steps in step_candidates:
            component_anchor = label_anchor and (not anchor_positive_only or float(alpha) > 0.0)
            candidate = propagate_predictions(
                logits,
                target.edge_index.to(device),
                float(alpha),
                int(steps),
                mode=mode,
                labels=y if component_anchor else None,
                anchor_mask=selection_anchor_mask if component_anchor else None,
            )
            value = accuracy(candidate, y, selection_mask)
            validation_rows.append(
                {
                    "alpha": float(alpha),
                    "steps": int(steps),
                    "validation_accuracy": value,
                    "selection_score": value - baseline_value if relative_validation else value,
                    "validation_gain": value - baseline_value,
                    "label_anchor": component_anchor,
                }
            )
            candidate_rows.append(
                {
                    "selection_score": value - baseline_value if relative_validation else value,
                    "validation_accuracy": value,
                    "alpha": float(alpha),
                    "steps": int(steps),
                    "label_anchor": component_anchor,
                }
            )
    candidate_rows.sort(key=lambda row: row["selection_score"], reverse=True)
    selected = candidate_rows[:ensemble_topk]
    selected_logits = [
        propagate_predictions(
            logits,
            target.edge_index.to(device),
            float(row["alpha"]),
            int(row["steps"]),
            mode=mode,
            labels=y if bool(row["label_anchor"]) else None,
            anchor_mask=final_anchor_mask if bool(row["label_anchor"]) else None,
        )
        for row in selected
    ]
    propagated = ensemble_log_probabilities(selected_logits)
    summary = {
        "enabled": True,
        "mode": mode,
        "alpha": selected[0]["alpha"],
        "steps": selected[0]["steps"],
        "ensemble_topk": len(selected),
        "label_anchor": label_anchor,
        "label_anchor_positive_only": anchor_positive_only,
        "relative_validation": relative_validation,
        "baseline_validation_accuracy": baseline_value,
        "best_validation_accuracy": selected[0]["validation_accuracy"],
        "validation": validation_rows,
        "components": [
            {
                "alpha": row["alpha"],
                "steps": row["steps"],
                "validation_accuracy": row["validation_accuracy"],
                "selection_score": row["selection_score"],
                "label_anchor": row["label_anchor"],
            }
            for row in selected
        ],
    }
    return propagated, summary


def run_ougp_tta(
    args,
    source,
    target,
    device: torch.device,
    target_loader=None,
) -> dict[str, Any]:
    tta_args = build_245_args(args, Path(args.out_dir))
    tta_args.original_num_nodes = source.num_nodes
    tta_args.original_num_edges = int(source.edge_index.size(1))
    set_seed(args.seed)
    ougp_model, source_result, source_history = train_source_variant(
        tta_args, source, device, variant="ougp"
    )
    source_x, source_y = source.x.to(device), source.y.to(device)
    masks = deployment_mask_set(ougp_model, mode="quantile_binary", temperature=getattr(tta_args, "temp_end", 0.5))
    hard_masks = ougp_model.get_training_hard_masks()
    parameter_values = hard_masks[1] if hard_masks is not None else masks.parameter.values
    parameter_support = parameter_values.detach().bool()
    source_graph_values = torch.ones_like(masks.graph.values)
    source_deployment = MaterializedGCNDeployment(
        ougp_model,
        source_graph_values,
        parameter_values,
        source_x.dtype,
        graph_support=torch.ones(source.edge_index.size(1), device=device, dtype=torch.bool),
        param_support=parameter_support,
    ).to(device)
    source_deployment.eval()
    for parameter in source_deployment.parameters():
        parameter.requires_grad_(False)
    controller, source_energy, tta_training = train_exp121_controller(
        tta_args,
        source_deployment,
        source_x,
        source_y,
        source.train_mask.to(device),
        source.val_mask.to(device),
    )
    if target is None:
        if target_loader is None:
            raise RuntimeError("Twitch target loader is required for target inference.")
        target = target_loader()
    target_x, target_y = target.x.to(device), target.y.to(device)
    target_deployment = MaterializedGCNDeployment(
        ougp_model,
        torch.ones(target.edge_index.size(1), device=device),
        parameter_values,
        target_x.dtype,
        edge_index_override=target.edge_index.to(device),
        num_nodes_override=target.num_nodes,
        graph_support=torch.ones(target.edge_index.size(1), device=device, dtype=torch.bool),
        param_support=parameter_support,
    ).to(device)
    target_deployment.eval()
    for module in (source_deployment, target_deployment):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    runtime_controller, _ = gated_controller(
        controller,
        target_deployment,
        target_x,
        torch.ones(target.num_nodes, device=device, dtype=torch.bool),
        source_energy,
        tta_args,
    )
    runtime_controller.eval()
    with torch.no_grad():
        source_logits = source_deployment(source_x)
        frozen_target_logits = target_deployment(target_x)
        tta_logits = target_deployment.forward_with_stateless_direct_tta(target_x, runtime_controller)
    tta_logits, propagation_summary = apply_tta_prediction_propagation(
        tta_logits,
        target,
        tta_args,
        device,
    )
    source_acc = accuracy(source_logits, source_y, source.test_mask.to(device))
    frozen_acc = accuracy(frozen_target_logits, target_y, target.test_mask.to(device))
    target_acc = accuracy(tta_logits, target_y, target.test_mask.to(device))
    latency = measure_latency(
        lambda: target_deployment.forward_with_stateless_direct_tta(target_x, runtime_controller),
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    non_loop_edges = int((target.edge_index[0] != target.edge_index[1]).sum().item())
    base_cost = elementwise_weight_pruning_flops_metrics(
        num_nodes=target.num_nodes,
        input_dim=target.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=target.num_classes,
        original_edge_count=non_loop_edges,
        deployed_edge_count=non_loop_edges,
        retained_parameter_count=int(parameter_support.sum().item()),
        total_parameter_count=int(parameter_support.numel()),
        hidden_layers=1,
    )
    tta_ops = float(2 * target.num_nodes * args.hidden_dim + 3 * target.num_nodes * target.num_classes)
    tta_cost = dict(base_cost)
    tta_cost["deployed_flops"] = float(base_cost["deployed_flops"] + tta_ops)
    tta_cost["flops_reduction"] = 1.0 - tta_cost["deployed_flops"] / float(base_cost["dense_reference_flops"])
    tta_cost["flops_reduction_source"] = "full target graph + quantile-binary weight entries + stateless EXP245 affine gate"
    return {
        "method": "ougp_tta",
        "method_label": "OUGP+TTA",
        "source_domains": list(args.source_domains),
        "target_domain": args.target_domain,
        "source_num_nodes": source.num_nodes,
        "source_num_edges": int(source.edge_index.size(1)),
        "target_num_nodes": target.num_nodes,
        "target_num_edges": int(target.edge_index.size(1)),
        "input_dim": source.num_features,
        "hidden_dim": args.hidden_dim,
        "materialized_hidden_dim": args.hidden_dim,
        "source_test_accuracy": source_acc,
        "target_accuracy": target_acc,
        "frozen_ougp_target_accuracy": frozen_acc,
        "tta_gain_vs_frozen_ougp": target_acc - frozen_acc,
        "deployment_mode": "full target graph",
        "parameter_granularity": "weight-entry",
        "parameter_sparsity": 1.0 - float(parameter_support.float().mean().item()),
        "graph_sparsity": 0.0,
        "hardening_mode": "quantile_binary",
        "tta_config_source": "build_245_args in scripts/twitch_lodo_experiment.py",
        "tta_state_mode": "stateless",
        "tta_test_mode": "energy_gated",
        "energy_gate_mode": str(tta_args.energy_gate_mode),
        "target_gate_scope": "all_target_nodes",
        "tta_training_summary": tta_training,
        "tta_prediction_propagation_summary": propagation_summary,
        "source_result": source_result,
        "source_history_epochs": len(source_history),
        "implementation_status": "current EXP245 OUGP+TTA path",
        **latency,
        **tta_cost,
    }


def run_ougp(
    args,
    source,
    target,
    device: torch.device,
    target_loader=None,
) -> dict[str, Any]:
    """Run the frozen OUGP deployment branch with the EXP184 hardening rule."""

    tta_args = build_184_args(args, Path(args.out_dir))
    tta_args.original_num_nodes = source.num_nodes
    tta_args.original_num_edges = int(source.edge_index.size(1))
    set_seed(args.seed)
    ougp_model, source_result, source_history = train_source_variant(
        tta_args, source, device, variant="ougp"
    )
    source_x, source_y = source.x.to(device), source.y.to(device)
    masks = deployment_mask_set(
        ougp_model,
        mode="quantile_binary",
        temperature=getattr(tta_args, "temp_end", 0.5),
    )
    hard_masks = ougp_model.get_training_hard_masks()
    parameter_values = hard_masks[1] if hard_masks is not None else masks.parameter.values
    parameter_support = parameter_values.detach().bool()
    source_deployment = MaterializedGCNDeployment(
        ougp_model,
        torch.ones(source.edge_index.size(1), device=device),
        parameter_values,
        source_x.dtype,
        graph_support=torch.ones(source.edge_index.size(1), device=device, dtype=torch.bool),
        param_support=parameter_support,
    ).to(device)
    if target is None:
        if target_loader is None:
            raise RuntimeError("Twitch target loader is required for target inference.")
        target = target_loader()
    target_x, target_y = target.x.to(device), target.y.to(device)
    target_deployment = MaterializedGCNDeployment(
        ougp_model,
        torch.ones(target.edge_index.size(1), device=device),
        parameter_values,
        target_x.dtype,
        edge_index_override=target.edge_index.to(device),
        num_nodes_override=target.num_nodes,
        graph_support=torch.ones(target.edge_index.size(1), device=device, dtype=torch.bool),
        param_support=parameter_support,
    ).to(device)
    source_deployment.eval()
    target_deployment.eval()
    for module in (source_deployment, target_deployment):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    with torch.no_grad():
        source_logits = source_deployment(source_x)
        target_logits = target_deployment(target_x)
    source_acc = accuracy(source_logits, source_y, source.test_mask.to(device))
    target_acc = accuracy(target_logits, target_y, target.test_mask.to(device))
    latency = measure_latency(
        lambda: target_deployment(target_x),
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    non_loop_edges = int((target.edge_index[0] != target.edge_index[1]).sum().item())
    cost = elementwise_weight_pruning_flops_metrics(
        num_nodes=target.num_nodes,
        input_dim=target.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=target.num_classes,
        original_edge_count=non_loop_edges,
        deployed_edge_count=non_loop_edges,
        retained_parameter_count=int(parameter_support.sum().item()),
        total_parameter_count=int(parameter_support.numel()),
        hidden_layers=1,
    )
    return {
        "method": "ougp",
        "method_label": "OUGP",
        "source_domains": list(args.source_domains),
        "target_domain": args.target_domain,
        "source_num_nodes": source.num_nodes,
        "source_num_edges": int(source.edge_index.size(1)),
        "target_num_nodes": target.num_nodes,
        "target_num_edges": int(target.edge_index.size(1)),
        "input_dim": source.num_features,
        "hidden_dim": args.hidden_dim,
        "materialized_hidden_dim": args.hidden_dim,
        "source_test_accuracy": source_acc,
        "target_accuracy": target_acc,
        "source_best_val_accuracy": source_result.get(
            "best_val_accuracy", source_result.get("best_val_acc")
        ),
        "source_graph_sparsity": 0.0,
        "target_graph_sparsity": 0.0,
        "parameter_granularity": "weight-entry",
        "parameter_sparsity": 1.0 - float(parameter_support.float().mean().item()),
        "deployment_mode": "full target graph",
        "hardening_mode": "quantile_binary",
        "source_history_epochs": len(source_history),
        "source_result": source_result,
        "implementation_status": "current OUGP deployment path",
        **latency,
        **cost,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="EXP257_TWITCH_LODO_MAIN_TABLE_DIM128")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--target-domain", choices=TWITCH_DOMAINS, required=True)
    parser.add_argument("--source-domains", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--tta-epochs", type=int, default=60)
    parser.add_argument("--graph-sparsity", type=float, default=0.30)
    parser.add_argument("--param-sparsity", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.source_domains = [domain for domain in TWITCH_DOMAINS if domain != args.target_domain]
    if args.hidden_dim not in {32, 128}:
        raise ValueError("This experiment supports hidden_dim=32 or hidden_dim=128.")
    if args.method in {"ougp", "ougp_tta"}:
        device = torch.device(args.device)
        source, _ = load_twitch_source(args.data_root, args.target_domain, split_seed=args.seed)
        target_loader = lambda: load_twitch_target(
            args.data_root, args.target_domain, split_seed=args.seed
        )
        if args.method == "ougp_tta":
            result = run_ougp_tta(args, source, None, device, target_loader=target_loader)
        else:
            result = run_ougp(args, source, None, device, target_loader=target_loader)
    else:
        set_seed(args.seed)
        device = torch.device(args.device)
        source, _ = load_twitch_source(args.data_root, args.target_domain, split_seed=args.seed)
        target_loader = lambda: load_twitch_target(
            args.data_root, args.target_domain, split_seed=args.seed
        )
        result = run_standard(
            args, source, None, args.method, device, target_loader=target_loader
        )
    result.update(
        {
            "experiment": args.experiment_name,
            "seed": args.seed,
            "split_mode": "stratified_2_1_1",
            "split_ratio": "2:1:1",
            "data_protocol": "clean",
            "target_graph_policy": "full_target_graph",
            "target_loaded_after_source_selection": True,
            "target_evaluated_after_source_selection": True,
            "target_evaluation_count": 1,
            "source_domains": args.source_domains,
            "target_domain": args.target_domain,
            "config": vars(args),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
