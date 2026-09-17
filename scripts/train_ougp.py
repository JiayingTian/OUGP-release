"""Run the first OUGP citation-network case study.

The default is intentionally small enough for CPU sanity runs:

    PYTHONPATH=src python scripts/run_case_study.py --dataset cora --epochs 80 --seeds 0

It saves per-run JSON plus aggregate CSV/Markdown files under
`results/ougp_case_study/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ougp.data import CitationGraph, apply_split_mode, load_graph_dataset
from ougp.deployment import MaterializedGCNDeployment, deployment_mask_set, deployment_masks
from ougp.model import OUGPConfig, OUGPGCN, sparse_gcn_mm
from ougp.trace import PruningTraceRecorder
from deployed_costs import deployment_flops_metrics


BACKBONES = ("gcn", "sage", "gat", "deepgcn")
GRAPH_SCORE_INITS = ("constant", "random", "degree", "similarity", "topofeat")
PARAM_SCORE_INITS = ("constant", "random", "magnitude")

VARIANTS = {
    "dense": dict(use_graph_pruning=False, use_param_pruning=False, use_memory=False, use_cross=False),
    "graph_only": dict(use_graph_pruning=True, use_param_pruning=False, use_memory=True, use_cross=False),
    "param_only": dict(use_graph_pruning=False, use_param_pruning=True, use_memory=True, use_cross=False),
    "dual_static": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=False, use_cross=False),
    "random_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="random",
        param_score_init="random",
        freeze_pruning_scores=True,
    ),
    "degree_magnitude_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    ),
    "similarity_magnitude_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    ),
    "dropedge_static": dict(
        use_graph_pruning=True,
        use_param_pruning=False,
        use_memory=False,
        use_cross=False,
        graph_score_init="random",
        freeze_pruning_scores=True,
    ),
    "neuralsparse_graph_dynamic": dict(
        use_graph_pruning=True,
        use_param_pruning=False,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        freeze_pruning_scores=False,
    ),
    "neuralsparse_dual_dynamic": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        param_score_init="magnitude",
        freeze_pruning_scores=False,
    ),
    "dspar_graph_static": dict(
        use_graph_pruning=True,
        use_param_pruning=False,
        use_memory=False,
        use_cross=False,
        graph_score_init="topofeat",
        freeze_pruning_scores=True,
    ),
    "dspar_dual_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="topofeat",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    ),
    "magnitude_param_static": dict(
        use_graph_pruning=False,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    ),
    "snip_param_static": dict(
        use_graph_pruning=False,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        freeze_pruning_scores=True,
        gradient_param_init="snip",
    ),
    "grasp_param_static": dict(
        use_graph_pruning=False,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        freeze_pruning_scores=True,
        gradient_param_init="grasp",
    ),
    "degree_gradient_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        freeze_pruning_scores=True,
        gradient_param_init="snip",
    ),
    "similarity_gradient_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        freeze_pruning_scores=True,
        gradient_param_init="snip",
    ),
    "degree_grasp_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        freeze_pruning_scores=True,
        gradient_param_init="grasp",
    ),
    "similarity_grasp_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        freeze_pruning_scores=True,
        gradient_param_init="grasp",
    ),
    "lottery_param_static": dict(
        use_graph_pruning=False,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        freeze_pruning_scores=True,
        lottery_param_init=True,
    ),
    "degree_lottery_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        freeze_pruning_scores=True,
        lottery_param_init=True,
    ),
    "similarity_lottery_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        freeze_pruning_scores=True,
        lottery_param_init=True,
    ),
    "rigl_param_dynamic": dict(
        use_graph_pruning=False,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        freeze_pruning_scores=True,
        rigl_param_update=True,
    ),
    "degree_rigl_dynamic": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        freeze_pruning_scores=True,
        rigl_param_update=True,
    ),
    "similarity_rigl_dynamic": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        freeze_pruning_scores=True,
        rigl_param_update=True,
    ),
    "ace_eagles_unified_dynamic": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="topofeat",
        param_score_init="magnitude",
        freeze_pruning_scores=False,
        unified_score_update=True,
        variant_budget_lambda=0.05,
    ),
    "serial_degree_magnitude_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
        serial_pruning=True,
    ),
    "ougp_no_cross": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=True, use_cross=False),
    "ougp": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=True, use_cross=True),
}


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
    return float(allocated), float(reserved)


def timed_section_start(device: torch.device) -> float:
    sync_if_cuda(device)
    return time.perf_counter()


def timed_section_end(device: torch.device, start: float) -> float:
    sync_if_cuda(device)
    return time.perf_counter() - start

VARIANT_PRIVATE_KEYS = {
    "gradient_param_init",
    "lottery_param_init",
    "rigl_param_update",
    "serial_pruning",
    "unified_score_update",
    "variant_budget_lambda",
}


def build_ougp_cfg_kwargs(args: argparse.Namespace, data: CitationGraph, edge_index: torch.Tensor, seed: int, variant: str) -> dict:
    param_sparsity = getattr(args, "param_sparsity", None)
    if param_sparsity is None:
        param_sparsity = getattr(args, "channel_sparsity")
    cfg_kwargs = dict(
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=data.num_classes,
        num_nodes=data.num_nodes,
        num_edges=edge_index.size(1),
        num_gnn_layers=args.num_gnn_layers,
        memory_rank=args.memory_rank,
        graph_target_keep=1.0 - args.graph_sparsity,
        param_target_keep=1.0 - param_sparsity,
        graph_gamma=args.graph_gamma,
        param_gamma=args.param_gamma,
        graph_score_scale_decay=args.graph_score_scale_decay,
        graph_score_scale_min=args.graph_score_scale_min,
        graph_score_scale_max=args.graph_score_scale_max,
        graph_correction_clip=args.graph_correction_clip,
        param_score_scale_decay=args.param_score_scale_decay,
        param_score_scale_min=args.param_score_scale_min,
        param_score_scale_max=args.param_score_scale_max,
        param_correction_clip=args.param_correction_clip,
        cross_gamma=args.cross_gamma,
        use_hidden_coupling=args.use_hidden_coupling,
        hidden_coupling_mix_graph=args.hidden_coupling_mix_graph,
        hidden_coupling_mix_param=args.hidden_coupling_mix_param,
        hidden_coupling_interval=args.hidden_coupling_interval,
        hidden_coupling_start_epoch=args.hidden_coupling_start_epoch,
        hidden_coupling_layer_norm_weight=args.hidden_coupling_layer_norm_weight,
        hidden_coupling_interaction_weight=args.hidden_coupling_interaction_weight,
        hidden_coupling_relation_weight=args.hidden_coupling_relation_weight,
        hidden_coupling_param_damage_weight=args.hidden_coupling_param_damage_weight,
        hidden_coupling_graph_damage_weight=args.hidden_coupling_graph_damage_weight,
        write_beta=args.write_beta,
        write_lambda=args.write_lambda,
        event_gamma=args.event_gamma,
        event_beta=args.event_beta,
        event_decay=args.event_decay,
        event_top_k=args.event_top_k,
        recall_gamma=args.recall_gamma,
        recall_beta=args.recall_beta,
        recall_decay=args.recall_decay,
        recall_top_k=args.recall_top_k,
        use_steering_memory=args.use_steering_memory,
        steer_gamma=args.steer_gamma,
        steer_beta=args.steer_beta,
        steer_lambda=args.steer_lambda,
        hard_masks=args.hard_eval,
        backbone=args.backbone,
        budget_target=args.budget_target,
        memory_write_mode=args.memory_write_mode,
        graph_memory_granularity=args.graph_memory_granularity,
        graph_memory_layout=args.graph_memory_layout,
        use_graph_full_branch=args.use_graph_full_branch,
        use_graph_grad_branch=args.use_graph_grad_branch,
        use_graph_branch_gates=args.use_graph_branch_gates,
        param_memory_layout=args.param_memory_layout,
        param_pruning_granularity=args.param_pruning_granularity,
        param_policy_granularity=args.param_policy_granularity,
        graph_score_init=args.graph_score_init,
        param_score_init=args.param_score_init,
        freeze_pruning_scores=args.freeze_pruning_scores,
        seed=seed,
    )
    variant_options = dict(VARIANTS[variant])
    for key in VARIANT_PRIVATE_KEYS:
        variant_options.pop(key, None)
    cfg_kwargs.update(variant_options)
    return cfg_kwargs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def snapshot_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone a deployment checkpoint without retaining GPU storage references."""

    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


@torch.no_grad()
def evaluate_metric(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, task_type: str) -> float:
    if task_type == "multilabel":
        y_true = y[mask].detach().cpu().numpy()
        y_score = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        return float(roc_auc_score(y_true, y_score, average="macro"))
    return accuracy(logits, y, mask)


@torch.no_grad()
def evaluate_hard_deployment_checkpoint(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    task_type: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluate the final hard-deployment protocol without changing training gradients."""

    was_training = model.training
    original_cfg = model.cfg
    final_cfg = OUGPConfig(
        **{
            **asdict(original_cfg),
            "graph_target_keep": 1.0 - float(args.graph_sparsity),
            "param_target_keep": 1.0 - float(args.param_sparsity),
        }
    )
    model.cfg = final_cfg
    try:
        masks = deployment_mask_set(
            model,
            mode=str(getattr(args, "hardening_mode", "quantile_binary")),
            temperature=float(args.temp_end),
        )
    finally:
        model.cfg = original_cfg
    if str(getattr(args, "deployment_graph_policy", "hardened")) == "full":
        graph_values = torch.ones_like(masks.graph.values)
        graph_support = torch.ones_like(masks.graph.support, dtype=torch.bool)
    else:
        graph_values = masks.graph.values
        graph_support = masks.graph.support
    deployment = MaterializedGCNDeployment(
        model,
        graph_values,
        masks.parameter.values,
        x.dtype,
        graph_support=graph_support,
        param_support=masks.parameter.support,
    ).to(x.device)
    deployment.eval()
    logits = deployment(x)
    if was_training:
        model.train()
    return {
        "hard_train_acc": evaluate_metric(logits, y, train_mask, task_type),
        "hard_val_acc": evaluate_metric(logits, y, val_mask, task_type),
        "hard_test_acc": evaluate_metric(logits, y, test_mask, task_type),
        "hard_graph_sparsity": 1.0 - float(graph_support.float().mean().item()),
        "hard_parameter_sparsity": 1.0 - masks.parameter.realized_keep_rate,
    }


@torch.no_grad()
def profile_deployment_latency(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    test_mask: torch.Tensor,
    task_type: str,
    temperature: float,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    """Profile online, fixed-mask, and materialized deployment inference latency."""

    if repeats <= 0:
        return {}
    device = x.device
    was_training = model.training
    model.eval()
    graph_mask, param_mask = deployment_masks(model)
    graph_mask = graph_mask.to(device=device, dtype=x.dtype)
    param_mask = param_mask.to(device=device, dtype=x.dtype)

    def time_forward(fn) -> tuple[float, torch.Tensor]:
        last_logits = None
        for _ in range(max(0, warmup)):
            last_logits = fn()
        sync_if_cuda(device)
        start = time.perf_counter()
        for _ in range(repeats):
            last_logits = fn()
        sync_if_cuda(device)
        assert last_logits is not None
        return (time.perf_counter() - start) / float(repeats), last_logits

    online_latency, online_logits = time_forward(lambda: model(x, temperature=temperature)[0])
    fixed_latency, fixed_logits = time_forward(lambda: model(x, temperature=temperature, fixed_masks=(graph_mask, param_mask))[0])

    materialized_latency = 0.0
    materialized_test_metric = 0.0
    materialized_nodes_per_sec = 0.0
    materialized_edges_per_sec = 0.0
    materialized_build_sec = 0.0
    materialized_kept_edges = 0.0
    materialized_kept_edges_with_self_loops = 0.0
    materialized_kept_channels = 0.0
    if model.cfg.backbone == "gcn":
        build_start = timed_section_start(device)
        deployment_model = MaterializedGCNDeployment(model, graph_mask, param_mask, x.dtype).to(device)
        deployment_model.eval()
        materialized_build_sec = timed_section_end(device, build_start)
        materialized_kept_edges = float(deployment_model.materialized_edge_count)
        materialized_kept_edges_with_self_loops = float(deployment_model.materialized_edge_count_with_self_loops)
        materialized_kept_channels = float(deployment_model.materialized_channel_count)
        materialized_latency, materialized_logits = time_forward(lambda: deployment_model(x))
        materialized_test_metric = evaluate_metric(materialized_logits, y, test_mask, task_type)
        materialized_nodes_per_sec = float(x.size(0) / max(materialized_latency, 1e-12))
        materialized_edges_per_sec = float(materialized_kept_edges_with_self_loops / max(materialized_latency, 1e-12))

    kept_edges = materialized_kept_edges if materialized_kept_edges > 0 else float(graph_mask.detach().float().sum().item())
    kept_channels = materialized_kept_channels if materialized_kept_channels > 0 else float(param_mask.detach().float().sum().item())
    if was_training:
        model.train()
    return {
        "online_inference_latency_sec": online_latency,
        "online_inference_test_metric": evaluate_metric(online_logits, y, test_mask, task_type),
        "fixed_mask_inference_latency_sec": fixed_latency,
        "fixed_mask_inference_test_metric": evaluate_metric(fixed_logits, y, test_mask, task_type),
        "fixed_mask_nodes_per_sec": float(x.size(0) / max(fixed_latency, 1e-12)),
        "fixed_mask_edges_per_sec": float(graph_mask.numel() / max(fixed_latency, 1e-12)),
        "materialized_sparse_inference_latency_sec": materialized_latency,
        "materialized_sparse_inference_test_metric": materialized_test_metric,
        "materialized_sparse_nodes_per_sec": materialized_nodes_per_sec,
        "materialized_sparse_edges_per_sec": materialized_edges_per_sec,
        "materialized_build_sec": materialized_build_sec,
        "materialized_kept_edges": kept_edges,
        "materialized_kept_edges_with_self_loops": materialized_kept_edges_with_self_loops,
        "materialized_kept_channels": kept_channels,
        "materialized_graph_sparsity": 1.0 - kept_edges / max(float(graph_mask.numel()), 1.0),
        "materialized_channel_sparsity": 1.0 - kept_channels / max(float(param_mask.numel()), 1.0),
        "fixed_mask_speedup_vs_online": online_latency / max(fixed_latency, 1e-12),
        "materialized_speedup_vs_online": online_latency / max(materialized_latency, 1e-12),
        "materialized_speedup_vs_fixed_mask": fixed_latency / max(materialized_latency, 1e-12),
    }


def temperature_at(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = epoch / (epochs - 1)
    return start * (end / start) ** ratio


def target_keep_at(epoch: int, epochs: int, warmup_epochs: int, target_keep: float) -> float:
    if epoch < warmup_epochs:
        return 1.0
    span = max(1, epochs - warmup_epochs)
    ratio = min(1.0, (epoch - warmup_epochs + 1) / span)
    return 1.0 - ratio * (1.0 - target_keep)


def serial_param_keep_at(epoch: int, epochs: int, warmup_epochs: int, target_keep: float) -> float:
    """Delay parameter pruning until graph pruning has had a dedicated phase."""

    serial_warmup = warmup_epochs + max(1, (epochs - warmup_epochs) // 2)
    return target_keep_at(epoch, epochs, serial_warmup, target_keep)


def periodic_prune_rate(
    completed_epoch: int, epochs: int, interval: int, target_sparsity: float
) -> float:
    if interval <= 0:
        raise ValueError("mask hardening interval must be positive.")
    total_events = max(1, (epochs + interval - 1) // interval)
    completed_events = min(total_events, completed_epoch // interval)
    return float(target_sparsity) * float(completed_events) / float(total_events)


@torch.no_grad()
def current_epoch_hard_masks(
    model: OUGPGCN,
    args: argparse.Namespace,
    completed_epoch: int,
    *,
    full_graph: bool,
    temperature: float | None = None,
):
    """Harden the latest scores at the pruning budget reached by this epoch."""

    graph_sparsity = periodic_prune_rate(
        completed_epoch, args.epochs, args.mask_hardening_interval, args.graph_sparsity
    )
    param_sparsity = periodic_prune_rate(
        completed_epoch, args.epochs, args.mask_hardening_interval, args.param_sparsity
    )
    original_cfg = model.cfg
    model.cfg = OUGPConfig(
        **{
            **asdict(original_cfg),
            "graph_target_keep": 1.0 if full_graph else 1.0 - graph_sparsity,
            "param_target_keep": 1.0 - param_sparsity,
        }
    )
    try:
        masks = deployment_mask_set(
            model,
            mode=str(args.training_mask_hardening_mode),
            temperature=float(
                temperature
                if temperature is not None
                else temperature_at(
                    max(0, completed_epoch - 1),
                    args.epochs,
                    args.temp_start,
                    args.temp_end,
                )
            ),
        )
    finally:
        model.cfg = original_cfg
    return masks, graph_sparsity, param_sparsity


@torch.no_grad()
def refresh_periodic_training_masks(
    model: OUGPGCN, args: argparse.Namespace, completed_epoch: int
) -> dict[str, float]:
    was_training = model.training
    model.clear_training_hard_masks()
    masks, graph_sparsity, param_sparsity = current_epoch_hard_masks(
        model,
        args,
        completed_epoch,
        full_graph=False,
    )
    model.set_training_hard_masks(masks.graph.values, masks.parameter.values)
    if was_training:
        model.train()
    return {
        "mask_hardening_refreshed": 1.0,
        "training_hard_graph_sparsity": 1.0 - masks.graph.realized_keep_rate,
        "training_hard_parameter_sparsity": 1.0 - masks.parameter.realized_keep_rate,
    }


@torch.no_grad()
def evaluate_current_epoch_hard_mask(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    task_type: str,
    args: argparse.Namespace,
    completed_epoch: int,
    temperature: float,
) -> dict[str, float]:
    masks, _, param_sparsity = current_epoch_hard_masks(
        model,
        args,
        completed_epoch,
        full_graph=True,
        temperature=temperature,
    )
    param_values = masks.parameter.values.to(device=x.device, dtype=x.dtype)
    graph_values = torch.ones(model.cfg.num_edges, device=x.device, dtype=x.dtype)
    deployment = MaterializedGCNDeployment(
        model,
        graph_values,
        param_values,
        x.dtype,
        graph_support=torch.ones_like(graph_values, dtype=torch.bool),
        param_support=param_values.detach().bool(),
    ).to(x.device)
    deployment.eval()
    logits = deployment(x)
    return {
        "hard_train_acc": evaluate_metric(logits, y, train_mask, task_type),
        "hard_val_acc": evaluate_metric(logits, y, val_mask, task_type),
        "hard_test_acc": evaluate_metric(logits, y, test_mask, task_type),
        "hard_graph_sparsity": 0.0,
        "hard_parameter_sparsity": 1.0 - float(param_values.float().mean().item()),
        "validation_hard_mask_current_epoch": 1.0,
        "validation_hard_mask_target_parameter_sparsity": param_sparsity,
    }


def channel_snip_scores(model: OUGPGCN) -> torch.Tensor:
    """Aggregate |w * grad| saliency into one score per hidden channel."""

    def row_score(layer: torch.nn.Linear) -> torch.Tensor:
        if layer.weight.grad is None:
            return torch.zeros(layer.weight.size(0), device=layer.weight.device)
        return (layer.weight.detach() * layer.weight.grad.detach()).abs().sum(dim=1)

    def col_score(layer: torch.nn.Linear) -> torch.Tensor:
        if layer.weight.grad is None:
            return torch.zeros(layer.weight.size(1), device=layer.weight.device)
        return (layer.weight.detach() * layer.weight.grad.detach()).abs().sum(dim=0)

    scores = row_score(model.lin1) + col_score(model.lin2)
    if model.cfg.backbone == "sage":
        if model.sage_lin1_neigh is None or model.sage_lin2_neigh is None:
            raise RuntimeError("GraphSAGE layers were not initialized.")
        scores = scores + row_score(model.sage_lin1_neigh) + col_score(model.sage_lin2_neigh)
    elif model.cfg.backbone == "gat":
        if model.gat_attn1_src is not None and model.gat_attn1_src.grad is not None:
            scores = scores + (model.gat_attn1_src.detach() * model.gat_attn1_src.grad.detach()).abs()
        if model.gat_attn1_dst is not None and model.gat_attn1_dst.grad is not None:
            scores = scores + (model.gat_attn1_dst.detach() * model.gat_attn1_dst.grad.detach()).abs()
    elif model.cfg.backbone == "deepgcn":
        for layer in model.deep_hidden_lins:
            scores = scores + row_score(layer) + col_score(layer)
    return scores


def channel_grasp_scores(model: OUGPGCN, second_grads: dict[int, torch.Tensor | None]) -> torch.Tensor:
    """Aggregate GraSP-style second-order saliency into one score per channel."""

    def saliency(param: torch.Tensor) -> torch.Tensor:
        grad = second_grads.get(id(param))
        if grad is None:
            return torch.zeros_like(param)
        return (-param.detach() * grad.detach()).abs()

    def row_score(layer: torch.nn.Linear) -> torch.Tensor:
        return saliency(layer.weight).sum(dim=1)

    def col_score(layer: torch.nn.Linear) -> torch.Tensor:
        return saliency(layer.weight).sum(dim=0)

    scores = row_score(model.lin1) + col_score(model.lin2)
    if model.cfg.backbone == "sage":
        if model.sage_lin1_neigh is None or model.sage_lin2_neigh is None:
            raise RuntimeError("GraphSAGE layers were not initialized.")
        scores = scores + row_score(model.sage_lin1_neigh) + col_score(model.sage_lin2_neigh)
    elif model.cfg.backbone == "gat":
        if model.gat_attn1_src is not None:
            scores = scores + saliency(model.gat_attn1_src)
        if model.gat_attn1_dst is not None:
            scores = scores + saliency(model.gat_attn1_dst)
    elif model.cfg.backbone == "deepgcn":
        for layer in model.deep_hidden_lins:
            scores = scores + row_score(layer) + col_score(layer)
    return scores


def initialize_param_scores_from_gradient(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    task_type: str,
    method: str = "snip",
) -> None:
    """Initialize channel pruning scores with early gradient saliency."""

    original_cfg = model.cfg
    model.cfg = OUGPConfig(
        **{
            **asdict(original_cfg),
            "use_graph_pruning": False,
            "use_param_pruning": False,
            "use_memory": False,
            "use_cross": False,
            "graph_target_keep": 1.0,
            "param_target_keep": 1.0,
        }
    )
    model.zero_grad(set_to_none=True)
    logits, _ = model(x, temperature=1.0)
    if task_type == "multilabel":
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
    else:
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
    if method == "snip":
        loss.backward()
        scores = channel_snip_scores(model)
    elif method == "grasp":
        params = [param for param in model.parameters() if param.requires_grad]
        first_grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grad_energy = sum(grad.pow(2).sum() for grad in first_grads if grad is not None)
        second = torch.autograd.grad(grad_energy, params, allow_unused=True)
        scores = channel_grasp_scores(model, {id(param): grad for param, grad in zip(params, second)})
    else:
        raise ValueError(f"Unknown gradient parameter init method {method!r}.")
    with torch.no_grad():
        model.param_logits.copy_(model.normalized_init_scores(scores).to(model.param_logits.device, model.param_logits.dtype))
    model.zero_grad(set_to_none=True)
    model.cfg = original_cfg


def initialize_param_scores_from_lottery_pretrain(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    task_type: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    """Lottery-ticket-style channel score: pretrain dense, score, then rewind."""

    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_cfg = model.cfg
    dense_cfg = OUGPConfig(
        **{
            **asdict(original_cfg),
            "use_graph_pruning": False,
            "use_param_pruning": False,
            "use_memory": False,
            "use_cross": False,
            "graph_target_keep": 1.0,
            "param_target_keep": 1.0,
        }
    )
    model.cfg = dense_cfg

    if epochs > 0:
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, temperature=1.0)
            if task_type == "multilabel":
                loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
            else:
                loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        scores = model.initial_param_scores("magnitude")
    model.load_state_dict(initial_state, strict=False)
    model.cfg = original_cfg
    model.reset_memory()
    with torch.no_grad():
        model.param_logits.copy_(scores.to(model.param_logits.device, model.param_logits.dtype))
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def update_param_scores_from_rigl(model: OUGPGCN, gradient_alpha: float) -> dict[str, float]:
    """Refresh channel scores with a RigL-like magnitude/gradient signal."""

    grad_scores = channel_snip_scores(model)
    mag_scores = model.initial_param_scores("magnitude")
    grad_unit = model.normalized_init_scores(grad_scores).to(mag_scores.device, mag_scores.dtype)
    alpha = float(max(0.0, min(1.0, gradient_alpha)))
    scores = (1.0 - alpha) * mag_scores + alpha * grad_unit
    scores = model.normalized_init_scores(scores).to(model.param_logits.device, model.param_logits.dtype)
    model.param_logits.copy_(scores)
    return {
        "rigl_param_updates": 1.0,
        "rigl_param_score_mean": float(scores.detach().float().mean().item()),
        "rigl_param_score_std": float(scores.detach().float().std(unbiased=False).item()),
    }


@torch.no_grad()
def update_scores_from_unified_dual(model: OUGPGCN, gradient_alpha: float) -> dict[str, float]:
    """ACE/EAGLES-inspired centralized refresh of graph and channel scores."""

    graph_base = model.initial_edge_scores("topofeat")
    if model.edge_logits.grad is None:
        graph_grad = torch.zeros_like(graph_base)
    else:
        graph_grad = (model.edge_logits.detach() * model.edge_logits.grad.detach()).abs()
    graph_grad = model.normalized_init_scores(graph_grad).to(graph_base.device, graph_base.dtype)

    param_base = model.initial_param_scores("magnitude")
    param_grad = channel_snip_scores(model)
    param_grad = model.normalized_init_scores(param_grad).to(param_base.device, param_base.dtype)

    alpha = float(max(0.0, min(1.0, gradient_alpha)))
    graph_scores = model.normalized_init_scores((1.0 - alpha) * graph_base + alpha * graph_grad)
    param_scores = model.normalized_init_scores((1.0 - alpha) * param_base + alpha * param_grad)
    model.edge_logits.copy_(graph_scores.to(model.edge_logits.device, model.edge_logits.dtype))
    model.param_logits.copy_(param_scores.to(model.param_logits.device, model.param_logits.dtype))
    return {
        "unified_score_updates": 1.0,
        "unified_graph_score_std": float(graph_scores.detach().float().std(unbiased=False).item()),
        "unified_param_score_std": float(param_scores.detach().float().std(unbiased=False).item()),
    }


@torch.no_grad()
def recall_diagnostics(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    task_type: str,
    temperature: float,
    effective_eps: float,
) -> dict[str, float]:
    """Compare evaluation masks with recall disabled vs. enabled."""

    original_cfg = model.cfg
    original_recall_gamma = float(original_cfg.recall_gamma)

    model.cfg = OUGPConfig(**{**asdict(original_cfg), "recall_gamma": 0.0})
    logits_before, before_mask_stats = model(x, temperature=temperature)
    before_param_mask = (
        model.last_param_mask.detach().float().clone()
        if model.last_param_mask is not None
        else torch.ones(original_cfg.hidden_dim, device=x.device)
    )
    val_before = evaluate_metric(logits_before, y, val_mask, task_type)
    test_before = evaluate_metric(logits_before, y, test_mask, task_type)

    model.cfg = original_cfg
    logits_after, after_mask_stats = model(x, temperature=temperature)
    after_param_mask = (
        model.last_param_mask.detach().float().clone()
        if model.last_param_mask is not None
        else torch.ones(original_cfg.hidden_dim, device=x.device)
    )
    val_after = evaluate_metric(logits_after, y, val_mask, task_type)
    test_after = evaluate_metric(logits_after, y, test_mask, task_type)

    recall_bias = model.param_memory.recall_bias.detach().float().to(after_param_mask.device)
    if recall_bias.numel() != after_param_mask.numel() or original_recall_gamma == 0.0:
        recalled = torch.zeros_like(after_param_mask, dtype=torch.bool)
    else:
        recalled = recall_bias > 0
    effective_recovered = recalled & (after_param_mask > before_param_mask + effective_eps)

    return {
        "recall_diag_param_sparsity": float(1.0 - after_mask_stats["param_keep"]),
        "recall_diag_dropped_parameters": float((1.0 - after_param_mask).sum().item()),
        "recall_diag_hard_dropped_parameters": float((after_param_mask < 0.5).sum().item()),
        "recall_diag_recalled_parameters": float(recalled.sum().item()),
        "recall_diag_effective_recovered_parameters": float(effective_recovered.sum().item()),
        "recall_diag_val_acc_before_recall": val_before,
        "recall_diag_val_acc_after_recall": val_after,
        "recall_diag_test_acc_before_recall": test_before,
        "recall_diag_test_acc_after_recall": test_after,
        "recall_diag_param_keep_before_recall": float(before_mask_stats["param_keep"]),
        "recall_diag_param_keep_after_recall": float(after_mask_stats["param_keep"]),
    }


def run_one(
    args: argparse.Namespace,
    dataset,
    variant: str,
    seed: int,
    out_dir: Path,
    return_artifacts: bool = False,
) -> dict[str, float | int | str] | tuple[dict[str, float | int | str], OUGPGCN, list[dict[str, float | int | str]]]:
    set_seed(seed)
    device = torch.device(args.device)
    data = dataset
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    cfg_kwargs = dict(
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=data.num_classes,
        num_nodes=data.num_nodes,
        num_edges=edge_index.size(1),
        num_gnn_layers=args.num_gnn_layers,
        memory_rank=args.memory_rank,
        graph_target_keep=1.0 - args.graph_sparsity,
        param_target_keep=1.0 - args.param_sparsity,
        graph_gamma=args.graph_gamma,
        param_gamma=args.param_gamma,
        graph_score_scale_decay=args.graph_score_scale_decay,
        graph_score_scale_min=args.graph_score_scale_min,
        graph_score_scale_max=args.graph_score_scale_max,
        graph_correction_clip=args.graph_correction_clip,
        param_score_scale_decay=args.param_score_scale_decay,
        param_score_scale_min=args.param_score_scale_min,
        param_score_scale_max=args.param_score_scale_max,
        param_correction_clip=args.param_correction_clip,
        cross_gamma=args.cross_gamma,
        use_hidden_coupling=args.use_hidden_coupling,
        hidden_coupling_mix_graph=args.hidden_coupling_mix_graph,
        hidden_coupling_mix_param=args.hidden_coupling_mix_param,
        hidden_coupling_interval=args.hidden_coupling_interval,
        hidden_coupling_start_epoch=args.hidden_coupling_start_epoch,
        hidden_coupling_layer_norm_weight=args.hidden_coupling_layer_norm_weight,
        hidden_coupling_interaction_weight=args.hidden_coupling_interaction_weight,
        hidden_coupling_relation_weight=args.hidden_coupling_relation_weight,
        hidden_coupling_param_damage_weight=args.hidden_coupling_param_damage_weight,
        hidden_coupling_graph_damage_weight=args.hidden_coupling_graph_damage_weight,
        write_beta=args.write_beta,
        write_lambda=args.write_lambda,
        event_gamma=args.event_gamma,
        event_beta=args.event_beta,
        event_decay=args.event_decay,
        event_top_k=args.event_top_k,
        recall_gamma=args.recall_gamma,
        recall_beta=args.recall_beta,
        recall_decay=args.recall_decay,
        recall_top_k=args.recall_top_k,
        use_steering_memory=args.use_steering_memory,
        steer_gamma=args.steer_gamma,
        steer_beta=args.steer_beta,
        steer_lambda=args.steer_lambda,
        hard_masks=args.hard_eval,
        backbone=args.backbone,
        budget_target=args.budget_target,
        memory_write_mode=args.memory_write_mode,
        graph_memory_granularity=args.graph_memory_granularity,
        graph_memory_layout=args.graph_memory_layout,
        use_graph_full_branch=args.use_graph_full_branch,
        use_graph_grad_branch=args.use_graph_grad_branch,
        use_graph_branch_gates=args.use_graph_branch_gates,
        param_memory_layout=args.param_memory_layout,
        param_pruning_granularity=args.param_pruning_granularity,
        param_policy_granularity=args.param_policy_granularity,
        graph_score_init=args.graph_score_init,
        param_score_init=args.param_score_init,
        freeze_pruning_scores=args.freeze_pruning_scores,
        seed=seed,
    )
    variant_options = dict(VARIANTS[variant])
    gradient_param_init = variant_options.pop("gradient_param_init", False)
    if gradient_param_init is True:
        gradient_param_init = "snip"
    if gradient_param_init not in (False, "snip", "grasp"):
        raise ValueError(f"Unknown gradient parameter init setting {gradient_param_init!r}.")
    lottery_param_init = bool(variant_options.pop("lottery_param_init", False))
    rigl_param_update = bool(variant_options.pop("rigl_param_update", False))
    serial_pruning = bool(variant_options.pop("serial_pruning", False))
    unified_score_update = bool(variant_options.pop("unified_score_update", False))
    variant_budget_lambda = variant_options.pop("variant_budget_lambda", None)
    private_leftovers = VARIANT_PRIVATE_KEYS.intersection(variant_options)
    if private_leftovers:
        raise ValueError(f"Unhandled private variant keys: {sorted(private_leftovers)}")
    cfg_kwargs.update(variant_options)
    model = OUGPGCN(OUGPConfig(**cfg_kwargs), edge_index=edge_index, x=x).to(device)
    if gradient_param_init:
        initialize_param_scores_from_gradient(model, x, y, train_mask, data.task_type, method=str(gradient_param_init))
    if lottery_param_init:
        initialize_param_scores_from_lottery_pretrain(
            model,
            x,
            y,
            train_mask,
            data.task_type,
            epochs=args.lottery_pretrain_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace_recorder = (
        PruningTraceRecorder(out_dir, args.dataset, variant, seed)
        if args.trace_pruning and variant_options.get("use_graph_pruning", cfg_kwargs["use_graph_pruning"])
        else None
    )

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    best_hard_val = -1.0
    best_hard_test = -1.0
    best_hard_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    total_train_forward_sec = 0.0
    total_train_mask_compute_sec = 0.0
    total_backward_sec = 0.0
    total_eval_mask_compute_sec = 0.0
    total_policy_write_sec = 0.0
    total_optimizer_sec = 0.0
    total_eval_forward_sec = 0.0
    total_epoch_sec = 0.0

    for epoch in range(args.epochs):
        epoch_start = timed_section_start(device)
        model.train()
        keep_g = target_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.graph_sparsity)
        if serial_pruning:
            keep_w = serial_param_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.param_sparsity)
        else:
            keep_w = target_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.param_sparsity)
        model.cfg = OUGPConfig(**{**asdict(model.cfg), "graph_target_keep": keep_g, "param_target_keep": keep_w})
        temperature = temperature_at(epoch, args.epochs, args.temp_start, args.temp_end)

        optimizer.zero_grad(set_to_none=True)
        train_forward_start = timed_section_start(device)
        capture_hidden_states = (
            model.cfg.use_memory
            and model.cfg.use_hidden_coupling
            and epoch >= model.cfg.hidden_coupling_start_epoch
            and model.cfg.hidden_coupling_interval > 0
            and (epoch - model.cfg.hidden_coupling_start_epoch) % model.cfg.hidden_coupling_interval == 0
        )
        hidden_states = None
        if capture_hidden_states:
            logits, mask_stats, hidden_states = model(
                x,
                temperature=temperature,
                return_hidden_states=True,
                retain_hidden_grad=True,
            )
        else:
            logits, mask_stats = model(x, temperature=temperature)
        if data.task_type == "multilabel":
            task_loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        else:
            task_loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss = task_loss + args.sparsity_lambda * model.regularization()
        effective_budget_lambda = float(args.budget_lambda if variant_budget_lambda is None else variant_budget_lambda)
        if effective_budget_lambda != 0.0:
            loss = loss + effective_budget_lambda * model.resource_regularization()
        train_forward_sec = timed_section_end(device, train_forward_start)
        train_mask_compute_sec = float(mask_stats.get("mask_compute_sec", 0.0))
        backward_start = timed_section_start(device)
        loss.backward()
        backward_sec = timed_section_end(device, backward_start)
        rigl_stats = {}
        if (
            rigl_param_update
            and epoch >= args.warmup_epochs
            and (epoch - args.warmup_epochs) % args.rigl_update_interval == 0
        ):
            rigl_stats = update_param_scores_from_rigl(model, args.rigl_gradient_alpha)
        unified_stats = {}
        if (
            unified_score_update
            and epoch >= args.warmup_epochs
            and (epoch - args.warmup_epochs) % args.unified_update_interval == 0
        ):
            unified_stats = update_scores_from_unified_dual(model, args.unified_gradient_alpha)
        policy_write_start = timed_section_start(device)
        memory_stats = model.write_memories(
            x=x,
            temperature=temperature,
            epoch=epoch,
            hidden_states=hidden_states,
        )
        policy_write_sec = timed_section_end(device, policy_write_start)
        trace_stats = {}
        if trace_recorder is not None and epoch % args.trace_every == 0:
            graph_events = model.graph_trace_snapshot(
                dataset=args.dataset,
                variant=variant,
                seed=seed,
                epoch=epoch,
                top_k=args.trace_top_k,
            )
            trace_recorder.record_graph_events(graph_events)
            trace_stats["trace_graph_events"] = len(graph_events)
        optimizer_start = timed_section_start(device)
        optimizer.step()
        optimizer_sec = timed_section_end(device, optimizer_start)
        churn_stats = model.churn()
        periodic_refresh_stats: dict[str, float] = {}
        if (
            args.training_mask_mode == "periodic_ste"
            and (epoch + 1) % args.mask_hardening_interval == 0
        ):
            periodic_refresh_stats = refresh_periodic_training_masks(
                model, args, completed_epoch=epoch + 1
            )

        model.eval()
        with torch.no_grad():
            eval_forward_start = timed_section_start(device)
            eval_logits, eval_mask_stats = model(x, temperature=max(args.temp_end, temperature))
            eval_forward_sec = timed_section_end(device, eval_forward_start)
            eval_mask_compute_sec = float(eval_mask_stats.get("mask_compute_sec", 0.0))
            eval_resource_stats = model.resource_stats()
            soft_val_acc = evaluate_metric(eval_logits, y, val_mask, data.task_type)
            soft_test_acc = evaluate_metric(eval_logits, y, test_mask, data.task_type)
            soft_train_acc = evaluate_metric(eval_logits, y, train_mask, data.task_type)
            if args.training_mask_mode == "periodic_ste":
                current_hard_stats = evaluate_current_epoch_hard_mask(
                    model,
                    x,
                    y,
                    train_mask,
                    val_mask,
                    test_mask,
                    data.task_type,
                    args,
                    completed_epoch=epoch + 1,
                    temperature=max(args.temp_end, temperature),
                )
                train_acc = current_hard_stats["hard_train_acc"]
                val_acc = current_hard_stats["hard_val_acc"]
                test_acc = current_hard_stats["hard_test_acc"]
            else:
                current_hard_stats = {}
                train_acc = soft_train_acc
                val_acc = soft_val_acc
                test_acc = soft_test_acc
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch
                if args.checkpoint_selection == "best_val":
                    best_state = snapshot_module_state(model)

        hard_checkpoint_stats: dict[str, float] = {}
        if args.checkpoint_selection == "best_hard_val":
            if args.training_mask_mode == "periodic_ste":
                hard_checkpoint_stats = dict(current_hard_stats)
            else:
                hard_checkpoint_stats = evaluate_hard_deployment_checkpoint(
                    model, x, y, train_mask, val_mask, test_mask, data.task_type, args
                )
            if hard_checkpoint_stats["hard_val_acc"] > best_hard_val:
                best_hard_val = hard_checkpoint_stats["hard_val_acc"]
                best_hard_test = hard_checkpoint_stats["hard_test_acc"]
                best_hard_epoch = epoch
                best_state = snapshot_module_state(model)

        diagnostic_stats = {}
        should_log_epoch = args.verbose and (epoch % args.log_every == 0 or epoch == args.epochs - 1)
        if args.recall_diagnostics and should_log_epoch:
            model.eval()
            diagnostic_stats = recall_diagnostics(
                model,
                x,
                y,
                val_mask,
                test_mask,
                data.task_type,
                max(args.temp_end, temperature),
                args.recall_effect_eps,
            )
            if args.training_mask_mode != "periodic_ste":
                val_acc = diagnostic_stats["recall_diag_val_acc_after_recall"]
                test_acc = diagnostic_stats["recall_diag_test_acc_after_recall"]
        epoch_sec = timed_section_end(device, epoch_start)
        peak_allocated_mb, peak_reserved_mb = cuda_memory_mb(device)
        total_train_forward_sec += train_forward_sec
        total_train_mask_compute_sec += train_mask_compute_sec
        total_backward_sec += backward_sec
        total_eval_mask_compute_sec += eval_mask_compute_sec
        total_policy_write_sec += policy_write_sec
        total_optimizer_sec += optimizer_sec
        total_eval_forward_sec += eval_forward_sec
        total_epoch_sec += epoch_sec

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "task_loss": float(task_loss.item()),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "best_val": best_val,
            "best_test": best_test,
            "temperature": temperature,
            "epoch_time_sec": epoch_sec,
            "train_forward_sec": train_forward_sec,
            "train_mask_compute_sec": train_mask_compute_sec,
            "backward_sec": backward_sec,
            "policy_write_sec": policy_write_sec,
            "optimizer_step_sec": optimizer_sec,
            "eval_forward_sec": eval_forward_sec,
            "eval_mask_compute_sec": eval_mask_compute_sec,
            "mask_policy_lhcm_overhead_sec": train_mask_compute_sec + policy_write_sec,
            "nodes_per_sec_epoch": float(data.num_nodes / max(epoch_sec, 1e-12)),
            "edges_per_sec_epoch": float(edge_index.size(1) / max(epoch_sec, 1e-12)),
            "peak_gpu_allocated_mb": peak_allocated_mb,
            "peak_gpu_reserved_mb": peak_reserved_mb,
            **mask_stats,
            **eval_mask_stats,
            **eval_resource_stats,
            **churn_stats,
            **memory_stats,
            **rigl_stats,
            **unified_stats,
            **trace_stats,
            **periodic_refresh_stats,
            **current_hard_stats,
            "soft_train_acc": soft_train_acc,
            "soft_val_acc": soft_val_acc,
            "soft_test_acc": soft_test_acc,
            **hard_checkpoint_stats,
            **diagnostic_stats,
        }
        history.append(row)
        if should_log_epoch:
            message = (
                f"{variant}/seed{seed} epoch={epoch:03d} "
                f"loss={loss.item():.4f} val={val_acc:.4f} test={test_acc:.4f} "
                f"g_keep={eval_mask_stats['graph_keep']:.3f} p_keep={eval_mask_stats['param_keep']:.3f}"
            )
            if args.score_diagnostics:
                message += (
                    f" graph_logits_mean={eval_mask_stats['graph_logits_mean']:.6f}"
                    f" graph_logits_std={eval_mask_stats['graph_logits_std']:.6f}"
                    f" graph_memory_correction_mean={eval_mask_stats['graph_memory_correction_mean']:.6f}"
                    f" graph_memory_correction_std={eval_mask_stats['graph_memory_correction_std']:.6f}"
                    f" graph_memory_unit_correction_std={eval_mask_stats['graph_memory_unit_correction_std']:.6f}"
                    f" graph_memory_raw_correction_mean={eval_mask_stats['graph_memory_raw_correction_mean']:.6f}"
                    f" graph_memory_raw_correction_std={eval_mask_stats['graph_memory_raw_correction_std']:.6f}"
                    f" graph_score_scale={eval_mask_stats['graph_score_scale']:.6f}"
                    f" graph_memory_score_delta_std={eval_mask_stats['graph_memory_score_delta_std']:.6f}"
                    f" param_logits_mean={eval_mask_stats['param_logits_mean']:.6f}"
                    f" param_logits_std={eval_mask_stats['param_logits_std']:.6f}"
                    f" param_memory_correction_mean={eval_mask_stats['param_memory_correction_mean']:.6f}"
                    f" param_memory_correction_std={eval_mask_stats['param_memory_correction_std']:.6f}"
                    f" param_memory_unit_correction_std={eval_mask_stats['param_memory_unit_correction_std']:.6f}"
                    f" param_memory_raw_correction_mean={eval_mask_stats['param_memory_raw_correction_mean']:.6f}"
                    f" param_memory_raw_correction_std={eval_mask_stats['param_memory_raw_correction_std']:.6f}"
                    f" recall_correction_mean={eval_mask_stats['recall_correction_mean']:.6f}"
                    f" recall_correction_std={eval_mask_stats['recall_correction_std']:.6f}"
                    f" recall_unit_correction_std={eval_mask_stats['recall_unit_correction_std']:.6f}"
                    f" param_score_scale={eval_mask_stats['param_score_scale']:.6f}"
                    f" param_memory_score_delta_std={eval_mask_stats['param_memory_score_delta_std']:.6f}"
                    f" recall_score_delta_std={eval_mask_stats['recall_score_delta_std']:.6f}"
                )
            if diagnostic_stats:
                message += (
                    f" param_sparsity={diagnostic_stats['recall_diag_param_sparsity']:.3f}"
                    f" dropped_params={diagnostic_stats['recall_diag_dropped_parameters']:.1f}"
                    f" recalled_params={diagnostic_stats['recall_diag_recalled_parameters']:.0f}"
                    f" effective_recovered={diagnostic_stats['recall_diag_effective_recovered_parameters']:.0f}"
                    f" param_churn={churn_stats.get('param_churn', 0.0):.4f}"
                    f" acc_before_recall={diagnostic_stats['recall_diag_val_acc_before_recall']:.4f}"
                    f" acc_after_recall={diagnostic_stats['recall_diag_val_acc_after_recall']:.4f}"
                )
            print(message, flush=True)

    last_epoch = dict(history[-1])
    training_wall_time_sec = time.time() - start_time
    selected_hard_stats: dict[str, float] = {}
    if args.checkpoint_selection in {"best_val", "best_hard_val"}:
        if best_state is None:
            raise RuntimeError("No requested validation checkpoint was captured.")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        with torch.no_grad():
            selected_logits, selected_mask_stats = model(x, temperature=args.temp_end)
            selected_resource_stats = model.resource_stats()
            selected_train_acc = evaluate_metric(selected_logits, y, train_mask, data.task_type)
            selected_val_acc = evaluate_metric(selected_logits, y, val_mask, data.task_type)
            selected_test_acc = evaluate_metric(selected_logits, y, test_mask, data.task_type)
        final = {
            **last_epoch,
            **selected_mask_stats,
            **selected_resource_stats,
            "train_acc": selected_train_acc,
            "val_acc": selected_val_acc,
            "test_acc": selected_test_acc,
        }
        if args.training_mask_mode == "periodic_ste":
            selected_epoch = best_hard_epoch if args.checkpoint_selection == "best_hard_val" else best_epoch
            selected_hard_stats = evaluate_current_epoch_hard_mask(
                model,
                x,
                y,
                train_mask,
                val_mask,
                test_mask,
                data.task_type,
                args,
                completed_epoch=selected_epoch + 1,
                temperature=args.temp_end,
            )
            final.update(
                {
                    "train_acc": selected_hard_stats["hard_train_acc"],
                    "val_acc": selected_hard_stats["hard_val_acc"],
                    "test_acc": selected_hard_stats["hard_test_acc"],
                }
            )
        elif args.checkpoint_selection == "best_hard_val":
            selected_hard_stats = evaluate_hard_deployment_checkpoint(
                model, x, y, train_mask, val_mask, test_mask, data.task_type, args
            )

        if args.checkpoint_selection == "best_hard_val":
            selected_checkpoint_epoch = best_hard_epoch
        else:
            selected_checkpoint_epoch = best_epoch
    else:
        final = last_epoch
        selected_checkpoint_epoch = args.epochs - 1
    inference_latency_sec = 0.0
    inference_nodes_per_sec = 0.0
    inference_edges_per_sec = 0.0
    if args.profile_inference_repeats > 0:
        model.eval()
        with torch.no_grad():
            for _ in range(max(0, args.profile_inference_warmup)):
                model(x, temperature=args.temp_end)
            sync_if_cuda(device)
            inference_start = time.perf_counter()
            for _ in range(args.profile_inference_repeats):
                model(x, temperature=args.temp_end)
            sync_if_cuda(device)
            inference_latency_sec = (time.perf_counter() - inference_start) / float(args.profile_inference_repeats)
            inference_nodes_per_sec = float(data.num_nodes / max(inference_latency_sec, 1e-12))
            inference_edges_per_sec = float(edge_index.size(1) / max(inference_latency_sec, 1e-12))
    deployment_profile_start = time.time()
    deployment_profile = profile_deployment_latency(
        model,
        x,
        y,
        test_mask,
        data.task_type,
        args.temp_end,
        args.profile_deployment_repeats,
        args.profile_deployment_warmup,
    )
    deployment_profile_wall_time_sec = time.time() - deployment_profile_start
    peak_allocated_mb, peak_reserved_mb = cuda_memory_mb(device)
    deployed_hidden_dim = int(
        round(float(deployment_profile.get("materialized_kept_channels", args.hidden_dim)))
    )
    deployed_edge_count = int(
        round(float(deployment_profile.get("materialized_kept_edges", edge_index.size(1))))
    )
    deployment_flops = deployment_flops_metrics(
        num_nodes=data.num_nodes,
        input_dim=data.num_features,
        original_hidden_dim=args.hidden_dim,
        deployed_hidden_dim=deployed_hidden_dim,
        output_dim=data.num_classes,
        original_edge_count=edge_index.size(1),
        deployed_edge_count=deployed_edge_count,
        hidden_layers=max(1, args.num_gnn_layers - 1),
        flops_reduction_source="post-retraining hard-materialized GCN deployment",
    )
    dense_ops = float(final.get("dense_message_cost", 0.0)) + float(final.get("dense_parameter_count", 0.0))
    effective_ops = float(final.get("effective_message_cost", 0.0)) + float(final.get("effective_parameter_count", 0.0))
    result = {
        "dataset": args.dataset,
        "variant": variant,
        "seed": seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "best_hard_epoch": best_hard_epoch,
        "best_hard_val_acc": best_hard_val,
        "best_hard_test_acc": best_hard_test,
        "checkpoint_selection": args.checkpoint_selection,
        "training_mask_mode": args.training_mask_mode,
        "mask_hardening_interval": args.mask_hardening_interval,
        "training_mask_hardening_mode": args.training_mask_hardening_mode,
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "selected_checkpoint_train_acc": (
            selected_hard_stats.get("hard_train_acc", final["train_acc"])
        ),
        "selected_checkpoint_val_acc": (
            selected_hard_stats.get("hard_val_acc", final["val_acc"])
        ),
        "selected_checkpoint_test_acc": (
            selected_hard_stats.get("hard_test_acc", final["test_acc"])
        ),
        "selected_checkpoint_soft_train_acc": final["train_acc"],
        "selected_checkpoint_soft_val_acc": final["val_acc"],
        "selected_checkpoint_soft_test_acc": final["test_acc"],
        "selected_checkpoint_hard_train_acc": selected_hard_stats.get("hard_train_acc"),
        "selected_checkpoint_hard_val_acc": selected_hard_stats.get("hard_val_acc"),
        "selected_checkpoint_hard_test_acc": selected_hard_stats.get("hard_test_acc"),
        "last_epoch_train_acc": last_epoch["train_acc"],
        "last_epoch_val_acc": last_epoch["val_acc"],
        "last_epoch_test_acc": last_epoch["test_acc"],
        "final_train_acc": final["train_acc"],
        "final_val_acc": final["val_acc"],
        "final_test_acc": final["test_acc"],
        "graph_keep": final["graph_keep"],
        "graph_sparsity": 1.0 - final["graph_keep"],
        "param_keep": final["param_keep"],
        "param_sparsity": 1.0 - final["param_keep"],
        "param_pruning_granularity": model.cfg.param_pruning_granularity,
        "param_policy_granularity": model.resolved_param_policy_granularity,
        "weight_entry_keep": (
            final["param_keep"] if model.cfg.param_pruning_granularity == "weight" else 1.0
        ),
        "weight_entry_sparsity": (
            1.0 - final["param_keep"] if model.cfg.param_pruning_granularity == "weight" else 0.0
        ),
        "channel_keep": (
            final["param_keep"] if model.cfg.param_pruning_granularity == "channel" else 1.0
        ),
        "channel_sparsity": (
            1.0 - final["param_keep"] if model.cfg.param_pruning_granularity == "channel" else 0.0
        ),
        "graph_churn": final.get("graph_churn", 0.0),
        "param_churn": final.get("param_churn", 0.0),
        "channel_churn": final.get("param_churn", 0.0),
        "graph_policy_correction_norm": final.get("graph_policy_correction_norm", 0.0),
        "channel_policy_correction_norm": final.get("channel_policy_correction_norm", 0.0),
        "message_cost_ratio": final.get("message_cost_ratio", 1.0),
        "message_cost_reduction": final.get("message_cost_reduction", 0.0),
        "parameter_cost_ratio": final.get("parameter_cost_ratio", 1.0),
        "parameter_cost_reduction": final.get("parameter_cost_reduction", 0.0),
        "estimated_message_cost_reduction": final.get("message_cost_reduction", 0.0),
        "estimated_parameter_cost_reduction": final.get("parameter_cost_reduction", 0.0),
        "actual_parameter_count_reduction": final.get("parameter_cost_reduction", 0.0),
        "dense_message_cost": final.get("dense_message_cost", 0.0),
        "effective_message_cost": final.get("effective_message_cost", 0.0),
        "dense_parameter_count": final.get("dense_parameter_count", 0.0),
        "effective_parameter_count": final.get("effective_parameter_count", 0.0),
        "dense_effective_operation_cost": dense_ops,
        "effective_sparse_operation_cost": effective_ops,
        "effective_sparse_operation_reduction": 1.0 - effective_ops / max(dense_ops, 1.0),
        "memory_state_items": final.get("memory_state_items", 0.0),
        "memory_overhead_vs_dense_params": final.get("memory_overhead_vs_dense_params", 0.0),
        "avg_epoch_time_sec": total_epoch_sec / max(args.epochs, 1),
        "total_training_time_sec": total_epoch_sec,
        "training_wall_time_sec": training_wall_time_sec,
        "avg_train_forward_sec": total_train_forward_sec / max(args.epochs, 1),
        "avg_train_mask_compute_sec": total_train_mask_compute_sec / max(args.epochs, 1),
        "avg_backward_sec": total_backward_sec / max(args.epochs, 1),
        "avg_eval_mask_compute_sec": total_eval_mask_compute_sec / max(args.epochs, 1),
        "avg_policy_write_sec": total_policy_write_sec / max(args.epochs, 1),
        "avg_optimizer_step_sec": total_optimizer_sec / max(args.epochs, 1),
        "avg_eval_forward_sec": total_eval_forward_sec / max(args.epochs, 1),
        "policy_update_overhead_ratio": total_policy_write_sec / max(total_epoch_sec, 1e-12),
        "mask_policy_lhcm_overhead_sec": total_train_mask_compute_sec + total_policy_write_sec,
        "mask_policy_lhcm_overhead_ratio": (total_train_mask_compute_sec + total_policy_write_sec) / max(total_epoch_sec, 1e-12),
        "train_forward_overhead_ratio": total_train_forward_sec / max(total_epoch_sec, 1e-12),
        "backward_overhead_ratio": total_backward_sec / max(total_epoch_sec, 1e-12),
        "eval_forward_overhead_ratio": total_eval_forward_sec / max(total_epoch_sec, 1e-12),
        "nodes_per_sec_train_epoch": float(data.num_nodes * args.epochs / max(total_epoch_sec, 1e-12)),
        "edges_per_sec_train_epoch": float(edge_index.size(1) * args.epochs / max(total_epoch_sec, 1e-12)),
        "inference_latency_sec": inference_latency_sec,
        "inference_nodes_per_sec": inference_nodes_per_sec,
        "inference_edges_per_sec": inference_edges_per_sec,
        **deployment_profile,
        "deployment_profile_wall_time_sec": deployment_profile_wall_time_sec,
        "deployed_hidden_dim": deployed_hidden_dim,
        "deployed_edge_count": deployed_edge_count,
        **deployment_flops,
        "peak_gpu_allocated_mb": peak_allocated_mb,
        "peak_gpu_reserved_mb": peak_reserved_mb,
        "graph_memory_state_norm": float(model.graph_memory.state.norm().item()),
        "param_memory_state_norm": float(model.param_memory.state.norm().item()),
        "graph_recall_bias_norm": float(model.graph_memory.recall_bias.norm().item()),
        "param_recall_bias_norm": float(model.param_memory.recall_bias.norm().item()),
        "steering_memory_state_norm": float(model.steering_memory.state.norm().item()),
        "runtime_sec": training_wall_time_sec,
        "end_to_end_runtime_sec": time.time() - start_time,
        "num_nodes": data.num_nodes,
        "num_edges": edge_index.size(1),
        "original_num_nodes": int(getattr(args, "original_num_nodes", data.num_nodes)),
        "original_num_edges": int(getattr(args, "original_num_edges", edge_index.size(1))),
        "node_sample_size": int(args.node_sample_size),
        "node_sample_mode": args.node_sample_mode,
        "edge_sample_size": int(args.edge_sample_size),
        "hidden_dim": args.hidden_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "memory_rank": args.memory_rank,
        "memory_write_mode": args.memory_write_mode,
        "graph_memory_granularity": args.graph_memory_granularity,
        "graph_memory_layout": args.graph_memory_layout,
        "use_graph_full_branch": bool(args.use_graph_full_branch),
        "use_graph_grad_branch": bool(args.use_graph_grad_branch),
        "param_memory_layout": args.param_memory_layout,
        "graph_score_init": cfg_kwargs["graph_score_init"],
        "param_score_init": cfg_kwargs["param_score_init"],
        "freeze_pruning_scores": bool(cfg_kwargs["freeze_pruning_scores"]),
        "gradient_param_init": bool(gradient_param_init),
        "gradient_param_method": str(gradient_param_init) if gradient_param_init else "none",
        "lottery_param_init": bool(lottery_param_init),
        "lottery_pretrain_epochs": int(args.lottery_pretrain_epochs if lottery_param_init else 0),
        "rigl_param_update": bool(rigl_param_update),
        "rigl_update_interval": int(args.rigl_update_interval if rigl_param_update else 0),
        "rigl_gradient_alpha": float(args.rigl_gradient_alpha if rigl_param_update else 0.0),
        "unified_score_update": bool(unified_score_update),
        "unified_update_interval": int(args.unified_update_interval if unified_score_update else 0),
        "unified_gradient_alpha": float(args.unified_gradient_alpha if unified_score_update else 0.0),
        "effective_budget_lambda": float(args.budget_lambda if variant_budget_lambda is None else variant_budget_lambda),
        "serial_pruning": bool(serial_pruning),
        "backbone": args.backbone,
        "param_memory_slots": int(getattr(model.param_memory, "num_channels", 1)),
        "task_type": data.task_type,
        "metric_name": data.metric_name,
    }

    run_path = out_dir / f"{args.dataset}_{variant}_seed{seed}.json"
    run_path.write_text(json.dumps({"result": result, "history": history, "config": cfg_kwargs}, indent=2))
    if return_artifacts:
        return result, model, history
    return result


def aggregate(results: list[dict[str, float | int | str]], out_dir: Path, dataset: str) -> None:
    csv_path = out_dir / f"{dataset}_summary.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    by_variant: dict[str, list[dict[str, float | int | str]]] = {}
    for row in results:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    lines = [
        "# OUGP Case Study Summary",
        "",
        f"Dataset: `{dataset}`",
        f"Backbone: `{results[0].get('backbone', 'gcn')}`",
        "",
        "| Variant | Best Test Acc | Graph Sparsity | Channel Sparsity | Message Cost Red. | Parameter Count Red. | Runtime Sec | Epoch Sec | Online Latency ms | Fixed Mask Latency ms | Materialized Sparse Latency ms | Mat. Speedup vs Online | Peak GPU MB | Mask+Policy+LHCM Overhead | Policy Overhead |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, rows in sorted(by_variant.items()):
        def mean(key: str) -> float:
            return float(np.mean([float(r.get(key, 0.0)) for r in rows]))

        def std(key: str) -> float:
            return float(np.std([float(r.get(key, 0.0)) for r in rows]))

        def mean_with_fallback(key: str, fallback: str) -> float:
            values = [float(r[key]) if key in r else float(r.get(fallback, 0.0)) for r in rows]
            return float(np.mean(values))

        lines.append(
            f"| {variant} | {mean('best_test_acc'):.4f} +/- {std('best_test_acc'):.4f} | "
            f"{mean('graph_sparsity'):.3f} | {mean('param_sparsity'):.3f} | "
            f"{mean('message_cost_reduction'):.3f} | {mean('parameter_cost_reduction'):.3f} | "
            f"{mean('runtime_sec'):.2f} | {mean('avg_epoch_time_sec'):.4f} | "
            f"{1000.0 * mean_with_fallback('online_inference_latency_sec', 'inference_latency_sec'):.3f} | "
            f"{1000.0 * mean('fixed_mask_inference_latency_sec'):.3f} | "
            f"{1000.0 * mean('materialized_sparse_inference_latency_sec'):.3f} | "
            f"{mean('materialized_speedup_vs_online'):.3f} | "
            f"{mean('peak_gpu_allocated_mb'):.1f} | "
            f"{mean('mask_policy_lhcm_overhead_ratio'):.3f} | "
            f"{mean('policy_update_overhead_ratio'):.3f} |"
        )
    (out_dir / f"{dataset}_summary.md").write_text("\n".join(lines) + "\n")


def write_run_manifest(args: argparse.Namespace, results: list[dict[str, float | int | str]], out_dir: Path) -> None:
    command = " ".join(shlex.quote(part) for part in [sys.executable, "scripts/run_case_study.py", *sys.argv[1:]])
    (out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    manifest = {
        "command": command,
        "args": vars(args),
        "result_files": {
            "summary_csv": str(out_dir / f"{args.dataset}_summary.csv"),
            "summary_md": str(out_dir / f"{args.dataset}_summary.md"),
            "run_json_pattern": str(out_dir / f"{args.dataset}_*_seed*.json"),
        },
        "final_metrics": results,
        "log_note": "Per-epoch metrics are stored in each run JSON under the `history` key.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def maybe_sample_edges(dataset: CitationGraph, sample_size: int, seed: int) -> CitationGraph:
    if sample_size <= 0 or sample_size >= dataset.edge_index.size(1):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    edge_ids = torch.randperm(dataset.edge_index.size(1), generator=generator)[:sample_size]
    sampled_edge_index = dataset.edge_index[:, edge_ids].contiguous()
    return replace(dataset, edge_index=sampled_edge_index)


def _sample_from_mask(mask: torch.Tensor, max_items: int, generator: torch.Generator) -> torch.Tensor:
    indices = mask.cpu().nonzero(as_tuple=False).flatten()
    if max_items <= 0 or indices.numel() == 0:
        return indices[:0]
    order = torch.randperm(indices.numel(), generator=generator)[: min(max_items, indices.numel())]
    return indices[order]


def _split_seed_nodes(dataset: CitationGraph, sample_size: int, generator: torch.Generator) -> torch.Tensor:
    forced_budget = max(1, min(256, sample_size // 10))
    forced_parts = [
        _sample_from_mask(dataset.train_mask, forced_budget, generator),
        _sample_from_mask(dataset.val_mask, forced_budget, generator),
        _sample_from_mask(dataset.test_mask, forced_budget, generator),
    ]
    forced = torch.unique(torch.cat(forced_parts)) if forced_parts else torch.empty(0, dtype=torch.long)
    if forced.numel() > sample_size:
        forced = forced[torch.randperm(forced.numel(), generator=generator)[:sample_size]]
    return forced


def _fill_random_nodes(keep: torch.Tensor, remaining: int, generator: torch.Generator) -> int:
    if remaining <= 0:
        return 0
    candidates = torch.randperm(keep.numel(), generator=generator)
    added = 0
    for node in candidates.tolist():
        if added >= remaining:
            break
        if not bool(keep[node].item()):
            keep[node] = True
            added += 1
    return added


def _fill_frontier_nodes(dataset: CitationGraph, keep: torch.Tensor, remaining: int, generator: torch.Generator) -> int:
    if remaining <= 0:
        return 0
    row, col = dataset.edge_index.cpu()
    frontier = keep.clone()
    added = 0
    while added < remaining and bool(frontier.any().item()):
        src_frontier = frontier[row]
        dst_frontier = frontier[col]
        candidates = torch.cat([col[src_frontier], row[dst_frontier]])
        if candidates.numel() == 0:
            break
        candidates = torch.unique(candidates[~keep[candidates]])
        if candidates.numel() == 0:
            break
        order = torch.randperm(candidates.numel(), generator=generator)
        selected = candidates[order[: min(remaining - added, candidates.numel())]]
        keep[selected] = True
        added += int(selected.numel())
        frontier.zero_()
        frontier[selected] = True
    return added


def maybe_sample_nodes(dataset: CitationGraph, sample_size: int, seed: int, mode: str = "random") -> CitationGraph:
    """Build a node-sampled induced subgraph while preserving split coverage.

    This is a static subgraph smoke path for large-graph validation. It is not
    full mini-batch neighbor sampling, but it keeps node features, labels, split
    masks, and edge endpoints consistent after remapping to local node ids.
    """

    if sample_size <= 0 or sample_size >= dataset.num_nodes:
        return dataset
    if sample_size < 3:
        raise ValueError("--node-sample-size must be at least 3 when enabled.")
    if mode not in {"random", "frontier"}:
        raise ValueError("--node-sample-mode must be one of: random, frontier.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    forced = _split_seed_nodes(dataset, sample_size, generator)
    keep = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    keep[forced] = True
    remaining = sample_size - int(keep.sum().item())
    if remaining > 0:
        if mode == "frontier":
            remaining -= _fill_frontier_nodes(dataset, keep, remaining, generator)
        if remaining > 0:
            _fill_random_nodes(keep, remaining, generator)

    node_ids = keep.nonzero(as_tuple=False).flatten()
    old_to_new = torch.full((dataset.num_nodes,), -1, dtype=torch.long)
    old_to_new[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long)

    row, col = dataset.edge_index.cpu()
    edge_keep = keep[row] & keep[col]
    sampled_edge_index = old_to_new[dataset.edge_index.cpu()[:, edge_keep]].contiguous()

    sampled = replace(
        dataset,
        x=dataset.x[node_ids].contiguous(),
        y=dataset.y[node_ids].contiguous(),
        edge_index=sampled_edge_index,
        train_mask=dataset.train_mask[node_ids].contiguous(),
        val_mask=dataset.val_mask[node_ids].contiguous(),
        test_mask=dataset.test_mask[node_ids].contiguous(),
    )
    if sampled.train_mask.sum() == 0 or sampled.val_mask.sum() == 0 or sampled.test_mask.sum() == 0:
        raise RuntimeError(
            "Node sampling produced an empty train/val/test split. "
            "Increase --node-sample-size or use a different --node-sample-seed."
        )
    if sampled.edge_index.numel() == 0:
        raise RuntimeError(
            "Node sampling produced a subgraph with no edges. "
            "Increase --node-sample-size or use a different --node-sample-seed."
        )
    return sampled


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="cora",
        choices=[
            "cora",
            "citeseer",
            "pubmed",
            "photo",
            "computers",
            "coauthor_cs",
            "coauthor_physics",
            "wiki_cs",
            "actor",
            "acm",
            "udagcn_dblp",
            "udagcn_acm",
            "webkb_cornell",
            "webkb_texas",
            "webkb_wisconsin",
            "chameleon",
            "squirrel",
            "roman_empire",
            "amazon_ratings",
            "minesweeper",
            "tolokers",
            "citationfull_cora_ml",
            "citationfull_cora",
            "citationfull_dblp",
            "citationfull_citeseer",
            "citationfull_pubmed",
            "ogbn-arxiv",
            "ogbn-products",
            "ogbn-proteins",
        ],
    )
    parser.add_argument("--data-root", default="data/raw/planetoid")
    parser.add_argument("--split-mode", choices=["original", "stratified_2_1_1"], default="original")
    parser.add_argument("--out-dir", default="experiments/manual_case_study")
    parser.add_argument("--node-sample-size", type=int, default=0)
    parser.add_argument("--node-sample-seed", type=int, default=0)
    parser.add_argument("--node-sample-mode", choices=["random", "frontier"], default="random")
    parser.add_argument("--edge-sample-size", type=int, default=0)
    parser.add_argument("--edge-sample-seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=["dense", "dual_static", "ougp_no_cross", "ougp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["last", "best_val", "best_hard_val"],
        default="last",
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--training-mask-mode", choices=["soft", "periodic_ste"], default="soft"
    )
    parser.add_argument("--mask-hardening-interval", type=int, default=20)
    parser.add_argument(
        "--training-mask-hardening-mode",
        choices=["quantile_binary"],
        default="quantile_binary",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--backbone", choices=BACKBONES, default="gcn")
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--memory-rank", type=int, default=8)
    parser.add_argument("--memory-write-mode", choices=["residual", "feature", "none"], default="residual")
    parser.add_argument("--graph-memory-granularity", choices=["edge", "subgraph"], default="edge")
    parser.add_argument("--graph-memory-layout", choices=["single", "multi"], default="single")
    parser.add_argument("--use-graph-full-branch", dest="use_graph_full_branch", action="store_true")
    parser.add_argument("--no-graph-full-branch", dest="use_graph_full_branch", action="store_false")
    parser.add_argument("--use-graph-grad-branch", dest="use_graph_grad_branch", action="store_true")
    parser.add_argument("--no-graph-grad-branch", dest="use_graph_grad_branch", action="store_false")
    parser.add_argument("--use-graph-branch-gates", dest="use_graph_branch_gates", action="store_true")
    parser.add_argument("--no-graph-branch-gates", dest="use_graph_branch_gates", action="store_false")
    parser.set_defaults(use_graph_full_branch=True)
    parser.set_defaults(use_graph_grad_branch=True)
    parser.set_defaults(use_graph_branch_gates=True)
    parser.add_argument("--param-memory-layout", choices=["single", "multi"], default="single")
    parser.add_argument(
        "--param-pruning-granularity",
        choices=["channel", "weight"],
        default="weight",
    )
    parser.add_argument(
        "--param-policy-granularity",
        choices=["auto", "channel", "weight"],
        default="auto",
    )
    parser.add_argument("--graph-score-init", choices=GRAPH_SCORE_INITS, default="constant")
    parser.add_argument("--param-score-init", choices=PARAM_SCORE_INITS, default="constant")
    parser.add_argument("--freeze-pruning-scores", action="store_true")
    parser.add_argument("--lottery-pretrain-epochs", type=int, default=5)
    parser.add_argument("--rigl-update-interval", type=int, default=10)
    parser.add_argument("--rigl-gradient-alpha", type=float, default=0.5)
    parser.add_argument("--unified-update-interval", type=int, default=10)
    parser.add_argument("--unified-gradient-alpha", type=float, default=0.5)
    parser.add_argument("--graph-sparsity", type=float, default=0.30)
    parser.add_argument("--param-sparsity", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sparsity-lambda", type=float, default=0.08)
    parser.add_argument("--budget-lambda", type=float, default=0.0)
    parser.add_argument("--budget-target", type=float, default=0.70)
    parser.add_argument("--graph-gamma", type=float, default=0.35)
    parser.add_argument("--param-gamma", type=float, default=0.35)
    parser.add_argument("--graph-score-scale-decay", type=float, default=0.95)
    parser.add_argument("--graph-score-scale-min", type=float, default=0.02)
    parser.add_argument("--graph-score-scale-max", type=float, default=0.50)
    parser.add_argument("--graph-correction-clip", type=float, default=2.0)
    parser.add_argument("--param-score-scale-decay", type=float, default=0.95)
    parser.add_argument("--param-score-scale-min", type=float, default=0.02)
    parser.add_argument("--param-score-scale-max", type=float, default=0.50)
    parser.add_argument("--param-correction-clip", type=float, default=2.0)
    parser.add_argument("--cross-gamma", type=float, default=0.20)
    parser.add_argument("--use-hidden-coupling", action="store_true")
    parser.add_argument("--hidden-coupling-mix-graph", type=float, default=0.0)
    parser.add_argument("--hidden-coupling-mix-param", type=float, default=0.0)
    parser.add_argument("--hidden-coupling-interval", type=int, default=1)
    parser.add_argument("--hidden-coupling-start-epoch", type=int, default=0)
    parser.add_argument("--hidden-coupling-layer-norm-weight", type=float, default=1.0)
    parser.add_argument("--hidden-coupling-interaction-weight", type=float, default=1.0)
    parser.add_argument("--hidden-coupling-relation-weight", type=float, default=1.0)
    parser.add_argument("--hidden-coupling-param-damage-weight", type=float, default=1.0)
    parser.add_argument("--hidden-coupling-graph-damage-weight", type=float, default=1.0)
    parser.add_argument("--write-beta", type=float, default=0.12)
    parser.add_argument("--write-lambda", type=float, default=0.98)
    parser.add_argument("--event-gamma", type=float, default=0.0)
    parser.add_argument("--event-beta", type=float, default=0.10)
    parser.add_argument("--event-decay", type=float, default=0.95)
    parser.add_argument("--event-top-k", type=int, default=2000)
    parser.add_argument("--recall-gamma", type=float, default=0.0)
    parser.add_argument("--recall-beta", type=float, default=0.10)
    parser.add_argument("--recall-decay", type=float, default=0.95)
    parser.add_argument("--recall-top-k", type=int, default=2000)
    parser.add_argument("--use-steering-memory", action="store_true")
    parser.add_argument("--steer-gamma", type=float, default=0.0)
    parser.add_argument("--steer-beta", type=float, default=0.10)
    parser.add_argument("--steer-lambda", type=float, default=0.95)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--hard-eval", action="store_true")
    parser.add_argument("--trace-pruning", action="store_true")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--trace-top-k", type=int, default=200)
    parser.add_argument("--recall-diagnostics", action="store_true")
    parser.add_argument("--recall-effect-eps", type=float, default=1e-4)
    parser.add_argument("--profile-inference-repeats", type=int, default=0)
    parser.add_argument("--profile-inference-warmup", type=int, default=3)
    parser.add_argument("--profile-deployment-repeats", type=int, default=0)
    parser.add_argument("--profile-deployment-warmup", type=int, default=5)
    parser.add_argument("--score-diagnostics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def validate_gpu_budget(args: argparse.Namespace) -> None:
    if args.trace_every <= 0:
        raise ValueError("--trace-every must be a positive integer.")
    if args.trace_top_k < 0:
        raise ValueError("--trace-top-k must be non-negative.")
    if args.event_top_k < 0:
        raise ValueError("--event-top-k must be non-negative.")
    if args.recall_top_k < 0:
        raise ValueError("--recall-top-k must be non-negative.")
    if args.edge_sample_size < 0:
        raise ValueError("--edge-sample-size must be non-negative.")
    if args.node_sample_size < 0:
        raise ValueError("--node-sample-size must be non-negative.")
    if args.budget_lambda < 0:
        raise ValueError("--budget-lambda must be non-negative.")
    if args.lottery_pretrain_epochs < 0:
        raise ValueError("--lottery-pretrain-epochs must be non-negative.")
    if args.rigl_update_interval <= 0:
        raise ValueError("--rigl-update-interval must be positive.")
    if not 0.0 <= args.rigl_gradient_alpha <= 1.0:
        raise ValueError("--rigl-gradient-alpha must be in [0, 1].")
    if args.unified_update_interval <= 0:
        raise ValueError("--unified-update-interval must be positive.")
    if not 0.0 <= args.unified_gradient_alpha <= 1.0:
        raise ValueError("--unified-gradient-alpha must be in [0, 1].")
    if args.hidden_coupling_interval <= 0:
        raise ValueError("--hidden-coupling-interval must be positive.")
    if args.hidden_coupling_start_epoch < 0:
        raise ValueError("--hidden-coupling-start-epoch must be non-negative.")
    if args.profile_inference_repeats < 0:
        raise ValueError("--profile-inference-repeats must be non-negative.")
    if args.profile_inference_warmup < 0:
        raise ValueError("--profile-inference-warmup must be non-negative.")
    if args.profile_deployment_repeats < 0:
        raise ValueError("--profile-deployment-repeats must be non-negative.")
    if args.profile_deployment_warmup < 0:
        raise ValueError("--profile-deployment-warmup must be non-negative.")
    if not 0.0 <= args.hidden_coupling_mix_graph <= 1.0:
        raise ValueError("--hidden-coupling-mix-graph must be in [0, 1].")
    if not 0.0 <= args.hidden_coupling_mix_param <= 1.0:
        raise ValueError("--hidden-coupling-mix-param must be in [0, 1].")
    if not 0.0 < args.budget_target <= 1.0:
        raise ValueError("--budget-target must be in (0, 1].")
    if args.num_gnn_layers < 2:
        raise ValueError("--num-gnn-layers must be at least 2.")
    if args.backbone not in {"gcn", "deepgcn"} and args.num_gnn_layers != 2:
        raise ValueError("--num-gnn-layers is currently only configurable with --backbone gcn/deepgcn.")
    if args.backbone == "deepgcn" and args.num_gnn_layers < 3:
        raise ValueError("--backbone deepgcn requires --num-gnn-layers >= 3.")
    if args.max_gpus > 4:
        raise ValueError("This workspace policy allows at most 4 GPUs per run.")
    if args.device.startswith("cuda"):
        visible = torch.cuda.device_count()
        if visible > args.max_gpus:
            raise RuntimeError(
                f"{visible} CUDA devices are visible, but this run is capped at {args.max_gpus}. "
                f"Set CUDA_VISIBLE_DEVICES to at most {args.max_gpus} devices."
            )


def main() -> None:
    args = parse_args()
    validate_gpu_budget(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_graph_dataset(args.data_root, args.dataset)
    split_seed = int(args.seeds[0]) if len(args.seeds) == 1 else 0
    dataset = apply_split_mode(dataset, args.split_mode, split_seed)
    args.original_num_nodes = int(dataset.num_nodes)
    args.original_num_edges = int(dataset.edge_index.size(1))
    dataset = maybe_sample_nodes(dataset, args.node_sample_size, args.node_sample_seed, args.node_sample_mode)
    dataset = maybe_sample_edges(dataset, args.edge_sample_size, args.edge_sample_seed)
    results = []
    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        for seed in args.seeds:
            results.append(run_one(args, dataset, variant, seed, out_dir))
    aggregate(results, out_dir, args.dataset)
    write_run_manifest(args, results, out_dir)
    print(f"Wrote {len(results)} runs to {out_dir}")
    print((out_dir / f"{args.dataset}_summary.md").read_text())


if __name__ == "__main__":
    main()
