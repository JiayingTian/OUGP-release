#!/usr/bin/env python3
"""Clean quantile-hardened OUGP deployment and TTA evaluation.

The run uses one deterministic 2:1:1 split and reports clean test accuracy plus
test-inference latency for Dense, OUGP, and OUGP + the configured EXP121 TTA
test-time mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deployed_costs import (
    deployment_flops_metrics,
    elementwise_weight_pruning_flops_metrics,
    prediction_propagation_flops,
)
from ougp.data import CitationGraph, apply_split_mode, load_graph_dataset
from ougp.deployment import HARDENING_MODES, MaterializedGCNDeployment, deployment_mask_set, deployment_masks
from ougp.tta import ensemble_log_probabilities, propagate_predictions
from ougp.para_tta import adapt_parameter_mask
from tta_energy_shift_probe import build_parser as build_exp121_parser
from method_efficiency import (
    gated_controller,
    measure_latency_distribution_ms,
    train_exp121_controller,
)
from run_tta_smoke import accuracy, set_seed, train_source_variant
from training_flops import source_training_flops, tta_training_flops


IID_CASE_DATASETS = {
    "iid_cora": "cora",
    "iid_pubmed": "pubmed",
    "iid_cora_full": "citationfull_cora",
    "iid_dblp": "citationfull_dblp",
    "iid_acm": "acm",
}


def route_unified_case() -> None:
    """Route an optional unified IID/OOD case before normal argument parsing."""

    route_parser = argparse.ArgumentParser(add_help=False)
    route_parser.add_argument("--case", choices=[*IID_CASE_DATASETS, "ood_dblp_acmv9"])
    route_parser.add_argument("--deployment-graph-policy", choices=["hardened", "full"], default="full")
    route_parser.add_argument("--latency-repeats", type=int, default=20)
    route_parser.add_argument("--latency-warmup", type=int, default=5)
    route_parser.add_argument("--tta-test-mode", choices=["standard_affine", "energy_gated"], default="energy_gated")
    route_args, remaining = route_parser.parse_known_args()
    if route_args.case is None:
        return

    if route_args.case in IID_CASE_DATASETS:
        sys.argv = [
            sys.argv[0],
            *remaining,
            "--dataset", IID_CASE_DATASETS[route_args.case],
            "--data-root", "data/raw",
            "--source-checkpoint-key", IID_CASE_DATASETS[route_args.case],
            "--deployment-graph-policy", route_args.deployment_graph_policy,
            "--latency-repeats", str(route_args.latency_repeats),
            "--latency-warmup", str(route_args.latency_warmup),
            "--tta-test-mode", route_args.tta_test_mode,
        ]
        return

    if route_args.deployment_graph_policy != "full":
        raise ValueError("The unified OOD protocol requires full target-graph deployment.")
    if route_args.tta_test_mode != "energy_gated":
        raise ValueError("The unified OOD protocol exposes the energy-gated target path only.")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_ood_citation_para_tta.py")),
        *remaining,
        "--source-dataset", "dblp",
        "--target-dataset", "acm",
        "--data-root", "data/raw/udagcn",
        "--source-checkpoint-key", "udagcn_dblp",
        "--profile-inference-repeats", str(route_args.latency_repeats),
        "--profile-warmup", str(route_args.latency_warmup),
    ]
    return_code = subprocess.run(command, check=False).returncode
    if return_code == 0 and "--out-dir" in remaining:
        out_dir = Path(remaining[remaining.index("--out-dir") + 1])
        source_result = out_dir / "ood_result.json"
        if source_result.exists():
            shutil.copyfile(source_result, out_dir / "result.json")
    raise SystemExit(return_code)


def stratified_split_2_1_1(dataset: CitationGraph, seed: int) -> CitationGraph:
    if dataset.y.ndim != 1:
        raise ValueError("The quantile-hardening experiment requires single-label node classification.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    for label in torch.unique(dataset.y.cpu(), sorted=True):
        indices = torch.where(dataset.y.cpu() == label)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_end = int(round(0.50 * indices.numel()))
        val_end = train_end + int(round(0.25 * indices.numel()))
        train_end = min(max(1, train_end), indices.numel() - 2)
        val_end = min(max(train_end + 1, val_end), indices.numel() - 1)
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
    return CitationGraph(
        x=dataset.x,
        y=dataset.y,
        edge_index=dataset.edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        task_type=dataset.task_type,
        metric_name=dataset.metric_name,
    )


def stratified_mask_partition(
    labels: torch.Tensor,
    source_mask: torch.Tensor,
    fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a labeled mask into deterministic stratified anchor and holdout masks."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("propagation anchor train fraction must be in (0, 1).")
    cpu_labels = labels.detach().cpu()
    cpu_mask = source_mask.detach().cpu().bool()
    anchor = torch.zeros_like(cpu_mask)
    holdout = torch.zeros_like(cpu_mask)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for label in torch.unique(cpu_labels[cpu_mask], sorted=True):
        indices = torch.where(cpu_mask & (cpu_labels == label))[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        if indices.numel() <= 1:
            anchor[indices] = True
            continue
        split = max(1, min(indices.numel() - 1, int(round(fraction * indices.numel()))))
        anchor[indices[:split]] = True
        holdout[indices[split:]] = True
    return anchor.to(source_mask.device), holdout.to(source_mask.device)


def retained_stats(values: torch.Tensor, support: torch.Tensor, prefix: str) -> dict[str, float]:
    retained = values.detach().float()[support.detach().bool()]
    if retained.numel() == 0:
        return {f"{prefix}_retained_min": 0.0, f"{prefix}_retained_mean": 0.0, f"{prefix}_retained_max": 0.0}
    return {
        f"{prefix}_retained_min": float(retained.min().item()),
        f"{prefix}_retained_mean": float(retained.mean().item()),
        f"{prefix}_retained_max": float(retained.max().item()),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    route_unified_case()
    parser = build_exp121_parser()
    parser.description = __doc__
    parser.set_defaults(
        experiment_name="EXP173_CORA_QUANTILE_HARDENING_SMOKE",
        out_dir="experiments/exp173_cora_quantile_hardening_smoke/seed0",
        dataset="cora",
        data_root="data/raw/planetoid",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        epochs=200,
        backbone="gcn",
        num_gnn_layers=2,
        hidden_dim=32,
        graph_sparsity=0.30,
        param_sparsity=0.30,
        param_pruning_granularity="weight",
        use_hidden_coupling=True,
        hidden_coupling_mix_graph=0.20,
        hidden_coupling_mix_param=0.20,
        tta_train_mask_ratio=0.20,
        tta_train_shift_scope="train_mask",
        tta_adaptation_nodes="train_val",
        tta_epochs=60,
        tta_lr=0.001,
        tta_loss_mode="unsupervised",
        tta_lambda_energy=1.0,
        tta_energy_negative_mask_ratio=0.50,
        tta_energy_clean_margin=0.10,
        tta_lambda_consistency=0.10,
        tta_consistency_mask_ratio=0.20,
        tta_lambda_anchor=0.10,
        tta_lambda_teacher=1.0,
        tta_lambda_state=0.0,
        tta_state_mode="stateless",
        tta_affine_mode="direct",
        tta_energy_reduction="train_mask",
        tta_consistency_reduction="train_mask",
        tta_gamma_a=0.10,
        tta_gamma_b=0.05,
        tta_center_delta_a=True,
        tta_normalize_delta_a=True,
        tta_center_delta_b=True,
        tta_normalize_delta_b=True,
        tta_delta_normalization_floor=0.05,
        tta_joint_energy_gate=True,
        tta_joint_energy_gate_train_min=0.25,
        energy_gate_mode="signed_z_tanh",
        energy_gate_min=0.0,
        energy_gate_max=1.0,
        energy_gate_center_z=0.20,
        energy_gate_temperature=1.0,
        energy_gate_polarity=1.0,
        profile_inference_repeats=0,
        profile_deployment_repeats=0,
    )
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--hardening-mode", choices=HARDENING_MODES, default="quantile_truncated")
    parser.add_argument(
        "--deployment-graph-policy",
        choices=("hardened", "full"),
        default="hardened",
    )
    parser.add_argument(
        "--tta-test-mode",
        choices=("standard_affine", "energy_gated"),
        default="standard_affine",
    )
    parser.add_argument(
        "--tta-protocol",
        choices=("legacy_source_tta", "forward_only", "para_tta"),
        default="legacy_source_tta",
        help="Select the legacy, forward-only state+affine, or Para-TTA protocol.",
    )
    parser.add_argument("--para-tta-score-scale", type=float, default=0.50)
    parser.add_argument("--para-tta-energy-gate", action="store_true", default=False)
    parser.add_argument("--para-tta-gate-temperature", type=float, default=1.0)
    parser.add_argument("--para-tta-eps", type=float, default=1e-6)
    args = parser.parse_args()
    if args.backbone != "gcn" or args.num_gnn_layers != 2:
        raise ValueError("The quantile-hardening experiment requires a 2-layer GCN.")

    args.seeds = [args.seed]
    args.mask_ratio = args.tta_train_mask_ratio
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = apply_split_mode(
        load_graph_dataset(args.data_root, args.dataset, split_seed=args.seed),
        args.split_mode,
        args.seed,
    )
    args.original_num_nodes = dataset.num_nodes
    args.original_num_edges = int(dataset.edge_index.size(1))
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    dense_model, dense_source, dense_source_history = train_source_variant(
        args, dataset, device, variant="dense"
    )
    ougp_model, ougp_source, ougp_source_history = train_source_variant(
        args, dataset, device, variant="ougp"
    )

    dense_graph, dense_channel = deployment_masks(dense_model)
    dense_deployment = MaterializedGCNDeployment(dense_model, dense_graph, dense_channel, x.dtype).to(device)

    masks = deployment_mask_set(ougp_model, mode=args.hardening_mode, temperature=args.temp_end)
    training_hard_masks = (
        ougp_model.get_training_hard_masks()
        if args.training_mask_mode == "periodic_ste"
        else None
    )
    if training_hard_masks is None:
        trained_graph_values = masks.graph.values
        deployment_parameter_values = masks.parameter.values
        deployment_parameter_support = masks.parameter.support
    else:
        trained_graph_values, deployment_parameter_values = training_hard_masks
        deployment_parameter_support = deployment_parameter_values.detach().bool()
    if args.deployment_graph_policy == "full":
        deployment_graph_values = torch.ones_like(masks.graph.values)
        deployment_graph_support = torch.ones_like(masks.graph.support, dtype=torch.bool)
    else:
        deployment_graph_values = trained_graph_values
        deployment_graph_support = trained_graph_values.detach().bool()
    ougp_deployment = MaterializedGCNDeployment(
        ougp_model,
        deployment_graph_values,
        deployment_parameter_values,
        x.dtype,
        graph_support=deployment_graph_support,
        param_support=deployment_parameter_support,
    ).to(device)
    for deployment in (dense_deployment, ougp_deployment):
        deployment.eval()
        for parameter in deployment.parameters():
            parameter.requires_grad_(False)

    if args.tta_protocol == "forward_only":
        if str(args.tta_state_mode) != "full":
            raise ValueError("forward_only TTA requires --tta-state-mode full.")
        if str(args.tta_affine_mode) != "readout":
            raise ValueError("forward_only TTA requires --tta-affine-mode readout.")
        if bool(getattr(args, "tta_output_mlp", False)):
            raise ValueError("forward_only TTA requires --no-tta-output-mlp.")
        if bool(getattr(args, "tta_prediction_propagation", False)):
            raise ValueError("forward_only TTA requires --no-tta-prediction-propagation.")

    para_tta_result = None
    para_deployment = None
    if args.tta_protocol == "para_tta":
        # Compare full source and target feature distributions without labels or masks.
        para_tta_result = adapt_parameter_mask(
            ougp_model,
            source_param_mask=deployment_parameter_values,
            source_x=x,
            target_x=x,
            source_node_mask=None,
            target_node_mask=None,
            score_scale=args.para_tta_score_scale,
            energy_gate=args.para_tta_energy_gate,
            gate_temperature=args.para_tta_gate_temperature,
            eps=args.para_tta_eps,
        )
        para_deployment = MaterializedGCNDeployment(
            ougp_model,
            deployment_graph_values,
            para_tta_result.param_mask,
            x.dtype,
            graph_support=deployment_graph_support,
            param_support=para_tta_result.param_mask.detach().bool(),
        ).to(device)
        para_deployment.eval()
        for parameter in para_deployment.parameters():
            parameter.requires_grad_(False)
        controller = None
        source_energy = {}
        tta_training = {
            "protocol": "para-tta-forward-only",
            "target_optimizer_steps": 0,
            "target_training": False,
            "tta_training_flops": 0.0,
            "tta_training_epochs": 0.0,
        }
        x_test = x
        tta_stats_mask = None
        test_controller = None
        test_energy_gate = 0.0
        test_energy_gate_stats = {}
        tta_label = "para-TTA energy-conditioned parameter-mask"
    else:
        teacher_logits_override = None
        if str(getattr(args, "tta_teacher_source", "ougp")) == "dense":
            with torch.no_grad():
                teacher_logits_override = dense_deployment(x).detach()
        controller, source_energy, tta_training = train_exp121_controller(
            args,
            ougp_deployment,
            x,
            y,
            train_mask,
            val_mask,
            teacher_logits_override=teacher_logits_override,
        )
        x_test = x
        tta_stats_mask = train_mask if args.tta_protocol == "forward_only" else test_mask
        if args.tta_test_mode == "energy_gated":
            test_controller, test_energy_gate = gated_controller(
                controller, ougp_deployment, x_test, tta_stats_mask, source_energy, args
            )
            tta_label = (
                "forward-only energy-gated state+affine TTA"
                if args.tta_protocol == "forward_only"
                else "stateless energy-gated affine TTA"
            )
        else:
            controller.reset_affine_strength()
            test_controller = controller
            test_energy_gate = 1.0
            tta_label = "stateless direct affine TTA"
        test_controller.eval()
        test_energy_gate_stats = test_controller.affine_strength_stats()

    @torch.no_grad()
    def dense_forward():
        return dense_deployment(x_test)

    @torch.no_grad()
    def ougp_forward():
        return ougp_deployment(x_test)

    propagation_runtime: dict[str, object] = {
        "enabled": False,
        "alpha": 0.0,
        "steps": 1,
        "mode": "convex",
        "components": [],
        "label_anchor": False,
    }

    @torch.no_grad()
    def tta_forward():
        if args.tta_protocol == "para_tta":
            return para_deployment(x_test)
        runtime_controller = test_controller
        if args.tta_test_mode == "energy_gated":
            runtime_controller, _ = gated_controller(
                controller,
                ougp_deployment,
                x_test,
                tta_stats_mask,
                source_energy,
                args,
            )
            runtime_controller.eval()
        if args.tta_protocol == "forward_only":
            runtime_controller.reset_state()
            logits = runtime_controller.test_time_forward(
                ougp_deployment,
                x_test,
                stats_node_mask=tta_stats_mask,
                update_state=True,
                return_hidden_states=False,
            )
        else:
            logits = ougp_deployment.forward_with_stateless_direct_tta(
                x_test,
                runtime_controller,
            )
        if bool(propagation_runtime["enabled"]):
            components = list(propagation_runtime["components"])
            logits = ensemble_log_probabilities(
                [
                    propagate_predictions(
                        logits,
                        dataset.edge_index.to(device),
                        float(component["alpha"]),
                        int(component["steps"]),
                        mode=str(propagation_runtime["mode"]),
                        labels=y if bool(component.get("label_anchor", False)) else None,
                        anchor_mask=(train_mask | val_mask)
                        if bool(component.get("label_anchor", False))
                        else None,
                    )
                    for component in components
                ]
            )
        return logits

    with torch.no_grad():
        dense_logits = dense_forward()
        ougp_logits = ougp_forward()
        tta_logits = tta_forward()
        fixed_clean_logits, _ = ougp_model(
            x,
            temperature=args.temp_end,
            fixed_masks=(deployment_graph_values, deployment_parameter_values),
        )
        materialized_clean_logits = ougp_deployment(x)

    propagation_summary: dict[str, object] = {
        "enabled": False,
        "alpha": 0.0,
        "steps": 1,
        "validation": [],
    }
    if bool(getattr(args, "tta_prediction_propagation", False)):
        ensemble_topk = int(getattr(args, "tta_propagation_ensemble_topk", 1))
        label_anchor = bool(getattr(args, "tta_propagation_label_anchor", False))
        anchor_positive_only = bool(
            getattr(args, "tta_propagation_label_anchor_positive_only", False)
        )
        relative_validation = bool(
            getattr(args, "tta_propagation_relative_validation", False)
        )
        if ensemble_topk <= 0:
            raise ValueError("tta_propagation_ensemble_topk must be positive.")
        anchor_fraction = float(
            getattr(args, "tta_propagation_anchor_train_fraction", 0.8)
        )
        train_anchor_mask, train_holdout_mask = stratified_mask_partition(
            y,
            train_mask,
            anchor_fraction,
            args.seed + 173,
        )
        validation_rows: list[dict[str, object]] = []
        candidate_rows: list[tuple[float, float, float, int, torch.Tensor, bool]] = []
        for alpha in args.tta_propagation_alpha_candidates:
            for steps in args.tta_propagation_step_candidates:
                component_anchor = label_anchor and (
                    not anchor_positive_only or float(alpha) > 0.0
                )
                selection_mask = (
                    train_holdout_mask
                    if component_anchor and relative_validation
                    else val_mask
                )
                component_anchor_mask = (
                    train_anchor_mask
                    if component_anchor and relative_validation
                    else train_mask if component_anchor else None
                )
                candidate = propagate_predictions(
                    tta_logits,
                    dataset.edge_index.to(device),
                    alpha,
                    steps,
                    mode=args.tta_propagation_mode,
                    labels=y if component_anchor else None,
                    anchor_mask=component_anchor_mask,
                )
                value = accuracy(candidate, y, selection_mask)
                baseline_value = accuracy(tta_logits, y, selection_mask)
                selection_score = value - baseline_value if relative_validation else value
                validation_rows.append(
                    {
                        "alpha": float(alpha),
                        "steps": float(steps),
                        "validation_accuracy": value,
                        "selection_score": selection_score,
                        "label_anchor": component_anchor,
                        "selection_nodes": int(selection_mask.sum().item()),
                    }
                )
                candidate_rows.append(
                    (
                        selection_score,
                        value,
                        float(alpha),
                        int(steps),
                        candidate,
                        component_anchor,
                    )
                )
        selected = sorted(candidate_rows, key=lambda row: row[0], reverse=True)[:ensemble_topk]
        components = [
            {
                "alpha": row[2],
                "steps": row[3],
                "validation_accuracy": row[1],
                "selection_score": row[0],
                "label_anchor": row[5],
            }
            for row in selected
        ]
        tta_logits = ensemble_log_probabilities(
            [
                propagate_predictions(
                    tta_logits,
                    dataset.edge_index.to(device),
                    component["alpha"],
                    component["steps"],
                    mode=args.tta_propagation_mode,
                    labels=y if bool(component["label_anchor"]) else None,
                    anchor_mask=(train_mask | val_mask)
                    if bool(component["label_anchor"])
                    else None,
                )
                for component in components
            ]
        )
        best_score, best_accuracy, best_alpha, best_steps, _, _ = selected[0]
        propagation_summary = {
            "enabled": True,
            "mode": args.tta_propagation_mode,
            "alpha": best_alpha,
            "steps": best_steps,
            "ensemble_topk": len(components),
            "components": components,
            "label_anchor": label_anchor,
            "label_anchor_positive_only": anchor_positive_only,
            "relative_validation": relative_validation,
            "best_selection_score": best_score,
            "anchor_train_fraction": anchor_fraction,
            "selection_anchor_nodes": int(train_anchor_mask.sum().item())
            if label_anchor and relative_validation
            else int(train_mask.sum().item()) if label_anchor else 0,
            "test_anchor_nodes": int((train_mask | val_mask).sum().item()) if label_anchor else 0,
            "best_validation_accuracy": best_accuracy,
            "validation": validation_rows,
        }
        propagation_runtime.update(
            {
                "enabled": True,
                "mode": args.tta_propagation_mode,
                "alpha": best_alpha,
                "steps": best_steps,
                "components": components,
                "label_anchor": label_anchor,
            }
        )

    original_non_loop_edge_count = int(
        (dataset.edge_index[0] != dataset.edge_index[1]).sum().item()
    )

    dense_cost = deployment_flops_metrics(
        num_nodes=dataset.num_nodes,
        input_dim=dataset.num_features,
        original_hidden_dim=args.hidden_dim,
        deployed_hidden_dim=dense_deployment.materialized_channel_count,
        output_dim=dataset.num_classes,
        original_edge_count=original_non_loop_edge_count,
        deployed_edge_count=dense_deployment.materialized_edge_count,
        hidden_layers=dense_deployment.hidden_layer_count,
        flops_reduction_source="unpruned Dense GCN deployment",
    )
    if args.param_pruning_granularity == "weight":
        ougp_cost = elementwise_weight_pruning_flops_metrics(
            num_nodes=dataset.num_nodes,
            input_dim=dataset.num_features,
            hidden_dim=args.hidden_dim,
            output_dim=dataset.num_classes,
            original_edge_count=original_non_loop_edge_count,
            deployed_edge_count=ougp_deployment.materialized_edge_count,
            retained_parameter_count=ougp_deployment.materialized_parameter_count,
            total_parameter_count=ougp_deployment.materialized_parameter_total,
            hidden_layers=ougp_deployment.hidden_layer_count,
        )
    else:
        ougp_cost = deployment_flops_metrics(
            num_nodes=dataset.num_nodes,
            input_dim=dataset.num_features,
            original_hidden_dim=args.hidden_dim,
            deployed_hidden_dim=ougp_deployment.materialized_channel_count,
            output_dim=dataset.num_classes,
            original_edge_count=original_non_loop_edge_count,
            deployed_edge_count=ougp_deployment.materialized_edge_count,
            hidden_layers=ougp_deployment.hidden_layer_count,
            flops_reduction_source=f"{args.hardening_mode} graph support and hidden channels",
        )
    tta_elementwise_ops = (
        0.0
        if args.tta_protocol == "para_tta"
        else float(
            2
            * dataset.num_nodes
            * ougp_deployment.materialized_channel_count
            * ougp_deployment.hidden_layer_count
        )
    )
    gate_probe_flops = (
        float(ougp_cost["deployed_flops"])
        if args.tta_test_mode == "energy_gated"
        else 0.0
    )
    energy_gate_flops = (
        float(3 * dataset.num_nodes * dataset.num_classes)
        if args.tta_test_mode == "energy_gated"
        else 0.0
    )
    logit_residual_flops = 0.0
    if bool(getattr(args, "tta_logit_residual", False)):
        logit_residual_flops = float(
            2
            * dataset.num_nodes
            * ougp_deployment.materialized_channel_count
            * dataset.num_classes
            + 3 * dataset.num_nodes * dataset.num_classes
        )
    logit_affine_flops = (
        float(2 * dataset.num_nodes * dataset.num_classes)
        if bool(getattr(args, "tta_logit_affine", False))
        else 0.0
    )
    output_mlp_flops = 0.0
    if args.tta_protocol != "para_tta" and bool(getattr(args, "tta_output_mlp", False)):
        output_mlp_flops = float(
            4
            * dataset.num_nodes
            * ougp_deployment.materialized_channel_count
            * int(args.tta_output_mlp_rank)
            + 2 * dataset.num_nodes * ougp_deployment.materialized_channel_count
        )
    tta_cost = dict(ougp_cost)
    tta_cost["tta_elementwise_ops"] = tta_elementwise_ops
    tta_cost["energy_gate_probe_flops"] = gate_probe_flops
    tta_cost["energy_gate_reduction_flops"] = energy_gate_flops
    tta_cost["logit_residual_flops"] = logit_residual_flops
    tta_cost["logit_affine_flops"] = logit_affine_flops
    tta_cost["output_mlp_flops"] = output_mlp_flops
    propagation_flops = 0.0
    propagation_ensemble_merge_flops = 0.0
    propagation_label_anchor_flops = 0.0
    if bool(propagation_runtime["enabled"]):
        components = list(propagation_runtime["components"])
        propagation_flops = sum(
            prediction_propagation_flops(
                num_nodes=dataset.num_nodes,
                edge_count_with_self_loops=original_non_loop_edge_count + dataset.num_nodes,
                num_classes=dataset.num_classes,
                steps=int(component["steps"]),
                mode=str(propagation_runtime["mode"]),
            )
            for component in components
        )
        anchored_components = [
            component
            for component in components
            if bool(component.get("label_anchor", False))
        ]
        if anchored_components:
            test_anchor_nodes = int((train_mask | val_mask).sum().item())
            propagation_label_anchor_flops = float(
                sum(int(component["steps"]) for component in anchored_components)
                * test_anchor_nodes
                * dataset.num_classes
            )
        if len(components) > 1:
            propagation_ensemble_merge_flops = float(
                (2 * len(components) + 2) * dataset.num_nodes * dataset.num_classes
            )
        propagation_flops += propagation_label_anchor_flops + propagation_ensemble_merge_flops
    tta_cost["prediction_propagation_flops"] = propagation_flops
    tta_cost["propagation_ensemble_merge_flops"] = propagation_ensemble_merge_flops
    tta_cost["propagation_label_anchor_flops"] = propagation_label_anchor_flops
    tta_cost["fixed_self_loop_message_count_per_layer"] = float(dataset.num_nodes)
    adapted_forward_flops = (
        float(ougp_cost["deployed_flops"])
        + tta_elementwise_ops
        + logit_residual_flops
        + logit_affine_flops
        + output_mlp_flops
    )
    output_forward_flops = adapted_forward_flops + propagation_flops
    validation_forward_flops = (
        gate_probe_flops
        + energy_gate_flops
        + output_forward_flops
    )
    clean_energy_reference_flops = (
        float(ougp_cost["deployed_flops"])
        if float(getattr(args, "tta_lambda_energy", 0.0)) > 0.0
        and float(getattr(args, "tta_energy_negative_mask_ratio", 0.5)) == 0.0
        else 0.0
    )
    train_forward_flops = validation_forward_flops + clean_energy_reference_flops
    test_forward_flops = (
        gate_probe_flops
        + energy_gate_flops
        + adapted_forward_flops
        + propagation_flops
    )
    dense_reference_flops = float(ougp_cost["dense_reference_flops"])
    tta_cost["adapted_forward_flops"] = adapted_forward_flops
    tta_cost["adapted_forward_flops_reduction"] = 1.0 - adapted_forward_flops / dense_reference_flops
    tta_cost["train_forward_flops_per_epoch"] = train_forward_flops
    tta_cost["train_forward_ratio_vs_dense_forward"] = train_forward_flops / dense_reference_flops
    tta_cost["validation_forward_flops"] = validation_forward_flops
    tta_cost["validation_forward_ratio_vs_dense_forward"] = validation_forward_flops / dense_reference_flops
    tta_cost["test_forward_flops"] = test_forward_flops
    tta_cost["test_forward_ratio_vs_dense_forward"] = test_forward_flops / dense_reference_flops
    tta_cost["test_forward_flops_reduction"] = 1.0 - test_forward_flops / dense_reference_flops
    # Preserve the conventional one-output-forward metric used by pruning tables.
    tta_cost["deployed_flops"] = output_forward_flops
    tta_cost["flops_reduction"] = 1.0 - output_forward_flops / dense_reference_flops
    if args.tta_protocol == "para_tta":
        tta_cost["flops_reduction_source"] = (
            f"single Para-TTA output forward: {args.hardening_mode} compact backbone, "
            "fixed self-loops, forward-only parameter-mask selection"
        )
    else:
        tta_cost["flops_reduction_source"] = (
            f"single adapted output forward: {args.hardening_mode} compact backbone, fixed self-loops, "
            "stateless affine, logit correction, and prediction propagation"
        )

    dense_training = source_training_flops(
        history=dense_source_history,
        num_nodes=dataset.num_nodes,
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=dataset.num_classes,
        original_non_loop_edges=original_non_loop_edge_count,
        hidden_layers=dense_deployment.hidden_layer_count,
        param_granularity=args.param_pruning_granularity,
        use_hidden_coupling=False,
        use_memory=False,
        variant="dense",
    )
    ougp_training = source_training_flops(
        history=ougp_source_history,
        num_nodes=dataset.num_nodes,
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=dataset.num_classes,
        original_non_loop_edges=original_non_loop_edge_count,
        hidden_layers=ougp_deployment.hidden_layer_count,
        param_granularity=args.param_pruning_granularity,
        use_hidden_coupling=bool(args.use_hidden_coupling),
        use_memory=True,
        variant="ougp",
        hidden_coupling_interval=args.hidden_coupling_interval,
    )
    if args.tta_protocol == "para_tta":
        tta_training = {
            "tta_training_flops": 0.0,
            "tta_training_epochs": 0.0,
            "training_flops_scope": "source OUGP training only; Para-TTA is forward-only",
        }
    else:
        tta_training = tta_training_flops(
            args=vars(args),
            compact_forward_flops=float(ougp_cost["deployed_flops"]),
            adapted_forward_flops=adapted_forward_flops,
        )
    dense_training_total = float(dense_training["source_training_flops"])
    ougp_training_total = float(ougp_training["source_training_flops"])
    tta_training_total = ougp_training_total + float(tta_training["tta_training_flops"])
    training_by_method = (
        {
            **dense_training,
            "tta_training_flops": 0.0,
            "tta_training_epochs": 0.0,
            "training_flops": dense_training_total,
            "training_flops_ratio": 1.0,
            "training_flops_scope": "source training only; excludes validation/test/deployment",
        },
        {
            **ougp_training,
            "tta_training_flops": 0.0,
            "tta_training_epochs": 0.0,
            "training_flops": ougp_training_total,
            "training_flops_ratio": ougp_training_total / max(dense_training_total, 1.0),
            "training_flops_scope": "source training forward + estimated backward + reused hidden-state utility; excludes validation/test",
        },
        {
            **ougp_training,
            **tta_training,
            "training_flops": tta_training_total,
            "training_flops_ratio": tta_training_total / max(dense_training_total, 1.0),
            "training_flops_scope": (
                "OUGP source training + forward-only Para-TTA mask selection; excludes validation/test/deployment"
                if args.tta_protocol == "para_tta"
                else "OUGP source training + TTA optimization; excludes validation/test/deployment"
            ),
        },
    )

    hardening_label = args.hardening_mode.replace("_", "-")
    graph_policy_label = "full-graph test" if args.deployment_graph_policy == "full" else "pruned-graph test"
    rows = []
    for (method, logits, forward, cost), training_metrics in zip(
        (
            ("Dense", dense_logits, dense_forward, dense_cost),
            (f"OUGP {hardening_label} ({graph_policy_label})", ougp_logits, ougp_forward, ougp_cost),
            (
                f"OUGP {hardening_label} ({graph_policy_label}) + {tta_label}",
                tta_logits,
                tta_forward,
                tta_cost,
            ),
        ),
        training_by_method,
    ):
        rows.append(
            {
                "method": method,
                "clean_test_accuracy": accuracy(logits, y, test_mask),
                "test_nodes": int(test_mask.sum().item()),
                "test_latency_scope": "full-graph forward used for clean test evaluation",
                **measure_latency_distribution_ms(
                    forward, device, args.latency_repeats, args.latency_warmup
                ),
                **cost,
                **training_metrics,
            }
        )

    payload = {
        "experiment": args.experiment_name,
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint_selection": args.checkpoint_selection,
        "training_mask_mode": args.training_mask_mode,
        "mask_hardening_interval": args.mask_hardening_interval,
        "split": {
            "ratio": "2:1:1",
            "train_nodes": int(train_mask.sum().item()),
            "val_nodes": int(val_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
            "stratified": True,
        },
        "hardening": {
            "mode": args.hardening_mode,
            "graph_threshold": masks.graph.threshold,
            "parameter_threshold": masks.parameter.threshold,
            "parameter_granularity": args.param_pruning_granularity,
            "deployment_graph_policy": args.deployment_graph_policy,
            "trained_graph_hard_sparsity": 1.0 - float(trained_graph_values.float().mean().item()),
            "graph_realized_sparsity": 1.0 - float(deployment_graph_support.float().mean().item()),
            "deployment_edge_count": ougp_deployment.materialized_edge_count,
            "original_edge_count": int(dataset.edge_index.size(1)),
            "parameter_realized_sparsity": 1.0 - float(deployment_parameter_support.float().mean().item()),
            "reused_training_hard_mask": training_hard_masks is not None,
            "fixed_vs_materialized_max_abs": float(
                (fixed_clean_logits - materialized_clean_logits).abs().max().item()
            ),
            **retained_stats(masks.graph.values, masks.graph.support, "graph"),
            **retained_stats(deployment_parameter_values, deployment_parameter_support, "parameter"),
        },
        "test_feature_mask_ratio": 0.0,
        "tta_test_mode": tta_label,
        "tta_test_protocol": args.tta_test_mode,
        "tta_protocol": args.tta_protocol,
        "tta_gate_scope": (
            "target_input_feature_energy" if args.tta_protocol == "para_tta"
            else "train_mask" if args.tta_protocol == "forward_only" else "test_mask"
        ),
        "tta_interface": (
            "para_tta_forward_mask" if args.tta_protocol == "para_tta"
            else "test_time_forward" if args.tta_protocol == "forward_only"
            else "forward_with_stateless_direct_tta"
        ),
        "source_controller_pretraining": args.tta_protocol == "forward_only",
        "target_tta_training": False if args.tta_protocol in {"forward_only", "para_tta"} else None,
        "tta_training_scope": (
            "none_forward_only_mask_selection" if args.tta_protocol == "para_tta"
            else "source_controller_pretraining" if args.tta_protocol == "forward_only"
            else "legacy_controller_training"
        ),
        "tta_test_energy_gate": test_energy_gate,
        "tta_test_energy_gate_stats": test_energy_gate_stats,
        "tta_selected_gate_scale": (
            controller.selected_gate_scale if controller is not None else 0.0
        ),
        "tta_gate_learn_mode": args.tta_gate_learn_mode,
        "tta_teacher_source": args.tta_teacher_source,
        "tta_energy_clean_margin": args.tta_energy_clean_margin,
        "energy_gate_mode": args.energy_gate_mode,
        "energy_gate_center_z": args.energy_gate_center_z,
        "dense_source_result": dense_source,
        "ougp_source_result": ougp_source,
        "tta_training_summary": tta_training,
        "prediction_propagation_summary": propagation_summary,
        "para_tta_summary": para_tta_result.summary() if para_tta_result is not None else None,
        "para_tta_energy_gate": bool(args.para_tta_energy_gate) if args.tta_protocol == "para_tta" else None,
        "para_tta_gate_scope": "target_only" if args.tta_protocol == "para_tta" else None,
        "rows": rows,
    }
    if para_tta_result is not None:
        (out_dir / "para_tta_diagnostics.json").write_text(
            json.dumps(para_tta_result.diagnostics(), indent=2) + "\n", encoding="utf-8"
        )
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_rows(out_dir / "results.csv", rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
