#!/usr/bin/env python3
"""Twitch OOD benchmark for forward-only Para-Mask TTA.

The source model is trained with the existing EXP245 OUGP path.  At target
test time, unlabeled target features produce a channel-wise energy shift and
re-rank the fixed-budget parameter mask.  No target loss, backward pass,
optimizer, controller, output MLP, or prediction propagation is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deployed_costs import elementwise_weight_pruning_flops_metrics
from ougp.deployment import MaterializedGCNDeployment, deployment_mask_set
from ougp.para_tta import adapt_parameter_mask
from ougp.twitch import TWITCH_DOMAINS, load_twitch_lodo
from tta_energy_shift_probe import build_parser as build_tta_parser
from run_tta_smoke import accuracy, set_seed, train_source_variant


def build_245_args(args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    """Build the existing EXP245 OUGP source-training configuration."""

    tta = build_tta_parser().parse_args([])
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
        "tta_epochs": 0,
        "tta_output_mlp": False,
        "tta_prediction_propagation": False,
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


def measure_latency(forward_fn, device: torch.device, warmup: int, repeats: int) -> dict[str, float | int]:
    for _ in range(max(0, int(warmup))):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "target_latency_ms": (time.perf_counter() - start) * 1000.0 / max(1, int(repeats)),
        "latency_repeats": int(repeats),
    }


def _deployment(
    model,
    graph,
    x: torch.Tensor,
    parameter_values: torch.Tensor,
    device: torch.device,
) -> MaterializedGCNDeployment:
    support = parameter_values.detach().bool()
    deployment = MaterializedGCNDeployment(
        model,
        torch.ones(graph.edge_index.size(1), device=device),
        parameter_values,
        x.dtype,
        edge_index_override=graph.edge_index.to(device),
        num_nodes_override=graph.num_nodes,
        graph_support=torch.ones(
            graph.edge_index.size(1), device=device, dtype=torch.bool
        ),
        param_support=support,
    ).to(device)
    deployment.eval()
    for parameter in deployment.parameters():
        parameter.requires_grad_(False)
    return deployment


def run_para_tta(args: argparse.Namespace, source, target, device: torch.device) -> dict[str, Any]:
    tta_args = build_245_args(args, Path(args.out_dir))
    tta_args.original_num_nodes = source.num_nodes
    tta_args.original_num_edges = int(source.edge_index.size(1))

    # This is the unchanged source-side OUGP training path.  The Para-Mask
    # branch starts only after this model and its source mask are finalized.
    set_seed(args.seed)
    ougp_model, source_result, source_history = train_source_variant(
        tta_args, source, device, variant="ougp"
    )
    source_x = source.x.to(device)
    source_y = source.y.to(device)
    target_x = target.x.to(device)
    target_y = target.y.to(device)

    masks = deployment_mask_set(
        ougp_model,
        mode="quantile_binary",
        temperature=getattr(tta_args, "temp_end", 0.5),
    )
    hard_masks = ougp_model.get_training_hard_masks()
    source_parameter_values = (
        hard_masks[1] if hard_masks is not None else masks.parameter.values
    )
    source_deployment = _deployment(
        ougp_model, source, source_x, source_parameter_values, device
    )
    frozen_target_deployment = _deployment(
        ougp_model, target, target_x, source_parameter_values, device
    )

    # Forward-only target adaptation.  Both node masks remain None: the
    # protocol uses the available target feature matrix and no target labels.
    para_result = adapt_parameter_mask(
        ougp_model,
        source_param_mask=source_parameter_values,
        source_x=source_x,
        target_x=target_x,
        source_node_mask=None,
        target_node_mask=None,
        score_scale=args.para_tta_score_scale,
        energy_gate=args.para_tta_energy_gate,
        gate_temperature=args.para_tta_gate_temperature,
        eps=args.para_tta_eps,
    )
    para_target_deployment = _deployment(
        ougp_model, target, target_x, para_result.param_mask, device
    )

    with torch.no_grad():
        source_logits = source_deployment(source_x)
        frozen_target_logits = frozen_target_deployment(target_x)
        para_target_logits = para_target_deployment(target_x)

    source_acc = accuracy(source_logits, source_y, source.test_mask.to(device))
    frozen_target_acc = accuracy(
        frozen_target_logits, target_y, target.test_mask.to(device)
    )
    para_target_acc = accuracy(
        para_target_logits, target_y, target.test_mask.to(device)
    )

    latency = measure_latency(
        lambda: para_target_deployment(target_x),
        device,
        args.latency_warmup,
        args.latency_repeats,
    )
    non_loop_edges = int((target.edge_index[0] != target.edge_index[1]).sum().item())
    support = para_result.param_mask.detach().bool()
    cost = elementwise_weight_pruning_flops_metrics(
        num_nodes=target.num_nodes,
        input_dim=target.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=target.num_classes,
        original_edge_count=non_loop_edges,
        deployed_edge_count=non_loop_edges,
        retained_parameter_count=int(support.sum().item()),
        total_parameter_count=int(support.numel()),
        hidden_layers=1,
    )
    para_summary = para_result.summary()

    # Keep a separate diagnostic artifact so the result table stays compact.
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "para_tta_diagnostics.json").write_text(
        json.dumps(para_result.diagnostics(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "method": "ougp_para_tta",
        "method_label": "OUGP+Para-Mask TTA",
        "experiment_protocol": "EXP245 source OUGP + forward-only channel-energy Para-Mask TTA",
        "source_domains": list(args.source_domains),
        "target_domain": args.target_domain,
        "source_num_nodes": source.num_nodes,
        "source_num_edges": int(source.edge_index.size(1)),
        "target_num_nodes": target.num_nodes,
        "target_num_edges": int(target.edge_index.size(1)),
        "input_dim": source.num_features,
        "hidden_dim": args.hidden_dim,
        "source_test_accuracy": source_acc,
        "frozen_ougp_target_accuracy": frozen_target_acc,
        "target_accuracy": para_target_acc,
        "para_tta_gain_vs_frozen_ougp": para_target_acc - frozen_target_acc,
        "deployment_mode": "full target graph",
        "parameter_granularity": "weight-entry",
        "source_parameter_keep_rate": para_summary["source_keep_rate"],
        "target_parameter_keep_rate": para_summary["target_keep_rate"],
        "parameter_sparsity": 1.0 - para_summary["target_keep_rate"],
        "mask_churn": para_summary["mask_churn"],
        "energy_gate_enabled": para_summary["energy_gate_enabled"],
        "energy_gate": para_summary["gate"],
        "energy_gate_scope": "target_only",
        "para_tta": para_summary,
        "target_training": False,
        "target_labels_used_for_adaptation": False,
        "target_train_mask_used_for_adaptation": False,
        "optimizer_steps": 0,
        "output_mlp": False,
        "prediction_propagation": False,
        "controller": False,
        "source_result": source_result,
        "source_history_epochs": len(source_history),
        "hardware_note": "hidden_dim kept at the existing Twitch protocol value 32",
        "implementation_status": "release standalone Twitch Para-Mask path",
        **latency,
        **cost,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="EXP278_TWITCH_PARA_MASK_H200")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--target-domain", choices=TWITCH_DOMAINS, required=True)
    parser.add_argument("--source-domains", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--para-tta-score-scale", type=float, default=0.50)
    parser.add_argument("--para-tta-energy-gate", action="store_true", default=False)
    parser.add_argument("--para-tta-gate-temperature", type=float, default=1.0)
    parser.add_argument("--para-tta-eps", type=float, default=1e-6)
    parser.add_argument("--latency-warmup", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.source_domains = [domain for domain in TWITCH_DOMAINS if domain != args.target_domain]
    device = torch.device(args.device)
    source, target, _ = load_twitch_lodo(
        args.data_root, args.target_domain, split_seed=args.seed
    )
    result = run_para_tta(args, source, target, device)
    result.update(
        {
            "experiment": args.experiment_name,
            "seed": args.seed,
            "split_mode": "stratified_2_1_1",
            "split_ratio": "2:1:1",
            "data_protocol": "clean",
            "target_graph_policy": "full_target_graph",
            "source_domains": args.source_domains,
            "target_domain": args.target_domain,
            "config": vars(args),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
