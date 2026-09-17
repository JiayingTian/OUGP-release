"""EXP130: deployment efficiency of dense and materialized OUGP with EXP121 TTA."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ougp.data import load_graph_dataset
from ougp.deployment import MaterializedGCNDeployment, deployment_masks
from ougp.tta import feature_masking_shift, signed_z_tanh_gate_values
from tta_source_energy_anchor import (
    clone_controller_with_gamma,
    energy,
    energy_gate,
    make_controller,
    source_energy_stats,
)
from tta_energy_shift_probe import build_parser as build_exp121_parser
from run_tta_smoke import accuracy, set_seed, train_source_variant, train_tta_controller


def all_ones_masks(model, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.ones(model.cfg.num_edges, device=device, dtype=torch.float32),
        torch.ones(model.cfg.hidden_dim, device=device, dtype=torch.float32),
    )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency_distribution_ms(fn, device: torch.device, repeats: int, warmup: int) -> dict[str, float]:
    samples_ms: list[float] = []
    for _ in range(max(0, warmup)):
        fn()
    sync_if_cuda(device)
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        sync_if_cuda(device)
        samples_ms.append(1000.0 * (time.perf_counter() - start))
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(np.median(values)),
        "latency_std_ms": float(values.std(ddof=0)),
        "latency_p95_ms": float(np.percentile(values, 95)),
    }


def measure_peak_gpu_memory_mb(fn, device: torch.device) -> float:
    if device.type != "cuda":
        fn()
        return 0.0
    sync_if_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    sync_if_cuda(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def evaluate_method(
    name: str,
    deployment: MaterializedGCNDeployment,
    x_shifted: torch.Tensor,
    y: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    latency_repeats: int,
    latency_warmup: int,
    theoretical_flops: float,
    controller=None,
    stats_node_mask: torch.Tensor | None = None,
    tta_elementwise_ops: float = 0.0,
) -> dict[str, float | str]:
    deployment.eval()
    if controller is None:
        def run_forward():
            return deployment(x_shifted)
    else:
        controller.eval()

        def run_forward():
            # EXP121 deployment is stateless + direct: no state reset, Q/K/V,
            # readout, hidden statistics, or diagnostic reports are needed.
            if (
                controller.cfg.state_mode == "stateless"
                and controller.cfg.affine_mode == "direct"
                and hasattr(deployment, "forward_with_stateless_direct_tta")
            ):
                return deployment.forward_with_stateless_direct_tta(x_shifted, controller)
            controller.reset_state()
            return controller.test_time_forward(
                deployment,
                x_shifted,
                stats_node_mask=stats_node_mask,
                update_state=True,
            )

    with torch.no_grad():
        logits = run_forward()
    latency = measure_latency_distribution_ms(run_forward, device, latency_repeats, latency_warmup)
    peak_memory = measure_peak_gpu_memory_mb(run_forward, device)
    return {
        "method": name,
        "accuracy": accuracy(logits, y, test_mask),
        "peak_gpu_memory_mb": peak_memory,
        "theoretical_flops": float(theoretical_flops),
        "backbone_macs": float(theoretical_flops - tta_elementwise_ops) / 2.0,
        "tta_elementwise_ops": float(tta_elementwise_ops),
        "inference_path": (
            "materialized_gcn+stateless_direct_affine"
            if controller is not None
            else "materialized_gcn"
        ),
        **latency,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_exp121_parser()
    parser.set_defaults(
        out_dir="experiments/method_efficiency/seed0",
        epochs=200,
        hidden_dim=32,
        backbone="gcn",
        num_gnn_layers=2,
        graph_sparsity=0.30,
        param_sparsity=0.30,
        tta_state_mode="stateless",
        tta_affine_mode="direct",
    )
    parser.add_argument("--shift-mask-ratio", type=float, default=0.60)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--latency-warmup", type=int, default=20)
    return parser


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def materialize(model, graph_mask: torch.Tensor, channel_mask: torch.Tensor, x: torch.Tensor, device: torch.device):
    sync_if_cuda(device)
    start = time.perf_counter()
    deployment = MaterializedGCNDeployment(model, graph_mask, channel_mask, x.dtype).to(device)
    sync_if_cuda(device)
    return deployment, time.perf_counter() - start


def gcn_macs(deployment: MaterializedGCNDeployment, num_nodes: int, in_dim: int, out_dim: int) -> float:
    """MACs for sparse aggregation plus dense linear maps in this deployment."""

    channels = deployment.materialized_channel_count
    edges = int(deployment.adj._nnz())
    hidden_layers = deployment.hidden_layer_count
    aggregation = edges * in_dim + edges * channels * hidden_layers
    linear = num_nodes * in_dim * channels
    linear += max(0, hidden_layers - 1) * num_nodes * channels * channels
    linear += num_nodes * channels * out_dim
    return float(aggregation + linear)


def direct_tta_elementwise_ops(deployment: MaterializedGCNDeployment, num_nodes: int) -> float:
    """Elementwise operations actually executed by stateless direct affine.

    Each adapted hidden value performs one multiply and one add.  The fast
    deployment path applies this to every node at every hidden layer; it does
    not execute statistics, Q/K/V projections, state updates, or readout.
    """

    channels = deployment.materialized_channel_count
    layers = deployment.hidden_layer_count
    return float(2 * num_nodes * channels * layers)


def train_exp121_controller(
    args,
    deployment,
    x,
    y,
    train_mask,
    val_mask,
    teacher_logits_override=None,
):
    with torch.no_grad():
        source_logits = deployment(x)
    source_mask = train_mask | val_mask
    stats = source_energy_stats(source_logits, train_mask, val_mask)
    controller = make_controller(deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(x.device)
    train_mask_for_controller = source_mask if args.tta_adaptation_nodes == "train_val" else train_mask
    training = train_tta_controller(
        args,
        deployment,
        controller,
        x,
        y,
        train_mask_for_controller,
        source_energy_stats=stats,
        energy_gate_fn=lambda target: energy_gate(target, stats, args),
        energy_gate_mask=source_mask,
        validation_mask=val_mask,
        teacher_logits_override=teacher_logits_override,
    )
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller, stats, training


def gated_controller(controller, deployment, x_shifted, test_mask, source_stats, args):
    with torch.no_grad():
        all_target_energy = energy(deployment(x_shifted))
        target_energy = all_target_energy[test_mask]
    if str(getattr(args, "tta_gate_learn_mode", "mean")) == "node":
        if str(args.energy_gate_mode) != "signed_z_tanh":
            raise ValueError("tta_gate_learn_mode=node requires energy_gate_mode=signed_z_tanh.")
        node_gates = signed_z_tanh_gate_values(
            all_target_energy,
            source_stats["source_energy_mean"],
            source_stats["source_energy_std"],
            args.energy_gate_center_z,
            args.energy_gate_temperature,
            args.energy_gate_max,
            getattr(args, "energy_gate_polarity", 1.0),
        )
        applied = torch.zeros_like(node_gates)
        selected = test_mask.to(device=node_gates.device, dtype=torch.bool)
        applied[selected] = node_gates[selected]
        applied.mul_(controller.selected_gate_scale)
        cloned = clone_controller_with_gamma(controller, args.tta_gamma_a, args.tta_gamma_b)
        cloned.set_node_affine_strength(applied)
        selected_values = applied[selected]
        return cloned, float(selected_values.mean().item()) if selected_values.numel() else 0.0
    gate = energy_gate({"target_energy_mean": float(target_energy.mean().item())}, source_stats, args)
    gate *= controller.selected_gate_scale
    cloned = clone_controller_with_gamma(
        controller,
        args.tta_gamma_a,
        args.tta_gamma_b,
    )
    cloned.set_affine_strength(gate)
    return cloned, float(gate)


def write_csv(path: Path, seed: int, rows: list[dict[str, float | str]]) -> None:
    fields = ["seed"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{"seed": seed, **row} for row in rows])


def main() -> None:
    args = build_parser().parse_args()
    if args.backbone != "gcn" or args.num_gnn_layers != 2:
        raise ValueError("EXP130 is deployment-aligned and currently requires a 2-layer GCN.")
    args.seeds = [args.seed]
    args.mask_ratio = args.tta_train_mask_ratio
    device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_graph_dataset(args.data_root, args.dataset, split_seed=args.seed)
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    dense_model, dense_source, _ = train_source_variant(args, dataset, device, variant="dense")
    compact_model, compact_source, _ = train_source_variant(args, dataset, device, variant="ougp")
    dense_deployment, dense_materialization_sec = materialize(
        dense_model, *all_ones_masks(dense_model, device), x, device
    )
    compact_deployment, compact_materialization_sec = materialize(
        compact_model, *deployment_masks(compact_model), x, device
    )
    for deployment in (dense_deployment, compact_deployment):
        deployment.eval()
        for parameter in deployment.parameters():
            parameter.requires_grad_(False)

    dense_controller, dense_source_energy, dense_tta_train = train_exp121_controller(
        args, dense_deployment, x, y, train_mask, val_mask
    )
    compact_controller, compact_source_energy, compact_tta_train = train_exp121_controller(
        args, compact_deployment, x, y, train_mask, val_mask
    )

    shift_generator = torch.Generator(device=device).manual_seed(args.seed + 13000)
    x_shifted = feature_masking_shift(
        x, args.shift_mask_ratio, generator=shift_generator, node_mask=test_mask
    )
    dense_gated, dense_gate = gated_controller(
        dense_controller, dense_deployment, x_shifted, test_mask, dense_source_energy, args
    )
    compact_gated, compact_gate = gated_controller(
        compact_controller, compact_deployment, x_shifted, test_mask, compact_source_energy, args
    )

    base_dense_macs = gcn_macs(dense_deployment, dataset.num_nodes, dataset.num_features, dataset.num_classes)
    base_compact_macs = gcn_macs(compact_deployment, dataset.num_nodes, dataset.num_features, dataset.num_classes)
    dense_tta_ops = direct_tta_elementwise_ops(dense_deployment, dataset.num_nodes)
    compact_tta_ops = direct_tta_elementwise_ops(compact_deployment, dataset.num_nodes)
    methods = [
        evaluate_method("Dense", dense_deployment, x_shifted, y, test_mask, device, args.latency_repeats, args.latency_warmup, 2.0 * base_dense_macs),
        evaluate_method("Dense + EXP121 TTA", dense_deployment, x_shifted, y, test_mask, device, args.latency_repeats, args.latency_warmup, 2.0 * base_dense_macs + dense_tta_ops, dense_gated, test_mask, dense_tta_ops),
        evaluate_method("Frozen Materialized OUGP", compact_deployment, x_shifted, y, test_mask, device, args.latency_repeats, args.latency_warmup, 2.0 * base_compact_macs),
        evaluate_method("Frozen Materialized OUGP + EXP121 TTA", compact_deployment, x_shifted, y, test_mask, device, args.latency_repeats, args.latency_warmup, 2.0 * base_compact_macs + compact_tta_ops, compact_gated, test_mask, compact_tta_ops),
    ]
    for method in methods:
        method["macs"] = float(method["backbone_macs"]) + float(method["tta_elementwise_ops"]) / 2.0

    result = {
        "experiment": "method_efficiency",
        "dataset": args.dataset,
        "seed": args.seed,
        "shift_mask_ratio": args.shift_mask_ratio,
        "source": {"dense": dense_source, "ougp": compact_source},
        "materialization_sec": {"dense": dense_materialization_sec, "compact": compact_materialization_sec},
        "energy_gate": {"dense": dense_gate, "compact": compact_gate},
        "controller_training": {"dense": dense_tta_train, "compact": compact_tta_train},
        "methods": methods,
    }
    (out_dir / "efficiency_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "efficiency_metrics.csv", args.seed, methods)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
