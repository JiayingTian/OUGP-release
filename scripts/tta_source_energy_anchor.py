"""EXP108 source-energy anchored affine-only TTA on Cora GCN2.

This runner keeps the main OUGP + LHCM training/deployment chain intact:
source OUGP training -> hard materialization -> source-side affine controller
pretraining -> forward-only target evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ougp.data import load_graph_dataset
from ougp.deployment import MaterializedGCNDeployment, deployment_masks
from ougp.tta import (
    LayerwiseHiddenAffineTTA,
    TTAConfig,
    feature_masking_shift,
    signed_z_tanh_gate_values,
)
from scripts.train_ougp import build_arg_parser as build_case_study_arg_parser
from run_tta_smoke import (
    accuracy,
    latency_ms,
    module_param_delta_l2_norm,
    module_param_l2_norm,
    set_seed,
    summarize_reports,
    train_source_ougp,
    train_tta_controller,
)


def energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits.detach().float(), dim=-1)


def tensor_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_p50": float(torch.quantile(values, 0.50).item()),
        f"{prefix}_p75": float(torch.quantile(values, 0.75).item()),
        f"{prefix}_p90": float(torch.quantile(values, 0.90).item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def source_energy_stats(logits: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor) -> dict[str, float]:
    source_mask = train_mask.to(dtype=torch.bool) | val_mask.to(dtype=torch.bool)
    values = energy(logits)[source_mask]
    return tensor_stats(values, "source_energy")


def percentile_energy_gate(target_mean: float, source_p50: float, source_p95: float, gate_min: float, gate_max: float) -> float:
    denom = max(source_p95 - source_p50, 1e-12)
    gate = (target_mean - source_p50) / denom
    return float(min(gate_max, max(gate_min, gate)))


def z_sigmoid_energy_gate(
    target_mean: float,
    source_mean: float,
    source_std: float,
    center_z: float,
    temperature: float,
    gate_min: float,
    gate_max: float,
) -> float:
    z = (target_mean - source_mean) / max(source_std, 1e-12)
    raw = 1.0 / (1.0 + np.exp(-(z - center_z) / max(temperature, 1e-12)))
    return float(gate_min + (gate_max - gate_min) * raw)


def absolute_z_sigmoid_energy_gate(
    target_mean: float,
    source_mean: float,
    source_std: float,
    center_z: float,
    temperature: float,
    gate_min: float,
    gate_max: float,
) -> float:
    """Gate by the magnitude, rather than direction, of source-energy deviation."""

    z_shift = abs(target_mean - source_mean) / max(source_std, 1e-12)
    raw = 1.0 / (1.0 + np.exp(-(z_shift - center_z) / max(temperature, 1e-12)))
    return float(gate_min + (gate_max - gate_min) * raw)


def compatibility_rbf_energy_gate(
    target_mean: float,
    source_mean: float,
    source_std: float,
    center_z: float,
    temperature: float,
    gate_min: float,
    gate_max: float,
) -> float:
    """Trust a source-trained correction only near the source energy domain."""

    z_shift = abs(target_mean - source_mean) / max(source_std, 1e-12)
    residual = max(z_shift - float(center_z), 0.0)
    scaled = residual / max(float(temperature), 1e-12)
    raw = float(np.exp(-0.5 * scaled * scaled))
    return float(gate_min + (gate_max - gate_min) * raw)


def signed_z_tanh_energy_gate(
    target_mean: float,
    source_mean: float,
    source_std: float,
    center_z: float,
    temperature: float,
    gate_max: float,
    polarity: float,
) -> float:
    """Signed gate: magnitude is shift size, sign is energy-shift direction."""

    value = signed_z_tanh_gate_values(
        torch.tensor([target_mean]),
        source_mean,
        source_std,
        center_z,
        temperature,
        gate_max,
        polarity,
    )
    return float(value.item())


def energy_gate(target_stats: dict[str, float], source_stats: dict[str, float], args: argparse.Namespace) -> float:
    if args.energy_gate_mode == "percentile":
        return percentile_energy_gate(
            target_stats["target_energy_mean"],
            source_stats["source_energy_p50"],
            source_stats["source_energy_p95"],
            args.energy_gate_min,
            args.energy_gate_max,
        )
    if args.energy_gate_mode == "signed_z_tanh":
        return signed_z_tanh_energy_gate(
            target_stats["target_energy_mean"],
            source_stats["source_energy_mean"],
            source_stats["source_energy_std"],
            args.energy_gate_center_z,
            args.energy_gate_temperature,
            args.energy_gate_max,
            getattr(args, "energy_gate_polarity", 1.0),
        )
    if args.energy_gate_mode == "compatibility_rbf":
        return compatibility_rbf_energy_gate(
            target_stats["target_energy_mean"],
            source_stats["source_energy_mean"],
            source_stats["source_energy_std"],
            args.energy_gate_center_z,
            args.energy_gate_temperature,
            args.energy_gate_min,
            args.energy_gate_max,
        )
    gate_fn = absolute_z_sigmoid_energy_gate if args.energy_gate_mode == "absolute_z_sigmoid" else z_sigmoid_energy_gate
    return gate_fn(
        target_stats["target_energy_mean"],
        source_stats["source_energy_mean"],
        source_stats["source_energy_std"],
        args.energy_gate_center_z,
        args.energy_gate_temperature,
        args.energy_gate_min,
        args.energy_gate_max,
    )


def make_controller(
    deployment: MaterializedGCNDeployment,
    args: argparse.Namespace,
    gamma_a: float,
    gamma_b: float,
) -> LayerwiseHiddenAffineTTA:
    cfg = TTAConfig(
        rank=args.tta_rank,
        gamma_a=gamma_a,
        gamma_b=gamma_b,
        state_mode=getattr(args, "tta_state_mode", "stateless"),
        state_sharing=getattr(args, "tta_state_sharing", "per_layer"),
        state_input_mode=getattr(args, "tta_state_input_mode", "stats"),
        affine_mode=getattr(args, "tta_affine_mode", "direct"),
        center_delta_a=args.tta_center_delta_a,
        normalize_delta_a=args.tta_normalize_delta_a,
        center_delta_b=args.tta_center_delta_b,
        normalize_delta_b=args.tta_normalize_delta_b,
        delta_normalization_floor=getattr(args, "tta_delta_normalization_floor", 0.05),
        use_logit_affine=bool(getattr(args, "tta_logit_affine", False)),
        gamma_logit_scale=float(getattr(args, "tta_gamma_logit_scale", 0.10)),
        gamma_logit_bias=float(getattr(args, "tta_gamma_logit_bias", 0.10)),
        use_logit_residual=bool(getattr(args, "tta_logit_residual", False)),
        gamma_logit_residual=float(getattr(args, "tta_gamma_logit_residual", 1.0)),
        output_gate_mode=str(getattr(args, "tta_output_gate_mode", "signed")),
        learn_gate_calibration=bool(getattr(args, "tta_learn_gate_calibration", False)),
        gate_calibration_slope_init=float(
            getattr(args, "tta_gate_calibration_slope_init", 1.0)
        ),
        gate_calibration_bias_init=float(
            getattr(args, "tta_gate_calibration_bias_init", 0.0)
        ),
        gate_calibration_amplitude_init=float(
            getattr(args, "tta_gate_calibration_amplitude_init", 0.99)
        ),
        use_output_mlp=bool(getattr(args, "tta_output_mlp", False)),
        output_mlp_rank=int(getattr(args, "tta_output_mlp_rank", 16)),
        gamma_output_mlp=float(getattr(args, "tta_gamma_output_mlp", 1.0)),
    )
    return LayerwiseHiddenAffineTTA(
        deployment.hidden_layer_count,
        deployment.materialized_channel_count,
        cfg,
        output_dim=int(deployment.lin2_weight.size(0)),
    )


def clone_controller_with_gamma(
    trained: LayerwiseHiddenAffineTTA,
    gamma_a: float,
    gamma_b: float,
) -> LayerwiseHiddenAffineTTA:
    cloned = LayerwiseHiddenAffineTTA(
        trained.num_layers,
        trained.channel_dim,
        replace(trained.cfg, gamma_a=gamma_a, gamma_b=gamma_b),
        output_dim=trained.output_dim,
    ).to(next(trained.parameters()).device)
    cloned.load_state_dict(trained.state_dict(), strict=True)
    cloned.set_selected_gate_scale(trained.selected_gate_scale)
    cloned.reset_state()
    return cloned


@torch.no_grad()
def evaluate_controller(
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x_shifted: torch.Tensor,
    y: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[float, float, dict[str, float]]:
    controller.eval()
    controller.reset_state()
    logits, _, reports = controller.test_time_forward(
        deployment,
        x_shifted,
        stats_node_mask=test_mask,
        update_state=True,
        return_hidden_states=True,
    )
    latency = latency_ms(
        lambda: (
            controller.reset_state(),
            controller.test_time_forward(deployment, x_shifted, stats_node_mask=test_mask, update_state=True),
        )[1],
        device,
        repeats,
        warmup,
    )
    return accuracy(logits, y, test_mask), latency, summarize_reports(reports)


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = build_case_study_arg_parser()
    parser.set_defaults(
        out_dir="experiments/tta_source_energy_anchor/seed0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset="cora",
        data_root="data/raw/planetoid",
        epochs=200,
        hidden_dim=32,
        backbone="gcn",
        num_gnn_layers=2,
        memory_rank=8,
        graph_sparsity=0.30,
        param_sparsity=0.30,
        sparsity_lambda=0.08,
        graph_memory_layout="multi",
        param_memory_layout="multi",
        graph_score_init="topofeat",
        param_score_init="magnitude",
        use_hidden_coupling=True,
        hidden_coupling_mix_graph=0.20,
        hidden_coupling_mix_param=0.20,
        profile_inference_repeats=0,
        profile_deployment_repeats=0,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", default="tta_source_energy_anchor")
    parser.add_argument("--mask-sweep-ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--tta-train-mask-ratio", type=float, default=0.20)
    parser.add_argument("--tta-epochs", type=int, default=60)
    parser.add_argument("--tta-rank", type=int, default=8)
    parser.add_argument("--tta-lr", type=float, default=1e-3)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-loss-mode", choices=["supervised", "unsupervised"], default="unsupervised")
    parser.add_argument("--tta-lambda-energy", type=float, default=1.0)
    parser.add_argument("--tta-energy-negative-mask-ratio", type=float, default=0.50)
    parser.add_argument("--tta-energy-clean-margin", type=float, default=0.10)
    parser.add_argument("--tta-lambda-consistency", type=float, default=0.10)
    parser.add_argument("--tta-consistency-mask-ratio", type=float, default=0.20)
    parser.add_argument("--tta-lambda-anchor", type=float, default=0.10)
    parser.add_argument("--tta-lambda-state", type=float, default=0.0)
    parser.add_argument("--tta-energy-reduction", choices=["train_mask", "all"], default="train_mask")
    parser.add_argument("--tta-consistency-reduction", choices=["train_mask", "all"], default="train_mask")
    parser.add_argument("--tta-gamma-a", type=float, default=0.10)
    parser.add_argument("--tta-gamma-b", type=float, default=0.10)
    parser.add_argument("--tta-center-delta-a", action="store_true")
    parser.add_argument("--tta-normalize-delta-a", action="store_true")
    parser.add_argument("--tta-center-delta-b", action="store_true")
    parser.add_argument("--tta-normalize-delta-b", action="store_true")
    parser.add_argument("--energy-gate-mode", choices=["percentile", "z_sigmoid", "absolute_z_sigmoid"], default="percentile")
    parser.add_argument("--energy-gate-min", type=float, default=0.0)
    parser.add_argument("--energy-gate-max", type=float, default=1.0)
    parser.add_argument("--energy-gate-center-z", type=float, default=0.0)
    parser.add_argument("--energy-gate-temperature", type=float, default=1.0)
    parser.add_argument("--tta-joint-energy-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--latency-warmup", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.seeds = [args.seed]
    args.mask_ratio = args.tta_train_mask_ratio
    if args.backbone != "gcn" or args.num_gnn_layers != 2:
        raise ValueError("EXP108 is defined for 2-layer GCN materialized deployment.")

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_graph_dataset(args.data_root, args.dataset)
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    model, source_result, source_history = train_source_ougp(args, dataset, device)
    graph_mask, channel_mask = deployment_masks(model)
    deployment = MaterializedGCNDeployment(model, graph_mask, channel_mask, x.dtype).to(device)
    deployment.eval()
    for param in deployment.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        clean_logits = deployment(x)
    src_energy = source_energy_stats(clean_logits, train_mask, val_mask)

    controller = make_controller(deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    controller_param_norm_before = module_param_l2_norm(controller)
    controller_before = {name: param.detach().clone() for name, param in controller.named_parameters()}
    source_energy_mask = train_mask | val_mask
    train_tta_controller(
        args,
        deployment,
        controller,
        x,
        y,
        train_mask,
        source_energy_stats=src_energy,
        energy_gate_fn=lambda target: energy_gate(target, src_energy, args),
        energy_gate_mask=source_energy_mask,
    )
    controller_param_norm_after = module_param_l2_norm(controller)
    controller_training_drift = module_param_delta_l2_norm(controller_before, controller)

    rows: list[dict[str, float | int | str]] = []
    for offset, mask_ratio in enumerate(args.mask_sweep_ratios):
        generator = torch.Generator(device=device).manual_seed(args.seed + 10800 + offset)
        x_shifted = feature_masking_shift(x, mask_ratio, generator=generator, node_mask=test_mask)
        with torch.no_grad():
            frozen_logits = deployment(x_shifted)
        target_energy_values = energy(frozen_logits)[test_mask]
        target_stats = tensor_stats(target_energy_values, "target_energy")
        gate = energy_gate(target_stats, src_energy, args)
        gated_gamma_a = float(args.tta_gamma_a * gate)
        gated_gamma_b = float(args.tta_gamma_b * gate)

        base_acc, base_latency, base_report = evaluate_controller(
            deployment,
            controller,
            x_shifted,
            y,
            test_mask,
            device,
            args.latency_repeats,
            args.latency_warmup,
        )
        gated_controller = clone_controller_with_gamma(controller, gated_gamma_a, gated_gamma_b)
        gated_acc, gated_latency, gated_report = evaluate_controller(
            deployment,
            gated_controller,
            x_shifted,
            y,
            test_mask,
            device,
            args.latency_repeats,
            args.latency_warmup,
        )
        frozen_latency = latency_ms(lambda: deployment(x_shifted), device, args.latency_repeats, args.latency_warmup)
        source_std = max(src_energy["source_energy_std"], 1e-12)
        row: dict[str, float | int | str] = {
            "dataset": args.dataset,
            "seed": args.seed,
            "mask_ratio": float(mask_ratio),
            "graph_sparsity_target": float(args.graph_sparsity),
            "channel_sparsity_target": float(args.param_sparsity),
            "frozen_materialized_ougp_acc": accuracy(frozen_logits, y, test_mask),
            "affine_loss_acc": base_acc,
            "source_energy_anchor_affine_acc": gated_acc,
            "affine_loss_gain": base_acc - accuracy(frozen_logits, y, test_mask),
            "source_energy_anchor_gain": gated_acc - accuracy(frozen_logits, y, test_mask),
            "frozen_materialized_ougp_latency_ms": frozen_latency,
            "affine_loss_latency_ms": base_latency,
            "source_energy_anchor_latency_ms": gated_latency,
            "energy_gate": gate,
            "effective_gamma_a": gated_gamma_a,
            "effective_gamma_b": gated_gamma_b,
            "target_energy_z_gap": (target_stats["target_energy_mean"] - src_energy["source_energy_mean"]) / source_std,
            **target_stats,
        }
        row.update({f"affine_loss_{key}": value for key, value in base_report.items()})
        row.update({f"source_energy_anchor_{key}": value for key, value in gated_report.items()})
        rows.append(row)

    result = {
        "experiment": args.experiment_name,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "num_gnn_layers": args.num_gnn_layers,
        "seed": args.seed,
        "source_epochs": args.epochs,
        "tta_epochs": args.tta_epochs,
        "tta_loss_mode": args.tta_loss_mode,
        "tta_affine_mode": "direct",
        "tta_state_mode": "stateless",
        "tta_gamma_a": args.tta_gamma_a,
        "tta_gamma_b": args.tta_gamma_b,
        "tta_lambda_energy": args.tta_lambda_energy,
        "tta_energy_clean_margin": args.tta_energy_clean_margin,
        "tta_lambda_consistency": args.tta_lambda_consistency,
        "tta_lambda_anchor": args.tta_lambda_anchor,
        "tta_lambda_state": args.tta_lambda_state,
        "energy_gate_mode": args.energy_gate_mode,
        "energy_gate_min": args.energy_gate_min,
        "energy_gate_max": args.energy_gate_max,
        "energy_gate_center_z": args.energy_gate_center_z,
        "energy_gate_temperature": args.energy_gate_temperature,
        "source_energy_stats": src_energy,
        "source_case_study_result": source_result,
        "source_case_study_final_epoch": source_history[-1] if source_history else {},
        "source_cfg": asdict(model.cfg),
        "clean_frozen_materialized_ougp_acc": accuracy(clean_logits, y, test_mask),
        "materialized_kept_edges": deployment.materialized_edge_count,
        "materialized_kept_edges_with_self_loops": deployment.materialized_edge_count_with_self_loops,
        "materialized_kept_channels": deployment.materialized_channel_count,
        "controller_param_norm_before_training": controller_param_norm_before,
        "controller_param_norm_after_training": controller_param_norm_after,
        "controller_training_drift_l2": controller_training_drift,
        "mask_sweep": rows,
    }
    result_path = out_dir / "source_energy_tta_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_rows(out_dir / "source_energy_tta_sweep.csv", rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
