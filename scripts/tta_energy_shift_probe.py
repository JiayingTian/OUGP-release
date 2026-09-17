"""EXP109: test whether source-energy TTA responds usefully to feature shift.

Each dataset trains OUGP + LHCM once, materializes it once, and reuses that
fixed deployment for every shift and TTA ablation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from deployed_costs import deployment_flops_metrics
from ougp.data import load_graph_dataset
from ougp.deployment import (
    MaterializedGCNDeployment,
    SoftMaskedGCNDeployment,
    deployment_masks,
    soft_deployment_masks,
)
from ougp.tta import LayerwiseHiddenAffineTTA, feature_masking_shift, signed_z_tanh_gate_values
from tta_source_energy_anchor import (
    clone_controller_with_gamma,
    energy,
    energy_gate,
    make_controller,
    source_energy_stats,
    tensor_stats,
)
from scripts.train_ougp import (
    VARIANTS,
    build_arg_parser as build_case_study_arg_parser,
    maybe_sample_nodes,
)
from run_tta_smoke import (
    accuracy,
    module_param_delta_l2_norm,
    set_seed,
    summarize_reports,
    train_source_variant,
    train_tta_controller,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_case_study_arg_parser()
    # Keep the standard benchmark choices while exposing the local UDAGCN
    # citation-domain adapters used by the DBLPv8/ACMv9 experiments.
    for action in parser._actions:
        if action.dest == "dataset" and action.choices is not None:
            action.choices = list(action.choices) + ["udagcn_dblp", "udagcn_acm"]
            break
    parser.set_defaults(
        out_dir="experiments/tta_energy_shift_probe/seed0",
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
    parser.add_argument("--source-variant", choices=sorted(VARIANTS), default="ougp")
    parser.add_argument(
        "--include-dense-baseline",
        action="store_true",
        help="Train and evaluate a Dense GCN on the same source split and test views.",
    )
    parser.add_argument("--experiment-name", default="tta_energy_shift_probe")
    parser.add_argument(
        "--deployment-mode",
        choices=["hard_materialized", "soft_mask_diagnostic"],
        default="hard_materialized",
    )
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument(
        "--masking-seed-offset",
        type=int,
        default=None,
        help="Optional seed offset shared by every feature-masking evaluation view.",
    )
    parser.add_argument("--noise-sigmas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--scale-factors", nargs="+", type=float, default=[0.8, 1.2, 1.5])
    parser.add_argument(
        "--shift-types",
        nargs="+",
        choices=["masking", "gaussian_noise", "feature_scaling"],
        default=["masking", "gaussian_noise", "feature_scaling"],
    )
    parser.add_argument(
        "--gaussian-noise-mode",
        choices=["absolute", "node_relative"],
        default="absolute",
    )
    parser.add_argument("--tta-train-mask-ratio", type=float, default=0.20)
    parser.add_argument("--tta-epochs", type=int, default=60)
    parser.add_argument("--tta-log-every", type=int, default=1)
    parser.add_argument(
        "--tta-checkpoint-selection",
        choices=["last", "best_val"],
        default="last",
    )
    parser.add_argument(
        "--tta-strength-selection",
        choices=["fixed", "best_val"],
        default="fixed",
    )
    parser.add_argument(
        "--tta-strength-candidates",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--tta-rank", type=int, default=8)
    parser.add_argument("--source-checkpoint-root", type=str, default="")
    parser.add_argument(
        "--source-checkpoint-policy",
        choices=["off", "reuse", "refresh", "read_write"],
        default="off",
    )
    parser.add_argument("--source-checkpoint-key", type=str, default="")
    parser.add_argument("--tta-lr", type=float, default=1e-3)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-loss-mode", choices=["supervised", "unsupervised"], default="unsupervised")
    parser.add_argument("--tta-lambda-logits", type=float, default=1.0)
    parser.add_argument("--tta-lambda-energy", type=float, default=1.0)
    parser.add_argument("--tta-energy-negative-mask-ratio", type=float, default=0.50)
    parser.add_argument("--tta-energy-clean-margin", type=float, default=0.10)
    parser.add_argument("--tta-lambda-consistency", type=float, default=0.10)
    parser.add_argument("--tta-consistency-mask-ratio", type=float, default=0.20)
    parser.add_argument("--tta-lambda-anchor", type=float, default=0.10)
    parser.add_argument("--tta-lambda-state", type=float, default=0.0)
    parser.add_argument("--tta-state-mode", choices=["full", "stateless"], default="stateless")
    parser.add_argument("--tta-state-sharing", choices=["per_layer", "shared"], default="per_layer")
    parser.add_argument(
        "--tta-state-input-mode",
        choices=["stats", "hidden", "hidden_attention", "channel_attention"],
        default="stats",
    )
    parser.add_argument("--tta-affine-mode", choices=["direct", "readout"], default="direct")
    parser.add_argument("--tta-lambda-teacher", type=float, default=0.0)
    parser.add_argument(
        "--tta-teacher-source",
        choices=["ougp", "dense"],
        default="ougp",
    )
    parser.add_argument("--tta-lambda-entropy", type=float, default=0.0)
    parser.add_argument("--tta-train-shift-scope", choices=["all", "train_mask"], default="all")
    parser.add_argument("--tta-adaptation-nodes", choices=["train", "train_val"], default="train")
    parser.add_argument("--tta-energy-reduction", choices=["train_mask", "all"], default="train_mask")
    parser.add_argument("--tta-consistency-reduction", choices=["train_mask", "all"], default="train_mask")
    parser.add_argument("--tta-gamma-a", type=float, default=0.10)
    parser.add_argument("--tta-gamma-b", type=float, default=0.10)
    parser.add_argument("--tta-logit-affine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tta-gamma-logit-scale", type=float, default=0.10)
    parser.add_argument("--tta-gamma-logit-bias", type=float, default=0.10)
    parser.add_argument("--tta-logit-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tta-gamma-logit-residual", type=float, default=1.0)
    parser.add_argument(
        "--tta-output-gate-mode",
        choices=["signed", "absolute", "none"],
        default="signed",
    )
    parser.add_argument(
        "--tta-learn-gate-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tta-gate-calibration-slope-init", type=float, default=1.0)
    parser.add_argument("--tta-gate-calibration-bias-init", type=float, default=0.0)
    parser.add_argument("--tta-gate-calibration-amplitude-init", type=float, default=0.99)
    parser.add_argument("--tta-output-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tta-output-mlp-rank", type=int, default=16)
    parser.add_argument("--tta-gamma-output-mlp", type=float, default=1.0)
    parser.add_argument("--tta-target-adaptation-epochs", type=int, default=0)
    parser.add_argument("--tta-target-adaptation-lr", type=float, default=1e-4)
    parser.add_argument("--tta-target-gate-scale", type=float, default=0.20)
    parser.add_argument("--tta-target-lambda-entropy", type=float, default=1.0)
    parser.add_argument("--tta-target-lambda-diversity", type=float, default=1.0)
    parser.add_argument("--tta-target-lambda-teacher", type=float, default=1.0)
    parser.add_argument("--tta-target-lambda-anchor", type=float, default=0.01)
    parser.add_argument(
        "--tta-target-reset-controller",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tta-prediction-propagation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tta-propagation-alpha-candidates",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.4, 0.6],
    )
    parser.add_argument(
        "--tta-propagation-mode",
        choices=["convex", "logspace_signed"],
        default="convex",
    )
    parser.add_argument(
        "--tta-propagation-step-candidates",
        nargs="+",
        type=int,
        default=[1, 2, 4],
    )
    parser.add_argument("--tta-propagation-ensemble-topk", type=int, default=1)
    parser.add_argument(
        "--tta-propagation-label-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tta-propagation-label-anchor-positive-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tta-propagation-relative-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tta-propagation-anchor-train-fraction", type=float, default=0.8)
    parser.add_argument("--tta-center-delta-a", action="store_true")
    parser.add_argument("--tta-normalize-delta-a", action="store_true")
    parser.add_argument("--tta-center-delta-b", action="store_true")
    parser.add_argument("--tta-normalize-delta-b", action="store_true")
    parser.add_argument("--tta-delta-normalization-floor", type=float, default=0.05)
    parser.add_argument(
        "--energy-gate-mode",
        choices=["percentile", "z_sigmoid", "absolute_z_sigmoid", "signed_z_tanh", "compatibility_rbf"],
        default="percentile",
    )
    parser.add_argument("--energy-gate-min", type=float, default=0.0)
    parser.add_argument("--energy-gate-max", type=float, default=1.0)
    parser.add_argument("--energy-gate-center-z", type=float, default=0.0)
    parser.add_argument("--energy-gate-temperature", type=float, default=1.0)
    parser.add_argument("--energy-gate-polarity", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--tta-joint-energy-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tta-joint-energy-gate-train-min", type=float, default=0.0)
    parser.add_argument(
        "--tta-gate-learn-mode",
        choices=["mean", "node"],
        default="mean",
    )
    return parser


def shifted_features(
    x: torch.Tensor,
    test_mask: torch.Tensor,
    shift_type: str,
    level: float,
    seed: int,
    gaussian_noise_mode: str = "absolute",
) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    selected = test_mask.to(device=x.device, dtype=torch.bool)
    if shift_type == "clean":
        return x.clone()
    if shift_type == "masking":
        return feature_masking_shift(x, level, generator=generator, node_mask=selected)
    shifted = x.clone()
    if shift_type == "gaussian_noise":
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        selected_noise = noise[selected]
        if gaussian_noise_mode == "node_relative":
            feature_norm = x[selected].norm(dim=1, keepdim=True)
            noise_direction = selected_noise / selected_noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
            selected_noise = float(level) * feature_norm * noise_direction
        elif gaussian_noise_mode == "absolute":
            selected_noise = float(level) * selected_noise
        else:
            raise ValueError(f"Unknown Gaussian noise mode: {gaussian_noise_mode}")
        shifted[selected] = shifted[selected] + selected_noise
        return shifted
    if shift_type == "feature_scaling":
        shifted[selected] = shifted[selected] * float(level)
        return shifted
    raise ValueError(f"Unknown shift type: {shift_type}")


@torch.no_grad()
def evaluate_controller(
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    test_mask: torch.Tensor,
) -> tuple[float, torch.Tensor, dict[str, float]]:
    controller.eval()
    controller.reset_state()
    logits, _, reports = controller.test_time_forward(
        deployment,
        x,
        stats_node_mask=test_mask,
        update_state=True,
        return_hidden_states=True,
    )
    summary = summarize_reports(reports)
    keep = {
        "delta_a_norm_mean",
        "delta_a_mean_mean",
        "delta_a_std_mean",
        "delta_a_min_mean",
        "delta_a_max_mean",
        "delta_a_rms_mean",
        "delta_b_norm_mean",
        "delta_b_mean_mean",
        "delta_b_std_mean",
        "delta_b_min_mean",
        "delta_b_max_mean",
        "delta_b_rms_mean",
        "delta_a_for_scale_mean_mean",
        "delta_a_for_scale_std_mean",
        "delta_a_for_scale_min_mean",
        "delta_a_for_scale_max_mean",
        "delta_a_for_scale_rms_mean",
        "delta_b_for_bias_mean_mean",
        "delta_b_for_bias_std_mean",
        "delta_b_for_bias_min_mean",
        "delta_b_for_bias_max_mean",
        "delta_b_for_bias_rms_mean",
        "scale_mean_mean",
        "scale_std_mean",
        "scale_min_mean",
        "scale_max_mean",
        "scale_above_one_ratio_mean",
        "scale_below_one_ratio_mean",
        "bias_to_hidden_ratio_mean",
        "hidden_delta_norm_ratio_mean",
        "state_norm_mean",
        "state_write_norm_mean",
        "state_delta_norm_mean",
        "novelty_mean",
    }
    return accuracy(logits, y, test_mask), logits, {
        key: value for key, value in summary.items() if key in keep
    }


def energy_margin(logits: torch.Tensor, negative_logits: torch.Tensor, mask: torch.Tensor) -> float:
    target_mean = energy(logits)[mask].mean()
    negative_mean = energy(negative_logits)[mask].mean()
    return float((negative_mean - target_mean).item())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def decision_diagnostics(
    frozen_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Measure whether an affine update reaches the classifier decision boundary."""

    selected = mask.to(device=frozen_logits.device, dtype=torch.bool)
    frozen = frozen_logits[selected].float()
    adapted = adapted_logits[selected].float()
    labels = y[selected]
    frozen_pred = frozen.argmax(dim=-1)
    adapted_pred = adapted.argmax(dim=-1)
    changed = frozen_pred != adapted_pred
    frozen_correct = frozen_pred == labels
    adapted_correct = adapted_pred == labels
    sorted_frozen = frozen.topk(k=2, dim=-1).values
    sorted_adapted = adapted.topk(k=2, dim=-1).values
    return {
        "logit_max_abs_shift": float((adapted - frozen).abs().max().item()),
        "logit_mean_abs_shift": float((adapted - frozen).abs().mean().item()),
        "prediction_flip_count": float(changed.sum().item()),
        "prediction_flip_ratio": float(changed.float().mean().item()),
        "wrong_to_correct_count": float((changed & ~frozen_correct & adapted_correct).sum().item()),
        "correct_to_wrong_count": float((changed & frozen_correct & ~adapted_correct).sum().item()),
        "top1_margin_mean_delta": float(
            ((sorted_adapted[:, 0] - sorted_adapted[:, 1]) - (sorted_frozen[:, 0] - sorted_frozen[:, 1]))
            .mean()
            .item()
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.seeds = [args.seed]
    args.mask_ratio = args.tta_train_mask_ratio
    if args.backbone != "gcn" or args.num_gnn_layers != 2:
        raise ValueError("EXP109 requires a 2-layer GCN.")

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    if args.dataset.startswith("udagcn_") and not (data_root / "acm").exists():
        data_root = data_root / "udagcn"
    dataset = load_graph_dataset(data_root, args.dataset, split_seed=args.seed)
    args.original_num_nodes = int(dataset.num_nodes)
    args.original_num_edges = int(dataset.edge_index.size(1))
    dataset = maybe_sample_nodes(
        dataset,
        args.node_sample_size,
        args.node_sample_seed,
        args.node_sample_mode,
    )
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    dense_source_result = None
    dense_source_history = None
    dense_deployment = None
    dense_deployment_cost = None
    if args.include_dense_baseline:
        dense_model, dense_source_result, dense_source_history = train_source_variant(
            args, dataset, device, variant="dense"
        )
        dense_graph_mask, dense_channel_mask = deployment_masks(dense_model)
        dense_deployment = MaterializedGCNDeployment(
            dense_model, dense_graph_mask, dense_channel_mask, x.dtype
        ).to(device)
        dense_deployment.eval()
        for parameter in dense_deployment.parameters():
            parameter.requires_grad_(False)
        dense_deployment_cost = deployment_flops_metrics(
            num_nodes=dataset.num_nodes,
            input_dim=dataset.num_features,
            original_hidden_dim=args.hidden_dim,
            deployed_hidden_dim=dense_deployment.materialized_channel_count,
            output_dim=dataset.num_classes,
            original_edge_count=int(dataset.edge_index.size(1)),
            deployed_edge_count=dense_deployment.materialized_edge_count,
            hidden_layers=dense_deployment.hidden_layer_count,
            flops_reduction_source="unpruned Dense GCN deployment",
        )

    model, source_result, source_history = train_source_variant(
        args,
        dataset,
        device,
        variant=args.source_variant,
    )
    if args.deployment_mode == "soft_mask_diagnostic":
        graph_mask, channel_mask = soft_deployment_masks(model, args.temp_end)
        deployment = SoftMaskedGCNDeployment(model, graph_mask, channel_mask, x.dtype).to(device)
        deployment_cost = deployment_flops_metrics(
            num_nodes=dataset.num_nodes,
            input_dim=dataset.num_features,
            original_hidden_dim=args.hidden_dim,
            deployed_hidden_dim=args.hidden_dim,
            output_dim=dataset.num_classes,
            original_edge_count=int(dataset.edge_index.size(1)),
            deployed_edge_count=int(dataset.edge_index.size(1)),
            hidden_layers=deployment.hidden_layer_count,
            flops_reduction_source="soft-mask diagnostic; no compact deployment reduction",
        )
    else:
        graph_mask, channel_mask = deployment_masks(model)
        deployment = MaterializedGCNDeployment(model, graph_mask, channel_mask, x.dtype).to(device)
        deployment_cost = deployment_flops_metrics(
            num_nodes=dataset.num_nodes,
            input_dim=dataset.num_features,
            original_hidden_dim=args.hidden_dim,
            deployed_hidden_dim=deployment.materialized_channel_count,
            output_dim=dataset.num_classes,
            original_edge_count=int(dataset.edge_index.size(1)),
            deployed_edge_count=deployment.materialized_edge_count,
            hidden_layers=deployment.hidden_layer_count,
            flops_reduction_source="hard-materialized OUGP backbone before TTA affine correction",
        )
    deployment.eval()
    for parameter in deployment.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        clean_logits = deployment(x)
    source_stats = source_energy_stats(clean_logits, train_mask, val_mask)
    clean_test_energy = energy(clean_logits)[test_mask]
    clean_test_stats = tensor_stats(clean_test_energy, "clean_test_energy")

    initial_controller = make_controller(deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    initial_state = copy.deepcopy(initial_controller.state_dict())
    source_energy_mask = train_mask | val_mask
    controller_train_mask = source_energy_mask if args.tta_adaptation_nodes == "train_val" else train_mask

    no_energy_controller = make_controller(deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    no_energy_controller.load_state_dict(initial_state, strict=True)
    no_energy_args = copy.copy(args)
    no_energy_args.tta_lambda_energy = 0.0
    no_energy_args.tta_lambda_entropy = 0.0
    # This is the ungated affine baseline.  It must not inherit the joint
    # energy-gate switch used by the energy-conditioned controller below.
    no_energy_args.tta_joint_energy_gate = False
    no_energy_before = {name: parameter.detach().clone() for name, parameter in no_energy_controller.named_parameters()}
    no_energy_training_summary = train_tta_controller(
        no_energy_args, deployment, no_energy_controller, x, y, controller_train_mask
    )

    energy_controller = make_controller(deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    energy_controller.load_state_dict(initial_state, strict=True)
    energy_before = {name: parameter.detach().clone() for name, parameter in energy_controller.named_parameters()}
    energy_args = copy.copy(args)
    energy_args.tta_lambda_entropy = 0.0
    energy_training_summary = train_tta_controller(
        energy_args,
        deployment,
        energy_controller,
        x,
        y,
        controller_train_mask,
        source_energy_stats=source_stats,
        energy_gate_fn=lambda target: energy_gate(target, source_stats, args),
        energy_gate_mask=source_energy_mask,
    )

    entropy_controller = None
    entropy_training_summary = {}
    if float(getattr(args, "tta_lambda_entropy", 0.0)) > 0.0:
        entropy_controller = make_controller(deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
        entropy_controller.load_state_dict(initial_state, strict=True)
        entropy_args = copy.copy(args)
        entropy_args.tta_lambda_energy = 0.0
        entropy_args.tta_lambda_consistency = 0.0
        entropy_args.tta_lambda_anchor = 0.0
        entropy_args.tta_lambda_teacher = 0.0
        entropy_args.tta_lambda_state = 0.0
        entropy_args.tta_joint_energy_gate = False
        entropy_training_summary = train_tta_controller(
            entropy_args, deployment, entropy_controller, x, y, controller_train_mask
        )

    shifts = [("clean", 0.0)]
    if "masking" in args.shift_types:
        shifts.extend(("masking", value) for value in args.mask_ratios)
    if "gaussian_noise" in args.shift_types:
        shifts.extend(("gaussian_noise", value) for value in args.noise_sigmas)
    if "feature_scaling" in args.shift_types:
        shifts.extend(("feature_scaling", value) for value in args.scale_factors)

    rows: list[dict[str, object]] = []
    for index, (shift_type, shift_level) in enumerate(shifts):
        shift_seed = args.seed + 10900 + index
        if shift_type == "masking" and args.masking_seed_offset is not None:
            shift_seed = args.seed + int(args.masking_seed_offset)
        x_shifted = shifted_features(
            x,
            test_mask,
            shift_type,
            shift_level,
            shift_seed,
            gaussian_noise_mode=args.gaussian_noise_mode,
        )
        x_negative = feature_masking_shift(
            x_shifted,
            args.tta_energy_negative_mask_ratio,
            generator=torch.Generator(device=device).manual_seed(shift_seed + 1000),
            node_mask=test_mask,
        )
        with torch.no_grad():
            frozen_logits = deployment(x_shifted)
            frozen_negative_logits = deployment(x_negative)
            dense_logits = dense_deployment(x_shifted) if dense_deployment is not None else None

        all_target_energy = energy(frozen_logits)
        target_energy = all_target_energy[test_mask]
        target_stats = tensor_stats(target_energy, "target_energy")
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
            gate = float(applied[selected].mean().item()) if selected.any() else 0.0
            gated_controller = clone_controller_with_gamma(
                energy_controller,
                args.tta_gamma_a,
                args.tta_gamma_b,
            )
            gated_controller.set_node_affine_strength(applied)
        else:
            gate = energy_gate(target_stats, source_stats, args)
            gated_controller = clone_controller_with_gamma(
                energy_controller,
                args.tta_gamma_a * gate,
                args.tta_gamma_b * gate,
            )

        no_energy_acc, no_energy_logits, no_energy_report = evaluate_controller(
            deployment, no_energy_controller, x_shifted, y, test_mask
        )
        _, no_energy_negative_logits, _ = evaluate_controller(
            deployment, no_energy_controller, x_negative, y, test_mask
        )
        energy_acc, energy_logits, energy_report = evaluate_controller(
            deployment, energy_controller, x_shifted, y, test_mask
        )
        _, energy_negative_logits, _ = evaluate_controller(
            deployment, energy_controller, x_negative, y, test_mask
        )
        entropy_acc = None
        if entropy_controller is not None:
            entropy_acc, _, _ = evaluate_controller(
                deployment, entropy_controller, x_shifted, y, test_mask
            )
        gated_acc, gated_logits, gated_report = evaluate_controller(
            deployment, gated_controller, x_shifted, y, test_mask
        )
        _, gated_negative_logits, _ = evaluate_controller(
            deployment, gated_controller, x_negative, y, test_mask
        )

        clean_std = max(clean_test_stats["clean_test_energy_std"], 1e-12)
        source_std = max(source_stats["source_energy_std"], 1e-12)
        row: dict[str, object] = {
            "dataset": args.dataset,
            "seed": args.seed,
            "shift_type": shift_type,
            "shift_level": float(shift_level),
            "shift_seed": shift_seed,
            "gaussian_noise_mode": args.gaussian_noise_mode,
            "dense_materialized_acc": (
                accuracy(dense_logits, y, test_mask) if dense_logits is not None else None
            ),
            "frozen_materialized_ougp_acc": accuracy(frozen_logits, y, test_mask),
            "affine_without_energy_acc": no_energy_acc,
            "affine_with_energy_loss_acc": energy_acc,
            "energy_gated_affine_acc": gated_acc,
            "entropy_only_affine_acc": entropy_acc if entropy_acc is not None else None,
            "entropy_only_gain_vs_frozen": (entropy_acc - accuracy(frozen_logits, y, test_mask)) if entropy_acc is not None else None,
            "energy_loss_gain_vs_no_energy": energy_acc - no_energy_acc,
            "energy_gate_gain_vs_ungated": gated_acc - energy_acc,
            "energy_gated_gain_vs_frozen": gated_acc - accuracy(frozen_logits, y, test_mask),
            "energy_gate": gate,
            "effective_gamma_a": args.tta_gamma_a * gate,
            "effective_gamma_b": args.tta_gamma_b * gate,
            "target_vs_clean_energy_mean_delta": target_stats["target_energy_mean"]
            - clean_test_stats["clean_test_energy_mean"],
            "target_vs_clean_energy_effect_size": (
                target_stats["target_energy_mean"] - clean_test_stats["clean_test_energy_mean"]
            )
            / clean_std,
            "target_vs_source_energy_z": (
                target_stats["target_energy_mean"] - source_stats["source_energy_mean"]
            )
            / source_std,
            "paired_energy_increase_ratio": float((target_energy > clean_test_energy).float().mean().item()),
            "frozen_energy_margin": energy_margin(frozen_logits, frozen_negative_logits, test_mask),
            "affine_without_energy_margin": energy_margin(
                no_energy_logits, no_energy_negative_logits, test_mask
            ),
            "affine_with_energy_loss_margin": energy_margin(
                energy_logits, energy_negative_logits, test_mask
            ),
            "energy_gated_affine_margin": energy_margin(
                gated_logits, gated_negative_logits, test_mask
            ),
            **{
                f"energy_gated_{key}": value
                for key, value in decision_diagnostics(frozen_logits, gated_logits, y, test_mask).items()
            },
            **source_stats,
            **clean_test_stats,
            **target_stats,
        }
        row.update({f"no_energy_{key}": value for key, value in no_energy_report.items()})
        row.update({f"energy_loss_{key}": value for key, value in energy_report.items()})
        row.update({f"energy_gated_{key}": value for key, value in gated_report.items()})
        rows.append(row)

    result = {
        "experiment": args.experiment_name,
        "dataset": args.dataset,
        "seed": args.seed,
        "source_variant": args.source_variant,
        "source_training_runs": 1,
        "deployment_materializations": 1,
        "deployment_mode": args.deployment_mode,
        "deployment": {
            "materialized_edge_count": deployment.materialized_edge_count,
            "materialized_channel_count": deployment.materialized_channel_count,
            "graph_sparsity": 1.0 - float(graph_mask.float().mean().item()),
            "channel_sparsity": 1.0 - float(channel_mask.float().mean().item()),
            **deployment_cost,
        },
        "original_num_nodes": args.original_num_nodes,
        "original_num_edges": args.original_num_edges,
        "sampled_num_nodes": int(dataset.num_nodes),
        "sampled_num_edges": int(dataset.edge_index.size(1)),
        "node_sample_size": int(args.node_sample_size),
        "node_sample_seed": int(args.node_sample_seed),
        "node_sample_mode": args.node_sample_mode,
        "gaussian_noise_mode": args.gaussian_noise_mode,
        "source_result": source_result,
        "source_history_epochs": len(source_history),
        "dense_source_training_runs": 1 if dense_source_result is not None else 0,
        "dense_source_result": dense_source_result,
        "dense_source_history_epochs": (
            len(dense_source_history) if dense_source_history is not None else 0
        ),
        "dense_deployment": (
            {
                "materialized_edge_count": dense_deployment.materialized_edge_count,
                "materialized_channel_count": dense_deployment.materialized_channel_count,
                **dense_deployment_cost,
            }
            if dense_deployment is not None and dense_deployment_cost is not None
            else None
        ),
        "source_energy_stats": source_stats,
        "clean_test_energy_stats": clean_test_stats,
        "config": {
            "tta_gate_learn_mode": args.tta_gate_learn_mode,
            "energy_gate_mode": args.energy_gate_mode,
            "energy_gate_min": args.energy_gate_min,
            "energy_gate_max": args.energy_gate_max,
            "energy_gate_center_z": args.energy_gate_center_z,
            "energy_gate_temperature": args.energy_gate_temperature,
            "energy_gate_polarity": args.energy_gate_polarity,
            "tta_epochs": args.tta_epochs,
            "tta_lr": args.tta_lr,
            "tta_weight_decay": args.tta_weight_decay,
            "tta_loss_mode": args.tta_loss_mode,
            "tta_lambda_energy": args.tta_lambda_energy,
            "tta_lambda_consistency": args.tta_lambda_consistency,
            "tta_lambda_anchor": args.tta_lambda_anchor,
            "tta_lambda_state": args.tta_lambda_state,
            "tta_lambda_teacher": args.tta_lambda_teacher,
            "tta_lambda_entropy": args.tta_lambda_entropy,
            "tta_affine_mode": args.tta_affine_mode,
            "tta_state_mode": args.tta_state_mode,
            "tta_state_sharing": args.tta_state_sharing,
            "tta_state_input_mode": args.tta_state_input_mode,
            "tta_train_mask_ratio": args.tta_train_mask_ratio,
            "tta_gamma_a": args.tta_gamma_a,
            "tta_gamma_b": args.tta_gamma_b,
            "tta_delta_normalization_floor": args.tta_delta_normalization_floor,
            "tta_joint_energy_gate": args.tta_joint_energy_gate,
            "tta_joint_energy_gate_train_min": args.tta_joint_energy_gate_train_min,
            "tta_train_shift_scope": args.tta_train_shift_scope,
            "tta_adaptation_nodes": args.tta_adaptation_nodes,
            "masking_seed_offset": args.masking_seed_offset,
        },
        "no_energy_controller_drift": module_param_delta_l2_norm(no_energy_before, no_energy_controller),
        "energy_controller_drift": module_param_delta_l2_norm(energy_before, energy_controller),
        "no_energy_training_summary": no_energy_training_summary,
        "energy_training_summary": energy_training_summary,
        "entropy_training_summary": entropy_training_summary,
        "rows": rows,
    }
    (out_dir / "energy_shift_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(out_dir / "energy_shift_probe.csv", rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
