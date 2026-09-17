"""Minimal OUGP + LHCM test-time adaptation smoke run.

Default smoke:

    PYTHONPATH=src python scripts/run_tta_smoke.py --dataset cora --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ougp.data import load_graph_dataset
from ougp.deployment import MaterializedGCNDeployment, deployment_masks
from ougp.model import OUGPConfig, OUGPGCN
from ougp.tta import (
    LayerwiseHiddenAffineTTA,
    TTAConfig,
    feature_masking_shift,
    signed_z_tanh_gate_values,
)
from scripts.train_ougp import build_arg_parser as build_case_study_arg_parser
from scripts.train_ougp import run_one as run_case_study_one
from tta_training_recorder import TTATrainingRecorder, controller_delta_stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


def tensor_l2_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def tensor_mean_abs(x: torch.Tensor) -> float:
    return float(x.detach().float().abs().mean().item())


def module_param_l2_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        total += float(param.detach().float().norm().item()) ** 2
    return float(total**0.5)


def module_param_delta_l2_norm(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    total = 0.0
    for name, param in module.named_parameters():
        delta = param.detach() - before[name]
        total += float(delta.float().norm().item()) ** 2
    return float(total**0.5)


def symmetric_kl_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric KL between two feature views.  Pass mask=None to use all nodes."""

    if mask is not None:
        logits_a = logits_a[mask]
        logits_b = logits_b[mask]
    log_prob_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
    log_prob_b = torch.nn.functional.log_softmax(logits_b, dim=-1)
    prob_a = log_prob_a.detach().exp()
    prob_b = log_prob_b.detach().exp()
    return 0.5 * (
        torch.nn.functional.kl_div(log_prob_a, prob_b, reduction="batchmean")
        + torch.nn.functional.kl_div(log_prob_b, prob_a, reduction="batchmean")
    )


def summarize_reports(reports: list[dict[str, object]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not reports:
        return summary
    for key in (
        "state_norm",
        "delta_a_norm",
        "delta_b_norm",
        "delta_a_mean",
        "delta_a_std",
        "delta_a_min",
        "delta_a_max",
        "delta_a_rms",
        "delta_b_mean",
        "delta_b_std",
        "delta_b_min",
        "delta_b_max",
        "delta_b_rms",
        "delta_a_for_scale_mean",
        "delta_a_for_scale_std",
        "delta_a_for_scale_min",
        "delta_a_for_scale_max",
        "delta_a_for_scale_rms",
        "delta_b_for_bias_mean",
        "delta_b_for_bias_std",
        "delta_b_for_bias_min",
        "delta_b_for_bias_max",
        "delta_b_for_bias_rms",
        "scale_mean",
        "scale_std",
        "scale_min",
        "scale_max",
        "negative_scale_ratio",
        "scale_above_one_ratio",
        "scale_below_one_ratio",
        "bias_to_hidden_ratio",
        "hidden_delta_norm_ratio",
        "num_state_tokens",
        "hidden_token_norm",
        "state_write_norm",
        "state_delta_norm",
        "novelty",
        "token_attention_entropy",
        "token_attention_max",
        "hidden_norm_before",
        "hidden_norm_after",
    ):
        values = [float(report[key]) for report in reports if key in report]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=0))
            summary[f"{key}_max"] = float(np.max(values))
            summary[f"{key}_min"] = float(np.min(values))
    return summary


def feature_shift_stats(x: torch.Tensor, shifted: torch.Tensor) -> dict[str, float]:
    x_f = x.detach().float()
    shifted_f = shifted.detach().float()
    nonzero = x_f != 0
    masked_nonzero = nonzero & (shifted_f == 0)
    changed = shifted_f != x_f
    total = max(1, x_f.numel())
    nonzero_count = max(1, int(nonzero.sum().item()))
    return {
        "input_zero_ratio": float((x_f == 0).float().mean().item()),
        "shifted_zero_ratio": float((shifted_f == 0).float().mean().item()),
        "actual_nonzero_mask_ratio": float(masked_nonzero.sum().item() / nonzero_count),
        "overall_changed_ratio": float(changed.float().mean().item()),
        "input_l1_norm": float(x_f.abs().sum().item()),
        "shifted_l1_norm": float(shifted_f.abs().sum().item()),
        "input_l2_norm": float(x_f.norm().item()),
        "shifted_l2_norm": float(shifted_f.norm().item()),
        "masked_nonzero_count": float(masked_nonzero.sum().item()),
        "nonzero_feature_count": float(nonzero.sum().item()),
        "overall_zero_ratio_delta": float((shifted_f == 0).float().mean().item() - (x_f == 0).float().mean().item()),
        "masked_nonzero_ratio_of_all": float(masked_nonzero.sum().item() / total),
    }


@torch.no_grad()
def evaluate_masking_ratios(
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    seed: int,
    ratios: list[float],
    reset_each_ratio: bool = True,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    if not reset_each_ratio:
        controller.reset_state()
    for offset, ratio in enumerate(ratios):
        generator = torch.Generator(device=device).manual_seed(seed + 3000 + offset)
        shifted = feature_masking_shift(x, ratio, generator=generator, node_mask=test_mask)
        frozen_logits = deployment(shifted)
        if reset_each_ratio:
            controller.reset_state()
        tta_logits, _, tta_reports = controller.test_time_forward(
            deployment,
            shifted,
            update_state=True,
            return_hidden_states=True,
        )
        stats = feature_shift_stats(x, shifted)
        stats.update(
            {
                "mask_ratio": float(ratio),
                "frozen_acc": accuracy(frozen_logits, y, test_mask),
                "tta_acc": accuracy(tta_logits, y, test_mask),
                "tta_report_state_norm_mean": float(np.mean([r["state_norm"] for r in tta_reports])) if tta_reports else 0.0,
                "tta_report_delta_a_norm_mean": float(np.mean([r["delta_a_norm"] for r in tta_reports])) if tta_reports else 0.0,
                "tta_report_delta_b_norm_mean": float(np.mean([r["delta_b_norm"] for r in tta_reports])) if tta_reports else 0.0,
            }
        )
        results.append(stats)
    return results


def save_source_checkpoint(
    path: Path,
    model: OUGPGCN,
    result: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    """Atomically save a device-neutral frozen source model checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    hard_masks = model.get_training_hard_masks()
    transient_state_keys = {"last_graph_score", "last_param_score"}
    payload = {
        "cfg": asdict(model.cfg),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name not in transient_state_keys
        },
        "training_hard_masks": (
            None
            if hard_masks is None
            else tuple(mask.detach().cpu().clone() for mask in hard_masks)
        ),
        "result": result,
        "history": history,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_source_checkpoint(path: Path, dataset, device: torch.device):
    """Reconstruct a frozen OUGP model from a canonical source checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = OUGPGCN(
        OUGPConfig(**payload["cfg"]),
        edge_index=dataset.edge_index.to(device),
        x=dataset.x.to(device),
    ).to(device)
    state_dict = dict(payload["state_dict"])
    state_dict.pop("last_graph_score", None)
    state_dict.pop("last_param_score", None)
    model.load_state_dict(state_dict, strict=True)
    hard_masks = payload.get("training_hard_masks")
    if hard_masks is not None:
        model.set_training_hard_masks(
            hard_masks[0].to(device),
            hard_masks[1].to(device),
        )
    model.eval()
    result = dict(payload["result"])
    result["source_checkpoint_reused"] = True
    result["source_checkpoint_path"] = str(path)
    return model, result, list(payload.get("history", []))


def train_source_variant(args: argparse.Namespace, dataset, device: torch.device, variant: str):
    source_out_dir = Path(args.out_dir) / f"source_case_study_{variant}"
    source_out_dir.mkdir(parents=True, exist_ok=True)
    policy = str(getattr(args, "source_checkpoint_policy", "off"))
    checkpoint_root = str(getattr(args, "source_checkpoint_root", ""))
    checkpoint_key = str(getattr(args, "source_checkpoint_key", "") or args.dataset)
    checkpoint_path = (
        Path(checkpoint_root) / checkpoint_key / f"seed{args.seed}" / f"{variant}.pt"
        if checkpoint_root
        else None
    )
    if policy == "reuse":
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"source checkpoint not found: {checkpoint_path}")
        return load_source_checkpoint(checkpoint_path, dataset, device)
    if policy == "read_write" and checkpoint_path is not None and checkpoint_path.exists():
        return load_source_checkpoint(checkpoint_path, dataset, device)
    result, model, history = run_case_study_one(
        args,
        dataset,
        variant=variant,
        seed=args.seed,
        out_dir=source_out_dir,
        return_artifacts=True,
    )
    model.to(device)
    model.eval()
    if policy in {"refresh", "read_write"}:
        if checkpoint_path is None:
            raise ValueError("source_checkpoint_root is required when checkpoint caching is enabled")
        result = dict(result)
        result["source_checkpoint_reused"] = False
        result["source_checkpoint_path"] = str(checkpoint_path)
        save_source_checkpoint(checkpoint_path, model, result, history)
    return model, result, history


def train_source_ougp(args: argparse.Namespace, dataset, device: torch.device):
    return train_source_variant(args, dataset, device, variant="ougp")


def train_tta_controller(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    source_energy_stats: dict[str, float] | None = None,
    energy_gate_fn=None,
    energy_gate_mask: torch.Tensor | None = None,
    validation_mask: torch.Tensor | None = None,
    teacher_logits_override: torch.Tensor | None = None,
    history_name: str = "tta_training",
) -> dict[str, object]:
    loss_mode = str(getattr(args, "tta_loss_mode", "supervised"))
    if loss_mode == "unsupervised":
        return _train_tta_unsupervised(
            args, deployment, controller, x, y, train_mask,
            source_energy_stats=source_energy_stats,
            energy_gate_fn=energy_gate_fn,
            energy_gate_mask=energy_gate_mask,
            validation_mask=validation_mask,
            teacher_logits_override=teacher_logits_override,
            history_name=history_name,
        )
    return _train_tta_supervised(
        args, deployment, controller, x, y, train_mask,
        source_energy_stats=source_energy_stats,
        energy_gate_fn=energy_gate_fn,
        energy_gate_mask=energy_gate_mask,
        validation_mask=validation_mask,
        teacher_logits_override=teacher_logits_override,
        history_name=history_name,
    )


def _source_shift_node_mask(args: argparse.Namespace, train_mask: torch.Tensor) -> torch.Tensor | None:
    """Choose the source-side node scope used to simulate test-time shifts."""

    scope = str(getattr(args, "tta_train_shift_scope", "all"))
    if scope == "all":
        return None
    if scope == "train_mask":
        return train_mask
    raise ValueError("tta_train_shift_scope must be one of: all, train_mask.")


def _can_reuse_clean_stateless_tta_view(args: argparse.Namespace) -> bool:
    """Whether clean TTA views are identical and safe to reuse."""

    consistency_ratio = float(
        getattr(args, "tta_consistency_mask_ratio", getattr(args, "mask_ratio", 0.0))
    )
    return (
        float(getattr(args, "mask_ratio", 0.0)) == 0.0
        and consistency_ratio == 0.0
        and str(getattr(args, "tta_state_mode", "stateless")) == "stateless"
        and not bool(getattr(args, "tta_carry_state", False))
    )


def frozen_teacher_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Distill frozen clean predictions into a controller-facing shifted view."""

    if mask is not None:
        selected = mask.to(device=student_logits.device, dtype=torch.bool)
        student_logits = student_logits[selected]
        teacher_logits = teacher_logits[selected]
    if student_logits.numel() == 0:
        return student_logits.new_tensor(0.0)
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits, dim=-1),
        torch.nn.functional.softmax(teacher_logits.detach(), dim=-1),
        reduction="batchmean",
    )


def clean_source_centered_energy_loss(
    args: argparse.Namespace,
    adapted_logits: torch.Tensor,
    frozen_logits: torch.Tensor,
    source_energy_stats: dict[str, float] | None,
    nodes: torch.Tensor | slice,
) -> torch.Tensor:
    """Calibrate clean adapted energy toward a bounded source-centered target."""

    if source_energy_stats is None:
        raise ValueError("clean energy calibration requires source energy statistics.")
    margin = float(getattr(args, "tta_energy_clean_margin", 0.10))
    if margin <= 0.0:
        raise ValueError("tta_energy_clean_margin must be positive.")

    frozen_energy = -torch.logsumexp(frozen_logits.float(), dim=-1)
    adapted_energy = -torch.logsumexp(adapted_logits.float(), dim=-1)
    clean_gate = signed_z_tanh_gate_values(
        frozen_energy,
        source_energy_stats["source_energy_mean"],
        source_energy_stats["source_energy_std"],
        float(getattr(args, "energy_gate_center_z", 0.0)),
        float(getattr(args, "energy_gate_temperature", 1.0)),
        float(getattr(args, "energy_gate_max", 1.0)),
        float(getattr(args, "energy_gate_polarity", 1.0)),
    )
    target_energy = (frozen_energy - margin * clean_gate).detach()
    selected_adapted = adapted_energy[nodes]
    selected_target = target_energy[nodes]
    if selected_adapted.numel() == 0:
        return adapted_energy.new_tensor(0.0)
    return (selected_adapted - selected_target).square().mean()


@torch.no_grad()
def _apply_joint_energy_gate(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x_shift: torch.Tensor,
    source_energy_stats: dict[str, float] | None,
    energy_gate_fn,
    energy_gate_mask: torch.Tensor | None,
    frozen_logits_override: torch.Tensor | None = None,
) -> float:
    """Match source controller training to the test-time energy-gated affine path."""

    if not bool(getattr(args, "tta_joint_energy_gate", False)):
        controller.reset_affine_strength()
        return 1.0
    if source_energy_stats is None:
        raise ValueError("joint energy-gate training requires source energy statistics.")
    frozen_logits = (
        deployment(x_shift)
        if frozen_logits_override is None
        else frozen_logits_override
    )
    values = -torch.logsumexp(frozen_logits.float(), dim=-1)
    gate_mode = str(getattr(args, "tta_gate_learn_mode", "mean"))
    if gate_mode == "node":
        if str(getattr(args, "energy_gate_mode", "signed_z_tanh")) != "signed_z_tanh":
            raise ValueError("tta_gate_learn_mode=node requires energy_gate_mode=signed_z_tanh.")
        node_gates = signed_z_tanh_gate_values(
            values,
            source_energy_stats["source_energy_mean"],
            source_energy_stats["source_energy_std"],
            float(args.energy_gate_center_z),
            float(args.energy_gate_temperature),
            float(args.energy_gate_max),
            float(getattr(args, "energy_gate_polarity", 1.0)),
        )
        selected = (
            torch.ones_like(node_gates, dtype=torch.bool)
            if energy_gate_mask is None
            else energy_gate_mask.to(device=node_gates.device, dtype=torch.bool)
        )
        applied = torch.zeros_like(node_gates)
        applied[selected] = node_gates[selected]
        train_min = float(getattr(args, "tta_joint_energy_gate_train_min", 0.0))
        if train_min > 0.0:
            nonzero = applied != 0
            applied[nonzero] = torch.sign(applied[nonzero]) * torch.maximum(
                applied[nonzero].abs(),
                applied.new_tensor(train_min),
            )
        applied.clamp_(-1.0, 1.0)
        controller.set_node_affine_strength(applied)
        selected_values = applied[selected]
        return float(selected_values.mean().item()) if selected_values.numel() else 0.0
    if gate_mode != "mean":
        raise ValueError("tta_gate_learn_mode must be one of: mean, node.")
    if energy_gate_fn is None:
        raise ValueError("mean energy-gate training requires a gate function.")
    if energy_gate_mask is not None:
        values = values[energy_gate_mask.to(device=values.device, dtype=torch.bool)]
    target_stats = {"target_energy_mean": float(values.mean().item())}
    raw_gate = float(energy_gate_fn(target_stats))
    train_min = float(getattr(args, "tta_joint_energy_gate_train_min", 0.0))
    if raw_gate < 0.0:
        gate = -min(1.0, max(train_min, abs(raw_gate)))
    else:
        gate = min(1.0, max(train_min, raw_gate))
    controller.set_affine_strength(gate)
    return gate


def _gate_training_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "joint_train_gate_count": 0.0,
            "joint_train_gate_mean": 0.0,
            "joint_train_gate_min": 0.0,
            "joint_train_gate_max": 0.0,
        }
    return {
        "joint_train_gate_count": float(len(values)),
        "joint_train_gate_mean": float(sum(values) / len(values)),
        "joint_train_gate_min": float(min(values)),
        "joint_train_gate_max": float(max(values)),
    }


def _tta_recorder(args: argparse.Namespace, history_name: str) -> TTATrainingRecorder:
    return TTATrainingRecorder(
        getattr(args, "out_dir", "experiments/manual_tta"),
        name=history_name,
        total_epochs=int(args.tta_epochs),
        log_every=int(getattr(args, "tta_log_every", 1)),
    )


def _record_tta_epoch(
    recorder: TTATrainingRecorder,
    controller: LayerwiseHiddenAffineTTA,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_loss: torch.Tensor,
    gate: float,
    supervised_loss: torch.Tensor,
    energy_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    anchor_loss: torch.Tensor,
    teacher_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    state_loss: torch.Tensor,
    validation_accuracy: float,
) -> None:
    recorder.record(
        {
            "epoch": float(epoch),
            "total_loss": float(total_loss.detach().item()),
            "supervised_loss": float(supervised_loss.detach().item()),
            "energy_loss": float(energy_loss.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "anchor_loss": float(anchor_loss.detach().item()),
            "teacher_loss": float(teacher_loss.detach().item()),
            "entropy_loss": float(entropy_loss.detach().item()),
            "state_loss": float(state_loss.detach().item()),
            "validation_accuracy": float(validation_accuracy),
            "gate": float(gate),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **controller.affine_strength_stats(),
            **controller_delta_stats(controller),
        }
    )


@torch.no_grad()
def _evaluate_tta_validation(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    validation_mask: torch.Tensor | None,
    gate_mask: torch.Tensor | None,
    source_energy_stats: dict[str, float] | None,
    energy_gate_fn,
) -> float:
    if validation_mask is None or not bool(validation_mask.any()):
        return -1.0
    was_training = controller.training
    controller.eval()
    controller.reset_state()
    _apply_joint_energy_gate(
        args,
        deployment,
        controller,
        x,
        source_energy_stats,
        energy_gate_fn,
        gate_mask,
    )
    logits = deployment.forward_with_tta(x, controller, update_state=False)
    value = accuracy(logits, y, validation_mask)
    controller.train(was_training)
    return value


def _maybe_select_tta_checkpoint(
    args: argparse.Namespace,
    controller: LayerwiseHiddenAffineTTA,
    validation_accuracy: float,
    epoch: int,
    best: tuple[float, int, dict[str, torch.Tensor] | None],
) -> tuple[float, int, dict[str, torch.Tensor] | None]:
    best_accuracy, best_epoch, best_state = best
    if str(getattr(args, "tta_checkpoint_selection", "last")) != "best_val":
        return best
    if validation_accuracy > best_accuracy:
        return (
            validation_accuracy,
            epoch,
            {
                name: value.detach().cpu().clone()
                for name, value in controller.state_dict().items()
            },
        )
    return best


def _restore_tta_checkpoint(
    args: argparse.Namespace,
    controller: LayerwiseHiddenAffineTTA,
    best: tuple[float, int, dict[str, torch.Tensor] | None],
) -> dict[str, object]:
    best_accuracy, best_epoch, best_state = best
    if str(getattr(args, "tta_checkpoint_selection", "last")) == "best_val":
        if best_state is None:
            raise RuntimeError("TTA best_val requested but no validation checkpoint was captured.")
        controller.load_state_dict(best_state, strict=True)
    return {
        "checkpoint_selection": str(getattr(args, "tta_checkpoint_selection", "last")),
        "best_validation_accuracy": best_accuracy if best_state is not None else None,
        "best_validation_epoch": best_epoch if best_state is not None else None,
    }


@torch.no_grad()
def _select_tta_strength(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    validation_mask: torch.Tensor | None,
    gate_mask: torch.Tensor | None,
    source_energy_stats: dict[str, float] | None,
    energy_gate_fn,
) -> dict[str, object]:
    mode = str(getattr(args, "tta_strength_selection", "fixed"))
    if mode == "fixed":
        controller.set_selected_gate_scale(1.0)
        return {"strength_selection": mode, "selected_gate_scale": 1.0, "strength_validation": []}
    if validation_mask is None or not bool(validation_mask.any()):
        raise ValueError("tta_strength_selection=best_val requires a validation mask.")

    candidates = sorted({float(value) for value in args.tta_strength_candidates})
    if not candidates or candidates[0] < 0.0:
        raise ValueError("tta_strength_candidates must contain non-negative values.")
    results: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_scale = 0.0
    was_training = controller.training
    controller.eval()
    for scale in candidates:
        controller.reset_state()
        _apply_joint_energy_gate(
            args,
            deployment,
            controller,
            x,
            source_energy_stats,
            energy_gate_fn,
            gate_mask,
        )
        controller.scale_affine_strength(scale)
        logits = deployment.forward_with_tta(x, controller, update_state=False)
        value = accuracy(logits, y, validation_mask)
        results.append({"scale": scale, "validation_accuracy": value})
        if value > best_accuracy:
            best_accuracy = value
            best_scale = scale
    controller.reset_state()
    controller.reset_affine_strength()
    controller.set_selected_gate_scale(best_scale)
    controller.train(was_training)
    return {
        "strength_selection": mode,
        "selected_gate_scale": best_scale,
        "strength_validation": results,
    }


def _train_tta_supervised(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    source_energy_stats: dict[str, float] | None = None,
    energy_gate_fn=None,
    energy_gate_mask: torch.Tensor | None = None,
    validation_mask: torch.Tensor | None = None,
    teacher_logits_override: torch.Tensor | None = None,
    history_name: str = "tta_training",
) -> dict[str, object]:
    for param in deployment.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.tta_lr, weight_decay=args.tta_weight_decay)
    generator = torch.Generator(device=x.device).manual_seed(args.seed + 1000)
    negative_generator = torch.Generator(device=x.device).manual_seed(args.seed + 2000)
    train_gates: list[float] = []
    shift_node_mask = _source_shift_node_mask(args, train_mask)
    recorder = _tta_recorder(args, history_name)
    best_checkpoint: tuple[float, int, dict[str, torch.Tensor] | None] = (-1.0, -1, None)
    with torch.no_grad():
        teacher_logits = (
            deployment(x).detach()
            if teacher_logits_override is None
            else teacher_logits_override.detach().to(x.device)
        )
    clean_reuse = _can_reuse_clean_stateless_tta_view(args)
    cached_frozen_logits = None
    cached_gate = None
    if clean_reuse:
        cached_frozen_logits = (
            teacher_logits
            if teacher_logits_override is None
            else deployment(x).detach()
        )
        cached_gate = _apply_joint_energy_gate(
            args,
            deployment,
            controller,
            x,
            source_energy_stats,
            energy_gate_fn,
            energy_gate_mask,
            frozen_logits_override=cached_frozen_logits,
        )

    for epoch in range(1, args.tta_epochs + 1):
        controller.train()
        controller.reset_state()
        if clean_reuse:
            x_shift = x
            gate = float(cached_gate)
        else:
            x_shift = feature_masking_shift(
                x, args.mask_ratio, generator=generator, node_mask=shift_node_mask
            )
            gate = _apply_joint_energy_gate(
                args, deployment, controller, x_shift, source_energy_stats, energy_gate_fn, energy_gate_mask
            )
        train_gates.append(gate)
        optimizer.zero_grad(set_to_none=True)
        logits = deployment.forward_with_tta(x_shift, controller, update_state=True)
        supervised_loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])
        lambda_logits = float(getattr(args, "tta_lambda_logits", 1.0))
        loss = lambda_logits * supervised_loss
        anchor_loss = controller.anchor_loss()
        zero = loss.new_tensor(0.0)
        energy_loss = zero
        consistency_loss = zero
        teacher_loss = zero
        entropy_loss = zero
        state_loss = zero
        lambda_energy = float(getattr(args, "tta_lambda_energy", 0.0))
        if lambda_energy > 0.0:
            negative_ratio = float(getattr(args, "tta_energy_negative_mask_ratio", 0.50))
            if negative_ratio > 0.0:
                controller.reset_state()
                x_negative = feature_masking_shift(
                    x_shift, negative_ratio, generator=negative_generator, node_mask=shift_node_mask
                )
                negative_logits = deployment.forward_with_tta(x_negative, controller, update_state=True)
                target_energy = -torch.logsumexp(logits, dim=-1)
                negative_energy = -torch.logsumexp(negative_logits, dim=-1)
                energy_loss = target_energy[train_mask].mean() - negative_energy[train_mask].mean()
            else:
                if clean_reuse:
                    frozen_shift_logits = cached_frozen_logits
                else:
                    with torch.no_grad():
                        frozen_shift_logits = deployment(x_shift)
                energy_loss = clean_source_centered_energy_loss(
                    args,
                    logits,
                    frozen_shift_logits,
                    source_energy_stats,
                    train_mask,
                )
            loss = loss + lambda_energy * energy_loss
        lambda_consistency = float(getattr(args, "tta_lambda_consistency", 0.0))
        if lambda_consistency > 0.0:
            if clean_reuse:
                consistency_loss = logits.sum() * 0.0
            else:
                controller.reset_state()
                consistency_ratio = float(getattr(args, "tta_consistency_mask_ratio", args.mask_ratio))
                x_consistency = feature_masking_shift(
                    x_shift, consistency_ratio, generator=negative_generator, node_mask=shift_node_mask
                )
                consistency_logits = deployment.forward_with_tta(x_consistency, controller, update_state=True)
                consistency_loss = symmetric_kl_consistency_loss(logits, consistency_logits, train_mask)
            loss = loss + lambda_consistency * consistency_loss
        lambda_anchor = float(getattr(args, "tta_lambda_anchor", 0.0))
        if lambda_anchor > 0.0:
            loss = loss + lambda_anchor * anchor_loss
        lambda_teacher = float(getattr(args, "tta_lambda_teacher", 0.0))
        if lambda_teacher > 0.0:
            teacher_loss = frozen_teacher_consistency_loss(logits, teacher_logits, train_mask)
            loss = loss + lambda_teacher * teacher_loss
        loss.backward()
        optimizer.step()
        needs_validation = (
            str(getattr(args, "tta_checkpoint_selection", "last")) == "best_val"
            or str(getattr(args, "tta_strength_selection", "fixed")) == "best_val"
        )
        validation_accuracy = (
            _evaluate_tta_validation(
                args,
                deployment,
                controller,
                x,
                y,
                validation_mask,
                energy_gate_mask,
                source_energy_stats,
                energy_gate_fn,
            )
            if needs_validation
            else -1.0
        )
        best_checkpoint = _maybe_select_tta_checkpoint(
            args, controller, validation_accuracy, epoch, best_checkpoint
        )
        _record_tta_epoch(
            recorder,
            controller,
            optimizer,
            epoch=epoch,
            total_loss=loss,
            gate=gate,
            supervised_loss=supervised_loss,
            energy_loss=energy_loss,
            consistency_loss=consistency_loss,
            anchor_loss=anchor_loss,
            teacher_loss=teacher_loss,
            entropy_loss=entropy_loss,
            state_loss=state_loss,
            validation_accuracy=validation_accuracy,
        )
        optimizer.zero_grad(set_to_none=True)
        if not getattr(args, "tta_carry_state", False):
            controller.reset_state()
    controller.reset_state()
    controller.reset_affine_strength()
    checkpoint_summary = _restore_tta_checkpoint(args, controller, best_checkpoint)
    strength_summary = _select_tta_strength(
        args,
        deployment,
        controller,
        x,
        y,
        validation_mask,
        energy_gate_mask,
        source_energy_stats,
        energy_gate_fn,
    )
    return {
        **_gate_training_summary(train_gates),
        **checkpoint_summary,
        **strength_summary,
        **recorder.summary(),
    }


def _train_tta_unsupervised(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller: LayerwiseHiddenAffineTTA,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    source_energy_stats: dict[str, float] | None = None,
    energy_gate_fn=None,
    energy_gate_mask: torch.Tensor | None = None,
    validation_mask: torch.Tensor | None = None,
    teacher_logits_override: torch.Tensor | None = None,
    history_name: str = "tta_training",
) -> dict[str, object]:
    """Pure unsupervised TTA: L = λ_e·L_energy + λ_c·L_consistency + λ_a·L_anchor + λ_s·L_state."""
    for param in deployment.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.tta_lr, weight_decay=args.tta_weight_decay)

    gen = torch.Generator(device=x.device).manual_seed(args.seed + 1000)
    gen_neg = torch.Generator(device=x.device).manual_seed(args.seed + 2000)
    gen_cons = torch.Generator(device=x.device).manual_seed(args.seed + 3000)

    λ_e = float(getattr(args, "tta_lambda_energy", 0.0))
    λ_c = float(getattr(args, "tta_lambda_consistency", 0.0))
    λ_a = float(getattr(args, "tta_lambda_anchor", 0.0))
    λ_s = float(getattr(args, "tta_lambda_state", 0.0))
    λ_t = float(getattr(args, "tta_lambda_teacher", 0.0))
    λ_h = float(getattr(args, "tta_lambda_entropy", 0.0))
    recorder = _tta_recorder(args, history_name)
    best_checkpoint: tuple[float, int, dict[str, torch.Tensor] | None] = (-1.0, -1, None)

    # Some comparison runners intentionally invoke this helper with every
    # unsupervised loss disabled. There is no differentiable objective in
    # that case, so skip the optimizer loop instead of calling backward() on
    # a constant zero tensor.
    if max(λ_e, λ_c, λ_a, λ_s, λ_t, λ_h) <= 0.0:
        controller.reset_state()
        controller.reset_affine_strength()
        return {**_gate_training_summary([]), **recorder.summary()}

    energy_reduction = str(getattr(args, "tta_energy_reduction", "all"))
    cons_reduction = str(getattr(args, "tta_consistency_reduction", "all"))
    energy_nodes = train_mask if energy_reduction == "train_mask" else slice(None)
    cons_nodes = train_mask if cons_reduction == "train_mask" else None  # None → all nodes in KL loss
    train_gates: list[float] = []
    shift_node_mask = _source_shift_node_mask(args, train_mask)
    with torch.no_grad():
        teacher_logits = (
            deployment(x).detach()
            if teacher_logits_override is None
            else teacher_logits_override.detach().to(x.device)
        )
    clean_reuse = _can_reuse_clean_stateless_tta_view(args)
    cached_frozen_logits = None
    cached_gate = None
    if clean_reuse:
        cached_frozen_logits = (
            teacher_logits
            if teacher_logits_override is None
            else deployment(x).detach()
        )
        cached_gate = _apply_joint_energy_gate(
            args,
            deployment,
            controller,
            x,
            source_energy_stats,
            energy_gate_fn,
            energy_gate_mask,
            frozen_logits_override=cached_frozen_logits,
        )

    for epoch in range(1, args.tta_epochs + 1):
        controller.train()
        optimizer.zero_grad(set_to_none=True)

        # === Forward 1: main target view ===
        controller.reset_state()
        if clean_reuse:
            x_shift = x
            gate = float(cached_gate)
        else:
            x_shift = feature_masking_shift(
                x, args.mask_ratio, generator=gen, node_mask=shift_node_mask
            )
            gate = _apply_joint_energy_gate(
                args, deployment, controller, x_shift, source_energy_stats, energy_gate_fn, energy_gate_mask
            )
        train_gates.append(gate)
        logits = deployment.forward_with_tta(x_shift, controller, update_state=True)
        anchor_loss = controller.anchor_loss()
        state_loss = controller.state_regularization() if λ_s > 0.0 else torch.tensor(0.0, device=x.device)

        loss = torch.tensor(0.0, device=x.device)
        zero = loss.new_tensor(0.0)
        supervised_loss = zero
        energy_loss = zero
        consistency_loss = zero
        teacher_loss = zero
        entropy_loss = zero

        # === Entropy minimization baseline (optional) ===
        if λ_h > 0.0:
            probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            entropy_values = -(probabilities * probabilities.log()).sum(dim=-1)
            entropy_loss = entropy_values[energy_nodes].mean()
            loss = loss + λ_h * entropy_loss

        # === Energy loss: contrastive with negatives, source-centered when clean ===
        if λ_e > 0.0:
            neg_ratio = float(getattr(args, "tta_energy_negative_mask_ratio", 0.50))
            if neg_ratio > 0.0:
                controller.reset_state()
                x_neg = feature_masking_shift(
                    x_shift, neg_ratio, generator=gen_neg, node_mask=shift_node_mask
                )
                neg_logits = deployment.forward_with_tta(x_neg, controller, update_state=True)
                target_e = -torch.logsumexp(logits, dim=-1)
                neg_e = -torch.logsumexp(neg_logits, dim=-1)
                energy_loss = target_e[energy_nodes].mean() - neg_e[energy_nodes].mean()
            else:
                if clean_reuse:
                    frozen_shift_logits = cached_frozen_logits
                else:
                    with torch.no_grad():
                        frozen_shift_logits = deployment(x_shift)
                energy_loss = clean_source_centered_energy_loss(
                    args,
                    logits,
                    frozen_shift_logits,
                    source_energy_stats,
                    energy_nodes,
                )
            loss = loss + λ_e * energy_loss

        # === Consistency loss (symmetric KL) ===
        if λ_c > 0.0:
            if clean_reuse:
                consistency_loss = logits.sum() * 0.0
            else:
                controller.reset_state()
                cons_ratio = float(getattr(args, "tta_consistency_mask_ratio", args.mask_ratio))
                x_cons = feature_masking_shift(
                    x_shift, cons_ratio, generator=gen_cons, node_mask=shift_node_mask
                )
                cons_logits = deployment.forward_with_tta(x_cons, controller, update_state=True)
                consistency_loss = symmetric_kl_consistency_loss(logits, cons_logits, mask=cons_nodes)
            loss = loss + λ_c * consistency_loss

        # === Anchor loss ===
        if λ_a > 0.0:
            loss = loss + λ_a * anchor_loss

        # This is a source-training objective only. At deployment the compact
        # OUGP and controller remain frozen and TTA is a single forward pass.
        if λ_t > 0.0:
            teacher_loss = frozen_teacher_consistency_loss(logits, teacher_logits, cons_nodes)
            loss = loss + λ_t * teacher_loss

        # === State regularization ===
        if λ_s > 0.0:
            loss = loss + λ_s * state_loss

        loss.backward()
        optimizer.step()
        needs_validation = (
            str(getattr(args, "tta_checkpoint_selection", "last")) == "best_val"
            or str(getattr(args, "tta_strength_selection", "fixed")) == "best_val"
        )
        validation_accuracy = (
            _evaluate_tta_validation(
                args,
                deployment,
                controller,
                x,
                y,
                validation_mask,
                energy_gate_mask,
                source_energy_stats,
                energy_gate_fn,
            )
            if needs_validation
            else -1.0
        )
        best_checkpoint = _maybe_select_tta_checkpoint(
            args, controller, validation_accuracy, epoch, best_checkpoint
        )
        _record_tta_epoch(
            recorder,
            controller,
            optimizer,
            epoch=epoch,
            total_loss=loss,
            gate=gate,
            supervised_loss=supervised_loss,
            energy_loss=energy_loss,
            consistency_loss=consistency_loss,
            anchor_loss=anchor_loss,
            teacher_loss=teacher_loss,
            entropy_loss=entropy_loss,
            state_loss=state_loss,
            validation_accuracy=validation_accuracy,
        )
        optimizer.zero_grad(set_to_none=True)

        if not getattr(args, "tta_carry_state", False):
            controller.reset_state()

    controller.reset_state()
    controller.reset_affine_strength()
    checkpoint_summary = _restore_tta_checkpoint(args, controller, best_checkpoint)
    strength_summary = _select_tta_strength(
        args,
        deployment,
        controller,
        x,
        y,
        validation_mask,
        energy_gate_mask,
        source_energy_stats,
        energy_gate_fn,
    )
    return {
        **_gate_training_summary(train_gates),
        **checkpoint_summary,
        **strength_summary,
        **recorder.summary(),
    }


@torch.no_grad()
def latency_ms(fn, device: torch.device, repeats: int, warmup: int) -> float:
    if repeats <= 0:
        return 0.0
    for _ in range(max(0, warmup)):
        fn()
    sync_if_cuda(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync_if_cuda(device)
    return 1000.0 * (time.perf_counter() - start) / float(repeats)


def main() -> None:
    parser = build_case_study_arg_parser()
    parser.set_defaults(
        out_dir="experiments/exp076_tta_minimal_cora_gcn4",
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=200,
        hidden_dim=32,
        backbone="gcn",
        num_gnn_layers=4,
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
    parser.add_argument("--tta-epochs", type=int, default=60)
    parser.add_argument("--tta-rank", type=int, default=8)
    parser.add_argument("--tta-state-mode", choices=["full", "stateless"], default="full")
    parser.add_argument("--tta-state-sharing", choices=["per_layer", "shared"], default="per_layer")
    parser.add_argument("--tta-carry-state", action="store_true")
    parser.add_argument("--tta-eval-carry-state", action="store_true")
    parser.add_argument("--mask-ratio", type=float, default=0.20)
    parser.add_argument("--mask-sweep-ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--tta-lr", type=float, default=1e-3)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-lambda-consistency", type=float, default=0.0)
    parser.add_argument("--tta-consistency-mask-ratio", type=float, default=0.20)
    parser.add_argument("--tta-lambda-energy", type=float, default=0.0)
    parser.add_argument("--tta-energy-negative-mask-ratio", type=float, default=0.50)
    parser.add_argument("--tta-energy-clean-margin", type=float, default=0.10)
    parser.add_argument("--tta-lambda-anchor", type=float, default=0.0)
    parser.add_argument("--tta-lambda-state", type=float, default=0.0)
    parser.add_argument("--tta-loss-mode", choices=["supervised", "unsupervised"], default="supervised")
    parser.add_argument("--tta-energy-reduction", choices=["train_mask", "all"], default="all")
    parser.add_argument("--tta-consistency-reduction", choices=["train_mask", "all"], default="all")
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--latency-warmup", type=int, default=5)
    args = parser.parse_args()
    args.seeds = [args.seed]
    if args.backbone != "gcn":
        raise ValueError("TTA materialization currently uses MaterializedGCNDeployment; pass --backbone gcn.")

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_graph_dataset(args.data_root, args.dataset)
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    model, source_result, source_history = train_source_ougp(args, dataset, device)
    with torch.no_grad():
        source_soft_logits, source_soft_stats = model(x, temperature=args.temp_end)
    graph_mask, channel_mask = deployment_masks(model)
    deployment = MaterializedGCNDeployment(model, graph_mask, channel_mask, x.dtype).to(device)
    deployment.eval()
    for param in deployment.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        source_fixed_logits, source_fixed_stats = model(
            x,
            temperature=args.temp_end,
            fixed_masks=(graph_mask, channel_mask),
        )
        materialized_logits = deployment(x)
    materialized_vs_fixed_logits_max_abs_diff = float((materialized_logits - source_fixed_logits).detach().abs().max().item())
    source_soft_vs_fixed_logits_max_abs_diff = float((source_soft_logits - source_fixed_logits).detach().abs().max().item())

    zero_controller = LayerwiseHiddenAffineTTA(
        deployment.hidden_layer_count,
        deployment.materialized_channel_count,
        TTAConfig(rank=args.tta_rank, state_mode=args.tta_state_mode, state_sharing=args.tta_state_sharing),
    ).to(device)
    with torch.no_grad():
        frozen_logits = deployment(x)
        zero_controller.reset_state()
        zero_logits, _, zero_reports = zero_controller.test_time_forward(
            deployment,
            x,
            update_state=True,
            return_hidden_states=True,
        )
    zero_state_max_abs_diff = float((zero_logits - frozen_logits).detach().abs().max().item())

    controller = LayerwiseHiddenAffineTTA(
        deployment.hidden_layer_count,
        deployment.materialized_channel_count,
        TTAConfig(rank=args.tta_rank, state_mode=args.tta_state_mode, state_sharing=args.tta_state_sharing),
    ).to(device)
    controller_param_norm_before = module_param_l2_norm(controller)
    controller_params_before = {name: param.detach().clone() for name, param in controller.named_parameters()}
    train_tta_controller(args, deployment, controller, x, y, train_mask)
    controller_param_norm_after = module_param_l2_norm(controller)
    controller_param_delta_norm = module_param_delta_l2_norm(controller_params_before, controller)

    eval_generator = torch.Generator(device=device).manual_seed(args.seed + 2000)
    x_masked = feature_masking_shift(x, args.mask_ratio, generator=eval_generator, node_mask=test_mask)
    controller.eval()
    for param in controller.parameters():
        param.grad = None
    state_before = controller.states.detach().clone()
    controller_params_before = {name: param.detach().clone() for name, param in controller.named_parameters()}
    deployment_params_before = {name: param.detach().clone() for name, param in deployment.named_parameters()}
    adj_indices_before = deployment.adj.indices().clone()
    adj_values_before = deployment.adj.values().clone()
    keep_channels_before = deployment.keep_channels.clone()

    with torch.no_grad():
        frozen_clean_logits = deployment(x)
        frozen_masked_logits = deployment(x_masked)
        controller.reset_state()
        tta_clean_logits = controller.test_time_forward(deployment, x, update_state=True)
        if not args.tta_eval_carry_state:
            controller.reset_state()
        tta_masked_logits, hidden_states, reports = controller.test_time_forward(
            deployment,
            x_masked,
            update_state=True,
            return_hidden_states=True,
        )
    mask_sweep = evaluate_masking_ratios(
        deployment,
        controller,
        x,
        y,
        test_mask,
        device,
        args.seed,
        args.mask_sweep_ratios,
        reset_each_ratio=not args.tta_eval_carry_state,
    )

    no_test_grads = all(param.grad is None for param in deployment.parameters()) and all(
        param.grad is None for param in controller.parameters()
    )
    masks_fixed = bool(
        torch.equal(deployment.adj.indices(), adj_indices_before)
        and torch.allclose(deployment.adj.values(), adj_values_before)
        and torch.equal(deployment.keep_channels, keep_channels_before)
    )
    params_fixed = all(torch.allclose(param, deployment_params_before[name]) for name, param in deployment.named_parameters()) and all(
        torch.allclose(param, controller_params_before[name]) for name, param in controller.named_parameters()
    )
    only_state_changed = params_fixed and masks_fixed and not torch.allclose(controller.states, state_before)

    frozen_latency = latency_ms(lambda: deployment(x_masked), device, args.latency_repeats, args.latency_warmup)
    tta_latency = latency_ms(
        (
            lambda: controller.test_time_forward(deployment, x_masked, update_state=True)
            if args.tta_eval_carry_state
            else (controller.reset_state(), controller.test_time_forward(deployment, x_masked, update_state=True))[1]
        ),
        device,
        args.latency_repeats,
        args.latency_warmup,
    )

    result = {
        "dataset": args.dataset,
        "backbone": "gcn",
        "num_gnn_layers": args.num_gnn_layers,
        "seed": args.seed,
        "shift_none": True,
        "shift_feature_masking_ratio": args.mask_ratio,
        "shift_node_scope": "test_mask",
        "mask_sweep_ratios": args.mask_sweep_ratios,
        "source_epochs": args.epochs,
        "tta_epochs": args.tta_epochs,
        "tta_energy_clean_margin": args.tta_energy_clean_margin,
        "tta_state_mode": args.tta_state_mode,
        "tta_state_sharing": args.tta_state_sharing,
        "tta_carry_state": bool(args.tta_carry_state),
        "tta_eval_carry_state": bool(args.tta_eval_carry_state),
        "source_case_study_result": source_result,
        "source_case_study_final_epoch": source_history[-1] if source_history else {},
        "source_cfg": asdict(model.cfg),
        "lhcm_enabled": bool(args.use_hidden_coupling),
        "graph_sparsity_target": args.graph_sparsity,
        "channel_sparsity_target": args.param_sparsity,
        "materialized_kept_edges": deployment.materialized_edge_count,
        "materialized_kept_edges_with_self_loops": deployment.materialized_edge_count_with_self_loops,
        "materialized_kept_channels": deployment.materialized_channel_count,
        "source_soft_clean_acc": accuracy(source_soft_logits, y, test_mask),
        "source_soft_clean_val_acc": accuracy(source_soft_logits, y, dataset.val_mask.to(device)),
        "source_fixed_clean_acc": accuracy(source_fixed_logits, y, test_mask),
        "materialized_clean_acc": accuracy(materialized_logits, y, test_mask),
        "materialized_vs_fixed_logits_max_abs_diff": materialized_vs_fixed_logits_max_abs_diff,
        "source_soft_vs_fixed_logits_max_abs_diff": source_soft_vs_fixed_logits_max_abs_diff,
        "zero_state_max_abs_diff": zero_state_max_abs_diff,
        "frozen_clean_acc": accuracy(frozen_clean_logits, y, test_mask),
        "frozen_masked_acc": accuracy(frozen_masked_logits, y, test_mask),
        "tta_clean_acc": accuracy(tta_clean_logits, y, test_mask),
        "tta_masked_acc": accuracy(tta_masked_logits, y, test_mask),
        "frozen_masked_latency_ms": frozen_latency,
        "tta_masked_latency_ms": tta_latency,
        "masks_fixed": masks_fixed,
        "params_fixed_during_test": params_fixed,
        "no_test_grads": no_test_grads,
        "only_state_changed_during_test": only_state_changed,
        "state_norm_after_masked_test": float(controller.states.detach().float().norm().item()),
        "controller_param_norm_before_training": controller_param_norm_before,
        "controller_param_norm_after_training": controller_param_norm_after,
        "controller_param_delta_norm_after_training": controller_param_delta_norm,
        "zero_init_shape_report": zero_reports,
        "masked_tta_shape_report": reports,
        "masked_tta_report_summary": summarize_reports(reports),
        "hidden_state_shapes": [
            {"layer": item["layer"], "kind": item["kind"], "shape": tuple(item["tensor"].shape)}
            for item in hidden_states
        ],
        "mask_sweep": mask_sweep,
    }
    result_path = out_dir / "tta_smoke_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
