"""Forward-only energy-shift parameter-mask adaptation (ESPM).

ESPM uses unlabeled target features only. It never creates an optimizer, loss,
controller, Q/K/V projections, or trainable target-side state. A channel-wise
energy shift changes parameter scores, and a fixed-budget Top-K mask is
materialized for the target forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParaTTAMaskResult:
    param_mask: torch.Tensor
    input_z: torch.Tensor
    hidden_z: torch.Tensor
    score_scale: float
    energy_gate_enabled: bool
    gate: float
    source_energy_mean: torch.Tensor
    source_energy_std: torch.Tensor
    target_energy_mean: torch.Tensor
    source_keep_rate: float
    target_keep_rate: float
    mask_churn: float

    def summary(self) -> dict[str, float | int | str]:
        z = self.input_z.detach().float()
        hidden_z = self.hidden_z.detach().float()
        return {
            "protocol": "para-tta",
            "target_training": False,
            "optimizer_steps": 0,
            "input_channel_count": int(z.numel()),
            "hidden_channel_count": int(hidden_z.numel()),
            "input_z_mean": float(z.mean().item()) if z.numel() else 0.0,
            "input_z_std": float(z.std(unbiased=False).item()) if z.numel() else 0.0,
            "input_z_rms": float(z.pow(2).mean().sqrt().item()) if z.numel() else 0.0,
            "input_z_min": float(z.min().item()) if z.numel() else 0.0,
            "input_z_max": float(z.max().item()) if z.numel() else 0.0,
            "hidden_z_mean": float(hidden_z.mean().item()) if hidden_z.numel() else 0.0,
            "hidden_z_std": float(hidden_z.std(unbiased=False).item()) if hidden_z.numel() else 0.0,
            "score_scale": float(self.score_scale),
            "energy_gate_enabled": bool(self.energy_gate_enabled),
            "gate": float(self.gate),
            "source_keep_rate": float(self.source_keep_rate),
            "target_keep_rate": float(self.target_keep_rate),
            "mask_churn": float(self.mask_churn),
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            **self.summary(),
            "input_z": self.input_z.detach().float().cpu().tolist(),
            "hidden_z": self.hidden_z.detach().float().cpu().tolist(),
            "source_energy_mean": self.source_energy_mean.detach().float().cpu().tolist(),
            "source_energy_std": self.source_energy_std.detach().float().cpu().tolist(),
            "target_energy_mean": self.target_energy_mean.detach().float().cpu().tolist(),
        }


def _masked_rows(values: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
    if node_mask is None:
        return values
    mask = node_mask.to(device=values.device, dtype=torch.bool)
    if mask.numel() != values.size(0):
        raise ValueError("Para-TTA node mask must match the feature row count.")
    if not bool(mask.any().item()):
        raise ValueError("Para-TTA energy statistics require at least one node.")
    return values[mask]


def _energy_statistics(
    x: torch.Tensor,
    node_mask: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    energy = x.detach().float().square()
    selected = _masked_rows(energy, node_mask)
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(float(eps))
    return mean, std


def _hidden_z_from_input_z(model, input_z: torch.Tensor, eps: float) -> torch.Tensor:
    first_weight = model.lin1.weight.detach().float()
    if first_weight.size(1) != input_z.numel():
        raise ValueError(
            "Input energy channel count does not match the first GCN layer: "
            f"{input_z.numel()} vs {first_weight.size(1)}."
        )
    routing = first_weight.abs()
    routing = routing / routing.sum(dim=1, keepdim=True).clamp_min(float(eps))
    hidden_z = routing @ input_z.to(device=routing.device, dtype=routing.dtype)
    return hidden_z


def _parameter_signal(model, input_z: torch.Tensor, hidden_z: torch.Tensor) -> torch.Tensor:
    """Map input-channel shift to every GCN weight entry in stable mask order."""

    signals: list[torch.Tensor] = []
    weights = model.parameter_weight_tensors()
    for index, weight in enumerate(weights):
        if index == 0:
            signal = input_z.to(device=weight.device, dtype=weight.dtype).unsqueeze(0).expand_as(weight)
        elif index == len(weights) - 1:
            signal = hidden_z.to(device=weight.device, dtype=weight.dtype).unsqueeze(0).expand_as(weight)
        else:
            hidden = hidden_z.to(device=weight.device, dtype=weight.dtype)
            signal = 0.5 * (hidden[:, None] + hidden[None, :]).expand_as(weight)
        signals.append(signal.reshape(-1))
    return torch.cat(signals, dim=0)


def _topk_like(mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    support = mask.detach().float().flatten() > 0.5
    keep_count = int(support.sum().item())
    if keep_count <= 0:
        keep_count = max(1, int(round(mask.numel() * 0.5)))
    keep_count = min(keep_count, scores.numel())
    selected = torch.topk(scores.detach().float().flatten(), k=keep_count, sorted=False).indices
    output = torch.zeros_like(scores.detach().float().flatten())
    output[selected] = 1.0
    return output.reshape_as(mask).to(dtype=mask.dtype)


@torch.no_grad()
def adapt_parameter_mask(
    model,
    source_param_mask: torch.Tensor,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    source_node_mask: torch.Tensor | None = None,
    target_node_mask: torch.Tensor | None = None,
    score_scale: float = 0.50,
    energy_gate: bool = False,
    gate_temperature: float = 1.0,
    eps: float = 1e-6,
) -> ParaTTAMaskResult:
    """Re-rank the fixed-budget parameter mask from input feature energy.

    The channel-wise energy shift z changes parameter scores while preserving
    the source keep budget. The scalar energy gate is optional and is enabled
    only for OOD target adaptation.

    ``target_node_mask`` is intentionally optional and should remain ``None``
    for the requested protocol: all available target features are used, with
    no target train/test labels or train-node mask.
    """

    if gate_temperature <= 0.0:
        raise ValueError("para_tta_gate_temperature must be positive.")
    if source_x.size(1) != target_x.size(1):
        raise ValueError("Source and target feature dimensions must match for Para-TTA.")
    source_mean, source_std = _energy_statistics(source_x, source_node_mask, eps)
    target_mean, _ = _energy_statistics(target_x, target_node_mask, eps)
    input_z = (target_mean - source_mean) / source_std
    input_z = torch.nan_to_num(input_z, nan=0.0, posinf=0.0, neginf=0.0)
    hidden_z = _hidden_z_from_input_z(model, input_z, eps)
    gate = 1.0
    if energy_gate:
        input_rms = input_z.float().pow(2).mean().sqrt()
        gate = float(torch.tanh(input_rms / float(gate_temperature)).item())

    base_mask = source_param_mask.detach().clone()
    if base_mask.numel() != model.param_item_count:
        raise ValueError(
            "Para-TTA currently requires a full deployment parameter mask: "
            f"{base_mask.numel()} vs {model.param_item_count}."
        )
    if model.cfg.param_pruning_granularity == "channel":
        raw_signal = hidden_z.to(device=base_mask.device, dtype=base_mask.dtype)
    else:
        raw_signal = _parameter_signal(model, input_z, hidden_z).to(
            device=base_mask.device, dtype=base_mask.dtype
        )
    signal_rms = raw_signal.float().pow(2).mean().sqrt().clamp_min(float(eps))
    normalized_signal = raw_signal / signal_rms

    if model.last_param_score is None:
        base_scores = base_mask.detach().float()
    else:
        base_scores = model.last_param_score.detach().float().clone()
    if base_scores.numel() != base_mask.numel():
        raise ValueError("Source parameter scores and mask must have the same size.")
    score_std = base_scores.std(unbiased=False).clamp_min(float(eps))
    adjusted_scores = base_scores + float(score_scale) * gate * score_std * normalized_signal.float()
    target_mask = _topk_like(base_mask, adjusted_scores).to(device=base_mask.device)
    source_support = base_mask.float() > 0.5
    target_support = target_mask.float() > 0.5
    churn = float((source_support != target_support).float().mean().item())
    return ParaTTAMaskResult(
        param_mask=target_mask,
        input_z=input_z.detach(),
        hidden_z=hidden_z.detach(),
        score_scale=float(score_scale),
        energy_gate_enabled=bool(energy_gate),
        gate=float(gate),
        source_energy_mean=source_mean.detach(),
        source_energy_std=source_std.detach(),
        target_energy_mean=target_mean.detach(),
        source_keep_rate=float(source_support.float().mean().item()),
        target_keep_rate=float(target_support.float().mean().item()),
        mask_churn=churn,
    )
