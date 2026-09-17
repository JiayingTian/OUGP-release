"""Incremental recorder for source-trained TTA controller optimization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from ougp.tta import LayerwiseHiddenAffineTTA


def controller_delta_stats(controller: LayerwiseHiddenAffineTTA) -> dict[str, float]:
    """Summarize learned direct affine parameters across all hidden layers."""

    stats: dict[str, float] = {}
    for name, parameters in (
        ("delta_a", controller.direct_delta_a),
        ("delta_b", controller.direct_delta_b),
    ):
        flat = torch.cat([parameter.detach().float().flatten() for parameter in parameters])
        stats.update(
            {
                f"{name}_mean": float(flat.mean().item()),
                f"{name}_std": float(flat.std(unbiased=False).item()),
                f"{name}_min": float(flat.min().item()),
                f"{name}_max": float(flat.max().item()),
                f"{name}_rms": float(flat.square().mean().sqrt().item()),
                f"{name}_norm": float(flat.norm().item()),
            }
        )
    if controller.direct_logit_scale is not None and controller.direct_logit_bias is not None:
        for name, parameter in (
            ("logit_scale", controller.direct_logit_scale),
            ("logit_bias", controller.direct_logit_bias),
        ):
            flat = parameter.detach().float().flatten()
            stats.update(
                {
                    f"{name}_mean": float(flat.mean().item()),
                    f"{name}_std": float(flat.std(unbiased=False).item()),
                    f"{name}_rms": float(flat.square().mean().sqrt().item()),
                    f"{name}_norm": float(flat.norm().item()),
                }
            )
    if controller.direct_logit_residual_weight is not None:
        weight = controller.direct_logit_residual_weight.detach().float()
        stats.update(
            {
                "logit_residual_weight_rms": float(weight.square().mean().sqrt().item()),
                "logit_residual_weight_norm": float(weight.norm().item()),
            }
        )
    if controller.gate_calibration_slope is not None:
        stats.update(
            {
                "gate_calibration_slope": float(
                    controller.gate_calibration_slope.detach().item()
                ),
                "gate_calibration_bias": float(
                    controller.gate_calibration_bias.detach().item()
                ),
                "gate_calibration_amplitude": float(
                    torch.sigmoid(
                        controller.gate_calibration_amplitude_logit.detach()
                    ).item()
                ),
            }
        )
    if controller.output_mlp_up is not None:
        weight = controller.output_mlp_up.weight.detach().float()
        stats.update(
            {
                "output_mlp_up_rms": float(weight.square().mean().sqrt().item()),
                "output_mlp_up_norm": float(weight.norm().item()),
            }
        )
    return stats


class TTATrainingRecorder:
    """Write one durable TTA training record per epoch and mirror it to stdout."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        name: str,
        total_epochs: int,
        log_every: int,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_epochs = int(total_epochs)
        self.log_every = max(1, int(log_every))
        self.jsonl_path = self.out_dir / f"{name}_history.jsonl"
        self.csv_path = self.out_dir / f"{name}_history.csv"
        self.jsonl_path.write_text("", encoding="utf-8")
        self.csv_path.write_text("", encoding="utf-8")
        self.history: list[dict[str, float]] = []

    def record(self, row: dict[str, float]) -> None:
        normalized = {key: float(value) for key, value in row.items()}
        self.history.append(normalized)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, sort_keys=True) + "\n")
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(normalized))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(normalized)

        epoch = int(normalized["epoch"])
        if epoch == 1 or epoch == self.total_epochs or epoch % self.log_every == 0:
            print(
                "TTA "
                f"epoch={epoch:03d}/{self.total_epochs:03d} "
                f"loss={normalized['total_loss']:.6f} "
                f"energy={normalized['energy_loss']:.6f} "
                f"consistency={normalized['consistency_loss']:.6f} "
                f"anchor={normalized['anchor_loss']:.6f} "
                f"teacher={normalized['teacher_loss']:.6f} "
                f"entropy={normalized['entropy_loss']:.6f} "
                f"val={normalized['validation_accuracy']:.6f} "
                f"gate_mean={normalized['gate_mean']:.6f} "
                f"gate_min={normalized['gate_min']:.6f} "
                f"gate_max={normalized['gate_max']:.6f} "
                f"delta_a_rms={normalized['delta_a_rms']:.6f} "
                f"delta_b_rms={normalized['delta_b_rms']:.6f}",
                flush=True,
            )

    def summary(self) -> dict[str, object]:
        return {
            "history_count": len(self.history),
            "history_jsonl": str(self.jsonl_path),
            "history_csv": str(self.csv_path),
            "history": self.history,
        }
