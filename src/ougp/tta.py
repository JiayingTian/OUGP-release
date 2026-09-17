"""Forward-only test-time adaptation for materialized OUGP GCNs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TTAConfig:
    rank: int = 8
    gamma_a: float = 0.10
    gamma_b: float = 0.10
    readout_hidden_dim: int | None = None
    state_mode: str = "full"
    state_sharing: str = "per_layer"
    center_delta_a: bool = False
    normalize_delta_a: bool = False
    center_delta_b: bool = False
    normalize_delta_b: bool = False
    delta_normalization_floor: float = 0.05
    scale_strategy: str = "linear"
    alpha_a: float | None = None
    state_input_mode: str = "stats"
    token_attention_temperature: float = 1.0
    affine_mode: str = "readout"
    use_logit_affine: bool = False
    gamma_logit_scale: float = 0.10
    gamma_logit_bias: float = 0.10
    use_logit_residual: bool = False
    gamma_logit_residual: float = 1.0
    output_gate_mode: str = "signed"
    learn_gate_calibration: bool = False
    gate_calibration_slope_init: float = 1.0
    gate_calibration_bias_init: float = 0.0
    gate_calibration_amplitude_init: float = 0.99
    use_output_mlp: bool = False
    output_mlp_rank: int = 16
    gamma_output_mlp: float = 1.0


def feature_masking_shift(
    x: torch.Tensor,
    mask_ratio: float = 0.20,
    generator: torch.Generator | None = None,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply element-wise feature masking, optionally only on selected nodes."""

    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1].")
    keep_prob = 1.0 - float(mask_ratio)
    keep = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) < keep_prob
    if node_mask is None:
        return x * keep.to(dtype=x.dtype)
    if node_mask.ndim != 1 or node_mask.numel() != x.size(0):
        raise ValueError("node_mask must have shape [num_nodes].")
    shifted = x.clone()
    selected = node_mask.to(device=x.device, dtype=torch.bool)
    shifted[selected] = shifted[selected] * keep[selected].to(dtype=x.dtype)
    return shifted


def signed_z_tanh_gate_values(
    target_energy: torch.Tensor,
    source_mean: float,
    source_std: float,
    center_z: float,
    temperature: float,
    gate_max: float,
    polarity: float,
) -> torch.Tensor:
    """Return one signed energy gate per target item."""

    z = (target_energy.float() - float(source_mean)) / max(float(source_std), 1e-12)
    residual = torch.sign(z) * (z.abs() - float(center_z)).clamp_min(0.0)
    return abs(float(gate_max)) * float(polarity) * torch.tanh(
        residual / max(float(temperature), 1e-12)
    )


@torch.no_grad()
def propagate_predictions(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    steps: int,
    mode: str = "convex",
    labels: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diffuse class probabilities over the graph without updating the GNN."""

    if mode not in {"convex", "logspace_signed"}:
        raise ValueError("propagation mode must be one of: convex, logspace_signed.")
    if mode == "convex" and not 0.0 <= alpha <= 1.0:
        raise ValueError("convex propagation alpha must be in [0, 1].")
    if mode == "logspace_signed" and not -1.0 <= alpha <= 1.0:
        raise ValueError("logspace-signed propagation alpha must be in [-1, 1].")
    if steps <= 0:
        raise ValueError("propagation steps must be positive.")
    if (labels is None) != (anchor_mask is None):
        raise ValueError("labels and anchor_mask must be provided together.")
    probabilities = torch.softmax(logits, dim=-1)
    anchor_values = None
    anchors = None
    if labels is not None and anchor_mask is not None:
        if labels.ndim != 1 or labels.numel() != logits.size(0):
            raise ValueError("labels must have shape [num_nodes].")
        if anchor_mask.ndim != 1 or anchor_mask.numel() != logits.size(0):
            raise ValueError("anchor_mask must have shape [num_nodes].")
        anchors = anchor_mask.to(device=logits.device, dtype=torch.bool)
        anchor_values = F.one_hot(
            labels.to(device=logits.device, dtype=torch.long),
            num_classes=logits.size(1),
        ).to(dtype=logits.dtype)
        probabilities = probabilities.clone()
        probabilities[anchors] = anchor_values[anchors]
    source, target = edge_index.to(logits.device)
    degree = torch.bincount(target, minlength=logits.size(0)).to(logits.dtype).clamp_min(1.0)
    adjacency = torch.sparse_coo_tensor(
        torch.stack([target, source]),
        torch.ones(source.numel(), device=logits.device, dtype=logits.dtype),
        (logits.size(0), logits.size(0)),
        device=logits.device,
    ).coalesce()
    initial = probabilities
    current = probabilities
    for _ in range(steps):
        neighbor_mean = torch.sparse.mm(adjacency, current) / degree.unsqueeze(1)
        if mode == "convex":
            current = (1.0 - alpha) * initial + alpha * neighbor_mean
        else:
            initial_log = initial.clamp_min(1e-8).log()
            neighbor_log = neighbor_mean.clamp_min(1e-8).log()
            current = torch.softmax(
                initial_log + float(alpha) * (neighbor_log - initial_log),
                dim=-1,
            )
        if anchors is not None and anchor_values is not None:
            current = current.clone()
            current[anchors] = anchor_values[anchors]
    return current.clamp_min(1e-8).log()


def ensemble_log_probabilities(log_probabilities: list[torch.Tensor]) -> torch.Tensor:
    """Average predictive probabilities and return normalized log-probabilities."""

    if not log_probabilities:
        raise ValueError("log_probabilities must not be empty.")
    reference_shape = log_probabilities[0].shape
    if any(values.shape != reference_shape for values in log_probabilities):
        raise ValueError("all log-probability tensors must have the same shape.")
    probabilities = torch.stack([values.exp() for values in log_probabilities], dim=0)
    return probabilities.mean(dim=0).clamp_min(1e-8).log()


class LayerwiseHiddenAffineTTA(nn.Module):
    """Layer-wise online state that produces channel affine corrections.

    The test-time protocol is forward-only:
    H_l -> channel statistics -> Q/K/V/gate -> update S_l -> read S_l @ q_l
    -> Readout -> channel-wise affine correction.
    """

    def __init__(
        self,
        num_layers: int,
        channel_dim: int,
        cfg: TTAConfig | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if channel_dim <= 0:
            raise ValueError("channel_dim must be positive.")
        self.num_layers = int(num_layers)
        self.channel_dim = int(channel_dim)
        self.output_dim = int(output_dim) if output_dim is not None else None
        self.cfg = cfg or TTAConfig()
        if self.cfg.rank <= 0:
            raise ValueError("rank must be positive.")
        if self.cfg.state_mode not in {"full", "stateless"}:
            raise ValueError("state_mode must be one of: full, stateless.")
        if self.cfg.state_sharing not in {"per_layer", "shared"}:
            raise ValueError("state_sharing must be one of: per_layer, shared.")
        if self.cfg.scale_strategy not in {"linear", "exp_tanh"}:
            raise ValueError("scale_strategy must be one of: linear, exp_tanh.")
        if self.cfg.state_input_mode not in {"stats", "hidden", "hidden_attention", "channel_attention"}:
            raise ValueError("state_input_mode must be one of: stats, hidden, hidden_attention, channel_attention.")
        if self.cfg.token_attention_temperature <= 0:
            raise ValueError("token_attention_temperature must be positive.")
        if self.cfg.delta_normalization_floor <= 0:
            raise ValueError("delta_normalization_floor must be positive.")
        if self.cfg.affine_mode not in {"readout", "direct"}:
            raise ValueError("affine_mode must be one of: readout, direct.")
        if self.cfg.output_gate_mode not in {"signed", "absolute", "none"}:
            raise ValueError("output_gate_mode must be one of: signed, absolute, none.")
        if not 0.0 < self.cfg.gate_calibration_amplitude_init < 1.0:
            raise ValueError("gate_calibration_amplitude_init must be in (0, 1).")
        if self.cfg.output_mlp_rank <= 0:
            raise ValueError("output_mlp_rank must be positive.")

        rank = int(self.cfg.rank)
        stats_dim = 3 * self.channel_dim
        readout_hidden_dim = int(self.cfg.readout_hidden_dim or max(self.channel_dim, 2 * rank))

        self.q_proj = nn.ModuleList([nn.Linear(stats_dim, rank) for _ in range(self.num_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(stats_dim, rank) for _ in range(self.num_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(stats_dim, rank) for _ in range(self.num_layers)])
        self.gate_proj = nn.ModuleList([nn.Linear(stats_dim, 2 * rank) for _ in range(self.num_layers)])
        self.hidden_q_proj = nn.ModuleList([nn.Linear(self.channel_dim, rank) for _ in range(self.num_layers)])
        self.hidden_k_proj = nn.ModuleList([nn.Linear(self.channel_dim, rank) for _ in range(self.num_layers)])
        self.hidden_v_proj = nn.ModuleList([nn.Linear(self.channel_dim, rank) for _ in range(self.num_layers)])
        self.hidden_gate_proj = nn.ModuleList([nn.Linear(self.channel_dim, 2 * rank) for _ in range(self.num_layers)])
        self.readout = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(rank, readout_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(readout_hidden_dim, 2 * self.channel_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        for head in self.readout:
            final = head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("readout final layer must be nn.Linear.")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.direct_delta_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.channel_dim)) for _ in range(self.num_layers)]
        )
        self.direct_delta_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.channel_dim)) for _ in range(self.num_layers)]
        )
        if self.cfg.use_logit_affine:
            if self.output_dim is None or self.output_dim <= 0:
                raise ValueError("use_logit_affine requires a positive output_dim.")
            self.direct_logit_scale = nn.Parameter(torch.zeros(self.output_dim))
            self.direct_logit_bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("direct_logit_scale", None)
            self.register_parameter("direct_logit_bias", None)
        if self.cfg.use_logit_residual:
            if self.output_dim is None or self.output_dim <= 0:
                raise ValueError("use_logit_residual requires a positive output_dim.")
            self.direct_logit_residual_weight = nn.Parameter(
                torch.zeros(self.output_dim, self.channel_dim)
            )
            self.direct_logit_residual_bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("direct_logit_residual_weight", None)
            self.register_parameter("direct_logit_residual_bias", None)
        if self.cfg.learn_gate_calibration:
            self.gate_calibration_slope = nn.Parameter(
                torch.tensor(float(self.cfg.gate_calibration_slope_init))
            )
            self.gate_calibration_bias = nn.Parameter(
                torch.tensor(float(self.cfg.gate_calibration_bias_init))
            )
            amplitude = torch.tensor(float(self.cfg.gate_calibration_amplitude_init))
            self.gate_calibration_amplitude_logit = nn.Parameter(torch.logit(amplitude))
        else:
            self.register_parameter("gate_calibration_slope", None)
            self.register_parameter("gate_calibration_bias", None)
            self.register_parameter("gate_calibration_amplitude_logit", None)
        if self.cfg.use_output_mlp:
            self.output_mlp_down = nn.Linear(self.channel_dim, self.cfg.output_mlp_rank)
            self.output_mlp_up = nn.Linear(self.cfg.output_mlp_rank, self.channel_dim)
            nn.init.zeros_(self.output_mlp_up.weight)
            nn.init.zeros_(self.output_mlp_up.bias)
        else:
            self.output_mlp_down = None
            self.output_mlp_up = None

        state_count = 1 if self.cfg.state_sharing == "shared" else self.num_layers
        self.register_buffer("states", torch.zeros(state_count, rank, rank))
        # Runtime-only strength is set by an energy gate.  It is deliberately
        # not checkpointed: source-pretrained weights should be reusable with
        # the gate computed from each target episode.
        self.register_buffer("_affine_strength", torch.ones(()), persistent=False)
        self._node_affine_strength: torch.Tensor | None = None
        self.selected_gate_scale = 1.0
        self.last_layer_reports: list[dict[str, object]] = []
        self._anchor_losses: list[torch.Tensor] = []
        self._state_losses: list[torch.Tensor] = []

    def reset_state(self) -> None:
        self.states.zero_()
        self.last_layer_reports = []
        self._anchor_losses = []
        self._state_losses = []

    def reset_forward_losses(self) -> None:
        self._anchor_losses = []
        self._state_losses = []

    def anchor_loss(self) -> torch.Tensor:
        if self._anchor_losses:
            return torch.stack(self._anchor_losses).mean()
        return self.states.new_tensor(0.0)

    def state_regularization(self) -> torch.Tensor:
        """Return the differentiable per-forward state-change penalty."""

        if self._state_losses:
            return torch.stack(self._state_losses).mean()
        return self.states.new_tensor(0.0)

    @torch.no_grad()
    def set_affine_strength(self, value: float) -> None:
        """Set the scalar multiplier supplied by an energy gate for one episode."""

        if not -1.0 <= float(value) <= 1.0:
            raise ValueError("affine strength must be in [-1, 1].")
        self._node_affine_strength = None
        self._affine_strength.fill_(float(value))

    @torch.no_grad()
    def set_node_affine_strength(self, values: torch.Tensor) -> None:
        """Set one runtime energy-gate strength per graph node."""

        if values.ndim != 1:
            raise ValueError("node affine strength must have shape [num_nodes].")
        detached = values.detach().float()
        if detached.numel() > 0 and (detached.min() < -1.0 or detached.max() > 1.0):
            raise ValueError("node affine strength must be in [-1, 1].")
        self._node_affine_strength = detached.clone()

    def affine_strength_stats(self) -> dict[str, float]:
        values = (
            self._affine_strength.detach().float().reshape(1)
            if self._node_affine_strength is None
            else self._node_affine_strength.detach().float()
        )
        values = self._calibrate_strength(values)
        return {
            "gate_mean": float(values.mean().item()),
            "gate_abs_mean": float(values.abs().mean().item()),
            "gate_min": float(values.min().item()),
            "gate_max": float(values.max().item()),
            "gate_positive_ratio": float((values > 0).float().mean().item()),
            "gate_negative_ratio": float((values < 0).float().mean().item()),
            "gate_zero_ratio": float((values == 0).float().mean().item()),
        }

    def _calibrate_strength(self, values: torch.Tensor) -> torch.Tensor:
        if not self.cfg.learn_gate_calibration:
            return values
        if (
            self.gate_calibration_slope is None
            or self.gate_calibration_bias is None
            or self.gate_calibration_amplitude_logit is None
        ):
            raise RuntimeError("gate calibration parameters are unavailable.")
        bounded = values.clamp(-0.999999, 0.999999)
        latent = torch.atanh(bounded)
        amplitude = torch.sigmoid(self.gate_calibration_amplitude_logit)
        return amplitude * torch.tanh(
            self.gate_calibration_slope * latent + self.gate_calibration_bias
        )

    @torch.no_grad()
    def scale_affine_strength(self, factor: float) -> None:
        if factor < 0.0:
            raise ValueError("affine strength scale must be non-negative.")
        if self._node_affine_strength is None:
            self._affine_strength.mul_(float(factor))
        else:
            self._node_affine_strength.mul_(float(factor))

    def set_selected_gate_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("selected gate scale must be non-negative.")
        self.selected_gate_scale = float(value)

    def _strength_for_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._node_affine_strength is None:
            raw = self._affine_strength.to(device=hidden.device, dtype=hidden.dtype)
            return self._calibrate_strength(raw)
        if self._node_affine_strength.numel() != hidden.size(0):
            raise ValueError(
                "node affine strength must match the hidden-state node dimension; "
                f"got {self._node_affine_strength.numel()} vs {hidden.size(0)}."
            )
        raw = self._node_affine_strength.to(
            device=hidden.device, dtype=hidden.dtype
        ).unsqueeze(1)
        return self._calibrate_strength(raw)

    @torch.no_grad()
    def reset_affine_strength(self) -> None:
        self._node_affine_strength = None
        self._affine_strength.fill_(1.0)

    @torch.no_grad()
    def apply_stateless_direct_affine(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the trained direct affine correction without state-side work.

        EXP121/EXP130 use stateless direct affine at deployment. The learned
        direct deltas are already the controller output, so test inference does
        not need hidden statistics, Q/K/V projections, readout, or reports.
        """

        if self.cfg.state_mode != "stateless" or self.cfg.affine_mode != "direct":
            raise ValueError("fast direct affine requires stateless + direct mode")
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer is out of range for this controller.")

        delta_a = self.direct_delta_a[layer].to(device=hidden.device, dtype=hidden.dtype)
        delta_b = self.direct_delta_b[layer].to(device=hidden.device, dtype=hidden.dtype)
        delta_a_for_scale = delta_a
        if self.cfg.center_delta_a:
            delta_a_for_scale = delta_a_for_scale - delta_a_for_scale.mean()
        if self.cfg.normalize_delta_a:
            floor = float(self.cfg.delta_normalization_floor)
            delta_a_for_scale = delta_a_for_scale / (
                delta_a_for_scale.pow(2).mean() + floor * floor
            ).sqrt()

        delta_b_for_bias = delta_b
        if self.cfg.center_delta_b:
            delta_b_for_bias = delta_b_for_bias - delta_b_for_bias.mean()
        if self.cfg.normalize_delta_b:
            floor = float(self.cfg.delta_normalization_floor)
            delta_b_for_bias = delta_b_for_bias / (
                delta_b_for_bias.pow(2).mean() + floor * floor
            ).sqrt()

        strength = self._strength_for_hidden(hidden)
        if self.cfg.scale_strategy == "linear":
            scale = 1.0 + float(self.cfg.gamma_a) * strength * delta_a_for_scale
        else:
            alpha_a = float(self.cfg.alpha_a if self.cfg.alpha_a is not None else self.cfg.gamma_a)
            scale = torch.exp(alpha_a * strength * torch.tanh(delta_a_for_scale))
        bias = float(self.cfg.gamma_b) * strength * delta_b_for_bias
        return scale * hidden + bias

    def apply_logit_affine(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply an optional class-wise residual affine calibration."""

        if not self.cfg.use_logit_affine:
            return logits
        if self.direct_logit_scale is None or self.direct_logit_bias is None:
            raise RuntimeError("logit affine parameters are unavailable.")
        if logits.size(-1) != self.direct_logit_scale.numel():
            raise ValueError(
                f"logit dimension mismatch: {logits.size(-1)} vs {self.direct_logit_scale.numel()}."
            )
        strength = self._strength_for_hidden(logits)
        scale = 1.0 + float(self.cfg.gamma_logit_scale) * strength * self.direct_logit_scale
        bias = float(self.cfg.gamma_logit_bias) * strength * self.direct_logit_bias
        self._anchor_losses.append((scale - 1.0).square().mean() + bias.square().mean())
        return scale * logits + bias

    def apply_logit_residual(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Apply an optional residual classifier while keeping backbone weights frozen."""

        adapted = logits
        if self.cfg.use_logit_residual:
            if (
                self.direct_logit_residual_weight is None
                or self.direct_logit_residual_bias is None
            ):
                raise RuntimeError("logit residual parameters are unavailable.")
            strength = self._strength_for_hidden(hidden)
            if self.cfg.output_gate_mode == "absolute":
                strength = strength.abs()
            elif self.cfg.output_gate_mode == "none":
                strength = torch.ones_like(strength)
            residual = F.linear(
                hidden,
                self.direct_logit_residual_weight,
                self.direct_logit_residual_bias,
            )
            correction = float(self.cfg.gamma_logit_residual) * strength * residual
            self._anchor_losses.append(correction.square().mean())
            adapted = adapted + correction
        return self.apply_logit_affine(adapted)

    def apply_output_hidden_adapter(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply an optional nonlinear residual adapter before the frozen classifier."""

        if not self.cfg.use_output_mlp:
            return hidden
        if self.output_mlp_down is None or self.output_mlp_up is None:
            raise RuntimeError("output MLP adapter modules are unavailable.")
        strength = self._strength_for_hidden(hidden)
        if self.cfg.output_gate_mode == "absolute":
            strength = strength.abs()
        elif self.cfg.output_gate_mode == "none":
            strength = torch.ones_like(strength)
        residual = self.output_mlp_up(F.relu(self.output_mlp_down(hidden)))
        correction = float(self.cfg.gamma_output_mlp) * strength * residual
        self._anchor_losses.append(correction.square().mean())
        return hidden + correction

    def state_index(self, layer: int) -> int:
        if self.cfg.state_sharing == "shared":
            return 0
        return layer

    def hidden_statistics(
        self,
        hidden: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [num_nodes, channel_dim].")
        if hidden.size(1) != self.channel_dim:
            raise ValueError("hidden channel dimension does not match controller channel_dim.")
        selected = hidden
        if node_mask is not None:
            if node_mask.ndim != 1 or node_mask.numel() != hidden.size(0):
                raise ValueError("node_mask must have shape [num_nodes].")
            selected = hidden[node_mask.to(device=hidden.device, dtype=torch.bool)]
            if selected.numel() == 0:
                selected = hidden
        mean = selected.mean(dim=0)
        std = selected.std(dim=0, unbiased=False)
        abs_mean = selected.abs().mean(dim=0)
        return torch.cat([mean, std, abs_mean], dim=0)

    def select_hidden_tokens(
        self,
        hidden: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [num_nodes, channel_dim].")
        if hidden.size(1) != self.channel_dim:
            raise ValueError("hidden channel dimension does not match controller channel_dim.")
        if node_mask is None:
            return hidden
        if node_mask.ndim != 1 or node_mask.numel() != hidden.size(0):
            raise ValueError("node_mask must have shape [num_nodes].")
        selected = hidden[node_mask.to(device=hidden.device, dtype=torch.bool)]
        return hidden if selected.numel() == 0 else selected

    def channel_hidden_tokens(
        self,
        hidden: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Represent each surviving channel by its co-activation profile."""

        selected = self.select_hidden_tokens(hidden, node_mask=node_mask)
        selected = selected.float()
        selected = selected - selected.mean(dim=0, keepdim=True)
        selected = F.normalize(selected, dim=0, eps=1e-12)
        tokens = selected.t() @ selected
        return tokens.to(device=hidden.device, dtype=hidden.dtype)

    def adapt_layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer is out of range for this controller.")

        hidden_norm_before = float(hidden.detach().float().norm().item())
        stats = self.hidden_statistics(hidden, node_mask=node_mask)
        state_input_mode = self.cfg.state_input_mode
        if state_input_mode in {"hidden", "hidden_attention", "channel_attention"}:
            if state_input_mode == "channel_attention":
                tokens = self.channel_hidden_tokens(hidden, node_mask=node_mask)
            else:
                tokens = self.select_hidden_tokens(hidden, node_mask=node_mask)
            q_tokens = F.normalize(torch.tanh(self.hidden_q_proj[layer](tokens)), dim=-1, eps=1e-12)
            k_tokens = F.normalize(torch.tanh(self.hidden_k_proj[layer](tokens)), dim=-1, eps=1e-12)
            v_tokens = self.hidden_v_proj[layer](tokens)
            gate_tokens = torch.sigmoid(self.hidden_gate_proj[layer](tokens))
            lambda_tokens, beta_tokens = gate_tokens.chunk(2, dim=-1)
            token_weight = torch.full(
                (tokens.size(0),),
                1.0 / max(1, tokens.size(0)),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            q = F.normalize(torch.sum(q_tokens * token_weight.unsqueeze(-1), dim=0), dim=0, eps=1e-12)
            k = F.normalize(torch.sum(k_tokens * token_weight.unsqueeze(-1), dim=0), dim=0, eps=1e-12)
            v = torch.sum(v_tokens * token_weight.unsqueeze(-1), dim=0)
            lambda_vec = torch.sum(lambda_tokens * token_weight.unsqueeze(-1), dim=0)
            beta_vec = torch.sum(beta_tokens * token_weight.unsqueeze(-1), dim=0)
            num_state_tokens = int(tokens.size(0))
            hidden_token_norm = float(tokens.detach().float().norm(dim=1).mean().item())
        else:
            q = F.normalize(torch.tanh(self.q_proj[layer](stats)), dim=0, eps=1e-12)
            k = F.normalize(torch.tanh(self.k_proj[layer](stats)), dim=0, eps=1e-12)
            v = self.v_proj[layer](stats)
            gate = torch.sigmoid(self.gate_proj[layer](stats))
            lambda_vec, beta_vec = gate.chunk(2, dim=0)
            num_state_tokens = 1
            hidden_token_norm = 0.0

        state_idx = self.state_index(layer)
        prev_state = self.states[state_idx].detach().to(device=hidden.device, dtype=hidden.dtype).clone()
        token_attention_entropy = 0.0
        token_attention_max = 1.0
        if self.cfg.state_mode == "stateless":
            read_vec = q
            new_state = prev_state
            if self.cfg.affine_mode == "direct":
                delta_ab = torch.cat([self.direct_delta_a[layer], self.direct_delta_b[layer]], dim=0)
            else:
                delta_ab = self.readout[layer](read_vec)
        else:
            current = prev_state @ k
            delta = v - current
            if state_input_mode in {"hidden", "hidden_attention", "channel_attention"}:
                current_tokens = k_tokens @ prev_state.t()
                delta_tokens = v_tokens - current_tokens
                if state_input_mode in {"hidden_attention", "channel_attention"}:
                    novelty_scores = delta_tokens.detach().float().norm(dim=-1)
                    temperature = float(self.cfg.token_attention_temperature)
                    token_weight = torch.softmax(
                        novelty_scores / max(temperature, 1e-12),
                        dim=0,
                    ).to(device=tokens.device, dtype=tokens.dtype)
                    q = F.normalize(torch.sum(q_tokens * token_weight.unsqueeze(-1), dim=0), dim=0, eps=1e-12)
                    k = F.normalize(torch.sum(k_tokens * token_weight.unsqueeze(-1), dim=0), dim=0, eps=1e-12)
                    v = torch.sum(v_tokens * token_weight.unsqueeze(-1), dim=0)
                    lambda_vec = torch.sum(lambda_tokens * token_weight.unsqueeze(-1), dim=0)
                    beta_vec = torch.sum(beta_tokens * token_weight.unsqueeze(-1), dim=0)
                    current = prev_state @ k
                    delta = v - current
                    token_weight_f = token_weight.detach().float()
                    token_attention_entropy = float(
                        (-(token_weight_f * token_weight_f.clamp_min(1e-12).log()).sum()).item()
                    )
                    token_attention_max = float(token_weight_f.max().item())
                write_tokens = delta_tokens.unsqueeze(2) * k_tokens.unsqueeze(1)
                write = torch.sum(write_tokens * token_weight.view(-1, 1, 1), dim=0)
                delta_for_report = delta_tokens
            else:
                write = delta.unsqueeze(1) * k.unsqueeze(0)
                delta_for_report = delta
            new_state = lambda_vec.unsqueeze(1) * prev_state + beta_vec.unsqueeze(1) * write
            self._state_losses.append((new_state - prev_state).pow(2).mean())
            if state_input_mode in {"hidden_attention", "channel_attention"}:
                read_tokens = q_tokens @ new_state.t()
                read_vec = torch.sum(read_tokens * token_weight.unsqueeze(-1), dim=0)
            else:
                read_vec = new_state @ q
            delta_ab = self.readout[layer](read_vec)
        delta_a, delta_b = delta_ab.chunk(2, dim=0)
        delta_a_for_scale = delta_a
        if self.cfg.center_delta_a:
            delta_a_for_scale = delta_a_for_scale - delta_a_for_scale.mean()
        if self.cfg.normalize_delta_a:
            # Direct affine parameters start at zero. A detached 1e-12 RMS
            # denominator makes their first gradient explode, while a global
            # (uncentred) scale leaves GCN argmax predictions unchanged.
            # This smooth normalizer is finite at identity and approaches RMS
            # normalization only after a channel-wise direction is learned.
            floor = float(self.cfg.delta_normalization_floor)
            normalizer = (delta_a_for_scale.pow(2).mean() + floor * floor).sqrt()
            delta_a_for_scale = delta_a_for_scale / normalizer
        delta_b_for_bias = delta_b
        if self.cfg.center_delta_b:
            delta_b_for_bias = delta_b_for_bias - delta_b_for_bias.mean()
        if self.cfg.normalize_delta_b:
            floor = float(self.cfg.delta_normalization_floor)
            normalizer = (delta_b_for_bias.pow(2).mean() + floor * floor).sqrt()
            delta_b_for_bias = delta_b_for_bias / normalizer

        strength = self._strength_for_hidden(hidden)
        if self.cfg.scale_strategy == "linear":
            scale = 1.0 + float(self.cfg.gamma_a) * strength * delta_a_for_scale
        else:
            alpha_a = float(self.cfg.alpha_a if self.cfg.alpha_a is not None else self.cfg.gamma_a)
            alpha_a = alpha_a * strength
            scale = torch.exp(alpha_a * torch.tanh(delta_a_for_scale))
        bias = float(self.cfg.gamma_b) * strength * delta_b_for_bias
        self._anchor_losses.append((scale - 1.0).pow(2).mean() + bias.pow(2).mean())
        adapted = scale * hidden + bias
        hidden_norm_after = float(adapted.detach().float().norm().item())
        bias_norm = float(bias.detach().float().norm().item())
        delta_a_f = delta_a.detach().float()
        delta_b_f = delta_b.detach().float()
        delta_a_for_scale_f = delta_a_for_scale.detach().float()
        delta_b_for_bias_f = delta_b_for_bias.detach().float()
        delta_a_rms = float(delta_a.detach().float().pow(2).mean().sqrt().item())
        delta_b_rms = float(delta_b.detach().float().pow(2).mean().sqrt().item())
        delta_a_for_scale_rms = float(delta_a_for_scale.detach().float().pow(2).mean().sqrt().item())
        delta_b_for_bias_rms = float(delta_b_for_bias.detach().float().pow(2).mean().sqrt().item())
        scale_f = scale.detach().float()
        scale_mean = float(scale_f.mean().item())
        scale_std = float(scale_f.std(unbiased=False).item())
        negative_scale_ratio = float((scale_f < 0).float().mean().item())
        scale_above_one_ratio = float((scale_f > 1.0).float().mean().item())
        scale_below_one_ratio = float((scale_f < 1.0).float().mean().item())
        bias_to_hidden_ratio = bias_norm / max(hidden_norm_before, 1e-12)
        hidden_delta_norm_ratio = float(
            (adapted - hidden).detach().float().norm().item() / max(hidden_norm_before, 1e-12)
        )
        state_write_norm = float(write.detach().float().norm().item()) if self.cfg.state_mode != "stateless" else 0.0
        state_delta_norm = float(delta_for_report.detach().float().norm().item()) if self.cfg.state_mode != "stateless" else 0.0
        novelty = 0.0
        if self.cfg.state_mode != "stateless":
            novelty = state_delta_norm / max(float(v.detach().float().norm().item()), 1e-12)

        if update_state:
            with torch.no_grad():
                self.states[state_idx].copy_(new_state.detach().to(device=self.states.device, dtype=self.states.dtype))

        affine_strength_stats = self.affine_strength_stats()
        report: dict[str, object] = {
            "layer": layer,
            "hidden_shape": tuple(hidden.shape),
            "statistics_shape": tuple(stats.shape),
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "v_shape": tuple(v.shape),
            "state_index": state_idx,
            "state_shape": tuple(self.states[state_idx].shape),
            "delta_a_shape": tuple(delta_a.shape),
            "delta_b_shape": tuple(delta_b.shape),
            "state_norm": float(self.states[state_idx].detach().float().norm().item()),
            "delta_a_norm": float(delta_a.detach().float().norm().item()),
            "delta_b_norm": float(delta_b.detach().float().norm().item()),
            "delta_a_mean": float(delta_a_f.mean().item()),
            "delta_a_std": float(delta_a_f.std(unbiased=False).item()),
            "delta_a_min": float(delta_a_f.min().item()),
            "delta_a_max": float(delta_a_f.max().item()),
            "delta_a_rms": delta_a_rms,
            "delta_b_mean": float(delta_b_f.mean().item()),
            "delta_b_std": float(delta_b_f.std(unbiased=False).item()),
            "delta_b_min": float(delta_b_f.min().item()),
            "delta_b_max": float(delta_b_f.max().item()),
            "delta_b_rms": delta_b_rms,
            "delta_a_for_scale_mean": float(delta_a_for_scale_f.mean().item()),
            "delta_a_for_scale_std": float(delta_a_for_scale_f.std(unbiased=False).item()),
            "delta_a_for_scale_min": float(delta_a_for_scale_f.min().item()),
            "delta_a_for_scale_max": float(delta_a_for_scale_f.max().item()),
            "delta_a_for_scale_rms": delta_a_for_scale_rms,
            "delta_b_for_bias_mean": float(delta_b_for_bias_f.mean().item()),
            "delta_b_for_bias_std": float(delta_b_for_bias_f.std(unbiased=False).item()),
            "delta_b_for_bias_min": float(delta_b_for_bias_f.min().item()),
            "delta_b_for_bias_max": float(delta_b_for_bias_f.max().item()),
            "delta_b_for_bias_rms": delta_b_for_bias_rms,
            "scale_mean": scale_mean,
            "scale_std": scale_std,
            "scale_min": float(scale_f.min().item()),
            "scale_max": float(scale_f.max().item()),
            "negative_scale_ratio": negative_scale_ratio,
            "scale_above_one_ratio": scale_above_one_ratio,
            "scale_below_one_ratio": scale_below_one_ratio,
            "bias_to_hidden_ratio": bias_to_hidden_ratio,
            "hidden_delta_norm_ratio": hidden_delta_norm_ratio,
            "hidden_norm_before": hidden_norm_before,
            "hidden_norm_after": hidden_norm_after,
            "state_mode": self.cfg.state_mode,
            "state_sharing": self.cfg.state_sharing,
            "state_input_mode": state_input_mode,
            "affine_mode": self.cfg.affine_mode,
            "affine_strength": affine_strength_stats["gate_mean"],
            "effective_gamma_a": float(self.cfg.gamma_a * affine_strength_stats["gate_mean"]),
            "effective_gamma_b": float(self.cfg.gamma_b * affine_strength_stats["gate_mean"]),
            "num_state_tokens": float(num_state_tokens),
            "hidden_token_norm": hidden_token_norm,
            "state_write_norm": state_write_norm,
            "state_delta_norm": state_delta_norm,
            "novelty": novelty,
            "token_attention_entropy": token_attention_entropy if self.cfg.state_mode != "stateless" else 0.0,
            "token_attention_max": token_attention_max if self.cfg.state_mode != "stateless" else 0.0,
        }
        return adapted, report

    @torch.no_grad()
    def test_time_forward(
        self,
        deployment_model,
        x: torch.Tensor,
        stats_node_mask: torch.Tensor | None = None,
        update_state: bool = True,
        return_hidden_states: bool = False,
    ):
        was_training = self.training
        self.eval()
        try:
            if (
                not return_hidden_states
                and self.cfg.state_mode == "stateless"
                and self.cfg.affine_mode == "direct"
                and hasattr(deployment_model, "forward_with_stateless_direct_tta")
            ):
                output = deployment_model.forward_with_stateless_direct_tta(x, self)
            else:
                output = deployment_model.forward_with_tta(
                    x,
                    self,
                    stats_node_mask=stats_node_mask,
                    update_state=update_state,
                    return_hidden_states=return_hidden_states,
                )
        finally:
            self.train(was_training)
        if return_hidden_states:
            logits, hidden_states, reports = output
            self.last_layer_reports = list(reports)
            return logits, hidden_states, reports
        self.last_layer_reports = []
        return output


def save_tta_controller_checkpoint(
    path: str | Path,
    controller: LayerwiseHiddenAffineTTA,
    *,
    variant: str,
    train_metadata: dict[str, object] | None = None,
) -> None:
    """Persist a source-pretrained TTA controller for later evaluation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "cfg": asdict(controller.cfg),
            "num_layers": controller.num_layers,
            "channel_dim": controller.channel_dim,
            "state_dict": controller.state_dict(),
            "train_metadata": train_metadata or {},
        },
        path,
    )


def load_tta_controller_checkpoint(path: str | Path, controller: LayerwiseHiddenAffineTTA) -> dict[str, object]:
    """Load a source-pretrained TTA controller and validate compatible shapes."""

    device = next(controller.parameters()).device
    checkpoint = torch.load(Path(path), map_location=device)
    loaded_cfg = checkpoint.get("cfg", {})
    expected = {
        "num_layers": controller.num_layers,
        "channel_dim": controller.channel_dim,
        "rank": controller.cfg.rank,
        "state_mode": controller.cfg.state_mode,
        "state_input_mode": controller.cfg.state_input_mode,
        "affine_mode": controller.cfg.affine_mode,
    }
    loaded = {
        "num_layers": checkpoint.get("num_layers"),
        "channel_dim": checkpoint.get("channel_dim"),
        "rank": loaded_cfg.get("rank"),
        "state_mode": loaded_cfg.get("state_mode"),
        "state_input_mode": loaded_cfg.get("state_input_mode"),
        "affine_mode": loaded_cfg.get("affine_mode"),
    }
    mismatches = {key: (expected[key], loaded.get(key)) for key in expected if expected[key] != loaded.get(key)}
    if mismatches:
        raise ValueError(f"TTA controller checkpoint shape/config mismatch: {mismatches}")
    controller.load_state_dict(checkpoint["state_dict"], strict=True)
    controller.reset_state()
    return checkpoint
