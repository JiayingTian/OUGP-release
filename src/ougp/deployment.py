"""Deployment-time compact GCN utilities for OUGP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ougp.model import OUGPGCN, append_unit_self_loops, symmetric_norm


HARDENING_MODES = ("topk_binary", "quantile_binary", "quantile_truncated")


@dataclass(frozen=True)
class HardenedMask:
    """Structural support and retained values for one deployment pruning lane."""

    support: torch.Tensor
    values: torch.Tensor
    threshold: float
    mode: str
    target_keep_rate: float
    realized_keep_rate: float


@dataclass(frozen=True)
class DeploymentMaskSet:
    graph: HardenedMask
    parameter: HardenedMask

    @property
    def channel(self) -> HardenedMask:
        """Backward-compatible alias for legacy channel-pruning callers."""

        return self.parameter


def hard_topk_mask(scores: torch.Tensor, keep_rate: float) -> torch.Tensor:
    """Return a binary deployment mask with exactly the requested keep budget."""

    if scores.numel() == 0:
        return torch.zeros_like(scores)
    keep_rate = float(min(1.0, max(0.0, keep_rate)))
    keep_count = int(round(scores.numel() * keep_rate))
    keep_count = max(1, min(scores.numel(), keep_count))
    indices = torch.topk(scores.detach().float(), k=keep_count).indices
    mask = torch.zeros_like(scores, dtype=torch.float32)
    mask[indices] = 1.0
    return mask.to(device=scores.device, dtype=scores.dtype)


def _keep_count(numel: int, keep_rate: float) -> int:
    if numel <= 0:
        return 0
    keep_rate = float(min(1.0, max(0.0, keep_rate)))
    return max(1, min(numel, int(round(numel * keep_rate))))


def _quantile_support(values: torch.Tensor, keep_rate: float) -> tuple[torch.Tensor, float]:
    """Return an exact-size support using a pruning quantile and deterministic ties."""

    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool), 0.0
    keep_count = _keep_count(flat.numel(), keep_rate)
    prune_rate = 1.0 - float(min(1.0, max(0.0, keep_rate)))
    threshold_tensor = torch.quantile(flat, prune_rate)
    support = flat > threshold_tensor
    selected = int(support.sum().item())
    if selected < keep_count:
        remaining = torch.nonzero(~support, as_tuple=False).flatten()
        remaining_values = flat.index_select(0, remaining)
        order = torch.argsort(remaining_values, descending=True, stable=True)
        support[remaining.index_select(0, order[: keep_count - selected])] = True
    elif selected > keep_count:
        selected_indices = torch.nonzero(support, as_tuple=False).flatten()
        selected_values = flat.index_select(0, selected_indices)
        order = torch.argsort(selected_values, descending=True, stable=True)
        support.zero_()
        support[selected_indices.index_select(0, order[:keep_count])] = True
    return support.reshape_as(values), float(threshold_tensor.item())


def harden_soft_mask(soft_mask: torch.Tensor, keep_rate: float, mode: str) -> HardenedMask:
    """Harden a continuous mask into binary or truncated deployment values."""

    if mode not in {"quantile_binary", "quantile_truncated"}:
        raise ValueError(f"Unsupported soft-mask hardening mode: {mode!r}")
    support, threshold = _quantile_support(soft_mask, keep_rate)
    if mode == "quantile_binary":
        values = support.to(dtype=soft_mask.dtype)
    else:
        values = soft_mask.detach().clone() * support.to(dtype=soft_mask.dtype)
    realized = float(support.detach().float().mean().item()) if support.numel() else 0.0
    return HardenedMask(
        support=support,
        values=values,
        threshold=threshold,
        mode=mode,
        target_keep_rate=float(keep_rate),
        realized_keep_rate=realized,
    )


def _binary_mask_spec(scores: torch.Tensor, keep_rate: float) -> HardenedMask:
    values = hard_topk_mask(scores, keep_rate)
    support = values.detach().bool()
    kept_scores = scores.detach().float()[support]
    threshold = float(kept_scores.min().item()) if kept_scores.numel() else 0.0
    realized = float(support.float().mean().item()) if support.numel() else 0.0
    return HardenedMask(
        support=support,
        values=values,
        threshold=threshold,
        mode="topk_binary",
        target_keep_rate=float(keep_rate),
        realized_keep_rate=realized,
    )


def deployment_mask_set(
    model: OUGPGCN,
    *,
    mode: str = "topk_binary",
    temperature: float = 0.5,
) -> DeploymentMaskSet:
    """Build graph/parameter deployment support and values for one hardening mode."""

    if mode not in HARDENING_MODES:
        raise ValueError(f"Unknown hardening mode {mode!r}; expected one of {HARDENING_MODES}.")
    if mode == "topk_binary":
        if model.cfg.use_graph_pruning and model.last_graph_score is not None:
            graph = _binary_mask_spec(model.last_graph_score, model.cfg.graph_target_keep)
        else:
            graph_scores = torch.ones(model.cfg.num_edges, device=model.base_edge_index.device)
            graph = _binary_mask_spec(graph_scores, 1.0)
        if model.cfg.use_param_pruning and model.last_param_score is not None:
            parameter = _binary_mask_spec(model.last_param_score, model.cfg.param_target_keep)
        else:
            parameter_scores = torch.ones(model.param_item_count, device=model.base_edge_index.device)
            parameter = _binary_mask_spec(parameter_scores, 1.0)
        return DeploymentMaskSet(graph=graph, parameter=parameter)

    soft_graph, soft_parameter = soft_deployment_masks(model, temperature)
    graph = harden_soft_mask(
        soft_graph,
        model.cfg.graph_target_keep if model.cfg.use_graph_pruning else 1.0,
        mode,
    )
    parameter = harden_soft_mask(
        soft_parameter,
        model.cfg.param_target_keep if model.cfg.use_param_pruning else 1.0,
        mode,
    )
    return DeploymentMaskSet(graph=graph, parameter=parameter)


def deployment_masks(model: OUGPGCN) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze final pruning scores into hard masks for deployment-style profiling."""

    if model.cfg.use_graph_pruning and model.last_graph_score is not None:
        graph_mask = hard_topk_mask(model.last_graph_score, model.cfg.graph_target_keep)
    elif model.last_graph_mask is not None:
        graph_mask = torch.ones_like(model.last_graph_mask)
    else:
        graph_mask = torch.ones(model.cfg.num_edges, device=model.base_edge_index.device)

    if model.cfg.use_param_pruning and model.last_param_score is not None:
        param_mask = hard_topk_mask(model.last_param_score, model.cfg.param_target_keep)
    elif model.last_param_mask is not None:
        param_mask = torch.ones_like(model.last_param_mask)
    else:
        param_mask = torch.ones(model.param_item_count, device=model.base_edge_index.device)
    return graph_mask, param_mask


def soft_deployment_masks(model: OUGPGCN, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze the continuous source masks for a soft-mask diagnostic forward."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        graph_mask, param_mask, _ = model.masks(temperature)
    model.train(was_training)
    return graph_mask.detach().clone(), param_mask.detach().clone()


class MaterializedGCNDeployment(torch.nn.Module):
    """GCN deployment graph with pruned edges/channels materialized once."""

    def __init__(
        self,
        model: OUGPGCN,
        graph_mask: torch.Tensor,
        param_mask: torch.Tensor,
        x_dtype: torch.dtype,
        edge_index_override: torch.Tensor | None = None,
        num_nodes_override: int | None = None,
        graph_support: torch.Tensor | None = None,
        param_support: torch.Tensor | None = None,
    ):
        super().__init__()
        if model.cfg.backbone != "gcn":
            raise ValueError("materialized deployment currently supports --backbone gcn only.")

        self.param_pruning_granularity = model.cfg.param_pruning_granularity
        parameter_support = (
            param_mask.detach().float() > 0.5
            if param_support is None
            else param_support.detach().bool()
        )
        if parameter_support.numel() != param_mask.numel():
            raise ValueError("param_support must match param_mask.")
        if self.param_pruning_granularity == "weight":
            keep_channels = torch.arange(model.cfg.hidden_dim, device=model.lin1.weight.device)
            channel_values = torch.ones(
                model.cfg.hidden_dim, device=model.lin1.weight.device, dtype=model.lin1.weight.dtype
            )
            parameter_masks = model.split_parameter_mask(param_mask.detach())
        else:
            keep_channels = torch.nonzero(parameter_support, as_tuple=False).flatten()
            if keep_channels.numel() == 0:
                keep_channels = torch.topk(param_mask.detach().float(), k=1).indices
            keep_channels = keep_channels.to(device=model.lin1.weight.device)
            channel_values = param_mask.detach().index_select(0, keep_channels).to(
                device=model.lin1.weight.device,
                dtype=model.lin1.weight.dtype,
            )
            parameter_masks = []

        base_edge_index = model.base_edge_index if edge_index_override is None else edge_index_override
        non_loop_mask = base_edge_index[0] != base_edge_index[1]
        if not bool(non_loop_mask.all()):
            base_edge_index = base_edge_index[:, non_loop_mask]
            graph_mask = graph_mask[non_loop_mask]
            if graph_support is not None:
                graph_support = graph_support[non_loop_mask]
        keep_edges = (
            graph_mask.detach().float() > 0.5
            if graph_support is None
            else graph_support.detach().bool()
        )
        if keep_edges.numel() != base_edge_index.size(1):
            raise ValueError(
                "graph_mask must match the deployment edge index; "
                f"got {keep_edges.numel()} vs {base_edge_index.size(1)}."
            )
        pruned_edge_index = base_edge_index[:, keep_edges]
        pruned_edge_values = graph_mask.detach().to(dtype=x_dtype).flatten()[keep_edges]
        num_nodes = model.cfg.num_nodes if num_nodes_override is None else int(num_nodes_override)
        edge_index, edge_weight = append_unit_self_loops(
            pruned_edge_index,
            pruned_edge_values,
            num_nodes,
        )
        norm_weight = symmetric_norm(edge_index, edge_weight, num_nodes)
        adj = torch.sparse_coo_tensor(edge_index, norm_weight, (num_nodes, num_nodes), device=edge_index.device)
        self.register_buffer("adj", adj.coalesce())
        self.register_buffer("keep_channels", keep_channels)
        self.register_buffer("channel_values", channel_values)

        if self.param_pruning_granularity == "weight":
            self.lin1_weight = torch.nn.Parameter(
                (model.lin1.weight.detach() * parameter_masks[0]).clone(), requires_grad=False
            )
            self.hidden_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        (layer.weight.detach() * parameter_masks[index + 1]).clone(),
                        requires_grad=False,
                    )
                    for index, layer in enumerate(model.deep_hidden_lins)
                ]
            )
            self.lin2_weight = torch.nn.Parameter(
                (model.lin2.weight.detach() * parameter_masks[-1]).clone(), requires_grad=False
            )
        else:
            self.lin1_weight = torch.nn.Parameter(
                model.lin1.weight.detach().index_select(0, keep_channels).clone(),
                requires_grad=False,
            )
            self.hidden_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        layer.weight.detach().index_select(0, keep_channels).index_select(1, keep_channels).clone(),
                        requires_grad=False,
                    )
                    for layer in model.deep_hidden_lins
                ]
            )
            self.lin2_weight = torch.nn.Parameter(
                model.lin2.weight.detach().index_select(1, keep_channels).clone(),
                requires_grad=False,
            )
        self.materialized_edge_count = int(pruned_edge_index.size(1))
        self.materialized_edge_count_with_self_loops = int(edge_index.size(1))
        self.materialized_channel_count = int(keep_channels.numel())
        self.materialized_parameter_count = int(parameter_support.sum().item())
        self.materialized_parameter_total = int(parameter_support.numel())

    @property
    def hidden_layer_count(self) -> int:
        return 1 + len(self.hidden_weights)

    def _record_hidden(
        self,
        hidden_states: list[dict[str, torch.Tensor | str | int]],
        layer: int,
        kind: str,
        value: torch.Tensor,
    ) -> None:
        hidden_states.append({"layer": layer, "kind": kind, "tensor": value})

    def forward(
        self,
        x: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor | str | int]]]:
        hidden_states: list[dict[str, torch.Tensor | str | int]] = []

        h = torch.sparse.mm(self.adj, x)
        h = F.linear(h, self.lin1_weight)
        if return_hidden_states:
            self._record_hidden(hidden_states, 0, "pre_activation", h)
        h = F.relu(h) * self.channel_values
        if return_hidden_states:
            self._record_hidden(hidden_states, 0, "activation", h)

        for layer_idx, hidden_weight in enumerate(self.hidden_weights, start=1):
            h = torch.sparse.mm(self.adj, h)
            h = F.linear(h, hidden_weight)
            if return_hidden_states:
                self._record_hidden(hidden_states, layer_idx, "pre_activation", h)
            h = F.relu(h) * self.channel_values
            if return_hidden_states:
                self._record_hidden(hidden_states, layer_idx, "activation", h)

        h = torch.sparse.mm(self.adj, h)
        logits = F.linear(h, self.lin2_weight)
        if return_hidden_states:
            self._record_hidden(hidden_states, self.hidden_layer_count, "logits", logits)
            return logits, hidden_states
        return logits

    def forward_with_tta(
        self,
        x: torch.Tensor,
        controller: Any,
        stats_node_mask: torch.Tensor | None = None,
        update_state: bool = True,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor | str | int]], list[dict[str, object]]]:
        hidden_states: list[dict[str, torch.Tensor | str | int]] = []
        reports: list[dict[str, object]] = []
        if hasattr(controller, "reset_forward_losses"):
            controller.reset_forward_losses()

        h = torch.sparse.mm(self.adj, x)
        h = F.linear(h, self.lin1_weight)
        if return_hidden_states:
            self._record_hidden(hidden_states, 0, "pre_activation", h)
        h = F.relu(h) * self.channel_values
        h, report = controller.adapt_layer(0, h, node_mask=stats_node_mask, update_state=update_state)
        reports.append(report)
        if return_hidden_states:
            self._record_hidden(hidden_states, 0, "activation", h)

        for layer_idx, hidden_weight in enumerate(self.hidden_weights, start=1):
            h = torch.sparse.mm(self.adj, h)
            h = F.linear(h, hidden_weight)
            if return_hidden_states:
                self._record_hidden(hidden_states, layer_idx, "pre_activation", h)
            h = F.relu(h) * self.channel_values
            h, report = controller.adapt_layer(layer_idx, h, node_mask=stats_node_mask, update_state=update_state)
            reports.append(report)
            if return_hidden_states:
                self._record_hidden(hidden_states, layer_idx, "activation", h)

        h = torch.sparse.mm(self.adj, h)
        h = controller.apply_output_hidden_adapter(h)
        logits = F.linear(h, self.lin2_weight)
        logits = controller.apply_logit_residual(h, logits)
        if return_hidden_states:
            self._record_hidden(hidden_states, self.hidden_layer_count, "logits", logits)
            return logits, hidden_states, reports
        return logits

    @torch.no_grad()
    def forward_with_stateless_direct_tta(
        self,
        x: torch.Tensor,
        controller: Any,
    ) -> torch.Tensor:
        """Fast deployment path for EXP121's stateless direct affine mode."""

        h = torch.sparse.mm(self.adj, x)
        h = F.linear(h, self.lin1_weight)
        h = F.relu(h) * self.channel_values
        h = controller.apply_stateless_direct_affine(0, h)

        for layer_idx, hidden_weight in enumerate(self.hidden_weights, start=1):
            h = torch.sparse.mm(self.adj, h)
            h = F.linear(h, hidden_weight)
            h = F.relu(h) * self.channel_values
            h = controller.apply_stateless_direct_affine(layer_idx, h)

        h = torch.sparse.mm(self.adj, h)
        h = controller.apply_output_hidden_adapter(h)
        logits = F.linear(h, self.lin2_weight)
        return controller.apply_logit_residual(h, logits)


class SoftMaskedGCNDeployment(torch.nn.Module):
    """Frozen continuous-mask GCN used only to measure the soft-to-hard gap."""

    def __init__(
        self,
        model: OUGPGCN,
        graph_mask: torch.Tensor,
        param_mask: torch.Tensor,
        x_dtype: torch.dtype,
    ):
        super().__init__()
        if model.cfg.backbone != "gcn":
            raise ValueError("soft-mask deployment currently supports --backbone gcn only.")
        if graph_mask.numel() != model.base_edge_index.size(1):
            raise ValueError("graph_mask must match the base edge index.")
        if param_mask.numel() != model.cfg.hidden_dim:
            raise ValueError("param_mask must match the hidden dimension.")

        num_nodes = model.cfg.num_nodes
        edge_index, edge_weight = append_unit_self_loops(
            model.base_edge_index,
            graph_mask.detach().to(dtype=x_dtype),
            num_nodes,
        )
        norm_weight = symmetric_norm(edge_index, edge_weight, num_nodes)
        adj = torch.sparse_coo_tensor(edge_index, norm_weight, (num_nodes, num_nodes), device=edge_index.device)
        self.register_buffer("adj", adj.coalesce())
        self.register_buffer("param_mask", param_mask.detach().clone())
        self.lin1_weight = torch.nn.Parameter(model.lin1.weight.detach().clone(), requires_grad=False)
        self.hidden_weights = torch.nn.ParameterList(
            [torch.nn.Parameter(layer.weight.detach().clone(), requires_grad=False) for layer in model.deep_hidden_lins]
        )
        self.lin2_weight = torch.nn.Parameter(model.lin2.weight.detach().clone(), requires_grad=False)
        self.materialized_edge_count = int(model.base_edge_index.size(1))
        self.materialized_edge_count_with_self_loops = int(edge_index.size(1))
        self.materialized_channel_count = int(model.cfg.hidden_dim)

    @property
    def hidden_layer_count(self) -> int:
        return 1 + len(self.hidden_weights)

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(F.linear(torch.sparse.mm(self.adj, x), self.lin1_weight)) * self.param_mask
        for weight in self.hidden_weights:
            h = F.relu(F.linear(torch.sparse.mm(self.adj, h), weight)) * self.param_mask
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(torch.sparse.mm(self.adj, self._hidden(x)), self.lin2_weight)

    def forward_with_tta(
        self,
        x: torch.Tensor,
        controller: Any,
        stats_node_mask: torch.Tensor | None = None,
        update_state: bool = True,
        return_hidden_states: bool = False,
    ):
        hidden_states: list[dict[str, torch.Tensor | str | int]] = []
        reports: list[dict[str, object]] = []
        h = F.relu(F.linear(torch.sparse.mm(self.adj, x), self.lin1_weight)) * self.param_mask
        h, report = controller.adapt_layer(0, h, node_mask=stats_node_mask, update_state=update_state)
        reports.append(report)
        if return_hidden_states:
            hidden_states.append({"layer": 0, "kind": "activation", "tensor": h})
        for layer_idx, weight in enumerate(self.hidden_weights, start=1):
            h = F.relu(F.linear(torch.sparse.mm(self.adj, h), weight)) * self.param_mask
            h, report = controller.adapt_layer(layer_idx, h, node_mask=stats_node_mask, update_state=update_state)
            reports.append(report)
            if return_hidden_states:
                hidden_states.append({"layer": layer_idx, "kind": "activation", "tensor": h})
        propagated = torch.sparse.mm(self.adj, h)
        propagated = controller.apply_output_hidden_adapter(propagated)
        logits = F.linear(propagated, self.lin2_weight)
        logits = controller.apply_logit_residual(propagated, logits)
        if return_hidden_states:
            hidden_states.append({"layer": self.hidden_layer_count, "kind": "logits", "tensor": logits})
            return logits, hidden_states, reports
        return logits

    @torch.no_grad()
    def forward_with_stateless_direct_tta(self, x: torch.Tensor, controller: Any) -> torch.Tensor:
        h = F.relu(F.linear(torch.sparse.mm(self.adj, x), self.lin1_weight)) * self.param_mask
        h = controller.apply_stateless_direct_affine(0, h)
        for layer_idx, weight in enumerate(self.hidden_weights, start=1):
            h = F.relu(F.linear(torch.sparse.mm(self.adj, h), weight)) * self.param_mask
            h = controller.apply_stateless_direct_affine(layer_idx, h)
        propagated = torch.sparse.mm(self.adj, h)
        propagated = controller.apply_output_hidden_adapter(propagated)
        logits = F.linear(propagated, self.lin2_weight)
        return controller.apply_logit_residual(propagated, logits)
