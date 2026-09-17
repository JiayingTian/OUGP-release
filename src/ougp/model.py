"""OUGP modules for the first citation-network case study."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch
from torch import nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def normalize_utility_signal(utility: torch.Tensor) -> torch.Tensor:
    utility = utility.detach().float()
    if utility.numel() <= 1:
        return utility / utility.abs().mean().clamp_min(1e-6)
    return (utility - utility.mean()) / utility.std(unbiased=False).clamp_min(1e-6)


def ste_sigmoid(logits: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
    soft = torch.sigmoid(logits / temperature)
    if not hard:
        return soft
    hard_mask = (soft >= 0.5).to(soft.dtype)
    return hard_mask.detach() - soft.detach() + soft


def budgeted_sigmoid(logits: torch.Tensor, keep_rate: float, temperature: float, hard: bool) -> torch.Tensor:
    keep_rate = float(min(0.999, max(0.001, keep_rate)))
    threshold = torch.quantile(logits.detach().float(), 1.0 - keep_rate)
    prior = torch.logit(logits.new_tensor(keep_rate), eps=1e-6)
    soft = torch.sigmoid((logits - threshold) / temperature + prior)
    soft = (soft * (keep_rate / soft.detach().mean().clamp_min(1e-6))).clamp(0.0, 1.0)
    if not hard:
        return soft
    hard_mask = (logits >= threshold).to(logits.dtype)
    return hard_mask.detach() - soft.detach() + soft


def symmetric_norm(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    degree = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    degree.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def append_unit_self_loops(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discard input loops and append exactly one fixed unit loop per node."""

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1):
        raise ValueError("edge_weight must have shape [num_edges].")
    row, col = edge_index
    non_loop_mask = row != col
    loop_nodes = torch.arange(num_nodes, device=edge_index.device)
    loop_index = torch.stack([loop_nodes, loop_nodes], dim=0)
    return (
        torch.cat([edge_index[:, non_loop_mask], loop_index], dim=1),
        torch.cat(
            [
                edge_weight[non_loop_mask],
                torch.ones(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype),
            ],
            dim=0,
        ),
    )


def sparse_gcn_mm(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=x.device)
    return torch.sparse.mm(adj.coalesce(), x)


def sparse_mean_mm(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, _ = edge_index
    degree = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    degree.scatter_add_(0, row, edge_weight)
    norm_weight = edge_weight / degree[row].clamp_min(1e-12)
    return sparse_gcn_mm(edge_index, norm_weight, x, num_nodes)


def sparse_gat_mm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    x: torch.Tensor,
    attn_src: torch.Tensor,
    attn_dst: torch.Tensor,
    num_nodes: int,
    negative_slope: float = 0.2,
) -> torch.Tensor:
    row, col = edge_index
    logits = (x[col] * attn_src).sum(dim=-1) + (x[row] * attn_dst).sum(dim=-1)
    logits = F.leaky_relu(logits, negative_slope=negative_slope)
    max_per_row = torch.full((num_nodes,), -torch.inf, device=x.device, dtype=x.dtype)
    max_per_row.scatter_reduce_(0, row, logits, reduce="amax", include_self=True)
    exp_logits = torch.exp(logits - max_per_row[row]) * edge_weight.clamp_min(0.0)
    denom = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    denom.scatter_add_(0, row, exp_logits)
    attn_weight = exp_logits / denom[row].clamp_min(1e-12)
    return sparse_gcn_mm(edge_index, attn_weight, x, num_nodes)


@dataclass(frozen=True)
class OUGPConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_nodes: int
    num_edges: int
    num_gnn_layers: int = 2
    edge_context_dim: int = 15
    param_context_dim: int = 6
    memory_rank: int = 8
    feature_context_dim: int = 6
    graph_target_keep: float = 0.70
    param_target_keep: float = 0.70
    graph_gamma: float = 0.35
    param_gamma: float = 0.35
    graph_score_scale_decay: float = 0.95
    graph_score_scale_min: float = 0.02
    graph_score_scale_max: float = 0.50
    graph_correction_clip: float = 2.0
    param_score_scale_decay: float = 0.95
    param_score_scale_min: float = 0.02
    param_score_scale_max: float = 0.50
    param_correction_clip: float = 2.0
    cross_gamma: float = 0.20
    use_hidden_coupling: bool = False
    hidden_coupling_mix_graph: float = 0.0
    hidden_coupling_mix_param: float = 0.0
    hidden_coupling_interval: int = 1
    hidden_coupling_start_epoch: int = 0
    hidden_coupling_layer_norm_weight: float = 1.0
    hidden_coupling_interaction_weight: float = 1.0
    hidden_coupling_relation_weight: float = 1.0
    hidden_coupling_param_damage_weight: float = 1.0
    hidden_coupling_graph_damage_weight: float = 1.0
    write_beta: float = 0.12
    write_lambda: float = 0.98
    event_gamma: float = 0.0
    event_beta: float = 0.10
    event_decay: float = 0.95
    event_top_k: int = 2000
    recall_gamma: float = 0.0
    recall_beta: float = 0.10
    recall_decay: float = 0.95
    recall_top_k: int = 2000
    use_steering_memory: bool = False
    steer_context_dim: int = 10
    steer_gamma: float = 0.0
    steer_beta: float = 0.10
    steer_lambda: float = 0.95
    hard_masks: bool = False
    use_graph_pruning: bool = True
    use_param_pruning: bool = True
    use_memory: bool = True
    use_cross: bool = True
    backbone: str = "gcn"
    budget_target: float = 0.70
    memory_write_mode: str = "residual"
    graph_memory_granularity: str = "edge"
    graph_memory_layout: str = "single"
    use_graph_full_branch: bool = True
    use_graph_grad_branch: bool = True
    use_graph_branch_gates: bool = True
    param_memory_layout: str = "single"
    param_pruning_granularity: str = "channel"
    param_policy_granularity: str = "auto"
    graph_score_init: str = "constant"
    param_score_init: str = "constant"
    freeze_pruning_scores: bool = False
    seed: int = 0


class OnlinePruningMemory(nn.Module):
    """Fixed-capacity residual utility memory."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        write_beta: float,
        write_lambda: float,
        event_items: int = 0,
        event_beta: float = 0.10,
        event_decay: float = 0.95,
        recall_items: int = 0,
        recall_beta: float = 0.10,
        recall_decay: float = 0.95,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.event_beta = event_beta
        self.event_decay = event_decay
        self.recall_beta = recall_beta
        self.recall_decay = recall_decay
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.read_head = nn.Linear(rank, 1, bias=False)
        self.utility_head = nn.Linear(rank, 1, bias=False)
        self.register_buffer("state", torch.zeros(rank, rank))
        self.register_buffer("event_bias", torch.zeros(max(0, event_items)))
        self.register_buffer("recall_bias", torch.zeros(max(0, recall_items)))

    def reset_state(self) -> None:
        self.state.zero_()
        self.event_bias.zero_()
        self.recall_bias.zero_()

    def project_qkv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = l2_normalize(torch.tanh(self.q_proj(context)))
        k = l2_normalize(torch.tanh(self.k_proj(context)))
        v = self.v_proj(context)
        return q, k, v

    def read(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.project_qkv(context)
        read_vec = q @ self.state.t()
        correction = self.read_head(read_vec).squeeze(-1)
        return correction, k, v, read_vec

    @torch.no_grad()
    def write(self, context: torch.Tensor, utility: torch.Tensor, mode: str = "residual") -> dict[str, float]:
        if mode not in {"residual", "feature", "none"}:
            raise ValueError("memory write mode must be one of: residual, feature, none.")
        if mode == "none":
            return {
                "write_mode": 0.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
            }
        if context.numel() == 0:
            return {
                "write_mode": 1.0 if mode == "residual" else 2.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
            }
        _, k, v = self.project_qkv(context.detach())
        pred_vec = k @ self.state.t()
        pred = self.utility_head(pred_vec).squeeze(-1)
        utility = normalize_utility_signal(utility)
        residual = utility - pred
        if mode == "residual":
            target_v = v * residual.tanh().unsqueeze(-1)
        else:
            target_v = v
        current = k @ self.state.t()
        delta_vec = target_v - current
        delta_state = torch.einsum("br,bk->rk", delta_vec, k) / max(1, context.size(0))
        self.state.mul_(self.write_lambda).add_(self.write_beta * delta_state)
        self.state.clamp_(-5.0, 5.0)
        return {
            "write_mode": 1.0 if mode == "residual" else 2.0,
            "utility_mean": float(utility.mean().item()),
            "residual_mean": float(residual.abs().mean().item()),
            "state_norm": float(self.state.norm().item()),
        }

    @torch.no_grad()
    def event_correction(self, reference: torch.Tensor) -> torch.Tensor:
        if self.event_bias.numel() != reference.numel():
            return torch.zeros_like(reference)
        return self.event_bias.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def recall_correction(self, reference: torch.Tensor) -> torch.Tensor:
        if self.recall_bias.numel() != reference.numel():
            return torch.zeros_like(reference)
        return self.recall_bias.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def write_events(self, event_delta: torch.Tensor, top_k: int) -> dict[str, float]:
        if self.event_bias.numel() == 0 or event_delta.numel() == 0:
            return {"updates": 0.0, "bias_norm": float(self.event_bias.norm().item())}
        if self.event_bias.numel() != event_delta.numel():
            raise ValueError("event_delta shape must match event_bias shape.")

        self.event_bias.mul_(self.event_decay)
        event_delta = event_delta.detach().float().to(self.event_bias.device)
        active = event_delta.abs() > 0
        if not bool(active.any().item()):
            return {"updates": 0.0, "bias_norm": float(self.event_bias.norm().item())}

        if top_k > 0 and int(active.sum().item()) > top_k:
            indices = torch.topk(event_delta.abs(), k=top_k).indices
            self.event_bias[indices] += self.event_beta * event_delta[indices]
            updates = float(indices.numel())
        else:
            self.event_bias.add_(self.event_beta * event_delta)
            updates = float(active.sum().item())
        self.event_bias.clamp_(-5.0, 5.0)
        return {
            "updates": updates,
            "bias_mean": float(self.event_bias.mean().item()),
            "bias_abs_mean": float(self.event_bias.abs().mean().item()),
            "bias_norm": float(self.event_bias.norm().item()),
        }

    @torch.no_grad()
    def write_recall(self, recall_delta: torch.Tensor, top_k: int) -> dict[str, float]:
        if self.recall_bias.numel() == 0 or recall_delta.numel() == 0:
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}
        if self.recall_bias.numel() != recall_delta.numel():
            raise ValueError("recall_delta shape must match recall_bias shape.")

        self.recall_bias.mul_(self.recall_decay)
        recall_delta = recall_delta.detach().float().to(self.recall_bias.device)
        active = recall_delta.abs() > 0
        if not bool(active.any().item()):
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}

        if top_k > 0 and int(active.sum().item()) > top_k:
            indices = torch.topk(recall_delta.abs(), k=top_k).indices
            self.recall_bias[indices] += self.recall_beta * recall_delta[indices]
            updates = float(indices.numel())
        else:
            self.recall_bias.add_(self.recall_beta * recall_delta)
            updates = float(active.sum().item())
        self.recall_bias.clamp_(-5.0, 5.0)
        return {
            "updates": updates,
            "bias_mean": float(self.recall_bias.mean().item()),
            "bias_abs_mean": float(self.recall_bias.abs().mean().item()),
            "bias_norm": float(self.recall_bias.norm().item()),
        }


class ChannelPruningMemory(nn.Module):
    """Channel-specific residual utility memory for parameter pruning."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        num_channels: int,
        write_beta: float,
        write_lambda: float,
        recall_beta: float = 0.10,
        recall_decay: float = 0.95,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.num_channels = num_channels
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.recall_beta = recall_beta
        self.recall_decay = recall_decay
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.read_head = nn.Linear(rank, 1, bias=False)
        self.utility_head = nn.Linear(rank, 1, bias=False)
        self.register_buffer("state", torch.zeros(num_channels, rank, rank))
        self.register_buffer("event_bias", torch.zeros(0))
        self.register_buffer("recall_bias", torch.zeros(num_channels))

    def reset_state(self) -> None:
        self.state.zero_()
        self.recall_bias.zero_()

    def project_qkv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = l2_normalize(torch.tanh(self.q_proj(context)))
        k = l2_normalize(torch.tanh(self.k_proj(context)))
        v = self.v_proj(context)
        return q, k, v

    def _validate_context(self, context: torch.Tensor) -> None:
        if context.size(0) != self.num_channels:
            raise ValueError("ChannelPruningMemory context rows must match num_channels.")

    def read(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_context(context)
        q, k, v = self.project_qkv(context)
        read_vec = torch.einsum("ck,cok->co", q, self.state)
        correction = self.read_head(read_vec).squeeze(-1)
        return correction, k, v, read_vec

    @torch.no_grad()
    def write(self, context: torch.Tensor, utility: torch.Tensor, mode: str = "residual") -> dict[str, float]:
        if mode not in {"residual", "feature", "none"}:
            raise ValueError("memory write mode must be one of: residual, feature, none.")
        if mode == "none":
            channel_norm = self.state.detach().float().norm(dim=(1, 2))
            return {
                "write_mode": 0.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
                "channel_state_norm_mean": float(channel_norm.mean().item()),
                "channel_state_norm_std": float(channel_norm.std(unbiased=False).item()),
            }
        if context.numel() == 0:
            channel_norm = self.state.detach().float().norm(dim=(1, 2))
            return {
                "write_mode": 1.0 if mode == "residual" else 2.0,
                "utility_mean": 0.0,
                "residual_mean": 0.0,
                "state_norm": float(self.state.norm().item()),
                "channel_state_norm_mean": float(channel_norm.mean().item()),
                "channel_state_norm_std": float(channel_norm.std(unbiased=False).item()),
            }
        self._validate_context(context)
        _, k, v = self.project_qkv(context.detach())
        pred_vec = torch.einsum("ck,cok->co", k, self.state)
        pred = self.utility_head(pred_vec).squeeze(-1)
        utility = normalize_utility_signal(utility)
        residual = utility - pred
        if mode == "residual":
            target_v = v * residual.tanh().unsqueeze(-1)
        else:
            target_v = v
        current = torch.einsum("ck,cok->co", k, self.state)
        delta_vec = target_v - current
        delta_state = torch.einsum("co,ck->cok", delta_vec, k)
        self.state.mul_(self.write_lambda).add_(self.write_beta * delta_state)
        self.state.clamp_(-5.0, 5.0)
        channel_norm = self.state.detach().float().norm(dim=(1, 2))
        return {
            "write_mode": 1.0 if mode == "residual" else 2.0,
            "utility_mean": float(utility.mean().item()),
            "residual_mean": float(residual.abs().mean().item()),
            "state_norm": float(self.state.norm().item()),
            "channel_state_norm_mean": float(channel_norm.mean().item()),
            "channel_state_norm_std": float(channel_norm.std(unbiased=False).item()),
        }

    @torch.no_grad()
    def recall_correction(self, reference: torch.Tensor) -> torch.Tensor:
        if self.recall_bias.numel() != reference.numel():
            return torch.zeros_like(reference)
        return self.recall_bias.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def write_recall(self, recall_delta: torch.Tensor, top_k: int) -> dict[str, float]:
        if self.recall_bias.numel() == 0 or recall_delta.numel() == 0:
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}
        if self.recall_bias.numel() != recall_delta.numel():
            raise ValueError("recall_delta shape must match recall_bias shape.")

        self.recall_bias.mul_(self.recall_decay)
        recall_delta = recall_delta.detach().float().to(self.recall_bias.device)
        active = recall_delta.abs() > 0
        if not bool(active.any().item()):
            return {"updates": 0.0, "bias_norm": float(self.recall_bias.norm().item())}

        if top_k > 0 and int(active.sum().item()) > top_k:
            indices = torch.topk(recall_delta.abs(), k=top_k).indices
            self.recall_bias[indices] += self.recall_beta * recall_delta[indices]
            updates = float(indices.numel())
        else:
            self.recall_bias.add_(self.recall_beta * recall_delta)
            updates = float(active.sum().item())
        self.recall_bias.clamp_(-5.0, 5.0)
        return {
            "updates": updates,
            "bias_mean": float(self.recall_bias.mean().item()),
            "bias_abs_mean": float(self.recall_bias.abs().mean().item()),
            "bias_norm": float(self.recall_bias.norm().item()),
        }


class MemorySteeringMLP(nn.Module):
    """Delta-memory-style global steering state for hidden representation repair."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        hidden_dim: int,
        write_beta: float,
        write_lambda: float,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.write_beta = write_beta
        self.write_lambda = write_lambda
        self.q_proj = nn.Linear(context_dim, rank)
        self.k_proj = nn.Linear(context_dim, rank)
        self.v_proj = nn.Linear(context_dim, rank)
        self.beta_proj = nn.Linear(context_dim, rank)
        self.target_proj = nn.Linear(hidden_dim, rank)
        self.steer_head = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.steer_head[-1].weight)
        nn.init.zeros_(self.steer_head[-1].bias)
        self.register_buffer("state", torch.zeros(rank, rank))

    def reset_state(self) -> None:
        self.state.zero_()

    def project_qkv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = l2_normalize(torch.tanh(self.q_proj(context)))
        k = l2_normalize(torch.tanh(self.k_proj(context)))
        v = self.v_proj(context)
        beta = torch.sigmoid(self.beta_proj(context))
        return q, k, v, beta

    def read(self, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        q, _, _, _ = self.project_qkv(context)
        read_vec = q @ self.state.t()
        delta_h = self.steer_head(read_vec).squeeze(0)
        stats = {
            "steer_delta_norm": float(delta_h.detach().float().norm().item()),
            "steer_state_norm": float(self.state.detach().float().norm().item()),
        }
        return delta_h, stats

    @torch.no_grad()
    def write(self, context: torch.Tensor, target_delta: torch.Tensor) -> dict[str, float]:
        if context.numel() == 0 or target_delta.numel() == 0:
            return {"target_norm": 0.0, "residual_mean": 0.0, "state_norm": float(self.state.norm().item())}
        _, k, v, beta = self.project_qkv(context.detach())
        target_value = self.target_proj(target_delta.detach().float().unsqueeze(0))
        pred_value = k @ self.state.t()
        residual = target_value - pred_value
        write_value = v + residual.tanh()
        delta_state = torch.einsum("br,bk->rk", write_value, k) / max(1, context.size(0))
        beta_vec = beta.mean(dim=0).unsqueeze(-1)
        self.state.mul_(self.write_lambda).add_(self.write_beta * beta_vec * delta_state)
        self.state.clamp_(-5.0, 5.0)
        return {
            "target_norm": float(target_delta.detach().float().norm().item()),
            "residual_mean": float(residual.abs().mean().item()),
            "beta_mean": float(beta.mean().item()),
            "state_norm": float(self.state.norm().item()),
        }


class OUGPGCN(nn.Module):
    """Masked GNN backbones with online graph and channel pruning memories."""

    def __init__(self, cfg: OUGPConfig, edge_index: torch.Tensor, x: torch.Tensor):
        super().__init__()
        non_loop_mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, non_loop_mask]
        if int(cfg.num_edges) != int(edge_index.size(1)):
            cfg = replace(cfg, num_edges=int(edge_index.size(1)))
        self.cfg = cfg
        if cfg.memory_write_mode not in {"residual", "feature", "none"}:
            raise ValueError("cfg.memory_write_mode must be one of: residual, feature, none.")
        if cfg.graph_memory_granularity not in {"edge", "subgraph"}:
            raise ValueError("cfg.graph_memory_granularity must be one of: edge, subgraph.")
        if cfg.graph_memory_layout not in {"single", "multi"}:
            raise ValueError("cfg.graph_memory_layout must be one of: single, multi.")
        if cfg.param_memory_layout not in {"single", "multi"}:
            raise ValueError("cfg.param_memory_layout must be one of: single, multi.")
        if cfg.param_pruning_granularity not in {"channel", "weight"}:
            raise ValueError("cfg.param_pruning_granularity must be one of: channel, weight.")
        if cfg.param_policy_granularity not in {"auto", "channel", "weight"}:
            raise ValueError("cfg.param_policy_granularity must be one of: auto, channel, weight.")
        if cfg.param_pruning_granularity == "channel" and cfg.param_policy_granularity == "weight":
            raise ValueError("weight-level policy requires weight-entry pruning granularity.")
        if cfg.graph_score_init not in {"constant", "random", "degree", "similarity", "topofeat"}:
            raise ValueError("cfg.graph_score_init must be one of: constant, random, degree, similarity, topofeat.")
        if cfg.param_score_init not in {"constant", "random", "magnitude"}:
            raise ValueError("cfg.param_score_init must be one of: constant, random, magnitude.")
        self.register_buffer("base_edge_index", edge_index)
        self.register_buffer("x_ref", x)
        self.lin1 = nn.Linear(cfg.in_dim, cfg.hidden_dim, bias=False)
        self.lin2 = nn.Linear(cfg.hidden_dim, cfg.out_dim, bias=False)
        if cfg.backbone not in {"gcn", "sage", "gat", "deepgcn"}:
            raise ValueError("cfg.backbone must be one of: 'gcn', 'sage', 'gat', 'deepgcn'.")
        if cfg.backbone in {"sage", "gat"} and cfg.num_gnn_layers != 2:
            raise ValueError("cfg.num_gnn_layers is currently only configurable for the 'gcn' and 'deepgcn' backbones.")
        if cfg.backbone == "gcn" and cfg.num_gnn_layers < 2:
            raise ValueError("gcn requires cfg.num_gnn_layers >= 2.")
        if cfg.backbone == "deepgcn" and cfg.num_gnn_layers < 3:
            raise ValueError("deepgcn requires cfg.num_gnn_layers >= 3.")
        self.deep_hidden_lins = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False) for _ in range(max(0, cfg.num_gnn_layers - 2))]
        )
        if cfg.param_pruning_granularity == "weight" and cfg.backbone != "gcn":
            raise ValueError("element-wise weight pruning currently supports the GCN backbone only.")
        if cfg.backbone == "sage":
            self.sage_lin1_neigh = nn.Linear(cfg.in_dim, cfg.hidden_dim, bias=False)
            self.sage_lin2_neigh = nn.Linear(cfg.hidden_dim, cfg.out_dim, bias=False)
        else:
            self.sage_lin1_neigh = None
            self.sage_lin2_neigh = None
        if cfg.backbone == "gat":
            self.gat_attn1_src = nn.Parameter(torch.empty(cfg.hidden_dim))
            self.gat_attn1_dst = nn.Parameter(torch.empty(cfg.hidden_dim))
            self.gat_attn2_src = nn.Parameter(torch.empty(cfg.out_dim))
            self.gat_attn2_dst = nn.Parameter(torch.empty(cfg.out_dim))
            nn.init.xavier_uniform_(self.gat_attn1_src.unsqueeze(0))
            nn.init.xavier_uniform_(self.gat_attn1_dst.unsqueeze(0))
            nn.init.xavier_uniform_(self.gat_attn2_src.unsqueeze(0))
            nn.init.xavier_uniform_(self.gat_attn2_dst.unsqueeze(0))
        else:
            self.gat_attn1_src = None
            self.gat_attn1_dst = None
            self.gat_attn2_src = None
            self.gat_attn2_dst = None
        edge_init = self.initial_edge_scores(cfg.graph_score_init)
        param_init = self.initial_param_scores(cfg.param_score_init)
        self.edge_logits = nn.Parameter(edge_init, requires_grad=not cfg.freeze_pruning_scores)
        self.param_logits = nn.Parameter(param_init, requires_grad=not cfg.freeze_pruning_scores)
        self.graph_branch_logits = nn.Parameter(torch.zeros(4), requires_grad=cfg.use_graph_branch_gates)
        self.register_buffer("graph_logit_scale_ema", torch.tensor(float(cfg.graph_score_scale_min)))
        self.register_buffer("param_logit_scale_ema", torch.tensor(float(cfg.param_score_scale_min)))

        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        rand = torch.randn(cfg.in_dim, cfg.feature_context_dim, generator=generator) / cfg.in_dim**0.5
        self.register_buffer("feature_proj", rand)

        self.graph_memory = OnlinePruningMemory(
            cfg.edge_context_dim,
            cfg.memory_rank,
            cfg.write_beta,
            cfg.write_lambda,
            event_items=cfg.num_edges,
            event_beta=cfg.event_beta,
            event_decay=cfg.event_decay,
            recall_items=cfg.num_edges,
            recall_beta=cfg.recall_beta,
            recall_decay=cfg.recall_decay,
        )
        if cfg.graph_memory_layout == "multi":
            self.graph_memory_topo = OnlinePruningMemory(
                cfg.edge_context_dim,
                cfg.memory_rank,
                cfg.write_beta,
                cfg.write_lambda,
            )
            self.graph_memory_feat = OnlinePruningMemory(
                cfg.edge_context_dim,
                cfg.memory_rank,
                cfg.write_beta,
                cfg.write_lambda,
            )
            if cfg.use_graph_grad_branch:
                self.graph_memory_grad = OnlinePruningMemory(
                    cfg.edge_context_dim,
                    cfg.memory_rank,
                    cfg.write_beta,
                    cfg.write_lambda,
                )
            else:
                self.graph_memory_grad = None
        else:
            self.graph_memory_topo = None
            self.graph_memory_feat = None
            self.graph_memory_grad = None
        if self.uses_channel_parameter_policy:
            self.param_memory = ChannelPruningMemory(
                cfg.param_context_dim,
                cfg.memory_rank,
                cfg.hidden_dim,
                cfg.write_beta,
                cfg.write_lambda,
                recall_beta=cfg.recall_beta,
                recall_decay=cfg.recall_decay,
            )
        else:
            self.param_memory = OnlinePruningMemory(
                cfg.param_context_dim,
                cfg.memory_rank,
                cfg.write_beta,
                cfg.write_lambda,
                recall_items=self.param_item_count,
                recall_beta=cfg.recall_beta,
                recall_decay=cfg.recall_decay,
            )
        if cfg.param_memory_layout == "multi":
            self.param_memory_layer = OnlinePruningMemory(
                cfg.param_context_dim,
                cfg.memory_rank,
                cfg.write_beta,
                cfg.write_lambda,
            )
        else:
            self.param_memory_layer = None
        self.steering_memory = MemorySteeringMLP(
            cfg.steer_context_dim,
            cfg.memory_rank,
            cfg.hidden_dim,
            cfg.steer_beta,
            cfg.steer_lambda,
        )

        self.last_graph_score: torch.Tensor | None = None
        self.last_param_score: torch.Tensor | None = None
        self.last_graph_mask: torch.Tensor | None = None
        self.last_param_mask: torch.Tensor | None = None
        self.last_graph_utility: torch.Tensor | None = None
        self.last_param_utility: torch.Tensor | None = None
        self.prev_graph_mask: torch.Tensor | None = None
        self.prev_param_mask: torch.Tensor | None = None
        self.trace_prev_graph_mask: torch.Tensor | None = None
        self.event_prev_graph_mask: torch.Tensor | None = None
        self.recall_prev_graph_mask: torch.Tensor | None = None
        self.recall_prev_param_mask: torch.Tensor | None = None
        self.last_steering_context: torch.Tensor | None = None
        self.last_steered_hidden: torch.Tensor | None = None
        self.last_hidden_states: list[dict[str, object]] | None = None
        self.training_hard_graph_mask: torch.Tensor | None = None
        self.training_hard_param_mask: torch.Tensor | None = None

    def normalized_init_scores(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float()
        std = scores.std(unbiased=False)
        if float(std.item()) <= 1e-8:
            return torch.full_like(scores, 2.0)
        return (scores - scores.mean()) / std.clamp_min(1e-8)

    def edge_similarity_scores(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """Compute edge cosine scores without materializing an ``[E, F]`` tensor."""

        x_norm = l2_normalize(self.x_ref.detach().float())
        scores = torch.empty(row.numel(), device=x_norm.device, dtype=x_norm.dtype)
        chunk_size = 4_096
        for start in range(0, row.numel(), chunk_size):
            end = min(start + chunk_size, row.numel())
            scores[start:end] = (x_norm[row[start:end]] * x_norm[col[start:end]]).sum(dim=-1)
        return scores

    def initial_edge_scores(self, mode: str) -> torch.Tensor:
        if mode == "constant":
            return torch.full((self.cfg.num_edges,), 2.0)
        generator = torch.Generator(device=self.base_edge_index.device)
        generator.manual_seed(self.cfg.seed)
        if mode == "random":
            scores = torch.randn(self.cfg.num_edges, generator=generator, device=self.base_edge_index.device)
        elif mode == "degree":
            row, col = self.base_edge_index
            degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(self.base_edge_index.device)
            scores = degree[row] + degree[col]
        elif mode == "similarity":
            row, col = self.base_edge_index
            scores = self.edge_similarity_scores(row, col)
        elif mode == "topofeat":
            row, col = self.base_edge_index
            degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(self.base_edge_index.device)
            degree_scores = self.normalized_init_scores(degree[row] + degree[col])
            similarity_scores = self.normalized_init_scores(self.edge_similarity_scores(row, col))
            scores = 0.5 * degree_scores + 0.5 * similarity_scores
        else:
            raise ValueError(f"Unknown graph score init mode {mode!r}.")
        return self.normalized_init_scores(scores)

    def initial_param_scores(self, mode: str) -> torch.Tensor:
        param_count = self.param_item_count
        if mode == "constant":
            return torch.full((param_count,), 2.0, device=self.lin1.weight.device)
        generator = torch.Generator(device=self.lin1.weight.device)
        generator.manual_seed(self.cfg.seed + 17)
        if mode == "random":
            scores = torch.randn(param_count, generator=generator, device=self.lin1.weight.device)
        elif mode == "magnitude":
            if self.cfg.param_pruning_granularity == "weight":
                scores = torch.cat(
                    [weight.detach().abs().flatten() for weight in self.parameter_weight_tensors()]
                )
            else:
                w1_norm = self.lin1.weight.detach().t().norm(dim=0)
                w2_norm = self.lin2.weight.detach().norm(dim=0)
                if self.cfg.backbone == "deepgcn" and len(self.deep_hidden_lins) > 0:
                    hidden_norms = [layer.weight.detach().norm(dim=0) for layer in self.deep_hidden_lins]
                    w2_norm = torch.stack([w2_norm, *hidden_norms], dim=0).mean(dim=0)
                scores = w1_norm + w2_norm
        else:
            raise ValueError(f"Unknown parameter score init mode {mode!r}.")
        return self.normalized_init_scores(scores)

    def parameter_weight_tensors(self) -> list[torch.Tensor]:
        """Return prunable GCN weights in their stable global-mask order."""

        return [self.lin1.weight, *(layer.weight for layer in self.deep_hidden_lins), self.lin2.weight]

    @property
    def param_item_count(self) -> int:
        if self.cfg.param_pruning_granularity == "weight":
            return sum(weight.numel() for weight in self.parameter_weight_tensors())
        return self.cfg.hidden_dim

    @property
    def resolved_param_policy_granularity(self) -> str:
        if self.cfg.param_policy_granularity == "auto":
            return self.cfg.param_pruning_granularity
        return self.cfg.param_policy_granularity

    @property
    def uses_channel_parameter_policy(self) -> bool:
        return self.resolved_param_policy_granularity == "channel"

    @property
    def param_policy_item_count(self) -> int:
        return self.cfg.hidden_dim if self.uses_channel_parameter_policy else self.param_item_count

    def set_training_hard_masks(
        self, graph_mask: torch.Tensor, param_mask: torch.Tensor
    ) -> None:
        if graph_mask.numel() != self.cfg.num_edges:
            raise ValueError("training graph hard mask must match num_edges.")
        if param_mask.numel() != self.param_item_count:
            raise ValueError("training parameter hard mask must match param_item_count.")
        self.training_hard_graph_mask = graph_mask.detach().to(
            device=self.edge_logits.device, dtype=self.edge_logits.dtype
        ).clone()
        self.training_hard_param_mask = param_mask.detach().to(
            device=self.param_logits.device, dtype=self.param_logits.dtype
        ).clone()

    def clear_training_hard_masks(self) -> None:
        self.training_hard_graph_mask = None
        self.training_hard_param_mask = None

    def get_training_hard_masks(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.training_hard_graph_mask is None or self.training_hard_param_mask is None:
            return None
        return self.training_hard_graph_mask, self.training_hard_param_mask

    def split_parameter_mask(self, param_mask: torch.Tensor) -> list[torch.Tensor]:
        """Reshape a global flat element mask back to the original weight matrices."""

        if self.cfg.param_pruning_granularity != "weight":
            raise ValueError("split_parameter_mask requires weight pruning granularity.")
        flat = param_mask.flatten()
        if flat.numel() != self.param_item_count:
            raise ValueError(
                f"parameter mask has {flat.numel()} entries; expected {self.param_item_count}."
            )
        masks: list[torch.Tensor] = []
        offset = 0
        for weight in self.parameter_weight_tensors():
            count = weight.numel()
            masks.append(flat[offset : offset + count].reshape_as(weight))
            offset += count
        return masks

    def aggregate_weight_entries_to_channels(self, values: torch.Tensor) -> torch.Tensor:
        """Aggregate W1 rows and W2 columns into one utility per hidden channel."""

        masks = self.split_parameter_mask(values)
        contributions = [masks[0].mean(dim=1)]
        for hidden_mask in masks[1:-1]:
            contributions.append(0.5 * (hidden_mask.mean(dim=1) + hidden_mask.mean(dim=0)))
        contributions.append(masks[-1].mean(dim=0))
        return torch.stack(contributions, dim=0).mean(dim=0)

    def expand_channel_utility_to_weight_entries(self, utility: torch.Tensor) -> torch.Tensor:
        """Broadcast hidden-level LHCU evidence to each incident weight entry."""

        if utility.numel() != self.cfg.hidden_dim:
            raise ValueError("hidden utility must match hidden_dim.")
        expanded = [utility[:, None].expand_as(self.lin1.weight)]
        for layer in self.deep_hidden_lins:
            expanded.append(0.5 * (utility[:, None] + utility[None, :]).expand_as(layer.weight))
        expanded.append(utility[None, :].expand_as(self.lin2.weight))
        return torch.cat([item.reshape(-1) for item in expanded])

    def reset_memory(self) -> None:
        self.graph_memory.reset_state()
        if self.graph_memory_topo is not None:
            self.graph_memory_topo.reset_state()
        if self.graph_memory_feat is not None:
            self.graph_memory_feat.reset_state()
        if self.graph_memory_grad is not None:
            self.graph_memory_grad.reset_state()
        self.param_memory.reset_state()
        if self.param_memory_layer is not None:
            self.param_memory_layer.reset_state()
        self.steering_memory.reset_state()
        self.graph_logit_scale_ema.fill_(float(self.cfg.graph_score_scale_min))
        self.param_logit_scale_ema.fill_(float(self.cfg.param_score_scale_min))

    def dense_edge_count(self) -> torch.Tensor:
        edge_count = self.edge_logits.new_tensor(float(self.cfg.num_edges))
        if self.cfg.backbone in {"gcn", "gat", "deepgcn"}:
            edge_count = edge_count + float(self.cfg.num_nodes)
        return edge_count

    def effective_edge_count(self) -> torch.Tensor:
        if self.last_graph_mask is None:
            kept = self.edge_logits.new_tensor(float(self.cfg.num_edges))
        else:
            kept = self.last_graph_mask.sum()
        if self.cfg.backbone in {"gcn", "gat", "deepgcn"}:
            kept = kept + float(self.cfg.num_nodes)
        return kept

    def dense_parameter_count(self) -> torch.Tensor:
        base = self.edge_logits.new_tensor(float(self.cfg.in_dim * self.cfg.hidden_dim + self.cfg.hidden_dim * self.cfg.out_dim))
        if self.cfg.backbone == "sage":
            base = base * 2.0
        elif self.cfg.backbone == "gat":
            base = base + float(2 * self.cfg.hidden_dim + 2 * self.cfg.out_dim)
        elif self.cfg.backbone == "deepgcn":
            base = base + float(max(0, self.cfg.num_gnn_layers - 2) * self.cfg.hidden_dim * self.cfg.hidden_dim)
        return base

    def effective_parameter_count(self) -> torch.Tensor:
        if self.cfg.param_pruning_granularity == "weight":
            if self.last_param_mask is None:
                return self.edge_logits.new_tensor(float(self.param_item_count))
            return self.last_param_mask.sum()
        if self.last_param_mask is None:
            hidden_keep = self.edge_logits.new_tensor(float(self.cfg.hidden_dim))
        else:
            hidden_keep = self.last_param_mask.sum()
        base = self.edge_logits.new_tensor(float(self.cfg.in_dim)) * hidden_keep
        base = base + hidden_keep * float(self.cfg.out_dim)
        if self.cfg.backbone == "sage":
            base = base * 2.0
        elif self.cfg.backbone == "gat":
            base = base + 2.0 * hidden_keep + float(2 * self.cfg.out_dim)
        elif self.cfg.backbone == "deepgcn":
            base = base + float(max(0, self.cfg.num_gnn_layers - 2)) * hidden_keep * hidden_keep
        return base

    def dense_message_cost(self) -> torch.Tensor:
        dense_edges = self.dense_edge_count()
        if self.cfg.backbone in {"gcn", "deepgcn"}:
            dims = float(self.cfg.in_dim + (self.cfg.num_gnn_layers - 1) * self.cfg.hidden_dim)
        elif self.cfg.backbone == "sage":
            dims = float(self.cfg.in_dim + self.cfg.hidden_dim)
        else:
            dims = float(3 * self.cfg.hidden_dim + 3 * self.cfg.out_dim)
        return dense_edges * dims

    def effective_message_cost(self) -> torch.Tensor:
        effective_edges = self.effective_edge_count()
        if self.cfg.param_pruning_granularity == "weight" or self.last_param_mask is None:
            hidden_keep = self.edge_logits.new_tensor(float(self.cfg.hidden_dim))
        else:
            hidden_keep = self.last_param_mask.sum()
        if self.cfg.backbone == "gcn":
            dims = self.edge_logits.new_tensor(float(self.cfg.in_dim)) + hidden_keep
        elif self.cfg.backbone == "deepgcn":
            dims = self.edge_logits.new_tensor(float(self.cfg.in_dim)) + float(self.cfg.num_gnn_layers - 1) * hidden_keep
        elif self.cfg.backbone == "sage":
            dims = self.edge_logits.new_tensor(float(self.cfg.in_dim)) + hidden_keep
        else:
            dims = 3.0 * hidden_keep + float(3 * self.cfg.out_dim)
        return effective_edges * dims

    def memory_state_items(self) -> torch.Tensor:
        graph_items = float(self.cfg.memory_rank * self.cfg.memory_rank)
        if self.cfg.graph_memory_layout == "multi":
            graph_items = float(2 + int(self.cfg.use_graph_full_branch) + int(self.cfg.use_graph_grad_branch)) * float(
                self.cfg.memory_rank * self.cfg.memory_rank
            )
        if self.uses_channel_parameter_policy:
            param_items = float(self.cfg.hidden_dim * self.cfg.memory_rank * self.cfg.memory_rank)
        else:
            param_items = float(self.cfg.memory_rank * self.cfg.memory_rank)
        if self.cfg.param_memory_layout == "multi":
            param_items += float(self.cfg.memory_rank * self.cfg.memory_rank)
        recall_items = float(self.cfg.num_edges + self.param_policy_item_count)
        steering_items = float(self.cfg.memory_rank * self.cfg.memory_rank)
        return self.edge_logits.new_tensor(graph_items + param_items + recall_items + steering_items)

    def resource_regularization(self) -> torch.Tensor:
        message_ratio = self.effective_message_cost() / self.dense_message_cost().clamp_min(1.0)
        param_ratio = self.effective_parameter_count() / self.dense_parameter_count().clamp_min(1.0)
        target = self.edge_logits.new_tensor(float(self.cfg.budget_target))
        return 0.5 * ((message_ratio - target).abs() + (param_ratio - target).abs())

    @torch.no_grad()
    def resource_stats(self) -> dict[str, float]:
        dense_message = self.dense_message_cost().detach().float()
        effective_message = self.effective_message_cost().detach().float()
        dense_params = self.dense_parameter_count().detach().float()
        effective_params = self.effective_parameter_count().detach().float()
        memory_items = self.memory_state_items().detach().float()
        return {
            "dense_message_cost": float(dense_message.item()),
            "effective_message_cost": float(effective_message.item()),
            "message_cost_ratio": float((effective_message / dense_message.clamp_min(1.0)).item()),
            "message_cost_reduction": float((1.0 - effective_message / dense_message.clamp_min(1.0)).item()),
            "dense_parameter_count": float(dense_params.item()),
            "effective_parameter_count": float(effective_params.item()),
            "parameter_cost_ratio": float((effective_params / dense_params.clamp_min(1.0)).item()),
            "parameter_cost_reduction": float((1.0 - effective_params / dense_params.clamp_min(1.0)).item()),
            "memory_state_items": float(memory_items.item()),
            "memory_overhead_vs_dense_params": float((memory_items / dense_params.clamp_min(1.0)).item()),
        }

    def edge_context(self, param_keep: torch.Tensor) -> torch.Tensor:
        row, col = self.base_edge_index
        x_proj = self.x_ref @ self.feature_proj
        degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(x_proj.device)
        degree = torch.log1p(degree / degree.mean().clamp_min(1.0)).unsqueeze(-1)
        param_ctx = param_keep.expand(row.numel(), 1)
        return torch.cat([x_proj[row], x_proj[col], degree[row], degree[col], param_ctx], dim=-1)

    def graph_branch_contexts(self, graph_ctx: torch.Tensor, graph_signal: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if graph_ctx.size(-1) != self.cfg.edge_context_dim:
            raise ValueError("graph_ctx last dimension must match cfg.edge_context_dim.")
        feat_dim = self.cfg.feature_context_dim
        src_feat = graph_ctx[:, :feat_dim]
        dst_feat = graph_ctx[:, feat_dim : 2 * feat_dim]
        src_deg = graph_ctx[:, 2 * feat_dim : 2 * feat_dim + 1]
        dst_deg = graph_ctx[:, 2 * feat_dim + 1 : 2 * feat_dim + 2]
        param_ctx = graph_ctx[:, -1:]
        if graph_signal is None:
            graph_signal = self.edge_logits.detach().float()
        zeros_feat = torch.zeros_like(src_feat)
        zeros_deg = torch.zeros_like(src_deg)
        branches = {
            "full": graph_ctx,
            "topo": torch.cat([zeros_feat, zeros_feat, src_deg, dst_deg, param_ctx], dim=-1),
            "feat": torch.cat([src_feat, dst_feat, zeros_deg, zeros_deg, param_ctx], dim=-1),
        }
        if self.cfg.use_graph_grad_branch:
            grad_signal = graph_signal.detach().float().to(graph_ctx.device, graph_ctx.dtype).unsqueeze(-1)
            grad_signal = torch.tanh(grad_signal)
            branches["grad"] = torch.cat([zeros_feat, zeros_feat, src_deg, dst_deg, grad_signal], dim=-1)
        return branches

    def param_context(self, graph_keep: torch.Tensor) -> torch.Tensor:
        if self.cfg.param_pruning_granularity == "weight" and not self.uses_channel_parameter_policy:
            contexts: list[torch.Tensor] = []
            logits = self.param_logits.detach()
            offset = 0
            weights = self.parameter_weight_tensors()
            for layer_index, weight in enumerate(weights):
                rows, cols = weight.shape
                count = weight.numel()
                magnitude = weight.detach().abs().flatten()
                magnitude = magnitude / magnitude.mean().clamp_min(1e-8)
                row_id = torch.arange(rows, device=weight.device, dtype=weight.dtype)
                col_id = torch.arange(cols, device=weight.device, dtype=weight.dtype)
                row_id = (row_id / max(1, rows - 1))[:, None].expand(rows, cols).reshape(-1)
                col_id = (col_id / max(1, cols - 1))[None, :].expand(rows, cols).reshape(-1)
                soft_score = torch.sigmoid(logits[offset : offset + count]).to(weight.dtype)
                graph_ctx = graph_keep.to(weight.dtype).expand(count)
                layer_id = weight.new_full((count,), float(layer_index) / max(1, len(weights) - 1))
                contexts.append(
                    torch.stack(
                        [magnitude, row_id, col_id, soft_score, graph_ctx, layer_id], dim=-1
                    )
                )
                offset += count
            return torch.cat(contexts, dim=0)
        w1_norm = self.lin1.weight.t().norm(dim=0, keepdim=True).t()
        w2_norm = self.lin2.weight.norm(dim=0, keepdim=True).t()
        if self.cfg.backbone == "sage":
            if self.sage_lin1_neigh is None or self.sage_lin2_neigh is None:
                raise RuntimeError("GraphSAGE layers were not initialized.")
            w1_norm = torch.stack(
                [w1_norm, self.sage_lin1_neigh.weight.t().norm(dim=0, keepdim=True).t()],
                dim=0,
            ).mean(dim=0)
            w2_norm = torch.stack(
                [w2_norm, self.sage_lin2_neigh.weight.norm(dim=0, keepdim=True).t()],
                dim=0,
            ).mean(dim=0)
        elif self.cfg.backbone == "gat":
            if self.gat_attn1_src is None or self.gat_attn1_dst is None:
                raise RuntimeError("GAT attention parameters were not initialized.")
            attn_norm = torch.stack([self.gat_attn1_src.abs(), self.gat_attn1_dst.abs()], dim=0).mean(dim=0).unsqueeze(-1)
            w1_norm = 0.5 * (w1_norm + attn_norm)
        elif self.cfg.backbone == "deepgcn" and len(self.deep_hidden_lins) > 0:
            hidden_norms = [layer.weight.norm(dim=0, keepdim=True).t() for layer in self.deep_hidden_lins]
            w2_norm = torch.stack([w2_norm, *hidden_norms], dim=0).mean(dim=0)
        channel_id = torch.linspace(0, 1, self.cfg.hidden_dim, device=w1_norm.device).unsqueeze(-1)
        if self.cfg.param_pruning_granularity == "weight":
            channel_soft_score = self.aggregate_weight_entries_to_channels(
                torch.sigmoid(self.param_logits).detach()
            )
        else:
            channel_soft_score = torch.sigmoid(self.param_logits).detach()
        sparsity = channel_soft_score.unsqueeze(-1)
        graph_ctx = graph_keep.expand(self.cfg.hidden_dim, 1)
        bias = torch.ones_like(graph_ctx)
        return torch.cat([w1_norm, w2_norm, channel_id, sparsity, graph_ctx, bias], dim=-1)

    def normalized_param_signal(self, signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.detach().float().mean().to(signal.dtype)
        std = centered.detach().float().std(unbiased=False)
        if float(std.item()) <= 1e-6:
            return torch.zeros_like(signal)
        normalized = centered / std.to(signal.dtype)
        return normalized.clamp(-self.cfg.param_correction_clip, self.cfg.param_correction_clip)

    def normalized_graph_signal(self, signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.detach().float().mean().to(signal.dtype)
        std = centered.detach().float().std(unbiased=False)
        if float(std.item()) <= 1e-6:
            return torch.zeros_like(signal)
        normalized = centered / std.to(signal.dtype)
        return normalized.clamp(-self.cfg.graph_correction_clip, self.cfg.graph_correction_clip)

    def graph_score_scale(self) -> torch.Tensor:
        current = self.edge_logits.detach().float().std(unbiased=False)
        current = current.clamp(self.cfg.graph_score_scale_min, self.cfg.graph_score_scale_max)
        if self.training:
            decay = float(self.cfg.graph_score_scale_decay)
            self.graph_logit_scale_ema.mul_(decay).add_((1.0 - decay) * current.to(self.graph_logit_scale_ema.device))
        return self.graph_logit_scale_ema.to(device=self.edge_logits.device, dtype=self.edge_logits.dtype)

    def param_score_scale(self) -> torch.Tensor:
        current = self.param_logits.detach().float().std(unbiased=False)
        current = current.clamp(self.cfg.param_score_scale_min, self.cfg.param_score_scale_max)
        if self.training:
            decay = float(self.cfg.param_score_scale_decay)
            self.param_logit_scale_ema.mul_(decay).add_((1.0 - decay) * current.to(self.param_logit_scale_ema.device))
        return self.param_logit_scale_ema.to(device=self.param_logits.device, dtype=self.param_logits.dtype)

    def masks(self, temperature: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        base_graph_keep = torch.sigmoid(self.edge_logits).mean().detach()
        base_param_keep = torch.sigmoid(self.param_logits).mean().detach()
        graph_scale = self.graph_score_scale()
        param_scale = self.param_score_scale()
        policy_param_score = (
            self.aggregate_weight_entries_to_channels(self.param_logits)
            if self.cfg.param_pruning_granularity == "weight" and self.uses_channel_parameter_policy
            else self.param_logits
        )

        if self.cfg.use_memory:
            graph_cross_ctx = base_param_keep if self.cfg.use_cross else self.edge_logits.new_tensor(self.cfg.param_target_keep)
            param_cross_ctx = base_graph_keep if self.cfg.use_cross else self.edge_logits.new_tensor(self.cfg.graph_target_keep)
            graph_ctx = self.edge_context(graph_cross_ctx)
            param_ctx = self.param_context(param_cross_ctx)
            raw_graph_corr, graph_branch_stats = self.graph_correction(graph_ctx)
            unit_graph_corr = self.normalized_graph_signal(raw_graph_corr)
            graph_corr = graph_scale * unit_graph_corr
            raw_policy_param_corr, param_branch_stats = self.param_correction(param_ctx)
            unit_policy_param_corr = self.normalized_param_signal(raw_policy_param_corr)
            policy_param_corr = param_scale * unit_policy_param_corr
        else:
            raw_graph_corr = torch.zeros_like(self.edge_logits)
            graph_corr = torch.zeros_like(self.edge_logits)
            graph_branch_stats = {}
            unit_graph_corr = torch.zeros_like(self.edge_logits)
            raw_policy_param_corr = torch.zeros_like(policy_param_score)
            param_branch_stats = {}
            unit_policy_param_corr = torch.zeros_like(policy_param_score)
            policy_param_corr = torch.zeros_like(policy_param_score)

        if self.cfg.param_pruning_granularity == "weight" and self.uses_channel_parameter_policy:
            raw_param_corr = self.expand_channel_utility_to_weight_entries(raw_policy_param_corr)
            unit_param_corr = self.expand_channel_utility_to_weight_entries(unit_policy_param_corr)
            param_corr = self.expand_channel_utility_to_weight_entries(policy_param_corr)
        else:
            raw_param_corr = raw_policy_param_corr
            unit_param_corr = unit_policy_param_corr
            param_corr = policy_param_corr

        graph_score = self.edge_logits
        param_score = self.param_logits
        raw_graph_event_corr = self.graph_memory.event_correction(graph_score)
        unit_graph_event_corr = self.normalized_graph_signal(raw_graph_event_corr)
        graph_event_corr = graph_scale * unit_graph_event_corr
        raw_graph_recall_corr = self.graph_memory.recall_correction(graph_score)
        unit_graph_recall_corr = self.normalized_graph_signal(raw_graph_recall_corr)
        graph_recall_corr = graph_scale * unit_graph_recall_corr
        raw_policy_param_recall_corr = self.param_memory.recall_correction(policy_param_score)
        unit_policy_param_recall_corr = self.normalized_param_signal(raw_policy_param_recall_corr)
        policy_param_recall_corr = param_scale * unit_policy_param_recall_corr
        if self.cfg.param_pruning_granularity == "weight" and self.uses_channel_parameter_policy:
            raw_param_recall_corr = self.expand_channel_utility_to_weight_entries(raw_policy_param_recall_corr)
            unit_param_recall_corr = self.expand_channel_utility_to_weight_entries(unit_policy_param_recall_corr)
            param_recall_corr = self.expand_channel_utility_to_weight_entries(policy_param_recall_corr)
        else:
            raw_param_recall_corr = raw_policy_param_recall_corr
            unit_param_recall_corr = unit_policy_param_recall_corr
            param_recall_corr = policy_param_recall_corr

        if self.cfg.use_memory:
            graph_score = graph_score + self.cfg.graph_gamma * graph_corr
            param_score = param_score + self.cfg.param_gamma * param_corr
            if self.cfg.use_graph_pruning and self.cfg.event_gamma != 0.0:
                graph_score = graph_score + self.cfg.event_gamma * graph_event_corr
            if self.cfg.use_graph_pruning and self.cfg.recall_gamma != 0.0:
                graph_score = graph_score + self.cfg.recall_gamma * graph_recall_corr
            if self.cfg.use_param_pruning and self.cfg.recall_gamma != 0.0:
                param_score = param_score + self.cfg.recall_gamma * param_recall_corr
        graph_keep_target = self.cfg.graph_target_keep
        param_keep_target = self.cfg.param_target_keep

        if self.cfg.use_graph_pruning and graph_keep_target < 0.999:
            graph_mask = budgeted_sigmoid(
                graph_score,
                graph_keep_target,
                temperature,
                self.cfg.hard_masks and not self.training,
            )
        else:
            graph_mask = torch.ones_like(graph_score)
        if self.cfg.use_param_pruning and param_keep_target < 0.999:
            param_mask = budgeted_sigmoid(
                param_score,
                param_keep_target,
                temperature,
                self.cfg.hard_masks and not self.training,
            )
        else:
            param_mask = torch.ones_like(param_score)

        active_hard_masks = self.get_training_hard_masks()
        if active_hard_masks is not None and self.training:
            hard_graph_mask, hard_param_mask = active_hard_masks
            graph_mask = hard_graph_mask.detach() - graph_mask.detach() + graph_mask
            param_mask = hard_param_mask.detach() - param_mask.detach() + param_mask

        self.last_graph_score = graph_score
        self.last_param_score = param_score
        self.last_graph_mask = graph_mask
        self.last_param_mask = param_mask
        stats = {
            "graph_keep": float(graph_mask.detach().mean().item()),
            "param_keep": float(param_mask.detach().mean().item()),
            "graph_logits_mean": float(self.edge_logits.detach().float().mean().item()),
            "graph_logits_std": float(self.edge_logits.detach().float().std(unbiased=False).item()),
            "graph_memory_correction_mean": float(graph_corr.detach().float().mean().item()),
            "graph_memory_correction_std": float(graph_corr.detach().float().std(unbiased=False).item()),
            "graph_policy_correction_norm": float(graph_corr.detach().float().norm().item()),
            "graph_memory_unit_correction_mean": float(unit_graph_corr.detach().float().mean().item()),
            "graph_memory_unit_correction_std": float(unit_graph_corr.detach().float().std(unbiased=False).item()),
            "graph_memory_raw_correction_mean": float(raw_graph_corr.detach().float().mean().item()),
            "graph_memory_raw_correction_std": float(raw_graph_corr.detach().float().std(unbiased=False).item()),
            "graph_event_correction_mean": float(graph_event_corr.detach().float().mean().item()),
            "graph_event_correction_std": float(graph_event_corr.detach().float().std(unbiased=False).item()),
            "graph_recall_correction_mean": float(graph_recall_corr.detach().float().mean().item()),
            "graph_recall_correction_std": float(graph_recall_corr.detach().float().std(unbiased=False).item()),
            "graph_score_scale": float(graph_scale.detach().float().item()),
            "graph_memory_score_delta_std": float((self.cfg.graph_gamma * graph_corr).detach().float().std(unbiased=False).item()),
            "graph_event_score_delta_std": float((self.cfg.event_gamma * graph_event_corr).detach().float().std(unbiased=False).item()),
            "graph_recall_score_delta_std": float((self.cfg.recall_gamma * graph_recall_corr).detach().float().std(unbiased=False).item()),
            "param_logits_mean": float(self.param_logits.detach().float().mean().item()),
            "param_logits_std": float(self.param_logits.detach().float().std(unbiased=False).item()),
            "param_memory_correction_mean": float(param_corr.detach().float().mean().item()),
            "param_memory_correction_std": float(param_corr.detach().float().std(unbiased=False).item()),
            "channel_policy_correction_norm": float(param_corr.detach().float().norm().item()),
            "param_memory_unit_correction_mean": float(unit_param_corr.detach().float().mean().item()),
            "param_memory_unit_correction_std": float(unit_param_corr.detach().float().std(unbiased=False).item()),
            "param_memory_raw_correction_mean": float(raw_param_corr.detach().float().mean().item()),
            "param_memory_raw_correction_std": float(raw_param_corr.detach().float().std(unbiased=False).item()),
            "recall_correction_mean": float(param_recall_corr.detach().float().mean().item()),
            "recall_correction_std": float(param_recall_corr.detach().float().std(unbiased=False).item()),
            "recall_unit_correction_mean": float(unit_param_recall_corr.detach().float().mean().item()),
            "recall_unit_correction_std": float(unit_param_recall_corr.detach().float().std(unbiased=False).item()),
            "param_score_scale": float(param_scale.detach().float().item()),
            "param_memory_score_delta_std": float((self.cfg.param_gamma * param_corr).detach().float().std(unbiased=False).item()),
            "recall_score_delta_std": float((self.cfg.recall_gamma * param_recall_corr).detach().float().std(unbiased=False).item()),
        }
        stats.update(graph_branch_stats)
        stats.update(param_branch_stats)
        return graph_mask, param_mask, stats

    def steering_context(
        self,
        hidden: torch.Tensor,
        graph_mask: torch.Tensor,
        param_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_detached = hidden.detach().float()
        graph_mask_detached = graph_mask.detach().float()
        param_mask_detached = param_mask.detach().float()
        values = [
            graph_mask_detached.mean(),
            param_mask_detached.mean(),
            hidden.new_tensor(self.cfg.graph_target_keep).float(),
            hidden.new_tensor(self.cfg.param_target_keep).float(),
            graph_mask_detached.std(unbiased=False),
            param_mask_detached.std(unbiased=False),
            hidden_detached.norm(dim=-1).mean(),
            hidden_detached.std(unbiased=False),
            self.graph_memory.state.detach().float().norm().to(hidden.device),
            self.param_memory.state.detach().float().norm().to(hidden.device),
        ]
        context = torch.stack([v.to(device=hidden.device, dtype=hidden.dtype) for v in values]).unsqueeze(0)
        if context.size(-1) != self.cfg.steer_context_dim:
            raise ValueError("steering_context size must match cfg.steer_context_dim.")
        return context

    def _record_hidden(
        self,
        hidden_states: list[dict[str, torch.Tensor | str | int]],
        layer: int,
        kind: str,
        value: torch.Tensor,
        retain_hidden_grad: bool,
    ) -> None:
        if retain_hidden_grad and value.requires_grad:
            value.retain_grad()
        hidden_states.append({"layer": layer, "kind": kind, "tensor": value})

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        return_hidden_states: bool = False,
        fixed_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
        retain_hidden_grad: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]] | tuple[torch.Tensor, dict[str, float], list[dict[str, torch.Tensor | str | int]]]:
        hidden_states: list[dict[str, torch.Tensor | str | int]] = []
        capture_hidden_states = return_hidden_states
        capture_hidden_grad = retain_hidden_grad
        if fixed_masks is None:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            mask_start = time.perf_counter()
            graph_mask, param_mask, stats = self.masks(temperature)
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            stats["mask_compute_sec"] = time.perf_counter() - mask_start
        else:
            graph_mask, param_mask = fixed_masks
            stats = {
                "graph_keep": float(graph_mask.detach().float().mean().item()),
                "param_keep": float(param_mask.detach().float().mean().item()),
                "mask_compute_sec": 0.0,
            }
        num_nodes = x.size(0)
        weight_masks = (
            self.split_parameter_mask(param_mask)
            if self.cfg.param_pruning_granularity == "weight"
            else []
        )
        if self.cfg.backbone in {"gcn", "deepgcn"}:
            edge_index, edge_weight = append_unit_self_loops(
                self.base_edge_index,
                graph_mask,
                num_nodes,
            )
            norm_weight = symmetric_norm(edge_index, edge_weight, num_nodes)
            h = sparse_gcn_mm(edge_index, norm_weight, x, num_nodes)
            if self.cfg.param_pruning_granularity == "weight":
                h = F.linear(h, self.lin1.weight * weight_masks[0])
            else:
                h = self.lin1(h)
            if capture_hidden_states:
                self._record_hidden(hidden_states, 0, "pre_activation", h, capture_hidden_grad)
        elif self.cfg.backbone == "sage":
            edge_index = self.base_edge_index
            edge_weight = graph_mask
            if self.sage_lin1_neigh is None or self.sage_lin2_neigh is None:
                raise RuntimeError("GraphSAGE layers were not initialized.")
            neigh = sparse_mean_mm(edge_index, edge_weight, x, num_nodes)
            h = self.lin1(x) + self.sage_lin1_neigh(neigh)
            if capture_hidden_states:
                self._record_hidden(hidden_states, 0, "pre_activation", h, capture_hidden_grad)
        elif self.cfg.backbone == "gat":
            edge_index, edge_weight = append_unit_self_loops(
                self.base_edge_index,
                graph_mask,
                num_nodes,
            )
            if self.gat_attn1_src is None or self.gat_attn1_dst is None:
                raise RuntimeError("GAT attention parameters were not initialized.")
            h_linear = self.lin1(x)
            h = sparse_gat_mm(edge_index, edge_weight, h_linear, self.gat_attn1_src, self.gat_attn1_dst, num_nodes)
            if capture_hidden_states:
                self._record_hidden(hidden_states, 0, "pre_activation", h, capture_hidden_grad)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone!r}.")

        self.last_steering_context = None
        self.last_steered_hidden = None
        if self.cfg.use_memory and self.cfg.use_steering_memory and self.cfg.steer_gamma != 0.0:
            steering_context = self.steering_context(h, graph_mask, param_mask)
            delta_h, steering_stats = self.steering_memory.read(steering_context)
            h = h + self.cfg.steer_gamma * delta_h.unsqueeze(0).to(dtype=h.dtype, device=h.device)
            self.last_steering_context = steering_context.detach()
            if self.training and h.requires_grad:
                h.retain_grad()
                self.last_steered_hidden = h
            stats.update({f"steering_{key}": value for key, value in steering_stats.items()})
        h = F.relu(h)
        if self.cfg.param_pruning_granularity == "channel":
            h = h * param_mask
        if capture_hidden_states:
            self._record_hidden(hidden_states, 0, "activation", h, capture_hidden_grad)
        h = F.dropout(h, p=0.5, training=self.training)
        if self.cfg.backbone == "gcn":
            for layer_idx, layer in enumerate(self.deep_hidden_lins, start=1):
                h = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
                if self.cfg.param_pruning_granularity == "weight":
                    h = F.linear(h, layer.weight * weight_masks[layer_idx])
                else:
                    h = layer(h)
                if capture_hidden_states:
                    self._record_hidden(hidden_states, layer_idx, "pre_activation", h, capture_hidden_grad)
                h = F.relu(h)
                if self.cfg.param_pruning_granularity == "channel":
                    h = h * param_mask
                if capture_hidden_states:
                    self._record_hidden(hidden_states, layer_idx, "activation", h, capture_hidden_grad)
                h = F.dropout(h, p=0.5, training=self.training)
            h = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
            if self.cfg.param_pruning_granularity == "weight":
                out = F.linear(h, self.lin2.weight * weight_masks[-1])
            else:
                out = self.lin2(h)
        elif self.cfg.backbone == "deepgcn":
            for layer in self.deep_hidden_lins:
                h_next = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
                h_next = layer(h_next)
                if capture_hidden_states:
                    layer_idx = len([item for item in hidden_states if item["kind"] == "pre_activation"])
                    self._record_hidden(hidden_states, layer_idx, "pre_activation", h_next, capture_hidden_grad)
                h = F.relu(h + h_next) * param_mask
                if capture_hidden_states:
                    self._record_hidden(hidden_states, layer_idx, "activation", h, capture_hidden_grad)
                h = F.dropout(h, p=0.5, training=self.training)
            h = sparse_gcn_mm(edge_index, norm_weight, h, num_nodes)
            out = self.lin2(h)
        elif self.cfg.backbone == "sage":
            neigh = sparse_mean_mm(edge_index, edge_weight, h, num_nodes)
            out = self.lin2(h) + self.sage_lin2_neigh(neigh)
        else:
            if self.gat_attn2_src is None or self.gat_attn2_dst is None:
                raise RuntimeError("GAT attention parameters were not initialized.")
            out_linear = self.lin2(h)
            out = sparse_gat_mm(edge_index, edge_weight, out_linear, self.gat_attn2_src, self.gat_attn2_dst, num_nodes)
        if capture_hidden_states:
            self._record_hidden(hidden_states, self.cfg.num_gnn_layers - 1, "logits", out, capture_hidden_grad)
        if capture_hidden_states:
            self.last_hidden_states = hidden_states
        else:
            self.last_hidden_states = None
        if return_hidden_states:
            return out, stats, hidden_states
        return out, stats

    def regularization(self) -> torch.Tensor:
        reg = self.edge_logits.new_tensor(0.0)
        if self.cfg.use_graph_pruning and self.last_graph_mask is not None:
            reg = reg + (self.last_graph_mask.mean() - self.cfg.graph_target_keep).abs()
        if self.cfg.use_param_pruning and self.last_param_mask is not None:
            reg = reg + (self.last_param_mask.mean() - self.cfg.param_target_keep).abs()
        return reg

    @torch.no_grad()
    def churn(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.last_graph_mask is not None:
            graph_mask = self.last_graph_mask.detach().float()
            if self.prev_graph_mask is None:
                out["graph_churn"] = 0.0
            else:
                out["graph_churn"] = float((graph_mask - self.prev_graph_mask).abs().mean().item())
            self.prev_graph_mask = graph_mask
        if self.last_param_mask is not None:
            param_mask = self.last_param_mask.detach().float()
            if self.prev_param_mask is None:
                out["param_churn"] = 0.0
            else:
                out["param_churn"] = float((param_mask - self.prev_param_mask).abs().mean().item())
            self.prev_param_mask = param_mask
        return out

    @torch.no_grad()
    def graph_event_tensors(self, graph_utility: torch.Tensor) -> dict[str, torch.Tensor]:
        row, col = self.base_edge_index
        graph_utility = graph_utility.detach().float().to(row.device)
        degree = torch.bincount(row, minlength=self.cfg.num_nodes).float().to(row.device)
        log_degree = torch.log1p(degree)
        feature_norm = self.x_ref.norm(dim=1).float().to(row.device)

        node_utility_sum = torch.zeros(self.cfg.num_nodes, device=row.device)
        node_utility_count = torch.zeros(self.cfg.num_nodes, device=row.device)
        node_utility_sum.scatter_add_(0, row, graph_utility)
        node_utility_sum.scatter_add_(0, col, graph_utility)
        ones = torch.ones_like(graph_utility)
        node_utility_count.scatter_add_(0, row, ones)
        node_utility_count.scatter_add_(0, col, ones)
        node_utility = node_utility_sum / node_utility_count.clamp_min(1.0)

        def minmax(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)

        degree_score = minmax(log_degree)
        feature_score = minmax(feature_norm)
        node_utility_score = minmax(node_utility)
        node_importance = (degree_score + feature_score + node_utility_score) / 3.0
        utility_score = minmax(graph_utility)
        edge_importance = 0.5 * utility_score + 0.25 * (node_importance[row] + node_importance[col])

        return {
            "row": row,
            "col": col,
            "degree": degree,
            "feature_norm": feature_norm,
            "graph_utility": graph_utility,
            "node_importance": node_importance,
            "edge_importance": edge_importance,
        }

    @torch.no_grad()
    def write_graph_event_memory(self, graph_utility: torch.Tensor) -> dict[str, float]:
        if self.last_graph_mask is None or not self.cfg.use_graph_pruning:
            return {}
        graph_mask = self.last_graph_mask.detach().float()
        if self.event_prev_graph_mask is None:
            self.event_prev_graph_mask = graph_mask.clone()
            return {
                "graph_event_updates": 0.0,
                "graph_event_bias_norm": float(self.graph_memory.event_bias.norm().item()),
            }

        mask_drop = (self.event_prev_graph_mask.to(graph_mask.device) - graph_mask).clamp_min(0.0)
        self.event_prev_graph_mask = graph_mask.clone()
        if not bool((mask_drop > 0).any().item()):
            return {
                "graph_event_updates": 0.0,
                "graph_event_bias_norm": float(self.graph_memory.event_bias.norm().item()),
            }

        event = self.graph_event_tensors(graph_utility)
        edge_importance = event["edge_importance"]
        centered_importance = edge_importance - edge_importance.mean()
        scaled_importance = centered_importance / centered_importance.std(unbiased=False).clamp_min(1e-6)
        event_delta = mask_drop * scaled_importance.clamp(-3.0, 3.0)
        event_stats = self.graph_memory.write_events(event_delta, self.cfg.event_top_k)
        return {f"graph_event_{key}": value for key, value in event_stats.items()}

    @torch.no_grad()
    def recall_delta_from_drop(self, previous_mask: torch.Tensor, current_mask: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
        mask_drop = (previous_mask.to(current_mask.device) - current_mask).clamp_min(0.0)
        if not bool((mask_drop > 0).any().item()):
            return torch.zeros_like(current_mask)
        utility = utility.detach().float().to(current_mask.device)
        scaled_utility = (utility - utility.mean()) / utility.std(unbiased=False).clamp_min(1e-6)
        important_after_drop = scaled_utility.clamp_min(0.0).clamp_max(3.0)
        return mask_drop * important_after_drop

    @torch.no_grad()
    def write_graph_recall_memory(self, graph_utility: torch.Tensor) -> dict[str, float]:
        if self.last_graph_mask is None or not self.cfg.use_graph_pruning:
            return {}
        graph_mask = self.last_graph_mask.detach().float()
        if self.recall_prev_graph_mask is None:
            self.recall_prev_graph_mask = graph_mask.clone()
            return {
                "graph_recall_updates": 0.0,
                "graph_recall_bias_norm": float(self.graph_memory.recall_bias.norm().item()),
            }
        recall_delta = self.recall_delta_from_drop(self.recall_prev_graph_mask, graph_mask, graph_utility)
        self.recall_prev_graph_mask = graph_mask.clone()
        recall_stats = self.graph_memory.write_recall(recall_delta, self.cfg.recall_top_k)
        return {f"graph_recall_{key}": value for key, value in recall_stats.items()}

    @torch.no_grad()
    def write_param_recall_memory(self, param_utility: torch.Tensor) -> dict[str, float]:
        if self.last_param_mask is None or not self.cfg.use_param_pruning:
            return {}
        param_mask = self.last_param_mask.detach().float()
        if self.cfg.param_pruning_granularity == "weight" and self.uses_channel_parameter_policy:
            param_mask = self.aggregate_weight_entries_to_channels(param_mask)
        if self.recall_prev_param_mask is None:
            self.recall_prev_param_mask = param_mask.clone()
            return {
                "param_recall_updates": 0.0,
                "param_recall_bias_norm": float(self.param_memory.recall_bias.norm().item()),
            }
        recall_delta = self.recall_delta_from_drop(self.recall_prev_param_mask, param_mask, param_utility)
        self.recall_prev_param_mask = param_mask.clone()
        recall_stats = self.param_memory.write_recall(recall_delta, self.cfg.recall_top_k)
        return {f"param_recall_{key}": value for key, value in recall_stats.items()}

    @torch.no_grad()
    def write_steering_memory(self) -> dict[str, float]:
        if (
            not self.cfg.use_steering_memory
            or self.last_steering_context is None
            or self.last_steered_hidden is None
            or self.last_steered_hidden.grad is None
        ):
            return {}
        grad = self.last_steered_hidden.grad.detach().float()
        target_delta = -grad.mean(dim=0)
        target_delta = target_delta / target_delta.norm().clamp_min(1e-6)
        steering_stats = self.steering_memory.write(self.last_steering_context, target_delta)
        return {f"steering_memory_{key}": value for key, value in steering_stats.items()}

    @torch.no_grad()
    def graph_trace_snapshot(
        self,
        dataset: str,
        variant: str,
        seed: int,
        epoch: int,
        top_k: int,
    ) -> list[dict[str, float | int | str]]:
        if self.last_graph_mask is None or self.last_graph_score is None or top_k <= 0:
            return []
        graph_mask = self.last_graph_mask.detach().float()
        if self.trace_prev_graph_mask is None:
            self.trace_prev_graph_mask = graph_mask.clone()
            return []

        mask_delta = (self.trace_prev_graph_mask.to(graph_mask.device) - graph_mask).clamp_min(0.0)
        self.trace_prev_graph_mask = graph_mask.clone()
        positive = mask_delta > 0
        if not bool(positive.any().item()):
            return []

        k = min(int(top_k), int(positive.sum().item()))
        top_values, edge_ids = torch.topk(mask_delta.masked_fill(~positive, -1.0), k=k)
        keep = top_values > 0
        if not bool(keep.any().item()):
            return []
        edge_ids = edge_ids[keep]

        graph_utility = (
            self.last_graph_utility.detach().float()
            if self.last_graph_utility is not None
            else torch.zeros_like(graph_mask)
        )
        event = self.graph_event_tensors(graph_utility)
        row = event["row"]
        col = event["col"]
        degree = event["degree"]
        feature_norm = event["feature_norm"]
        node_importance = event["node_importance"]
        edge_importance_tensor = event["edge_importance"]

        current_graph_keep = float(graph_mask.mean().item())
        current_param_keep = (
            float(self.last_param_mask.detach().float().mean().item())
            if self.last_param_mask is not None
            else float(self.cfg.param_target_keep)
        )

        rows = []
        for edge_id_tensor in edge_ids.detach().cpu():
            edge_id = int(edge_id_tensor.item())
            src = int(row[edge_id].item())
            dst = int(col[edge_id].item())
            src_importance = float(node_importance[src].item())
            dst_importance = float(node_importance[dst].item())
            edge_importance = float(edge_importance_tensor[edge_id].item())
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "edge_id": edge_id,
                    "src_node": src,
                    "dst_node": dst,
                    "prev_mask": float((graph_mask[edge_id] + mask_delta[edge_id]).item()),
                    "current_mask": float(graph_mask[edge_id].item()),
                    "mask_delta": float(mask_delta[edge_id].item()),
                    "graph_score": float(self.last_graph_score.detach().float()[edge_id].item()),
                    "graph_utility": float(graph_utility[edge_id].item()),
                    "src_degree": float(degree[src].item()),
                    "dst_degree": float(degree[dst].item()),
                    "src_feature_norm": float(feature_norm[src].item()),
                    "dst_feature_norm": float(feature_norm[dst].item()),
                    "src_node_importance": src_importance,
                    "dst_node_importance": dst_importance,
                    "edge_importance": edge_importance,
                    "graph_keep": current_graph_keep,
                    "param_keep": current_param_keep,
                }
            )
        return rows

    @torch.no_grad()
    def hidden_coupling_utilities(
        self,
        hidden_states: list[dict[str, object]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Build hidden utility from the already-computed training forward.

        The activation-gradient product is a local sensitivity signal. It
        provides node utility for the graph lane and channel utility for the
        parameter lane without launching counterfactual mask forwards.
        """

        empty_stats = {
            "hidden_coupling_available": 0.0,
            "hidden_coupling_forward_reused": 1.0,
            "hidden_coupling_extra_forwards": 0.0,
        }
        if hidden_states is None:
            empty_stats["hidden_coupling_forward_reused"] = 0.0
            return torch.zeros_like(self.edge_logits), torch.zeros_like(self.param_logits), empty_stats

        layer_signals: list[tuple[int, torch.Tensor, bool]] = []
        for item in hidden_states:
            if item.get("kind") != "activation":
                continue
            value = item.get("tensor")
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                continue
            activation = value.detach().float()
            gradient = value.grad
            if gradient is None:
                signal = activation.abs()
                gradient_available = False
            else:
                signal = (activation * gradient.detach().float()).abs()
                gradient_available = True
            layer_signals.append((int(item.get("layer", 0)), signal, gradient_available))

        if not layer_signals:
            empty_stats["hidden_coupling_forward_reused"] = 0.0
            return torch.zeros_like(self.edge_logits), torch.zeros_like(self.param_logits), empty_stats

        row, col = self.base_edge_index
        layer_strength = torch.stack([signal.mean() for _, signal, _ in layer_signals])
        weights = torch.softmax(layer_strength, dim=0)
        graph_utility = torch.zeros_like(self.edge_logits)
        param_utility = self.param_logits.new_zeros(self.param_policy_item_count)
        channel_means: list[torch.Tensor] = []
        node_means: list[torch.Tensor] = []
        for weight, (_, signal, _) in zip(weights, layer_signals):
            channel_signal = signal.mean(dim=0)
            node_signal = signal.mean(dim=1)
            channel_means.append(channel_signal)
            node_means.append(node_signal)
            graph_score = 0.5 * (node_signal[row] + node_signal[col])
            normalized_graph = normalize_utility_signal(graph_score).to(
                device=self.edge_logits.device, dtype=self.edge_logits.dtype
            )
            normalized_param = normalize_utility_signal(channel_signal).to(
                device=self.param_logits.device, dtype=self.param_logits.dtype
            )
            if self.cfg.param_pruning_granularity == "weight" and not self.uses_channel_parameter_policy:
                normalized_param = self.expand_channel_utility_to_weight_entries(normalized_param)
            graph_utility = graph_utility + weight * normalized_graph
            param_utility = param_utility + weight * normalized_param

        gradient_available = any(available for _, _, available in layer_signals)
        channel_signal = torch.stack(channel_means).mean(dim=0)
        node_signal = torch.stack(node_means).mean(dim=0)
        stats: dict[str, float] = {
            "hidden_coupling_available": 1.0,
            "hidden_coupling_forward_reused": 1.0,
            "hidden_coupling_extra_forwards": 0.0,
            "hidden_coupling_gradient_available": float(gradient_available),
            "hidden_coupling_layer_count": float(len(layer_signals)),
            "hidden_coupling_activation_utility_mean": float(channel_signal.mean().item()),
            "hidden_coupling_activation_utility_std": float(channel_signal.std(unbiased=False).item()),
            "hidden_coupling_node_utility_mean": float(node_signal.mean().item()),
            "hidden_coupling_node_utility_std": float(node_signal.std(unbiased=False).item()),
            "hidden_graph_utility_mean": float(graph_utility.detach().float().mean().item()),
            "hidden_graph_utility_std": float(graph_utility.detach().float().std(unbiased=False).item()),
            "hidden_param_utility_mean": float(param_utility.detach().float().mean().item()),
            "hidden_param_utility_std": float(param_utility.detach().float().std(unbiased=False).item()),
            # Retain legacy numeric fields for downstream CSV schemas.
            "hidden_coupling_cosine_mean": 0.0,
            "hidden_coupling_interaction_ratio_mean": 0.0,
            "hidden_coupling_delta_g_norm_mean": 0.0,
            "hidden_coupling_delta_p_norm_mean": 0.0,
        }
        for (layer, _, _), weight in zip(layer_signals, weights.detach().float().tolist()):
            stats[f"hidden_coupling_layer_{layer}_weight"] = float(weight)
            stats[f"hidden_coupling_layer_{layer}_cosine"] = 0.0
            stats[f"hidden_coupling_layer_{layer}_interaction_ratio"] = 0.0
        return graph_utility, param_utility, stats

    @torch.no_grad()
    def write_memories(
        self,
        x: torch.Tensor | None = None,
        temperature: float = 1.0,
        epoch: int | None = None,
        hidden_states: list[dict[str, object]] | None = None,
    ) -> dict[str, float]:
        stats: dict[str, float] = {}
        if hidden_states is None:
            hidden_states = self.last_hidden_states
        graph_utility = None
        param_utility = None
        if self.edge_logits.grad is not None and self.last_graph_mask is not None:
            graph_utility = (self.edge_logits.grad.detach() * self.last_graph_mask.detach()).abs()
            self.last_graph_utility = graph_utility
        if self.param_logits.grad is not None and self.last_param_mask is not None:
            param_utility = (self.param_logits.grad.detach() * self.last_param_mask.detach()).abs()
            if self.cfg.param_pruning_granularity == "weight" and self.uses_channel_parameter_policy:
                param_utility = self.aggregate_weight_entries_to_channels(param_utility)
            self.last_param_utility = param_utility
        if not self.cfg.use_memory:
            return stats
        if self.cfg.memory_write_mode == "none":
            stats["memory_write_skipped"] = 1.0
            stats["memory_write_mode"] = 0.0
            return stats
        if (
            self.cfg.use_hidden_coupling
            and (epoch is None or epoch >= self.cfg.hidden_coupling_start_epoch)
            and self.cfg.hidden_coupling_interval > 0
            and (epoch is None or (epoch - self.cfg.hidden_coupling_start_epoch) % self.cfg.hidden_coupling_interval == 0)
        ):
            hidden_graph_utility, hidden_param_utility, hidden_stats = self.hidden_coupling_utilities(hidden_states)
            stats.update(hidden_stats)
            if graph_utility is not None and self.cfg.hidden_coupling_mix_graph > 0.0:
                mix = float(self.cfg.hidden_coupling_mix_graph)
                graph_utility = (1.0 - mix) * normalize_utility_signal(graph_utility).to(
                    device=graph_utility.device, dtype=graph_utility.dtype
                ) + mix * hidden_graph_utility.to(device=graph_utility.device, dtype=graph_utility.dtype)
                self.last_graph_utility = graph_utility
                stats["hidden_coupling_graph_mix"] = mix
                stats["graph_memory_mixed_utility_std"] = float(graph_utility.detach().float().std(unbiased=False).item())
            if param_utility is not None and self.cfg.hidden_coupling_mix_param > 0.0:
                mix = float(self.cfg.hidden_coupling_mix_param)
                param_utility = (1.0 - mix) * normalize_utility_signal(param_utility).to(
                    device=param_utility.device, dtype=param_utility.dtype
                ) + mix * hidden_param_utility.to(device=param_utility.device, dtype=param_utility.dtype)
                self.last_param_utility = param_utility
                stats["hidden_coupling_param_mix"] = mix
                stats["param_memory_mixed_utility_std"] = float(param_utility.detach().float().std(unbiased=False).item())
        elif self.cfg.use_hidden_coupling:
            stats["hidden_coupling_available"] = 0.0
        if graph_utility is not None:
            graph_cross_ctx = (
                self.last_param_mask.detach().mean()
                if self.cfg.use_cross
                else self.edge_logits.new_tensor(self.cfg.param_target_keep)
            )
            graph_ctx = self.edge_context(graph_cross_ctx)
            graph_write_stats = self.write_graph_memories(graph_ctx, graph_utility)
            stats.update(graph_write_stats)
            stats.update(self.write_graph_event_memory(graph_utility))
            stats.update(self.write_graph_recall_memory(graph_utility))
        if param_utility is not None:
            param_cross_ctx = (
                self.last_graph_mask.detach().mean()
                if self.cfg.use_cross
                else self.edge_logits.new_tensor(self.cfg.graph_target_keep)
            )
            param_ctx = self.param_context(param_cross_ctx)
            param_stats = self.write_param_memories(param_ctx, param_utility)
            stats.update(param_stats)
            stats.update(self.write_param_recall_memory(param_utility))
        stats.update(self.write_steering_memory())
        return stats

    @torch.no_grad()
    def graph_memory_write_inputs(
        self,
        graph_ctx: torch.Tensor,
        graph_utility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.cfg.graph_memory_granularity == "edge":
            return graph_ctx, graph_utility, int(graph_ctx.size(0))
        subgraph_ctx = graph_ctx.detach().float().mean(dim=0, keepdim=True).to(device=graph_ctx.device, dtype=graph_ctx.dtype)
        subgraph_utility = graph_utility.detach().float().mean().reshape(1).to(device=graph_utility.device, dtype=graph_utility.dtype)
        return subgraph_ctx, subgraph_utility, 1

    def graph_correction(self, graph_ctx: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.cfg.graph_memory_layout == "single" or self.graph_memory_topo is None or self.graph_memory_feat is None:
            full_corr, _, _, _ = self.graph_memory.read(graph_ctx)
            return full_corr, {
                "graph_memory_branch_count": 1.0,
                "graph_memory_full_correction_std": float(full_corr.detach().float().std(unbiased=False).item()),
                "graph_branch_gate_full": 1.0,
            }
        branches = self.graph_branch_contexts(graph_ctx)
        branch_names: list[str] = []
        topo_corr, _, _, _ = self.graph_memory_topo.read(branches["topo"])
        feat_corr, _, _, _ = self.graph_memory_feat.read(branches["feat"])
        branch_corrs = [topo_corr, feat_corr]
        branch_names.extend(["topo", "feat"])
        stats = {
            "graph_memory_topo_correction_std": float(topo_corr.detach().float().std(unbiased=False).item()),
            "graph_memory_feat_correction_std": float(feat_corr.detach().float().std(unbiased=False).item()),
        }
        if self.cfg.use_graph_full_branch:
            full_corr, _, _, _ = self.graph_memory.read(branches["full"])
            branch_corrs.insert(0, full_corr)
            branch_names.insert(0, "full")
            stats["graph_memory_full_correction_std"] = float(full_corr.detach().float().std(unbiased=False).item())
        if self.cfg.use_graph_grad_branch and self.graph_memory_grad is not None and "grad" in branches:
            grad_corr, _, _, _ = self.graph_memory_grad.read(branches["grad"])
            branch_corrs.append(grad_corr)
            branch_names.append("grad")
            stats["graph_memory_grad_correction_std"] = float(grad_corr.detach().float().std(unbiased=False).item())
        stacked = torch.stack(branch_corrs, dim=0)
        if self.cfg.use_graph_branch_gates:
            branch_index = {"full": 0, "topo": 1, "feat": 2, "grad": 3}
            active_logits = torch.stack([self.graph_branch_logits[branch_index[name]] for name in branch_names])
            branch_weights = torch.softmax(active_logits, dim=0).to(dtype=stacked.dtype, device=stacked.device)
            combined = (branch_weights.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0)
        else:
            branch_weights = stacked.new_full((len(branch_corrs),), 1.0 / float(len(branch_corrs)))
            combined = stacked.mean(dim=0)
        stats["graph_memory_branch_count"] = float(len(branch_corrs))
        stats["graph_memory_combined_correction_std"] = float(combined.detach().float().std(unbiased=False).item())
        for name, weight in zip(branch_names, branch_weights.detach().float().tolist()):
            stats[f"graph_branch_gate_{name}"] = float(weight)
        return combined, stats

    @torch.no_grad()
    def write_graph_memories(self, graph_ctx: torch.Tensor, graph_utility: torch.Tensor) -> dict[str, float]:
        stats: dict[str, float] = {}
        graph_write_items_total = 0.0
        if self.cfg.graph_memory_layout == "single" or self.cfg.use_graph_full_branch:
            graph_write_ctx, graph_write_utility, graph_write_items = self.graph_memory_write_inputs(graph_ctx, graph_utility)
            graph_stats = self.graph_memory.write(graph_write_ctx, graph_write_utility, mode=self.cfg.memory_write_mode)
            stats.update({f"graph_memory_{key}": value for key, value in graph_stats.items()})
            graph_write_items_total += float(graph_write_items)
        else:
            stats["graph_memory_state_norm"] = float(self.graph_memory.state.norm().item())
            stats["graph_memory_write_mode"] = 0.0
            stats["graph_memory_utility_mean"] = 0.0
            stats["graph_memory_residual_mean"] = 0.0
        stats["graph_memory_write_items"] = graph_write_items_total
        stats["graph_memory_write_granularity"] = 1.0 if self.cfg.graph_memory_granularity == "edge" else 2.0
        stats["graph_memory_branch_count"] = 1.0 if self.cfg.graph_memory_layout == "single" else 2.0 + float(self.cfg.use_graph_full_branch) + float(self.cfg.use_graph_grad_branch)
        if self.cfg.graph_memory_layout == "multi" and self.graph_memory_topo is not None and self.graph_memory_feat is not None:
            branches = self.graph_branch_contexts(graph_ctx, graph_utility)
            topo_ctx, topo_utility, topo_items = self.graph_memory_write_inputs(branches["topo"], graph_utility)
            feat_ctx, feat_utility, feat_items = self.graph_memory_write_inputs(branches["feat"], graph_utility)
            topo_stats = self.graph_memory_topo.write(topo_ctx, topo_utility, mode=self.cfg.memory_write_mode)
            feat_stats = self.graph_memory_feat.write(feat_ctx, feat_utility, mode=self.cfg.memory_write_mode)
            stats.update({f"graph_memory_topo_{key}": value for key, value in topo_stats.items()})
            stats.update({f"graph_memory_feat_{key}": value for key, value in feat_stats.items()})
            stats["graph_memory_topo_write_items"] = float(topo_items)
            stats["graph_memory_feat_write_items"] = float(feat_items)
            graph_write_items_total += float(topo_items + feat_items)
            if self.cfg.use_graph_grad_branch and self.graph_memory_grad is not None and "grad" in branches:
                grad_ctx, grad_utility, grad_items = self.graph_memory_write_inputs(branches["grad"], graph_utility)
                grad_stats = self.graph_memory_grad.write(grad_ctx, grad_utility, mode=self.cfg.memory_write_mode)
                stats.update({f"graph_memory_grad_{key}": value for key, value in grad_stats.items()})
                stats["graph_memory_grad_write_items"] = float(grad_items)
                graph_write_items_total += float(grad_items)
            stats["graph_memory_write_items"] = graph_write_items_total
        return stats

    def param_correction(self, param_ctx: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        raw_channel_corr, _, _, _ = self.param_memory.read(param_ctx)
        if self.cfg.param_memory_layout == "single" or self.param_memory_layer is None:
            return raw_channel_corr, {
                "param_memory_branch_count": 1.0,
                "param_memory_channel_raw_correction_std": float(raw_channel_corr.detach().float().std(unbiased=False).item()),
            }
        raw_layer_corr, _, _, _ = self.param_memory_layer.read(param_ctx)
        combined = 0.5 * (raw_channel_corr + raw_layer_corr)
        return combined, {
            "param_memory_branch_count": 2.0,
            "param_memory_channel_raw_correction_std": float(raw_channel_corr.detach().float().std(unbiased=False).item()),
            "param_memory_layer_raw_correction_std": float(raw_layer_corr.detach().float().std(unbiased=False).item()),
            "param_memory_combined_raw_correction_std": float(combined.detach().float().std(unbiased=False).item()),
        }

    @torch.no_grad()
    def write_param_memories(self, param_ctx: torch.Tensor, param_utility: torch.Tensor) -> dict[str, float]:
        channel_stats = self.param_memory.write(param_ctx, param_utility, mode=self.cfg.memory_write_mode)
        stats = {f"param_memory_{key}": value for key, value in channel_stats.items()}
        stats["param_memory_branch_count"] = 1.0 if self.cfg.param_memory_layout == "single" else 2.0
        if self.cfg.param_memory_layout == "multi" and self.param_memory_layer is not None:
            layer_stats = self.param_memory_layer.write(param_ctx, param_utility, mode=self.cfg.memory_write_mode)
            stats.update({f"param_memory_layer_{key}": value for key, value in layer_stats.items()})
            stats["param_memory_layer_state_norm"] = float(self.param_memory_layer.state.norm().item())
        return stats
