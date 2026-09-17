"""Training-only FLOPs estimates for the unified OUGP experiment runner.

The counts deliberately exclude validation, test, deployment materialization,
and latency profiling. A training step is reported as forward FLOPs plus an
estimated two-forward-equivalent backward pass. Hidden utility is counted as
the small activation-gradient, reduction, and edge-mapping arithmetic applied
to the already-computed hidden states; no counterfactual GCN forward is added.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def gcn_forward_flops(
    *,
    num_nodes: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    original_non_loop_edges: int,
    hidden_layers: int,
    graph_keep: float = 1.0,
    param_keep: float = 1.0,
    param_granularity: str = "weight",
) -> float:
    """Count one GCN forward pass under graph/parameter keep rates.

    Self-loops are fixed and therefore not multiplied by ``graph_keep``.  For
    weight-entry pruning, the hidden dimension remains unchanged and only the
    nonzero linear terms are reduced.  Multiply-adds count as two FLOPs.
    """

    if param_granularity not in {"weight", "channel"}:
        raise ValueError(f"unknown parameter granularity: {param_granularity!r}")
    nodes = float(num_nodes)
    hidden = float(hidden_dim)
    input_size = float(input_dim)
    output = float(output_dim)
    loops = nodes
    edges = max(0.0, float(original_non_loop_edges)) * max(0.0, min(1.0, float(graph_keep))) + loops
    layers = max(1, int(hidden_layers))

    if param_granularity == "weight":
        total_params = input_size * hidden + hidden * output
        if layers > 1:
            total_params += float(layers - 1) * hidden * hidden
        linear = nodes * total_params * max(0.0, min(1.0, float(param_keep)))
        hidden_message_dim = hidden
    else:
        kept_hidden = hidden * max(0.0, min(1.0, float(param_keep)))
        linear = nodes * (
            input_size * kept_hidden
            + float(layers - 1) * kept_hidden * kept_hidden
            + kept_hidden * output
        )
        hidden_message_dim = kept_hidden

    aggregation = edges * (input_size + float(layers) * hidden_message_dim)
    return 2.0 * (aggregation + linear)


def _keep_from_history(row: Mapping[str, object], sparsity_key: str, keep_key: str) -> float:
    hard_sparsity = row.get(sparsity_key)
    if hard_sparsity is not None:
        return max(0.0, min(1.0, 1.0 - float(hard_sparsity)))
    return max(0.0, min(1.0, float(row.get(keep_key, 1.0))))


def hidden_state_utility_flops(
    *,
    num_nodes: int,
    hidden_dim: int,
    original_non_loop_edges: int,
    hidden_layers: int,
) -> float:
    """Estimate utility arithmetic over hidden states already in memory."""

    nodes = float(num_nodes)
    hidden = float(hidden_dim)
    edges = float(max(0, original_non_loop_edges))
    layers = float(max(1, hidden_layers))
    # activation*gradient, abs/reductions, edge endpoint mapping, and small
    # normalization/broadcast operations. This is not a second GCN forward.
    return layers * (6.0 * nodes * hidden + 6.0 * edges + 8.0 * hidden)


def source_training_flops(
    *,
    history: Sequence[Mapping[str, object]],
    num_nodes: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    original_non_loop_edges: int,
    hidden_layers: int,
    param_granularity: str,
    use_hidden_coupling: bool,
    use_memory: bool,
    variant: str,
    hidden_coupling_interval: int = 1,
) -> dict[str, float | str]:
    """Estimate source-training FLOPs from per-epoch hard-mask statistics."""

    if not history:
        raise ValueError("source training FLOPs require a non-empty history")

    forward_total = 0.0
    backward_total = 0.0
    hidden_utility_total = 0.0
    interval = max(1, int(hidden_coupling_interval))
    for epoch_index, row in enumerate(history):
        if variant == "dense":
            graph_keep = 1.0
            param_keep = 1.0
        else:
            graph_keep = _keep_from_history(
                row, "training_hard_graph_sparsity", "graph_keep"
            )
            param_keep = _keep_from_history(
                row, "training_hard_parameter_sparsity", "param_keep"
            )
        normal_forward = gcn_forward_flops(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            original_non_loop_edges=original_non_loop_edges,
            hidden_layers=hidden_layers,
            graph_keep=graph_keep,
            param_keep=param_keep,
            param_granularity=param_granularity,
        )
        forward_total += normal_forward
        # Standard first-order training estimate: backward ~= 2 forward.
        backward_total += 2.0 * normal_forward

        if (
            variant != "dense"
            and use_memory
            and use_hidden_coupling
            and epoch_index % interval == 0
        ):
            hidden_utility_total += hidden_state_utility_flops(
                num_nodes=num_nodes,
                hidden_dim=hidden_dim,
                original_non_loop_edges=original_non_loop_edges,
                hidden_layers=hidden_layers,
            )

    total = forward_total + backward_total + hidden_utility_total
    return {
        "source_training_flops_forward": forward_total,
        "source_training_flops_backward": backward_total,
        "source_training_flops_hidden_utility": hidden_utility_total,
        "source_training_flops": total,
        "source_training_epochs": float(len(history)),
        "source_training_hidden_coupling_interval": float(interval),
        "source_training_flops_scope": "source forward + estimated backward + reused hidden-state utility; excludes validation/test",
    }


def tta_training_flops(
    *,
    args: Mapping[str, object],
    compact_forward_flops: float,
    adapted_forward_flops: float,
) -> dict[str, float | str]:
    """Estimate FLOPs executed by the TTA optimization loop only.

    Validation forwards used for checkpoint selection are intentionally
    excluded.  Frozen energy-reference forwards contribute to forward FLOPs
    but have no backward cost.
    """

    epochs = max(0, int(args.get("tta_epochs", 0)))
    lambda_energy = float(args.get("tta_lambda_energy", 0.0))
    negative_ratio = float(args.get("tta_energy_negative_mask_ratio", 0.5))
    lambda_consistency = float(args.get("tta_lambda_consistency", 0.0))
    consistency_ratio = float(
        args.get("tta_consistency_mask_ratio", args.get("mask_ratio", 0.0))
    )
    clean_reuse = (
        float(args.get("mask_ratio", 0.0)) == 0.0
        and consistency_ratio == 0.0
        and str(args.get("tta_state_mode", "stateless")) == "stateless"
        and not bool(args.get("tta_carry_state", False))
    )

    forward_per_epoch = float(adapted_forward_flops)
    differentiable_forward_per_epoch = float(adapted_forward_flops)
    frozen_forward_per_epoch = 0.0
    if lambda_energy > 0.0 and not clean_reuse:
        if negative_ratio > 0.0:
            forward_per_epoch += float(adapted_forward_flops)
            differentiable_forward_per_epoch += float(adapted_forward_flops)
        else:
            forward_per_epoch += float(compact_forward_flops)
            frozen_forward_per_epoch += float(compact_forward_flops)
    if lambda_consistency > 0.0 and not clean_reuse:
        forward_per_epoch += float(adapted_forward_flops)
        differentiable_forward_per_epoch += float(adapted_forward_flops)

    backward_per_epoch = 2.0 * differentiable_forward_per_epoch
    total_per_epoch = forward_per_epoch + backward_per_epoch
    frozen_setup = float(compact_forward_flops)
    return {
        "tta_training_flops_forward_per_epoch": forward_per_epoch,
        "tta_training_flops_backward_per_epoch": backward_per_epoch,
        "tta_training_flops_frozen_forward_per_epoch": frozen_forward_per_epoch,
        "tta_training_clean_forward_reuse": float(clean_reuse),
        "tta_training_flops_per_epoch": total_per_epoch,
        "tta_training_flops_frozen_setup": frozen_setup,
        "tta_training_flops_forward": forward_per_epoch * epochs + frozen_setup,
        "tta_training_flops_backward": backward_per_epoch * epochs,
        "tta_training_flops": total_per_epoch * epochs + frozen_setup,
        "tta_training_epochs": float(epochs),
        "tta_training_flops_scope": "TTA optimization forward + estimated backward; excludes TTA validation/test/deployment",
    }
