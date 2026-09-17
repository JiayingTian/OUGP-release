"""Deployment-grounded operation counts for two-layer GCN comparisons."""

from __future__ import annotations


def gcn_macs(
    *,
    num_nodes: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    edge_count_with_self_loops: int,
    hidden_layers: int = 1,
) -> float:
    """Count MACs for the materialized GCN operator used by this project.

    The count includes sparse aggregation and dense linear operators.  A
    masked dense weight matrix keeps the dense linear cost; only a structured
    channel materialization changes that term.
    """

    edges = int(edge_count_with_self_loops)
    nodes = int(num_nodes)
    hidden = int(hidden_dim)
    aggregation = edges * int(input_dim) + edges * hidden * int(hidden_layers)
    linear = nodes * int(input_dim) * hidden
    linear += max(0, int(hidden_layers) - 1) * nodes * hidden * hidden
    linear += nodes * hidden * int(output_dim)
    return float(aggregation + linear)


def deployment_flops_metrics(
    *,
    num_nodes: int,
    input_dim: int,
    original_hidden_dim: int,
    deployed_hidden_dim: int,
    output_dim: int,
    original_edge_count: int,
    deployed_edge_count: int,
    hidden_layers: int = 1,
    flops_reduction_source: str = "materialized graph edges and hidden-channel dimensions",
) -> dict[str, float | str]:
    """Return deployed and dense-reference FLOPs with explicit provenance."""

    dense_edges = int(original_edge_count) + int(num_nodes)
    deployed_edges = int(deployed_edge_count) + int(num_nodes)
    dense_macs = gcn_macs(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=original_hidden_dim,
        output_dim=output_dim,
        edge_count_with_self_loops=dense_edges,
        hidden_layers=hidden_layers,
    )
    deployed_macs = gcn_macs(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=deployed_hidden_dim,
        output_dim=output_dim,
        edge_count_with_self_loops=deployed_edges,
        hidden_layers=hidden_layers,
    )
    reduction = 1.0 - deployed_macs / max(dense_macs, 1.0)
    return {
        "dense_reference_macs": dense_macs,
        "dense_reference_flops": 2.0 * dense_macs,
        "deployed_macs": deployed_macs,
        "deployed_flops": 2.0 * deployed_macs,
        "flops_reduction": reduction,
        "flops_reduction_source": flops_reduction_source,
    }


def elementwise_weight_pruning_flops_metrics(
    *,
    num_nodes: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    original_edge_count: int,
    deployed_edge_count: int,
    retained_parameter_count: int,
    total_parameter_count: int,
    hidden_layers: int = 1,
) -> dict[str, float | str]:
    """Count effective nonzero FLOPs for graph plus unstructured weight pruning.

    Dense PyTorch linear kernels still execute the full matrix shape. The
    effective count treats each retained weight as one MAC per node, while the
    dense-kernel count exposes the hardware boundary explicitly.
    """

    dense_edges = int(original_edge_count) + int(num_nodes)
    deployed_edges = int(deployed_edge_count) + int(num_nodes)
    dense_aggregation = dense_edges * int(input_dim)
    dense_aggregation += dense_edges * int(hidden_dim) * int(hidden_layers)
    deployed_aggregation = deployed_edges * int(input_dim)
    deployed_aggregation += deployed_edges * int(hidden_dim) * int(hidden_layers)
    dense_linear = int(num_nodes) * int(total_parameter_count)
    effective_linear = int(num_nodes) * int(retained_parameter_count)
    dense_macs = float(dense_aggregation + dense_linear)
    effective_macs = float(deployed_aggregation + effective_linear)
    dense_kernel_macs = float(deployed_aggregation + dense_linear)
    return {
        "dense_reference_macs": dense_macs,
        "dense_reference_flops": 2.0 * dense_macs,
        "deployed_macs": effective_macs,
        "deployed_flops": 2.0 * effective_macs,
        "flops_reduction": 1.0 - effective_macs / max(dense_macs, 1.0),
        "flops_reduction_source": "effective nonzero MACs from materialized edges and element-wise weight masks",
        "dense_kernel_deployed_macs": dense_kernel_macs,
        "dense_kernel_deployed_flops": 2.0 * dense_kernel_macs,
        "dense_kernel_flops_reduction": 1.0 - dense_kernel_macs / max(dense_macs, 1.0),
        "retained_parameter_count": float(retained_parameter_count),
        "total_parameter_count": float(total_parameter_count),
        "parameter_entry_sparsity": 1.0 - float(retained_parameter_count) / max(float(total_parameter_count), 1.0),
    }


def prediction_propagation_flops(
    *,
    num_nodes: int,
    edge_count_with_self_loops: int,
    num_classes: int,
    steps: int,
    mode: str,
) -> float:
    """Return theoretical arithmetic FLOPs for deployed prediction propagation."""

    if steps <= 0:
        return 0.0
    sparse_mm = 2 * int(edge_count_with_self_loops) * int(num_classes)
    if mode == "convex":
        node_ops = 4 * int(num_nodes) * int(num_classes)
    elif mode == "logspace_signed":
        node_ops = 8 * int(num_nodes) * int(num_classes)
    else:
        raise ValueError(f"unknown propagation mode: {mode!r}")
    return float(int(steps) * (sparse_mm + node_ops))
