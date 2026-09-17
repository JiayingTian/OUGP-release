"""Twitch multi-domain data utilities for leave-one-domain-out experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ougp.data import CitationGraph


TWITCH_DOMAINS = ("DE", "EN", "ES", "FR", "PT", "RU")


def _split_2_1_1(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a deterministic label-stratified 50/25/25 node split."""

    labels = labels.detach().cpu().long()
    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros(labels.numel(), dtype=torch.bool)
    test = torch.zeros(labels.numel(), dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == label)[0]
        if indices.numel() < 3:
            raise ValueError(f"Twitch class {int(label)} has fewer than three nodes.")
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_end = min(max(1, int(round(0.50 * indices.numel()))), indices.numel() - 2)
        val_end = min(max(train_end + 1, train_end + int(round(0.25 * indices.numel()))), indices.numel() - 1)
        train[indices[:train_end]] = True
        val[indices[train_end:val_end]] = True
        test[indices[val_end:]] = True
    return train, val, test


def _raw_path(root: str | Path, domain: str) -> Path:
    domain = str(domain).upper()
    candidates = (
        Path(root) / "twitch" / domain / "raw" / f"{domain}.npz",
        Path(root) / "Twitch" / domain / "raw" / f"{domain}.npz",
        Path(root) / domain / "raw" / f"{domain}.npz",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing Twitch domain {domain}; searched: " + ", ".join(str(path) for path in candidates)
    )


def load_twitch_domain(root: str | Path, domain: str, split_seed: int = 0) -> CitationGraph:
    """Load one official Twitch domain and apply the project 2:1:1 split."""

    domain = str(domain).upper()
    if domain not in TWITCH_DOMAINS:
        raise ValueError(f"Unknown Twitch domain {domain!r}; expected {TWITCH_DOMAINS}.")
    data = np.load(_raw_path(root, domain), allow_pickle=True)
    x = torch.from_numpy(np.asarray(data["features"], dtype=np.float32)).contiguous()
    y = torch.from_numpy(np.asarray(data["target"], dtype=np.int64)).long().contiguous()
    edge_array = np.asarray(data["edges"], dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError(f"Unexpected {domain} edge shape: {edge_array.shape}.")
    edge_index = torch.from_numpy(edge_array.T).long().contiguous()
    non_loop = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, non_loop].contiguous()
    train_mask, val_mask, test_mask = _split_2_1_1(y, split_seed)
    return CitationGraph(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        task_type="binary",
        metric_name="accuracy",
    )


def combine_twitch_domains(
    graphs: list[tuple[str, CitationGraph]],
) -> tuple[CitationGraph, dict[str, dict[str, int]]]:
    """Build a disjoint-union source graph with no cross-domain edges."""

    if not graphs:
        raise ValueError("At least one source Twitch domain is required.")
    feature_dims = {graph.num_features for _, graph in graphs}
    class_counts = {graph.num_classes for _, graph in graphs}
    if len(feature_dims) != 1:
        raise ValueError(f"Twitch source feature dimensions differ: {sorted(feature_dims)}")
    if class_counts != {2}:
        raise ValueError(f"Twitch source class counts differ or are not binary: {sorted(class_counts)}")

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    trains: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    tests: list[torch.Tensor] = []
    metadata: dict[str, dict[str, int]] = {}
    offset = 0
    for domain, graph in graphs:
        xs.append(graph.x)
        ys.append(graph.y)
        edges.append(graph.edge_index + offset)
        trains.append(graph.train_mask)
        vals.append(graph.val_mask)
        tests.append(graph.test_mask)
        metadata[domain] = {
            "node_offset": offset,
            "num_nodes": graph.num_nodes,
            "num_edges": int(graph.edge_index.size(1)),
            "train_nodes": int(graph.train_mask.sum().item()),
            "val_nodes": int(graph.val_mask.sum().item()),
            "test_nodes": int(graph.test_mask.sum().item()),
        }
        offset += graph.num_nodes
    source = CitationGraph(
        x=torch.cat(xs, dim=0),
        y=torch.cat(ys, dim=0),
        edge_index=torch.cat(edges, dim=1),
        train_mask=torch.cat(trains, dim=0),
        val_mask=torch.cat(vals, dim=0),
        test_mask=torch.cat(tests, dim=0),
        task_type="binary",
        metric_name="accuracy",
    )
    return source, metadata


def load_twitch_lodo(
    root: str | Path,
    target_domain: str,
    split_seed: int = 0,
) -> tuple[CitationGraph, CitationGraph, dict[str, dict[str, int]]]:
    """Load five source domains and one held-out target domain."""

    target_domain = str(target_domain).upper()
    if target_domain not in TWITCH_DOMAINS:
        raise ValueError(f"Unknown Twitch target domain {target_domain!r}.")
    source_domains = [domain for domain in TWITCH_DOMAINS if domain != target_domain]
    source_graphs = [
        (domain, load_twitch_domain(root, domain, split_seed=split_seed))
        for domain in source_domains
    ]
    source, metadata = combine_twitch_domains(source_graphs)
    target = load_twitch_domain(root, target_domain, split_seed=split_seed)
    metadata["target"] = {
        "num_nodes": target.num_nodes,
        "num_edges": int(target.edge_index.size(1)),
        "train_nodes": int(target.train_mask.sum().item()),
        "val_nodes": int(target.val_mask.sum().item()),
        "test_nodes": int(target.test_mask.sum().item()),
    }
    return source, target, metadata
