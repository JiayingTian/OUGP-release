"""Pruning event tracing utilities for OUGP experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


GRAPH_EVENT_FIELDS = [
    "dataset",
    "variant",
    "seed",
    "epoch",
    "edge_id",
    "src_node",
    "dst_node",
    "prev_mask",
    "current_mask",
    "mask_delta",
    "graph_score",
    "graph_utility",
    "src_degree",
    "dst_degree",
    "src_feature_norm",
    "dst_feature_norm",
    "src_node_importance",
    "dst_node_importance",
    "edge_importance",
    "graph_keep",
    "param_keep",
]


class PruningTraceRecorder:
    """Append-only recorder for interpretable pruning events.

    OnlinePruningMemory remains the compact training-time memory. This recorder is
    deliberately separate: it writes sampled pruning events for later analysis and
    does not feed anything back into the model.
    """

    def __init__(self, out_dir: Path, dataset: str, variant: str, seed: int):
        self.out_dir = out_dir / "pruning_trace"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.variant = variant
        self.seed = seed
        self.graph_path = self.out_dir / f"{dataset}_{variant}_seed{seed}_graph_events.csv"

    def record_graph_events(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        write_header = not self.graph_path.exists()
        with self.graph_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRAPH_EVENT_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
