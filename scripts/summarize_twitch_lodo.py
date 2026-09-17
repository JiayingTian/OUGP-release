#!/usr/bin/env python3
"""Summarize the unified Twitch LODO benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = [
    "dense",
    "ugs",
    "cgp",
    "unifews",
    "ace_glt",
    "dspar",
    "sgcn",
    "adaptivegcn",
    "lsp_p",
    "neuralsparse",
    "grasp",
    "gcnp",
    "ougp",
    "ougp_tta",
]
METHOD_LABELS = {
    "dense": "Dense",
    "ugs": "UGS",
    "cgp": "CGP",
    "unifews": "UniFews",
    "ace_glt": "ACE-GLT",
    "dspar": "DSpar",
    "sgcn": "SGCN",
    "adaptivegcn": "AdaptiveGCN",
    "lsp_p": "LSP-P",
    "neuralsparse": "NeuralSparse",
    "grasp": "GraSP",
    "gcnp": "GCNP",
    "ougp": "OUGP",
    "ougp_tta": "OUGP+TTA",
}
DOMAINS = ("DE", "EN", "ES", "FR", "PT", "RU")


def read_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": str(payload["method"]),
                "method_label": str(payload.get("method_label", METHOD_LABELS.get(str(payload["method"]), payload["method"]))),
                "target_domain": str(payload["target_domain"]),
                "seed": int(payload["seed"]),
                "source_test_accuracy": float(payload["source_test_accuracy"]),
                "target_accuracy": float(payload["target_accuracy"]),
                "target_latency_ms": float(payload["target_latency_ms"]),
                "flops_reduction": float(payload["flops_reduction"]),
                "target_flops_ratio": 1.0 - float(payload["flops_reduction"]),
                "source_graph_sparsity": float(payload.get("source_graph_sparsity", 0.0)),
                "target_parameter_sparsity": float(payload.get("target_parameter_sparsity", payload.get("parameter_sparsity", 0.0))),
                "hidden_dim": int(payload["hidden_dim"]),
                "materialized_hidden_dim": int(payload["materialized_hidden_dim"]),
                "data_protocol": str(payload.get("data_protocol", "")),
                "split_ratio": str(payload.get("split_ratio", "")),
                "target_graph_policy": str(payload.get("target_graph_policy", "")),
                "evidence_path": str(path),
            }
        )
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_rows(root)
    if not rows:
        raise RuntimeError(f"No result.json files found under {root}")

    with (root / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_domain"]), str(row["method"]))].append(row)

    aggregate: list[dict[str, object]] = []
    for domain in DOMAINS:
        for method in METHOD_ORDER:
            items = grouped.get((domain, method), [])
            if not items:
                continue
            target_acc = [float(item["target_accuracy"]) for item in items]
            aggregate.append(
                {
                    "target_domain": domain,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seeds": len(items),
                    "target_accuracy_mean": mean(target_acc),
                    "target_accuracy_std": pstdev(target_acc),
                    "source_accuracy_mean": mean([float(item["source_test_accuracy"]) for item in items]),
                    "target_latency_ms_mean": mean([float(item["target_latency_ms"]) for item in items]),
                    "target_flops_ratio_mean": mean([float(item["target_flops_ratio"]) for item in items]),
                    "flops_reduction_mean": mean([float(item["flops_reduction"]) for item in items]),
                    "source_graph_sparsity_mean": mean([float(item["source_graph_sparsity"]) for item in items]),
                    "target_parameter_sparsity_mean": mean([float(item["target_parameter_sparsity"]) for item in items]),
                    "hidden_dim": int(items[0]["hidden_dim"]),
                    "materialized_hidden_dim_mean": mean([float(item["materialized_hidden_dim"]) for item in items]),
                }
            )

    fields = list(aggregate[0])
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregate:
        by_method[str(row["method"])].append(row)
    overall: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        items = by_method.get(method, [])
        if not items:
            continue
        overall.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "target_domains": len(items),
                "target_accuracy_mean": mean([float(item["target_accuracy_mean"]) for item in items]),
                "target_accuracy_domain_std": pstdev([float(item["target_accuracy_mean"]) for item in items]),
                "target_flops_ratio_mean": mean([float(item["target_flops_ratio_mean"]) for item in items]),
                "flops_reduction_mean": mean([float(item["flops_reduction_mean"]) for item in items]),
                "target_latency_ms_mean": mean([float(item["target_latency_ms_mean"]) for item in items]),
            }
        )
    with (root / "overall_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall[0]))
        writer.writeheader()
        writer.writerows(overall)

    lines = [
        "# EXP257 Twitch LODO Main-Table Summary",
        "",
        "Protocol: five Twitch source domains -> one unseen target domain; clean features; source stratified 2:1:1; target full graph; hidden dimension 128; seeds 0--3.",
        "",
        "## Per-target domain",
        "",
        "| Target | Method | Seeds | Target Acc. | Std. | FLOPs ratio | FLOPs red. | Latency (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['target_domain']} | {row['method_label']} | {row['seeds']} | "
            f"{100 * float(row['target_accuracy_mean']):.2f} | "
            f"{100 * float(row['target_accuracy_std']):.2f} | "
            f"{float(row['target_flops_ratio_mean']):.4f} | "
            f"{100 * float(row['flops_reduction_mean']):.2f}% | "
            f"{float(row['target_latency_ms_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Across six target domains",
            "",
            "| Method | Target domains | Target Acc. | Domain Std. | FLOPs ratio | FLOPs red. | Latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overall:
        lines.append(
            f"| {row['method_label']} | {row['target_domains']} | "
            f"{100 * float(row['target_accuracy_mean']):.2f} | "
            f"{100 * float(row['target_accuracy_domain_std']):.2f} | "
            f"{float(row['target_flops_ratio_mean']):.4f} | "
            f"{100 * float(row['flops_reduction_mean']):.2f}% | "
            f"{float(row['target_latency_ms_mean']):.3f} |"
        )
    (root / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {root / 'per_seed.csv'}")
    print(f"Wrote {root / 'aggregate.csv'}")
    print(f"Wrote {root / 'overall_summary.csv'}")


if __name__ == "__main__":
    main()

