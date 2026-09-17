#!/usr/bin/env python3
"""Summarize a complete TTA seed sweep without cherry-picking favorable seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


BASELINE_COLUMNS = {
    "cora": "cora_acc_percent",
    "pubmed": "pubmed_acc_percent",
    "citationfull_cora": "cora_full_acc_percent",
    "citationfull_dblp": "dblp_acc_percent",
    "acm": "acm_acc_percent",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = list(csv.DictReader(args.baseline_csv.open(encoding="utf-8")))
    baselines: dict[str, list[tuple[str, float]]] = {}
    for dataset, column in BASELINE_COLUMNS.items():
        baselines[dataset] = sorted(
            [
                (row["method"], float(row[column]) / 100.0)
                for row in baseline_rows
                if row["method"] != "Dense" and row[column]
            ],
            key=lambda item: item[1],
            reverse=True,
        )

    grouped: dict[str, list[dict[str, float]]] = {}
    per_seed: list[dict[str, object]] = []
    for path in sorted(args.root.glob("iid_*/seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        ougp = next(
            float(row["clean_test_accuracy"])
            for row in rows
            if str(row["method"]).startswith("OUGP")
            and "+ stateless" not in str(row["method"])
        )
        tta = next(
            float(row["clean_test_accuracy"])
            for row in rows
            if "+ stateless" in str(row["method"])
        )
        record = {
            "seed": float(payload["seed"]),
            "ougp": ougp,
            "tta": tta,
            "gain": tta - ougp,
        }
        dataset = str(payload["dataset"])
        grouped.setdefault(dataset, []).append(record)
        per_seed.append({"dataset": dataset, **record})

    summary: list[dict[str, object]] = []
    for dataset, values in sorted(grouped.items()):
        tta_values = [value["tta"] for value in values]
        gains = [value["gain"] for value in values]
        first_method, first_accuracy = baselines[dataset][0]
        second_method, second_accuracy = baselines[dataset][1]
        mean_accuracy = statistics.mean(tta_values)
        rank = 1 + sum(accuracy > mean_accuracy for _, accuracy in baselines[dataset])
        best = max(values, key=lambda value: value["tta"])
        summary.append(
            {
                "dataset": dataset,
                "seeds": len(values),
                "tta_accuracy_mean": mean_accuracy,
                "tta_accuracy_std": statistics.pstdev(tta_values),
                "ougp_gain_mean": statistics.mean(gains),
                "ougp_gain_std": statistics.pstdev(gains),
                "positive_gain_seeds": sum(gain > 0.0 for gain in gains),
                "baseline_rank_excluding_dense": rank,
                "first_baseline": first_method,
                "first_baseline_accuracy": first_accuracy,
                "gap_to_first": mean_accuracy - first_accuracy,
                "second_baseline": second_method,
                "second_baseline_accuracy": second_accuracy,
                "gap_to_second": mean_accuracy - second_accuracy,
                "best_seed_diagnostic": int(best["seed"]),
                "best_seed_accuracy_diagnostic": best["tta"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    per_seed_path = args.output.with_name(args.output.stem + "_per_seed.csv")
    with per_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    for row in summary:
        print(
            f"{row['dataset']}: mean={100 * float(row['tta_accuracy_mean']):.3f}% "
            f"std={100 * float(row['tta_accuracy_std']):.3f}% "
            f"rank={row['baseline_rank_excluding_dense']} "
            f"gap_first={100 * float(row['gap_to_first']):+.3f}pp "
            f"gap_second={100 * float(row['gap_to_second']):+.3f}pp"
        )
    print(f"summary={args.output}")
    print(f"per_seed={per_seed_path}")


if __name__ == "__main__":
    main()
