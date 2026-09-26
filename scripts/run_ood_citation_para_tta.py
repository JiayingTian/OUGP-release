"""OOD citation Para-TTA: forward-only input-energy parameter-mask adaptation."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ougp.data import apply_split_mode, load_udagcn_domain
from ougp.deployment import MaterializedGCNDeployment, deployment_mask_set, deployment_masks
from ougp.para_tta import adapt_parameter_mask
from ood_citation_transfer import build_parser as build_base_parser, pad_graph_to_dim
from run_tta_smoke import accuracy, set_seed, train_source_ougp, train_source_variant


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = build_base_parser()
    parser.set_defaults(
        experiment_name="EXP276_OOD_CITATION_PARA_TTA",
        out_dir="experiments/exp276_ood_citation_para_tta/source_to_target/seed0",
        profile_inference_repeats=0,
        profile_warmup=0,
    )
    parser.add_argument("--para-tta-score-scale", type=float, default=0.50)
    parser.add_argument("--para-tta-energy-gate", action="store_true", default=False)
    parser.add_argument("--para-tta-gate-temperature", type=float, default=1.0)
    parser.add_argument("--para-tta-eps", type=float, default=1e-6)
    parser.add_argument("--tta-protocol", choices=("para_tta",), default="para_tta")
    return parser


def timed_forward(fn, device: torch.device, repeats: int, warmup: int) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        logits = fn()
    if repeats <= 0:
        return logits, 0.0
    for _ in range(max(0, warmup)):
        with torch.no_grad():
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return logits, (time.perf_counter() - start) * 1000.0 / repeats


def main() -> None:
    args = build_parser().parse_args()
    if args.backbone != "gcn" or args.num_gnn_layers != 2:
        raise ValueError("Citation Para-TTA requires a 2-layer GCN.")
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = apply_split_mode(
        load_udagcn_domain(args.data_root, args.source_dataset, split_seed=args.seed),
        args.split_mode,
        args.seed,
    )
    source_x = source.x.to(device)
    source_y = source.y.to(device)
    source_train = source.train_mask.to(device)
    source_val = source.val_mask.to(device)
    source_test = source.test_mask.to(device)
    feature_dim = source.num_features
    args.dataset = args.source_dataset
    args.original_num_nodes = source.num_nodes
    args.original_num_edges = int(source.edge_index.size(1))
    args.node_sample_size = 0
    args.node_sample_seed = 0
    args.node_sample_mode = "none"
    args.mask_ratio = args.tta_train_mask_ratio
    args.seeds = [args.seed]

    source_model, source_result, source_history = train_source_ougp(args, source, device)
    source_masks = deployment_mask_set(
        source_model,
        mode=args.hardening_mode,
        temperature=args.temp_end,
    )
    source_graph_mask = torch.ones_like(source_masks.graph.values)
    source_param_mask = source_masks.parameter.values
    source_deployment = MaterializedGCNDeployment(
        source_model,
        source_graph_mask,
        source_param_mask,
        source_x.dtype,
        graph_support=torch.ones_like(source_masks.graph.support, dtype=torch.bool),
        param_support=source_masks.parameter.support,
    ).to(device)
    source_deployment.eval()
    source_logits, source_latency = timed_forward(
        lambda: source_deployment(source_x),
        device,
        args.profile_inference_repeats,
        args.profile_warmup,
    )
    source_acc = accuracy(source_logits, source_y, source_test)

    target = apply_split_mode(
        load_udagcn_domain(args.data_root, args.target_dataset, split_seed=args.seed),
        args.split_mode,
        args.seed,
    )
    target = pad_graph_to_dim(target, feature_dim)
    if source.num_classes != target.num_classes:
        raise ValueError(
            f"Source/target class spaces differ: {source.num_classes} vs {target.num_classes}."
        )
    target_x = target.x.to(device)
    target_y = target.y.to(device)
    target_test = target.test_mask.to(device)
    target_graph_mask = torch.ones(
        target.edge_index.size(1),
        device=device,
        dtype=source_graph_mask.dtype,
    )

    frozen_target_deployment = MaterializedGCNDeployment(
        source_model,
        target_graph_mask,
        source_param_mask,
        target_x.dtype,
        edge_index_override=target.edge_index.to(device),
        num_nodes_override=target.num_nodes,
        param_support=source_masks.parameter.support,
    ).to(device)
    frozen_target_deployment.eval()
    frozen_logits, frozen_latency = timed_forward(
        lambda: frozen_target_deployment(target_x),
        device,
        args.profile_inference_repeats,
        args.profile_warmup,
    )

    # Para-TTA uses every available target feature row, not target train nodes.
    para_result = adapt_parameter_mask(
        source_model,
        source_param_mask=source_param_mask,
        source_x=source_x,
        target_x=target_x,
        source_node_mask=None,
        target_node_mask=None,
        score_scale=args.para_tta_score_scale,
        energy_gate=args.para_tta_energy_gate,
        gate_temperature=args.para_tta_gate_temperature,
        eps=args.para_tta_eps,
    )
    para_target_deployment = MaterializedGCNDeployment(
        source_model,
        target_graph_mask,
        para_result.param_mask,
        target_x.dtype,
        edge_index_override=target.edge_index.to(device),
        num_nodes_override=target.num_nodes,
        param_support=para_result.param_mask.detach().bool(),
    ).to(device)
    para_target_deployment.eval()
    para_logits, para_latency = timed_forward(
        lambda: para_target_deployment(target_x),
        device,
        args.profile_inference_repeats,
        args.profile_warmup,
    )

    dense_target_acc = None
    dense_source_result = None
    dense_source_history = None
    if args.include_dense_baseline:
        set_seed(args.seed)
        dense_model, dense_source_result, dense_source_history = train_source_variant(
            args, source, device, variant="dense"
        )
        dense_graph_mask, dense_param_mask = deployment_masks(dense_model)
        dense_target_deployment = MaterializedGCNDeployment(
            dense_model,
            torch.ones_like(target_graph_mask),
            dense_param_mask,
            target_x.dtype,
            edge_index_override=target.edge_index.to(device),
            num_nodes_override=target.num_nodes,
        ).to(device)
        dense_target_deployment.eval()
        dense_logits, dense_latency = timed_forward(
            lambda: dense_target_deployment(target_x),
            device,
            args.profile_inference_repeats,
            args.profile_warmup,
        )
        dense_target_acc = accuracy(dense_logits, target_y, target_test)
    else:
        dense_latency = 0.0

    frozen_acc = accuracy(frozen_logits, target_y, target_test)
    para_acc = accuracy(para_logits, target_y, target_test)
    row = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "seed": args.seed,
        "source_acc": source_acc,
        "target_frozen_ougp_acc": frozen_acc,
        "target_para_tta_acc": para_acc,
        "target_para_tta_gain_vs_frozen": para_acc - frozen_acc,
        "target_test_nodes": int(target_test.sum().item()),
        "source_graph_policy": "full_source_graph", "target_graph_policy": "full_target_graph",
        "target_adaptation_features": "all_target_features",
        "target_labels_used_for_adaptation": False,
        "target_optimizer_steps": 0,
        "source_param_keep_rate": para_result.source_keep_rate,
        "para_tta_param_keep_rate": para_result.target_keep_rate,
        "para_tta_mask_churn": para_result.mask_churn,
        "para_tta_score_scale": args.para_tta_score_scale,
        "para_tta_energy_gate": args.para_tta_energy_gate,
        "para_tta_gate_scope": "target_only",
        "para_tta_input_z_rms": para_result.summary()["input_z_rms"],
        "frozen_latency_ms": frozen_latency,
        "para_tta_latency_ms": para_latency,
        "dense_latency_ms": dense_latency,
    }
    if dense_target_acc is not None:
        row["dense_target_acc"] = dense_target_acc

    result = {
        "experiment": args.experiment_name,
        "protocol": f"{args.source_dataset.upper()} source -> {args.target_dataset.upper()} target OOD",
        "method": "OUGP + para-TTA",
        "source_training_runs": 1,
        "target_tta_training": False,
        "target_optimizer_steps": 0,
        "target_labels_used_for_adaptation": False,
        "target_adaptation_protocol": "test-time input-energy matrix -> channel-wise z -> tanh(norm(z)) target gate -> fixed-budget parameter-mask re-ranking",
        "source_training_mask": "source train nodes only for source energy statistics",
        "target_adaptation_features": "all target graph features; no target train mask",
        "source_result": source_result,
        "dense_source_result": dense_source_result,
        "source_history_epochs": len(source_history),
        "dense_source_history_epochs": len(dense_source_history) if dense_source_history is not None else 0,
        "para_tta_summary": para_result.summary(),
        "row": row,
        "config": vars(args),
        "notes": [
            "OUGP source training and source graph/parameter pruning are unchanged.",
            "Para-TTA does not use Q/K/V, loss, backward, optimizer, target train nodes, or target labels.",
            "The global parameter keep count is preserved; only weight-mask distribution is re-ranked.",
            "Energy gate is disabled for source deployment and enabled only for target ESPM mask re-ranking.",
            "Target graph edges are full and target features are used before target propagation.",
        ],
    }
    (out_dir / "ood_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "para_tta_diagnostics.json").write_text(
        json.dumps(para_result.diagnostics(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rows(out_dir / "ood_result.csv", [row])
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
