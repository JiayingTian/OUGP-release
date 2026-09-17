"""EXP134: DBLPv8 -> ACMv9 OOD evaluation with the EXP121 method.

Protocol:
  DBLP source training (OUGP + LHCM) -> source hard materialization ->
  source-only energy-loss affine controller training -> ACM target forward-only
  evaluation.  Source edge masks are not copied to ACM; the target graph is
  kept intact while the source-learned channel mask and weights are reused.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from ougp.data import CitationGraph, apply_split_mode, load_udagcn_domain
from ougp.deployment import (
    HARDENING_MODES,
    MaterializedGCNDeployment,
    deployment_mask_set,
    deployment_masks,
)
from ougp.tta import propagate_predictions, signed_z_tanh_gate_values
from scripts.train_ougp import build_arg_parser as build_case_study_arg_parser
from tta_source_energy_anchor import (
    energy,
    energy_gate,
    make_controller,
    source_energy_stats,
    tensor_stats,
    clone_controller_with_gamma,
    evaluate_controller,
)
from tta_energy_shift_probe import build_parser as build_exp121_parser
from run_tta_smoke import (
    accuracy,
    set_seed,
    train_source_ougp,
    train_source_variant,
    train_tta_controller,
)


@torch.no_grad()
def select_prediction_propagation(
    args: argparse.Namespace,
    source_logits: torch.Tensor,
    source_edge_index: torch.Tensor,
    source_y: torch.Tensor,
    source_val: torch.Tensor,
) -> dict[str, object]:
    if not bool(getattr(args, "tta_prediction_propagation", False)):
        return {"enabled": False, "alpha": 0.0, "steps": 1, "validation": []}
    results: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_alpha = 0.0
    best_steps = 1
    for alpha in args.tta_propagation_alpha_candidates:
        for steps in args.tta_propagation_step_candidates:
            propagated = propagate_predictions(
                source_logits,
                source_edge_index,
                alpha,
                steps,
                mode=args.tta_propagation_mode,
            )
            value = accuracy(propagated, source_y, source_val)
            results.append(
                {"alpha": float(alpha), "steps": float(steps), "validation_accuracy": value}
            )
            if value > best_accuracy:
                best_accuracy = value
                best_alpha = float(alpha)
                best_steps = int(steps)
    return {
        "enabled": True,
        "mode": args.tta_propagation_mode,
        "alpha": best_alpha,
        "steps": best_steps,
        "best_validation_accuracy": best_accuracy,
        "validation": results,
    }


def adapt_controller_on_unlabeled_target(
    args: argparse.Namespace,
    deployment: MaterializedGCNDeployment,
    controller,
    x: torch.Tensor,
    frozen_logits: torch.Tensor,
    node_gates: torch.Tensor,
    diagnostic_y: torch.Tensor | None = None,
    diagnostic_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    epochs = int(getattr(args, "tta_target_adaptation_epochs", 0))
    if epochs <= 0:
        return {"epochs": 0, "history": []}
    for parameter in deployment.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=float(args.tta_target_adaptation_lr),
    )
    teacher_prob = torch.softmax(frozen_logits.detach(), dim=-1)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        controller.train()
        controller.reset_state()
        controller.set_node_affine_strength(node_gates)
        optimizer.zero_grad(set_to_none=True)
        logits = deployment.forward_with_tta(x, controller, update_state=False)
        probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-8)
        conditional_entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
        marginal = probabilities.mean(dim=0).clamp_min(1e-8)
        marginal_entropy = -(marginal * marginal.log()).sum()
        teacher_loss = torch.nn.functional.kl_div(
            torch.log_softmax(logits, dim=-1),
            teacher_prob,
            reduction="batchmean",
        )
        anchor_loss = controller.anchor_loss()
        loss = (
            float(args.tta_target_lambda_entropy) * conditional_entropy
            - float(args.tta_target_lambda_diversity) * marginal_entropy
            + float(args.tta_target_lambda_teacher) * teacher_loss
            + float(args.tta_target_lambda_anchor) * anchor_loss
        )
        loss.backward()
        optimizer.step()
        value = float(loss.detach().item())
        history.append(
            {
                "epoch": float(epoch),
                "loss": value,
                "conditional_entropy": float(conditional_entropy.detach().item()),
                "marginal_entropy": float(marginal_entropy.detach().item()),
                "teacher_loss": float(teacher_loss.detach().item()),
                "anchor_loss": float(anchor_loss.detach().item()),
                "diagnostic_accuracy": (
                    accuracy(logits.detach(), diagnostic_y, diagnostic_mask)
                    if diagnostic_y is not None and diagnostic_mask is not None
                    else -1.0
                ),
            }
        )
        if value < best_loss:
            best_loss = value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in controller.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Target TTA did not capture a checkpoint.")
    controller.load_state_dict(best_state, strict=True)
    controller.eval()
    controller.reset_state()
    controller.set_node_affine_strength(node_gates)
    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_objective": best_loss,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_exp121_parser()
    parser.set_defaults(
        out_dir="experiments/ood_citation_transfer/dblp_to_acm/seed0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        backbone="gcn",
        num_gnn_layers=2,
        hidden_dim=32,
        epochs=200,
        graph_sparsity=0.30,
        param_sparsity=0.30,
        use_hidden_coupling=True,
        hidden_coupling_mix_graph=0.20,
        hidden_coupling_mix_param=0.20,
        profile_inference_repeats=0,
        profile_deployment_repeats=0,
    )
    parser.add_argument("--source-dataset", default="dblp")
    parser.add_argument("--target-dataset", default="acm")
    parser.add_argument("--target-feature-mask-ratio", type=float, default=0.0)
    parser.add_argument("--profile-warmup", type=int, default=0)
    parser.add_argument("--hardening-mode", choices=HARDENING_MODES, default="topk_binary")
    return parser


def pad_feature_dims(source: CitationGraph, target: CitationGraph) -> tuple[CitationGraph, CitationGraph, int]:
    feature_dim = max(source.num_features, target.num_features)

    def pad(graph: CitationGraph) -> CitationGraph:
        if graph.num_features == feature_dim:
            return graph
        padded = torch.zeros((graph.num_nodes, feature_dim), dtype=graph.x.dtype)
        padded[:, : graph.num_features] = graph.x
        return replace(graph, x=padded)

    return pad(source), pad(target), feature_dim


def pad_graph_to_dim(graph: CitationGraph, feature_dim: int) -> CitationGraph:
    """Pad a graph after its domain is loaded to the source model dimension."""
    if graph.num_features > feature_dim:
        raise ValueError(
            f"Target feature dimension {graph.num_features} exceeds source model dimension {feature_dim}."
        )
    if graph.num_features == feature_dim:
        return graph
    padded = torch.zeros((graph.num_nodes, feature_dim), dtype=graph.x.dtype)
    padded[:, : graph.num_features] = graph.x
    return replace(graph, x=padded)


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


def main() -> None:
    args = build_parser().parse_args()
    if args.backbone != "gcn" or args.num_gnn_layers != 2:
        raise ValueError("EXP134 follows EXP121 and requires a 2-layer GCN.")
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = apply_split_mode(
        load_udagcn_domain(args.data_root, args.source_dataset, split_seed=args.seed),
        args.split_mode,
        args.seed,
    )
    feature_dim = source.num_features
    args.dataset = args.source_dataset
    args.original_num_nodes = source.num_nodes
    args.original_num_edges = int(source.edge_index.size(1))
    args.node_sample_size = 0
    args.node_sample_seed = 0
    args.node_sample_mode = "none"
    args.mask_ratio = args.tta_train_mask_ratio
    args.seeds = [args.seed]
    source_x = source.x.to(device)
    source_y = source.y.to(device)
    source_train = source.train_mask.to(device)
    source_val = source.val_mask.to(device)

    dense_source_result = None
    dense_source_history = None
    dense_target_acc = None
    source_model, source_result, source_history = train_source_ougp(args, source, device)
    masks = deployment_mask_set(
        source_model,
        mode=args.hardening_mode,
        temperature=args.temp_end,
    )
    source_graph_mask = masks.graph.values
    source_channel_mask = masks.channel.values
    source_deployment = MaterializedGCNDeployment(
        source_model,
        source_graph_mask,
        source_channel_mask,
        source_x.dtype,
        graph_support=masks.graph.support,
        param_support=masks.channel.support,
    ).to(device)
    source_deployment.eval()
    for parameter in source_deployment.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        source_clean_logits = source_deployment(source_x)
    source_stats = source_energy_stats(source_clean_logits, source_train, source_val)
    source_acc = accuracy(source_clean_logits, source_y, source.test_mask.to(device))

    source_energy_mask = source_train | source_val
    controller_train_mask = source_energy_mask if args.tta_adaptation_nodes == "train_val" else source_train
    initial_controller = make_controller(source_deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    initial_state = copy.deepcopy(initial_controller.state_dict())

    no_energy_controller = make_controller(source_deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    no_energy_controller.load_state_dict(initial_state, strict=True)
    no_energy_args = copy.copy(args)
    no_energy_args.tta_lambda_energy = 0.0
    no_energy_args.tta_joint_energy_gate = False
    no_energy_summary = train_tta_controller(
        no_energy_args,
        source_deployment,
        no_energy_controller,
        source_x,
        source_y,
        controller_train_mask,
        validation_mask=source_val,
    )

    energy_controller = make_controller(source_deployment, args, args.tta_gamma_a, args.tta_gamma_b).to(device)
    energy_controller.load_state_dict(initial_state, strict=True)
    energy_summary = train_tta_controller(
        args,
        source_deployment,
        energy_controller,
        source_x,
        source_y,
        controller_train_mask,
        source_energy_stats=source_stats,
        energy_gate_fn=lambda stats: energy_gate(stats, source_stats, args),
        energy_gate_mask=source_energy_mask,
        validation_mask=source_val,
    )

    # Load target only after source training, pruning hardening, and source
    # validation-based controller selection are complete.
    target = apply_split_mode(
        load_udagcn_domain(args.data_root, args.target_dataset, split_seed=args.seed),
        args.split_mode,
        args.seed,
    )
    target = pad_graph_to_dim(target, feature_dim)
    if source.num_classes != target.num_classes:
        raise ValueError(
            f"DBLP/ACM class spaces differ: {source.num_classes} vs {target.num_classes}."
        )
    target_x = target.x.to(device)
    target_y = target.y.to(device)
    target_test = target.test_mask.to(device)
    target_graph_mask = torch.ones(
        target.edge_index.size(1), device=device, dtype=source_graph_mask.dtype
    )
    target_deployment = MaterializedGCNDeployment(
        source_model,
        target_graph_mask,
        source_channel_mask,
        target_x.dtype,
        edge_index_override=target.edge_index.to(device),
        num_nodes_override=target.num_nodes,
        param_support=masks.channel.support,
    ).to(device)
    target_deployment.eval()
    for parameter in target_deployment.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        target_frozen_logits = target_deployment(target_x)
    target_energy_values = energy(target_frozen_logits)
    target_stats = tensor_stats(target_energy_values, "target_energy")
    target_gate = energy_gate(target_stats, source_stats, args)

    if str(getattr(args, "tta_gate_learn_mode", "mean")) == "node":
        if str(args.energy_gate_mode) != "signed_z_tanh":
            raise ValueError("tta_gate_learn_mode=node requires energy_gate_mode=signed_z_tanh.")
        raw_target_node_gates = signed_z_tanh_gate_values(
            target_energy_values,
            source_stats["source_energy_mean"],
            source_stats["source_energy_std"],
            args.energy_gate_center_z,
            args.energy_gate_temperature,
            args.energy_gate_max,
            getattr(args, "energy_gate_polarity", 1.0),
        )
        if bool(getattr(args, "tta_target_reset_controller", False)):
            target_node_gates = raw_target_node_gates * float(args.tta_target_gate_scale)
            gated_controller = make_controller(
                target_deployment,
                args,
                args.tta_gamma_a,
                args.tta_gamma_b,
            ).to(device)
        else:
            target_node_gates = raw_target_node_gates * energy_controller.selected_gate_scale
            gated_controller = clone_controller_with_gamma(
                energy_controller,
                args.tta_gamma_a,
                args.tta_gamma_b,
            ).to(device)
        gated_controller.set_node_affine_strength(target_node_gates)
        target_gate = float(target_node_gates.mean().item())
    else:
        target_gate *= energy_controller.selected_gate_scale
        gated_controller = clone_controller_with_gamma(
            energy_controller,
            args.tta_gamma_a,
            args.tta_gamma_b,
        ).to(device)
        gated_controller.set_affine_strength(target_gate)
    target_adaptation_summary = adapt_controller_on_unlabeled_target(
        args,
        target_deployment,
        gated_controller,
        target_x,
        target_frozen_logits,
        (
            target_node_gates
            if str(getattr(args, "tta_gate_learn_mode", "mean")) == "node"
            else torch.full(
                (target_x.size(0),),
                float(target_gate),
                device=target_x.device,
                dtype=target_x.dtype,
            )
        ),
        diagnostic_y=None,
        diagnostic_mask=None,
    )
    no_energy_acc, no_energy_latency, no_energy_report = evaluate_controller(
        target_deployment, no_energy_controller, target_x, target_y, target_test,
        device, args.profile_inference_repeats, args.profile_warmup
    )
    energy_acc, energy_latency, energy_report = evaluate_controller(
        target_deployment, energy_controller, target_x, target_y, target_test,
        device, args.profile_inference_repeats, args.profile_warmup
    )
    gated_acc, gated_latency, gated_report = evaluate_controller(
        target_deployment, gated_controller, target_x, target_y, target_test,
        device, args.profile_inference_repeats, args.profile_warmup
    )
    propagation_summary = select_prediction_propagation(
        args,
        source_clean_logits,
        source.edge_index.to(device),
        source_y,
        source_val,
    )
    if bool(propagation_summary["enabled"]):
        with torch.no_grad():
            gated_controller.reset_state()
            gated_logits = gated_controller.test_time_forward(
                target_deployment,
                target_x,
                stats_node_mask=target_test,
                update_state=False,
            )
            gated_logits = propagate_predictions(
                gated_logits,
                target.edge_index.to(device),
                float(propagation_summary["alpha"]),
                int(propagation_summary["steps"]),
                mode=args.tta_propagation_mode,
            )
            gated_acc = accuracy(gated_logits, target_y, target_test)

    # Train Dense last so the optional baseline cannot perturb OUGP or TTA RNG state.
    if args.include_dense_baseline:
        set_seed(args.seed)
        dense_model, dense_source_result, dense_source_history = train_source_variant(
            args, source, device, variant="dense"
        )
        _, dense_channel_mask = deployment_masks(dense_model)
        dense_target_graph_mask = torch.ones(
            target.edge_index.size(1), device=device, dtype=dense_channel_mask.dtype
        )
        dense_target_deployment = MaterializedGCNDeployment(
            dense_model,
            dense_target_graph_mask,
            dense_channel_mask,
            target_x.dtype,
            edge_index_override=target.edge_index.to(device),
            num_nodes_override=target.num_nodes,
        ).to(device)
        dense_target_deployment.eval()
        for parameter in dense_target_deployment.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            dense_target_logits = dense_target_deployment(target_x)
        dense_target_acc = accuracy(dense_target_logits, target_y, target_test)

    row = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "seed": args.seed,
        "source_num_nodes": source.num_nodes,
        "source_num_edges": int(source.edge_index.size(1)),
        "target_num_nodes": target.num_nodes,
        "target_num_edges": int(target.edge_index.size(1)),
        "aligned_feature_dim": feature_dim,
        "source_num_classes": source.num_classes,
        "target_test_nodes": int(target_test.sum().item()),
        "target_graph_policy": "full_target_graph",
        "source_channel_keep": float(source_channel_mask.mean().item()),
        "source_graph_keep": float(source_graph_mask.mean().item()),
        "hardening_mode": args.hardening_mode,
        "source_graph_hardening_threshold": masks.graph.threshold,
        "source_channel_hardening_threshold": masks.channel.threshold,
        "frozen_materialized_ougp_acc": accuracy(target_frozen_logits, target_y, target_test),
        "affine_without_energy_acc": no_energy_acc,
        "affine_with_energy_loss_acc": energy_acc,
        "energy_gated_affine_acc": gated_acc,
        "energy_loss_gain_vs_frozen": energy_acc - accuracy(target_frozen_logits, target_y, target_test),
        "energy_gate_gain_vs_energy_loss": gated_acc - energy_acc,
        "energy_gated_gain_vs_frozen": gated_acc - accuracy(target_frozen_logits, target_y, target_test),
        "target_energy_gate": target_gate,
        "selected_gate_scale": energy_controller.selected_gate_scale,
        "effective_gamma_a": args.tta_gamma_a * target_gate,
        "effective_gamma_b": args.tta_gamma_b * target_gate,
        "source_controller_train_uses_target_labels": False,
        "target_labels_used_for_adaptation": False,
        "no_energy_latency_ms": no_energy_latency,
        "energy_loss_latency_ms": energy_latency,
        "energy_gated_latency_ms": gated_latency,
        **source_stats,
        **target_stats,
        **{f"no_energy_{key}": value for key, value in no_energy_report.items()},
        **{f"energy_loss_{key}": value for key, value in energy_report.items()},
        **{f"energy_gated_{key}": value for key, value in gated_report.items()},
    }
    if dense_target_acc is not None:
        row["dense_target_acc"] = dense_target_acc
    protocol = f"{args.source_dataset.upper()} source -> {args.target_dataset.upper()} target OOD"
    result = {
        "experiment": args.experiment_name,
        "protocol": protocol,
        "split": {
            "mode": args.split_mode,
            "source_train_nodes": int(source_train.sum().item()),
            "source_val_nodes": int(source_val.sum().item()),
            "target_test_nodes": int(target_test.sum().item()),
        },
        "method": "EXP121 energy-loss + affine TTA",
        "source_training_runs": 1,
        "dense_source_training_runs": 1 if dense_source_result is not None else 0,
        "source_history_epochs": len(source_history),
        "dense_source_history_epochs": (
            len(dense_source_history) if dense_source_history is not None else 0
        ),
        "source_result": source_result,
        "dense_source_result": dense_source_result,
        "source_energy_stats": source_stats,
        "target_energy_stats": target_stats,
        "target_gate": target_gate,
        "row": row,
        "config": vars(args),
        "notes": [
            "Source edge mask is not copied to target because edge indices are graph-specific.",
            "Target uses the full ACM graph and the source-trained channel mask/weights.",
            "Target labels are used only for final accuracy evaluation.",
        ],
        "no_energy_training_summary": no_energy_summary,
        "energy_training_summary": energy_summary,
        "target_adaptation_summary": target_adaptation_summary,
        "prediction_propagation_summary": propagation_summary,
    }
    (out_dir / "ood_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_rows(out_dir / "ood_result.csv", [row])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
