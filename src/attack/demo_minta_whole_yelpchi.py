import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src import loss_fn
from src.attack.attacker import MintaAttacker
from src.attack.surrogate_pipeline import SurrogateAttackPipeline
from src.models import HeteroNN, classifier_engine
from src.utils.hetero_dataset_util import GraphDataLoader


@dataclass
class WholeGraphYelpChiDataModule:
    target: str
    root: str
    hetero_datasets: Dict[str, List[Any]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "config_path": str(repo_root / "data" / "YelpChi" / "meta" / "YelpChi.json"),
        "mat_path": str(repo_root / "data" / "YelpChi" / "data" / "YelpChi.mat"),
        "out_dir": str(repo_root / "outputs" / "minta_whole_yelpchi_demo"),
    }


def _build_parser() -> argparse.ArgumentParser:
    defaults = _resolve_default_paths()
    parser = argparse.ArgumentParser("MintA demo on whole YelpChi graph (no subgraph split).")
    parser.add_argument("--config-path", type=str, default=defaults["config_path"])
    parser.add_argument("--mat-path", type=str, default=defaults["mat_path"])
    parser.add_argument("--out-dir", type=str, default=defaults["out_dir"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--trials", type=int, default=3)

    parser.add_argument("--net", type=str, default="HeteroSAGE", choices=["HeteroSAGE", "HeteroGAT", "RGCN"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=100)

    parser.add_argument("--victim-epochs", type=int, default=200)
    parser.add_argument("--victim-patience", type=int, default=10)
    parser.add_argument("--victim-min-delta", type=float, default=1e-3)
    parser.add_argument("--train-log-interval", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--perturb-ratio", type=float, default=0.15)
    parser.add_argument("--adv-nodes-test-size", type=int, default=100)
    parser.add_argument("--surrogate-train-size", type=int, default=4000)
    parser.add_argument("--surrogate-hidden", type=int, default=64)
    parser.add_argument("--surrogate-epochs", type=int, default=50)
    parser.add_argument("--surrogate-lr", type=float, default=0.01)
    parser.add_argument("--surrogate-patience", type=int, default=10)
    parser.add_argument("--surrogate-min-delta", type=float, default=1e-3)

    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--allow-undetected-targets", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    return parser


def _prepare_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_whole_graph(config_path: str, mat_path: str):
    ctx = GraphDataLoader.load_from_config(config_path, mat_path, device=None)
    g = ctx.g
    target = ctx.target_node
    nclass = int(g[target].y.max().item()) + 1
    return g, target, nclass


def _build_engine(graph, target: str, nclass: int, args, device: str):
    model = HeteroNN.HeteroGNN_classifier(
        net=args.net,
        target_node=target,
        metadata=graph.metadata(),
        nhid=args.hidden_dim,
        nclass=nclass,
        nlayer=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    train_mask = graph[target].train_mask
    class_count = torch.bincount(graph[target].y[train_mask].view(-1), minlength=nclass).to(torch.float)
    criterion = loss_fn.IMB_LOSS("ce", nclass, class_count.detach().cpu().numpy(), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        verbose=False,
    )

    return classifier_engine.HeteroClassifierEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        target_node=target,
        device=device,
    )


def _run_trial(base_graph, target: str, nclass: int, args, device: str, trial_idx: int) -> Dict[str, Any]:
    trial_seed = args.seed + trial_idx
    _set_seed(trial_seed)

    trial_root = os.path.join(args.out_dir, f"trial_{trial_idx:02d}")
    os.makedirs(trial_root, exist_ok=True)

    # One whole graph for each split (no partitioning), cloned to avoid accidental in-place coupling.
    dm = WholeGraphYelpChiDataModule(
        target=target,
        root=trial_root,
        hetero_datasets={
            "train": [base_graph.clone()],
            "val": [base_graph.clone()],
            "test": [base_graph.clone()],
        },
    )

    attacker = MintaAttacker(
        dataset_module=dm,
        perturb_ratio=args.perturb_ratio,
        adv_nodes_test_size=args.adv_nodes_test_size,
        positive_label=args.positive_label,
        only_attack_correctly_detected=(not args.allow_undetected_targets),
        surrogate_train_size=args.surrogate_train_size,
        surrogate_hidden=args.surrogate_hidden,
        surrogate_epochs=args.surrogate_epochs,
        surrogate_lr=args.surrogate_lr,
        enable_feature_perturb=(not args.structure_only),
        target_node_type=target,
        seed=trial_seed,
        device=device,
        surrogate_early_stop_patience=args.surrogate_patience,
        surrogate_early_stop_min_delta=args.surrogate_min_delta,
    )

    engine = _build_engine(base_graph.clone(), target, nclass, args, device)
    pipeline = SurrogateAttackPipeline(
        dataset_module=dm,
        attacker=attacker,
        classifier_engine=engine,
        defender=None,
        device=device,
        target_node=target,
    )

    summary = pipeline.minta_evasion(
        split="test",
        train_split="train",
        train_epochs=args.victim_epochs,
        shuffle=False,
        log_every=args.log_every,
        eval_split="test",
        include_defended=False,
        positive_label=args.positive_label,
        early_stop_patience=args.victim_patience,
        early_stop_min_delta=args.victim_min_delta,
        train_log_interval=args.train_log_interval,
    )

    attacked_detail = summary["details"]["attacked"]["per_sample"][0] if summary["details"]["attacked"]["per_sample"] else {}
    attack_info = attacked_detail.get("attack_info", {})
    metrics = summary["metrics"]
    trial_result = {
        "trial": trial_idx,
        "seed": trial_seed,
        "clean_micro_acc": metrics.get("clean_micro_acc"),
        "attacked_micro_acc": metrics.get("attacked_micro_acc"),
        "clean_macro_f1": metrics.get("clean_macro_f1"),
        "attacked_macro_f1": metrics.get("attacked_macro_f1"),
        "attacked_target_asr_good": metrics.get("attacked_target_asr_good"),
        "attacked_target_asr_bad": metrics.get("attacked_target_asr_bad"),
        "attacked_target_asr_post": metrics.get("attacked_target_asr_post"),
        "attacked_target_nfr": metrics.get("attacked_target_nfr"),
        "attacked_target_eval_count": metrics.get("attacked_target_eval_count"),
        "attack_info": attack_info,
    }
    return trial_result


def _mean_ignore_nan(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def main() -> None:
    args = _build_parser().parse_args()
    device = _prepare_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    base_graph, target, nclass = _load_whole_graph(args.config_path, args.mat_path)
    results = []
    for t in range(args.trials):
        trial_res = _run_trial(base_graph, target, nclass, args, device, t)
        results.append(trial_res)
        info = trial_res.get("attack_info", {})
        print(
            f"[trial {t:02d}] clean_acc={trial_res['clean_micro_acc']:.4f} "
            f"attacked_acc={trial_res['attacked_micro_acc']:.4f} "
            f"asr_good={trial_res['attacked_target_asr_good']} "
            f"asr_post={trial_res['attacked_target_asr_post']} "
            f"edges(+/-)=({info.get('num_added_edges')}/{info.get('num_removed_edges')}) "
            f"feat_l1={info.get('feature_delta_l1')}"
        )

    aggregate = {
        "trials": args.trials,
        "mean_clean_micro_acc": _mean_ignore_nan([r["clean_micro_acc"] for r in results]),
        "mean_attacked_micro_acc": _mean_ignore_nan([r["attacked_micro_acc"] for r in results]),
        "mean_attacked_target_asr_good": _mean_ignore_nan([r["attacked_target_asr_good"] for r in results]),
        "mean_attacked_target_asr_post": _mean_ignore_nan([r["attacked_target_asr_post"] for r in results]),
        "mean_attacked_target_nfr": _mean_ignore_nan([r["attacked_target_nfr"] for r in results]),
    }

    payload = {
        "config": vars(args),
        "device_used": device,
        "target_node_type": target,
        "num_classes": nclass,
        "aggregate": aggregate,
        "trials": results,
    }
    out_path = os.path.join(args.out_dir, "whole_yelpchi_minta_demo_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== Aggregate ===")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
