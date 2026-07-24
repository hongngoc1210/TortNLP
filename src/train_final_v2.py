"""Train the final two-phase MTL architecture.

Phase 1
-------
Joint RE + TP training with a global-only TP path.  This learns a stable shared
representation, strong RE heads, and a global verdict anchor.

Phase 2
-------
Load the phase-1 checkpoint into the final architecture:
  - exact-identity RE and TP adapters,
  - detached RE probabilities for TP,
  - rationale/fallback pooling initialized fallback-dominant,
  - global verdict logit plus a small rationale residual correction.

Examples
--------
Train both phases from scratch:
    python src/train_final_v2.py --config config/final_v2.yaml

Reuse the completed joint_no_rationale ablation:
    python src/train_final_v2.py \
      --config config/final_v2.yaml \
      --phase1-checkpoint outputs/ablations/joint_no_rationale/best_model.pt
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
from pathlib import Path

# Must be set before importing torch.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import numpy as np
import torch
import yaml

from losses.factory import build_multitask_loss
from models.factory import build_model_stages
from trainer.engine import Trainer
from trainer.final_v2 import (
    build_phase2_optimizer_and_scheduler,
    configure_phase2_trainability,
    load_checkpoint,
    load_phase1_checkpoint,
    save_checkpoint,
)
from trainer.scheduler import TeacherForcingScheduler
from trainer.train_pipeline import (
    build_loaders,
    build_optimizer_and_scheduler,
)


def deep_update(base: dict, override: dict) -> dict:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(output.get(key), dict)
        ):
            output[key] = deep_update(
                output[key],
                value,
            )
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[CUDA:{label}] allocated={allocated:.2f} GiB | "
        f"reserved={reserved:.2f} GiB | peak={peak:.2f} GiB",
        flush=True,
    )


def cleanup_memory(label: str | None = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        if label:
            print_cuda_memory(label)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def save_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def save_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            payload,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def trainer_from_components(
    stages,
    loss_fn,
    optimizer,
    scheduler,
    cfg,
):
    stage1, stage2, stage3, stage4 = stages
    tf_cfg = cfg.get("teacher_forcing", {})
    tf_scheduler = TeacherForcingScheduler(
        start=float(tf_cfg.get("start", 0.0)),
        end=float(tf_cfg.get("end", 0.0)),
        epochs=int(tf_cfg.get("epochs", 1)),
    )
    ablation = cfg.get("ablation", {})

    return Trainer(
        stage1,
        stage2,
        stage3,
        stage4,
        loss_fn,
        optimizer,
        tf_scheduler,
        lr_scheduler=scheduler,
        device=cfg["system"]["device"],
        grad_accum_steps=int(
            cfg["training"].get(
                "grad_accum_steps",
                1,
            )
        ),
        max_grad_norm=float(
            cfg["training"].get(
                "max_grad_norm",
                1.0,
            )
        ),
        use_amp=bool(
            cfg["training"].get("use_amp", True)
        ),
        task_mode=ablation.get("task_mode", "joint"),
        tp_input_mode=ablation.get(
            "tp_input_mode",
            "rationale",
        ),
        train_rationale_source=ablation.get(
            "train_rationale_source",
            "predicted",
        ),
        eval_rationale_source=ablation.get(
            "eval_rationale_source",
            "predicted",
        ),
        gradient_method=ablation.get(
            "gradient_method",
            "standard",
        ),
        grad_diagnostics_every=int(
            ablation.get(
                "grad_diagnostics_every",
                0,
            )
        ),
    )


def phase1_config(base_cfg: dict) -> dict:
    phase = base_cfg.get("phase1", {})
    cfg = copy.deepcopy(base_cfg)
    cfg = deep_update(
        cfg,
        {
            "training": {
                "epochs": int(phase.get("epochs", 20)),
                "batch_size": int(
                    phase.get(
                        "batch_size",
                        base_cfg["training"].get("batch_size", 1),
                    )
                ),
                "grad_accum_steps": int(
                    phase.get(
                        "grad_accum_steps",
                        base_cfg["training"].get("grad_accum_steps", 64),
                    )
                ),
                "lr": float(phase.get("lr", 2e-5)),
                "encoder_lr_multiplier": float(
                    phase.get("encoder_lr_multiplier", 0.1)
                ),
                "early_stopping_patience": int(
                    phase.get("early_stopping_patience", 5)
                ),
            },
            "ablation": {
                "task_mode": "joint",
                "tp_input_mode": "global_only",
                "train_rationale_source": "predicted",
                "eval_rationale_source": "predicted",
                "use_task_adapters": False,
                "detach_rationale_for_tp": False,
                "use_global_residual": False,
                "gradient_method": "standard",
                "grad_diagnostics_every": 0,
            },
            "teacher_forcing": {
                "start": 0.0,
                "end": 0.0,
                "epochs": 1,
            },
        },
    )
    return cfg


def phase2_config(base_cfg: dict) -> dict:
    phase = base_cfg.get("phase2", {})
    cfg = copy.deepcopy(base_cfg)
    cfg = deep_update(
        cfg,
        {
            "training": {
                "epochs": int(phase.get("epochs", 15)),
                "batch_size": int(
                    phase.get(
                        "batch_size",
                        base_cfg["training"].get("batch_size", 1),
                    )
                ),
                "grad_accum_steps": int(
                    phase.get(
                        "grad_accum_steps",
                        base_cfg["training"].get("grad_accum_steps", 64),
                    )
                ),
                "early_stopping_patience": int(
                    phase.get("early_stopping_patience", 5)
                ),
            },
            "ablation": {
                "task_mode": "joint",
                "tp_input_mode": "rationale",
                "train_rationale_source": "predicted",
                "eval_rationale_source": "predicted",
                "use_task_adapters": True,
                "adapter_bottleneck": int(
                    phase.get("adapter_bottleneck", 128)
                ),
                "adapter_dropout": float(
                    phase.get("adapter_dropout", 0.1)
                ),
                "detach_rationale_for_tp": True,
                "use_global_residual": True,
                "rationale_scale_init": float(
                    phase.get("rationale_scale_init", -1.5)
                ),
                "mix_gate_init": float(
                    phase.get("mix_gate_init", -1.5)
                ),
                "gradient_method": "standard",
                "grad_diagnostics_every": int(
                    phase.get("grad_diagnostics_every", 0)
                ),
            },
            "teacher_forcing": {
                "start": 0.0,
                "end": 0.0,
                "epochs": 1,
            },
        },
    )
    return cfg


def weighted_score(metrics: dict, cfg: dict) -> dict:
    re_f1 = float(metrics["re_f1"])
    tp_acc = float(metrics["tp_acc"])
    re_weight = float(cfg.get("re_weight", 0.4))
    tp_weight = float(cfg.get("tp_weight", 0.6))
    reference_re = float(cfg["reference_re_f1"])
    max_re_drop = float(cfg.get("max_re_drop", 0.015))
    penalty_weight = float(cfg.get("re_drop_penalty", 5.0))

    floor = reference_re - max_re_drop
    raw = re_weight * re_f1 + tp_weight * tp_acc
    drop = max(0.0, floor - re_f1)
    guarded = raw - penalty_weight * drop

    return {
        "raw_score": raw,
        "guarded_score": guarded,
        "re_floor": floor,
        "re_drop_below_floor": drop,
        "eligible": re_f1 >= floor,
    }


def train_phase1(
    cfg: dict,
    train_loader,
    dev_loader,
    save_dir: Path,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(save_dir / "resolved_config.yaml", cfg)

    stages = build_model_stages(
        cfg,
        device=cfg["system"]["device"],
    )
    loss_fn = build_multitask_loss(cfg)

    epochs = int(cfg["training"]["epochs"])
    grad_accum = int(
        cfg["training"].get("grad_accum_steps", 1)
    )
    updates_per_epoch = (
        len(train_loader) + grad_accum - 1
    ) // grad_accum
    total_updates = epochs * updates_per_epoch

    optimizer, scheduler = (
        build_optimizer_and_scheduler(
            *stages,
            cfg,
            total_updates,
        )
    )
    trainer = trainer_from_components(
        stages,
        loss_fn,
        optimizer,
        scheduler,
        cfg,
    )

    best_score = float("-inf")
    no_improve = 0
    patience = int(
        cfg["training"].get(
            "early_stopping_patience",
            5,
        )
    )
    history = []
    checkpoint_path = save_dir / "best_model.pt"

    print("\n=== Phase 1: joint global-only pretraining ===")
    print(
        f"batches={len(train_loader)}, "
        f"grad_accum={grad_accum}, "
        f"updates/epoch={updates_per_epoch}",
        flush=True,
    )

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(
            train_loader,
            epoch,
        )
        dev = trainer.evaluate(
            dev_loader,
            return_dict=True,
        )
        score = (
            float(dev["re_f1"])
            + float(dev["tp_acc"])
        ) / 2.0

        row = {
            "epoch": epoch + 1,
            **trainer.last_train_stats,
            **{f"dev_{key}": value for key, value in dev.items()},
            "selection_score": score,
            "lr": [
                group["lr"]
                for group in optimizer.param_groups
            ],
        }
        history.append(row)

        print(
            f"Phase1 {epoch + 1:02d} | "
            f"loss={train_loss:.4f} | "
            f"RE={dev['re_f1']:.4f} "
            f"TP={dev['tp_acc']:.4f} "
            f"score={score:.4f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            no_improve = 0
            save_checkpoint(
                checkpoint_path,
                *stages,
                cfg=cfg,
                epoch=epoch + 1,
                metrics=dev,
                extra={"phase": 1},
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Phase 1 early stopping", flush=True)
                break

    save_json(save_dir / "training_log.json", history)
    return checkpoint_path


def train_phase2(
    cfg: dict,
    phase1_checkpoint: Path,
    train_loader,
    dev_loader,
    test_loader,
    save_dir: Path,
) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(save_dir / "resolved_config.yaml", cfg)

    cleanup_memory("before_phase2_model")
    stages = build_model_stages(
        cfg,
        device=cfg["system"]["device"],
    )
    stage1, stage2, stage3, stage4 = stages
    print_cuda_memory("after_phase2_model")

    transfer = load_phase1_checkpoint(
        phase1_checkpoint,
        *stages,
        device=cfg["system"]["device"],
    )
    save_json(
        save_dir / "phase1_load_report.json",
        transfer["load_report"],
    )

    phase1_metrics = transfer["metadata"].get(
        "metrics",
        {},
    )
    del transfer
    cleanup_memory("after_phase1_transfer_cleanup")
    reference_re = phase1_metrics.get("re_f1")
    if not finite(reference_re):
        reference_re = cfg.get("phase2", {}).get(
            "reference_re_f1"
        )
    if not finite(reference_re):
        raise RuntimeError(
            "Phase-1 checkpoint has no finite metrics.re_f1. "
            "Set phase2.reference_re_f1 in final_v2.yaml."
        )

    phase2_cfg = cfg.get("phase2", {})
    schedule_report = configure_phase2_trainability(
        *stages,
        epoch=0,
        cfg=phase2_cfg,
    )

    epochs = int(cfg["training"]["epochs"])
    grad_accum = int(
        cfg["training"].get("grad_accum_steps", 1)
    )
    updates_per_epoch = (
        len(train_loader) + grad_accum - 1
    ) // grad_accum
    total_updates = epochs * updates_per_epoch

    optimizer, scheduler = (
        build_phase2_optimizer_and_scheduler(
            *stages,
            cfg=phase2_cfg,
            num_training_steps=total_updates,
        )
    )
    print_cuda_memory("after_phase2_optimizer")
    loss_fn = build_multitask_loss(cfg)
    trainer = trainer_from_components(
        stages,
        loss_fn,
        optimizer,
        scheduler,
        cfg,
    )

    scoring_cfg = dict(
        phase2_cfg.get("selection", {})
    )
    scoring_cfg["reference_re_f1"] = float(
        reference_re
    )

    best_guarded = float("-inf")
    no_improve = 0
    patience = int(
        cfg["training"].get(
            "early_stopping_patience",
            5,
        )
    )
    history = []
    checkpoint_path = save_dir / "best_model.pt"

    print("\n=== Phase 2: detached rationale residual ===")
    print(
        f"phase1_RE={reference_re:.4f}, "
        f"RE_floor={reference_re - float(scoring_cfg.get('max_re_drop', 0.015)):.4f}",
        flush=True,
    )
    print("Initial trainability:", schedule_report, flush=True)

    for epoch in range(epochs):
        schedule_report = configure_phase2_trainability(
            *stages,
            epoch=epoch,
            cfg=phase2_cfg,
        )
        train_loss = trainer.train_epoch(
            train_loader,
            epoch,
        )
        dev = trainer.evaluate(
            dev_loader,
            return_dict=True,
        )
        score_info = weighted_score(
            dev,
            scoring_cfg,
        )

        row = {
            "epoch": epoch + 1,
            **trainer.last_train_stats,
            **{f"dev_{key}": value for key, value in dev.items()},
            **score_info,
            **schedule_report,
            "lr": {
                group.get("name", str(index)): group["lr"]
                for index, group in enumerate(
                    optimizer.param_groups
                )
            },
        }
        history.append(row)

        print(
            f"Phase2 {epoch + 1:02d} "
            f"[{schedule_report['schedule_phase']}] | "
            f"loss={train_loss:.4f} | "
            f"RE={dev['re_f1']:.4f} "
            f"TP={dev['tp_acc']:.4f} | "
            f"score={score_info['guarded_score']:.4f} "
            f"eligible={score_info['eligible']}",
            flush=True,
        )

        if score_info["guarded_score"] > best_guarded:
            best_guarded = score_info["guarded_score"]
            no_improve = 0
            save_checkpoint(
                checkpoint_path,
                *stages,
                cfg=cfg,
                epoch=epoch + 1,
                metrics=dev,
                extra={
                    "phase": 2,
                    "phase1_checkpoint": str(
                        phase1_checkpoint
                    ),
                    "phase1_metrics": phase1_metrics,
                    "selection": score_info,
                    "trainability": schedule_report,
                },
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Phase 2 early stopping", flush=True)
                break

    save_json(save_dir / "training_log.json", history)

    checkpoint = load_checkpoint(
        checkpoint_path,
        *stages,
        device=cfg["system"]["device"],
    )

    rationale_sources = cfg.get(
        "evaluation",
        {},
    ).get(
        "rationale_sources",
        ["predicted", "gold", "no_rationale", "random"],
    )
    dev_rationale_ablation = {
        source: trainer.evaluate(
            dev_loader,
            rationale_source=source,
            return_dict=True,
        )
        for source in rationale_sources
    }

    test = trainer.evaluate(
        test_loader,
        rationale_source="predicted",
        return_dict=True,
    )
    predictions = trainer.predict(
        test_loader,
        rationale_source="predicted",
    )

    with open(
        save_dir / "test_predictions.jsonl",
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions:
            file.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                )
                + "\n"
            )

    results = {
        "architecture": "final_v2_detached_rationale_residual",
        "phase1_checkpoint": str(phase1_checkpoint),
        "phase1_metrics": phase1_metrics,
        "best_epoch": checkpoint.get("epoch"),
        "best_dev": checkpoint.get("metrics"),
        "best_selection": checkpoint.get(
            "extra",
            {},
        ).get("selection"),
        "dev_rationale_ablation": dev_rationale_ablation,
        "test": test,
    }
    save_json(save_dir / "results.json", results)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/final_v2.yaml",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        help=(
            "Optional completed joint_no_rationale checkpoint. "
            "When supplied, phase 1 is skipped."
        ),
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Require phase1.checkpoint from config and skip training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(args.config)
    seed_everything(int(base_cfg.get("seed", 42)))

    output_root = Path(
        base_cfg["system"].get(
            "save_dir",
            "outputs/final_v2",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    save_yaml(
        output_root / "base_resolved_config.yaml",
        base_cfg,
    )

    explicit_checkpoint = (
        args.phase1_checkpoint
        or base_cfg.get("phase1", {}).get("checkpoint")
    )

    if explicit_checkpoint:
        phase1_checkpoint = Path(explicit_checkpoint)
        if not phase1_checkpoint.exists():
            raise FileNotFoundError(
                f"Phase-1 checkpoint not found: {phase1_checkpoint}"
            )
        print(
            "Using existing phase-1 checkpoint:",
            phase1_checkpoint,
            flush=True,
        )
    else:
        if args.skip_phase1:
            raise ValueError(
                "--skip-phase1 requires --phase1-checkpoint or "
                "phase1.checkpoint in config."
            )
        p1_cfg = phase1_config(base_cfg)
        p1_train_loader, p1_dev_loader, _ = build_loaders(
            p1_cfg
        )
        phase1_checkpoint = train_phase1(
            p1_cfg,
            p1_train_loader,
            p1_dev_loader,
            output_root / "phase1_global",
        )
        del p1_train_loader
        del p1_dev_loader
        cleanup_memory("after_phase1_cleanup")

    p2_cfg = phase2_config(base_cfg)
    train_loader, dev_loader, test_loader = build_loaders(
        p2_cfg
    )
    print(
        "Phase-2 loader:",
        {
            "batch_size": train_loader.batch_size,
            "grad_accum_steps": p2_cfg["training"]["grad_accum_steps"],
            "raw_batches": len(train_loader),
        },
        flush=True,
    )
    results = train_phase2(
        p2_cfg,
        phase1_checkpoint,
        train_loader,
        dev_loader,
        test_loader,
        output_root / "phase2_final",
    )

    print("\nFinal V2 results")
    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
