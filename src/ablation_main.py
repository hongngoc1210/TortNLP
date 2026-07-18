"""Train named RE/TP ablations from config/ablation_suite.yaml.

Examples:
    python src/ablation_main.py --experiment tp_only
    python src/ablation_main.py --experiment joint_predicted,task_adapters
    python src/ablation_main.py --experiment all
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

import numpy as np
import torch
import yaml

from losses.factory import build_multitask_loss
from models.factory import build_model_stages
from trainer.engine import Trainer
from trainer.scheduler import TeacherForcingScheduler
from trainer.train_pipeline import build_loaders, build_optimizer_and_scheduler


def deep_update(base: dict, override: dict) -> dict:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_update(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism can reduce speed, so keep benchmark disabled but do not force
    # deterministic algorithms that some Transformer kernels do not support.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def save_model(path, stage1, stage2, stage3, stage4, cfg, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "stage1": stage1.state_dict(),
            "stage2": stage2.state_dict(),
            "stage3": stage3.state_dict(),
            "stage4": stage4.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_model(path, stage1, stage2, stage3, stage4, device):
    checkpoint = torch.load(path, map_location=device)
    stage1.load_state_dict(checkpoint["stage1"])
    stage2.load_state_dict(checkpoint["stage2"])
    stage3.load_state_dict(checkpoint["stage3"])
    stage4.load_state_dict(checkpoint["stage4"])
    return checkpoint


def finite_mean(values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("-inf")


def selection_score(metrics: dict, selection_metric: str) -> float:
    if selection_metric == "tp":
        return metrics["tp_acc"] if math.isfinite(metrics["tp_acc"]) else float("-inf")
    if selection_metric == "re":
        return metrics["re_f1"] if math.isfinite(metrics["re_f1"]) else float("-inf")
    if selection_metric == "mean":
        return finite_mean([metrics["re_f1"], metrics["tp_acc"]])
    raise ValueError(f"Unknown selection_metric={selection_metric!r}")


def build_trainer(cfg, train_loader):
    device = cfg["system"]["device"]
    stage1, stage2, stage3, stage4 = build_model_stages(cfg, device=device)
    loss_fn = build_multitask_loss(cfg)

    epochs = int(cfg["training"]["epochs"])
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    updates_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    optimizer_updates = epochs * updates_per_epoch

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        stage1, stage2, stage3, stage4, cfg, optimizer_updates
    )

    tf_cfg = cfg.get("teacher_forcing", {})
    tf_scheduler = TeacherForcingScheduler(
        start=float(tf_cfg.get("start", 1.0)),
        end=float(tf_cfg.get("end", 0.0)),
        epochs=int(tf_cfg.get("epochs", 10)),
    )

    ablation = cfg.get("ablation", {})
    trainer = Trainer(
        stage1,
        stage2,
        stage3,
        stage4,
        loss_fn,
        optimizer,
        tf_scheduler,
        lr_scheduler=lr_scheduler,
        device=device,
        grad_accum_steps=grad_accum,
        max_grad_norm=float(cfg["training"].get("max_grad_norm", 1.0)),
        use_amp=bool(cfg["training"].get("use_amp", True)),
        task_mode=ablation.get("task_mode", "joint"),
        tp_input_mode=ablation.get("tp_input_mode", "rationale"),
        train_rationale_source=ablation.get(
            "train_rationale_source", "teacher_forcing"
        ),
        eval_rationale_source=ablation.get(
            "eval_rationale_source", "predicted"
        ),
        gradient_method=ablation.get("gradient_method", "standard"),
        grad_diagnostics_every=int(ablation.get("grad_diagnostics_every", 0)),
    )
    return trainer, (stage1, stage2, stage3, stage4), optimizer, optimizer_updates


def run_experiment(name: str, cfg: dict):
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    base_save_dir = cfg["system"]["save_dir"]
    save_dir = os.path.join(base_save_dir, name)
    cfg["system"]["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "resolved_config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=True)

    train_loader, dev_loader, test_loader = build_loaders(cfg)
    trainer, stages, optimizer, optimizer_updates = build_trainer(cfg, train_loader)
    stage1, stage2, stage3, stage4 = stages

    print(f"\n=== Experiment: {name} ===")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Raw batches/epoch: {len(train_loader)}")
    print(f"Gradient accumulation: {trainer.grad_accum_steps}")
    print(
        "Optimizer updates/epoch:",
        (len(train_loader) + trainer.grad_accum_steps - 1) // trainer.grad_accum_steps,
    )
    print(f"Total optimizer updates: {optimizer_updates}")
    print("Ablation:", cfg.get("ablation", {}))

    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("early_stopping_patience", 5))
    selection_metric = cfg.get("evaluation", {}).get("selection_metric", "mean")
    best_score = float("-inf")
    no_improve = 0
    history = []
    checkpoint_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        dev_metrics = trainer.evaluate(dev_loader, return_dict=True)
        score = selection_score(dev_metrics, selection_metric)
        train_stats = dict(trainer.last_train_stats)
        current_lr = [group["lr"] for group in optimizer.param_groups]

        row = {
            "epoch": epoch + 1,
            **train_stats,
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
            "selection_score": score,
            "lr": current_lr,
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"loss={train_loss:.4f} re_loss={train_stats['loss_re']:.4f} "
            f"tp_loss={train_stats['loss_tp']:.4f} | "
            f"RE={dev_metrics['re_f1']:.4f} TP={dev_metrics['tp_acc']:.4f} | "
            f"grad_cos={train_stats['grad_cosine']:.4f}"
        )

        if score > best_score:
            best_score = score
            no_improve = 0
            save_model(
                checkpoint_path,
                stage1,
                stage2,
                stage3,
                stage4,
                cfg,
                epoch + 1,
                dev_metrics,
            )
            print("Saved best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    with open(os.path.join(save_dir, "training_log.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2, allow_nan=True)

    checkpoint = load_model(
        checkpoint_path, stage1, stage2, stage3, stage4, cfg["system"]["device"]
    )

    results = {
        "experiment": name,
        "best_epoch": checkpoint.get("epoch"),
        "best_dev": checkpoint.get("metrics"),
        "dev_rationale_ablation": {},
        "test": trainer.evaluate(test_loader, return_dict=True),
    }

    evaluation_cfg = cfg.get("evaluation", {})
    if (
        evaluation_cfg.get("compare_rationale_sources", False)
        and cfg.get("ablation", {}).get("tp_input_mode") == "rationale"
        and cfg.get("ablation", {}).get("task_mode") != "re_only"
    ):
        for source in evaluation_cfg.get(
            "rationale_sources", ["predicted", "gold", "no_rationale", "random"]
        ):
            results["dev_rationale_ablation"][source] = trainer.evaluate(
                dev_loader,
                rationale_source=source,
                return_dict=True,
            )

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, allow_nan=True)

    print("Final results:")
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True))
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config/config.yaml")
    parser.add_argument("--suite-config", default="config/ablation_suite.yaml")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name, comma-separated names, or 'all'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    suite = load_yaml(args.suite_config).get("experiments", {})

    if args.experiment == "all":
        names = list(suite)
    else:
        names = [name.strip() for name in args.experiment.split(",") if name.strip()]

    unknown = [name for name in names if name not in suite]
    if unknown:
        raise KeyError(f"Unknown experiments: {unknown}. Available: {list(suite)}")

    summary = {}
    for name in names:
        cfg = deep_update(base_cfg, suite[name])
        summary[name] = run_experiment(name, cfg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(summary) > 1:
        summary_path = Path(base_cfg["system"]["save_dir"]) / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        print("Saved suite summary to", summary_path)


if __name__ == "__main__":
    main()
