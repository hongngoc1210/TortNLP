"""Evaluate one existing checkpoint under predicted/gold/no/random rationale.

This accepts both checkpoints produced by the earlier simplified project and
new ablation checkpoints. Newly added optional parameters are loaded with
``strict=False`` and are not used by the default full-rationale evaluation.
"""

from __future__ import annotations

import argparse
import json

import torch
import yaml

from ablation_main import build_trainer, deep_update, load_yaml
from trainer.train_pipeline import build_loaders


def load_compatible(module, state_dict, name):
    result = module.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[{name}] missing optional/new keys:", result.missing_keys)
    if result.unexpected_keys:
        print(f"[{name}] unexpected old keys:", result.unexpected_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--sources",
        default="predicted,gold,no_rationale,random",
        help="Comma-separated rationale sources",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    # Force the full rationale architecture for this diagnostic.
    cfg = deep_update(
        cfg,
        {
            "ablation": {
                "task_mode": "joint",
                "tp_input_mode": "rationale",
                "train_rationale_source": "predicted",
                "eval_rationale_source": "predicted",
                "use_task_adapters": False,
                "use_global_residual": False,
                "gradient_method": "standard",
            }
        },
    )

    train_loader, dev_loader, test_loader = build_loaders(cfg)
    trainer, stages, _, _ = build_trainer(cfg, train_loader)
    stage1, stage2, stage3, stage4 = stages

    checkpoint = torch.load(args.checkpoint, map_location=cfg["system"]["device"])
    load_compatible(stage1, checkpoint["stage1"], "stage1")
    load_compatible(stage2, checkpoint["stage2"], "stage2")
    load_compatible(stage3, checkpoint["stage3"], "stage3")
    load_compatible(stage4, checkpoint["stage4"], "stage4")

    loader = dev_loader if args.split == "dev" else test_loader
    results = {}
    for source in [item.strip() for item in args.sources.split(",") if item.strip()]:
        results[source] = trainer.evaluate(
            loader,
            rationale_source=source,
            return_dict=True,
        )

    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
