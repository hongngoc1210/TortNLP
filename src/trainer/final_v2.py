"""Utilities for two-phase training of the final MTL architecture."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def count_trainable(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _module_list(value: Any):
    if isinstance(value, (nn.ModuleList, list, tuple)):
        return value
    return None


def find_transformer_layers(hf_model: nn.Module):
    """Find the repeated Transformer block list across common HF layouts."""
    candidates = [
        (hf_model, "layers"),
        (hf_model, "layer"),
        (getattr(hf_model, "encoder", None), "layers"),
        (getattr(hf_model, "encoder", None), "layer"),
        (getattr(hf_model, "transformer", None), "layers"),
        (getattr(hf_model, "transformer", None), "layer"),
    ]

    for parent, attribute in candidates:
        if parent is None or not hasattr(parent, attribute):
            continue
        layers = _module_list(getattr(parent, attribute))
        if layers is not None:
            return layers

    return None


def configure_phase2_trainability(
    stage1: nn.Module,
    stage2: nn.Module,
    stage3: nn.Module,
    stage4: nn.Module,
    epoch: int,
    cfg: dict,
) -> dict:
    """Apply the conservative phase-2 freeze/unfreeze schedule.

    Warm-up:
      - Stage 1 and RE are frozen.
      - Only TP adapter/pooling and verdict head are trained.

    After warm-up:
      - Stage-1 fusion can be unfrozen.
      - Only the last N pretrained encoder blocks are unfrozen.
      - RE remains frozen by default; RE loss still regularizes the shared
        representation once Stage 1 is unfrozen.
    """
    warmup_epochs = int(cfg.get("head_warmup_epochs", 2))
    unfreeze_top_n = max(
        0,
        int(cfg.get("unfreeze_top_encoder_layers", 2)),
    )
    unfreeze_stage1_fusion = bool(
        cfg.get("unfreeze_stage1_fusion", True)
    )
    keep_re_frozen = bool(
        cfg.get("keep_re_frozen", True)
    )

    set_requires_grad(stage1, False)
    set_requires_grad(stage2, False)
    set_requires_grad(stage3, True)
    set_requires_grad(stage4, True)

    if bool(cfg.get("freeze_global_anchor", True)):
        global_anchor = getattr(stage4, "global_verdict_mlp", None)
        if global_anchor is not None:
            set_requires_grad(global_anchor, False)

    phase_name = "tp_head_warmup"
    unfrozen_encoder_layers = 0

    if epoch >= warmup_epochs:
        phase_name = "partial_shared_finetune"

        if unfreeze_stage1_fusion:
            # Unfreeze Stage-1 modules except the expensive pretrained encoder.
            for name, parameter in stage1.named_parameters():
                if not name.startswith("encoder."):
                    parameter.requires_grad_(True)

        hf_model = stage1.encoder.encoder
        layers = find_transformer_layers(hf_model)
        if layers is None:
            if unfreeze_top_n > 0:
                raise RuntimeError(
                    "Could not locate Transformer layers for partial unfreeze. "
                    "Inspect stage1.encoder.encoder and extend "
                    "find_transformer_layers()."
                )
        else:
            for layer in list(layers)[-unfreeze_top_n:]:
                set_requires_grad(layer, True)
            unfrozen_encoder_layers = min(
                unfreeze_top_n,
                len(layers),
            )

        if not keep_re_frozen:
            set_requires_grad(stage2, True)

    return {
        "schedule_phase": phase_name,
        "unfrozen_encoder_layers": unfrozen_encoder_layers,
        "stage1_trainable": count_trainable(stage1),
        "stage2_trainable": count_trainable(stage2),
        "stage3_trainable": count_trainable(stage3),
        "stage4_trainable": count_trainable(stage4),
    }


def _unique_parameters(
    parameters: Iterable[nn.Parameter],
    seen: set[int],
):
    output = []
    for parameter in parameters:
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        output.append(parameter)
    return output


def build_phase2_optimizer_and_scheduler(
    stage1: nn.Module,
    stage2: nn.Module,
    stage3: nn.Module,
    stage4: nn.Module,
    cfg: dict,
    num_training_steps: int,
):
    """Build named LR groups and include currently frozen parameters.

    Frozen parameters are intentionally retained in the optimizer groups so
    they can be unfrozen later without rebuilding the optimizer or losing the
    scheduler state.
    """
    base_lr = float(cfg.get("lr", 2e-5))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.06))

    encoder_lr = float(
        cfg.get("encoder_lr", base_lr * 0.1)
    )
    shared_lr = float(
        cfg.get("shared_lr", base_lr * 0.5)
    )
    re_lr = float(cfg.get("re_lr", base_lr * 0.1))
    tp_pool_lr = float(
        cfg.get("tp_pool_lr", base_lr)
    )
    tp_head_lr = float(
        cfg.get("tp_head_lr", base_lr)
    )

    seen: set[int] = set()
    encoder_params = _unique_parameters(
        stage1.encoder.parameters(),
        seen,
    )
    shared_params = _unique_parameters(
        (
            parameter
            for name, parameter in stage1.named_parameters()
            if not name.startswith("encoder.")
        ),
        seen,
    )
    re_params = _unique_parameters(
        stage2.parameters(),
        seen,
    )
    tp_pool_params = _unique_parameters(
        stage3.parameters(),
        seen,
    )
    tp_head_params = _unique_parameters(
        stage4.parameters(),
        seen,
    )

    groups = [
        {
            "params": encoder_params,
            "lr": encoder_lr,
            "name": "pretrained_encoder",
        },
        {
            "params": shared_params,
            "lr": shared_lr,
            "name": "stage1_fusion",
        },
        {
            "params": re_params,
            "lr": re_lr,
            "name": "re_branch",
        },
        {
            "params": tp_pool_params,
            "lr": tp_pool_lr,
            "name": "tp_adapter_pooling",
        },
        {
            "params": tp_head_params,
            "lr": tp_head_lr,
            "name": "tp_verdict_head",
        },
    ]
    groups = [group for group in groups if group["params"]]

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=weight_decay,
    )

    num_training_steps = max(
        1,
        int(num_training_steps),
    )
    num_warmup_steps = int(
        num_training_steps * warmup_ratio
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def _compatible_subset(
    module: nn.Module,
    source_state: dict,
    allowed_prefixes: tuple[str, ...] | None = None,
):
    target_state = module.state_dict()
    compatible = {}
    skipped = {}

    for key, value in source_state.items():
        if allowed_prefixes is not None and not key.startswith(
            allowed_prefixes
        ):
            skipped[key] = "prefix_filtered"
            continue
        if key not in target_state:
            skipped[key] = "missing_target_key"
            continue
        if target_state[key].shape != value.shape:
            skipped[key] = (
                f"shape {tuple(value.shape)} -> "
                f"{tuple(target_state[key].shape)}"
            )
            continue
        compatible[key] = value

    return compatible, skipped


def load_phase1_checkpoint(
    checkpoint_path: str | Path,
    stage1: nn.Module,
    stage2: nn.Module,
    stage3: nn.Module,
    stage4: nn.Module,
    device: str | torch.device,
) -> dict:
    """Load a joint-no-rationale checkpoint into the final architecture.

    Stage 1 is loaded fully.  RE heads are loaded by compatible keys while the
    newly introduced exact-identity adapter stays initialized.  Stage 3 is not
    loaded because it was unused in phase 1.  Only the trained global verdict
    anchor is loaded from Stage 4.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    reports = {}

    stage1_state, skipped = _compatible_subset(
        stage1,
        checkpoint["stage1"],
    )
    stage1.load_state_dict(stage1_state, strict=False)
    reports["stage1"] = {
        "loaded": len(stage1_state),
        "skipped": skipped,
    }

    stage2_state, skipped = _compatible_subset(
        stage2,
        checkpoint["stage2"],
    )
    stage2.load_state_dict(stage2_state, strict=False)
    reports["stage2"] = {
        "loaded": len(stage2_state),
        "skipped": skipped,
    }

    # Stage 3 is deliberately fresh in phase 2.
    reports["stage3"] = {
        "loaded": 0,
        "skipped": {
            "all": "phase1 global-only did not train Stage 3"
        },
    }

    stage4_state, skipped = _compatible_subset(
        stage4,
        checkpoint["stage4"],
        allowed_prefixes=("global_verdict_mlp.",),
    )
    stage4.load_state_dict(stage4_state, strict=False)
    reports["stage4"] = {
        "loaded": len(stage4_state),
        "skipped": skipped,
    }

    return {
        "checkpoint": checkpoint,
        "load_report": reports,
    }


def save_checkpoint(
    path: str | Path,
    stage1: nn.Module,
    stage2: nn.Module,
    stage3: nn.Module,
    stage4: nn.Module,
    cfg: dict,
    epoch: int,
    metrics: dict,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {
            "stage1": stage1.state_dict(),
            "stage2": stage2.state_dict(),
            "stage3": stage3.state_dict(),
            "stage4": stage4.state_dict(),
            "config": cfg,
            "epoch": int(epoch),
            "metrics": metrics,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    stage1: nn.Module,
    stage2: nn.Module,
    stage3: nn.Module,
    stage4: nn.Module,
    device: str | torch.device,
) -> dict:
    checkpoint = torch.load(
        path,
        map_location=device,
    )
    stage1.load_state_dict(checkpoint["stage1"])
    stage2.load_state_dict(checkpoint["stage2"])
    stage3.load_state_dict(checkpoint["stage3"])
    stage4.load_state_dict(checkpoint["stage4"])
    return checkpoint
