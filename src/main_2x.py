"""
main_2x.py  —  Multi-GPU training với PyTorch DDP (2× GPU)

Chạy bằng torchrun:
  torchrun --nproc_per_node=2 main_2x.py
  torchrun --nproc_per_node=2 main_2x.py --resume
  torchrun --nproc_per_node=2 main_2x.py --eval-only
  torchrun --nproc_per_node=2 main_2x.py --predict-only

Lưu ý DDP:
  - Mỗi GPU chạy 1 process riêng (rank 0 và rank 1)
  - Gradient sync tự động qua all-reduce sau mỗi backward
  - Evaluation + checkpoint chỉ thực hiện ở rank 0
  - set_epoch() trên DistributedSampler để shuffle khác nhau mỗi epoch
"""

import os
import argparse
import yaml
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from trainer.train_pipeline2x import (
    build_loaders,
    build_models,
    build_optimizer_and_scheduler,
)
from losses.multitask_loss import MultiTaskLoss
from trainer.engine        import Trainer
from trainer.scheduler     import TeacherForcingScheduler


# =============================================================================
# DDP setup / teardown
# =============================================================================

def setup_ddp():
    """Khởi tạo process group NCCL, trả về (rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


# =============================================================================
# Config
# =============================================================================

def load_config(path="config/config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["lr"]           = float(cfg["training"]["lr"])
    cfg["training"]["weight_decay"] = float(cfg["training"]["weight_decay"])
    cfg["training"]["batch_size"]   = int(cfg["training"]["batch_size"])
    cfg["training"]["epochs"]       = int(cfg["training"]["epochs"])
    return cfg


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _ckpt_dir(cfg) -> str:
    return cfg["system"]["save_dir"]


def _unwrap(model):
    """Lấy raw model từ DDP wrapper để lấy state_dict đúng key."""
    return model.module if isinstance(model, DDP) else model


def save_best_model(cfg, stage1, stage2, stage3, stage4):
    """
    BUG 3 + BUG 8 FIX:
      - Lưu đủ cả 4 stages (thêm stage3)
      - Unwrap DDP trước khi lấy state_dict → key không bị tiền tố "module."
    """
    path = os.path.join(_ckpt_dir(cfg), "best_model.pt")
    os.makedirs(_ckpt_dir(cfg), exist_ok=True)

    torch.save(
        {
            "stage1": _unwrap(stage1).state_dict(),
            "stage2": _unwrap(stage2).state_dict(),
            "stage3": _unwrap(stage3).state_dict(),   # BUG 3 FIX
            "stage4": _unwrap(stage4).state_dict(),
        },
        path,
    )
    return path


def save_checkpoint(
    cfg, epoch,
    stage1, stage2, stage3, stage4,
    optimizer, lr_scheduler, scaler,
    history, best_score, no_improve_epochs,
):
    """Lưu toàn bộ trạng thái để resume (chỉ rank 0 gọi)."""
    path = os.path.join(_ckpt_dir(cfg), "last_checkpoint.pt")
    os.makedirs(_ckpt_dir(cfg), exist_ok=True)

    torch.save(
        {
            "epoch":             epoch,
            "best_score":        best_score,
            "no_improve_epochs": no_improve_epochs,
            "history":           history,
            "stage1": _unwrap(stage1).state_dict(),
            "stage2": _unwrap(stage2).state_dict(),
            "stage3": _unwrap(stage3).state_dict(),
            "stage4": _unwrap(stage4).state_dict(),
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "scaler":       scaler.state_dict(),
        },
        path,
    )
    return path


def load_best_model(cfg, stage1, stage2, stage3, stage4, device):
    """
    BUG 8 FIX: load vào raw model (trước DDP wrap) với map_location đúng.
    Sau khi load xong, DDP wrapper tự broadcast weights sang GPU khác
    qua broadcast_buffers hoặc manual dist.broadcast.
    """
    path = os.path.join(_ckpt_dir(cfg), "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"best_model.pt not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)

    _unwrap(stage1).load_state_dict(ckpt["stage1"])
    _unwrap(stage2).load_state_dict(ckpt["stage2"])
    _unwrap(stage3).load_state_dict(ckpt["stage3"])
    _unwrap(stage4).load_state_dict(ckpt["stage4"])


def load_checkpoint(
    path, device,
    stage1, stage2, stage3, stage4,
    optimizer, lr_scheduler, scaler,
):
    """Load trạng thái đầy đủ để resume. Trả về (start_epoch, best_score, no_improve, history)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    _unwrap(stage1).load_state_dict(ckpt["stage1"])
    _unwrap(stage2).load_state_dict(ckpt["stage2"])
    _unwrap(stage3).load_state_dict(ckpt["stage3"])
    _unwrap(stage4).load_state_dict(ckpt["stage4"])

    optimizer.load_state_dict(ckpt["optimizer"])

    # Di chuyển optimizer state sang đúng device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    if lr_scheduler is not None and ckpt.get("lr_scheduler"):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    return (
        ckpt["epoch"] + 1,
        ckpt["best_score"],
        ckpt["no_improve_epochs"],
        ckpt["history"],
    )


# =============================================================================
# Helpers
# =============================================================================

def _empty_history():
    return {
        "epoch":           [],
        "train_loss":      [], "train_loss_re": [], "train_loss_td": [],
        "dev_re_f1_P":     [], "dev_re_f1_D":   [], "dev_re_f1_macro": [],
        "dev_td_acc":      [], "dev_td_f1":      [],
        "score":           [],
        "lr":              [],
    }


def _append_history(history, epoch, loss_dict, metrics, lr):
    history["epoch"].append(epoch)
    history["train_loss"].append(loss_dict["loss"])
    history["train_loss_re"].append(loss_dict["loss_re"])
    history["train_loss_td"].append(loss_dict["loss_td"])
    history["dev_re_f1_P"].append(metrics["re_f1_P"])
    history["dev_re_f1_D"].append(metrics["re_f1_D"])
    history["dev_re_f1_macro"].append(metrics["re_f1_macro"])
    history["dev_td_acc"].append(metrics["td_acc"])
    history["dev_td_f1"].append(metrics["td_f1"])
    history["score"].append(metrics["score"])
    history["lr"].append(lr)


def save_predictions(path, preds):
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# =============================================================================
# main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--ckpt",         default=None)
    parser.add_argument("--eval-only",    action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- DDP init ----
    rank, world_size = setup_ddp()
    device = f"cuda:{rank}"

    if rank == 0:
        os.makedirs(_ckpt_dir(cfg), exist_ok=True)

    # ---- data ----
    train_loader, dev_loader, test_loader = build_loaders(cfg, rank, world_size)

    # ---- models (raw, chưa wrap DDP) ----
    stage1, stage2, stage3, stage4 = build_models(cfg, device)

    # BUG 1 FIX: wrap tất cả 4 stages bằng DDP
    # find_unused_parameters=True cần thiết vì LabelConditionedAttention
    # chỉ active khi training + có gt → một số param có thể không có grad
    ddp_kwargs = dict(
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
    )
    stage1 = DDP(stage1, **ddp_kwargs)
    stage2 = DDP(stage2, **ddp_kwargs)
    stage3 = DDP(stage3, **ddp_kwargs)
    stage4 = DDP(stage4, **ddp_kwargs)

    # ---- loss ----
    loss_fn = MultiTaskLoss()

    # ---- optimizer + scheduler ----
    epochs    = cfg["training"]["epochs"]
    # Mỗi GPU xử lý 1/world_size steps → num_steps tính theo train_loader của rank 0
    num_steps = epochs * len(train_loader)

    # BUG 6 FIX: dùng build_optimizer_and_scheduler với layer-wise LR
    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        stage1, stage2, stage3, stage4,
        cfg,
        num_steps=num_steps,
        ddp=True,
    )

    # ---- teacher forcing ----
    tf_scheduler = TeacherForcingScheduler(
        start  = cfg["teacher_forcing"]["start"],
        end    = cfg["teacher_forcing"]["end"],
        epochs = cfg["teacher_forcing"]["epochs"],
    )

    # ---- trainer ----
    trainer = Trainer(
        stage1, stage2, stage3, stage4,
        loss_fn, optimizer, tf_scheduler,
        lr_scheduler     = lr_scheduler,
        device           = device,
        grad_accum_steps = cfg["training"].get("grad_accum_steps", 1),
        max_grad_norm    = cfg["training"].get("max_grad_norm", 1.0),
    )

    # =========================================================================
    # --eval-only
    # =========================================================================

    if args.eval_only:
        if rank == 0:
            load_best_model(cfg, stage1, stage2, stage3, stage4, device)
            metrics = trainer.evaluate(tqdm(dev_loader, desc="Dev"))
            print("\n=== Dev Results ===")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            test_metrics = trainer.evaluate(tqdm(test_loader, desc="Test"))
            print("\n=== Test Results ===")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")
        cleanup_ddp()
        return

    # =========================================================================
    # --predict-only
    # =========================================================================

    if args.predict_only:
        if rank == 0:
            load_best_model(cfg, stage1, stage2, stage3, stage4, device)
            preds = trainer.predict(tqdm(test_loader, desc="Predict"))
            out   = os.path.join(_ckpt_dir(cfg), "test_predictions.jsonl")
            save_predictions(out, preds)
            print(f"Predictions saved to: {out}")
        cleanup_ddp()
        return

    # =========================================================================
    # Resume
    # =========================================================================

    start_epoch       = 0
    best_score        = 0.0
    no_improve_epochs = 0
    history           = _empty_history()

    if args.resume:
        ckpt_path = args.ckpt or os.path.join(_ckpt_dir(cfg), "last_checkpoint.pt")

        # Chỉ rank 0 load checkpoint rồi broadcast
        if rank == 0:
            start_epoch, best_score, no_improve_epochs, history = load_checkpoint(
                ckpt_path, device,
                stage1, stage2, stage3, stage4,
                optimizer, lr_scheduler, trainer.scaler,
            )
            print(f"[rank 0] Resumed from epoch {start_epoch}")

        # Broadcast start_epoch sang tất cả ranks để đồng bộ vòng lặp
        _t = torch.tensor(start_epoch, device=device)
        dist.broadcast(_t, src=0)
        start_epoch = int(_t.item())

        # DDP tự sync weights qua all-reduce gradient,
        # nhưng để chắc chắn sau resume → broadcast model params từ rank 0
        for m in [stage1, stage2, stage3, stage4]:
            for p in m.parameters():
                dist.broadcast(p.data, src=0)

        if start_epoch >= epochs:
            if rank == 0:
                _final_eval_and_predict(cfg, trainer, dev_loader, test_loader, history)
            cleanup_ddp()
            return

    patience = cfg["training"]["early_stopping_patience"]

    # =========================================================================
    # Training loop
    # =========================================================================

    if rank == 0:
        print(f"\nStarting DDP training ({world_size} GPUs) from epoch {start_epoch + 1}/{epochs}")

    for epoch in range(start_epoch, epochs):

        # BUG: set_epoch phải được gọi mỗi epoch để shuffle khác nhau
        train_loader.sampler.set_epoch(epoch)

        # rank 0 dùng tqdm, rank 1 dùng loader trực tiếp
        loader = tqdm(train_loader, desc=f"[GPU{rank}] Training") if rank == 0 \
                 else train_loader

        # BUG 4 FIX: train_epoch trả về dict, không phải scalar
        loss_dict = trainer.train_epoch(loader, epoch)

        # Barrier: đảm bảo tất cả GPU xong train trước khi rank 0 evaluate
        dist.barrier()

        # ---- Evaluation chỉ ở rank 0 ----
        if rank == 0:

            # BUG 5 FIX: evaluate() trả về dict
            metrics    = trainer.evaluate(tqdm(dev_loader, desc="Dev"))
            score      = metrics["score"]
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch + 1}/{epochs} | "
                f"loss={loss_dict['loss']:.4f} "
                f"re={loss_dict['loss_re']:.4f} "
                f"td={loss_dict['loss_td']:.4f} | "
                f"RE_F1_P={metrics['re_f1_P']:.4f} "
                f"RE_F1_D={metrics['re_f1_D']:.4f} "
                f"RE_macro={metrics['re_f1_macro']:.4f} | "
                f"TD_f1={metrics['td_f1']:.4f} "
                f"score={score:.4f}"
            )

            _append_history(history, epoch, loss_dict, metrics, current_lr)

            # ---- best model checkpoint ----
            if score > best_score:
                best_score        = score
                no_improve_epochs = 0
                path = save_best_model(cfg, stage1, stage2, stage3, stage4)
                print(f"  → Saved best model (score={best_score:.4f}) → {path}")
            else:
                no_improve_epochs += 1
                print(f"  → No improvement ({no_improve_epochs}/{patience})")

            # ---- last checkpoint (resume) ----
            ckpt_path = save_checkpoint(
                cfg, epoch,
                stage1, stage2, stage3, stage4,
                optimizer, lr_scheduler, trainer.scaler,
                history, best_score, no_improve_epochs,
            )
            print(f"  → Saved last checkpoint → {ckpt_path}")

            # Broadcast early_stop signal sang rank 1
            stop_flag = torch.tensor(
                1 if no_improve_epochs >= patience else 0,
                device=device,
            )
        else:
            stop_flag = torch.tensor(0, device=device)

        # Đồng bộ early stopping: rank 0 broadcast quyết định dừng
        dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if rank == 0:
                print("\nEarly stopping triggered")
            break

    # =========================================================================
    # Final evaluation + predictions (rank 0 only)
    # =========================================================================

    if rank == 0:
        _final_eval_and_predict(cfg, trainer, dev_loader, test_loader, history)

    cleanup_ddp()


# =============================================================================
# Final eval + predict
# =============================================================================

def _final_eval_and_predict(cfg, trainer, dev_loader, test_loader, history):

    device = next(trainer.stage1.parameters()).device

    load_best_model(cfg, trainer.stage1, trainer.stage2,
                    trainer.stage3, trainer.stage4, device)

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tqdm(test_loader, desc="Test"))

    print("\n=== Final Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    log_path = os.path.join(cfg["system"]["save_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {log_path}")

    preds    = trainer.predict(tqdm(test_loader, desc="Predict"))
    out_path = os.path.join(cfg["system"]["save_dir"], "test_predictions.jsonl")
    save_predictions(out_path, preds)
    print(f"Predictions saved to: {out_path}")
    print("\nTask done!")


if __name__ == "__main__":
    main()