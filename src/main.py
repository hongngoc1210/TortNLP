import os
import sys
import argparse
import yaml
import torch
import json
from tqdm import tqdm

from trainer.train_pipeline import (
    build_loaders,
    build_optimizer_and_scheduler,
)

from models.shared_encoder import Stage1Encoder
from models.re_module      import RationableExtraction
from models.pooling        import RationalePooling
from models.td_head        import TDHead

from losses.multitask_loss import MultiTaskLoss

from trainer.engine    import Trainer
from trainer.scheduler import TeacherForcingScheduler


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
# Model factory — tái sử dụng ở nhiều nơi
# =============================================================================

def build_models(cfg, device):
    """Khởi tạo 4 stage models và trả về cùng hidden size."""

    stage1 = Stage1Encoder(
        model_name          = cfg["model"]["encoder_name"],
        use_self_attention  = cfg["model"].get("use_self_attention",   True),   # [A1]
        self_attn_heads     = cfg["model"].get("self_attn_heads",      4),      # [A1]
        self_attn_dropout   = cfg["model"].get("self_attn_dropout",    0.1),    # [A1]
        cross_attn_heads    = cfg["model"].get("cross_attn_heads",     4),
        cross_attn_dropout  = cfg["model"].get("cross_attn_dropout",   0.1),
        use_cross_attention = cfg["model"].get("use_cross_attention",   True),
    ).to(device)

    hidden = stage1.encoder.hidden_size

    stage2 = RationableExtraction(hidden).to(device)
    stage3 = RationalePooling(                              # [A2]
        hidden               = hidden,
        contrastive_temp     = cfg.get("loss", {}).get("contrastive_temp",     0.07),
        use_contrastive_loss = cfg.get("loss", {}).get("use_contrastive_loss", True),
    ).to(device)
    stage4 = TDHead(
        hidden         = hidden,
        num_heads      = cfg["model"].get("td_num_heads",    4),
        dropout        = cfg["model"].get("td_dropout",      0.2),
        use_label_attn = cfg["model"].get("use_label_attn",  True),
        num_experts    = cfg["model"].get("num_experts",      4),   # [A3]
    ).to(device)

    return stage1, stage2, stage3, stage4


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _ckpt_dir(cfg) -> str:
    return cfg["system"]["save_dir"]


def save_best_model(cfg, stage1, stage2, stage3, stage4):
    """Lưu chỉ model weights — dùng để load lại khi eval / predict."""

    path = os.path.join(_ckpt_dir(cfg), "best_model.pt")
    os.makedirs(_ckpt_dir(cfg), exist_ok=True)

    torch.save(
        {
            "stage1": stage1.state_dict(),
            "stage2": stage2.state_dict(),
            "stage3": stage3.state_dict(),
            "stage4": stage4.state_dict(),
        },
        path,
    )

    return path


def save_checkpoint(
    cfg,
    epoch,
    stage1, stage2, stage3, stage4,
    optimizer,
    lr_scheduler,
    scaler,
    history,
    best_score,
    no_improve_epochs,
):
    """
    Lưu trạng thái đầy đủ để resume:
      - model weights (tất cả 4 stages)
      - optimizer state_dict
      - lr_scheduler state_dict
      - AMP GradScaler state_dict
      - history (metrics log)
      - epoch hiện tại
      - best_score và no_improve_epochs
    """

    path = os.path.join(_ckpt_dir(cfg), "last_checkpoint.pt")
    os.makedirs(_ckpt_dir(cfg), exist_ok=True)

    torch.save(
        {
            # --- training state ---
            "epoch":             epoch,
            "best_score":        best_score,
            "no_improve_epochs": no_improve_epochs,
            "history":           history,

            # --- model weights ---
            "stage1": stage1.state_dict(),
            "stage2": stage2.state_dict(),
            "stage3": stage3.state_dict(),
            "stage4": stage4.state_dict(),

            # --- optimizer & scheduler ---
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler":       scaler.state_dict(),
        },
        path,
    )

    return path


def load_best_model(cfg, stage1, stage2, stage3, stage4, device="cpu"):
    """Load chỉ model weights từ best_model.pt."""

    path = os.path.join(_ckpt_dir(cfg), "best_model.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"best_model.pt not found at: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)

    stage1.load_state_dict(ckpt["stage1"])
    stage2.load_state_dict(ckpt["stage2"])
    stage3.load_state_dict(ckpt["stage3"])
    stage4.load_state_dict(ckpt["stage4"])

    print(f"[resume] Loaded best model weights from: {path}")


def load_checkpoint(
    path,
    stage1, stage2, stage3, stage4,
    optimizer,
    lr_scheduler,
    scaler,
    device="cpu",
):
    """
    Load toàn bộ trạng thái từ last_checkpoint.pt.
    Trả về (start_epoch, best_score, no_improve_epochs, history).
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False cần thiết để load optimizer / scheduler state
    # (chứa Python objects, không phải tensor thuần)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # --- model weights ---
    stage1.load_state_dict(ckpt["stage1"])
    stage2.load_state_dict(ckpt["stage2"])
    stage3.load_state_dict(ckpt["stage3"])
    stage4.load_state_dict(ckpt["stage4"])

    # --- optimizer ---
    optimizer.load_state_dict(ckpt["optimizer"])

    # Di chuyển optimizer state sang đúng device
    # (AdamW lưu moment tensors, cần move thủ công)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    # --- lr_scheduler ---
    if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    # --- AMP scaler ---
    if ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch       = ckpt["epoch"] + 1   # tiếp tục từ epoch kế tiếp
    best_score        = ckpt["best_score"]
    no_improve_epochs = ckpt["no_improve_epochs"]
    history           = ckpt["history"]

    print(
        f"[resume] Loaded checkpoint from: {path}\n"
        f"         Resuming from epoch {start_epoch} | "
        f"best_score={best_score:.4f} | "
        f"no_improve={no_improve_epochs}"
    )

    return start_epoch, best_score, no_improve_epochs, history


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
# Main
# =============================================================================

def main():

    # ---- argument parsing ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume training từ last_checkpoint.pt")
    parser.add_argument("--ckpt",         default=None,
                        help="Path checkpoint cụ thể để resume (override mặc định)")
    parser.add_argument("--eval-only",    action="store_true",
                        help="Chỉ evaluate best_model.pt, không train")
    parser.add_argument("--predict-only", action="store_true",
                        help="Chỉ generate predictions, không train/eval")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg["system"]["device"]

    os.makedirs(_ckpt_dir(cfg), exist_ok=True)

    # ---- data ----
    train_loader, dev_loader, test_loader = build_loaders(cfg)

    # ---- models ----
    stage1, stage2, stage3, stage4 = build_models(cfg, device)

    # ---- loss ----
    lc = cfg.get("loss", {})
    loss_fn = MultiTaskLoss(
        lambda_re             = lc.get("lambda_re",             1.0),
        lambda_td             = lc.get("lambda_td",             1.0),
        lambda_contrastive    = lc.get("lambda_contrastive",    0.1),   # [A2]
        lambda_moe            = lc.get("lambda_moe",            0.01),  # [A3]
        lambda_consistency    = lc.get("lambda_consistency",    0.5),   # [B4]
        uncertainty_weighting = lc.get("uncertainty_weighting", True),  # [B3]
        focal_gamma           = lc.get("focal_gamma",           2.0),   # [B1]
        focal_alpha           = lc.get("focal_alpha",           0.25),  # [B1]
        asl_gamma_pos         = lc.get("asl_gamma_pos",         0.0),   # [B2]
        asl_gamma_neg         = lc.get("asl_gamma_neg",         4.0),   # [B2]
        asl_clip              = lc.get("asl_clip",              0.05),  # [B2]
        consistency_margin    = lc.get("consistency_margin",    0.1),   # [B4]
    )

    # ---- optimizer + lr scheduler ----
    epochs    = cfg["training"]["epochs"]
    num_steps = epochs * len(train_loader)

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        stage1, stage2, stage3, stage4, cfg, num_steps, loss_fn=loss_fn
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
    # --eval-only: load best weights rồi evaluate, xong thoát
    # =========================================================================

    if args.eval_only:
        load_best_model(cfg, stage1, stage2, stage3, stage4, device)

        print("\nEvaluating on dev set...")
        dev_metrics = trainer.evaluate(tqdm(dev_loader, desc="Dev"))
        print("\n=== Dev Results ===")
        for k, v in dev_metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(tqdm(test_loader, desc="Test"))
        print("\n=== Test Results ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        return

    # =========================================================================
    # --predict-only: load best weights rồi predict, xong thoát
    # =========================================================================

    if args.predict_only:
        load_best_model(cfg, stage1, stage2, stage3, stage4, device)

        print("\nGenerating predictions...")
        preds = trainer.predict(tqdm(test_loader, desc="Predict"))
        out   = os.path.join(_ckpt_dir(cfg), "test_predictions.jsonl")
        save_predictions(out, preds)
        print(f"Predictions saved to: {out}")

        return

    # =========================================================================
    # Resume state
    # =========================================================================

    start_epoch       = 0
    best_score        = 0.0
    no_improve_epochs = 0
    history           = _empty_history()

    if args.resume:

        # Tự động tìm last_checkpoint.pt nếu --ckpt không chỉ định
        ckpt_path = args.ckpt or os.path.join(_ckpt_dir(cfg), "last_checkpoint.pt")

        start_epoch, best_score, no_improve_epochs, history = load_checkpoint(
            ckpt_path,
            stage1, stage2, stage3, stage4,
            optimizer, lr_scheduler, trainer.scaler,
            device,
        )

        # Kiểm tra xem đã train đủ epoch chưa
        if start_epoch >= epochs:
            print(
                f"[resume] Checkpoint đã train đủ {epochs} epochs. "
                "Chỉ chạy final evaluation."
            )
            _final_eval_and_predict(cfg, trainer, dev_loader, test_loader, history)
            return

    patience = cfg["training"]["early_stopping_patience"]

    # =========================================================================
    # Training loop
    # =========================================================================

    print(f"\nStarting training from epoch {start_epoch + 1}/{epochs}")

    for epoch in range(start_epoch, epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ---- train ----
        loss_dict = trainer.train_epoch(
            tqdm(train_loader, desc="Training"), epoch
        )

        # ---- eval ----
        metrics = trainer.evaluate(tqdm(dev_loader, desc="Dev"))

        score      = metrics["score"]
        current_lr = optimizer.param_groups[0]["lr"]

        collapse_warn = ""
        if loss_dict.get("zero_loss_steps", 0) > 0:
            collapse_warn = f"  ⚠ LOSS COLLAPSE ({loss_dict['zero_loss_steps']} zero-steps)"

        print(
            f"  loss={loss_dict['loss']:.4f} "
            f"re={loss_dict['loss_re']:.4f} "
            f"td={loss_dict['loss_td']:.4f} | "
            f"RE_F1_P={metrics['re_f1_P']:.4f}@t={metrics['re_threshold_P']:.2f} "
            f"RE_F1_D={metrics['re_f1_D']:.4f}@t={metrics['re_threshold_D']:.2f} "
            f"RE_macro={metrics['re_f1_macro']:.4f} | "
            f"TD_f1={metrics['td_f1']:.4f}@t={metrics['td_threshold']:.2f} "
            f"TD_acc={metrics['td_acc']:.4f} | "
            f"score={score:.4f}"
            + collapse_warn
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

        # ---- last checkpoint (luôn lưu sau mỗi epoch) ----
        ckpt_path = save_checkpoint(
            cfg, epoch,
            stage1, stage2, stage3, stage4,
            optimizer, lr_scheduler, trainer.scaler,
            history, best_score, no_improve_epochs,
        )
        print(f"  → Saved last checkpoint → {ckpt_path}")

        # ---- early stopping ----
        if no_improve_epochs >= patience:
            print("\nEarly stopping triggered")
            break

    # =========================================================================
    # Final evaluation + predictions
    # =========================================================================

    _final_eval_and_predict(cfg, trainer, dev_loader, test_loader, history)


# =============================================================================
# Final eval + predict (tái sử dụng ở cuối training và --eval-only)
# =============================================================================

def _final_eval_and_predict(cfg, trainer, dev_loader, test_loader, history):

    device = cfg["system"]["device"]

    # Load best weights trước khi evaluate
    stage1 = trainer.stage1
    stage2 = trainer.stage2
    stage3 = trainer.stage3
    stage4 = trainer.stage4

    load_best_model(cfg, stage1, stage2, stage3, stage4, device)

    # ---- test evaluation ----
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tqdm(test_loader, desc="Test"))

    print("\n=== Final Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- save training log ----
    log_path = os.path.join(cfg["system"]["save_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {log_path}")

    # ---- predictions ----
    print("\nGenerating predictions...")
    preds = trainer.predict(tqdm(test_loader, desc="Predict"))

    out_path = os.path.join(cfg["system"]["save_dir"], "test_predictions.jsonl")
    save_predictions(out_path, preds)
    print(f"Predictions saved to: {out_path}")
    print("\nTask done!")


if __name__ == "__main__":
    main()