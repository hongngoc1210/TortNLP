import os
import yaml
import torch
import json
from tqdm import tqdm

from trainer.train_pipeline import build_loaders, build_optimizer_and_scheduler

from models.factory import build_model_stages
from losses.factory import build_multitask_loss

from trainer.engine import Trainer
from trainer.scheduler import TeacherForcingScheduler


# =============================================================================
# load_config
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
# save_predictions
# =============================================================================

def save_predictions(path, preds):

    with open(path, "w", encoding="utf-8") as f:

        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# =============================================================================
# save_model / load_model
# =============================================================================

def save_model(path, stage1, stage2, stage3, stage4):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "stage1": stage1.state_dict(),
            "stage2": stage2.state_dict(),
            "stage3": stage3.state_dict(),
            "stage4": stage4.state_dict(),
        },
        path
    )


def load_model(path, stage1, stage2, stage3, stage4):

    ckpt = torch.load(path, map_location=next(stage1.parameters()).device)

    stage1.load_state_dict(ckpt["stage1"])
    stage2.load_state_dict(ckpt["stage2"])
    stage3.load_state_dict(ckpt["stage3"])
    stage4.load_state_dict(ckpt["stage4"])


# =============================================================================
# main
# =============================================================================

def main():

    cfg = load_config()

    device = cfg["system"]["device"]

    os.makedirs(cfg["system"]["save_dir"], exist_ok=True)

    # ---- Load dataset ----

    train_loader, dev_loader, test_loader = build_loaders(cfg)

    # ---- Models ----

    stage1, stage2, stage3, stage4 = build_model_stages(cfg, device=device)

    # ---- Loss ----

    loss_fn = build_multitask_loss(cfg)

    # ---- Optimizer & scheduler ----

    epochs = cfg["training"]["epochs"]
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    updates_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    num_steps = epochs * updates_per_epoch

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        stage1, stage2, stage3, stage4, cfg, num_steps
    )

    # ---- Teacher forcing ----

    tf_scheduler = TeacherForcingScheduler(
        start  = cfg["teacher_forcing"]["start"],
        end    = cfg["teacher_forcing"]["end"],
        epochs = cfg["teacher_forcing"]["epochs"],
    )

    trainer = Trainer(
        stage1, stage2, stage3, stage4,
        loss_fn, optimizer, tf_scheduler,
        lr_scheduler     = lr_scheduler,
        device           = device,
        grad_accum_steps = cfg["training"].get("grad_accum_steps", 1),
        max_grad_norm    = cfg["training"].get("max_grad_norm",    1.0),
        use_amp           = cfg["training"].get("use_amp", True),
        task_mode=cfg.get("ablation", {}).get("task_mode", "joint"),
        tp_input_mode=cfg.get("ablation", {}).get("tp_input_mode", "rationale"),
        train_rationale_source=cfg.get("ablation", {}).get(
            "train_rationale_source", "teacher_forcing"
        ),
        eval_rationale_source=cfg.get("ablation", {}).get(
            "eval_rationale_source", "predicted"
        ),
        gradient_method=cfg.get("ablation", {}).get("gradient_method", "standard"),
        grad_diagnostics_every=cfg.get("ablation", {}).get(
            "grad_diagnostics_every", 0
        ),
    )

    # ---- Training loop ----

    best_score        = 0
    no_improve_epochs = 0
    patience          = cfg["training"]["early_stopping_patience"]

    history = {
        "epoch":      [],
        "train_loss": [],
        "dev_re_f1":  [],
        "dev_td_acc": [],
        "score":      [],
        "lr":         []
    }

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_bar = tqdm(train_loader, desc="Training")
        loss      = trainer.train_epoch(train_bar, epoch)

        dev_bar       = tqdm(dev_loader, desc="Dev")
        re_f1, td_acc = trainer.evaluate(dev_bar)

        score      = (re_f1 + td_acc) / 2
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["dev_re_f1"].append(re_f1)
        history["dev_td_acc"].append(td_acc)
        history["score"].append(score)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch} "
            f"Loss {loss:.4f} "
            f"RE_F1 {re_f1:.4f} "
            f"TD_ACC {td_acc:.4f}"
        )

        if score > best_score:

            best_score        = score
            no_improve_epochs = 0

            path = os.path.join(cfg["system"]["save_dir"], "best_model.pt")

            save_model(path, stage1, stage2, stage3, stage4)

            print("Saved best model")

        else:

            no_improve_epochs += 1

        if no_improve_epochs >= patience:

            print("\nEarly stopping triggered")
            break

    # ---- Final evaluation ----

    print("\nLoading best model for final evaluation...")

    load_model(
        os.path.join(cfg["system"]["save_dir"], "best_model.pt"),
        stage1, stage2, stage3, stage4
    )

    print("\nEvaluating on test set...")

    test_bar      = tqdm(test_loader, desc="Test")
    re_f1, td_acc = trainer.evaluate(test_bar)

    print("\nFinal Test Results")
    print(f"RE_F1: {re_f1:.4f}")
    print(f"TD_ACC: {td_acc:.4f}")

    # ---- Save training log ----

    log_path = os.path.join(cfg["system"]["save_dir"], "training_log.json")

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print("Training history saved to:", log_path)

    # ---- Save predictions ----

    print("\nGenerating predictions...")

    test_bar = tqdm(test_loader, desc="Predict")
    preds    = trainer.predict(test_bar)

    output_path = os.path.join(cfg["system"]["save_dir"], "test_predictions.jsonl")

    save_predictions(output_path, preds)

    print("\nPredictions saved to:", output_path)
    print("\nTask done!")


if __name__ == "__main__":
    main()