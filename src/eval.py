import os
import argparse
from functools import partial

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils          import collate_fn, seed_worker
from data_utils.dataset  import LegalDataset
from data_utils.preprocessing import build_dataset
from data_utils.split    import train_dev_test_split

from models.factory import build_model_stages

from trainer.metrics import compute_re_f1, compute_td_accuracy


# =============================================================================
# load_config
# =============================================================================

def load_config(path: str) -> dict:

    with open(path) as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["lr"]           = float(cfg["training"]["lr"])
    cfg["training"]["weight_decay"] = float(cfg["training"]["weight_decay"])
    cfg["training"]["batch_size"]   = int(cfg["training"]["batch_size"])
    cfg["training"]["epochs"]       = int(cfg["training"]["epochs"])

    return cfg


# =============================================================================
# build_models
# =============================================================================

def build_models(cfg: dict, device: str):

    return build_model_stages(cfg, device=device)


# =============================================================================
# load_checkpoint
# =============================================================================

def load_checkpoint(ckpt_path: str, stage1, stage2, stage3, stage4, device: str):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    def _strip_ddp(state_dict: dict) -> dict:
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    stage1.load_state_dict(_strip_ddp(ckpt["stage1"]))
    stage2.load_state_dict(_strip_ddp(ckpt["stage2"]))
    stage3.load_state_dict(_strip_ddp(ckpt["stage3"]))
    stage4.load_state_dict(_strip_ddp(ckpt["stage4"]))

    print("Checkpoint loaded successfully.")


# =============================================================================
# build_loader
# =============================================================================

def build_loader(cfg: dict, split: str) -> DataLoader:

    samples = build_dataset(cfg["data"]["train_path"])

    train_samples, dev_samples, test_samples = train_dev_test_split(samples)

    split_map = {
        "train": train_samples,
        "dev":   dev_samples,
        "test":  test_samples,
    }

    if split not in split_map:
        raise ValueError(f"split phải là 'train', 'dev', hoặc 'test', nhận: {split}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    dataset = LegalDataset(split_map[split])

    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    loader = DataLoader(
        dataset,
        batch_size     = cfg["training"]["batch_size"],
        shuffle        = False,
        num_workers    = cfg["system"]["num_workers"],
        pin_memory     = True,
        collate_fn     = partial(collate_fn, tokenizer=tokenizer),
        worker_init_fn = seed_worker,
        generator      = g,
    )

    return loader


# =============================================================================
# evaluate
# =============================================================================

@torch.no_grad()
def evaluate(stage1, stage2, stage3, stage4, loader, device: str, use_amp: bool) -> dict:

    stage1.eval()
    stage2.eval()
    stage3.eval()
    stage4.eval()

    re_preds_P, re_labels_P = [], []
    re_preds_D, re_labels_D = [], []
    td_preds,   td_labels   = [], []

    for batch in tqdm(loader, desc="Evaluating"):

        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda", enabled=use_amp):

            s1 = stage1(batch)
            s2 = stage2(s1)
            s3 = stage3(s1, s2, batch)
            s4 = stage4(s1, s3)

        re_preds_P.append(s2["rP_hat"].cpu())
        re_preds_D.append(s2["rD_hat"].cpu())
        re_labels_P.append(batch["R_P"].cpu())
        re_labels_D.append(batch["R_D"].cpu())

        td_preds.append(s4["T_hat"].cpu())
        td_labels.append(batch["T"].cpu())

    re_preds_P  = torch.cat(re_preds_P)
    re_preds_D  = torch.cat(re_preds_D)
    re_labels_P = torch.cat(re_labels_P)
    re_labels_D = torch.cat(re_labels_D)

    td_preds  = torch.cat(td_preds)
    td_labels = torch.cat(td_labels)

    # ---- tính metrics dùng API cũ ----

    re_f1_P   = compute_re_f1(re_preds_P,  re_labels_P)
    re_f1_D   = compute_re_f1(re_preds_D,  re_labels_D)
    re_f1_all = compute_re_f1(
        torch.cat([re_preds_P,  re_preds_D]),
        torch.cat([re_labels_P, re_labels_D]),
    )
    re_macro  = (re_f1_P + re_f1_D) / 2.0

    td_acc = compute_td_accuracy(td_preds, td_labels)

    return {
        "tp_accuracy":           td_acc,
        "re_f1_true":            re_f1_all,
        "re_macro_f1":           re_macro,
        "re_f1_true_plaintiff":  re_f1_P,
        "re_macro_f1_plaintiff": re_f1_P,
        "re_f1_true_defendant":  re_f1_D,
        "re_macro_f1_defendant": re_f1_D,
        "num_samples":           td_labels.numel(),
        "score":                 (re_macro + td_acc) / 2.0,
    }


# =============================================================================
# print_metrics
# =============================================================================

def print_metrics(tag: str, metrics: dict):

    print(f"\n{'=' * 50}")
    print(tag)
    print('=' * 50)
    print(f"  tp_accuracy             : {metrics['tp_accuracy']:.6f}")
    print(f"  re_f1_true              : {metrics['re_f1_true']:.6f}")
    print(f"  re_macro_f1             : {metrics['re_macro_f1']:.6f}")
    print(f"  re_f1_true_plaintiff    : {metrics['re_f1_true_plaintiff']:.6f}")
    print(f"  re_macro_f1_plaintiff   : {metrics['re_macro_f1_plaintiff']:.6f}")
    print(f"  re_f1_true_defendant    : {metrics['re_f1_true_defendant']:.6f}")
    print(f"  re_macro_f1_defendant   : {metrics['re_macro_f1_defendant']:.6f}")
    print(f"  num_samples             : {metrics['num_samples']}")
    print(f"  score (composite)       : {metrics['score']:.6f}")
    print('=' * 50)


# =============================================================================
# main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(description="Evaluate best_model.pt")

    parser.add_argument(
        "--ckpt",    required=True,
        help="Path tới checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--config",  default="config/config.yaml",
        help="Path tới config.yaml",
    )
    parser.add_argument(
        "--split",   default="test",
        choices=["train", "dev", "test"],
        help="Split để evaluate (mặc định: test)",
    )
    parser.add_argument(
        "--device",  default=None,
        help="'cuda', 'cpu', ... (mặc định: tự detect)",
    )
    parser.add_argument(
        "--no-amp",  action="store_true",
        help="Tắt AMP — dùng khi chạy CPU",
    )

    args = parser.parse_args()

    # ---- device ----

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_amp = (not args.no_amp) and ("cuda" in device)

    print(f"Device  : {device}")
    print(f"Use AMP : {use_amp}")

    # ---- config ----

    cfg = load_config(args.config)

    # ---- loader ----

    print(f"\nLoading '{args.split}' split...")

    loader = build_loader(cfg, args.split)

    print(f"Samples : {len(loader.dataset)}")

    # ---- models ----

    print("\nBuilding models...")

    stage1, stage2, stage3, stage4 = build_models(cfg, device)

    # ---- checkpoint ----

    load_checkpoint(args.ckpt, stage1, stage2, stage3, stage4, device)

    # ---- evaluate ----

    metrics = evaluate(stage1, stage2, stage3, stage4, loader, device, use_amp)

    print_metrics(f"Evaluation — split: {args.split}", metrics)


if __name__ == "__main__":
    main()