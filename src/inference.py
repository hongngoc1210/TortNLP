import os
import json
import argparse
from functools import partial

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from models.shared_encoder import Stage1Encoder
from models.re_module      import RationableExtraction
from models.pooling        import RationalePooling
from models.td_head        import TDHead


# =============================================================================
# Data — phiên bản nhẹ cho inference (không cần label)
# =============================================================================

def load_test_jsonl(path: str) -> list:
    """
    Đọc file JSONL của test set cuộc thi.
    Các trường label (court_decision, is_accepted) có thể vắng mặt hoặc null.
    """
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] line {line_no} malformed JSON, skipped: {e}")

    print(f"Loaded {len(samples)} test cases from: {path}")
    return samples


def normalize_test_case(case: dict) -> dict:
    """
    Parse một case từ JSON thô → dict chuẩn.
    Không giả định label tồn tại — mọi label đều là -1.
    Giữ lại text gốc của claims để đưa vào output.
    """

    # undisputed facts
    facts  = [f.get("description", "") for f in case.get("undisputed_facts", [])]
    U_text = " ".join(facts)

    # plaintiff claims
    P_claims = []
    for c in case.get("plaintiff_claims", []):
        P_claims.append(c.get("description", ""))

    # defendant claims
    D_claims = []
    for c in case.get("defendant_claims", []):
        D_claims.append(c.get("description", ""))

    return {
        "tort_id":  case.get("tort_id", ""),
        "U":        U_text,
        "P":        P_claims,
        "D":        D_claims,
        # Label placeholder — không dùng trong inference, cần để collate_fn tương thích
        "R_P": torch.full((len(P_claims),), -1, dtype=torch.float),
        "R_D": torch.full((len(D_claims),), -1, dtype=torch.float),
        "T":   torch.tensor(-1.0),
        # Giữ text gốc để ghi vào output
        "P_claims": P_claims,
        "D_claims": D_claims,
    }


class TestDataset(Dataset):

    def __init__(self, path: str):
        raw           = load_test_jsonl(path)
        self.samples  = [normalize_test_case(c) for c in raw]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_test(batch, tokenizer, max_len: int = 512):
    """
    Collate cho inference — giống collate_fn training nhưng
    giữ thêm P_claims/D_claims text để ghi vào output.
    """

    U_texts      = []
    P_texts      = []
    D_texts      = []
    sample_map_P = []
    sample_map_D = []
    R_P_list     = []
    R_D_list     = []
    T_list       = []
    tort_ids     = []
    P_claims_all = []   # list of list — text gốc
    D_claims_all = []

    for i, sample in enumerate(batch):

        tort_ids.append(sample["tort_id"])
        U_texts.append(sample["U"])

        P_claims_all.append(sample["P_claims"])
        D_claims_all.append(sample["D_claims"])

        for p in sample["P"]:
            P_texts.append(p)
            sample_map_P.append(i)

        r_p = sample["R_P"]
        R_P_list.extend(r_p.tolist() if torch.is_tensor(r_p) else list(r_p))

        for d in sample["D"]:
            D_texts.append(d)
            sample_map_D.append(i)

        r_d = sample["R_D"]
        R_D_list.extend(r_d.tolist() if torch.is_tensor(r_d) else list(r_d))

        t = sample["T"]
        T_list.append(t.item() if torch.is_tensor(t) else t)

    # tokenize U
    U_tok = tokenizer(
        U_texts, padding=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )

    # tokenize P
    if len(P_texts) == 0:
        P_tok = {
            "input_ids":      torch.zeros((0, max_len), dtype=torch.long),
            "attention_mask": torch.zeros((0, max_len), dtype=torch.long),
        }
    else:
        P_tok = tokenizer(
            P_texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )

    # tokenize D
    if len(D_texts) == 0:
        D_tok = {
            "input_ids":      torch.zeros((0, max_len), dtype=torch.long),
            "attention_mask": torch.zeros((0, max_len), dtype=torch.long),
        }
    else:
        D_tok = tokenizer(
            D_texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )

    return {
        "tort_id":          tort_ids,
        "U_input_ids":      U_tok["input_ids"],
        "U_attention_mask": U_tok["attention_mask"],
        "P_input_ids":      P_tok["input_ids"],
        "P_attention_mask": P_tok["attention_mask"],
        "D_input_ids":      D_tok["input_ids"],
        "D_attention_mask": D_tok["attention_mask"],
        "sample_map_P":     torch.tensor(sample_map_P, dtype=torch.long),
        "sample_map_D":     torch.tensor(sample_map_D, dtype=torch.long),
        "R_P":              torch.tensor(R_P_list, dtype=torch.float),
        "R_D":              torch.tensor(R_D_list, dtype=torch.float),
        "T":                torch.tensor(T_list,   dtype=torch.float),
        # text gốc (không phải tensor, chỉ dùng để ghi output)
        "P_claims": P_claims_all,
        "D_claims": D_claims_all,
    }


# =============================================================================
# Model loading
# =============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["lr"]           = float(cfg["training"]["lr"])
    cfg["training"]["weight_decay"] = float(cfg["training"]["weight_decay"])
    cfg["training"]["batch_size"]   = int(cfg["training"]["batch_size"])
    cfg["training"]["epochs"]       = int(cfg["training"]["epochs"])
    return cfg


def build_models(cfg: dict, device: str):
    """Khởi tạo đúng architecture theo config, chưa load weights."""

    stage1 = Stage1Encoder(
        model_name          = cfg["model"]["encoder_name"],
        cross_attn_heads    = cfg["model"].get("cross_attn_heads",    4),
        cross_attn_dropout  = cfg["model"].get("cross_attn_dropout",  0.1),
        use_cross_attention = cfg["model"].get("use_cross_attention",  True),
    ).to(device)

    hidden = stage1.encoder.hidden_size

    stage2 = RationableExtraction(hidden).to(device)
    stage3 = RationalePooling(hidden).to(device)
    stage4 = TDHead(
        hidden         = hidden,
        num_heads      = cfg["model"].get("td_num_heads",   4),
        dropout        = cfg["model"].get("td_dropout",     0.2),
        use_label_attn = cfg["model"].get("use_label_attn", True),
    ).to(device)

    return stage1, stage2, stage3, stage4


def load_checkpoint(ckpt_path: str, stage1, stage2, stage3, stage4, device: str):
    """
    Load weights từ checkpoint.
    Hỗ trợ cả best_model.pt (chỉ weights) và last_checkpoint.pt (full state).
    Tự động strip tiền tố "module." nếu checkpoint được lưu từ DDP.
    """

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # best_model.pt chỉ chứa tensors → weights_only=True an toàn
    # last_checkpoint.pt chứa optimizer state → cần weights_only=False
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    def _strip_ddp(state_dict: dict) -> dict:
        """Xoá tiền tố 'module.' do DDP thêm vào khi save."""
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k[len("module."):] if k.startswith("module.") else k
            new_sd[new_k] = v
        return new_sd

    stage1.load_state_dict(_strip_ddp(ckpt["stage1"]))
    stage2.load_state_dict(_strip_ddp(ckpt["stage2"]))
    stage3.load_state_dict(_strip_ddp(ckpt["stage3"]))
    stage4.load_state_dict(_strip_ddp(ckpt["stage4"]))

    print("Checkpoint loaded successfully.")


# =============================================================================
# Inference engine
# =============================================================================

@torch.no_grad()
def run_inference(
    stage1, stage2, stage3, stage4,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    threshold: float,
) -> list:
    """
    Chạy forward pass trên toàn bộ test set.
    Trả về list[dict] — mỗi dict là prediction cho 1 case.
    """

    stage1.eval()
    stage2.eval()
    stage3.eval()
    stage4.eval()

    results = []

    for batch in tqdm(loader, desc="Inference"):

        # move tensors to device, giữ lại non-tensor (tort_id, P_claims, D_claims)
        batch_gpu = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda", enabled=use_amp):
            s1 = stage1(batch_gpu)
            s2 = stage2(s1)
            s3 = stage3(s1, s2, batch_gpu)
            s4 = stage4(s1, s3)

        # pull predictions to CPU
        rP_hat = s2["rP_hat"].cpu()
        rD_hat = s2["rD_hat"].cpu()
        T_hat  = s4["T_hat"].cpu()

        map_P = batch_gpu["sample_map_P"].cpu()
        map_D = batch_gpu["sample_map_D"].cpu()

        for i, tort_id in enumerate(batch["tort_id"]):

            # lấy RE scores của case i
            p_idx = (map_P == i).nonzero(as_tuple=True)[0]
            d_idx = (map_D == i).nonzero(as_tuple=True)[0]

            rP_i = rP_hat[p_idx].tolist()
            rD_i = rD_hat[d_idx].tolist()
            T_i  = float(T_hat[i])

            results.append({
                "tort_id":  tort_id,
                # TD prediction
                "T_hat":    round(T_i, 6),
                "T_pred":   int(T_i >= threshold),
                # RE predictions — plaintiff
                "rP_hat":   [round(v, 6) for v in rP_i],
                "rP_pred":  [int(v >= threshold) for v in rP_i],
                # RE predictions — defendant
                "rD_hat":   [round(v, 6) for v in rD_i],
                "rD_pred":  [int(v >= threshold) for v in rD_i],
                # claim texts gốc để đối chiếu
                "P_claims": batch["P_claims"][i],
                "D_claims": batch["D_claims"][i],
            })

    return results


# =============================================================================
# Output helpers
# =============================================================================

def save_jsonl(path: str, records: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} predictions → {path}")


def save_submission(path: str, records: list):
    """
    Format nộp bài tối giản: chỉ tort_id + T_pred + rP_pred + rD_pred.
    Không có claim text (nhẹ hơn để upload).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "tort_id": r["tort_id"],
                "T_pred":  r["T_pred"],
                "rP_pred": r["rP_pred"],
                "rD_pred": r["rD_pred"],
            }, ensure_ascii=False) + "\n")
    print(f"Saved submission ({len(records)} cases) → {path}")


def print_summary(results: list, threshold: float):
    """In thống kê nhanh về predictions."""

    n          = len(results)
    n_pos      = sum(r["T_pred"] for r in results)
    n_neg      = n - n_pos
    avg_conf   = sum(r["T_hat"] for r in results) / max(n, 1)

    rP_counts  = [sum(r["rP_pred"]) for r in results]
    rD_counts  = [sum(r["rD_pred"]) for r in results]
    avg_rP     = sum(rP_counts) / max(n, 1)
    avg_rD     = sum(rD_counts) / max(n, 1)

    print("\n" + "=" * 50)
    print("Inference Summary")
    print("=" * 50)
    print(f"  Total cases        : {n}")
    print(f"  TD threshold       : {threshold}")
    print(f"  T_pred=1 (plaintiff wins) : {n_pos} ({n_pos/max(n,1)*100:.1f}%)")
    print(f"  T_pred=0 (defendant wins) : {n_neg} ({n_neg/max(n,1)*100:.1f}%)")
    print(f"  Avg T_hat (confidence)    : {avg_conf:.4f}")
    print(f"  Avg accepted P claims     : {avg_rP:.2f}")
    print(f"  Avg accepted D claims     : {avg_rD:.2f}")
    print("=" * 50)


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(description="COLIEE Tort inference")

    parser.add_argument(
        "--test-path",   required=True,
        help="Path tới file test JSONL của cuộc thi",
    )
    parser.add_argument(
        "--ckpt",        required=True,
        help="Path tới checkpoint (best_model.pt hoặc last_checkpoint.pt)",
    )
    parser.add_argument(
        "--config",      default="config/config.yaml",
        help="Path tới config.yaml (mặc định: config/config.yaml)",
    )
    parser.add_argument(
        "--output-path", default=None,
        help="Path output JSONL đầy đủ (mặc định: <ckpt_dir>/inference_output.jsonl)",
    )
    parser.add_argument(
        "--submission-path", default=None,
        help="Path file nộp bài tối giản (mặc định: <ckpt_dir>/submission.jsonl)",
    )
    parser.add_argument(
        "--batch-size",  type=int, default=None,
        help="Override batch size từ config",
    )
    parser.add_argument(
        "--threshold",   type=float, default=0.5,
        help="Ngưỡng để chuyển xác suất → nhị phân (mặc định: 0.5)",
    )
    parser.add_argument(
        "--device",      default=None,
        help="'cuda', 'cuda:0', 'cpu', ... (mặc định: tự detect)",
    )
    parser.add_argument(
        "--no-amp",      action="store_true",
        help="Tắt AMP (mixed precision) — dùng khi chạy CPU",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
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

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    # ---- output paths ----
    ckpt_dir = os.path.dirname(args.ckpt) or "."

    output_path     = args.output_path     or os.path.join(ckpt_dir, "inference_output.jsonl")
    submission_path = args.submission_path or os.path.join(ckpt_dir, "submission.jsonl")

    # ---- tokenizer ----
    print(f"\nLoading tokenizer: {cfg['model']['encoder_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["encoder_name"],
        use_fast=True,
    )

    # ---- dataset + loader ----
    print(f"\nLoading test data: {args.test_path}")
    dataset = TestDataset(args.test_path)

    loader = DataLoader(
        dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,        # inference không shuffle
        num_workers = args.num_workers,
        pin_memory  = ("cuda" in device),
        collate_fn  = partial(collate_test, tokenizer=tokenizer),
    )

    # ---- models ----
    print("\nBuilding models...")
    stage1, stage2, stage3, stage4 = build_models(cfg, device)

    # ---- load checkpoint ----
    load_checkpoint(args.ckpt, stage1, stage2, stage3, stage4, device)

    # ---- inference ----
    print(f"\nRunning inference on {len(dataset)} cases...")
    results = run_inference(
        stage1, stage2, stage3, stage4,
        loader, device, use_amp,
        threshold=args.threshold,
    )

    # ---- save ----
    save_jsonl(output_path, results)
    save_submission(submission_path, results)

    # ---- summary ----
    print_summary(results, args.threshold)


if __name__ == "__main__":
    main()