import os
import json
import argparse
import inspect
from copy import deepcopy
from functools import partial

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from models.factory import build_model_stages


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

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge dict đệ quy; giá trị trong override được ưu tiên."""
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def resolve_config_path(config_arg: str | None, ckpt_path: str) -> str:
    """
    Chọn config theo thứ tự:
      1. --config do người dùng truyền;
      2. resolved_config.yaml nằm cùng checkpoint;
      3. config/final_v2.yaml;
      4. config/config.yaml.

    Final V2 phải được build bằng đúng resolved config để các task adapter
    (re_adapter/tp_adapter) tồn tại trước khi load state_dict.
    """

    if config_arg:
        if not os.path.exists(config_arg):
            raise FileNotFoundError(f"Config không tìm thấy: {config_arg}")
        return config_arg

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidates = [
        os.path.join(ckpt_dir, "resolved_config.yaml"),
        os.path.join(ckpt_dir, "config.yaml"),
        "config/final_v2.yaml",
        "config/config.yaml",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Không tự tìm thấy config. Hãy truyền --config, ví dụ:\n"
        "  --config config/final_v2.yaml"
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config không hợp lệ hoặc rỗng: {path}")

    # Chỉ ép kiểu những trường thực sự tồn tại.
    training = cfg.setdefault("training", {})

    for key in ("lr", "weight_decay", "encoder_lr_multiplier"):
        if key in training and training[key] is not None:
            training[key] = float(training[key])

    for key in (
        "batch_size",
        "grad_accum_steps",
        "epochs",
        "phase1_epochs",
        "phase2_epochs",
    ):
        if key in training and training[key] is not None:
            training[key] = int(training[key])

    return cfg


def read_checkpoint(ckpt_path: str) -> dict:
    """
    Đọc checkpoint một lần trên CPU. Sau đó checkpoint vừa được dùng để
    hiệu chỉnh config, vừa được dùng để load weights.
    """

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt_path}")

    print(f"Reading checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint phải là dict, nhận được: {type(ckpt).__name__}"
        )

    required = ("stage1", "stage2", "stage3", "stage4")
    missing = [key for key in required if key not in ckpt]

    if missing:
        raise KeyError(
            "Checkpoint thiếu state_dict: "
            + ", ".join(missing)
            + f". Các key hiện có: {sorted(ckpt.keys())}"
        )

    return ckpt


def _strip_ddp(state_dict: dict) -> dict:
    """Xoá tiền tố module. do DistributedDataParallel thêm khi save."""
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _set_architecture_option(cfg: dict, key: str, value) -> None:
    """
    Cập nhật mọi vị trí đã có key và đồng thời ghi vào các section thường
    được factory của Final V2 sử dụng. Các key dư trong YAML không ảnh hưởng
    tới model nếu factory không đọc section đó.
    """

    found = False

    def visit(node):
        nonlocal found

        if not isinstance(node, dict):
            return

        if key in node:
            node[key] = value
            found = True

        for child in node.values():
            if isinstance(child, dict):
                visit(child)

    visit(cfg)

    # Bảo đảm tương thích với các cách tổ chức config cũ/mới.
    cfg[key] = value

    for section_name in ("model", "ablation", "architecture", "final_v2"):
        cfg.setdefault(section_name, {})[key] = value

    if found:
        print(f"  Config override from checkpoint: {key}={value}")


def _infer_adapter_bottleneck(*state_dicts: dict) -> int | None:
    """
    Adapter down projection thường có shape:
        [adapter_bottleneck, hidden_size]
    """

    candidate_suffixes = (
        "re_adapter.down.weight",
        "tp_adapter.down.weight",
        "adapter.down.weight",
    )

    for state_dict in state_dicts:
        for key, tensor in state_dict.items():
            if (
                any(key.endswith(suffix) for suffix in candidate_suffixes)
                and torch.is_tensor(tensor)
                and tensor.ndim == 2
            ):
                return int(tensor.shape[0])

    return None


def adapt_config_to_checkpoint(cfg: dict, ckpt: dict) -> dict:
    """
    Đồng bộ các option kiến trúc có thể suy ra chắc chắn từ checkpoint.

    Lỗi cũ xảy ra vì checkpoint có re_adapter.* nhưng inference build Stage 2
    bằng config mặc định không bật task adapters.
    """

    # Nếu full checkpoint có lưu config, ưu tiên config lúc train.
    for config_key in ("resolved_config", "config", "cfg"):
        embedded = ckpt.get(config_key)
        if isinstance(embedded, dict):
            print(f"Using config embedded in checkpoint key: {config_key}")
            cfg = _deep_merge(cfg, embedded)
            break

    stage2_sd = _strip_ddp(ckpt["stage2"])
    stage3_sd = _strip_ddp(ckpt["stage3"])
    stage4_sd = _strip_ddp(ckpt["stage4"])

    all_keys = tuple(stage2_sd) + tuple(stage3_sd) + tuple(stage4_sd)

    has_re_adapter = any(
        key.startswith("re_adapter.") or ".re_adapter." in key
        for key in all_keys
    )
    has_tp_adapter = any(
        key.startswith("tp_adapter.") or ".tp_adapter." in key
        for key in all_keys
    )

    if has_re_adapter or has_tp_adapter:
        _set_architecture_option(cfg, "use_task_adapters", True)

        bottleneck = _infer_adapter_bottleneck(
            stage2_sd,
            stage3_sd,
            stage4_sd,
        )
        if bottleneck is not None:
            _set_architecture_option(
                cfg,
                "adapter_bottleneck",
                bottleneck,
            )

    return cfg


def build_models(cfg: dict, device: str):
    """Khởi tạo architecture theo resolved config/checkpoint."""

    return build_model_stages(cfg, device=device)


def _load_one_stage(
    stage_name: str,
    module: torch.nn.Module,
    state_dict: dict,
) -> None:
    """
    Load strict để không âm thầm bỏ qua adapter hoặc layer đã được train.
    In chẩn đoán rõ hơn nếu config vẫn không khớp checkpoint.
    """

    state_dict = _strip_ddp(state_dict)

    model_keys = set(module.state_dict())
    checkpoint_keys = set(state_dict)

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)

    if missing or unexpected:
        details = [f"{stage_name} architecture không khớp checkpoint."]

        if missing:
            details.append(
                "Missing keys trong checkpoint:\n  - "
                + "\n  - ".join(missing[:30])
            )

        if unexpected:
            details.append(
                "Unexpected keys trong checkpoint:\n  - "
                + "\n  - ".join(unexpected[:30])
            )

        if any("re_adapter." in key for key in unexpected):
            details.append(
                "Checkpoint có RE adapter nhưng model inference chưa tạo "
                "re_adapter. Hãy dùng resolved_config.yaml của lần train "
                "hoặc config/final_v2.yaml."
            )

        if any("tp_adapter." in key for key in unexpected):
            details.append(
                "Checkpoint có TP adapter nhưng model inference chưa tạo "
                "tp_adapter. Hãy dùng resolved_config.yaml của lần train "
                "hoặc config/final_v2.yaml."
            )

        raise RuntimeError("\n".join(details))

    module.load_state_dict(state_dict, strict=True)


def load_checkpoint(
    ckpt: dict,
    stage1,
    stage2,
    stage3,
    stage4,
):
    """Load weights đã đọc sẵn vào bốn stage."""

    print("Loading checkpoint weights...")

    _load_one_stage("stage1", stage1, ckpt["stage1"])
    _load_one_stage("stage2", stage2, ckpt["stage2"])
    _load_one_stage("stage3", stage3, ckpt["stage3"])
    _load_one_stage("stage4", stage4, ckpt["stage4"])

    print("Checkpoint loaded successfully.")


def _find_config_value(cfg: dict, key: str, default=None):
    """Tìm key đệ quy trong config."""

    if key in cfg:
        return cfg[key]

    for value in cfg.values():
        if isinstance(value, dict):
            found = _find_config_value(value, key, default=None)
            if found is not None:
                return found

    return default


def _call_stage4(stage4, s1, s3, input_mode: str):
    """
    Final V2 mới hỗ trợ input_mode; vẫn tương thích implementation cũ không
    có keyword này.
    """

    parameters = inspect.signature(stage4.forward).parameters

    if "input_mode" in parameters:
        return stage4(s1, s3, input_mode=input_mode)

    return stage4(s1, s3)


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
    tp_input_mode: str = "rationale",
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

        device_type = "cuda" if "cuda" in device else "cpu"
        with torch.amp.autocast(device_type, enabled=use_amp):
            s1 = stage1(batch_gpu)
            s2 = stage2(s1)
            s3 = stage3(s1, s2, batch_gpu)
            s4 = _call_stage4(
                stage4,
                s1,
                s3,
                input_mode=tp_input_mode,
            )

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
        "--config",      default=None,
        help=(
            "Path tới config. Nếu bỏ qua, script ưu tiên "
            "<ckpt_dir>/resolved_config.yaml rồi config/final_v2.yaml."
        ),
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

    # ---- checkpoint + config ----
    ckpt = read_checkpoint(args.ckpt)

    config_path = resolve_config_path(args.config, args.ckpt)
    print(f"Config  : {config_path}")

    cfg = load_config(config_path)
    cfg = adapt_config_to_checkpoint(cfg, ckpt)

    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size

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

    inference_batch_size = int(
        cfg.get("training", {}).get("batch_size", 1)
    )
    max_len = int(_find_config_value(cfg, "max_len", default=512))

    print(f"Inference batch size: {inference_batch_size}")
    print(f"Tokenizer max length: {max_len}")

    loader = DataLoader(
        dataset,
        batch_size  = inference_batch_size,
        shuffle     = False,        # inference không shuffle
        num_workers = args.num_workers,
        pin_memory  = ("cuda" in device),
        collate_fn  = partial(
            collate_test,
            tokenizer=tokenizer,
            max_len=max_len,
        ),
    )

    # ---- models ----
    print("\nBuilding models...")
    stage1, stage2, stage3, stage4 = build_models(cfg, device)

    # ---- load checkpoint ----
    load_checkpoint(ckpt, stage1, stage2, stage3, stage4)

    # ---- inference ----
    print(f"\nRunning inference on {len(dataset)} cases...")
    tp_input_mode = str(
        _find_config_value(cfg, "tp_input_mode", default="rationale")
    )

    results = run_inference(
        stage1, stage2, stage3, stage4,
        loader, device, use_amp,
        threshold=args.threshold,
        tp_input_mode=tp_input_mode,
    )

    # ---- save ----
    save_jsonl(output_path, results)
    save_submission(submission_path, results)

    # ---- summary ----
    print_summary(results, args.threshold)


if __name__ == "__main__":
    main()