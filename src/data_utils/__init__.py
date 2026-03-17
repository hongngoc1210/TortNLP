import torch
import numpy as np


# =============================================================================
# Seed worker — đảm bảo DataLoader reproducible với num_workers > 0
# =============================================================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


# =============================================================================
# collate_fn
# =============================================================================

def collate_fn(batch, tokenizer, max_len=512):
    """
    FIX #18: R_P/R_D từ LegalDataset đã là tensor → dùng torch.stack/cat
             thay vì torch.tensor() lại (sẽ raise error với list of tensors).
    """

    U_texts    = []
    P_texts    = []
    D_texts    = []

    sample_map_P = []
    sample_map_D = []

    R_P_list   = []
    R_D_list   = []

    T_list     = []
    tort_ids   = []

    for i, sample in enumerate(batch):

        tort_ids.append(sample["tort_id"])
        U_texts.append(sample["U"])

        # ---------- plaintiff ----------
        for p in sample["P"]:
            P_texts.append(p)
            sample_map_P.append(i)

        # FIX: sample["R_P"] có thể là tensor hoặc list — chuẩn hoá về list float
        r_p = sample["R_P"]
        if torch.is_tensor(r_p):
            R_P_list.extend(r_p.tolist())
        else:
            R_P_list.extend(list(r_p))

        # ---------- defendant ----------
        for d in sample["D"]:
            D_texts.append(d)
            sample_map_D.append(i)

        r_d = sample["R_D"]
        if torch.is_tensor(r_d):
            R_D_list.extend(r_d.tolist())
        else:
            R_D_list.extend(list(r_d))

        # ---------- verdict ----------
        t = sample["T"]
        T_list.append(t.item() if torch.is_tensor(t) else t)

    # ---------- tokenize U ----------

    U_tok = tokenizer(
        U_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # ---------- tokenize P ----------

    if len(P_texts) == 0:
        P_tok = {
            "input_ids":      torch.zeros((0, max_len), dtype=torch.long),
            "attention_mask": torch.zeros((0, max_len), dtype=torch.long),
        }
    else:
        P_tok = tokenizer(
            P_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    # ---------- tokenize D ----------

    if len(D_texts) == 0:
        D_tok = {
            "input_ids":      torch.zeros((0, max_len), dtype=torch.long),
            "attention_mask": torch.zeros((0, max_len), dtype=torch.long),
        }
    else:
        D_tok = tokenizer(
            D_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    return {
        "tort_id": tort_ids,

        "U_input_ids":      U_tok["input_ids"],
        "U_attention_mask": U_tok["attention_mask"],

        "P_input_ids":      P_tok["input_ids"],
        "P_attention_mask": P_tok["attention_mask"],

        "D_input_ids":      D_tok["input_ids"],
        "D_attention_mask": D_tok["attention_mask"],

        "sample_map_P": torch.tensor(sample_map_P, dtype=torch.long),
        "sample_map_D": torch.tensor(sample_map_D, dtype=torch.long),

        # FIX: build từ plain Python list → không bao giờ raise
        "R_P": torch.tensor(R_P_list, dtype=torch.float),
        "R_D": torch.tensor(R_D_list, dtype=torch.float),
        "T":   torch.tensor(T_list,   dtype=torch.float),
    }