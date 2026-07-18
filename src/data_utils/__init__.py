import torch
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch, tokenizer, max_len=512):

    U_texts = []
    P_texts = []
    D_texts = []

    sample_map_P = []
    sample_map_D = []

    R_P = []
    R_D = []

    T = []
    tort_ids = []

    for i, sample in enumerate(batch):

        tort_ids.append(sample["tort_id"])

        U_texts.append(sample["U"])

        # ---------- plaintiff ----------
        for p in sample["P"]:
            P_texts.append(p)
            sample_map_P.append(i)

        R_P.extend(sample["R_P"])

        # ---------- defendant ----------
        for d in sample["D"]:
            D_texts.append(d)
            sample_map_D.append(i)

        R_D.extend(sample["R_D"])

        T.append(sample["T"])

    # ---------- tokenize U ----------

    U_tok = tokenizer(
        U_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
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
            return_tensors="pt"
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
            return_tensors="pt"
        )

    # ---------- tensor labels ----------

    R_P = torch.tensor(R_P, dtype=torch.float)
    R_D = torch.tensor(R_D, dtype=torch.float)

    T = torch.tensor(T, dtype=torch.float)

    # ---------- output ----------

    batch_out = {

        "tort_id": tort_ids,

        "U_input_ids":      U_tok["input_ids"],
        "U_attention_mask": U_tok["attention_mask"],

        "P_input_ids":      P_tok["input_ids"],
        "P_attention_mask": P_tok["attention_mask"],

        "D_input_ids":      D_tok["input_ids"],
        "D_attention_mask": D_tok["attention_mask"],

        "sample_map_P": torch.tensor(sample_map_P, dtype=torch.long),
        "sample_map_D": torch.tensor(sample_map_D, dtype=torch.long),

        "R_P": R_P,
        "R_D": R_D,

        "T": T
    }

    return batch_out