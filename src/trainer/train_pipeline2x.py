import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from data_utils.preprocessing import build_dataset
from data_utils.dataset       import LegalDataset
from data_utils               import collate_fn, seed_worker
from data_utils.split         import train_dev_test_split

from models.shared_encoder import Stage1Encoder
from models.re_module      import RationableExtraction
from models.pooling        import RationalePooling
from models.td_head        import TDHead

from losses.multitask_loss import MultiTaskLoss
from trainer.engine        import Trainer
from trainer.scheduler     import TeacherForcingScheduler


# =============================================================================
# build_loaders
# =============================================================================

def build_loaders(cfg, rank: int, world_size: int):
    """
    - train_loader : DistributedSampler (mỗi GPU thấy 1/world_size data)
    - dev_loader   : không sampler → rank 0 evaluate toàn bộ dev set
    - test_loader  : không sampler → rank 0 evaluate toàn bộ test set
    """

    samples = build_dataset(cfg["data"]["train_path"])
    train_samples, dev_samples, test_samples = train_dev_test_split(samples)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["encoder_name"],
        use_fast=True,
    )

    train_dataset = LegalDataset(train_samples)
    dev_dataset   = LegalDataset(dev_samples)
    test_dataset  = LegalDataset(test_samples)

    # BUG 11 FIX: seed_worker + generator để reproducible
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42) + rank)   # seed khác nhau mỗi GPU

    num_workers = cfg["system"].get("num_workers", 2)

    collate = partial(collate_fn, tokenizer=tokenizer)

    # ---- train loader với DistributedSampler ----
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,           # shuffle=False khi dùng sampler
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # ---- dev/test loader không cần sampler (chỉ rank 0 dùng) ----
    eval_common = dict(
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=g,
    )

    dev_loader  = DataLoader(dev_dataset,  **eval_common)
    test_loader = DataLoader(test_dataset, **eval_common)

    return train_loader, dev_loader, test_loader


# =============================================================================
# build_models  — trả về model gốc (chưa DDP wrap)
# =============================================================================

def build_models(cfg, device):
    """
    BUG 7 FIX: truyền đầy đủ params mới cho Stage1Encoder và TDHead.
    Trả về raw models — DDP wrapping được làm ở main_2x.py sau khi
    sync BatchNorm (nếu có) và kiểm soát find_unused_parameters.
    """

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


# =============================================================================
# build_optimizer_and_scheduler  — layer-wise LR + cosine warmup
# =============================================================================

def build_optimizer_and_scheduler(
    stage1, stage2, stage3, stage4,
    cfg,
    num_steps: int,
    ddp: bool = True,
):
    """
    BUG 6 FIX: layer-wise LR + cosine warmup (giống train_pipeline.py).

    Khi DDP=True, model đã được wrap → params nằm trong .module
    Hàm này luôn nhận raw model (trước wrap) hoặc sau wrap đều OK
    vì chúng ta dùng .parameters() trực tiếp.
    """

    lr     = cfg["training"]["lr"]
    wd     = cfg["training"]["weight_decay"]
    warmup = cfg["training"].get("warmup_ratio", 0.06)

    # Lấy raw model nếu đã wrap DDP
    def _raw(m):
        return m.module if hasattr(m, "module") else m

    encoder_params    = list(_raw(stage1).encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}

    cross_attn_params = []
    if hasattr(_raw(stage1), "cross_attn"):
        cross_attn_params = list(_raw(stage1).cross_attn.parameters())
    cross_attn_ids = {id(p) for p in cross_attn_params}

    other_params = [
        p for m in [stage1, stage2, stage3, stage4]
        for p in m.parameters()
        if id(p) not in encoder_param_ids and id(p) not in cross_attn_ids
    ]

    param_groups = [
        {"params": encoder_params,    "lr": lr * 0.1, "name": "encoder"},
        {"params": cross_attn_params, "lr": lr,       "name": "cross_attn"},
        {"params": other_params,      "lr": lr,       "name": "heads"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    num_warmup = int(num_steps * warmup)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = num_warmup,
        num_training_steps = num_steps,
    )

    return optimizer, lr_scheduler