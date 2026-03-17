import torch
from torch.utils.data import DataLoader
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

from trainer.engine    import Trainer
from trainer.scheduler import TeacherForcingScheduler


# =============================================================================
# build_loaders
# =============================================================================

def build_loaders(cfg):

    samples = build_dataset(cfg["data"]["train_path"])

    train_samples, dev_samples, test_samples = train_dev_test_split(samples)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_dataset = LegalDataset(train_samples)
    dev_dataset   = LegalDataset(dev_samples)
    test_dataset  = LegalDataset(test_samples)

    # FIX #19: seed_worker + generator để DataLoader reproducible
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    common = dict(
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["system"]["num_workers"],
        pin_memory  = True,
        collate_fn  = partial(collate_fn, tokenizer=tokenizer),
        worker_init_fn = seed_worker,
        generator      = g,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **common)
    dev_loader   = DataLoader(dev_dataset,   shuffle=False, **common)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **common)

    return train_loader, dev_loader, test_loader


# =============================================================================
# build_optimizer_and_scheduler
# =============================================================================

def build_optimizer_and_scheduler(stage1, stage2, stage3, stage4, cfg, num_steps, loss_fn=None):
    """
    FIX #9: Layer-wise learning rates
      - encoder backbone  : lr × 0.1   (pretrained, fine-tune conservatively)
      - cross-attention   : lr × 1.0   (new module, train normally)
      - RE / pooling / TD : lr × 1.0
    FIX #5: Warmup + cosine decay scheduler
    """

    lr     = cfg["training"]["lr"]
    wd     = cfg["training"]["weight_decay"]
    warmup = cfg["training"].get("warmup_ratio", 0.06)

    # --- collect param groups ---
    encoder_params    = list(stage1.encoder.parameters())
    encoder_param_ids = set(id(p) for p in encoder_params)

    cross_attn_params = []
    if hasattr(stage1, "cross_attn"):
        cross_attn_params = list(stage1.cross_attn.parameters())

    other_params = [
        p for m in [stage1, stage2, stage3, stage4]
        for p in m.parameters()
        if id(p) not in encoder_param_ids
        and not any(p is cp for cp in cross_attn_params)
    ]

    # [B3] log_var params từ uncertainty weighting — lr lớn hơn vì hội tụ nhanh
    loss_params = []
    if loss_fn is not None:
        loss_params = [p for p in loss_fn.parameters()]

    param_groups = [
        {"params": encoder_params,    "lr": lr * 0.1, "name": "encoder"},
        {"params": cross_attn_params, "lr": lr,       "name": "cross_attn"},
        {"params": other_params,      "lr": lr,       "name": "heads"},
        {"params": loss_params,       "lr": lr * 10,  "name": "loss_vars"},  # [B3]
    ]

    # drop empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    # warmup + cosine decay
    num_warmup = int(num_steps * warmup)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps    = num_warmup,
        num_training_steps  = num_steps,
    )

    return optimizer, lr_scheduler


# =============================================================================
# train_one_config  (helper, dùng trong sweep / hparam search)
# =============================================================================

def train_one_config(cfg):

    device = cfg["system"]["device"]

    train_loader, dev_loader, test_loader = build_loaders(cfg)

    # FIX #8: truyền đầy đủ config cho Stage1Encoder
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
    stage3 = RationalePooling(
        hidden               = hidden,
        contrastive_temp     = cfg.get("loss", {}).get("contrastive_temp",     0.07),  # [A2]
        use_contrastive_loss = cfg.get("loss", {}).get("use_contrastive_loss", True),  # [A2]
    ).to(device)
    stage4 = TDHead(
        hidden          = hidden,
        num_heads       = cfg["model"].get("td_num_heads",    4),
        dropout         = cfg["model"].get("td_dropout",      0.2),
        use_label_attn  = cfg["model"].get("use_label_attn",  True),
        num_experts     = cfg["model"].get("num_experts",      4),   # [A3]
    ).to(device)

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

    epochs     = cfg["training"]["epochs"]
    num_steps  = epochs * len(train_loader)

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        stage1, stage2, stage3, stage4, cfg, num_steps, loss_fn=loss_fn
    )

    tf_scheduler = TeacherForcingScheduler(
        start  = cfg["teacher_forcing"]["start"],
        end    = cfg["teacher_forcing"]["end"],
        epochs = cfg["teacher_forcing"]["epochs"],
    )

    trainer = Trainer(
        stage1, stage2, stage3, stage4,
        loss_fn, optimizer, tf_scheduler,
        lr_scheduler  = lr_scheduler,
        device        = device,
        grad_accum_steps = cfg["training"].get("grad_accum_steps", 1),
        max_grad_norm    = cfg["training"].get("max_grad_norm", 1.0),
    )

    for epoch in range(epochs):
        trainer.train_epoch(train_loader, epoch)

    metrics = trainer.evaluate(dev_loader)

    return metrics