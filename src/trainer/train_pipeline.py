import torch
from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from data_utils.preprocessing import build_dataset
from data_utils.dataset       import LegalDataset
from data_utils               import collate_fn, seed_worker
from data_utils.split         import train_dev_test_split

from models.factory import build_model_stages
from losses.factory import build_multitask_loss

from trainer.engine    import Trainer
from trainer.scheduler import TeacherForcingScheduler


# =============================================================================
# build_loaders
# =============================================================================

def build_loaders(cfg):

    samples = build_dataset(cfg["data"]["train_path"])

    train_samples, dev_samples, test_samples = train_dev_test_split(
        samples, seed=int(cfg.get("seed", 42))
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_dataset = LegalDataset(train_samples)
    dev_dataset   = LegalDataset(dev_samples)
    test_dataset  = LegalDataset(test_samples)

    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    common = dict(
        batch_size     = cfg["training"]["batch_size"],
        num_workers    = cfg["system"]["num_workers"],
        pin_memory     = True,
        collate_fn     = partial(collate_fn, tokenizer=tokenizer),
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

def build_optimizer_and_scheduler(stage1, stage2, stage3, stage4, cfg, num_steps):
    """Build AdamW with conservative encoder fine-tuning.

    ``num_steps`` must be the number of *optimizer updates*, not the number of
    raw batches.  With gradient accumulation this is:
        epochs * ceil(len(train_loader) / grad_accum_steps)
    """

    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    warmup = float(cfg["training"].get("warmup_ratio", 0.06))
    encoder_lr_multiplier = float(
        cfg["training"].get("encoder_lr_multiplier", 0.1)
    )

    encoder_params = [p for p in stage1.encoder.parameters() if p.requires_grad]
    encoder_param_ids = set(id(p) for p in encoder_params)

    other_params = [
        p
        for module in [stage1, stage2, stage3, stage4]
        for p in module.parameters()
        if p.requires_grad and id(p) not in encoder_param_ids
    ]

    param_groups = [
        {
            "params": encoder_params,
            "lr": lr * encoder_lr_multiplier,
            "name": "encoder",
        },
        {"params": other_params, "lr": lr, "name": "heads"},
    ]
    param_groups = [group for group in param_groups if group["params"]]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    num_steps = max(1, int(num_steps))
    num_warmup = int(num_steps * warmup)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_steps,
    )
    return optimizer, lr_scheduler


# =============================================================================
# train_one_config  (helper, dùng trong sweep / hparam search)
# =============================================================================

def train_one_config(cfg):

    device = cfg["system"]["device"]

    train_loader, dev_loader, test_loader = build_loaders(cfg)

    stage1, stage2, stage3, stage4 = build_model_stages(cfg, device=device)

    loss_fn = build_multitask_loss(cfg)

    epochs    = cfg["training"]["epochs"]
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    updates_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    num_steps = epochs * updates_per_epoch

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        stage1, stage2, stage3, stage4, cfg, num_steps
    )

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

    for epoch in range(epochs):
        trainer.train_epoch(train_loader, epoch)

    re_f1, td_acc = trainer.evaluate(dev_loader)

    return re_f1, td_acc