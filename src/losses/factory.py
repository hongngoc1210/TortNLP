from .multitask_loss import MultiTaskLoss


def build_multitask_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    ablation_cfg = cfg.get("ablation", {})
    return MultiTaskLoss(
        weight_re=float(loss_cfg.get("weight_re", 0.33)),
        weight_tp=float(loss_cfg.get("weight_tp", 0.67)),
        task_mode=ablation_cfg.get("task_mode", "joint"),
        re_side_reduction=loss_cfg.get("re_side_reduction", "mean"),
    )
