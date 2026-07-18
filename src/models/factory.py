"""Factories for constructing the four CAER-MTL stages from config.yaml."""

from __future__ import annotations

from typing import Optional

from .pooling import RationalePooling
from .re_module import RationableExtraction
from .shared_encoder import Stage1Encoder
from .td_head import TDHead


def build_model_stages(cfg: dict, device: Optional[str] = None):
    model_cfg = cfg.get("model", {})
    ablation_cfg = cfg.get("ablation", {})

    stage1 = Stage1Encoder(
        model_name=model_cfg.get(
            "encoder_name", "sbintuitions/modernbert-ja-310m"
        ),
        claim_chunk_size=model_cfg.get("claim_chunk_size", 64),
        num_heads=model_cfg.get(
            "fusion_heads", model_cfg.get("cross_attn_heads", 8)
        ),
        topk_fact_tokens=model_cfg.get("topk_fact_tokens", 16),
        topk_opponents=model_cfg.get("topk_opponents", 3),
        dropout=model_cfg.get(
            "fusion_dropout", model_cfg.get("cross_attn_dropout", 0.1)
        ),
    )

    hidden = stage1.encoder.hidden_size
    use_task_adapters = bool(ablation_cfg.get("use_task_adapters", False))
    adapter_bottleneck = int(ablation_cfg.get("adapter_bottleneck", 128))
    adapter_dropout = float(ablation_cfg.get("adapter_dropout", 0.1))

    stage2 = RationableExtraction(
        hidden,
        use_task_adapter=use_task_adapters,
        adapter_bottleneck=adapter_bottleneck,
        adapter_dropout=adapter_dropout,
        head_dropout=float(model_cfg.get("re_dropout", 0.1)),
    )

    stage3 = RationalePooling(
        hidden=hidden,
        topk_claims=model_cfg.get("topk_claims", 5),
        dropout=model_cfg.get("aggregation_dropout", 0.1),
        use_task_adapter=use_task_adapters,
        adapter_bottleneck=adapter_bottleneck,
        adapter_dropout=adapter_dropout,
        detach_rationale_for_tp=bool(
            ablation_cfg.get("detach_rationale_for_tp", False)
        ),
    )

    stage4 = TDHead(
        hidden=hidden,
        num_heads=model_cfg.get("td_num_heads", 8),
        dropout=model_cfg.get("td_dropout", 0.2),
        input_mode=ablation_cfg.get("tp_input_mode", "rationale"),
        use_global_residual=bool(
            ablation_cfg.get("use_global_residual", False)
        ),
        rationale_scale_init=float(
            ablation_cfg.get("rationale_scale_init", -1.5)
        ),
    )

    if device is not None:
        stage1 = stage1.to(device)
        stage2 = stage2.to(device)
        stage3 = stage3.to(device)
        stage4 = stage4.to(device)

    return stage1, stage2, stage3, stage4
