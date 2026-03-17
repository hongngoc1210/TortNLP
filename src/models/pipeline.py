"""
pipeline.py
Full model wrapper kết nối tất cả stages:

  Stage 1: Stage1Encoder      (Enc + FiLM + P↔D CrossAttn)
  Stage 2: RationableExtraction (RE heads)
  Stage 3: RationalePooling    (robust soft pooling)
  Stage 4: TDHead              (FeatureAttention + REStatsGate + LabelAttn + MLP)

Teacher forcing (Stage B training):
  Khi truyền gt_rP / gt_rD vào forward(), pooling dùng mixed signal
  r̃ = η*r_gt + (1-η)*r̂  và TDHead nhận label stats từ ground-truth.
  η giảm dần epoch-by-epoch (caller tự set pipeline.eta).
"""

import math
import torch
import torch.nn as nn

from .shared_encoder import Stage1Encoder
from .re_module       import RationableExtraction
from .pooling         import RationalePooling
from .td_head         import TDHead, TDLoss


class LegalPipeline(nn.Module):

    def __init__(
        self,
        model_name:          str   = "sbintuitions/modernbert-ja-310m",
        # Stage 1
        claim_chunk_size:    int   = 64,
        cross_attn_heads:    int   = 4,
        cross_attn_dropout:  float = 0.1,
        use_cross_attention: bool  = True,
        # Stage 3
        pool_tau:            float = 1.0,
        # Stage 4
        td_num_heads:        int   = 4,
        td_dropout:          float = 0.2,
        use_label_attn:      bool  = True,
        # teacher forcing
        eta:                 float = 1.0,   # 1.0 = full teacher, 0.0 = model only
    ):
        super().__init__()

        # --- Stage 1 ---
        self.stage1 = Stage1Encoder(
            model_name          = model_name,
            claim_chunk_size    = claim_chunk_size,
            cross_attn_heads    = cross_attn_heads,
            cross_attn_dropout  = cross_attn_dropout,
            use_cross_attention = use_cross_attention,
        )

        hidden = self.stage1.encoder.hidden_size

        # --- Stage 2 ---
        self.stage2 = RationableExtraction(hidden)

        # --- Stage 3 ---
        self.stage3 = RationalePooling(hidden, tau=pool_tau)

        # --- Stage 4 ---
        self.stage4 = TDHead(
            hidden         = hidden,
            num_heads      = td_num_heads,
            dropout        = td_dropout,
            use_label_attn = use_label_attn,
        )

        # teacher forcing mixing coefficient (caller sets this each epoch)
        self.eta = eta

    # -------------------------------------------------------------------------
    # _mixed_rationale:
    #   Tạo r̃ = η * r_gt + (1-η) * r_hat để dùng trong pooling
    #   r_gt và r_hat phải cùng shape và cùng device
    # -------------------------------------------------------------------------

    def _mixed_rationale(
        self,
        r_hat: torch.Tensor,
        r_gt:  torch.Tensor | None,
    ) -> torch.Tensor:

        if r_gt is None or not self.training or self.eta == 0.0:
            return r_hat

        # r_gt có thể là binary {0,1} hoặc soft — đều ok
        return self.eta * r_gt.float() + (1.0 - self.eta) * r_hat

    # -------------------------------------------------------------------------
    # _gt_stats:
    #   Tính RE statistics từ ground-truth rationale (cho label_attn)
    # -------------------------------------------------------------------------

    @staticmethod
    def _gt_stats(
        r_gt:       torch.Tensor,   # [N_claims]  binary
        sample_map: torch.Tensor,   # [N_claims]  case index
        batch_size: int,
        device:     torch.device,
    ) -> torch.Tensor:
        """
        Trả về [B, 4] stats (max, mean, sum, entropy) từ ground-truth rationale.
        Dùng làm input cho LabelConditionedAttention.
        """

        stats = torch.zeros(batch_size, 4, device=device)

        for case_id in range(batch_size):

            idx = (sample_map == case_id).nonzero(as_tuple=True)[0]

            if len(idx) == 0:
                continue

            r = r_gt[idx].float()

            # treat binary r as "soft" weights (all equal if all 0/1)
            r_sum = r.sum().clamp(min=1e-8)
            w     = r / r_sum
            w     = w.clamp(min=1e-8)

            stats[case_id, 0] = r.max()
            stats[case_id, 1] = r.mean()
            stats[case_id, 2] = r.sum()
            stats[case_id, 3] = -(w * w.log()).sum()   # entropy

        return stats

    # -------------------------------------------------------------------------

    def forward(
        self,
        batch:  dict,
        # ground-truth rationale (optional, chỉ dùng khi training Stage B/C)
        gt_rP:  torch.Tensor | None = None,   # [N_P_claims] binary
        gt_rD:  torch.Tensor | None = None,   # [N_D_claims] binary
    ) -> dict:

        device     = batch["U_input_ids"].device
        batch_size = batch["U_input_ids"].size(0)

        # ------------------------------------------------------------------ #
        # Stage 1: encode + FiLM + P↔D cross-attention                       #
        # ------------------------------------------------------------------ #

        s1 = self.stage1(batch)

        # ------------------------------------------------------------------ #
        # Stage 2: RE heads                                                   #
        # ------------------------------------------------------------------ #

        s2 = self.stage2(s1)

        # ------------------------------------------------------------------ #
        # Teacher forcing: trộn r̂ và r_gt cho pooling                        #
        # ------------------------------------------------------------------ #

        r_P_pool = self._mixed_rationale(s2["rP_hat"], gt_rP)
        r_D_pool = self._mixed_rationale(s2["rD_hat"], gt_rD)

        # Build mixed stage2 output để truyền vào pooling
        s2_pool = {**s2, "rP_hat": r_P_pool, "rD_hat": r_D_pool}

        # ------------------------------------------------------------------ #
        # Stage 3: robust soft pooling                                        #
        # ------------------------------------------------------------------ #

        s3 = self.stage3(s1, s2_pool, batch)

        # ------------------------------------------------------------------ #
        # Stage 4: TD prediction                                              #
        #   Tính gt_stats cho LabelConditionedAttention nếu có gt_rP/gt_rD   #
        # ------------------------------------------------------------------ #

        gt_stats_P = None
        gt_stats_D = None

        if self.training and gt_rP is not None and gt_rD is not None:
            gt_stats_P = self._gt_stats(
                gt_rP, batch["sample_map_P"], batch_size, device
            )
            gt_stats_D = self._gt_stats(
                gt_rD, batch["sample_map_D"], batch_size, device
            )

        s4 = self.stage4(
            stage1_out = s1,
            stage3_out = s3,
            gt_stats_P = gt_stats_P,
            gt_stats_D = gt_stats_D,
        )

        # ------------------------------------------------------------------ #
        # Merge all outputs                                                   #
        # ------------------------------------------------------------------ #

        return {
            # RE predictions
            "logits_P": s2["logits_P"],
            "logits_D": s2["logits_D"],
            "rP_hat":   s2["rP_hat"],
            "rD_hat":   s2["rD_hat"],
            # TD predictions
            "T_logit":  s4["T_logit"],
            "T_hat":    s4["T_hat"],
            # Pooled representations (useful for debugging)
            "H_re_p":   s3["H_re_p"],
            "H_re_d":   s3["H_re_d"],
        }


# =============================================================================
#  CombinedLoss
#  λ_RE * L_RE + λ_T * L_TD
# =============================================================================

class CombinedLoss(nn.Module):

    def __init__(self, lambda_re: float = 0.5, lambda_td: float = 0.5):

        super().__init__()

        self.lambda_re = lambda_re
        self.lambda_td = lambda_td

        self.bce = nn.BCEWithLogitsLoss()
        self.td_loss = TDLoss()

    def forward(self, outputs: dict, batch: dict) -> dict:

        # RE loss (plaintiff)
        loss_re_p = torch.tensor(0.0, device=outputs["T_logit"].device)
        if outputs["logits_P"].numel() > 0 and "rP" in batch:
            loss_re_p = self.bce(outputs["logits_P"], batch["rP"].float())

        # RE loss (defendant)
        loss_re_d = torch.tensor(0.0, device=outputs["T_logit"].device)
        if outputs["logits_D"].numel() > 0 and "rD" in batch:
            loss_re_d = self.bce(outputs["logits_D"], batch["rD"].float())

        loss_re = loss_re_p + loss_re_d

        # TD loss
        loss_td = self.td_loss(outputs, batch)

        total = self.lambda_re * loss_re + self.lambda_td * loss_td

        return {
            "loss":      total,
            "loss_re":   loss_re,
            "loss_re_p": loss_re_p,
            "loss_re_d": loss_re_d,
            "loss_td":   loss_td,
        }