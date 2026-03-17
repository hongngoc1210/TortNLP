"""
multitask_loss.py

Tích hợp tất cả losses:
  L_RE          : [B1] Focal Loss cho RE
  L_TD          : [B2] Asymmetric Loss cho TD
  L_contrastive : [A2] Supervised Contrastive Loss từ pooling
  L_moe_balance : [A3] Load Balance Loss từ MoE
  L_consistency : [B4] Consistency loss — RE phải consistent với TD

[B3] Dynamic λ weighting dùng Uncertainty Weighting (Kendall et al. 2018):
  L = (1/2σ_re²)*L_RE + log(σ_re)
    + (1/2σ_td²)*L_TD + log(σ_td)
  Model tự học σ_re, σ_td — không cần tune tay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .re_loss import RELoss
from .td_loss import TDLoss


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        # Fixed weights (dùng khi uncertainty_weighting=False)
        lambda_re:            float = 2.0,   # tăng lên 2.0 để bù focal scale thấp
        lambda_td:            float = 1.0,
        lambda_contrastive:   float = 0.1,   # [A2]
        lambda_moe:           float = 0.01,  # [A3] nhỏ — chỉ balance
        lambda_consistency:   float = 0.3,   # [B4] giảm xuống để không át RE
        # [B3] Uncertainty weighting — để False khi RE chưa ổn định
        # Bật lại sau khi RE đã học được (RE_F1 > 0.1)
        uncertainty_weighting: bool = False,
        # Loss configs
        focal_gamma:   float = 0.5,   # [B1] giảm thêm → tránh loss collapse
        focal_alpha:   float = 0.75,  # [B1] up-weight positives (minority)
        asl_gamma_pos: float = 0.0,   # [B2]
        asl_gamma_neg: float = 2.0,   # [B2] giảm từ 4.0 → tránh loss_td → 0
        asl_clip:      float = 0.0,   # [B2] tắt clip → tránh easy neg vanish
        # [B4] consistency margin
        consistency_margin: float = 0.1,
    ):
        super().__init__()

        self.re_loss = RELoss(gamma=focal_gamma, alpha=focal_alpha)
        self.td_loss = TDLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)

        self.lambda_re          = lambda_re
        self.lambda_td          = lambda_td
        self.lambda_contrastive = lambda_contrastive
        self.lambda_moe         = lambda_moe
        self.lambda_consistency = lambda_consistency
        self.consistency_margin = consistency_margin

        # [B3] Uncertainty weighting: học log(σ²) để tránh σ âm
        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting:
            # log_var = log(σ²); σ² > 0 luôn đảm bảo qua exp
            self.log_var_re = nn.Parameter(torch.zeros(1))
            self.log_var_td = nn.Parameter(torch.zeros(1))

        # Counter để chỉ in NaN warning 1 lần, không spam mỗi step
        self._nan_warned = False

    # -------------------------------------------------------------------------
    # [B4] Consistency loss
    # -------------------------------------------------------------------------

    def _consistency_loss(
        self,
        re_outputs: dict,
        td_outputs: dict,
        batch:      dict,
    ) -> torch.Tensor:
        """
        Nếu T=1 (plaintiff thắng): mean(rP_hat) phải > mean(rD_hat) + margin
        Nếu T=0 (defendant thắng): mean(rD_hat) phải > mean(rP_hat) + margin
        Chỉ tính trên case có T label hợp lệ.
        """
        T        = batch["T"].float()
        valid_T  = T >= 0

        if not valid_T.any():
            return torch.tensor(0.0, device=T.device)

        rP_hat   = re_outputs["rP_hat"]
        rD_hat   = re_outputs["rD_hat"]
        map_P    = batch["sample_map_P"]
        map_D    = batch["sample_map_D"]
        B        = T.size(0)

        # Mean RE score per case [B]
        mean_rP = torch.zeros(B, device=T.device)
        mean_rD = torch.zeros(B, device=T.device)

        for i in range(B):
            idx_P = (map_P == i).nonzero(as_tuple=True)[0]
            idx_D = (map_D == i).nonzero(as_tuple=True)[0]
            if len(idx_P) > 0:
                mean_rP[i] = rP_hat[idx_P].mean()
            if len(idx_D) > 0:
                mean_rD[i] = rD_hat[idx_D].mean()

        # expected_sign: +1 nếu T=1, -1 nếu T=0
        expected_sign = 2 * T - 1   # [B]
        score_gap     = mean_rP - mean_rD   # [B]

        # Hinge loss: penalty khi score_gap không đúng chiều
        loss = F.relu(
            self.consistency_margin - expected_sign * score_gap
        )

        return loss[valid_T].mean()

    # -------------------------------------------------------------------------

    def forward(
        self,
        re_outputs: dict,
        td_outputs: dict,
        batch:      dict,
    ) -> tuple:

        loss_re  = self.re_loss(re_outputs, batch)
        loss_td  = self.td_loss(td_outputs, batch)

        # [A2] Contrastive loss từ pooling output
        loss_contrastive = td_outputs.get(
            "loss_contrastive",
            torch.tensor(0.0, device=loss_td.device)
        )
        # Nếu pooling không trả về, thử lấy từ re_outputs (tùy routing)
        if loss_contrastive.item() == 0.0:
            loss_contrastive = re_outputs.get(
                "loss_contrastive",
                torch.tensor(0.0, device=loss_re.device)
            )

        # [A3] MoE load balance loss từ TD output
        loss_moe = td_outputs.get(
            "load_balance_loss",
            torch.tensor(0.0, device=loss_td.device)
        )

        # [B4] Consistency loss
        loss_consistency = self._consistency_loss(re_outputs, td_outputs, batch)

        # [B3] Weighting
        if self.uncertainty_weighting:
            # L = (1/2σ_re²)*L_RE + log(σ_re) + (1/2σ_td²)*L_TD + log(σ_td)
            # log_var = log(σ²) → 1/σ² = exp(-log_var), log(σ) = 0.5*log_var
            loss = (
                torch.exp(-self.log_var_re) * loss_re  + 0.5 * self.log_var_re
              + torch.exp(-self.log_var_td) * loss_td  + 0.5 * self.log_var_td
            )
        else:
            loss = self.lambda_re * loss_re + self.lambda_td * loss_td

        # Auxiliary losses với fixed weights (không scale bằng uncertainty)
        loss = (
            loss
            + self.lambda_contrastive * loss_contrastive   # [A2]
            + self.lambda_moe         * loss_moe            # [A3]
            + self.lambda_consistency * loss_consistency    # [B4]
        )

        # NaN/Inf guard — fallback về core losses, chỉ log 1 lần
        if torch.isnan(loss) or torch.isinf(loss):
            if not self._nan_warned:
                nan_or_inf = 'NaN' if torch.isnan(loss) else 'Inf'
                print(
                    f"[MultiTaskLoss] {nan_or_inf} in auxiliary losses "
                    f"(loss_re={loss_re.item():.4f}, loss_td={loss_td.item():.4f}, "
                    f"loss_cons={loss_consistency.item():.4f}). "
                    f"Falling back to core losses. (warning suppressed after this)"
                )
                self._nan_warned = True
            loss = self.lambda_re * loss_re + self.lambda_td * loss_td

        return loss, loss_re, loss_td