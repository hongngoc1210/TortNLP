"""
td_head.py

Modules trong file này (theo thứ tự gọi trong TDHead.forward):
  1. REStatsGate              — gate H_re_p/d theo confidence của RE
  2. FeatureAttention         — self-attention trên N feature tokens [A2 existing]
  3. LabelConditionedAttention — teacher forcing signal
  4. TDMoE                   — [A3] Mixture of Experts thay thế MLP đơn lẻ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  REStatsGate
# =============================================================================

class REStatsGate(nn.Module):
    """Gate H_re_p và H_re_d theo độ tự tin của RE."""

    def __init__(self, hidden: int, stats_dim: int = 4):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(stats_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.norm_p = nn.LayerNorm(hidden)
        self.norm_d = nn.LayerNorm(hidden)

    def forward(self, H_re_p, H_re_d, stats_P, stats_D):
        gate = self.gate_mlp(torch.cat([stats_P, stats_D], dim=-1))
        return self.norm_p(H_re_p * gate), self.norm_d(H_re_d * (1.0 - gate))


# =============================================================================
#  FeatureAttention
# =============================================================================

class FeatureAttention(nn.Module):
    """
    Self-attention trên N feature tokens, pool bằng learned CLS query.
    Tokens: [H_u, H_re_p_gated, H_re_d_gated, delta, prod]
    """

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden % num_heads == 0
        self.hidden    = hidden
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(hidden, hidden * 3, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_q   = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_q, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, H = tokens.shape
        nh, d   = self.num_heads, self.head_dim

        normed  = self.norm1(tokens)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)

        q = q.view(B, N, nh, d).transpose(1, 2)
        k = k.view(B, N, nh, d).transpose(1, 2)
        v = v.view(B, N, nh, d).transpose(1, 2)

        attn    = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        attn    = self.dropout(attn)
        ctx     = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, H)

        tokens  = tokens + self.proj(ctx)
        tokens  = tokens + self.ff(self.norm2(tokens))

        cls_q   = self.cls_q.expand(B, -1, -1).view(B, 1, nh, d).transpose(1, 2)
        k2      = tokens.view(B, N, nh, d).transpose(1, 2)
        pool_attn = F.softmax(torch.matmul(cls_q, k2.transpose(-2, -1)) * self.scale, dim=-1)
        pooled  = torch.matmul(pool_attn, k2).transpose(1, 2).contiguous().view(B, H)

        return pooled


# =============================================================================
#  LabelConditionedAttention
# =============================================================================

class LabelConditionedAttention(nn.Module):
    """Teacher forcing: prepend label token vào feature sequence."""

    def __init__(self, hidden: int, stats_dim: int = 4):
        super().__init__()
        self.label_proj = nn.Linear(stats_dim * 2, hidden)
        self.norm       = nn.LayerNorm(hidden)

    def forward(self, feature_tokens, stats_P_gt, stats_D_gt):
        label_vec   = self.label_proj(torch.cat([stats_P_gt, stats_D_gt], dim=-1))
        label_token = self.norm(label_vec).unsqueeze(1)               # [B, 1, H]
        return torch.cat([label_token, feature_tokens], dim=1)        # [B, N+1, H]


# =============================================================================
#  [A3] TDMoE — Mixture of Experts
#
#  Thay thế MLP đơn lẻ trong TD head. Mỗi expert chuyên về 1 loại pattern
#  (ví dụ: negligence, defamation, nuisance). Gate được điều khiển bởi H_u
#  (global facts) — loại case được phản ánh trong undisputed facts.
#
#  Thiết kế:
#    - num_experts expert MLPs, mỗi cái nhận pooled vector
#    - Soft gate từ H_u → [B, num_experts] weights
#    - Output = weighted sum của expert outputs
#    - load_balance_loss để tránh collapse về 1 expert
# =============================================================================

class TDMoE(nn.Module):

    def __init__(
        self,
        input_dim:    int,
        hidden:       int,
        num_experts:  int   = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.num_experts = num_experts

        # Mỗi expert là 1 MLP nhỏ
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            for _ in range(num_experts)
        ])

        # Gate: H_u → soft distribution trên experts
        # Dùng H_u vì global facts mô tả loại case tốt nhất
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, num_experts),
        )

        # Noise để khuyến khích exploration trong training (Shazeer 2017)
        self.gate_noise = nn.Linear(hidden, num_experts, bias=False)

    def forward(
        self,
        x:        torch.Tensor,   # [B, input_dim]  — pooled features
        H_u:      torch.Tensor,   # [B, hidden]      — gate signal
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits          : [B]       — TD prediction logits
          load_balance_loss : scalar  — auxiliary loss để balance experts
        """
        B = x.size(0)

        # Gate scores
        gate_logits = self.gate(H_u)   # [B, num_experts]

        if training:
            # Jitter noise để tránh collapse (standard MoE trick)
            noise = torch.randn_like(gate_logits) * F.softplus(self.gate_noise(H_u))
            gate_logits = gate_logits + noise

        gate_weights = F.softmax(gate_logits, dim=-1)   # [B, E]

        # Chạy tất cả experts, stack lại
        expert_logits = torch.stack(
            [expert(x).squeeze(-1) for expert in self.experts],
            dim=1,
        )   # [B, num_experts]

        # Weighted sum
        logits = (gate_weights * expert_logits).sum(dim=1)   # [B]

        # Load balance loss: khuyến khích phân phối đều trên experts
        # = variance của mean gate weights trên batch (thấp = đều)
        mean_gate = gate_weights.mean(dim=0)   # [E]
        load_balance_loss = (mean_gate * torch.log(mean_gate + 1e-8)).sum().neg()
        # Dùng entropy của distribution — cao = đều = tốt → minimize negative entropy

        return logits, load_balance_loss


# =============================================================================
#  TDHead (updated)
#
#  Thay đổi so với version trước:
#    [A3] TDMoE thay thế MLP đơn lẻ ở bước prediction cuối
#         gate dùng H_u, load_balance_loss được trả về để thêm vào total loss
# =============================================================================

class TDHead(nn.Module):

    def __init__(
        self,
        hidden:         int,
        num_heads:      int   = 4,
        dropout:        float = 0.2,
        stats_dim:      int   = 4,
        use_label_attn: bool  = True,
        num_experts:    int   = 4,    # [A3]
    ):
        super().__init__()

        self.use_label_attn = use_label_attn

        self.re_gate = REStatsGate(hidden, stats_dim)

        if use_label_attn:
            self.label_attn = LabelConditionedAttention(hidden, stats_dim)

        self.feat_attn = FeatureAttention(hidden, num_heads, dropout)

        # [A3] MoE thay thế MLP đơn
        # input_dim = pooled (hidden) + stats (8)
        self.moe = TDMoE(
            input_dim   = hidden + stats_dim * 2,
            hidden      = hidden,
            num_experts = num_experts,
            dropout     = dropout,
        )

    def forward(
        self,
        stage1_out:  dict,
        stage3_out:  dict,
        gt_stats_P:  torch.Tensor | None = None,
        gt_stats_D:  torch.Tensor | None = None,
    ) -> dict:

        H_u     = stage1_out["H_u"]
        H_re_p  = stage3_out["H_re_p"]
        H_re_d  = stage3_out["H_re_d"]
        stats_P = stage3_out["stats_P"]
        stats_D = stage3_out["stats_D"]

        # 1. RE Stats Gate
        H_re_p_g, H_re_d_g = self.re_gate(H_re_p, H_re_d, stats_P, stats_D)

        # 2. Feature tokens
        delta  = H_re_p_g - H_re_d_g
        prod   = H_re_p_g * H_re_d_g
        tokens = torch.stack([H_u, H_re_p_g, H_re_d_g, delta, prod], dim=1)

        # 3. Label-conditioned attention (training only)
        use_label = (
            self.use_label_attn and self.training
            and gt_stats_P is not None and gt_stats_D is not None
        )
        if use_label:
            tokens = self.label_attn(tokens, gt_stats_P, gt_stats_D)

        # 4. Feature attention
        pooled = self.feat_attn(tokens)   # [B, hidden]

        # 5. [A3] MoE prediction
        x = torch.cat([pooled, stats_P, stats_D], dim=-1)

        logits, load_balance_loss = self.moe(x, H_u, training=self.training)

        probs = torch.sigmoid(logits)

        return {
            "T_logit":           logits,
            "T_hat":             probs,
            "load_balance_loss": load_balance_loss,   # [A3] cho MultiTaskLoss
        }


# =============================================================================
#  TDLoss — interface không thay đổi
# =============================================================================

class TDLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict, batch: dict) -> torch.Tensor:
        logits = outputs["T_logit"]
        labels = batch["T"].float()
        mask   = labels >= 0
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
        return self.loss_fn(logits[mask], labels[mask])