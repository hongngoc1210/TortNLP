"""
claim_self_attention.py  [A1]

Mỗi claim attend sang các claims cùng phía TRƯỚC khi đưa vào RE head.
Lý do: trong văn bản pháp lý, các claims có quan hệ nhân quả với nhau —
claim sau thường dựa trên claim trước. Self-attention học được cấu trúc này.

Vị trí trong pipeline:
  FiLM conditioning → [A1] ClaimSelfAttention → P↔D CrossAttention → RE head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClaimSelfAttention(nn.Module):
    """
    Self-attention trên tất cả claims của 1 phía (P hoặc D).
    Xử lý variable-length claims qua sample_map (không padding).
    Output: h_out[i] = LayerNorm(h_in[i] + SelfAttn(h_in[i] → tất cả claims cùng case))
    """

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):

        super().__init__()

        assert hidden % num_heads == 0

        self.hidden    = hidden
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.qkv      = nn.Linear(hidden, hidden * 3, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

        self.norm    = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def _attend_one_case(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [N, hidden]  — tất cả claims của 1 case
        return : [N, hidden]  — context vectors
        """
        N  = h.size(0)
        nh = self.num_heads
        d  = self.head_dim

        qkv = self.qkv(h)                              # [N, 3H]
        q, k, v = qkv.chunk(3, dim=-1)                # [N, H] each

        q = q.view(N, nh, d).unsqueeze(0).transpose(1, 2)   # [1, nh, N, d]
        k = k.view(N, nh, d).unsqueeze(0).transpose(1, 2)
        v = v.view(N, nh, d).unsqueeze(0).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [1, nh, N, N]
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        ctx = torch.matmul(attn, v)                    # [1, nh, N, d]
        ctx = ctx.transpose(1, 2).contiguous().view(N, self.hidden)

        return self.out_proj(ctx)

    def forward(
        self,
        h:          torch.Tensor,   # [total_claims, hidden]
        sample_map: torch.Tensor,   # [total_claims]  — case index
        batch_size: int,
    ) -> torch.Tensor:
        """
        Trả về h_out: [total_claims, hidden]
        Mỗi claim được cập nhật bằng context từ các claims cùng case.
        """
        out = torch.zeros_like(h)

        for case_id in range(batch_size):
            idx = (sample_map == case_id).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            h_case = h[idx]                           # [N, hidden]
            ctx    = self._attend_one_case(h_case)    # [N, hidden]
            out[idx] = self.norm(h_case + ctx)        # residual + norm

        return out


class ClaimSelfAttentionLayer(nn.Module):
    """
    Wrapper áp dụng self-attention cho cả P và D.
    Được gọi trong Stage1Encoder sau FiLM, trước CrossAttention.
    """

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Weight không share giữa P và D — hai phía có cấu trúc argument khác nhau
        self.self_attn_P = ClaimSelfAttention(hidden, num_heads, dropout)
        self.self_attn_D = ClaimSelfAttention(hidden, num_heads, dropout)

    def forward(self, stage1_out: dict, batch: dict) -> dict:
        """
        Nhận stage1_out (sau FiLM), trả về dict mới với hP_cond/hD_cond đã
        được self-attend. Override in-place để pipeline downstream không thay đổi.
        """
        batch_size = batch["U_input_ids"].size(0)

        hP_updated = self.self_attn_P(
            stage1_out["hP_cond"],
            batch["sample_map_P"],
            batch_size,
        )
        hD_updated = self.self_attn_D(
            stage1_out["hD_cond"],
            batch["sample_map_D"],
            batch_size,
        )

        return {**stage1_out, "hP_cond": hP_updated, "hD_cond": hD_updated}