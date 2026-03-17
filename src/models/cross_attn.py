import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  ClaimCrossAttention
#  - Mỗi claim của một phía (P hoặc D) attend sang toàn bộ claims của phía kia
#  - Xử lý variable-length claims thông qua sample_map (không cần padding)
#  - Output: h_out = h_in + CrossAttn(Q=h_in, K/V=h_other)  (residual)
# =============================================================================

class ClaimCrossAttention(nn.Module):

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):

        super().__init__()

        assert hidden % num_heads == 0, \
            f"hidden ({hidden}) must be divisible by num_heads ({num_heads})"

        self.hidden    = hidden
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # Projection cho Q (từ phía mình), K/V (từ phía đối diện)
        self.q_proj  = nn.Linear(hidden, hidden, bias=False)
        self.kv_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

        self.dropout = nn.Dropout(dropout)

        # LayerNorm sau residual
        self.norm = nn.LayerNorm(hidden)

    # -------------------------------------------------------------------------
    # _attend_one_case:
    #   h_q   : [M, hidden]  — claims của phía "self"
    #   h_kv  : [N, hidden]  — claims của phía "other"
    #   trả về context vector cùng shape [M, hidden]
    # -------------------------------------------------------------------------

    def _attend_one_case(
        self,
        h_q: torch.Tensor,
        h_kv: torch.Tensor,
    ) -> torch.Tensor:

        M = h_q.size(0)
        N = h_kv.size(0)
        H = self.num_heads
        d = self.head_dim

        # Nếu phía kia không có claim nào thì trả về zero context
        if N == 0:
            return torch.zeros_like(h_q)

        # ----- project -----
        q = self.q_proj(h_q)           # [M, hidden]
        kv = self.kv_proj(h_kv)        # [N, 2*hidden]
        k, v = kv.chunk(2, dim=-1)     # [N, hidden] each

        # ----- reshape to multi-head -----
        # [1, H, M, d]  — batch_size=1 vì đây là 1 case
        q = q.view(M, H, d).unsqueeze(0).transpose(1, 2)   # [1, H, M, d]
        k = k.view(N, H, d).unsqueeze(0).transpose(1, 2)   # [1, H, N, d]
        v = v.view(N, H, d).unsqueeze(0).transpose(1, 2)   # [1, H, N, d]

        # ----- scaled dot-product attention -----
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [1, H, M, N]
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        ctx = torch.matmul(attn, v)    # [1, H, M, d]
        ctx = ctx.transpose(1, 2).contiguous().view(M, self.hidden)  # [M, hidden]

        return self.out_proj(ctx)

    # -------------------------------------------------------------------------
    # forward:
    #   Duyệt từng case trong batch theo sample_map, tránh padding-noise
    # -------------------------------------------------------------------------

    def forward(
        self,
        h_self:   torch.Tensor,   # [total_claims_self,  hidden]
        h_other:  torch.Tensor,   # [total_claims_other, hidden]
        map_self:  torch.Tensor,  # [total_claims_self]   — case index
        map_other: torch.Tensor,  # [total_claims_other]  — case index
        batch_size: int,
    ) -> torch.Tensor:
        """
        Trả về h_out: [total_claims_self, hidden]
        h_out[i] = LayerNorm( h_self[i] + CrossAttn(h_self[i] → h_other của cùng case) )
        """

        device = h_self.device
        out = torch.zeros_like(h_self)

        for case_id in range(batch_size):

            idx_self  = (map_self  == case_id).nonzero(as_tuple=True)[0]
            idx_other = (map_other == case_id).nonzero(as_tuple=True)[0]

            if len(idx_self) == 0:
                continue

            h_q  = h_self[idx_self]                          # [M, hidden]
            h_kv = h_other[idx_other] if len(idx_other) > 0 \
                   else torch.zeros(0, self.hidden, device=device)

            ctx = self._attend_one_case(h_q, h_kv)          # [M, hidden]

            # residual + norm
            out[idx_self] = self.norm(h_q + ctx)

        return out


# =============================================================================
#  ClaimsCrossAttentionLayer
#  Wrapper tiện gọi: xử lý cả hai chiều P→D và D→P cùng lúc
# =============================================================================

class ClaimsCrossAttentionLayer(nn.Module):

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):

        super().__init__()

        # Hai module độc lập (weight không share) để P→D và D→P học khác nhau
        self.cross_P2D = ClaimCrossAttention(hidden, num_heads, dropout)
        self.cross_D2P = ClaimCrossAttention(hidden, num_heads, dropout)

    def forward(self, stage1_out: dict, batch: dict) -> dict:
        """
        Nhận stage1_out từ Stage1Encoder, trả về dict mới với
        hP_cond và hD_cond đã được cross-attend.

        Keys thêm vào output:
          hP_ctx  — hP sau cross-attention (P→D)
          hD_ctx  — hD sau cross-attention (D→P)
        (hP_cond / hD_cond được ghi đè bằng giá trị mới để RE dùng)
        """

        hP = stage1_out["hP_cond"]   # [N_P, hidden]
        hD = stage1_out["hD_cond"]   # [N_D, hidden]

        map_P = batch["sample_map_P"]
        map_D = batch["sample_map_D"]
        batch_size = batch["U_input_ids"].size(0)

        # P attend sang D  →  P_ctx
        hP_ctx = self.cross_P2D(
            h_self=hP, h_other=hD,
            map_self=map_P, map_other=map_D,
            batch_size=batch_size,
        )

        # D attend sang P  →  D_ctx
        hD_ctx = self.cross_D2P(
            h_self=hD, h_other=hP,
            map_self=map_D, map_other=map_P,
            batch_size=batch_size,
        )

        # Build output — copy stage1_out và override hP/hD_cond
        out = {**stage1_out}
        out["hP_cond"] = hP_ctx
        out["hD_cond"] = hD_ctx
        out["hP_ctx"]  = hP_ctx   # alias nếu cần debug
        out["hD_ctx"]  = hD_ctx

        return out