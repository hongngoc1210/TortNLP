import torch
import torch.nn as nn

from .encode_text             import SharedEncoder
from .conditioning            import FiLMConditioner
from .cross_attn         import ClaimsCrossAttentionLayer
from .claim_selfattn    import ClaimSelfAttentionLayer   # [A1]


class Stage1Encoder(nn.Module):
    """
    Stage 1 pipeline (theo thứ tự):
      1. Encode U  →  H_u
      2. Encode P/D claims (chunked)
      3. FiLM conditioning với H_u
      4. [A1] Claim self-attention  (mỗi phía attend nội bộ)
      5. P↔D cross-attention        (hai phía attend lẫn nhau)
    """

    def __init__(
        self,
        model_name:             str   = "sbintuitions/modernbert-ja-310m",
        claim_chunk_size:       int   = 64,
        # cross-attention config
        cross_attn_heads:       int   = 4,
        cross_attn_dropout:     float = 0.1,
        use_cross_attention:    bool  = True,
        # [A1] self-attention config
        self_attn_heads:        int   = 4,
        self_attn_dropout:      float = 0.1,
        use_self_attention:     bool  = True,
    ):
        super().__init__()

        self.encoder     = SharedEncoder(model_name)
        hidden           = self.encoder.hidden_size
        self.conditioner = FiLMConditioner(hidden)

        # [A1] claim self-attention (after FiLM, before cross-attn)
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_attn = ClaimSelfAttentionLayer(
                hidden     = hidden,
                num_heads  = self_attn_heads,
                dropout    = self_attn_dropout,
            )

        # P↔D cross-attention
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = ClaimsCrossAttentionLayer(
                hidden     = hidden,
                num_heads  = cross_attn_heads,
                dropout    = cross_attn_dropout,
            )

        self.claim_chunk_size = claim_chunk_size

    # -------------------------------------------------------------------------

    def encode_chunked(self, input_ids, attention_mask):

        if input_ids.size(0) == 0:
            return torch.zeros(
                0, self.encoder.hidden_size,
                device=input_ids.device,
            )

        outputs = []
        chunk   = self.claim_chunk_size

        for i in range(0, input_ids.size(0), chunk):
            emb = self.encoder(input_ids[i:i+chunk], attention_mask[i:i+chunk])
            outputs.append(emb)

        return torch.cat(outputs, dim=0)

    # -------------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:

        # 1. Encode U
        H_u = self.encoder(batch["U_input_ids"], batch["U_attention_mask"])

        # 2. Encode claims
        h_P = self.encode_chunked(batch["P_input_ids"], batch["P_attention_mask"])
        h_D = self.encode_chunked(batch["D_input_ids"], batch["D_attention_mask"])

        # 3. FiLM conditioning
        hP_cond = self.conditioner(h_P, H_u, batch["sample_map_P"])
        hD_cond = self.conditioner(h_D, H_u, batch["sample_map_D"])

        stage1_out = {
            "H_u": H_u, "h_P": h_P, "h_D": h_D,
            "hP_cond": hP_cond, "hD_cond": hD_cond,
        }

        # 4. [A1] Claim self-attention (P attend P, D attend D)
        if self.use_self_attention:
            stage1_out = self.self_attn(stage1_out, batch)

        # 5. P↔D cross-attention (P attend D, D attend P)
        if self.use_cross_attention:
            stage1_out = self.cross_attn(stage1_out, batch)

        return stage1_out