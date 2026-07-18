"""Stage 1: Claim-Centric Evidence Fusion."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import masked_softmax, valid_num_heads
from .encode_text import SharedEncoder


class GlobalFactPool(nn.Module):
    """Create the global case representation H_u from fact-token states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        fact_tokens: torch.Tensor,
        fact_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(fact_tokens)).squeeze(-1)
        weights = masked_softmax(scores, fact_mask, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * fact_tokens, dim=1)
        return self.output_norm(pooled), weights


class ClaimEvidenceFusion(nn.Module):
    """
    Fuse four sources for each claim:
      1. its own encoded semantics,
      2. top-k fact-token evidence,
      3. top-k opposing claims,
      4. the global case representation H_u.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        topk_fact_tokens: int = 16,
        topk_opponents: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.topk_fact_tokens = max(1, int(topk_fact_tokens))
        self.topk_opponents = max(1, int(topk_opponents))

        heads = valid_num_heads(hidden_size, num_heads)

        self.fact_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fact_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fact_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.opp_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.opp_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.opp_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.component_proj = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(4)]
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _retrieve_fact_tokens(
        self,
        claims: torch.Tensor,
        fact_tokens: torch.Tensor,
        fact_mask: torch.Tensor,
        sample_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k token evidence from the corresponding U sequence."""

        num_claims = claims.size(0)
        if num_claims == 0:
            return (
                claims.new_zeros((0, self.hidden_size)),
                torch.empty((0, 0), dtype=torch.long, device=claims.device),
            )

        case_tokens = fact_tokens[sample_map]
        case_mask = fact_mask[sample_map]

        query = F.normalize(self.fact_query(claims), dim=-1)
        keys = F.normalize(self.fact_key(case_tokens), dim=-1)
        similarity = torch.einsum("nh,nlh->nl", query, keys)
        similarity = similarity.masked_fill(~case_mask, -1e4)

        k = min(self.topk_fact_tokens, case_tokens.size(1))
        top_scores, top_indices = similarity.topk(k=k, dim=-1)

        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        selected = case_tokens.gather(dim=1, index=gather_index)
        selected_mask = case_mask.gather(dim=1, index=top_indices)

        # At least one token is normally valid because U contains special tokens.
        safe_mask = selected_mask.clone()
        no_valid = ~safe_mask.any(dim=1)
        if no_valid.any():
            safe_mask[no_valid, 0] = True
            selected = selected.clone()
            selected[no_valid, 0] = 0.0

        attended, _ = self.fact_attention(
            query=claims.unsqueeze(1),
            key=selected,
            value=selected,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        attended = attended.squeeze(1)
        attended = torch.where(
            selected_mask.any(dim=1, keepdim=True),
            attended,
            torch.zeros_like(attended),
        )
        return attended, top_indices

    def _retrieve_opponents(
        self,
        claims: torch.Tensor,
        opponents: torch.Tensor,
        claim_map: torch.Tensor,
        opponent_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and attend to the top-k opposing claims in the same case."""

        num_claims = claims.size(0)
        evidence = claims.new_zeros((num_claims, self.hidden_size))
        indices = torch.full(
            (num_claims, self.topk_opponents),
            -1,
            dtype=torch.long,
            device=claims.device,
        )

        if num_claims == 0 or opponents.size(0) == 0:
            return evidence, indices

        outputs = []
        output_indices = []

        q_all = F.normalize(self.opp_query(claims), dim=-1)
        k_all = F.normalize(self.opp_key(opponents), dim=-1)

        for claim_idx in range(num_claims):
            same_case = torch.nonzero(
                opponent_map == claim_map[claim_idx], as_tuple=True
            )[0]

            if same_case.numel() == 0:
                outputs.append(claims.new_zeros(self.hidden_size))
                output_indices.append(indices[claim_idx])
                continue

            scores = torch.mv(k_all[same_case], q_all[claim_idx])
            k = min(self.topk_opponents, same_case.numel())
            local_top = torch.topk(scores, k=k, dim=0).indices
            selected_global = same_case[local_top]
            selected = opponents[selected_global].unsqueeze(0)

            attended, _ = self.opp_attention(
                query=claims[claim_idx : claim_idx + 1].unsqueeze(1),
                key=selected,
                value=selected,
                need_weights=False,
            )
            outputs.append(attended[0, 0])

            padded_idx = indices[claim_idx].clone()
            padded_idx[:k] = selected_global
            output_indices.append(padded_idx)

        return torch.stack(outputs, dim=0), torch.stack(output_indices, dim=0)

    def forward(
        self,
        claims: torch.Tensor,
        opponents: torch.Tensor,
        fact_tokens: torch.Tensor,
        fact_mask: torch.Tensor,
        global_context: torch.Tensor,
        claim_map: torch.Tensor,
        opponent_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if claims.size(0) == 0:
            empty_indices = torch.empty(
                (0, self.topk_opponents), dtype=torch.long, device=claims.device
            )
            return claims, {
                "fact_indices": torch.empty(
                    (0, 0), dtype=torch.long, device=claims.device
                ),
                "opponent_indices": empty_indices,
                "fusion_gates": claims.new_zeros((0, 4)),
            }

        fact_evidence, fact_indices = self._retrieve_fact_tokens(
            claims, fact_tokens, fact_mask, claim_map
        )
        opponent_evidence, opponent_indices = self._retrieve_opponents(
            claims, opponents, claim_map, opponent_map
        )
        case_context = global_context[claim_map]

        components = [claims, fact_evidence, opponent_evidence, case_context]
        gate_input = torch.cat(components, dim=-1)
        gates = torch.softmax(self.gate(gate_input), dim=-1)

        projected = torch.stack(
            [layer(component) for layer, component in zip(self.component_proj, components)],
            dim=1,
        )
        fused = torch.sum(gates.unsqueeze(-1) * projected, dim=1)
        fused = self.norm(claims + self.output(fused))

        return fused, {
            "fact_indices": fact_indices,
            "opponent_indices": opponent_indices,
            "fusion_gates": gates,
        }


class Stage1Encoder(nn.Module):
    """Stage 1 module with the same public interface as the old project."""

    def __init__(
        self,
        model_name: str = "sbintuitions/modernbert-ja-310m",
        claim_chunk_size: int = 64,
        num_heads: int = 8,
        topk_fact_tokens: int = 16,
        topk_opponents: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = SharedEncoder(model_name)
        hidden = self.encoder.hidden_size

        self.global_fact_pool = GlobalFactPool(hidden)
        self.plaintiff_fusion = ClaimEvidenceFusion(
            hidden,
            num_heads=num_heads,
            topk_fact_tokens=topk_fact_tokens,
            topk_opponents=topk_opponents,
            dropout=dropout,
        )
        self.defendant_fusion = ClaimEvidenceFusion(
            hidden,
            num_heads=num_heads,
            topk_fact_tokens=topk_fact_tokens,
            topk_opponents=topk_opponents,
            dropout=dropout,
        )

        self.claim_chunk_size = max(1, int(claim_chunk_size))

    def encode_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.size(0) == 0:
            return torch.zeros(
                0,
                self.encoder.hidden_size,
                device=input_ids.device,
                dtype=self.global_fact_pool.score.weight.dtype,
            )

        outputs = []
        for start in range(0, input_ids.size(0), self.claim_chunk_size):
            end = start + self.claim_chunk_size
            outputs.append(self.encoder(input_ids[start:end], attention_mask[start:end]))
        return torch.cat(outputs, dim=0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fact_outputs = self.encoder(
            batch["U_input_ids"],
            batch["U_attention_mask"],
            return_tokens=True,
        )
        fact_tokens = fact_outputs["tokens"]
        fact_mask = fact_outputs["attention_mask"]
        H_u, global_fact_attention = self.global_fact_pool(fact_tokens, fact_mask)

        h_P = self.encode_chunked(
            batch["P_input_ids"], batch["P_attention_mask"]
        )
        h_D = self.encode_chunked(
            batch["D_input_ids"], batch["D_attention_mask"]
        )

        sample_map_P = batch["sample_map_P"].long()
        sample_map_D = batch["sample_map_D"].long()

        hP_fused, diagnostics_P = self.plaintiff_fusion(
            claims=h_P,
            opponents=h_D,
            fact_tokens=fact_tokens,
            fact_mask=fact_mask,
            global_context=H_u,
            claim_map=sample_map_P,
            opponent_map=sample_map_D,
        )
        hD_fused, diagnostics_D = self.defendant_fusion(
            claims=h_D,
            opponents=h_P,
            fact_tokens=fact_tokens,
            fact_mask=fact_mask,
            global_context=H_u,
            claim_map=sample_map_D,
            opponent_map=sample_map_P,
        )

        return {
            "H_u": H_u,
            "fact_tokens": fact_tokens,
            "fact_mask": fact_mask,
            "global_fact_attention": global_fact_attention,
            # Raw vectors are kept for diagnostics/backward compatibility.
            "h_P": h_P,
            "h_D": h_D,
            # Existing Stage 2 expects these names.
            "hP_cond": hP_fused,
            "hD_cond": hD_fused,
            "plaintiff_fact_indices": diagnostics_P["fact_indices"],
            "defendant_fact_indices": diagnostics_D["fact_indices"],
            "plaintiff_opponent_indices": diagnostics_P["opponent_indices"],
            "defendant_opponent_indices": diagnostics_D["opponent_indices"],
            "plaintiff_fusion_gates": diagnostics_P["fusion_gates"],
            "defendant_fusion_gates": diagnostics_D["fusion_gates"],
        }
