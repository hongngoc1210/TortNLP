from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import masked_softmax, valid_num_heads
from .encode_text import SharedEncoder


class GlobalFactPool(nn.Module):
    """Create the global case representation ``H_u`` from fact-token states."""

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
        weights = masked_softmax(scores, fact_mask.bool(), dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * fact_tokens, dim=1)
        return self.output_norm(pooled), weights


class ClaimEvidenceFusion(nn.Module):
    """
    Fuse four sources for each claim:

    1. the claim's own encoded semantics,
    2. top-k fact-token evidence,
    3. top-k opposing claims from the same case,
    4. the global case representation ``H_u``.
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

        self.hidden_size = int(hidden_size)
        self.topk_fact_tokens = max(1, int(topk_fact_tokens))
        self.topk_opponents = max(1, int(topk_opponents))

        heads = valid_num_heads(self.hidden_size, num_heads)

        self.fact_query = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.fact_key = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.fact_attention = nn.MultiheadAttention(
            self.hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.opp_query = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.opp_key = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.opp_attention = nn.MultiheadAttention(
            self.hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.component_proj = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(4)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 4),
        )

        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def _empty_indices(
        self,
        rows: int,
        columns: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full(
            (rows, columns),
            -1,
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _check_sample_map(
        sample_map: torch.Tensor,
        num_cases: int,
        name: str,
    ) -> None:
        """Raise a clear error if a claim-to-case map is invalid."""

        if sample_map.numel() == 0:
            return

        minimum = int(sample_map.min().item())
        maximum = int(sample_map.max().item())

        if minimum < 0 or maximum >= num_cases:
            raise IndexError(
                f"{name} contains case indices outside [0, {num_cases - 1}]: "
                f"min={minimum}, max={maximum}."
            )

    def _retrieve_fact_tokens(
        self,
        claims: torch.Tensor,
        fact_tokens: torch.Tensor,
        fact_mask: torch.Tensor,
        sample_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve fact evidence case by case.

        The old implementation used ``fact_tokens[sample_map]``, producing a
        tensor shaped ``[num_claims, fact_length, hidden]``. That duplicated
        the same case facts once per claim and caused a large VRAM spike.

        This implementation projects facts once as
        ``[num_cases, fact_length, hidden]`` and only materializes the selected
        top-k tokens for the claims belonging to the current case.
        """

        num_claims = int(claims.size(0))
        num_cases = int(fact_tokens.size(0))

        if num_claims == 0:
            return (
                claims.new_zeros((0, self.hidden_size)),
                self._empty_indices(
                    0,
                    self.topk_fact_tokens,
                    claims.device,
                ),
            )

        self._check_sample_map(sample_map, num_cases, "sample_map")

        # Under AMP, these projected tensors can be FP16/BF16 even when
        # ``claims`` is FP32.
        queries = F.normalize(
            self.fact_query(claims),
            dim=-1,
        )
        fact_keys = F.normalize(
            self.fact_key(fact_tokens),
            dim=-1,
        )

        # Use the projected-query dtype, not claims.new_zeros(). This prevents:
        # index_copy_(): self Float and source Half.
        evidence = queries.new_zeros(
            (num_claims, self.hidden_size)
        )
        top_indices_out = self._empty_indices(
            num_claims,
            self.topk_fact_tokens,
            claims.device,
        )

        for case_id_tensor in sample_map.unique(sorted=False):
            claim_indices = torch.nonzero(
                sample_map == case_id_tensor,
                as_tuple=True,
            )[0]
            if claim_indices.numel() == 0:
                continue

            case_index = int(case_id_tensor.item())
            current_queries = queries.index_select(0, claim_indices)
            current_keys = fact_keys[case_index]
            current_tokens = fact_tokens[case_index]
            current_mask = fact_mask[case_index].bool()

            # [claims_in_case, fact_length]
            similarity = torch.matmul(
                current_queries,
                current_keys.transpose(0, 1),
            )
            similarity = similarity.masked_fill(
                ~current_mask.unsqueeze(0),
                -1e4,
            )

            k = min(
                self.topk_fact_tokens,
                int(current_tokens.size(0)),
            )
            top_indices = torch.topk(
                similarity,
                k=k,
                dim=-1,
            ).indices

            # [claims_in_case, k, hidden]
            selected_tokens = current_tokens[top_indices]
            selected_mask = current_mask[top_indices]

            # MultiheadAttention cannot receive a row where every key is
            # masked. Insert a zero placeholder for those rare rows.
            safe_mask = selected_mask.clone()
            no_valid_tokens = ~safe_mask.any(dim=1)
            if no_valid_tokens.any():
                safe_mask[no_valid_tokens, 0] = True
                selected_tokens = selected_tokens.clone()
                selected_tokens[no_valid_tokens, 0] = 0.0

            attended, _ = self.fact_attention(
                query=claims.index_select(0, claim_indices).unsqueeze(1),
                key=selected_tokens,
                value=selected_tokens,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            attended = attended.squeeze(1)

            # Rows with no valid fact tokens should contribute zero evidence.
            attended = torch.where(
                selected_mask.any(dim=1, keepdim=True),
                attended,
                torch.zeros_like(attended),
            )

            # ``index_copy`` requires identical destination/source dtypes.
            attended = attended.to(dtype=evidence.dtype)
            evidence = evidence.index_copy(
                0,
                claim_indices,
                attended,
            )
            top_indices_out[claim_indices, :k] = top_indices

        return evidence, top_indices_out

    def _retrieve_opponents(
        self,
        claims: torch.Tensor,
        opponents: torch.Tensor,
        claim_map: torch.Tensor,
        opponent_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and attend to top-k opposing claims from the same case.

        Retrieval is grouped by case instead of scanning the full opponent map
        separately for every claim.
        """

        num_claims = int(claims.size(0))
        indices = self._empty_indices(
            num_claims,
            self.topk_opponents,
            claims.device,
        )

        if num_claims == 0:
            return (
                claims.new_zeros((0, self.hidden_size)),
                indices,
            )

        q_all = F.normalize(
            self.opp_query(claims),
            dim=-1,
        )
        evidence = q_all.new_zeros(
            (num_claims, self.hidden_size)
        )

        if opponents.size(0) == 0:
            return evidence, indices

        k_all = F.normalize(
            self.opp_key(opponents),
            dim=-1,
        )

        for case_id_tensor in claim_map.unique(sorted=False):
            claim_indices = torch.nonzero(
                claim_map == case_id_tensor,
                as_tuple=True,
            )[0]
            opponent_indices = torch.nonzero(
                opponent_map == case_id_tensor,
                as_tuple=True,
            )[0]

            if (
                claim_indices.numel() == 0
                or opponent_indices.numel() == 0
            ):
                continue

            current_queries = q_all.index_select(
                0,
                claim_indices,
            )
            current_keys = k_all.index_select(
                0,
                opponent_indices,
            )

            # [claims_in_case, opponents_in_case]
            scores = torch.matmul(
                current_queries,
                current_keys.transpose(0, 1),
            )
            k = min(
                self.topk_opponents,
                int(opponent_indices.numel()),
            )
            local_top = torch.topk(
                scores,
                k=k,
                dim=-1,
            ).indices

            # Convert case-local indices to indices in the flattened opponent
            # tensor. Shape: [claims_in_case, k].
            selected_global = opponent_indices[local_top]
            selected_opponents = opponents[selected_global]

            attended, _ = self.opp_attention(
                query=claims.index_select(
                    0,
                    claim_indices,
                ).unsqueeze(1),
                key=selected_opponents,
                value=selected_opponents,
                need_weights=False,
            )
            attended = attended.squeeze(1)
            attended = attended.to(dtype=evidence.dtype)

            evidence = evidence.index_copy(
                0,
                claim_indices,
                attended,
            )
            indices[claim_indices, :k] = selected_global

        return evidence, indices

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
        num_claims = int(claims.size(0))

        if num_claims == 0:
            return claims, {
                "fact_indices": self._empty_indices(
                    0,
                    self.topk_fact_tokens,
                    claims.device,
                ),
                "opponent_indices": self._empty_indices(
                    0,
                    self.topk_opponents,
                    claims.device,
                ),
                "fusion_gates": claims.new_zeros((0, 4)),
            }

        claim_map = claim_map.long()
        opponent_map = opponent_map.long()

        self._check_sample_map(
            claim_map,
            int(global_context.size(0)),
            "claim_map",
        )

        fact_evidence, fact_indices = self._retrieve_fact_tokens(
            claims=claims,
            fact_tokens=fact_tokens,
            fact_mask=fact_mask,
            sample_map=claim_map,
        )
        opponent_evidence, opponent_indices = self._retrieve_opponents(
            claims=claims,
            opponents=opponents,
            claim_map=claim_map,
            opponent_map=opponent_map,
        )
        case_context = global_context.index_select(
            0,
            claim_map,
        )

        components = [
            claims,
            fact_evidence,
            opponent_evidence,
            case_context,
        ]

        # torch.cat promotes mixed FP16/FP32 inputs safely. Linear layers then
        # follow the active autocast policy.
        gate_input = torch.cat(components, dim=-1)
        gates = torch.softmax(
            self.gate(gate_input),
            dim=-1,
        )

        projected = torch.stack(
            [
                layer(component)
                for layer, component in zip(
                    self.component_proj,
                    components,
                )
            ],
            dim=1,
        )
        fused = torch.sum(
            gates.unsqueeze(-1) * projected,
            dim=1,
        )
        fused = self.norm(
            claims + self.output(fused)
        )

        return fused, {
            "fact_indices": fact_indices,
            "opponent_indices": opponent_indices,
            "fusion_gates": gates,
        }


class Stage1Encoder(nn.Module):
    """Stage 1 claim-centric encoder with a TP-only global fast path."""

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
        hidden_size = self.encoder.hidden_size

        self.global_fact_pool = GlobalFactPool(hidden_size)
        self.plaintiff_fusion = ClaimEvidenceFusion(
            hidden_size=hidden_size,
            num_heads=num_heads,
            topk_fact_tokens=topk_fact_tokens,
            topk_opponents=topk_opponents,
            dropout=dropout,
        )
        self.defendant_fusion = ClaimEvidenceFusion(
            hidden_size=hidden_size,
            num_heads=num_heads,
            topk_fact_tokens=topk_fact_tokens,
            topk_opponents=topk_opponents,
            dropout=dropout,
        )

        self.claim_chunk_size = max(
            1,
            int(claim_chunk_size),
        )

    def encode_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode flattened claims in smaller forward chunks."""

        if input_ids.size(0) == 0:
            return torch.zeros(
                (0, self.encoder.hidden_size),
                device=input_ids.device,
                dtype=self.global_fact_pool.score.weight.dtype,
            )

        outputs = []
        for start in range(
            0,
            input_ids.size(0),
            self.claim_chunk_size,
        ):
            end = min(
                start + self.claim_chunk_size,
                input_ids.size(0),
            )
            outputs.append(
                self.encoder(
                    input_ids[start:end],
                    attention_mask[start:end],
                )
            )

        return torch.cat(outputs, dim=0)

    def _encode_facts(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fact_outputs = self.encoder(
            batch["U_input_ids"],
            batch["U_attention_mask"],
            return_tokens=True,
        )
        fact_tokens = fact_outputs["tokens"]
        fact_mask = fact_outputs["attention_mask"].bool()
        H_u, global_fact_attention = self.global_fact_pool(
            fact_tokens,
            fact_mask,
        )
        return (
            fact_tokens,
            fact_mask,
            H_u,
            global_fact_attention,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        (
            fact_tokens,
            fact_mask,
            H_u,
            global_fact_attention,
        ) = self._encode_facts(batch)

        h_P = self.encode_chunked(
            batch["P_input_ids"],
            batch["P_attention_mask"],
        )
        h_D = self.encode_chunked(
            batch["D_input_ids"],
            batch["D_attention_mask"],
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
            # Raw vectors are retained for diagnostics/backward compatibility.
            "h_P": h_P,
            "h_D": h_D,
            # Existing Stage 2 expects these names.
            "hP_cond": hP_fused,
            "hD_cond": hD_fused,
            "plaintiff_fact_indices": diagnostics_P["fact_indices"],
            "defendant_fact_indices": diagnostics_D["fact_indices"],
            "plaintiff_opponent_indices": diagnostics_P[
                "opponent_indices"
            ],
            "defendant_opponent_indices": diagnostics_D[
                "opponent_indices"
            ],
            "plaintiff_fusion_gates": diagnostics_P[
                "fusion_gates"
            ],
            "defendant_fusion_gates": diagnostics_D[
                "fusion_gates"
            ],
        }

    def forward_global(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode only undisputed facts for the true TP-only ablation.

        The trainer must call this method instead of ``forward`` when:
        ``task_mode == "tp_only"`` and ``tp_input_mode == "global_only"``.
        """

        (
            fact_tokens,
            fact_mask,
            H_u,
            global_fact_attention,
        ) = self._encode_facts(batch)

        return {
            "H_u": H_u,
            "fact_tokens": fact_tokens,
            "fact_mask": fact_mask,
            "global_fact_attention": global_fact_attention,
        }