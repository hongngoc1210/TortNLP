"""
pooling.py

RationalePooling với [A2] Supervised Contrastive Loss.

[A2] Contrastive pooling loss:
  Ý tưởng: embedding của rationale claims (R=1) nên cluster lại gần nhau,
  xa với non-rationale claims (R=0) trong không gian vector.
  Điều này trực tiếp cải thiện chất lượng của H_re_p / H_re_d vì
  pooling sẽ tập trung vào signal thực sự của rationale.

  Loss dùng supervised contrastive (SimCLR-style) per case:
    - Anchor: 1 claim
    - Positive: các claims cùng label
    - Negative: các claims khác label
  Chỉ tính trên claims có label hợp lệ (>= 0).
  Trả về loss_contrastive để MultiTaskLoss cộng vào.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  [A2] SupervisedContrastiveLoss (per case, variable-length)
# =============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss cho tập claims variable-length.
    Tính per-case rồi average.

    Với mỗi case có N claims và binary labels (0/1):
      Với mỗi anchor i:
        positives = {j : label[j] == label[i], j != i}
        negatives = {j : label[j] != label[i]}
        loss_i = -log( sum_pos exp(sim(i,j)/T) / sum_{j!=i} exp(sim(i,j)/T) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _loss_one_case(
        self,
        h:      torch.Tensor,   # [N, hidden] — claim embeddings (L2 normalized)
        labels: torch.Tensor,   # [N]          — 0/1
    ) -> torch.Tensor:

        N = h.size(0)

        if N < 2:
            return torch.tensor(0.0, device=h.device)

        # QUAN TRỌNG: force float32 để tránh AMP float16 overflow.
        # exp(sim/0.07) với float16 overflow tại exp(11.1) → NaN trong logsumexp.
        # Contrastive loss phải luôn chạy trong float32.
        h = h.float()
        labels = labels.float()

        # L2 normalize để similarity = cosine
        h_norm = F.normalize(h, dim=-1)

        # Similarity matrix [N, N]
        sim = torch.matmul(h_norm, h_norm.T) / self.temperature

        # Mask tự tương đồng
        eye = torch.eye(N, device=h.device, dtype=torch.bool)
        sim.masked_fill_(eye, float("-inf"))

        # Positive mask: cùng label, khác index
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        pos_mask = label_eq & ~eye                              # [N, N]

        # Nếu không có positive pair nào trong case → trả về 0 giữ dtype/device
        # Dùng h.sum() * 0 thay vì torch.tensor(0.0) để giữ grad_fn trong AMP
        if not pos_mask.any():
            return (h.sum() * 0.0).float()

        # Log-softmax trên toàn hàng (loại trừ diagonal)
        log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)

        # Loss = -mean log-prob của các positive pairs
        loss = -(log_prob * pos_mask.float()).sum(dim=-1) \
               / pos_mask.float().sum(dim=-1).clamp(min=1)

        return loss.mean()

    def forward(
        self,
        h:          torch.Tensor,   # [total_claims, hidden]
        labels:     torch.Tensor,   # [total_claims]  — 0/1/-1
        sample_map: torch.Tensor,   # [total_claims]
        batch_size: int,
    ) -> torch.Tensor:

        losses = []

        for case_id in range(batch_size):

            idx = (sample_map == case_id).nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue

            h_case = h[idx]
            l_case = labels[idx]

            # Chỉ tính trên claims có label hợp lệ
            valid = l_case >= 0
            if valid.sum() < 2:
                continue

            loss_case = self._loss_one_case(h_case[valid], l_case[valid])
            losses.append(loss_case)

        if not losses:
            return torch.tensor(0.0, device=h.device)

        return torch.stack(losses).mean()


# =============================================================================
#  RationalePooling (updated with [A2])
# =============================================================================

class RationalePooling(nn.Module):

    def __init__(
        self,
        hidden:               int,
        tau:                  float = 1.0,
        # [A2] contrastive
        contrastive_temp:     float = 0.07,
        use_contrastive_loss: bool  = True,
    ):
        super().__init__()

        self.hidden = hidden
        self.tau    = tau

        # [A2]
        self.use_contrastive_loss = use_contrastive_loss
        if use_contrastive_loss:
            self.contrastive_loss = SupervisedContrastiveLoss(
                temperature=contrastive_temp
            )

    # -------------------------------------------------------------------------

    def pool_side(self, h, r, sample_map, batch_size):

        device = h.device
        H_pool = torch.zeros(batch_size, self.hidden, device=device)
        stats  = torch.zeros(batch_size, 4, device=device)

        for case_id in range(batch_size):

            idx = (sample_map == case_id).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            h_case = h[idx]
            r_case = r[idx]

            r_case  = torch.nan_to_num(r_case, nan=0.0, posinf=1.0, neginf=0.0)
            r_case  = torch.clamp(r_case, -10, 10)
            weights = torch.nan_to_num(torch.softmax(r_case / self.tau, dim=0))

            soft_pool = torch.sum(weights.unsqueeze(-1) * h_case, dim=0)
            mean_pool = torch.mean(h_case, dim=0)
            pooled    = 0.5 * soft_pool + 0.5 * mean_pool
            H_pool[case_id] = pooled

            stats[case_id, 0] = torch.max(r_case)
            stats[case_id, 1] = torch.mean(r_case)
            stats[case_id, 2] = torch.sum(r_case)
            w = torch.clamp(weights, min=1e-8)
            stats[case_id, 3] = -torch.sum(w * torch.log(w))

        return H_pool, stats

    # -------------------------------------------------------------------------

    def forward(self, stage1_out, stage2_out, batch):

        batch_size = batch["U_input_ids"].size(0)

        hP   = stage1_out["hP_cond"]
        hD   = stage1_out["hD_cond"]
        rP   = stage2_out["rP_hat"]
        rD   = stage2_out["rD_hat"]
        mapP = batch["sample_map_P"]
        mapD = batch["sample_map_D"]

        H_re_p, stats_P = self.pool_side(hP, rP, mapP, batch_size)
        H_re_d, stats_D = self.pool_side(hD, rD, mapD, batch_size)

        # [A2] Contrastive loss — chỉ tính khi training và có labels
        loss_contrastive = torch.tensor(0.0, device=hP.device)

        if self.use_contrastive_loss and self.training:

            R_P = batch.get("R_P")
            R_D = batch.get("R_D")

            if R_P is not None and R_D is not None:
                loss_P = self.contrastive_loss(hP, R_P, mapP, batch_size)
                loss_D = self.contrastive_loss(hD, R_D, mapD, batch_size)
                loss_contrastive = loss_P + loss_D

        return {
            "H_re_p":           H_re_p,
            "H_re_d":           H_re_d,
            "stats_P":          stats_P,
            "stats_D":          stats_D,
            "loss_contrastive": loss_contrastive,   # [A2]
        }