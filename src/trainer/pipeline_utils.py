import torch


def compute_gt_stats(
    r_gt:       torch.Tensor,   # [N_claims]  0/1/-1
    sample_map: torch.Tensor,   # [N_claims]
    batch_size: int,
    device:     torch.device,
) -> torch.Tensor:
    """
    Tính stats [B, 4] = (max, mean, sum, entropy) từ ground-truth rationale.
    Vị trí label = -1 bị bỏ qua.
    Dùng làm input cho LabelConditionedAttention trong TDHead.
    """

    stats = torch.zeros(batch_size, 4, device=device)

    for case_id in range(batch_size):

        idx = (sample_map == case_id).nonzero(as_tuple=True)[0]

        if len(idx) == 0:
            continue

        r = r_gt[idx].float()

        # Lọc -1
        valid_mask = r >= 0
        if valid_mask.sum() == 0:
            continue

        r = r[valid_mask]

        r_sum = r.sum().clamp(min=1e-8)
        w     = r / r_sum
        w     = w.clamp(min=1e-8)

        stats[case_id, 0] = r.max()
        stats[case_id, 1] = r.mean()
        stats[case_id, 2] = r.sum()
        stats[case_id, 3] = -(w * w.log()).sum()   # entropy

    return stats