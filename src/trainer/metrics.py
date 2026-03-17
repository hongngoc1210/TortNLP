"""
metrics.py

Fixes so với version trước:
  - find_best_threshold(): tìm threshold tối ưu trên dev set thay vì
    hardcode 0.5. Quan trọng trong 2-3 epoch đầu khi model chưa calibrate.
  - compute_re_f1() nhận thêm tham số threshold (default 0.5, override từ ngoài).
"""

import torch


# =============================================================================
# Helpers
# =============================================================================

def _filter_valid(preds: torch.Tensor, labels: torch.Tensor):
    mask = labels >= 0
    return preds[mask], labels[mask]


def _binary_f1(
    preds:     torch.Tensor,
    labels:    torch.Tensor,
    threshold: float = 0.5,
) -> float:

    preds_bin = (preds > threshold).float()

    tp = ((preds_bin == 1) & (labels == 1)).sum()
    fp = ((preds_bin == 1) & (labels == 0)).sum()
    fn = ((preds_bin == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.item()


# =============================================================================
# Optimal threshold search
# =============================================================================

def find_best_threshold(
    preds:  torch.Tensor,
    labels: torch.Tensor,
    n_steps: int = 50,
) -> tuple[float, float]:
    """
    Tìm threshold tối ưu hoá F1 trên một tập predictions.
    Trả về (best_threshold, best_f1).

    Dùng để override threshold=0.5 trong evaluate() — quan trọng
    vì 2-3 epoch đầu model chưa calibrate, logits RE gần 0.
    """
    preds, labels = _filter_valid(preds, labels)

    if labels.numel() == 0 or labels.sum() == 0:
        return 0.5, 0.0

    best_t  = 0.5
    best_f1 = 0.0

    for i in range(1, n_steps):
        t  = i / n_steps
        f1 = _binary_f1(preds, labels, threshold=t)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    return best_t, best_f1


# =============================================================================
# Public API
# =============================================================================

def compute_re_f1(
    preds_P:    torch.Tensor,
    labels_P:   torch.Tensor,
    preds_D:    torch.Tensor,
    labels_D:   torch.Tensor,
    threshold_P: float = 0.5,
    threshold_D: float = 0.5,
) -> dict:

    pP, lP = _filter_valid(preds_P, labels_P)
    pD, lD = _filter_valid(preds_D, labels_D)

    f1_p = _binary_f1(pP, lP, threshold_P) if lP.numel() > 0 else 0.0
    f1_d = _binary_f1(pD, lD, threshold_D) if lD.numel() > 0 else 0.0

    f1_macro = (f1_p + f1_d) / 2.0

    if pP.numel() > 0 or pD.numel() > 0:
        p_all = torch.cat([pP, pD]) if (pP.numel() > 0 and pD.numel() > 0) \
                else (pP if pP.numel() > 0 else pD)
        l_all = torch.cat([lP, lD]) if (lP.numel() > 0 and lD.numel() > 0) \
                else (lP if lP.numel() > 0 else lD)
        t_avg = (threshold_P + threshold_D) / 2
        f1_combined = _binary_f1(p_all, l_all, t_avg)
    else:
        f1_combined = 0.0

    return {
        "f1_P":        f1_p,
        "f1_D":        f1_d,
        "f1_macro":    f1_macro,
        "f1_combined": f1_combined,
    }


def compute_td_accuracy(
    preds:     torch.Tensor,
    labels:    torch.Tensor,
    threshold: float = 0.5,
) -> float:

    preds, labels = _filter_valid(preds, labels)
    if labels.numel() == 0:
        return 0.0

    return ((preds > threshold).float() == labels).float().mean().item()


def compute_td_f1(
    preds:     torch.Tensor,
    labels:    torch.Tensor,
    threshold: float = 0.5,
) -> float:

    preds, labels = _filter_valid(preds, labels)
    if labels.numel() == 0:
        return 0.0

    return _binary_f1(preds, labels, threshold)