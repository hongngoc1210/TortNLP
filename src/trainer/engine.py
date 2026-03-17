import torch
import torch.nn as nn

from trainer.metrics import compute_re_f1, compute_td_accuracy, compute_td_f1, find_best_threshold


class Trainer:

    def __init__(
        self,
        stage1,
        stage2,
        stage3,
        stage4,
        loss_fn,
        optimizer,
        tf_scheduler,
        lr_scheduler=None,        # [NEW] warmup + cosine decay
        device="cuda",
        grad_accum_steps=1,
        use_amp=True,
        max_grad_norm=1.0,        # [NEW] gradient clipping
    ):

        self.stage1 = stage1.to(device)
        self.stage2 = stage2.to(device)
        self.stage3 = stage3.to(device)
        self.stage4 = stage4.to(device)

        self.loss_fn          = loss_fn.to(device)   # move log_var params to device
        self.optimizer        = optimizer
        self.tf_scheduler     = tf_scheduler
        self.lr_scheduler     = lr_scheduler   # [NEW]

        self.device           = device
        self.grad_accum_steps = grad_accum_steps
        self.use_amp          = use_amp
        self.max_grad_norm    = max_grad_norm   # [NEW]

        # FIX #1: torch.cuda.amp.GradScaler deprecated → torch.amp.GradScaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # =========================================================================
    # train_epoch
    # =========================================================================

    def train_epoch(self, loader, epoch):

        self.stage1.train()
        self.stage2.train()
        self.stage3.train()
        self.stage4.train()

        eta = self.tf_scheduler.get_eta(epoch)

        total_loss    = 0.0
        total_re_loss = 0.0
        total_td_loss = 0.0
        total_cons_loss = 0.0
        zero_loss_steps = 0   # đếm số bước loss về 0 để detect collapse

        all_models = [self.stage1, self.stage2, self.stage3, self.stage4]

        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):

            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", enabled=self.use_amp):

                # ---- Stage 1 ----
                s1 = self.stage1(batch)

                # ---- Stage 2 ----
                s2 = self.stage2(s1)

                # ---- Teacher forcing (FIX #2: mask -1 labels) ----
                # R_P/R_D = -1 khi test set không có ground-truth
                # Mask ra: vị trí nào -1 thì dùng r_hat thuần
                if eta > 0:
                    rP_mix = _masked_mix(eta, batch["R_P"], s2["rP_hat"])
                    rD_mix = _masked_mix(eta, batch["R_D"], s2["rD_hat"])
                    s2["rP_hat"] = rP_mix
                    s2["rD_hat"] = rD_mix

                # ---- Stage 3 ----
                s3 = self.stage3(s1, s2, batch)

                # ---- Stage 4 (FIX #3: truyền gt_stats cho TDHead mới) ----
                gt_stats_P, gt_stats_D = _build_gt_stats(
                    batch, s3, self.device
                ) if eta > 0 else (None, None)

                s4 = self.stage4(
                    s1, s3,
                    gt_stats_P=gt_stats_P,
                    gt_stats_D=gt_stats_D,
                )

                # [A2] Bubble loss_contrastive từ pooling vào s4
                # MultiTaskLoss đọc từ td_outputs dict
                if "loss_contrastive" in s3:
                    s4["loss_contrastive"] = s3["loss_contrastive"]

                # ---- Loss ----
                # MultiTaskLoss.forward() returns tuple (loss, loss_re, loss_td)
                loss, loss_re, loss_td = self.loss_fn(s2, s4, batch)

                loss = loss / self.grad_accum_steps

            # Guard: skip backward nếu loss không hợp lệ
            # (NaN/Inf loss → scaler.step() sẽ crash với AssertionError)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:

                # Unscale trước khi clip và step
                self.scaler.unscale_(self.optimizer)

                # Skip step nếu gradient có NaN/Inf (scaler tự detect)
                nn.utils.clip_grad_norm_(
                    [p for m in all_models for p in m.parameters()],
                    self.max_grad_norm,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Step LR scheduler per optimizer-step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            step_loss = loss.item() * self.grad_accum_steps
            total_loss    += step_loss
            total_re_loss += loss_re.item()
            total_td_loss += loss_td.item()

            # Detect loss collapse: nếu total_loss < 1e-6 liên tiếp 50+ steps
            if step_loss < 1e-6:
                zero_loss_steps += 1
            else:
                zero_loss_steps = 0

            if zero_loss_steps == 50:
                import warnings
                warnings.warn(
                    f"[Epoch {epoch}] Loss đã về ~0 trong 50 steps liên tiếp! "
                    f"loss_re={loss_re.item():.6f}, loss_td={loss_td.item():.6f}. "
                    "Kiểm tra ASL gamma_neg và FocalLoss alpha.",
                    RuntimeWarning, stacklevel=2,
                )

        n = len(loader)

        # FIX #7: trả về dict thay vì scalar
        return {
            "loss":    total_loss    / n,
            "loss_re": total_re_loss / n,
            "loss_td": total_td_loss / n,
            "zero_loss_steps": zero_loss_steps,   # >0 nếu có dấu hiệu collapse
        }

    # =========================================================================
    # evaluate
    # =========================================================================

    def evaluate(self, loader):

        self.stage1.eval()
        self.stage2.eval()
        self.stage3.eval()
        self.stage4.eval()

        re_preds_P, re_labels_P = [], []
        re_preds_D, re_labels_D = [], []
        td_preds,   td_labels   = [], []

        with torch.no_grad():

            for batch in loader:

                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                with torch.amp.autocast("cuda", enabled=self.use_amp):

                    s1 = self.stage1(batch)
                    s2 = self.stage2(s1)
                    s3 = self.stage3(s1, s2, batch)
                    # FIX #3: inference → no gt_stats
                    s4 = self.stage4(s1, s3)

                # FIX #16: kéo P/D riêng
                re_preds_P.append(s2["rP_hat"].cpu())
                re_preds_D.append(s2["rD_hat"].cpu())
                re_labels_P.append(batch["R_P"].cpu())
                re_labels_D.append(batch["R_D"].cpu())

                td_preds.append(s4["T_hat"].cpu())
                td_labels.append(batch["T"].cpu())

        re_preds_P  = torch.cat(re_preds_P)
        re_preds_D  = torch.cat(re_preds_D)
        re_labels_P = torch.cat(re_labels_P)
        re_labels_D = torch.cat(re_labels_D)

        td_preds  = torch.cat(td_preds)
        td_labels = torch.cat(td_labels)

        # Tìm optimal threshold per-side thay vì hardcode 0.5
        # Quan trọng trong early epochs khi model chưa calibrate
        t_P, _ = find_best_threshold(re_preds_P, re_labels_P)
        t_D, _ = find_best_threshold(re_preds_D, re_labels_D)
        t_td,_ = find_best_threshold(td_preds,   td_labels)

        re_metrics = compute_re_f1(
            re_preds_P, re_labels_P,
            re_preds_D, re_labels_D,
            threshold_P=t_P,
            threshold_D=t_D,
        )
        td_acc = compute_td_accuracy(td_preds, td_labels, threshold=t_td)
        td_f1  = compute_td_f1(td_preds, td_labels, threshold=t_td)

        return {
            "re_f1_P":        re_metrics["f1_P"],
            "re_f1_D":        re_metrics["f1_D"],
            "re_f1_macro":    re_metrics["f1_macro"],
            "re_f1_combined": re_metrics["f1_combined"],
            "re_threshold_P": t_P,
            "re_threshold_D": t_D,
            "td_threshold":   t_td,
            "td_acc":         td_acc,
            "td_f1":          td_f1,
            # composite score: dùng td_f1 thay acc (robust hơn với imbalance)
            "score":          (re_metrics["f1_macro"] + td_f1) / 2.0,
        }

    # =========================================================================
    # predict
    # =========================================================================

    def predict(self, loader):

        self.stage1.eval()
        self.stage2.eval()
        self.stage3.eval()
        self.stage4.eval()

        predictions = []

        with torch.no_grad():

            for batch in loader:

                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                # FIX #15: cần autocast trong predict để tránh dtype mismatch
                with torch.amp.autocast("cuda", enabled=self.use_amp):

                    s1 = self.stage1(batch)
                    s2 = self.stage2(s1)
                    s3 = self.stage3(s1, s2, batch)
                    s4 = self.stage4(s1, s3)

                rP = s2["rP_hat"].cpu()
                rD = s2["rD_hat"].cpu()
                T  = s4["T_hat"].cpu()

                tort_ids   = batch["tort_id"]
                map_P      = batch["sample_map_P"].cpu()
                map_D      = batch["sample_map_D"].cpu()

                for i, tid in enumerate(tort_ids):

                    p_idx = (map_P == i).nonzero(as_tuple=True)[0]
                    d_idx = (map_D == i).nonzero(as_tuple=True)[0]

                    predictions.append({
                        "tort_id":    tid,
                        "T_hat":      float(T[i]),
                        "rP_hat":     rP[p_idx].tolist(),
                        "rD_hat":     rD[d_idx].tolist(),
                    })

        return predictions


# =============================================================================
# Helpers (module-level, không phải method)
# =============================================================================

def _masked_mix(
    eta:    float,
    r_gt:   torch.Tensor,   # [N] — có thể chứa -1
    r_hat:  torch.Tensor,   # [N]
) -> torch.Tensor:
    """
    FIX #2: teacher forcing chỉ áp dụng nơi label hợp lệ (>= 0).
    Vị trí label = -1 dùng r_hat thuần không trộn.
    """
    valid = (r_gt >= 0).float()
    mixed = eta * r_gt.float() * valid + (1.0 - eta * valid) * r_hat
    return mixed


def _build_gt_stats(
    batch:  dict,
    s3:     dict,
    device: torch.device,
) -> tuple:
    """
    FIX #3: tính gt_stats_P/D từ ground-truth R_P/R_D để truyền vào TDHead
    (LabelConditionedAttention).  Chỉ gọi khi eta > 0 (training).
    Trả về (stats_P, stats_D) mỗi cái shape [B, 4].
    """

    from trainer.pipeline_utils import compute_gt_stats   # lazy import tránh circular

    B        = batch["U_input_ids"].size(0)
    gt_stats_P = compute_gt_stats(batch["R_P"], batch["sample_map_P"], B, device)
    gt_stats_D = compute_gt_stats(batch["R_D"], batch["sample_map_D"], B, device)

    return gt_stats_P, gt_stats_D