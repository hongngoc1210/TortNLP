"""Training engine with reproducible MTL ablations and diagnostics.

Key fixes:
- Select the active task objective explicitly for re_only / tp_only / joint.
- Prefer graph-connected task objectives returned by loss_fn.task_objectives().
- Skip RE-only micro-batches that contain no valid RE supervision.
- Run training under torch.enable_grad() and emit actionable diagnostics if a
  loss is unexpectedly detached.
- Accumulate gradients by processed micro-batches rather than DataLoader index.
- Keep inactive stages in eval mode during single-task ablations.
"""

from __future__ import annotations

from typing import Iterable

import torch

from trainer.metrics import compute_re_f1, compute_td_accuracy


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
        lr_scheduler=None,
        device="cuda",
        grad_accum_steps=1,
        max_grad_norm=1.0,
        use_amp=True,
        task_mode="joint",
        tp_input_mode="rationale",
        train_rationale_source="teacher_forcing",
        eval_rationale_source="predicted",
        gradient_method="standard",
        grad_diagnostics_every=0,
    ):
        self.stage1 = stage1.to(device)
        self.stage2 = stage2.to(device)
        self.stage3 = stage3.to(device)
        self.stage4 = stage4.to(device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tf_scheduler = tf_scheduler
        self.lr_scheduler = lr_scheduler

        self.device = torch.device(device)
        self.device_type = self.device.type
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_grad_norm = float(max_grad_norm)
        self.task_mode = str(task_mode)
        self.tp_input_mode = str(tp_input_mode)
        self.train_rationale_source = str(train_rationale_source)
        self.eval_rationale_source = str(eval_rationale_source)
        self.gradient_method = str(gradient_method).lower()
        self.grad_diagnostics_every = max(0, int(grad_diagnostics_every))

        if self.task_mode not in {"joint", "re_only", "tp_only"}:
            raise ValueError(f"Unknown task_mode={self.task_mode!r}")
        if self.tp_input_mode not in {"rationale", "global_only"}:
            raise ValueError(f"Unknown tp_input_mode={self.tp_input_mode!r}")

        valid_sources = {
            "predicted",
            "teacher_forcing",
            "gold",
            "no_rationale",
            "random",
        }
        if self.train_rationale_source not in valid_sources:
            raise ValueError(
                f"Unknown train_rationale_source={self.train_rationale_source!r}"
            )
        if self.eval_rationale_source not in valid_sources:
            raise ValueError(
                f"Unknown eval_rationale_source={self.eval_rationale_source!r}"
            )
        if self.gradient_method not in {"standard", "pcgrad"}:
            raise ValueError("gradient_method must be 'standard' or 'pcgrad'")

        if self.gradient_method == "pcgrad":
            if self.task_mode != "joint":
                raise ValueError("PCGrad is only meaningful in joint task mode")
            if self.grad_accum_steps != 1:
                raise ValueError(
                    "This PCGrad implementation requires grad_accum_steps=1"
                )
            use_amp = False

        self.use_amp = bool(
            use_amp
            and self.device_type == "cuda"
            and torch.cuda.is_available()
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.last_train_stats = {}
        self.last_eval = {}

        self._validate_active_parameters()

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------

    def _active_training_modules(self):
        if self.task_mode == "re_only":
            return [self.stage1, self.stage2]

        if self.task_mode == "tp_only":
            if self.tp_input_mode == "global_only":
                return [self.stage1, self.stage4]
            return [self.stage1, self.stage2, self.stage3, self.stage4]

        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def _validate_active_parameters(self):
        active_parameters = self._trainable_parameters(
            self._active_training_modules()
        )
        if not active_parameters:
            raise RuntimeError(
                f"No trainable parameters for task_mode={self.task_mode!r}. "
                "Check requires_grad before constructing the optimizer."
            )

        optimizer_parameter_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        missing_from_optimizer = [
            parameter
            for parameter in active_parameters
            if id(parameter) not in optimizer_parameter_ids
        ]
        if missing_from_optimizer:
            missing_count = sum(
                parameter.numel() for parameter in missing_from_optimizer
            )
            raise RuntimeError(
                "Some active trainable parameters are missing from the optimizer "
                f"({missing_count:,} parameters). Configure requires_grad before "
                "building the optimizer."
            )

    # ------------------------------------------------------------------
    # Rationale controls
    # ------------------------------------------------------------------

    @staticmethod
    def _mix_rationale_scores(
        prediction: torch.Tensor,
        label: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        if prediction.numel() == 0:
            return prediction
        label = label.to(device=prediction.device, dtype=prediction.dtype)
        valid = label >= 0
        gold_or_prediction = torch.where(valid, label, prediction.detach())
        return eta * gold_or_prediction + (1.0 - eta) * prediction

    @staticmethod
    def _gold_rationale_scores(
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.numel() == 0:
            return prediction
        label = label.to(device=prediction.device, dtype=prediction.dtype)
        valid = label >= 0
        return torch.where(valid, label, prediction.detach())

    @staticmethod
    def _shuffle_within_cases(
        scores: torch.Tensor,
        sample_map: torch.Tensor,
    ) -> torch.Tensor:
        """Random-rationale control preserving each case's score distribution."""
        shuffled = scores.detach().clone()
        if scores.numel() == 0:
            return shuffled

        for case_id in sample_map.unique(sorted=False):
            indices = torch.nonzero(sample_map == case_id, as_tuple=True)[0]
            if indices.numel() > 1:
                permutation = indices[
                    torch.randperm(indices.numel(), device=indices.device)
                ]
                shuffled[indices] = scores.detach()[permutation]
        return shuffled

    # ------------------------------------------------------------------
    # Batch and mode utilities
    # ------------------------------------------------------------------

    def _move_batch(self, batch):
        return {
            key: (
                value.to(self.device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for key, value in batch.items()
        }

    def _set_train_mode(self):
        # Frozen modules stay in eval mode, which disables dropout and keeps the
        # phase-1 RE/global anchor stable during the phase-2 head warm-up.
        self.stage1.eval()
        self.stage2.eval()
        self.stage3.eval()
        self.stage4.eval()

        for module in self._active_training_modules():
            if any(
                parameter.requires_grad
                for parameter in module.parameters()
            ):
                module.train()

    def _set_eval_mode(self):
        self.stage1.eval()
        self.stage2.eval()
        self.stage3.eval()
        self.stage4.eval()

    @staticmethod
    def _count_valid_labels(labels) -> int:
        if labels is None or not torch.is_tensor(labels) or labels.numel() == 0:
            return 0
        valid = torch.isfinite(labels) & (labels >= 0)
        return int(valid.sum().item())

    def _has_re_supervision(self, batch: dict) -> bool:
        return (
            self._count_valid_labels(batch.get("R_P")) > 0
            or self._count_valid_labels(batch.get("R_D")) > 0
        )

    def _has_tp_supervision(self, batch: dict) -> bool:
        return self._count_valid_labels(batch.get("T")) > 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _prepare_pooling_inputs(
        self,
        s2: dict,
        batch: dict,
        source: str,
        eta: float,
    ):
        pooling_mode = "rationale"
        s2_pool = dict(s2)

        if source == "teacher_forcing":
            s2_pool["rP_for_pool"] = self._mix_rationale_scores(
                s2["rP_hat"], batch["R_P"], eta
            )
            s2_pool["rD_for_pool"] = self._mix_rationale_scores(
                s2["rD_hat"], batch["R_D"], eta
            )
        elif source == "gold":
            s2_pool["rP_for_pool"] = self._gold_rationale_scores(
                s2["rP_hat"], batch["R_P"]
            )
            s2_pool["rD_for_pool"] = self._gold_rationale_scores(
                s2["rD_hat"], batch["R_D"]
            )
        elif source == "random":
            s2_pool["rP_for_pool"] = self._shuffle_within_cases(
                s2["rP_hat"], batch["sample_map_P"].long()
            )
            s2_pool["rD_for_pool"] = self._shuffle_within_cases(
                s2["rD_hat"], batch["sample_map_D"].long()
            )
        elif source == "no_rationale":
            pooling_mode = "fallback_only"
        elif source != "predicted":
            raise ValueError(f"Unknown rationale source={source!r}")

        return s2_pool, pooling_mode

    def _forward_batch(
        self,
        batch: dict,
        epoch: int = 0,
        rationale_source: str | None = None,
    ):
        if self.task_mode == "tp_only" and self.tp_input_mode == "global_only":
            s1 = self.stage1.forward_global(batch)
            s4 = self.stage4(s1, None, input_mode="global_only")
            return s1, None, None, s4

        s1 = self.stage1(batch)
        need_re = self.task_mode != "tp_only" or self.tp_input_mode == "rationale"
        s2 = self.stage2(s1) if need_re else None

        if self.task_mode == "re_only":
            return s1, s2, None, None

        if self.tp_input_mode == "global_only":
            s4 = self.stage4(s1, None, input_mode="global_only")
            return s1, s2, None, s4

        if s2 is None:
            raise RuntimeError("Rationale TP mode requires Stage 2 outputs")

        source = rationale_source
        if source is None:
            source = (
                self.train_rationale_source
                if self.stage1.training
                else self.eval_rationale_source
            )

        eta = 0.0
        if self.tf_scheduler is not None:
            eta = float(self.tf_scheduler.get_eta(epoch))

        s2_pool, pooling_mode = self._prepare_pooling_inputs(
            s2=s2,
            batch=batch,
            source=source,
            eta=eta,
        )
        s3 = self.stage3(s1, s2_pool, batch, pooling_mode=pooling_mode)
        s4 = self.stage4(s1, s3, input_mode="rationale")
        return s1, s2, s3, s4

    # ------------------------------------------------------------------
    # Objective handling
    # ------------------------------------------------------------------

    @staticmethod
    def _as_scalar_tensor(value, reference=None):
        if torch.is_tensor(value):
            return value
        if reference is not None and torch.is_tensor(reference):
            return reference.new_tensor(float(value))
        return torch.tensor(float(value))

    @staticmethod
    def _first_graph_connected(*candidates):
        for candidate in candidates:
            if torch.is_tensor(candidate) and candidate.requires_grad:
                return candidate
        return None

    def _compute_task_losses(self, s2, s4, batch):
        """Compute only the objective required by the active ablation.

        Important:
        ``MultiTaskLoss.task_objectives`` assumes that both RE and TP outputs
        exist. Therefore it must not be called in ``re_only`` or ``tp_only``
        mode, where one output is intentionally ``None``.
        """

        if self.task_mode == "re_only":
            if s2 is None:
                raise RuntimeError(
                    "re_only requires Stage 2 outputs, but s2 is None."
                )

            # Prefer the dedicated RE criterion so the loss is not multiplied
            # by the joint-task weight (0.33).
            if hasattr(self.loss_fn, "re_loss"):
                loss_re = self.loss_fn.re_loss(s2, batch)
                loss = loss_re
                loss_tp = loss_re.detach().new_zeros(())
            else:
                loss, loss_re, loss_tp = self.loss_fn(
                    s2,
                    None,
                    batch,
                )

        elif self.task_mode == "tp_only":
            if s4 is None:
                raise RuntimeError(
                    "tp_only requires Stage 4 outputs, but s4 is None."
                )

            # Prefer the dedicated TP criterion so the loss is not multiplied
            # by the joint-task weight (0.67).
            if hasattr(self.loss_fn, "tp_loss"):
                loss_tp = self.loss_fn.tp_loss(s4, batch)
                loss = loss_tp
                loss_re = loss_tp.detach().new_zeros(())
            else:
                loss, loss_re, loss_tp = self.loss_fn(
                    None,
                    s4,
                    batch,
                )

        else:
            if s2 is None or s4 is None:
                raise RuntimeError(
                    "joint mode requires both Stage 2 and Stage 4 outputs."
                )

            loss, loss_re, loss_tp = self.loss_fn(
                s2,
                s4,
                batch,
            )

        if not torch.is_tensor(loss):
            raise TypeError(
                "The active training loss must be a torch.Tensor, "
                f"but received {type(loss).__name__}."
            )

        if loss.ndim != 0:
            loss = loss.mean()

        if not loss.requires_grad:
            self._raise_detached_loss_error(
                s2=s2,
                s4=s4,
                batch=batch,
                total_loss=loss,
                re_objective=(
                    loss_re if self.task_mode != "tp_only" else None
                ),
                tp_objective=(
                    loss_tp if self.task_mode != "re_only" else None
                ),
                loss_re=loss_re,
                loss_tp=loss_tp,
            )

        return loss, loss_re, loss_tp

    def _raise_detached_loss_error(
        self,
        *,
        s2,
        s4,
        batch,
        total_loss,
        re_objective,
        tp_objective,
        loss_re,
        loss_tp,
    ):
        def grad_state(value):
            if value is None:
                return "None"
            if not torch.is_tensor(value):
                return type(value).__name__
            grad_fn_name = type(value.grad_fn).__name__ if value.grad_fn else None
            scalar = float(value.detach().float().item())
            return (
                f"requires_grad={value.requires_grad}, "
                f"grad_fn={grad_fn_name}, value={scalar:.6f}"
            )

        logits_p = s2.get("logits_P") if isinstance(s2, dict) else None
        logits_d = s2.get("logits_D") if isinstance(s2, dict) else None
        t_hat = s4.get("T_hat") if isinstance(s4, dict) else None

        raise RuntimeError(
            "No graph-connected training objective was found.\n"
            f"task_mode={self.task_mode}\n"
            f"torch.is_grad_enabled()={torch.is_grad_enabled()}\n"
            f"valid_R_P={self._count_valid_labels(batch.get('R_P'))}\n"
            f"valid_R_D={self._count_valid_labels(batch.get('R_D'))}\n"
            f"valid_T={self._count_valid_labels(batch.get('T'))}\n"
            f"total_loss: {grad_state(total_loss)}\n"
            f"re_objective: {grad_state(re_objective)}\n"
            f"tp_objective: {grad_state(tp_objective)}\n"
            f"loss_re: {grad_state(loss_re)}\n"
            f"loss_tp: {grad_state(loss_tp)}\n"
            f"logits_P: {grad_state(logits_p)}\n"
            f"logits_D: {grad_state(logits_d)}\n"
            f"T_hat: {grad_state(t_hat)}\n"
            "Check the loss implementation and requires_grad configuration."
        )

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _trainable_parameters(modules: Iterable[torch.nn.Module]):
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def _dot_and_norms(grads_a, grads_b):
        dot = None
        norm_a = None
        norm_b = None

        for grad_a, grad_b in zip(grads_a, grads_b):
            if grad_a is None or grad_b is None:
                continue
            current_dot = torch.sum(grad_a * grad_b)
            current_a = torch.sum(grad_a * grad_a)
            current_b = torch.sum(grad_b * grad_b)
            dot = current_dot if dot is None else dot + current_dot
            norm_a = current_a if norm_a is None else norm_a + current_a
            norm_b = current_b if norm_b is None else norm_b + current_b

        if dot is None:
            device = None
            for grad in list(grads_a) + list(grads_b):
                if grad is not None:
                    device = grad.device
                    break
            device = device or torch.device("cpu")
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        return dot, norm_a, norm_b

    def _gradient_diagnostics(self, re_objective, tp_objective):
        if (
            not torch.is_tensor(re_objective)
            or not re_objective.requires_grad
            or not torch.is_tensor(tp_objective)
            or not tp_objective.requires_grad
        ):
            return {
                "grad_re_norm": 0.0,
                "grad_tp_norm": 0.0,
                "grad_cosine": 0.0,
            }

        shared_params = self._trainable_parameters([self.stage1])
        re_grads = torch.autograd.grad(
            re_objective,
            shared_params,
            retain_graph=True,
            allow_unused=True,
        )
        tp_grads = torch.autograd.grad(
            tp_objective,
            shared_params,
            retain_graph=True,
            allow_unused=True,
        )
        dot, norm_re_sq, norm_tp_sq = self._dot_and_norms(re_grads, tp_grads)
        norm_re = torch.sqrt(norm_re_sq.clamp_min(0.0))
        norm_tp = torch.sqrt(norm_tp_sq.clamp_min(0.0))
        cosine = dot / (norm_re * norm_tp + 1e-12)
        return {
            "grad_re_norm": float(norm_re.detach().item()),
            "grad_tp_norm": float(norm_tp.detach().item()),
            "grad_cosine": float(cosine.detach().item()),
        }

    def _pcgrad_backward(self, re_objective, tp_objective):
        """Symmetric two-task PCGrad on Stage 1/2 shared parameters."""
        if not re_objective.requires_grad or not tp_objective.requires_grad:
            raise RuntimeError(
                "PCGrad requires graph-connected RE and TP objectives."
            )

        shared_params = self._trainable_parameters([self.stage1, self.stage2])
        re_specific = []
        tp_specific = self._trainable_parameters([self.stage3, self.stage4])

        re_all = shared_params + re_specific
        tp_all = shared_params + tp_specific

        re_grads_all = torch.autograd.grad(
            re_objective,
            re_all,
            retain_graph=True,
            allow_unused=True,
        )
        tp_grads_all = torch.autograd.grad(
            tp_objective,
            tp_all,
            retain_graph=False,
            allow_unused=True,
        )

        n_shared = len(shared_params)
        re_shared = re_grads_all[:n_shared]
        tp_shared = tp_grads_all[:n_shared]
        dot, norm_re_sq, norm_tp_sq = self._dot_and_norms(
            re_shared,
            tp_shared,
        )

        conflict = bool(dot.detach().item() < 0.0)
        coefficient_re = dot / (norm_tp_sq + 1e-12)
        coefficient_tp = dot / (norm_re_sq + 1e-12)

        for parameter, grad_re, grad_tp in zip(
            shared_params,
            re_shared,
            tp_shared,
        ):
            if grad_re is None and grad_tp is None:
                parameter.grad = None
                continue
            if grad_re is None:
                parameter.grad = grad_tp.detach().clone()
                continue
            if grad_tp is None:
                parameter.grad = grad_re.detach().clone()
                continue

            if conflict:
                projected_re = grad_re - coefficient_re * grad_tp
                projected_tp = grad_tp - coefficient_tp * grad_re
                combined = projected_re + projected_tp
            else:
                combined = grad_re + grad_tp
            parameter.grad = combined.detach().clone()

        for parameter, grad in zip(re_specific, re_grads_all[n_shared:]):
            parameter.grad = None if grad is None else grad.detach().clone()
        for parameter, grad in zip(tp_specific, tp_grads_all[n_shared:]):
            parameter.grad = None if grad is None else grad.detach().clone()

        norm_re = torch.sqrt(norm_re_sq.clamp_min(0.0))
        norm_tp = torch.sqrt(norm_tp_sq.clamp_min(0.0))
        cosine = dot / (norm_re * norm_tp + 1e-12)
        return {
            "grad_re_norm": float(norm_re.detach().item()),
            "grad_tp_norm": float(norm_tp.detach().item()),
            "grad_cosine": float(cosine.detach().item()),
            "pcgrad_conflict": float(conflict),
        }

    def _parameters_with_grad(self):
        return [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        parameters = self._parameters_with_grad()
        if parameters:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, loader, epoch):
        self._set_train_mode()

        totals = {
            "loss": 0.0,
            "loss_re": 0.0,
            "loss_tp": 0.0,
            "steps": 0,
            "skipped_steps": 0,
            "optimizer_steps": 0,
            "grad_re_norm": 0.0,
            "grad_tp_norm": 0.0,
            "grad_cosine": 0.0,
            "pcgrad_conflict": 0.0,
            "grad_samples": 0,
        }

        self.optimizer.zero_grad(set_to_none=True)
        accumulated_micro_steps = 0

        with torch.enable_grad():
            for step, batch in enumerate(loader):
                batch = self._move_batch(batch)

                if step % 50 == 0:
                    num_u = batch["U_input_ids"].size(0)
                    num_p = batch["P_input_ids"].size(0)
                    num_d = batch["D_input_ids"].size(0)
                    tokens_u = int(batch["U_attention_mask"].sum().item())
                    tokens_p = int(batch["P_attention_mask"].sum().item())
                    tokens_d = int(batch["D_attention_mask"].sum().item())
                    print(
                        f"[Epoch {epoch} | Step {step}/{len(loader)}] "
                        f"cases={num_u} | P={num_p} | D={num_d} | "
                        f"tokens U/P/D={tokens_u}/{tokens_p}/{tokens_d}",
                        flush=True,
                    )

                if self.task_mode == "re_only" and not self._has_re_supervision(batch):
                    totals["skipped_steps"] += 1
                    continue
                if self.task_mode == "tp_only" and not self._has_tp_supervision(batch):
                    totals["skipped_steps"] += 1
                    continue

                if self.gradient_method == "pcgrad":
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(self.device_type, enabled=False):
                        s1, s2, s3, s4 = self._forward_batch(
                            batch,
                            epoch=epoch,
                        )
                        re_objective, tp_objective, loss_re, loss_tp = (
                            self.loss_fn.task_objectives(s2, s4, batch)
                        )
                    diag = self._pcgrad_backward(re_objective, tp_objective)

                    parameters = self._parameters_with_grad()
                    if parameters:
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            self.max_grad_norm,
                        )
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    loss = re_objective + tp_objective
                    totals["optimizer_steps"] += 1
                    for key, value in diag.items():
                        totals[key] += value
                    totals["grad_samples"] += 1

                else:
                    with torch.amp.autocast(
                        self.device_type,
                        enabled=self.use_amp,
                    ):
                        s1, s2, s3, s4 = self._forward_batch(
                            batch,
                            epoch=epoch,
                        )
                        loss, loss_re, loss_tp = self._compute_task_losses(
                            s2,
                            s4,
                            batch,
                        )
                        scaled_loss = loss / self.grad_accum_steps

                    if (
                        self.task_mode == "joint"
                        and self.grad_diagnostics_every > 0
                        and step % self.grad_diagnostics_every == 0
                    ):
                        re_objective, tp_objective, _, _ = (
                            self.loss_fn.task_objectives(s2, s4, batch)
                        )
                        diag = self._gradient_diagnostics(
                            re_objective,
                            tp_objective,
                        )
                        for key, value in diag.items():
                            totals[key] += value
                        totals["grad_samples"] += 1

                    self.scaler.scale(scaled_loss).backward()
                    accumulated_micro_steps += 1

                    if accumulated_micro_steps >= self.grad_accum_steps:
                        self._optimizer_step()
                        totals["optimizer_steps"] += 1
                        accumulated_micro_steps = 0

                totals["loss"] += float(loss.detach().float().item())
                totals["loss_re"] += float(loss_re.detach().float().item())
                totals["loss_tp"] += float(loss_tp.detach().float().item())
                totals["steps"] += 1

        if self.gradient_method == "standard" and accumulated_micro_steps > 0:
            correction = self.grad_accum_steps / accumulated_micro_steps
            if correction != 1.0:
                for parameter in self._parameters_with_grad():
                    parameter.grad.mul_(correction)
            self._optimizer_step()
            totals["optimizer_steps"] += 1

        steps = max(1, totals["steps"])
        grad_samples = max(1, totals["grad_samples"])
        self.last_train_stats = {
            "loss": totals["loss"] / steps,
            "loss_re": totals["loss_re"] / steps,
            "loss_tp": totals["loss_tp"] / steps,
            "processed_steps": totals["steps"],
            "skipped_steps": totals["skipped_steps"],
            "optimizer_steps": totals["optimizer_steps"],
            "grad_re_norm": totals["grad_re_norm"] / grad_samples,
            "grad_tp_norm": totals["grad_tp_norm"] / grad_samples,
            "grad_cosine": totals["grad_cosine"] / grad_samples,
            "pcgrad_conflict_rate": (
                totals["pcgrad_conflict"] / grad_samples
            ),
        }
        return self.last_train_stats["loss"]

    # ------------------------------------------------------------------
    # Evaluation and prediction
    # ------------------------------------------------------------------

    def evaluate(self, loader, rationale_source=None, return_dict=False):
        self._set_eval_mode()

        re_preds_p, re_labels_p = [], []
        re_preds_d, re_labels_d = [], []
        td_preds, td_labels = [], []
        gate_p, gate_d = [], []
        rationale_scales = []
        global_logits = []
        rationale_deltas = []

        source = rationale_source or self.eval_rationale_source

        with torch.inference_mode():
            for batch in loader:
                batch = self._move_batch(batch)
                with torch.amp.autocast(
                    self.device_type,
                    enabled=self.use_amp,
                ):
                    s1, s2, s3, s4 = self._forward_batch(
                        batch,
                        epoch=0,
                        rationale_source=source,
                    )

                if s2 is not None:
                    re_preds_p.append(s2["rP_hat"].detach().cpu())
                    re_labels_p.append(batch["R_P"].detach().cpu())
                    re_preds_d.append(s2["rD_hat"].detach().cpu())
                    re_labels_d.append(batch["R_D"].detach().cpu())
                if s4 is not None:
                    td_preds.append(s4["T_hat"].detach().cpu())
                    td_labels.append(batch["T"].detach().cpu())
                    if "rationale_scale" in s4:
                        scale = s4["rationale_scale"].detach().float().reshape(-1)
                        rationale_scales.append(scale.cpu())
                    if "global_T_logit" in s4:
                        global_logits.append(
                            s4["global_T_logit"].detach().float().reshape(-1).cpu()
                        )
                    if "rationale_delta_logit" in s4:
                        rationale_deltas.append(
                            s4["rationale_delta_logit"].detach().float().reshape(-1).cpu()
                        )
                if s3 is not None:
                    gate_p.append(s3["mix_gate_P"].reshape(-1).detach().cpu())
                    gate_d.append(s3["mix_gate_D"].reshape(-1).detach().cpu())

        def cat_or_empty(items):
            return torch.cat(items) if items else torch.empty(0)

        p_pred = cat_or_empty(re_preds_p)
        p_label = cat_or_empty(re_labels_p)
        d_pred = cat_or_empty(re_preds_d)
        d_label = cat_or_empty(re_labels_d)
        t_pred = cat_or_empty(td_preds)
        t_label = cat_or_empty(td_labels)

        re_p_f1 = (
            compute_re_f1(p_pred, p_label)
            if p_pred.numel()
            else float("nan")
        )
        re_d_f1 = (
            compute_re_f1(d_pred, d_label)
            if d_pred.numel()
            else float("nan")
        )
        if p_pred.numel() or d_pred.numel():
            re_f1 = compute_re_f1(
                torch.cat(
                    [value for value in (p_pred, d_pred) if value.numel()]
                ),
                torch.cat(
                    [value for value in (p_label, d_label) if value.numel()]
                ),
            )
        else:
            re_f1 = float("nan")

        tp_acc = (
            compute_td_accuracy(t_pred, t_label)
            if t_pred.numel()
            else float("nan")
        )

        gp = cat_or_empty(gate_p)
        gd = cat_or_empty(gate_d)
        rs = cat_or_empty(rationale_scales)
        gl = cat_or_empty(global_logits)
        rd = cat_or_empty(rationale_deltas)
        self.last_eval = {
            "re_f1": re_f1,
            "re_p_f1": re_p_f1,
            "re_d_f1": re_d_f1,
            "tp_acc": tp_acc,
            "rationale_source": source,
            "gate_p_mean": float(gp.mean().item()) if gp.numel() else float("nan"),
            "gate_d_mean": float(gd.mean().item()) if gd.numel() else float("nan"),
            "gate_p_saturation": (
                float(((gp < 0.1) | (gp > 0.9)).float().mean().item())
                if gp.numel()
                else float("nan")
            ),
            "gate_d_saturation": (
                float(((gd < 0.1) | (gd > 0.9)).float().mean().item())
                if gd.numel()
                else float("nan")
            ),
            "rationale_scale": (
                float(rs.mean().item()) if rs.numel() else float("nan")
            ),
            "global_logit_abs_mean": (
                float(gl.abs().mean().item()) if gl.numel() else float("nan")
            ),
            "rationale_delta_abs_mean": (
                float(rd.abs().mean().item()) if rd.numel() else float("nan")
            ),
        }

        if return_dict:
            return dict(self.last_eval)
        return re_f1, tp_acc

    def predict(self, loader, rationale_source=None):
        self._set_eval_mode()
        source = rationale_source or self.eval_rationale_source
        predictions = []

        with torch.inference_mode():
            for batch in loader:
                batch = self._move_batch(batch)
                with torch.amp.autocast(
                    self.device_type,
                    enabled=self.use_amp,
                ):
                    _, _, _, s4 = self._forward_batch(
                        batch,
                        epoch=0,
                        rationale_source=source,
                    )

                if s4 is None:
                    continue

                t_predictions = s4["T_hat"].detach().cpu()
                tort_ids = batch["tort_id"]
                for index, tort_id in enumerate(tort_ids):
                    predictions.append(
                        {
                            "tort_id": tort_id,
                            "T_hat": float(t_predictions[index]),
                        }
                    )

        return predictions
