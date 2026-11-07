from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base_transcoder import BaseTranscoder
from src.utils.config import compute_fvu, set_global_seed


class _JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        gate_hard = (pre_acts > threshold).to(dtype=pre_acts.dtype)
        ctx.save_for_backward(pre_acts, threshold, gate_hard)
        ctx.bandwidth = float(bandwidth)
        return pre_acts * gate_hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold, gate_hard = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        grad_pre_acts = grad_output * gate_hard
        if bandwidth <= 0.0:
            grad_threshold = None
        else:
            scaled = (pre_acts.float() - threshold.float()) / (
                bandwidth * math.sqrt(2.0)
            )
            pdf = torch.exp(-(scaled**2)) / (
                bandwidth * math.sqrt(2.0 * math.pi)
            )
            pdf = pdf.to(dtype=pre_acts.dtype)
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_threshold = -(grad_output * pre_acts * pdf).sum(dim=reduce_dims)
        return grad_pre_acts, grad_threshold, None


class FlatTranscoder(BaseTranscoder):
    """Flat transcoder with explicit activation semantics.

    Supported activation types:
      - `relu`: dense ReLU features
      - `topk`: per-token TopK features
      - `batchtopk`: global BatchTopK during training, with configurable eval mode
      - `jumprelu`: JumpReLU with hard gating and smoothed threshold gradients

    For `batchtopk`, the recommended eval mode is `threshold`, which mirrors the
    Matryoshka SAE implementation and provides stable single-example inference.
    Set `batchtopk_inference="batchtopk"` if you explicitly want batch-dependent
    eval behavior.
    """

    VALID_ACTIVATIONS = {"relu", "topk", "batchtopk", "jumprelu"}
    VALID_BATCHTOPK_INFERENCE = {"threshold", "batchtopk"}

    def __init__(self, cfg):
        super().__init__(cfg)
        set_global_seed(cfg.get("seed", 42))

        self.d_in = int(cfg.get("source_act_size", cfg["act_size"]))
        self.d_out = int(cfg.get("target_act_size", cfg["act_size"]))
        self.dict_size = int(cfg["dict_size"])
        self.d_sae = self.dict_size

        self.activation_type = str(cfg.get("activation_type", "batchtopk")).lower()
        if self.activation_type not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation_type={self.activation_type!r}; expected one of "
                f"{sorted(self.VALID_ACTIVATIONS)}."
            )

        self.k = int(cfg.get("k", cfg.get("top_k", 0)) or 0)
        if self.activation_type in {"topk", "batchtopk"} and self.k <= 0:
            raise ValueError(
                f"activation_type={self.activation_type!r} requires a positive k/top_k."
            )

        self.batchtopk_inference = str(
            cfg.get("batchtopk_inference", cfg.get("batchtopk_eval_mode", "threshold"))
        ).lower()
        if self.batchtopk_inference not in self.VALID_BATCHTOPK_INFERENCE:
            raise ValueError(
                "batchtopk_inference must be one of "
                f"{sorted(self.VALID_BATCHTOPK_INFERENCE)}."
            )

        self.subtract_decoder_bias_from_input = bool(
            cfg.get("subtract_decoder_bias_from_input", False)
        )
        self.input_scale = float(cfg.get("input_scale", 1.0))
        self.output_scale = float(cfg.get("output_scale", 1.0))
        self.config["input_scale"] = self.input_scale
        self.config["output_scale"] = self.output_scale

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.d_in, self.dict_size))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.dict_size, self.d_out))
        )
        self.b_dec = nn.Parameter(torch.zeros(self.d_out))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.init_skip_connection(cfg, self.d_in, self.d_out)

        # Joint x+y objective: an auxiliary source-space decoder head adds an
        # x-reconstruction loss alongside the y loss, mirroring the Matryoshka
        # joint head with a single prefix. Train-time only; encode/decode and
        # the eval adapter never touch this head.
        self.joint_source_weight = float(cfg.get("joint_source_weight", 0.0) or 0.0)
        cfg["joint_source_weight"] = self.joint_source_weight
        if self.joint_source_weight > 0.0:
            self.W_dec_src = nn.Parameter(
                torch.nn.init.kaiming_uniform_(torch.empty(self.dict_size, self.d_in))
            )
            self.b_dec_src = nn.Parameter(torch.zeros(self.d_in))
            self.W_dec_src.data[:] = self.W_dec_src / self.W_dec_src.norm(dim=-1, keepdim=True)

        if self.activation_type == "jumprelu":
            threshold_init = float(cfg.get("jumprelu_threshold_init", 1e-3))
            threshold_init = max(threshold_init, 1e-8)
            raw_init = math.log(math.expm1(threshold_init))
            self.threshold = nn.Parameter(torch.full((self.dict_size,), raw_init))
        else:
            self.register_buffer(
                "threshold",
                torch.tensor(-1.0, dtype=torch.float32, device=cfg["device"]),
            )

        self.register_buffer(
            "num_batches_not_active",
            torch.zeros(self.dict_size, dtype=torch.float32, device=cfg["device"]),
            persistent=False,
        )
        self.to(cfg["dtype"]).to(cfg["device"])

        # Persist resolved values so checkpoints remain self-describing.
        self.config["activation_type"] = self.activation_type
        self.config["k"] = self.k
        self.config["top_k"] = self.k
        self.config["group_sizes"] = [self.dict_size]
        self.config["batchtopk_inference"] = self.batchtopk_inference

    @property
    def group_sizes(self):
        return [self.dict_size]

    @property
    def group_indices(self):
        return [0, self.dict_size]

    @property
    def active_groups(self):
        return 1

    def _center_source(self, x: torch.Tensor) -> torch.Tensor:
        if not self.subtract_decoder_bias_from_input:
            return x
        return x - self.b_dec

    @torch.no_grad()
    def update_threshold(self, sparse_acts: torch.Tensor):
        threshold_start_step = self.config.get("threshold_start_step")
        if threshold_start_step is not None:
            current_step = int(
                getattr(self, "training_step", self.config.get("training_step", 0))
            )
            if current_step <= int(threshold_start_step):
                return

        positive_mask = sparse_acts > 0
        if not positive_mask.any():
            return

        min_positive = sparse_acts[positive_mask].min().detach().to(self.threshold.dtype)
        beta = float(self.config.get("threshold_beta", 0.99))
        beta = min(1.0, max(0.0, beta))
        if float(self.threshold.item()) < 0.0:
            self.threshold.copy_(min_positive)
        else:
            self.threshold.mul_(beta).add_(min_positive * (1.0 - beta))

    @torch.no_grad()
    def update_inactive_features(self, sparse_acts: torch.Tensor):
        dead_threshold = float(self.config.get("dead_feature_threshold", 1e-8))
        max_acts = sparse_acts.max(dim=0).values
        inactive = max_acts < dead_threshold
        self.num_batches_not_active += inactive.float()
        self.num_batches_not_active[~inactive] = 0

    def _reconstruction_loss(self, reconstruction: torch.Tensor, target: torch.Tensor):
        diff = reconstruction.float() - target.float()
        if str(self.config.get("loss_reduction", "mean")).lower() == "sum_mean":
            return diff.pow(2).sum(dim=-1).mean()
        return diff.pow(2).mean()

    def _auxk_reconstruction_loss(
        self, aux_reconstruction: torch.Tensor, residual: torch.Tensor
    ):
        loss = self._reconstruction_loss(aux_reconstruction, residual)
        if not bool(self.config.get("auxk_normalize", False)):
            return loss

        residual_mean = residual.float().mean(dim=0, keepdim=True)
        denom = self._reconstruction_loss(
            residual.float(), residual_mean.expand_as(residual)
        )
        eps = float(self.config.get("auxk_normalize_eps", 1e-8))
        return (loss / denom.clamp_min(eps)).nan_to_num(0.0)

    def _relu_activations(self, pre_acts: torch.Tensor):
        sparse_acts = F.relu(pre_acts)
        l0_proxy = (sparse_acts > 0).to(dtype=pre_acts.dtype)
        return sparse_acts, l0_proxy

    def _topk_activations(self, pre_acts: torch.Tensor):
        acts = F.relu(pre_acts)
        if self.k >= acts.shape[-1]:
            sparse_acts = acts
        else:
            topk_vals, topk_idx = torch.topk(acts, self.k, dim=-1)
            sparse_acts = torch.zeros_like(acts).scatter(-1, topk_idx, topk_vals)
        l0_proxy = (sparse_acts > 0).to(dtype=pre_acts.dtype)
        return sparse_acts, l0_proxy

    def _batchtopk_mask(self, acts: torch.Tensor) -> torch.Tensor:
        k_total = min(self.k * acts.shape[0], acts.numel())
        if k_total <= 0:
            return torch.zeros_like(acts)
        topk = torch.topk(acts.flatten(), k_total, dim=-1)
        return torch.zeros_like(acts.flatten()).scatter(-1, topk.indices, topk.values).reshape(
            acts.shape
        )

    def _batchtopk_activations(self, pre_acts: torch.Tensor):
        acts = F.relu(pre_acts)

        if self.training or self.batchtopk_inference == "batchtopk":
            sparse_acts = self._batchtopk_mask(acts)
            if self.training:
                self.update_threshold(sparse_acts)
        else:
            if float(self.threshold.item()) < 0.0:
                raise ValueError(
                    "BatchTopK threshold is unset. Load a trained checkpoint or run training first, "
                    "or set batchtopk_inference=\"batchtopk\"."
                )
            sparse_acts = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        l0_proxy = (sparse_acts > 0).to(dtype=pre_acts.dtype)
        return sparse_acts, l0_proxy

    def _jumprelu_activations(self, pre_acts: torch.Tensor):
        threshold = F.softplus(self.threshold)
        gate_hard = (pre_acts > threshold).to(dtype=pre_acts.dtype)
        bandwidth = float(self.config.get("jumprelu_kde_bandwidth", 1e-3))
        bandwidth = max(bandwidth, 1e-6)

        if self.training:
            scaled = (pre_acts.float() - threshold.float()) / (
                bandwidth * math.sqrt(2.0)
            )
            gate_soft = 0.5 * (1.0 + torch.erf(scaled))
            l0_proxy = gate_soft.to(dtype=pre_acts.dtype)
            sparse_acts = _JumpReLUFunction.apply(pre_acts, threshold, bandwidth)
        else:
            l0_proxy = gate_hard
            sparse_acts = pre_acts * gate_hard

        return sparse_acts, l0_proxy

    def compute_activations(self, x_source: torch.Tensor):
        x_centered = self._center_source(x_source)
        pre_acts = x_centered @ self.W_enc + self.b_enc

        if self.activation_type == "relu":
            sparse_acts, l0_proxy = self._relu_activations(pre_acts)
        elif self.activation_type == "topk":
            sparse_acts, l0_proxy = self._topk_activations(pre_acts)
        elif self.activation_type == "batchtopk":
            sparse_acts, l0_proxy = self._batchtopk_activations(pre_acts)
        else:
            sparse_acts, l0_proxy = self._jumprelu_activations(pre_acts)

        return pre_acts, sparse_acts, l0_proxy

    def _get_auxiliary_loss(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        pre_acts: torch.Tensor,
    ) -> torch.Tensor:
        dead_steps = int(
            self.config.get(
                "dead_steps_threshold", self.config.get("n_batches_to_dead", 20)
            )
        )
        dead_mask = self.num_batches_not_active >= dead_steps
        if dead_mask.sum() == 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        residual = target.float() - reconstruction.float()
        dead_pre = F.relu(pre_acts[:, dead_mask])
        if dead_pre.shape[1] == 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        top_k_aux = int(self.config.get("top_k_aux", max(1, self.k * 8)))
        if top_k_aux <= 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        k_aux = min(top_k_aux, dead_pre.shape[1])
        topk_vals, topk_idx = torch.topk(dead_pre, k_aux, dim=-1)
        dead_sparse = torch.zeros_like(dead_pre).scatter(-1, topk_idx, topk_vals)
        aux_recon = dead_sparse @ self.W_dec[dead_mask]
        aux_penalty = float(self.config.get("aux_penalty", 1 / 32))
        return aux_penalty * self._auxk_reconstruction_loss(aux_recon, residual)

    def _get_ghost_grad_loss(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        pre_acts: torch.Tensor,
        recon_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Ghost-grad style rescue loss for dead features.

        This is off by default. It gives features that have been inactive for
        `n_batches_to_dead` batches a differentiable activation path toward the
        current reconstruction residual without changing the normal forward pass.
        """
        ghost_penalty = float(self.config.get("ghost_grad_penalty", 0.0) or 0.0)
        if ghost_penalty <= 0.0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        dead_steps = int(
            self.config.get(
                "dead_steps_threshold", self.config.get("n_batches_to_dead", 20)
            )
        )
        dead_mask = self.num_batches_not_active >= dead_steps
        if dead_mask.sum() == 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        residual = (target.float() - reconstruction.float()).detach()
        ghost_pre = pre_acts[:, dead_mask]
        if ghost_pre.shape[1] == 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        if self.activation_type == "jumprelu":
            threshold = F.softplus(self.threshold)[dead_mask]
            ghost_pre = ghost_pre - threshold

        max_pre = float(self.config.get("ghost_grad_max_pre", 10.0))
        ghost_acts = torch.exp(ghost_pre.float().clamp(max=max_pre))
        ghost_recon = ghost_acts @ self.W_dec[dead_mask].float()

        eps = float(self.config.get("ghost_grad_eps", 1e-6))
        norm_ratio = float(self.config.get("ghost_grad_target_norm_ratio", 0.5))
        residual_norm = residual.norm(dim=-1, keepdim=True)
        ghost_norm = ghost_recon.norm(dim=-1, keepdim=True).clamp_min(eps)
        ghost_scale = (norm_ratio * residual_norm / ghost_norm).detach()
        ghost_recon = ghost_recon * ghost_scale

        raw_loss = (ghost_recon - residual).pow(2).mean()
        raw_loss = torch.nan_to_num(raw_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if bool(self.config.get("ghost_grad_rescale_loss", True)):
            denom = raw_loss.detach().clamp_min(eps)
            raw_loss = raw_loss * (recon_loss.detach() / denom)

        return ghost_penalty * raw_loss

    def forward(self, x_source: torch.Tensor, x_target: torch.Tensor):
        source_shape = x_source.shape
        raw_source_flat = x_source.reshape(-1, x_source.shape[-1])

        x_source, scale = self.preprocess_input(x_source)
        x_target = self.preprocess_target(x_target, scale)

        source_flat = x_source.reshape(-1, x_source.shape[-1])
        target_flat = x_target.reshape(-1, x_target.shape[-1])
        skip_scaled = None
        residual_target_flat = target_flat
        if self.has_skip_connection:
            skip_raw = self.compute_skip(raw_source_flat)
            skip_scaled = self.preprocess_target(skip_raw, scale).to(target_flat.dtype)
            residual_target_flat = target_flat - skip_scaled

        pre_acts, sparse_acts, l0_proxy = self.compute_activations(source_flat)
        self.update_inactive_features(sparse_acts)

        residual_reconstruction_scaled = sparse_acts @ self.W_dec + self.b_dec
        if skip_scaled is None:
            reconstruction_scaled = residual_reconstruction_scaled
        else:
            reconstruction_scaled = residual_reconstruction_scaled + skip_scaled
        reconstruction = self.postprocess_output(reconstruction_scaled, scale)
        if len(source_shape) == 3:
            reconstruction = reconstruction.reshape(source_shape[0], source_shape[1], -1)

        l2_loss = self._reconstruction_loss(
            residual_reconstruction_scaled, residual_target_flat
        )
        aux_loss = self._get_auxiliary_loss(
            residual_target_flat, residual_reconstruction_scaled, pre_acts
        )
        ghost_grad_loss = self._get_ghost_grad_loss(
            residual_target_flat, residual_reconstruction_scaled, pre_acts, l2_loss
        )

        act_l1_coef = float(self.config.get("act_l1_coef", 0.0) or 0.0)
        if act_l1_coef != 0.0 and self.activation_type != "jumprelu":
            l1_loss = act_l1_coef * sparse_acts.float().abs().sum(dim=-1).mean()
        else:
            l1_loss = torch.tensor(0.0, dtype=target_flat.dtype, device=target_flat.device)

        act_l0_coef = float(
            self.config.get("act_l0_coef_current", self.config.get("act_l0_coef", 0.0))
            or 0.0
        )
        if act_l0_coef != 0.0 and self.activation_type == "jumprelu":
            l0_loss = act_l0_coef * l0_proxy.float().sum(dim=-1).mean()
        else:
            l0_loss = torch.tensor(0.0, dtype=target_flat.dtype, device=target_flat.device)

        joint_source_loss = None
        if self.joint_source_weight > 0.0:
            src_recon = sparse_acts @ self.W_dec_src + self.b_dec_src
            src_l2_loss = self._reconstruction_loss(src_recon, source_flat)
            # Variance-matched scaling keeps the y-loss recipe identical to the
            # baseline while making joint_source_weight=1.0 mean "equal weight".
            with torch.no_grad():
                target_var = self._reconstruction_loss(
                    target_flat.float().mean(dim=0, keepdim=True).expand_as(target_flat), target_flat
                )
                source_var = self._reconstruction_loss(
                    source_flat.float().mean(dim=0, keepdim=True).expand_as(source_flat), source_flat
                )
                var_ratio = (target_var / source_var.clamp_min(1e-8)).nan_to_num(1.0)
            joint_source_loss = self.joint_source_weight * var_ratio * src_l2_loss

        l0_norm = (sparse_acts > 0).sum(dim=-1).float().mean()
        fvu_y = compute_fvu(target_flat, reconstruction_scaled)
        fvu_r = compute_fvu(residual_target_flat, residual_reconstruction_scaled)
        if skip_scaled is None:
            fvu_skip = torch.tensor(float("nan"), device=target_flat.device)
        else:
            fvu_skip = compute_fvu(target_flat, skip_scaled)

        total_loss = l2_loss + aux_loss + ghost_grad_loss + l1_loss + l0_loss
        if joint_source_loss is not None:
            total_loss = total_loss + joint_source_loss

        output = {
            "transcoder_out": reconstruction,
            "loss": total_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "aux_loss": aux_loss,
            "ghost_grad_loss": ghost_grad_loss,
            "l1_loss": l1_loss,
            "l0_loss": l0_loss,
            "fvu": fvu_y,
            "fvu_y": fvu_y,
            "fvu_r": fvu_r,
            "min_l2_loss": l2_loss,
            "max_l2_loss": l2_loss,
            "fvu_min": fvu_y,
            "fvu_max": fvu_y,
            "fvu_mean": fvu_y,
        }
        if self.has_skip_connection:
            main_eps = float(self.config.get("loss_ratio_eps", 1e-12))
            output.update(
                {
                    "fvu_skip": fvu_skip,
                    "aux_to_main_loss": aux_loss / l2_loss.detach().clamp_min(main_eps),
                    "residual_target_l2": residual_target_flat.float()
                    .pow(2)
                    .sum(dim=-1)
                    .mean()
                    .sqrt(),
                    "target_l2": target_flat.float().pow(2).sum(dim=-1).mean().sqrt(),
                }
            )
        if joint_source_loss is not None:
            output["joint_source_loss"] = joint_source_loss
        return output

    @torch.no_grad()
    def project_decoder_grad(self):
        super().project_decoder_grad()
        if self.joint_source_weight > 0.0 and self.W_dec_src.grad is not None:
            normed = self.W_dec_src / self.W_dec_src.norm(dim=-1, keepdim=True)
            proj = (self.W_dec_src.grad * normed).sum(-1, keepdim=True) * normed
            self.W_dec_src.grad -= proj

    @torch.no_grad()
    def renorm_decoder_weights(self):
        super().renorm_decoder_weights()
        if (
            self.joint_source_weight > 0.0
            and not bool(self.config.get("freeze_decoder", False))
        ):
            self.W_dec_src.data = self.W_dec_src / self.W_dec_src.norm(dim=-1, keepdim=True)

    def encode(self, x_source: torch.Tensor) -> torch.Tensor:
        original_shape = x_source.shape
        self._cache_decode_source(x_source)
        x_source, scale = self.preprocess_input(x_source)
        self.input_scale = float(scale)

        source_flat = x_source.reshape(-1, x_source.shape[-1])
        _, sparse_acts, _ = self.compute_activations(source_flat)

        if len(original_shape) == 3:
            return sparse_acts.reshape(original_shape[0], original_shape[1], -1)
        return sparse_acts

    def decode(
        self,
        feature_acts: torch.Tensor,
        x_source: torch.Tensor | None = None,
        include_skip: bool = True,
    ) -> torch.Tensor:
        original_shape = feature_acts.shape
        flat_acts = feature_acts.reshape(-1, feature_acts.shape[-1])
        reconstruction = flat_acts @ self.W_dec + self.b_dec
        reconstruction = self.postprocess_output(reconstruction, self.output_scale)
        if include_skip and self.has_skip_connection:
            source = self._resolve_decode_source(x_source, feature_acts)
            flat_source = source.reshape(-1, source.shape[-1])
            skip = self.compute_skip(flat_source).to(reconstruction.dtype)
            reconstruction = reconstruction + skip
        if len(original_shape) == 3:
            return reconstruction.reshape(original_shape[0], original_shape[1], -1)
        return reconstruction

    def decode_residual(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return self.decode(feature_acts, include_skip=False)
