"""Matryoshka transcoder with nested group structure."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import compute_fvu, set_global_seed
from src.model.base_transcoder import BaseTranscoder


class MatryoshkaTranscoder(BaseTranscoder):
    """Transcoder with nested Matryoshka structure for multi-scale features."""

    def __init__(self, cfg):
        super().__init__(cfg)
        set_global_seed(cfg["seed"])

        group_sizes = cfg["group_sizes"]
        self.group_sizes = group_sizes
        self.group_weights = cfg.get("group_weights")
        if self.group_weights is None:
            self.group_weights = [1.0] * len(group_sizes)
        if len(self.group_weights) != len(group_sizes):
            raise ValueError("group_weights must match group_sizes length.")
        self.group_weights = [float(w) for w in self.group_weights]
        cfg["group_weights"] = list(self.group_weights)
        self.loss_aggregation = str(cfg.get("matryoshka_loss_aggregation", "sum")).lower()
        if self.loss_aggregation not in {"sum", "mean"}:
            raise ValueError("matryoshka_loss_aggregation must be 'sum' or 'mean'.")
        cfg["matryoshka_loss_aggregation"] = self.loss_aggregation
        total_dict_size = sum(group_sizes)
        self.group_indices = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0).tolist())
        self.active_groups = len(group_sizes)
        cfg["activation_type"] = "batchtopk"
        self.batchtopk_mode = cfg.get("batchtopk_mode", "global")
        if self.batchtopk_mode not in {"global", "groupwise"}:
            raise ValueError(f"Unknown batchtopk_mode={self.batchtopk_mode!r}; expected 'global' or 'groupwise'.")
        self.group_k_per_token = None
        if self.batchtopk_mode == "groupwise":
            shares = cfg.get("group_k_shares")
            if shares is None:
                total_size = float(sum(group_sizes))
                shares = [size / total_size for size in group_sizes]
            else:
                if len(shares) != len(group_sizes):
                    raise ValueError("group_k_shares must match group_sizes length.")
                total_share = float(sum(shares))
                if total_share <= 0:
                    raise ValueError("group_k_shares must sum to a positive value.")
                shares = [float(s) / total_share for s in shares]
            raw = [share * cfg["top_k"] for share in shares]
            base = [int(math.floor(val)) for val in raw]
            remainder = int(cfg["top_k"]) - sum(base)
            if remainder > 0:
                frac = [val - b for val, b in zip(raw, base)]
                order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
                for idx in order[:remainder]:
                    base[idx] += 1
            self.group_k_per_token = base

        self.source_act_size = cfg.get("source_act_size", cfg["act_size"])
        self.target_act_size = cfg.get("target_act_size", cfg["act_size"])
        self.input_scale = float(cfg.get("input_scale", 1.0))
        self.output_scale = float(cfg.get("output_scale", 1.0))
        self.config["input_scale"] = self.input_scale
        self.config["output_scale"] = self.output_scale

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.source_act_size, total_dict_size)))
        self.b_enc = nn.Parameter(torch.zeros(total_dict_size))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(total_dict_size, self.target_act_size)))
        self.b_dec = nn.Parameter(torch.zeros(self.target_act_size))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.init_skip_connection(cfg, self.source_act_size, self.target_act_size)

        # Joint x+y objective: an auxiliary source-space decoder head gives every
        # prefix an x-reconstruction loss alongside the y loss, so early groups
        # are also rewarded for being broad source-space parents. Train-time
        # only; encode/decode and the eval adapter never touch this head.
        self.joint_source_weight = float(cfg.get("joint_source_weight", 0.0) or 0.0)
        cfg["joint_source_weight"] = self.joint_source_weight
        if self.joint_source_weight > 0.0:
            self.W_dec_src = nn.Parameter(
                torch.nn.init.kaiming_uniform_(torch.empty(total_dict_size, self.source_act_size))
            )
            self.b_dec_src = nn.Parameter(torch.zeros(self.source_act_size))
            self.W_dec_src.data[:] = self.W_dec_src / self.W_dec_src.norm(dim=-1, keepdim=True)

        self.register_buffer(
            'num_batches_not_active',
            torch.zeros(total_dict_size, dtype=torch.float32, device=cfg["device"]),
            persistent=False,
        )
        self.register_buffer('threshold', torch.tensor(-1.0))
        self.to(cfg["dtype"]).to(cfg["device"])

    @property
    def d_in(self):
        return self.source_act_size

    @property
    def d_sae(self):
        return sum(self.group_sizes)

    @torch.no_grad()
    def update_inactive_features(self, acts):
        """Track dead features using max activation threshold."""
        dead_threshold = self.config.get("dead_feature_threshold", 1e-8)
        max_acts = acts.max(dim=0).values
        self.num_batches_not_active += (max_acts < dead_threshold).float()
        self.num_batches_not_active[max_acts >= dead_threshold] = 0

    @torch.no_grad()
    def update_threshold(self, sparse_acts):
        """Track EMA of min active activation for threshold gating in eval."""
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

    def _reconstruction_loss(self, reconstruction, target):
        diff = reconstruction.float() - target.float()
        if str(self.config.get("loss_reduction", "mean")).lower() == "sum_mean":
            return diff.pow(2).sum(dim=-1).mean()
        return diff.pow(2).mean()

    def _auxk_reconstruction_loss(self, aux_reconstruction, residual):
        loss = self._reconstruction_loss(aux_reconstruction, residual)
        if not bool(self.config.get("auxk_normalize", False)):
            return loss

        residual_mean = residual.float().mean(dim=0, keepdim=True)
        denom = self._reconstruction_loss(
            residual.float(), residual_mean.expand_as(residual)
        )
        eps = float(self.config.get("auxk_normalize_eps", 1e-8))
        return (loss / denom.clamp_min(eps)).nan_to_num(0.0)

    def compute_activations(self, x_source):
        """Compute activations: BatchTopK in training, threshold gating in eval.

        Returns (pre_acts, acts_topk): pre-ReLU activations and sparse post-TopK activations.
        Note: the transcoder does not center by `b_dec`; `x_source` is passed through as-is.
        """
        pre_acts = x_source @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)

        if self.training:
            if self.batchtopk_mode == "groupwise":
                acts_topk = torch.zeros_like(acts)
                batch_size = x_source.shape[0]
                for i, group_k in enumerate(self.group_k_per_token):
                    if group_k <= 0:
                        continue
                    start_idx, end_idx = self.group_indices[i], self.group_indices[i + 1]
                    group_acts = acts[:, start_idx:end_idx]
                    k_total = min(group_k * batch_size, group_acts.numel())
                    topk = torch.topk(group_acts.flatten(), k_total, dim=-1)
                    group_topk = torch.zeros_like(group_acts.flatten()).scatter(
                        -1, topk.indices, topk.values
                    ).reshape(group_acts.shape)
                    acts_topk[:, start_idx:end_idx] = group_topk
            else:
                k_total = self.config["top_k"] * x_source.shape[0]
                topk = torch.topk(acts.flatten(), k_total, dim=-1)
                acts_topk = torch.zeros_like(acts.flatten()).scatter(
                    -1, topk.indices, topk.values
                ).reshape(acts.shape)
            self.update_threshold(acts_topk)
            return pre_acts, acts_topk
        else:
            if float(self.threshold.item()) <= 0.0:
                # Fallback to BatchTopK if threshold was never set
                # (old checkpoints have threshold=0.0, new ones use -1.0 sentinel)
                k_total = self.config["top_k"] * x_source.shape[0]
                topk = torch.topk(acts.flatten(), k_total, dim=-1)
                acts_topk = torch.zeros_like(acts.flatten()).scatter(
                    -1, topk.indices, topk.values
                ).reshape(acts.shape)
                return pre_acts, acts_topk
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
            return pre_acts, acts_topk

    def forward(self, x_source, x_target):
        """Forward pass with intermediate reconstructions for each group."""
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

        all_pre_acts, all_acts_topk = self.compute_activations(source_flat)

        residual_reconstruct_scaled = self.b_dec
        intermediate_reconstructs = []
        intermediate_full_reconstructs = []
        for i in range(self.active_groups):
            start_idx, end_idx = self.group_indices[i], self.group_indices[i+1]
            residual_reconstruct_scaled = (
                all_acts_topk[:, start_idx:end_idx] @ self.W_dec[start_idx:end_idx, :] + residual_reconstruct_scaled
            )
            intermediate_reconstructs.append(residual_reconstruct_scaled)
            if skip_scaled is None:
                intermediate_full_reconstructs.append(residual_reconstruct_scaled)
            else:
                intermediate_full_reconstructs.append(residual_reconstruct_scaled + skip_scaled)

        x_reconstruct_scaled = intermediate_full_reconstructs[-1]
        x_reconstruct = self.postprocess_output(x_reconstruct_scaled, scale)
        if len(source_shape) == 3:
            x_reconstruct = x_reconstruct.reshape(source_shape[0], source_shape[1], -1)
        self.update_inactive_features(all_acts_topk)
        
        l2_losses = [
            self._reconstruction_loss(recon, residual_target_flat)
            for recon in intermediate_reconstructs
        ]
        fvus = [compute_fvu(target_flat, r) for r in intermediate_full_reconstructs]
        fvus_r = [
            compute_fvu(residual_target_flat, r) for r in intermediate_reconstructs
        ]

        weighted_l2_losses = [
            weight * loss for weight, loss in zip(self.group_weights, l2_losses)
        ]
        if self.loss_aggregation == "mean":
            prefix_l2_loss = torch.stack(weighted_l2_losses).mean()
        else:
            prefix_l2_loss = sum(weighted_l2_losses)
        aux_loss = self.get_auxiliary_loss(
            residual_target_flat,
            residual_reconstruct_scaled,
            all_pre_acts,
            intermediate_reconstructs,
        )

        joint_source_loss = None
        if self.joint_source_weight > 0.0:
            src_recon = self.b_dec_src
            src_l2_losses = []
            for i in range(self.active_groups):
                start_idx, end_idx = self.group_indices[i], self.group_indices[i + 1]
                src_recon = (
                    all_acts_topk[:, start_idx:end_idx] @ self.W_dec_src[start_idx:end_idx, :] + src_recon
                )
                src_l2_losses.append(self._reconstruction_loss(src_recon, source_flat))
            weighted_src_losses = [
                weight * loss for weight, loss in zip(self.group_weights, src_l2_losses)
            ]
            if self.loss_aggregation == "mean":
                src_prefix_loss = torch.stack(weighted_src_losses).mean()
            else:
                src_prefix_loss = sum(weighted_src_losses)
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
            joint_source_loss = self.joint_source_weight * var_ratio * src_prefix_loss

        l0_norm = (all_acts_topk > 0).sum(dim=-1).float().mean()
        final_fvu = compute_fvu(target_flat, x_reconstruct_scaled)
        final_fvu_r = compute_fvu(residual_target_flat, residual_reconstruct_scaled)
        if skip_scaled is None:
            fvu_skip = torch.tensor(float("nan"), device=target_flat.device)
        else:
            fvu_skip = compute_fvu(target_flat, skip_scaled)

        B = all_acts_topk.shape[0]
        k_total = self.config["top_k"] * B
        
        per_group_l0 = []
        zero_active_fractions = []
        nonzero_shares = []
        
        for i in range(self.active_groups):
            start_idx, end_idx = self.group_indices[i], self.group_indices[i+1]
            group_acts = all_acts_topk[:, start_idx:end_idx]
            group_l0_per_token = (group_acts > 0).sum(dim=-1).float()
            per_group_l0.append(group_l0_per_token.mean())
            zero_active_fractions.append((group_l0_per_token == 0).float().mean())
            group_nonzeros = (group_acts > 0).sum().float()
            nonzero_shares.append(group_nonzeros / max(1, k_total))
        
        per_prefix_mse = l2_losses

        total_loss = prefix_l2_loss + aux_loss
        if joint_source_loss is not None:
            total_loss = total_loss + joint_source_loss

        output_dict = {
            "transcoder_out": x_reconstruct,
            "loss": total_loss,
            "l2_loss": prefix_l2_loss,
            "min_l2_loss": min(l2_losses) if l2_losses else prefix_l2_loss,
            "max_l2_loss": max(l2_losses) if l2_losses else prefix_l2_loss,
            "l0_norm": l0_norm,
            "aux_loss": aux_loss,
            "fvu": final_fvu,
            "fvu_y": final_fvu,
            "fvu_r": final_fvu_r,
            "fvu_min": min(fvus) if fvus else final_fvu,
            "fvu_max": max(fvus) if fvus else final_fvu,
            "fvu_mean": sum(fvus) / len(fvus) if fvus else final_fvu,
        }
        if self.has_skip_connection:
            main_eps = float(self.config.get("loss_ratio_eps", 1e-12))
            output_dict.update(
                {
                    "fvu_skip": fvu_skip,
                    "aux_to_main_loss": aux_loss
                    / prefix_l2_loss.detach().clamp_min(main_eps),
                    "residual_target_l2": residual_target_flat.float()
                    .pow(2)
                    .sum(dim=-1)
                    .mean()
                    .sqrt(),
                    "target_l2": target_flat.float().pow(2).sum(dim=-1).mean().sqrt(),
                }
            )
        
        if joint_source_loss is not None:
            output_dict["joint_source_loss"] = joint_source_loss

        # Add diagnostic metrics
        for i in range(self.active_groups):
            output_dict[f"diagnostic/l0_group_{i}"] = per_group_l0[i]
            output_dict[f"diagnostic/zero_active_fraction_group_{i}"] = zero_active_fractions[i]
            output_dict[f"diagnostic/nonzero_share_group_{i}"] = nonzero_shares[i]
            output_dict[f"diagnostic/prefix_mse_{i}"] = per_prefix_mse[i]
            output_dict[f"diagnostic/prefix_fvu_{i}"] = fvus[i]
            output_dict[f"diagnostic/prefix_fvu_r_{i}"] = fvus_r[i]
        
        return output_dict

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

    def get_auxiliary_loss(self, x_target, x_reconstruct, pre_acts, intermediate_reconstructs=None):
        """Per-group auxiliary loss for dead feature revival.

        When aux_residual_mode="per_prefix" (and intermediate_reconstructs is
        provided), each group's dead features target the residual from their
        own prefix reconstruction rather than the final reconstruction.  This
        gives early-group dead features a stronger, hierarchy-aligned revival
        signal.  Default ("final") uses the final residual for all groups,
        matching Bussmann et al.
        """
        aux_penalty = float(self.config.get("aux_penalty", 1 / 32))
        if aux_penalty <= 0.0:
            return torch.tensor(0.0, dtype=x_target.dtype, device=x_target.device)

        dead_steps = int(self.config.get("n_batches_to_dead", 20))
        dead_mask = self.num_batches_not_active >= dead_steps
        if dead_mask.sum() == 0:
            return torch.tensor(0.0, dtype=x_target.dtype, device=x_target.device)

        top_k_aux = int(self.config.get("top_k_aux", self.config["top_k"] * 8))
        if top_k_aux <= 0:
            return torch.tensor(0.0, dtype=x_target.dtype, device=x_target.device)

        if self.config.get("auxk_mode", "per_group") == "global":
            residual = x_target.float() - x_reconstruct.float()
            dead_pre = F.relu(pre_acts[:, dead_mask])
            if dead_pre.shape[1] == 0:
                return torch.tensor(0.0, dtype=x_target.dtype, device=x_target.device)

            k_aux = min(top_k_aux, dead_pre.shape[1])
            topk_vals, topk_idx = torch.topk(dead_pre, k_aux, dim=-1)
            dead_sparse = torch.zeros_like(dead_pre).scatter(-1, topk_idx, topk_vals)
            aux_recon = dead_sparse @ self.W_dec[dead_mask]
            return aux_penalty * self._auxk_reconstruction_loss(aux_recon, residual)

        use_per_prefix = (
            self.config.get("aux_residual_mode", "final") == "per_prefix"
            and intermediate_reconstructs is not None
        )
        if not use_per_prefix:
            final_residual = x_target.float() - x_reconstruct.float()

        aux_terms = []
        for i in range(self.active_groups):
            start_idx, end_idx = self.group_indices[i], self.group_indices[i + 1]
            group_dead = dead_mask[start_idx:end_idx]
            if group_dead.sum() == 0:
                continue

            group_pre = F.relu(pre_acts[:, start_idx:end_idx])
            dead_pre = group_pre[:, group_dead]
            if dead_pre.shape[1] == 0:
                continue

            k_aux = min(top_k_aux, dead_pre.shape[1])
            topk_vals, topk_idx = torch.topk(dead_pre, k_aux, dim=-1)
            dead_sparse = torch.zeros_like(dead_pre).scatter(-1, topk_idx, topk_vals)

            group_dec = self.W_dec[start_idx:end_idx][group_dead]
            aux_recon = dead_sparse @ group_dec

            if use_per_prefix:
                residual = x_target.float() - intermediate_reconstructs[i].float()
            else:
                residual = final_residual

            aux_terms.append(self._auxk_reconstruction_loss(aux_recon, residual))

        if not aux_terms:
            return torch.tensor(0.0, dtype=x_target.dtype, device=x_target.device)

        return aux_penalty * torch.stack(aux_terms).sum()

    @torch.no_grad()
    def calibrate_threshold(self, activation_store, num_batches=50):
        """Calibrate the eval threshold from a trained checkpoint.

        Run a few batches through BatchTopK (in train mode) to populate
        the EMA threshold, then switch back to eval mode.
        """
        was_training = self.training
        self.train()
        for i in range(num_batches):
            source_batch, _ = activation_store.next_batch()
            x, _ = self.preprocess_input(source_batch)
            x = x.reshape(-1, x.shape[-1])
            self.compute_activations(x)
        if not was_training:
            self.eval()
        return float(self.threshold.item())

    def encode(self, x_source):
        """Encode input to sparse features, masking inactive groups."""
        original_shape = x_source.shape
        self._cache_decode_source(x_source)
        x_source, scale = self.preprocess_input(x_source)
        self.input_scale = float(scale)

        x_source = x_source.reshape(-1, x_source.shape[-1])
        x_enc_in = x_source
        _, result = self.compute_activations(x_enc_in)

        max_act_index = self.group_indices[self.active_groups]
        result[:, max_act_index:] = 0

        if len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1], -1)
        return result

    def decode(self, acts_topk, x_source=None, include_skip=True):
        """Decode sparse features using only active groups."""
        original_shape = acts_topk.shape
        flat_acts = acts_topk.reshape(-1, acts_topk.shape[-1])
        max_feature_idx = self.group_indices[self.active_groups]
        acts_topk_active = flat_acts[:, :max_feature_idx]
        W_dec_active = self.W_dec[:max_feature_idx, :]
        residual = acts_topk_active @ W_dec_active + self.b_dec
        reconstruct = self.postprocess_output(residual, self.output_scale)
        if include_skip and self.has_skip_connection:
            source = self._resolve_decode_source(x_source, acts_topk)
            flat_source = source.reshape(-1, source.shape[-1])
            skip = self.compute_skip(flat_source).to(reconstruct.dtype)
            reconstruct = reconstruct + skip
        if len(original_shape) == 3:
            return reconstruct.reshape(original_shape[0], original_shape[1], -1)
        return reconstruct

    def decode_residual(self, acts_topk):
        return self.decode(acts_topk, include_skip=False)
