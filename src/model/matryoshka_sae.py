"""Matryoshka BatchTopK SAE for single-hook autoencoding."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import compute_fvu, set_global_seed


class MatryoshkaBatchTopKSAE(nn.Module):
    """Matryoshka SAE with global BatchTopK (train) and threshold gating (eval).

    Loss = sum of per-prefix MSEs + aux_penalty * sum of dead-feature MSEs.
    Based on Bussmann et al. (arXiv 2503.17547).
    """

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        set_global_seed(cfg.get("seed", 42))

        self.d_in = int(cfg.get("act_size"))
        self.dict_size = int(cfg["dict_size"])
        self.k = int(cfg.get("k", cfg.get("top_k", 0)))
        if self.k <= 0:
            raise ValueError("k must be a positive integer for BatchTopK SAE.")

        self.group_sizes = self._resolve_group_sizes(cfg)
        self.group_indices = [0]
        running = 0
        for size in self.group_sizes:
            running += int(size)
            self.group_indices.append(running)
        self.active_groups = len(self.group_sizes)
        self.loss_reduction = str(cfg.get("loss_reduction", "mean")).lower()
        if self.loss_reduction not in {"mean", "sum_mean"}:
            raise ValueError(
                f"Unsupported loss_reduction={self.loss_reduction!r}; expected 'mean' or 'sum_mean'."
            )
        self.skip_outer_loss = bool(cfg.get("skip_outer_loss", False))
        self.normalize_losses_by_num_matryoshka_steps = bool(
            cfg.get("normalize_losses_by_num_matryoshka_steps", False)
        )
        self.tied_encoder_decoder_init = bool(
            cfg.get("tied_encoder_decoder_init", False)
        )
        self.rescale_acts_by_decoder_norm = bool(
            cfg.get("rescale_acts_by_decoder_norm", False)
        )
        self.enforce_decoder_unit_norm = bool(
            cfg.get("enforce_decoder_unit_norm", True)
        )
        self.decoder_init_norm = cfg.get("decoder_init_norm", 1.0)

        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))
        self.W_enc = nn.Parameter(torch.empty(self.d_in, self.dict_size))
        self.W_dec = nn.Parameter(torch.empty(self.dict_size, self.d_in))
        torch.nn.init.kaiming_uniform_(self.W_dec)
        if self.decoder_init_norm is not None:
            self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            self.W_dec.data[:] = self.W_dec.data * float(self.decoder_init_norm)
        if self.tied_encoder_decoder_init:
            self.W_enc.data[:] = self.W_dec.data.T.clone().contiguous()
        else:
            torch.nn.init.kaiming_uniform_(self.W_enc)

        self.register_buffer(
            'num_batches_not_active',
            torch.zeros(self.dict_size, dtype=torch.float32, device=cfg["device"]),
            persistent=False,
        )
        self.register_buffer(
            "threshold",
            torch.tensor(-1.0, dtype=torch.float32, device=cfg["device"]),
        )
        self.to(cfg["dtype"]).to(cfg["device"])

        # Persist resolved values to config for checkpoint/eval portability.
        self.config["k"] = self.k
        self.config["top_k"] = self.k
        self.config["group_sizes"] = list(self.group_sizes)

    @property
    def d_sae(self):
        return self.dict_size

    def _resolve_group_sizes(self, cfg) -> list[int]:
        group_sizes = cfg.get("group_sizes")
        if isinstance(group_sizes, Iterable):
            sizes = [int(v) for v in group_sizes]
            if len(sizes) == 0:
                raise ValueError("group_sizes must contain at least one group.")
            if sum(sizes) != self.dict_size:
                raise ValueError(
                    f"group_sizes must sum to dict_size={self.dict_size}, got {sum(sizes)}."
                )
            return sizes

        fractions = cfg.get("group_fractions", [0.125, 0.125, 0.25, 0.5])
        if not isinstance(fractions, Iterable):
            raise ValueError("group_fractions must be a list of positive floats.")
        fractions = [float(v) for v in fractions]
        if len(fractions) == 0 or any(v <= 0 for v in fractions):
            raise ValueError("group_fractions must contain only positive values.")

        total = float(sum(fractions))
        fractions = [v / total for v in fractions]
        raw = [v * self.dict_size for v in fractions]
        sizes = [int(math.floor(v)) for v in raw]
        remainder = self.dict_size - sum(sizes)
        if remainder > 0:
            frac = [v - s for v, s in zip(raw, sizes)]
            order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
            for idx in order[:remainder]:
                sizes[idx] += 1

        if any(v <= 0 for v in sizes):
            raise ValueError(
                "Resolved group sizes contain zeros; increase dict_size or change group_fractions."
            )
        if sum(sizes) != self.dict_size:
            raise ValueError("Resolved group sizes do not sum to dict_size.")
        return sizes

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Constrain decoder rows to unit norm and project radial gradient."""
        if not self.enforce_decoder_unit_norm:
            return
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (
                (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            )
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def _reconstruction_loss(self, reconstruction, target):
        diff = reconstruction.float() - target.float()
        if self.loss_reduction == "sum_mean":
            return diff.pow(2).sum(dim=-1).mean()
        return diff.pow(2).mean()

    def _decode_slice(self, feature_acts, start_idx, end_idx):
        decoder = self.W_dec[start_idx:end_idx]
        if self.rescale_acts_by_decoder_norm:
            feature_acts = feature_acts * (1 / decoder.norm(dim=-1).clamp(min=1e-8))
        return feature_acts @ decoder

    @torch.no_grad()
    def update_threshold(self, sparse_acts):
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
    def update_inactive_features(self, sparse_acts):
        dead_threshold = float(self.config.get("dead_feature_threshold", 1e-8))
        max_acts = sparse_acts.max(dim=0).values
        inactive = max_acts < dead_threshold
        self.num_batches_not_active += inactive.float()
        self.num_batches_not_active[~inactive] = 0

    def compute_activations(self, x):
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        if self.rescale_acts_by_decoder_norm:
            pre_acts = pre_acts * self.W_dec.norm(dim=-1)
        acts = F.relu(pre_acts)

        if self.training:
            k_total = min(self.k * x.shape[0], acts.numel())
            topk = torch.topk(acts.flatten(), k_total, dim=-1)
            sparse_acts = torch.zeros_like(acts.flatten()).scatter(
                -1, topk.indices, topk.values
            ).reshape(acts.shape)
            self.update_threshold(sparse_acts)
        else:
            if float(self.threshold.item()) < 0.0:
                raise ValueError(
                    "SAE threshold is unset. Load a trained checkpoint or run training first."
                )
            sparse_acts = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        return pre_acts, sparse_acts

    def _get_auxiliary_loss(self, target, reconstruction, pre_acts, intermediate_reconstructs=None):
        dead_steps = int(
            self.config.get(
                "dead_steps_threshold", self.config.get("n_batches_to_dead", 20)
            )
        )
        dead_mask = self.num_batches_not_active >= dead_steps
        if dead_mask.sum() == 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        top_k_aux = int(self.config.get("top_k_aux", self.k * 8))
        if top_k_aux <= 0:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)
        use_per_prefix = (
            self.config.get("aux_residual_mode", "final") == "per_prefix"
            and intermediate_reconstructs is not None
        )
        if not use_per_prefix:
            residual = target.float() - reconstruction.float()

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

            group_decoder = self.W_dec[start_idx:end_idx][group_dead]
            if self.rescale_acts_by_decoder_norm:
                dead_sparse = dead_sparse * (
                    1 / group_decoder.norm(dim=-1).clamp(min=1e-8)
                )
            aux_recon = dead_sparse @ group_decoder
            if use_per_prefix:
                residual = target.float() - intermediate_reconstructs[i].float()
            aux_terms.append(self._reconstruction_loss(aux_recon, residual))

        if not aux_terms:
            return torch.tensor(0.0, dtype=target.dtype, device=target.device)

        aux_penalty = float(self.config.get("aux_penalty", 1 / 32))
        return aux_penalty * torch.stack(aux_terms).sum()

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        pre_acts, sparse_acts = self.compute_activations(x_flat)
        self.update_inactive_features(sparse_acts)

        reconstruction = self.b_dec.unsqueeze(0).expand_as(x_flat)
        prefix_losses = []
        prefix_fvus = []
        intermediate_reconstructs = []
        for i in range(self.active_groups):
            start_idx, end_idx = self.group_indices[i], self.group_indices[i + 1]
            reconstruction = (
                reconstruction + self._decode_slice(sparse_acts[:, start_idx:end_idx], start_idx, end_idx)
            )
            intermediate_reconstructs.append(reconstruction)
            prefix_losses.append(self._reconstruction_loss(reconstruction, x_flat))
            prefix_fvus.append(compute_fvu(x_flat, reconstruction))

        if not prefix_losses:
            raise RuntimeError("Matryoshka SAE requires at least one group.")

        prefix_loss_tensor = torch.stack(prefix_losses)
        if self.skip_outer_loss and prefix_loss_tensor.numel() > 1:
            prefix_loss_terms = prefix_loss_tensor[:-1]
        else:
            prefix_loss_terms = prefix_loss_tensor
        l2_loss = prefix_loss_terms.sum()
        if self.normalize_losses_by_num_matryoshka_steps and prefix_loss_terms.numel() > 0:
            l2_loss = l2_loss / prefix_loss_terms.numel()
        aux_loss = self._get_auxiliary_loss(
            x_flat,
            reconstruction,
            pre_acts,
            intermediate_reconstructs=intermediate_reconstructs,
        )
        l0_norm = (sparse_acts > 0).sum(dim=-1).float().mean()
        final_fvu = compute_fvu(x_flat, reconstruction)

        sae_out = reconstruction.reshape(original_shape)
        output = {
            "sae_out": sae_out,
            "loss": l2_loss + aux_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "aux_loss": aux_loss,
            "fvu": final_fvu,
            "min_l2_loss": prefix_loss_tensor.min(),
            "max_l2_loss": prefix_loss_tensor.max(),
            "fvu_min": min(prefix_fvus),
            "fvu_max": max(prefix_fvus),
            "fvu_mean": sum(prefix_fvus) / len(prefix_fvus),
            "threshold": self.threshold,
        }
        for i in range(self.active_groups):
            output[f"diagnostic/prefix_mse_{i}"] = prefix_losses[i]
            output[f"diagnostic/prefix_fvu_{i}"] = prefix_fvus[i]
        return output

    def encode(self, x):
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        _, sparse_acts = self.compute_activations(x_flat)
        if len(original_shape) == 3:
            return sparse_acts.reshape(original_shape[0], original_shape[1], -1)
        return sparse_acts

    def decode(self, feature_acts):
        original_shape = feature_acts.shape
        flat_acts = feature_acts.reshape(-1, feature_acts.shape[-1])
        if self.rescale_acts_by_decoder_norm:
            flat_acts = flat_acts * (1 / self.W_dec.norm(dim=-1).clamp(min=1e-8))
        reconstruction = flat_acts @ self.W_dec + self.b_dec
        if len(original_shape) == 3:
            return reconstruction.reshape(original_shape[0], original_shape[1], -1)
        return reconstruction
