from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class _TranscoderCfg:
    hook_name: str
    hook_head_index: int | None = None


class GoogleTranscoder(nn.Module):
    """
    Minimal wrapper that exposes Google Gemma Scope transcoders with the
    interface expected by SAE Bench evaluations.
    """

    def __init__(
        self,
        params_path: str | Path,
        layer: int,
        device: str = "cuda",
    ):
        super().__init__()
        path = Path(params_path)
        if not path.is_file():
            raise FileNotFoundError(f"No params.npz at {path}")

        weights = np.load(path)
        self.W_dec = torch.from_numpy(weights["W_dec"])
        self.W_enc = torch.from_numpy(weights["W_enc"])
        self.b_dec = torch.from_numpy(weights["b_dec"])
        self.b_enc = torch.from_numpy(weights["b_enc"])
        self.threshold = torch.from_numpy(weights["threshold"])

        self.d_in = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]
        self.layer = layer

        # Gemma-2 MLP input is exposed as the post-LN2 activations.
        source_hook = f"blocks.{layer}.ln2.hook_normalized"
        target_hook = f"blocks.{layer}.hook_mlp_out"

        self.cfg = _TranscoderCfg(hook_name=source_hook, hook_head_index=None)
        self.config = {
            "source_hook_point": source_hook,
            "target_hook_point": target_hook,
        }

        self.device = torch.device(device)
        self.training = False
        self.to(self.device)

    def _activate(self, pre: torch.Tensor) -> torch.Tensor:
        # JumpReLU: z * 1[z > θ]
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    def to(
        self, device: str | torch.device, dtype: torch.dtype | None = torch.float32
    ) -> "GoogleTranscoder":
        target_device = torch.device(device)
        target_dtype = dtype or torch.float32
        tensors = ["W_dec", "W_enc", "b_dec", "b_enc", "threshold"]
        for name in tensors:
            tensor = getattr(self, name)
            setattr(
                self,
                name,
                tensor.to(device=target_device, dtype=target_dtype),
            )
        self.device = target_device
        super().to(target_device)
        return self

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        original_shape = acts.shape
        target_dtype = self.W_enc.dtype
        flat = acts.reshape(-1, acts.shape[-1]).to(self.device, dtype=target_dtype)
        pre = flat @ self.W_enc + self.b_enc
        activated = self._activate(pre)
        return activated.reshape(*original_shape[:-1], self.d_sae)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        original_shape = features.shape
        target_dtype = self.W_dec.dtype
        flat = features.reshape(-1, features.shape[-1]).to(
            self.device, dtype=target_dtype
        )
        recon = flat @ self.W_dec + self.b_dec
        return recon.reshape(*original_shape[:-1], self.d_in)

    def preprocess_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        target_dtype = self.W_enc.dtype
        return x.to(self.device, dtype=target_dtype), None, None

    def compute_activations(
        self, x_source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.W_enc.dtype
        flat = x_source.to(self.device, dtype=target_dtype)
        pre = flat @ self.W_enc + self.b_enc
        acts = self._activate(pre)
        return acts, acts


def load_google_transcoder(
    layer: int,
    variant: str,
    *,
    width: str = "width_16k",
    weights_root: str = "models/gemma-scope-2b-pt-transcoders",
    device: str = "cuda:0",
) -> GoogleTranscoder:
    """
    Convenience helper for constructing a GoogleTranscoder from the checkpoint tree.
    """
    params_path = (
        Path(weights_root)
        / f"layer_{layer}"
        / width
        / variant
        / "params.npz"
    )
    return GoogleTranscoder(params_path=params_path, layer=layer, device=device)

