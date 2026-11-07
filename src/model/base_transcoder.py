import torch
import torch.nn as nn


class BaseTranscoder(nn.Module):
    """Base class for transcoder architectures with shared utility methods."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.input_scale = 1.0
        self.output_scale = 1.0

    def preprocess_input(self, x):
        """Scale inputs by input_scale (default 1.0, i.e. no-op)."""
        scale = float(self.input_scale) if self.input_scale else 1.0
        if scale == 1.0:
            return x, scale
        return x / scale, scale

    def preprocess_target(self, x, scale):
        """Scale targets using the same factor as inputs."""
        if scale == 1.0:
            return x
        return x / scale

    def postprocess_output(self, x_reconstruct, scale=None):
        """Rescale outputs back to the original activation scale."""
        scale = self.output_scale if scale is None else scale
        scale = float(scale) if scale is not None else 1.0
        if scale == 1.0:
            return x_reconstruct
        return x_reconstruct * scale

    def init_skip_connection(self, cfg, d_in: int, d_out: int):
        """Configure an optional affine skip map x @ A + c."""
        skip_mode = str(cfg.get("skip_mode", "none") or "none").lower()
        if skip_mode in {"", "false", "0"}:
            skip_mode = "none"
        if skip_mode not in {"none", "fixed", "learned"}:
            raise ValueError("skip_mode must be one of: none, fixed, learned.")

        self.skip_mode = skip_mode
        self.skip_path = cfg.get("skip_path")
        self._last_source_for_decode = None
        cfg["skip_mode"] = skip_mode

        if skip_mode == "none":
            return

        A = None
        c = None
        if self.skip_path:
            skip_state = torch.load(
                self.skip_path, map_location="cpu", weights_only=False
            )
            if not isinstance(skip_state, dict):
                raise ValueError(f"Skip artifact must be a dict, got {type(skip_state)}.")
            A = skip_state.get("A", skip_state.get("skip_A"))
            c = skip_state.get("c", skip_state.get("skip_c"))
            if A is None or c is None:
                raise KeyError("Skip artifact must contain tensors 'A' and 'c'.")
            A = A.detach().float()
            c = c.detach().float()
        elif skip_mode == "fixed":
            raise ValueError("skip_mode='fixed' requires cfg['skip_path'].")

        if A is None:
            A = torch.zeros(d_in, d_out, dtype=torch.float32)
        if c is None:
            c = torch.zeros(d_out, dtype=torch.float32)

        if tuple(A.shape) != (d_in, d_out):
            raise ValueError(
                f"Skip A has shape {tuple(A.shape)}, expected {(d_in, d_out)}."
            )
        if tuple(c.shape) != (d_out,):
            raise ValueError(f"Skip c has shape {tuple(c.shape)}, expected {(d_out,)}.")

        if skip_mode == "learned":
            self.skip_A = nn.Parameter(A)
            self.skip_c = nn.Parameter(c)
        else:
            self.register_buffer("skip_A", A, persistent=True)
            self.register_buffer("skip_c", c, persistent=True)

        cfg["skip_path"] = self.skip_path
        cfg["skip_trainable"] = skip_mode == "learned"

    @property
    def has_skip_connection(self) -> bool:
        return getattr(self, "skip_mode", "none") in {"fixed", "learned"}

    def compute_skip(self, x_source):
        if not self.has_skip_connection:
            return None
        skip_A = self.skip_A.to(device=x_source.device)
        skip_c = self.skip_c.to(device=x_source.device)
        return x_source.to(dtype=skip_A.dtype) @ skip_A + skip_c

    def _cache_decode_source(self, x_source):
        if self.has_skip_connection:
            self._last_source_for_decode = x_source.detach()

    def _resolve_decode_source(self, x_source, feature_acts):
        if not self.has_skip_connection:
            return None
        if x_source is None:
            x_source = self._last_source_for_decode
        if x_source is None:
            raise ValueError(
                "Skip transcoders need source activations for decode(). "
                "Call encode(source) first, or pass decode(..., x_source=source)."
            )
        if x_source.shape[:-1] != feature_acts.shape[:-1]:
            raise ValueError(
                "Decode source leading shape does not match feature activations: "
                f"source={tuple(x_source.shape)}, acts={tuple(feature_acts.shape)}."
            )
        return x_source

    @torch.no_grad()
    def project_decoder_grad(self):
        """Project out the gradient component parallel to each decoder direction.

        Must be called after backward() and before optimizer.step() so that Adam
        does not accumulate momentum in the constrained (radial) direction.
        """
        if self.W_dec.grad is None:
            return
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    @torch.no_grad()
    def renorm_decoder_weights(self):
        """Re-project decoder weights onto the unit sphere.

        Must be called after optimizer.step() so the forward pass always uses
        unit-norm decoder directions.
        """
        if bool(self.config.get("freeze_decoder", False)):
            return
        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Deprecated: use project_decoder_grad() + renorm_decoder_weights() separately."""
        self.project_decoder_grad()
        self.renorm_decoder_weights()
