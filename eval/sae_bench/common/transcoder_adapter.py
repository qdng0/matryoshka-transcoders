from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.matryoshka_transcoder import MatryoshkaTranscoder  # noqa: E402
from src.model.flat_transcoder import FlatTranscoder  # noqa: E402
from src.model.matryoshka_sae import MatryoshkaBatchTopKSAE  # noqa: E402
from eval.google_transcoder import load_google_transcoder  # noqa: E402
from src.utils.config import get_default_cfg  # noqa: E402
from src.utils.config import get_hook_name  # noqa: E402


def _infer_hook_layer(cfg: dict[str, Any]) -> int:
    for key in ("layer", "hook_layer", "target_layer", "source_layer"):
        if key in cfg and cfg[key] is not None:
            return int(cfg[key])
    hook_point = (
        cfg.get("hook_point")
        or cfg.get("source_hook_point")
        or cfg.get("target_hook_point", "")
    )
    for part in hook_point.split("."):
        if part.isdigit():
            return int(part)
        if part.startswith("layer_") and part[len("layer_") :].isdigit():
            return int(part[len("layer_") :])
    raise ValueError("Could not infer hook layer; please specify layer in config.")


def _infer_checkpoint_path(config_path: Path) -> Path:
    ckpt_dir = config_path.parent / "checkpoints"
    final_candidate = ckpt_dir / "final.pt"
    if final_candidate.exists():
        return final_candidate.resolve()
    checkpoints = sorted(ckpt_dir.glob("*.pt"))
    if checkpoints:
        return checkpoints[-1].resolve()
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


@dataclass
class TranscoderAdapterCfg:
    hook_layer: int
    d_sae: int
    hook_name: str
    source_hook_point: str
    target_hook_point: str
    hook_head_index: int | None = None
    has_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_layer": self.hook_layer,
            "d_sae": self.d_sae,
            "hook_name": self.hook_name,
            "source_hook_point": self.source_hook_point,
            "target_hook_point": self.target_hook_point,
            "hook_head_index": self.hook_head_index,
            "has_skip": self.has_skip,
        }


class TranscoderAdapter:
    """Expose a transcoder/SAE with an SAE-like interface for SAE Bench style evals."""

    def __init__(
        self,
        transcoder: Any,
        hook_layer: int,
        hook_name: str,
        source_hook_point: str,
        target_hook_point: str,
        hook_head_index: int | None = None,
    ):
        self.transcoder = transcoder
        d_sae = int(
            getattr(
                transcoder,
                "d_sae",
                getattr(transcoder, "dict_size", transcoder.W_dec.shape[0]),
            )
        )
        self.cfg = TranscoderAdapterCfg(
            hook_layer=hook_layer,
            d_sae=d_sae,
            hook_name=hook_name,
            source_hook_point=source_hook_point,
            target_hook_point=target_hook_point,
            hook_head_index=hook_head_index,
            has_skip=hasattr(transcoder, "skip_A") and hasattr(transcoder, "skip_c"),
        )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.transcoder.parameters()).device
        except StopIteration:
            return getattr(self.transcoder, "device", torch.device("cpu"))

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.transcoder.parameters()).dtype
        except StopIteration:
            return getattr(
                self.transcoder, "W_dec", torch.empty((), device=self.device)
            ).dtype

    @property
    def W_dec(self) -> torch.Tensor:
        return self.transcoder.W_dec

    @property
    def W_enc(self) -> torch.Tensor:
        return self.transcoder.W_enc

    @property
    def output_scale(self) -> float:
        """Scaling factor applied by decode() to map from internal to target space.

        Transcoders that divide encoder input by a scale and multiply decoder
        output by the same scale (e.g. Matryoshka/Flat transcoders) will have
        output_scale > 1.  Models without such preprocessing (SAEs, Google
        Gemma Scope transcoders) default to 1.0.
        """
        return float(getattr(self.transcoder, "output_scale", 1.0))

    @property
    def hook_name(self) -> str:
        return self.cfg.hook_name

    def encode(self, x: torch.Tensor, max_tokens: int = 4096) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        # Flatten to 2D for chunking, then restore shape
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        n_tokens = x_flat.shape[0]
        if n_tokens <= max_tokens:
            return self.transcoder.encode(x)
        # Chunk to avoid OOM from BatchTopK flattening large batches
        chunks = []
        for i in range(0, n_tokens, max_tokens):
            chunks.append(self.transcoder.encode(x_flat[i : i + max_tokens]))
        result = torch.cat(chunks, dim=0)
        if len(orig_shape) == 3:
            result = result.reshape(orig_shape[0], orig_shape[1], -1)
        return result

    @property
    def has_skip(self) -> bool:
        return hasattr(self.transcoder, "skip_A") and hasattr(self.transcoder, "skip_c")

    def skip(self, source: torch.Tensor) -> torch.Tensor:
        if not self.has_skip:
            raise ValueError("This transcoder does not have a skip connection.")
        source = source.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return source @ self.transcoder.skip_A + self.transcoder.skip_c

    def decode_sparse(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode only the sparse residual dictionary path."""
        return self.transcoder.decode(
            acts.to(device=self.device, dtype=self.dtype, non_blocking=True)
        )

    def decode(self, acts: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
        sparse_reconstruction = self.decode_sparse(acts)
        if self.has_skip and source is not None:
            return self.skip(source) + sparse_reconstruction
        return sparse_reconstruction

    def to(self, *args, **kwargs) -> "TranscoderAdapter":
        self.transcoder.to(*args, **kwargs)
        return self


def load_transcoder_adapter(
    config_path: Path | str,
    checkpoint_path: Path | str | None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Tuple[str, TranscoderAdapter]:
    config_path = Path(config_path).resolve()
    ckpt_path = (
        Path(checkpoint_path).resolve()
        if checkpoint_path
        else _infer_checkpoint_path(config_path)
    )
    with config_path.open("r") as f:
        run_cfg = json.load(f)

    cfg = get_default_cfg()
    cfg.update(run_cfg)

    # Backward compat: old configs without explicit scale used sqrt(act_size) default
    if "input_scale" not in run_cfg and "output_scale" not in run_cfg:
        act_size = int(cfg.get("act_size", 2304))
        cfg["input_scale"] = math.sqrt(act_size)
        cfg["output_scale"] = math.sqrt(act_size)

    cfg["device"] = device
    cfg["dtype"] = dtype
    cfg["model_dtype"] = dtype
    hook_layer = _infer_hook_layer(cfg)

    sae_mode = bool(cfg.get("sae_mode", False))
    if sae_mode:
        hook_point = cfg.get("hook_point") or cfg.get("source_hook_point")
        if not hook_point:
            hook_point = get_hook_name(
                cfg.get("site", "resid_post"),
                cfg.get("layer", hook_layer),
                cfg.get("model_name", "gemma-2-2b"),
            )
        source_hook_point = hook_point
        target_hook_point = hook_point
    else:
        source_hook_point = cfg.get("source_hook_point") or get_hook_name(
            cfg.get("source_site", cfg.get("site", "mlp_in")),
            cfg.get("source_layer", hook_layer),
            cfg.get("model_name", "gemma-2-2b"),
        )
        target_hook_point = cfg.get("target_hook_point") or get_hook_name(
            cfg.get("target_site", cfg.get("site", "mlp_out")),
            cfg.get("target_layer", hook_layer),
            cfg.get("model_name", "gemma-2-2b"),
        )
    hook_head_index = cfg.get("hook_head_index")
    hook_name = cfg.get("hook_name", source_hook_point)

    if sae_mode:
        transcoder = MatryoshkaBatchTopKSAE(cfg).to(device=device, dtype=dtype)
    else:
        # Choose transcoder class based on config
        group_sizes = cfg.get("group_sizes", [])
        if len(group_sizes) > 1:
            transcoder = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
        else:
            transcoder = FlatTranscoder(cfg).to(device=device, dtype=dtype)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "skip_A" in state and "skip_c" in state:
        skip_A = state["skip_A"]
        skip_c = state["skip_c"]
        if not hasattr(transcoder, "skip_A"):
            if bool(cfg.get("skip_trainable", False)):
                transcoder.register_parameter(
                    "skip_A",
                    torch.nn.Parameter(
                        torch.empty(
                            skip_A.shape,
                            device=device,
                            dtype=dtype,
                        )
                    ),
                )
                transcoder.register_parameter(
                    "skip_c",
                    torch.nn.Parameter(
                        torch.empty(
                            skip_c.shape,
                            device=device,
                            dtype=dtype,
                        )
                    ),
                )
            else:
                transcoder.register_buffer(
                    "skip_A",
                    torch.empty(skip_A.shape, device=device, dtype=dtype),
                    persistent=True,
                )
                transcoder.register_buffer(
                    "skip_c",
                    torch.empty(skip_c.shape, device=device, dtype=dtype),
                    persistent=True,
                )
    transcoder.load_state_dict(state)
    transcoder.eval()

    adapter = TranscoderAdapter(
        transcoder,
        hook_layer=hook_layer,
        hook_name=hook_name,
        source_hook_point=source_hook_point,
        target_hook_point=target_hook_point,
        hook_head_index=hook_head_index,
    )
    # NOTE: Many training runs use a generic filename like `config.json`. Using only
    # `config_path.stem` causes name collisions across different runs (e.g. topk vs
    # matryoshka), which in turn causes incorrect cache hits in eval artifacts.
    try:
        rel = config_path.relative_to(REPO_ROOT)
        variant = rel.with_suffix("").as_posix().replace("/", "_")
    except ValueError:
        # Fallback if config_path is outside the repo root.
        variant = f"{config_path.parent.name}_{config_path.stem}"

    name = f"{variant}_l{hook_layer}_{ckpt_path.stem}".replace("/", "_")
    return name, adapter


def load_google_transcoder_adapter(
    params_path: Path | str,
    layer: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Tuple[str, TranscoderAdapter]:
    """Load a Google Gemma Scope transcoder and wrap it for evaluations."""
    params_path = Path(params_path).resolve()
    variant = params_path.parent.name
    width = params_path.parent.parent.name
    weights_root = str(params_path.parents[3])  # models/gemma-scope-2b-pt-transcoders
    transcoder = load_google_transcoder(
        layer=layer,
        variant=variant,
        width=width,
        weights_root=weights_root,
        device=str(device),
    )
    transcoder = transcoder.to(device=device, dtype=dtype)
    source_hook_point = transcoder.config["source_hook_point"]
    hook_name = source_hook_point  # Default hook_name to source_hook_point
    adapter = TranscoderAdapter(
        transcoder,
        hook_layer=layer,
        hook_name=hook_name,
        source_hook_point=source_hook_point,
        target_hook_point=transcoder.config["target_hook_point"],
        hook_head_index=transcoder.config.get("hook_head_index"),
    )
    name = f"google_l{layer}_{Path(params_path).stem}".replace("/", "_")
    return name, adapter
