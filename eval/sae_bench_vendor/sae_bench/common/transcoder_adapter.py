from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import torch

EVAL_ROOT = Path(__file__).resolve().parents[2]  # .../eval
REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "sae_bench_vendor"
for p in (VENDOR_ROOT, REPO_ROOT, EVAL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.model.matryoshka_transcoder import MatryoshkaTranscoder  # noqa: E402
from src.model.flat_transcoder import FlatTranscoder  # noqa: E402
from eval.google_transcoder import load_google_transcoder  # noqa: E402
from src.utils.config import get_default_cfg  # noqa: E402
from src.utils.config import get_hook_name  # noqa: E402


def _infer_hook_layer(cfg: dict[str, Any]) -> int:
    for key in ("layer", "target_layer", "source_layer"):
        if key in cfg and cfg[key] is not None:
            return int(cfg[key])
    hook_point = cfg.get("source_hook_point") or cfg.get("target_hook_point", "")
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_layer": self.hook_layer,
            "d_sae": self.d_sae,
            "hook_name": self.hook_name,
            "source_hook_point": self.source_hook_point,
            "target_hook_point": self.target_hook_point,
            "hook_head_index": self.hook_head_index,
        }


class TranscoderAdapter:
    """Expose a transcoder with an SAE-like interface for SAE Bench style evals."""

    def __init__(
        self,
        transcoder: MatryoshkaTranscoder,
        hook_layer: int,
        hook_name: str,
        source_hook_point: str,
        target_hook_point: str,
        hook_head_index: int | None = None,
    ):
        self.transcoder = transcoder
        self.cfg = TranscoderAdapterCfg(
            hook_layer=hook_layer,
            d_sae=transcoder.d_sae,
            hook_name=hook_name,
            source_hook_point=source_hook_point,
            target_hook_point=target_hook_point,
            hook_head_index=hook_head_index,
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
    def hook_name(self) -> str:
        return self.cfg.hook_name

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.transcoder.encode(x)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.transcoder.decode(acts)

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
    cfg["device"] = device
    cfg["dtype"] = dtype
    cfg["model_dtype"] = dtype
    hook_layer = _infer_hook_layer(cfg)

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

    # Choose transcoder class based on config
    group_sizes = cfg.get("group_sizes", [])
    sae_type = cfg.get("sae_type", "")
    activation_type = cfg.get("activation_type", "")
    if sae_type == "topk" or activation_type in {"topk", "jumprelu"} or len(
        group_sizes
    ) <= 1:
        transcoder = FlatTranscoder(cfg).to(device=device, dtype=dtype)
    else:
        transcoder = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
    state = torch.load(ckpt_path, map_location=device)
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
    variant = config_path.stem
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
    adapter = TranscoderAdapter(
        transcoder,
        hook_layer=layer,
        source_hook_point=transcoder.config["source_hook_point"],
        target_hook_point=transcoder.config["target_hook_point"],
        hook_head_index=transcoder.config.get("hook_head_index"),
    )
    name = f"google_l{layer}_{Path(params_path).stem}".replace("/", "_")
    return name, adapter

