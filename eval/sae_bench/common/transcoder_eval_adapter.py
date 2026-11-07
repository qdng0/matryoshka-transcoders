from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.google_transcoder import load_google_transcoder  # noqa: E402
from eval.sae_bench.common.transcoder_adapter import (  # noqa: E402
    load_transcoder_adapter,
)


_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}


@dataclass
class TranscoderEvalCfg:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    is_transcoder: bool
    input_hook_name: str
    output_hook_name: str
    context_size: int | None = None
    hook_head_index: int | None = None
    architecture: str = "transcoder"
    apply_b_dec_to_input: bool | None = None
    finetuning_scaling_factor: bool | None = None
    activation_fn_str: str = ""
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dtype: str = "float32"
    device: str = ""
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: list[Any] | None = None
    training_tokens: int = -100000
    sae_lens_training_version: str | None = None
    neuronpedia_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "hook_layer": self.hook_layer,
            "hook_name": self.hook_name,
            "is_transcoder": self.is_transcoder,
            "input_hook_name": self.input_hook_name,
            "output_hook_name": self.output_hook_name,
            "context_size": self.context_size,
            "hook_head_index": self.hook_head_index,
            "architecture": self.architecture,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "activation_fn_str": self.activation_fn_str,
            "prepend_bos": self.prepend_bos,
            "normalize_activations": self.normalize_activations,
            "dtype": self.dtype,
            "device": self.device,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "seqpos_slice": self.seqpos_slice if self.seqpos_slice is not None else [None],
            "training_tokens": self.training_tokens,
            "sae_lens_training_version": self.sae_lens_training_version,
            "neuronpedia_id": self.neuronpedia_id,
        }


class TranscoderEvalAdapter:
    """Minimal SAE-like wrapper with paper-style transcoder metadata.

    This intentionally keeps the runtime surface small and avoids the extra
    convenience behaviors in the local adapter (for example encode chunking), so
    the evaluation path stays as close as possible to the original SAE Bench
    assumptions.
    """

    def __init__(self, transcoder: Any, cfg: TranscoderEvalCfg):
        self.transcoder = transcoder
        self.cfg = cfg

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
            return getattr(self.transcoder, "W_dec", torch.empty((), device=self.device)).dtype

    @property
    def W_dec(self) -> torch.Tensor:
        return self.transcoder.W_dec

    @property
    def W_enc(self) -> torch.Tensor:
        return self.transcoder.W_enc

    @property
    def hook_name(self) -> str:
        return self.cfg.hook_name

    @property
    def output_scale(self) -> float:
        return float(getattr(self.transcoder, "output_scale", 1.0))

    @property
    def source_hook_point(self) -> str:
        return self.cfg.input_hook_name

    @property
    def target_hook_point(self) -> str:
        return self.cfg.output_hook_name

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return self.transcoder.encode(x)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        acts = acts.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return self.transcoder.decode(acts)

    def to(self, *args, **kwargs) -> "TranscoderEvalAdapter":
        self.transcoder.to(*args, **kwargs)
        return self


def _dtype_to_str(dtype: torch.dtype) -> str:
    return _DTYPE_TO_STR.get(dtype, str(dtype).replace("torch.", ""))


def _normalize_model_name(model_name: str) -> str:
    if "gemma-2-2b" in model_name:
        return "gemma-2-2b"
    return model_name.replace("google/", "")


def _load_run_cfg(config_path: Path) -> dict[str, Any]:
    with config_path.open("r") as f:
        return json.load(f)


def load_transcoder_eval_adapter(
    config_path: Path | str,
    checkpoint_path: Path | str | None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Tuple[str, TranscoderEvalAdapter]:
    config_path = Path(config_path).resolve()
    run_cfg = _load_run_cfg(config_path)
    name, adapter = load_transcoder_adapter(config_path, checkpoint_path, device, dtype)

    d_in = int(
        getattr(
            adapter.transcoder,
            "d_in",
            run_cfg.get("source_act_size", run_cfg.get("act_size", 2304)),
        )
    )
    architecture = str(run_cfg.get("activation_type") or "transcoder")
    is_transcoder = not bool(run_cfg.get("sae_mode", False))

    cfg = TranscoderEvalCfg(
        model_name=_normalize_model_name(str(run_cfg.get("model_name", "gemma-2-2b"))),
        d_in=d_in,
        d_sae=int(adapter.cfg.d_sae),
        hook_layer=int(adapter.cfg.hook_layer),
        hook_name=str(adapter.cfg.hook_name),
        is_transcoder=is_transcoder,
        input_hook_name=str(adapter.cfg.source_hook_point),
        output_hook_name=str(adapter.cfg.target_hook_point),
        context_size=run_cfg.get("seq_len"),
        hook_head_index=adapter.cfg.hook_head_index,
        architecture=architecture,
        apply_b_dec_to_input=bool(run_cfg.get("subtract_decoder_bias_from_input", False)),
        finetuning_scaling_factor=False,
        activation_fn_str=architecture,
        prepend_bos=bool(run_cfg.get("prepend_bos", True)),
        normalize_activations="none",
        dtype=_dtype_to_str(dtype),
        device="",
        dataset_path=str(run_cfg.get("dataset_name", "")),
        dataset_trust_remote_code=bool(run_cfg.get("dataset_trust_remote_code", True)),
        seqpos_slice=[None],
        training_tokens=int(run_cfg.get("training_tokens", -100000)),
    )
    return name, TranscoderEvalAdapter(adapter.transcoder, cfg)


def load_google_transcoder_eval_adapter(
    params_path: Path | str,
    layer: int,
    device: str | torch.device,
    dtype: torch.dtype,
    model_name: str = "gemma-2-2b",
) -> Tuple[str, TranscoderEvalAdapter]:
    params_path = Path(params_path).resolve()
    variant = params_path.parent.name
    width = params_path.parent.parent.name
    weights_root = str(params_path.parents[3])
    transcoder = load_google_transcoder(
        layer=layer,
        variant=variant,
        width=width,
        weights_root=weights_root,
        device=str(device),
    )
    transcoder = transcoder.to(device=device, dtype=dtype)

    hook_name = transcoder.config["source_hook_point"]
    cfg = TranscoderEvalCfg(
        model_name=_normalize_model_name(model_name),
        d_in=int(transcoder.d_in),
        d_sae=int(transcoder.d_sae),
        hook_layer=int(layer),
        hook_name=hook_name,
        is_transcoder=True,
        input_hook_name=hook_name,
        output_hook_name=transcoder.config["target_hook_point"],
        context_size=None,
        hook_head_index=transcoder.config.get("hook_head_index"),
        architecture="jumprelu",
        apply_b_dec_to_input=None,
        finetuning_scaling_factor=None,
        activation_fn_str="",
        prepend_bos=True,
        normalize_activations="none",
        dtype=_dtype_to_str(dtype),
        device="",
        dataset_path="",
        dataset_trust_remote_code=True,
        seqpos_slice=[None],
        training_tokens=-100000,
        sae_lens_training_version=None,
        neuronpedia_id=None,
    )
    name = f"google_l{layer}_{params_path.stem}".replace("/", "_")
    return name, TranscoderEvalAdapter(transcoder, cfg)
