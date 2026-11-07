"""Train an x -> z_y bridge through a frozen target-space SAE decoder.

This experiment freezes the decoder from a trained y -> y SAE on ``mlp_out`` and
trains only an encoder/gating predictor from ``mlp_in``:

    z_hat_y = h(x)
    y_hat = D_y z_hat_y + b_y

The launcher supports both target decoders from the layer-12 output-SAE bridge
controls:

- BatchTopK y-SAE decoder with a flat BatchTopK predictor;
- Matryoshka y-SAE decoder with a Matryoshka predictor.

Use ``--predictor-arch`` to intentionally cross these choices.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.flat_transcoder import FlatTranscoder
from src.model.matryoshka_transcoder import MatryoshkaTranscoder
from src.model.transcoder_activation_store import TranscoderActivationsStore
from src.scripts.train_matryoshka_all_layers import (
    LAYER_TOP_K,
    parse_torch_dtype,
    slugify_tag,
)
from src.training.training import train_transcoder
from src.utils.config import (
    create_transcoder_config,
    get_default_cfg,
    load_hooked_transformer,
    post_init_cfg,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECODER_RUNS = {
    "batchtopk": REPO_ROOT
    / "results"
    / "training"
    / "batchtopk_sae"
    / "layer_12"
    / "122070"
    / "Apr-30_default2_sae_mlp_out_batchtopk_flat_layer12_k111_500M_pile-uncopyrighted_l12-output-bridge",
    "matryoshka": REPO_ROOT
    / "results"
    / "training"
    / "matryoshka_sae"
    / "layer_12"
    / "122070"
    / "Apr-30_default2_sae_mlp_out_matryoshka_matryoshka_layer12_k111_500M_pile-uncopyrighted_l12-output-bridge",
}


def parse_int_list(spec: str | None) -> list[int]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    if spec.startswith("[") and spec.endswith("]"):
        spec = spec[1:-1]
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def default_matryoshka_groups(dict_size: int) -> list[int]:
    return [
        dict_size // 16,
        dict_size // 16,
        dict_size // 8,
        dict_size // 4,
        dict_size // 2,
    ]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_decoder_config_path(decoder_arch: str) -> Path:
    return DEFAULT_DECODER_RUNS[decoder_arch] / "config.json"


def default_decoder_checkpoint_path(decoder_arch: str) -> Path:
    return DEFAULT_DECODER_RUNS[decoder_arch] / "checkpoints" / "final.pt"


def resolve_fold_ln(args: argparse.Namespace) -> bool:
    if args.fold_ln is not None:
        return bool(args.fold_ln)
    return "gemma" in args.model_name.lower()


def resolve_top_k(layer: int, decoder_cfg: dict, args: argparse.Namespace) -> int:
    if args.top_k is not None:
        return int(args.top_k)
    if decoder_cfg.get("top_k") is not None:
        return int(decoder_cfg["top_k"])
    if layer in LAYER_TOP_K:
        return int(LAYER_TOP_K[layer])
    raise ValueError(f"No top_k available for layer {layer}; pass --top-k.")


def resolve_predictor_arch(decoder_cfg: dict, args: argparse.Namespace) -> str:
    if args.predictor_arch != "auto":
        return args.predictor_arch
    groups = decoder_cfg.get("group_sizes") or []
    return "matryoshka" if len(groups) > 1 else "batchtopk"


def resolve_group_sizes(
    predictor_arch: str,
    decoder_cfg: dict,
    dict_size: int,
    args: argparse.Namespace,
) -> list[int]:
    explicit = parse_int_list(args.group_sizes)
    if explicit:
        if sum(explicit) != dict_size:
            raise ValueError(
                f"--group-sizes must sum to dict_size={dict_size}, got {sum(explicit)}."
            )
        return explicit
    if predictor_arch == "batchtopk":
        return [dict_size]
    decoder_groups = [int(v) for v in decoder_cfg.get("group_sizes", [])]
    if len(decoder_groups) > 1 and sum(decoder_groups) == dict_size:
        return decoder_groups
    return default_matryoshka_groups(dict_size)


def build_run_name(
    layer: int,
    top_k: int,
    predictor_arch: str,
    args: argparse.Namespace,
) -> str:
    token_tag = f"{args.num_tokens // 1_000_000}M"
    dataset_tag = slugify_tag(args.dataset_path.split("/")[-1])
    aux_tag = "auxoff" if args.aux_penalty <= 0.0 else "auxon"
    parts = [
        "frozen-dy-bridge",
        f"dy-{args.decoder_arch}",
        f"h-{predictor_arch}",
        f"layer{layer}",
        f"k{top_k}",
        aux_tag,
        token_tag,
        dataset_tag,
    ]
    if args.run_tag:
        parts.append(slugify_tag(args.run_tag))
    return "_".join(parts)


def build_config(
    layer: int,
    device: str,
    decoder_cfg: dict,
    decoder_config_path: Path,
    decoder_checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict:
    predictor_arch = resolve_predictor_arch(decoder_cfg, args)
    dict_size = int(args.dict_size or decoder_cfg["dict_size"])
    top_k = resolve_top_k(layer, decoder_cfg, args)
    group_sizes = resolve_group_sizes(predictor_arch, decoder_cfg, dict_size, args)
    top_k_aux = args.top_k_aux if args.top_k_aux is not None else top_k * 8
    run_name = build_run_name(layer, top_k, predictor_arch, args)
    transcoder_type = args.transcoder_type or (
        f"frozen_{args.decoder_arch}_dy_{predictor_arch}_bridge"
    )

    cfg = get_default_cfg()
    cfg.update(
        {
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "trust_remote_code": True,
            "fold_ln": resolve_fold_ln(args),
            "layer": layer,
            "num_tokens": args.num_tokens,
            "model_batch_size": args.model_batch_size,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "beta1": 0.0,
            "beta2": 0.999,
            "max_grad_norm": args.max_grad_norm,
            "model_dtype": parse_torch_dtype(args.model_dtype),
            "dtype": parse_torch_dtype(args.dtype),
            "device": device,
            "buffer_device": "cpu",
            "scheduler_type": "warmup_only",
            "warmup_steps": args.warmup_steps,
            "warmup_start_ratio": 0.1,
            "min_lr_ratio": 1.0,
            "input_scale": args.input_scale,
            "output_scale": args.output_scale,
            "dict_size": dict_size,
            "group_sizes": group_sizes,
            "activation_type": "batchtopk",
            "sae_type": "frozen_output_sae_decoder_bridge",
            "k": top_k,
            "top_k": top_k,
            "top_k_aux": top_k_aux,
            "aux_penalty": args.aux_penalty,
            "act_l1_coef": 0.0,
            "batchtopk_inference": args.batchtopk_inference,
            "batchtopk_mode": args.batchtopk_mode,
            "n_batches_to_dead": args.n_batches_to_dead,
            "dead_steps_threshold": args.n_batches_to_dead,
            "dead_feature_threshold": 1e-8,
            "threshold_beta": args.threshold_beta,
            "save_activation_samples": args.save_activation_samples,
            "sample_collection_freq": args.sample_collection_freq,
            "perf_log_freq": args.perf_log_freq,
            "checkpoint_freq": args.checkpoint_freq,
            "wandb_project": args.wandb_project,
            "experiment_description": run_name,
            "activation_cache_path": None,
            "activation_cache_size": 0,
            "freeze_decoder": True,
            "frozen_decoder_arch": args.decoder_arch,
            "frozen_decoder_config_path": str(decoder_config_path),
            "frozen_decoder_checkpoint_path": str(decoder_checkpoint_path),
            "predictor_arch": predictor_arch,
            "transcoder_type": transcoder_type,
            "run_tag": args.run_tag,
        }
    )

    cfg = create_transcoder_config(
        cfg,
        source_layer=layer,
        target_layer=layer,
        source_site="mlp_in",
        target_site="mlp_out",
    )
    cfg = post_init_cfg(cfg)
    cfg["name"] = run_name
    return cfg


def load_frozen_decoder(transcoder, decoder_checkpoint_path: Path, cfg: dict) -> None:
    state = torch.load(decoder_checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    required = ("W_dec", "b_dec")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Decoder checkpoint missing keys: {missing}")
    if tuple(state["W_dec"].shape) != tuple(transcoder.W_dec.shape):
        raise ValueError(
            f"W_dec shape mismatch: checkpoint={tuple(state['W_dec'].shape)} "
            f"model={tuple(transcoder.W_dec.shape)}"
        )
    if tuple(state["b_dec"].shape) != tuple(transcoder.b_dec.shape):
        raise ValueError(
            f"b_dec shape mismatch: checkpoint={tuple(state['b_dec'].shape)} "
            f"model={tuple(transcoder.b_dec.shape)}"
        )

    device = torch.device(cfg["device"])
    dtype = cfg["dtype"]
    with torch.no_grad():
        transcoder.W_dec.copy_(state["W_dec"].to(device=device, dtype=dtype))
        transcoder.b_dec.copy_(state["b_dec"].to(device=device, dtype=dtype))
    transcoder.W_dec.requires_grad_(False)
    transcoder.b_dec.requires_grad_(False)


def run_training(layer: int, device: str, args: argparse.Namespace) -> None:
    decoder_config_path = Path(
        args.decoder_config_path or default_decoder_config_path(args.decoder_arch)
    ).resolve()
    decoder_checkpoint_path = Path(
        args.decoder_checkpoint_path or default_decoder_checkpoint_path(args.decoder_arch)
    ).resolve()
    decoder_cfg = load_json(decoder_config_path)
    cfg = build_config(
        layer,
        device,
        decoder_cfg,
        decoder_config_path,
        decoder_checkpoint_path,
        args,
    )
    predictor_arch = cfg["predictor_arch"]
    steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(
        f"Training frozen-D_y bridge: layer={layer}, decoder={args.decoder_arch}, "
        f"predictor={predictor_arch}, top_k={cfg['top_k']}, "
        f"group_sizes={cfg['group_sizes']}, steps~={steps}, device={cfg['device']}, "
        f"model={cfg['model_name']}, dataset={cfg['dataset_path']}[{cfg['dataset_split']}], "
        f"decoder_checkpoint={decoder_checkpoint_path}"
    )

    transcoder_cls = MatryoshkaTranscoder if predictor_arch == "matryoshka" else FlatTranscoder
    transcoder = transcoder_cls(cfg).to(cfg["device"])
    load_frozen_decoder(transcoder, decoder_checkpoint_path, cfg)

    if args.dry_run:
        trainable = sum(p.numel() for p in transcoder.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in transcoder.parameters() if not p.requires_grad)
        print(
            f"Dry run OK: trainable_params={trainable:,}, frozen_params={frozen:,}, "
            f"W_dec_norm_mean={transcoder.W_dec.detach().float().norm(dim=-1).mean().item():.6f}"
        )
        return

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    train_transcoder(transcoder, activation_store, model, cfg)

    del transcoder, activation_store, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an x->z_y predictor through a frozen y-SAE decoder."
    )
    parser.add_argument("--decoder-arch", choices=["batchtopk", "matryoshka"], required=True)
    parser.add_argument("--predictor-arch", choices=["auto", "batchtopk", "matryoshka"], default="auto")
    parser.add_argument("--decoder-config-path", default="")
    parser.add_argument("--decoder-checkpoint-path", default="")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--dataset-path", default="monology/pile-uncopyrighted")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-tokens", type=int, default=500_000_000)
    parser.add_argument("--dict-size", type=int, default=None)
    parser.add_argument("--group-sizes", default="")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-k-aux", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--aux-penalty", type=float, default=1 / 32)
    parser.add_argument("--n-batches-to-dead", type=int, default=1000)
    parser.add_argument("--threshold-beta", type=float, default=0.99)
    parser.add_argument("--batchtopk-inference", choices=["threshold", "batchtopk"], default="threshold")
    parser.add_argument("--batchtopk-mode", choices=["global", "groupwise"], default="global")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--model-dtype", default="float32")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--wandb-project", default="default2-frozen-output-sae-decoder-bridge")
    parser.add_argument("--transcoder-type", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--perf-log-freq", type=int, default=20_000)
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--sample-collection-freq", type=int, default=80_000)
    parser.add_argument("--save-activation-samples", dest="save_activation_samples", action="store_true")
    parser.add_argument("--no-save-activation-samples", dest="save_activation_samples", action="store_false")
    parser.set_defaults(save_activation_samples=False)
    parser.add_argument("--fold-ln", dest="fold_ln", action="store_true", default=None)
    parser.add_argument("--no-fold-ln", dest="fold_ln", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_training(args.layer, args.device, args)


if __name__ == "__main__":
    main()
