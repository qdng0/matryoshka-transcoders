#!/usr/bin/env python3
"""Train fixed-skip Matryoshka or BatchTopK transcoders with default2 settings."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.flat_transcoder import FlatTranscoder  # noqa: E402
from src.model.matryoshka_transcoder import MatryoshkaTranscoder  # noqa: E402
from src.model.transcoder_activation_store import TranscoderActivationsStore  # noqa: E402
from src.scripts.train_matryoshka_all_layers import (  # noqa: E402
    LAYER_TOP_K,
    parse_int_list,
    parse_torch_dtype,
    slugify_tag,
)
from src.training.training import train_transcoder  # noqa: E402
from src.utils.config import (  # noqa: E402
    create_transcoder_config,
    get_default_cfg,
    load_hooked_transformer,
    post_init_cfg,
)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def resolve_top_k(layer: int, args: argparse.Namespace) -> int:
    if args.top_k is not None:
        return int(args.top_k)
    if args.model_name == "gemma-2-2b" and layer in LAYER_TOP_K:
        return int(LAYER_TOP_K[layer])
    raise ValueError(f"No default top_k configured for layer {layer}; pass --top-k.")


def resolve_group_sizes(args: argparse.Namespace) -> list[int]:
    group_sizes = parse_int_list(args.group_sizes)
    if group_sizes:
        if sum(group_sizes) != args.width:
            raise ValueError(
                f"--group-sizes must sum to --width={args.width}, got {sum(group_sizes)}."
            )
        return group_sizes
    return [
        args.width // 16,
        args.width // 16,
        args.width // 8,
        args.width // 4,
        args.width // 2,
    ]


def resolve_fold_ln(args: argparse.Namespace) -> bool:
    if args.fold_ln is not None:
        return bool(args.fold_ln)
    return "gemma" in args.model_name.lower()


def dataset_tag(path: str) -> str:
    return slugify_tag(path.split("/")[-1])


def build_run_name(args: argparse.Namespace, top_k: int) -> str:
    token_tag = f"{args.num_tokens // 1_000_000}M"
    arch_tag = "skip_matryoshka" if args.arch == "matryoshka" else "skip_batchtopk"
    parts = [
        f"{arch_tag}_layer{args.layer}",
        f"k{top_k}",
        args.skip_mode,
        token_tag,
        dataset_tag(args.dataset_path),
    ]
    if args.aux_penalty <= 0.0:
        parts.append("auxoff")
    else:
        parts.append("auxon")
    if args.run_tag:
        parts.append(slugify_tag(args.run_tag))
    return "_".join(parts)


def load_skip_metrics(skip_path: str) -> dict:
    metrics_path = Path(skip_path).with_name("ridge_metrics.json")
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_config(args: argparse.Namespace) -> dict:
    top_k = resolve_top_k(args.layer, args)
    top_k_aux = args.top_k_aux if args.top_k_aux is not None else top_k * 8
    run_name = build_run_name(args, top_k)
    skip_metrics = load_skip_metrics(args.skip_path) if args.skip_path else {}

    cfg = get_default_cfg()
    cfg.update(
        {
            "model_name": args.model_name,
            "seed": args.seed,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "dataset_shuffle_buffer": args.dataset_shuffle_buffer,
            "trust_remote_code": True,
            "fold_ln": resolve_fold_ln(args),
            "layer": args.layer,
            "num_tokens": args.num_tokens,
            "model_batch_size": args.model_batch_size,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_batches_in_buffer": args.num_batches_in_buffer,
            "lr": args.lr,
            "beta1": 0.0,
            "beta2": 0.999,
            "max_grad_norm": args.max_grad_norm,
            "model_dtype": parse_torch_dtype(args.model_dtype),
            "dtype": parse_torch_dtype(args.dtype),
            "device": resolve_device(args.device),
            "buffer_device": args.buffer_device,
            "scheduler_type": "warmup_only",
            "warmup_steps": args.warmup_steps,
            "warmup_start_ratio": 0.1,
            "min_lr_ratio": 1.0,
            "input_scale": args.input_scale,
            "output_scale": args.output_scale,
            "dict_size": args.width,
            "top_k": top_k,
            "k": top_k,
            "top_k_aux": top_k_aux,
            "aux_penalty": args.aux_penalty,
            "n_batches_to_dead": args.n_batches_to_dead,
            "dead_steps_threshold": args.n_batches_to_dead,
            "dead_feature_threshold": 1e-8,
            "save_activation_samples": args.save_activation_samples,
            "sample_collection_freq": args.sample_collection_freq,
            "perf_log_freq": args.perf_log_freq,
            "checkpoint_freq": args.checkpoint_freq,
            "wandb_project": args.wandb_project,
            "experiment_description": run_name,
            "activation_cache_path": None,
            "activation_cache_size": 0,
            "run_tag": args.run_tag,
            "skip_mode": args.skip_mode,
            "skip_path": str(Path(args.skip_path).resolve()) if args.skip_path else "",
            "skip_ridge_alpha": skip_metrics.get("selected_alpha"),
            "skip_ridge_lambda": skip_metrics.get("selected_lambda"),
            "skip_only_fvu": skip_metrics.get("heldout_fvu_skip"),
            "results_base_dir": args.results_base_dir,
        }
    )

    if args.arch == "matryoshka":
        cfg.update(
            {
                "group_sizes": resolve_group_sizes(args),
                "batchtopk_mode": args.batchtopk_mode,
                "sae_type": "topk",
                "transcoder_type": "skip_matryoshka",
            }
        )
    else:
        cfg.update(
            {
                "group_sizes": [args.width],
                "activation_type": "batchtopk",
                "batchtopk_mode": "global",
                "batchtopk_inference": args.batchtopk_inference,
                "sae_type": "batchtopk",
                "act_l1_coef": 0.0,
                "transcoder_type": "skip_batchtopk",
            }
        )

    cfg = create_transcoder_config(
        cfg,
        source_layer=args.layer,
        target_layer=args.layer,
        source_site=args.source_site,
        target_site=args.target_site,
    )
    if args.source_hook:
        cfg["source_hook_point"] = args.source_hook
    if args.target_hook:
        cfg["target_hook_point"] = args.target_hook
    cfg = post_init_cfg(cfg)
    cfg["name"] = run_name
    return cfg


def run_training(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(
        f"Training {args.arch} skip transcoder: layer={cfg['layer']} "
        f"top_k={cfg['top_k']} width={cfg['dict_size']} steps~={steps} "
        f"skip_mode={cfg['skip_mode']} skip_path={cfg['skip_path']} "
        f"skip_only_fvu={cfg.get('skip_only_fvu')} "
        f"dataset={cfg['dataset_path']}[{cfg['dataset_split']}] "
        f"device={cfg['device']} seed={cfg['seed']}"
    )

    if args.dry_run_config:
        print(json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))
        return

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    if args.arch == "matryoshka":
        transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    else:
        transcoder = FlatTranscoder(cfg).to(cfg["device"])
    train_transcoder(transcoder, activation_store, model, cfg)

    del transcoder, activation_store, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["matryoshka", "batchtopk"], required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--skip-path", type=str, required=True)
    parser.add_argument("--skip-mode", choices=["fixed", "learned"], default="fixed")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-name", type=str, default="gemma-2-2b")
    parser.add_argument("--source-site", type=str, default="mlp_in")
    parser.add_argument("--target-site", type=str, default="mlp_out")
    parser.add_argument("--source-hook", type=str, default="")
    parser.add_argument("--target-hook", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=500_000_000)
    parser.add_argument("--width", type=int, default=16_384)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--group-sizes", type=str, default="")
    parser.add_argument("--batchtopk-mode", choices=["global", "groupwise"], default="global")
    parser.add_argument(
        "--batchtopk-inference",
        choices=["threshold", "batchtopk"],
        default="threshold",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-batches-in-buffer", type=int, default=10)
    parser.add_argument("--buffer-device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--aux-penalty", type=float, default=1 / 32)
    parser.add_argument("--top-k-aux", type=int, default=None)
    parser.add_argument("--n-batches-to-dead", type=int, default=1000)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--model-dtype", type=str, default="float32")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--wandb-project", type=str, default="default2-skip-transcoder")
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="results/default2/skip_transcoders",
    )
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--perf-log-freq", type=int, default=20_000)
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--sample-collection-freq", type=int, default=80_000)
    parser.add_argument("--no-save-activation-samples", dest="save_activation_samples", action="store_false")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--fold-ln", dest="fold_ln", action="store_true", default=None)
    parser.add_argument("--no-fold-ln", dest="fold_ln", action="store_false")
    parser.set_defaults(save_activation_samples=True)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
