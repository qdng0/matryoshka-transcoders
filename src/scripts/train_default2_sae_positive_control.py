"""Train default2-like Matryoshka and BatchTopK SAEs for positive controls.

This launcher mirrors the default2 transcoder recipe where it can:

- Gemma-2-2B, folded layer norm;
- monology/pile-uncopyrighted;
- 500M activation tokens;
- width 16,384;
- the default2 per-layer TopK map;
- batch size 4096, seq_len 1024, lr 7e-5, warmup-only schedule;
- AuxK enabled with top_k_aux = 8 * top_k.

The default hook is ``mlp_in`` so the SAE reconstructs the same source-space
representation that the transcoders encode.  Pass ``--site mlp_out`` for the
target-space companion control.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.matryoshka_sae import MatryoshkaBatchTopKSAE
from src.model.transcoder_activation_store import TranscoderActivationsStore
from src.scripts.train_matryoshka_all_layers import LAYER_TOP_K, parse_torch_dtype
from src.training.train_sae import train_sae
from src.utils.config import (
    create_sae_config,
    get_default_cfg,
    load_hooked_transformer,
    post_init_cfg,
)


DEFAULT_LAYERS = "12"
DEFAULT_DATASET_PATH = "monology/pile-uncopyrighted"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_NUM_TOKENS = 500_000_000
DEFAULT_DICT_SIZE = 16_384
DEFAULT_BATCH_SIZE = 4096
DEFAULT_MODEL_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 1024
DEFAULT_LR = 7e-5
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_N_BATCHES_TO_DEAD = 1000
DEFAULT_PERF_LOG_FREQ = 20_000
DEFAULT_CHECKPOINT_FREQ = 200_000
DEFAULT_SAMPLE_COLLECTION_FREQ = 80_000


def parse_int_list(spec: str) -> list[int]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    if spec.startswith("[") and spec.endswith("]"):
        spec = spec[1:-1]
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-").lower()


def resolve_top_k(layer: int, args: argparse.Namespace) -> int:
    if args.top_k is not None:
        return int(args.top_k)
    if layer not in LAYER_TOP_K:
        raise ValueError(f"No default2 top_k configured for layer {layer}.")
    return int(LAYER_TOP_K[layer])


def resolve_group_sizes(arch: str, dict_size: int, group_sizes_arg: str) -> list[int]:
    if arch == "batchtopk":
        return [dict_size]
    explicit = parse_int_list(group_sizes_arg)
    if explicit:
        if sum(explicit) != dict_size:
            raise ValueError(
                f"--group-sizes must sum to dict_size={dict_size}, got {sum(explicit)}."
            )
        return explicit
    return [
        dict_size // 16,
        dict_size // 16,
        dict_size // 8,
        dict_size // 4,
        dict_size // 2,
    ]


def build_run_name(layer: int, top_k: int, group_sizes: list[int], args: argparse.Namespace) -> str:
    token_tag = f"{args.num_tokens // 1_000_000}M"
    group_tag = "flat" if len(group_sizes) == 1 else "matryoshka"
    parts = [
        "default2_sae",
        args.site,
        args.arch,
        group_tag,
        f"layer{layer}",
        f"k{top_k}",
        token_tag,
        slugify(args.dataset_path.split("/")[-1]),
    ]
    if args.run_tag:
        parts.append(slugify(args.run_tag))
    return "_".join(parts)


def build_config(layer: int, device: str, args: argparse.Namespace) -> dict:
    top_k = resolve_top_k(layer, args)
    group_sizes = resolve_group_sizes(args.arch, args.dict_size, args.group_sizes)
    top_k_aux = args.top_k_aux if args.top_k_aux is not None else top_k * 8
    run_name = build_run_name(layer, top_k, group_sizes, args)

    cfg = get_default_cfg()
    cfg.update(
        {
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "trust_remote_code": True,
            "fold_ln": args.fold_ln,
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
            "dict_size": args.dict_size,
            "group_sizes": group_sizes,
            "group_fractions": None,
            "k": top_k,
            "top_k": top_k,
            "top_k_aux": top_k_aux,
            "aux_penalty": args.aux_penalty,
            "n_batches_to_dead": args.n_batches_to_dead,
            "dead_steps_threshold": args.n_batches_to_dead,
            "dead_feature_threshold": 1e-8,
            "threshold_beta": 0.99,
            "sae_mode": True,
            "trainer_class": "MatryoshkaBatchTopKTrainer"
            if args.arch == "matryoshka"
            else "BatchTopKTrainer",
            "sae_type": "matryoshka_batchtopk_sae"
            if args.arch == "matryoshka"
            else "batchtopk_sae",
            "activation_type": "batchtopk",
            "save_activation_samples": args.save_activation_samples,
            "sample_collection_freq": args.sample_collection_freq,
            "perf_log_freq": args.perf_log_freq,
            "checkpoint_freq": args.checkpoint_freq,
            "activation_cache_path": None,
            "activation_cache_size": 0,
            "wandb_project": args.wandb_project,
            "experiment_description": run_name,
            "run_tag": args.run_tag,
        }
    )

    cfg = create_sae_config(cfg, layer=layer, site=args.site)
    cfg = post_init_cfg(cfg)
    cfg["name"] = run_name
    return cfg


def train_one(layer: int, device: str, args: argparse.Namespace) -> None:
    cfg = build_config(layer, device, args)
    steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(
        f"Training {args.arch} SAE: layer={layer}, site={args.site}, "
        f"hook={cfg['hook_point']}, dict_size={cfg['dict_size']}, "
        f"group_sizes={cfg['group_sizes']}, k={cfg['top_k']}, steps~={steps}, "
        f"device={device}, dtype={cfg['dtype']}, model_dtype={cfg['model_dtype']}, "
        f"dataset={cfg['dataset_path']}[{cfg['dataset_split']}]"
    )

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    sae = MatryoshkaBatchTopKSAE(cfg).to(cfg["device"])
    train_sae(sae, activation_store, model, cfg)

    del sae, activation_store, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train default2-like Matryoshka or BatchTopK SAE positive controls."
    )
    parser.add_argument("--arch", choices=["matryoshka", "batchtopk"], required=True)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--site", choices=["mlp_in", "mlp_out", "resid_post"], default="mlp_in")
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--dict-size", type=int, default=DEFAULT_DICT_SIZE)
    parser.add_argument("--group-sizes", default="")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-k-aux", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-batch-size", type=int, default=DEFAULT_MODEL_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--aux-penalty", type=float, default=1 / 32)
    parser.add_argument("--n-batches-to-dead", type=int, default=DEFAULT_N_BATCHES_TO_DEAD)
    parser.add_argument("--model-dtype", default="float32")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--fold-ln", dest="fold_ln", action="store_true", default=True)
    parser.add_argument("--no-fold-ln", dest="fold_ln", action="store_false")
    parser.add_argument("--wandb-project", default="default2-sae-positive-control")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--perf-log-freq", type=int, default=DEFAULT_PERF_LOG_FREQ)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--sample-collection-freq", type=int, default=DEFAULT_SAMPLE_COLLECTION_FREQ)
    parser.add_argument(
        "--no-save-activation-samples",
        dest="save_activation_samples",
        action="store_false",
    )
    parser.set_defaults(save_activation_samples=True)
    args = parser.parse_args()

    for layer in parse_int_list(args.layers):
        train_one(layer, args.device, args)


if __name__ == "__main__":
    main()
