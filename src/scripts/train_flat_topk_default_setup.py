import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.flat_transcoder import FlatTranscoder
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


def resolve_top_k(layer: int, args: argparse.Namespace) -> int:
    if args.top_k is not None:
        return int(args.top_k)
    if args.model_name == "gemma-2-2b" and layer in LAYER_TOP_K:
        return int(LAYER_TOP_K[layer])
    raise ValueError(
        f"No default top_k configured for layer {layer}. "
        "Pass --top-k for non-default models."
    )


def resolve_fold_ln(args: argparse.Namespace) -> bool:
    if args.fold_ln is not None:
        return bool(args.fold_ln)
    return "gemma" in args.model_name.lower()


def build_run_name(layer: int, args: argparse.Namespace, top_k: int) -> str:
    token_tag = f"{args.num_tokens // 1_000_000}M"
    if args.model_name == "gemma-2-2b" and args.top_k is None:
        return f"flat_topk_layer{layer}_k{top_k}_default_setup_{token_tag}_pile"

    prefix = ""
    if args.model_name != "gemma-2-2b":
        prefix = f"{slugify_tag(args.model_name.split('/')[-1])}_"
    return f"{prefix}flat_topk_layer{layer}_k{top_k}_default_setup_{token_tag}_pile"


def build_config(layer: int, device: str, args: argparse.Namespace):
    top_k = resolve_top_k(layer, args)
    top_k_aux = args.top_k_aux if args.top_k_aux is not None else top_k * 8
    run_name = build_run_name(layer, args, top_k)
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
            "dict_size": args.dict_size,
            "group_sizes": [args.dict_size],
            "activation_type": "topk",
            "sae_type": "topk",
            "k": top_k,
            "top_k": top_k,
            "top_k_aux": top_k_aux,
            "aux_penalty": args.aux_penalty,
            "act_l1_coef": 0.0,
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
        }
    )
    if args.transcoder_type:
        cfg["transcoder_type"] = args.transcoder_type

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


def run_training(layer: int, device: str, args: argparse.Namespace):
    cfg = build_config(layer, device, args)
    steps = cfg["num_tokens"] // cfg["batch_size"]
    print(
        f"Training flat TopK transcoder: layer={layer}, top_k={cfg['top_k']}, "
        f"dict_size={cfg['dict_size']}, steps~={steps}, device={cfg['device']}, "
        f"model={cfg['model_name']}, fold_ln={cfg['fold_ln']}, "
        f"dataset={cfg['dataset_path']}[{cfg['dataset_split']}]"
    )

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = FlatTranscoder(cfg).to(cfg["device"])
    train_transcoder(transcoder, activation_store, model, cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Train a flat TopK transcoder with default2-comparable settings."
    )
    parser.add_argument("--layer", type=int, required=True, help="Transformer layer index.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma-2-2b",
        help="HookedTransformer model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Per-token TopK value. For Gemma default2, the per-layer map is used "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="monology/pile-uncopyrighted",
        help="HuggingFace dataset path.",
    )
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--num-tokens", type=int, default=500_000_000, help="Total training tokens.")
    parser.add_argument("--dict-size", type=int, default=16_384, help="Dictionary size.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Activation batch size.")
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=4,
        help="Number of token sequences processed by the model per activation-store fetch.",
    )
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length.")
    parser.add_argument("--lr", type=float, default=7e-5, help="Learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps.")
    parser.add_argument("--aux-penalty", type=float, default=0.0, help="Auxiliary loss weight.")
    parser.add_argument(
        "--top-k-aux",
        type=int,
        default=None,
        help="Auxiliary loss TopK. Defaults to top_k * 8.",
    )
    parser.add_argument(
        "--n-batches-to-dead",
        type=int,
        default=1000,
        help="Dead-feature threshold in batches.",
    )
    parser.add_argument("--input-scale", type=float, default=1.0, help="Input scaling factor.")
    parser.add_argument("--output-scale", type=float, default=1.0, help="Output scaling factor.")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="float32",
        help="Torch dtype for the model weights.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Torch dtype for activations and transcoder parameters.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="flat-topk-default-setup",
        help="Weights & Biases project.",
    )
    parser.add_argument(
        "--transcoder-type",
        type=str,
        default=None,
        help="Optional result folder type under results/training/.",
    )
    parser.add_argument(
        "--fold-ln",
        dest="fold_ln",
        action="store_true",
        default=None,
        help="Force TransformerLens fold_ln=True.",
    )
    parser.add_argument(
        "--no-fold-ln",
        dest="fold_ln",
        action="store_false",
        help="Force TransformerLens fold_ln=False.",
    )
    parser.add_argument("--perf-log-freq", type=int, default=20_000, help="Perf logging frequency.")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000, help="Checkpoint frequency.")
    parser.add_argument(
        "--sample-collection-freq",
        type=int,
        default=80_000,
        help="Activation sample collection frequency.",
    )
    parser.add_argument(
        "--no-save-activation-samples",
        dest="save_activation_samples",
        action="store_false",
        help="Disable activation sample collection.",
    )
    parser.set_defaults(save_activation_samples=True)
    args = parser.parse_args()

    run_training(args.layer, args.device, args)


if __name__ == "__main__":
    main()
