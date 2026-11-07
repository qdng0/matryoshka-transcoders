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


def resolve_target_l0(layer: int, args: argparse.Namespace) -> float:
    if args.target_l0 is not None:
        return float(args.target_l0)
    if args.model_name == "gemma-2-2b" and layer in LAYER_TOP_K:
        return float(LAYER_TOP_K[layer])
    raise ValueError(
        f"No default L0 target configured for layer {layer}. "
        "Pass --target-l0 for non-default models."
    )


def resolve_fold_ln(args: argparse.Namespace) -> bool:
    if args.fold_ln is not None:
        return bool(args.fold_ln)
    return "gemma" in args.model_name.lower()


def build_run_name(layer: int, args: argparse.Namespace, target_l0: float) -> str:
    token_tag = f"{args.num_tokens // 1_000_000}M"
    l1_tag = f"{args.act_l1_coef:.2e}"
    rescue_tag = ""
    if args.ghost_grad_penalty > 0:
        ghost_tag = f"{args.ghost_grad_penalty:.2g}"
        rescue_tag += f"_ghost-{ghost_tag}"
    if args.aux_penalty <= 0:
        rescue_tag += "_auxoff"
    if args.model_name == "gemma-2-2b" and args.target_l0 is None:
        return f"flat_relu_layer{layer}_l1-{l1_tag}{rescue_tag}_default_setup_{token_tag}_pile"

    prefix = ""
    if args.model_name != "gemma-2-2b":
        prefix = f"{slugify_tag(args.model_name.split('/')[-1])}_"
    return (
        f"{prefix}flat_relu_layer{layer}_l1-{l1_tag}{rescue_tag}_target-l0-{target_l0:g}_"
        f"default_setup_{token_tag}_pile"
    )


def build_config(layer: int, device: str, args: argparse.Namespace):
    target_l0 = resolve_target_l0(layer, args)
    cfg = get_default_cfg()
    run_name = build_run_name(layer, args, target_l0)
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
            "activation_type": "relu",
            "sae_type": "relu",
            "k": 0,
            "top_k": 0,
            "top_k_aux": args.top_k_aux,
            "target_l0": target_l0,
            "aux_penalty": args.aux_penalty,
            "ghost_grad_penalty": args.ghost_grad_penalty,
            "ghost_grad_target_norm_ratio": args.ghost_grad_target_norm_ratio,
            "ghost_grad_max_pre": args.ghost_grad_max_pre,
            "ghost_grad_rescale_loss": not args.no_ghost_grad_rescale_loss,
            "act_l1_coef": args.act_l1_coef,
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
        f"Training flat ReLU transcoder: layer={layer}, "
        f"act_l1_coef={cfg['act_l1_coef']:.3g}, target_l0={cfg['target_l0']}, "
        f"dict_size={cfg['dict_size']}, steps~={steps}, device={cfg['device']}, "
        f"model={cfg['model_name']}, fold_ln={cfg['fold_ln']}, "
        f"dataset={cfg['dataset_path']}[{cfg['dataset_split']}], "
        f"aux_penalty={cfg['aux_penalty']}, "
        f"ghost_grad_penalty={cfg['ghost_grad_penalty']}"
    )

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = FlatTranscoder(cfg).to(cfg["device"])
    train_transcoder(transcoder, activation_store, model, cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Train a flat ReLU transcoder with default2-comparable settings."
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
        "--target-l0",
        type=float,
        default=None,
        help=(
            "Target average active features for run metadata. For Gemma default2, "
            "the per-layer map is used when omitted."
        ),
    )
    parser.add_argument(
        "--act-l1-coef",
        type=float,
        required=True,
        help="L1 sparsity coefficient for ReLU activations.",
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
    parser.add_argument("--aux-penalty", type=float, default=1 / 32, help="Auxiliary loss weight.")
    parser.add_argument("--top-k-aux", type=int, default=512, help="Auxiliary loss TopK.")
    parser.add_argument(
        "--ghost-grad-penalty",
        type=float,
        default=0.0,
        help="Ghost-grad rescue loss weight. Set >0 to enable.",
    )
    parser.add_argument(
        "--ghost-grad-target-norm-ratio",
        type=float,
        default=0.5,
        help="Scale ghost reconstruction norm to this fraction of residual norm.",
    )
    parser.add_argument(
        "--ghost-grad-max-pre",
        type=float,
        default=10.0,
        help="Clamp ghost pre-activations before exp() to avoid overflow.",
    )
    parser.add_argument(
        "--no-ghost-grad-rescale-loss",
        action="store_true",
        help="Do not rescale ghost loss to match the current reconstruction loss magnitude.",
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
        default="default2-flat-relu-pile-500m-auxon",
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
        "--save-activation-samples",
        dest="save_activation_samples",
        action="store_true",
        help=(
            "Enable activation sample collection. This is off by default for "
            "dense ReLU runs because the generic collector scans every feature."
        ),
    )
    parser.add_argument(
        "--no-save-activation-samples",
        dest="save_activation_samples",
        action="store_false",
        help="Disable activation sample collection.",
    )
    parser.set_defaults(save_activation_samples=False)
    args = parser.parse_args()

    run_training(args.layer, args.device, args)


if __name__ == "__main__":
    main()
