"""Train Llama-family Matryoshka transcoders on the Pile.

The NeurIPS artifact uses this launcher for the Llama-3.1-8B cross-model check.
It also supports the Llama-3.2 variants used in earlier sweeps.
"""

import os
import sys
import argparse
import gc
import multiprocessing as mp

import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.matryoshka_transcoder import MatryoshkaTranscoder
from src.model.transcoder_activation_store import TranscoderActivationsStore
from src.training.training import train_transcoder
from src.utils.config import (
    get_default_cfg,
    post_init_cfg,
    create_transcoder_config,
    get_model_config,
    load_hooked_transformer,
)
from src.scripts.train_matryoshka_all_layers import parse_torch_dtype

DEFAULT_TOP_K = 89
DEFAULT_DATASET_PATH = os.environ.get("DATASET_PATH", "monology/pile-uncopyrighted")
DEFAULT_DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
DEFAULT_NUM_TOKENS = int(os.environ.get("NUM_TOKENS", str(500_000_000)))
DEFAULT_DICT_SIZE = os.environ.get("DICT_SIZE")
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4096"))
DEFAULT_SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
DEFAULT_LR = float(os.environ.get("LR", "7e-5"))
DEFAULT_MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
DEFAULT_WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "1000"))
DEFAULT_AUX_PENALTY = float(os.environ.get("AUX_PENALTY", str(1 / 32)))
DEFAULT_N_BATCHES_TO_DEAD = int(os.environ.get("N_BATCHES_TO_DEAD", "1000"))
DEFAULT_MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "bfloat16")
DEFAULT_DTYPE = os.environ.get("DTYPE", "float32")
DEFAULT_ACTIVATION_CACHE_SIZE = int(os.environ.get("ACTIVATION_CACHE_SIZE", "0"))
DEFAULT_PERF_LOG_FREQ = int(os.environ.get("PERF_LOG_FREQ", "10000"))
DEFAULT_CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", "100000"))
DEFAULT_SAMPLE_COLLECTION_FREQ = int(os.environ.get("SAMPLE_COLLECTION_FREQ", "40000"))


def model_tag_for_name(model_name: str) -> str:
    model_lower = model_name.lower()
    if "llama-3.1-8b" in model_lower or "llama31-8b" in model_lower:
        return "llama31_8b"
    if "llama-3.2-3b" in model_lower or "llama32-3b" in model_lower:
        return "llama32_3b"
    if "llama-3.2-1b" in model_lower or "llama32-1b" in model_lower:
        return "llama32_1b"
    return model_name.split("/")[-1].replace("-", "_").lower()


def resolve_transcoder_type(model_name: str, explicit_type: str | None) -> str:
    if explicit_type:
        return explicit_type
    return f"matryoshka_{model_tag_for_name(model_name)}"


def resolve_fold_ln(args: argparse.Namespace) -> bool:
    if args.fold_ln is not None:
        return bool(args.fold_ln)
    return "gemma" in args.model.lower()


def build_config_for_layer(layer: int, device: str, args: argparse.Namespace):
    cfg = get_default_cfg()
    model_name = args.model
    model_cfg = get_model_config(model_name)
    act_size = model_cfg["act_size"]

    dict_size = int(args.dict_size or act_size * 8)
    num_tokens = int(args.num_tokens)
    top_k = int(args.top_k)

    group_sizes = [
        dict_size // 16,
        dict_size // 16,
        dict_size // 8,
        dict_size // 4,
        dict_size // 2,
    ]

    model_batch_size = args.model_batch_size
    if model_batch_size is None:
        if "8B" in model_name:
            model_batch_size = 2
        elif "3B" in model_name:
            model_batch_size = 2
        else:
            model_batch_size = 4

    model_tag = model_tag_for_name(model_name)
    token_tag = f"{num_tokens // 1_000_000}M"
    wandb_project = args.wandb_project or f"matryoshka-all-layers-{model_tag}-pile-{token_tag}"
    transcoder_type = resolve_transcoder_type(model_name, args.transcoder_type)

    cfg.update(
        {
            "model_name": model_name,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "trust_remote_code": True,
            "layer": layer,
            "num_tokens": num_tokens,
            "model_batch_size": model_batch_size,
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
            "fold_ln": resolve_fold_ln(args),
            "scheduler_type": "warmup_only",
            "warmup_steps": args.warmup_steps,
            "warmup_start_ratio": 0.1,
            "min_lr_ratio": 1.0,
            "input_scale": args.input_scale,
            "output_scale": args.output_scale,
            "dict_size": dict_size,
            "group_sizes": group_sizes,
            "top_k": top_k,
            "batchtopk_mode": "global",
            "top_k_aux": args.top_k_aux if args.top_k_aux is not None else top_k * 8,
            "aux_penalty": args.aux_penalty,
            "n_batches_to_dead": args.n_batches_to_dead,
            "save_activation_samples": True,
            "sample_collection_freq": args.sample_collection_freq,
            "perf_log_freq": args.perf_log_freq,
            "checkpoint_freq": args.checkpoint_freq,
            "transcoder_type": transcoder_type,
            "wandb_project": wandb_project,
            "experiment_description": f"matryoshka_layer{layer}_k{top_k}_global_{token_tag}_pile_{model_tag}",
            "activation_cache_path": None,
            "activation_cache_size": args.activation_cache_size,
        }
    )

    cfg = create_transcoder_config(
        cfg,
        source_layer=layer,
        target_layer=layer,
        source_site="mlp_in",
        target_site="mlp_out",
    )
    return post_init_cfg(cfg)


def parse_int_list(csv: str):
    if csv is None or csv == "":
        return []
    out = []
    for part in csv.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def parse_devices(devices_csv: str):
    if devices_csv is None or devices_csv.strip() == "":
        return []
    return [d.strip() for d in devices_csv.split(",") if d.strip()]


def is_layer_already_trained(layer: int, args: argparse.Namespace) -> bool:
    expected_steps = int(args.num_tokens // args.batch_size)
    model_type = resolve_transcoder_type(args.model, args.transcoder_type)
    layer_dir = Path("results/training") / model_type / f"layer_{layer}" / str(expected_steps)
    for final_ckpt in layer_dir.glob("*/checkpoints/final.pt"):
        config_path = final_ckpt.parent.parent / "config.json"
        if not config_path.exists():
            return True
        try:
            import json

            with config_path.open() as f:
                saved_cfg = json.load(f)
            if (
                saved_cfg.get("model_name") == args.model
                and int(saved_cfg.get("top_k", -1)) == int(args.top_k)
            ):
                return True
        except Exception:
            return True
    return False


def train_single_layer(layer: int, device: str, args: argparse.Namespace):
    cfg = build_config_for_layer(layer, device, args)
    steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(
        f"[Layer {layer} | device={device}] model={cfg['model_name']} dict={cfg['dict_size']} "
        f"top_k={cfg['top_k']} batchtopk_mode={cfg['batchtopk_mode']} steps≈{steps}"
    )

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    train_transcoder(transcoder, activation_store, model, cfg)

    del transcoder, activation_store, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def partition_layers(layers, num_parts: int):
    parts = [[] for _ in range(num_parts)]
    for idx, layer in enumerate(layers):
        parts[idx % num_parts].append(layer)
    return parts


def train_layers_on_device(device: str, layers, args: argparse.Namespace):
    if not layers:
        print(f"[{device}] No layers assigned, skipping.")
        return
    print(f"[{device}] Training layers: {layers}")
    for layer in layers:
        train_single_layer(layer, device=device, args=args)


def main():
    parser = argparse.ArgumentParser(
        description="Train Matryoshka transcoders for all Llama 3.2 layers on the Pile (500M tokens, 1x scaling)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name (default: meta-llama/Llama-3.2-1B). Use meta-llama/Llama-3.2-3B for the 3B variant.",
    )
    parser.add_argument("--skip-layers", type=str, default="", help="Comma-separated layer indices to skip.")
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma-separated devices (e.g. cuda:0,cuda:1). Auto-detected if empty.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="Explicit comma-separated layer indices to train. If empty, trains all model layers.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="HuggingFace dataset path (default: monology/pile-uncopyrighted).",
    )
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT, help="Dataset split.")
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS, help="Total training tokens.")
    parser.add_argument(
        "--dict-size",
        type=int,
        default=int(DEFAULT_DICT_SIZE) if DEFAULT_DICT_SIZE is not None else None,
        help="Dictionary size. Defaults to act_size * 8.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Activation batch size.")
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=None,
        help="Sequences processed per activation-store fetch. Defaults to 4 for 1B and 2 for 3B.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM, help="Gradient clipping norm.")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Warmup steps.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("TOP_K", str(DEFAULT_TOP_K))),
        help="BatchTopK average-L0 target.",
    )
    parser.add_argument(
        "--top-k-aux",
        type=int,
        default=int(os.environ["TOP_K_AUX"]) if "TOP_K_AUX" in os.environ else None,
        help="Auxiliary TopK budget. Defaults to top_k * 8.",
    )
    parser.add_argument("--aux-penalty", type=float, default=DEFAULT_AUX_PENALTY, help="Auxiliary loss weight.")
    parser.add_argument(
        "--n-batches-to-dead",
        type=int,
        default=DEFAULT_N_BATCHES_TO_DEAD,
        help="Dead-feature threshold in batches.",
    )
    parser.add_argument("--input-scale", type=float, default=1.0, help="Input scaling factor.")
    parser.add_argument("--output-scale", type=float, default=1.0, help="Output scaling factor.")
    parser.add_argument("--model-dtype", type=str, default=DEFAULT_MODEL_DTYPE, help="Torch dtype for model weights.")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, help="Torch dtype for transcoder parameters.")
    parser.add_argument("--activation-cache-size", type=int, default=DEFAULT_ACTIVATION_CACHE_SIZE)
    parser.add_argument("--perf-log-freq", type=int, default=DEFAULT_PERF_LOG_FREQ)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--sample-collection-freq", type=int, default=DEFAULT_SAMPLE_COLLECTION_FREQ)
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project.")
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
    args = parser.parse_args()

    model_name = args.model
    devices = parse_devices(args.devices)
    if not devices:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            devices = [f"cuda:{i}" for i in range(2)]
        elif torch.cuda.is_available():
            devices = ["cuda:0"]
        else:
            devices = ["cpu"]
    if any(d.startswith("cuda") for d in devices) and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA device is available.")
    if torch.cuda.is_available() and len(devices) > torch.cuda.device_count():
        print(
            f"Warning: requested {len(devices)} CUDA devices but only {torch.cuda.device_count()} available. "
            "Trimming to available devices."
        )
        devices = devices[: torch.cuda.device_count()]

    model_cfg = get_model_config(model_name)
    total_layers = int(model_cfg.get("n_layers", 16))

    explicit_layers = parse_int_list(args.layers)
    skip_layers = set(parse_int_list(args.skip_layers))

    all_layers = explicit_layers if explicit_layers else list(range(total_layers))
    all_layers = [l for l in all_layers if l not in skip_layers]

    already_trained = [l for l in all_layers if is_layer_already_trained(l, args)]
    layers_to_train = [l for l in all_layers if l not in already_trained]

    print(f"Model: {model_name} ({total_layers} layers, act_size={model_cfg['act_size']})")
    print(f"Dataset: {args.dataset_path}[{args.dataset_split}]")
    print(f"Tokens: {args.num_tokens:,}")
    print(f"Top K: {args.top_k}")
    print(f"Devices: {devices}")
    print(f"Skipping layers: {sorted(skip_layers)}")
    print(f"Already trained: {sorted(already_trained)}")
    print(f"Layers to train: {sorted(layers_to_train)}")

    if not layers_to_train:
        print("All selected layers already trained. Nothing to do.")
        return

    if len(devices) == 1:
        print(f"Single device mode on {devices[0]}")
        for layer in layers_to_train:
            train_single_layer(layer, device=devices[0], args=args)
    else:
        mp.set_start_method("spawn", force=True)
        layer_partitions = partition_layers(layers_to_train, len(devices))
        processes = []
        for dev, assigned_layers in zip(devices, layer_partitions):
            print(f"Assigning layers {assigned_layers} to device {dev}")
            p = mp.Process(
                target=train_layers_on_device,
                args=(dev, assigned_layers, args),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
