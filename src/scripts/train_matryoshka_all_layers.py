import os
import sys
import argparse
import gc
import multiprocessing as mp
import re

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

LAYER_TOP_K = {
    0: 115,
    1: 104,
    2: 87,
    3: 96,
    4: 88,
    5: 87,
    6: 95,
    7: 70,
    8: 92,
    9: 72,
    10: 88,
    11: 108,
    12: 111,
    13: 89,
    14: 81,
    15: 78,
    16: 87,
    17: 112,
    18: 99,
    19: 89,
    20: 88,
    21: 102,
    22: 117,
    23: 116,
    24: 96,
    25: 110,
}


DEFAULT_DATASET_PATH = os.environ.get("DATASET_PATH", "HuggingFaceFW/fineweb-edu")
DEFAULT_DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
DEFAULT_DICT_SIZE = int(os.environ.get("DICT_SIZE", str(16_384)))
DEFAULT_NUM_TOKENS = int(os.environ.get("NUM_TOKENS", str(4_000_000_000)))
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4096"))
DEFAULT_MODEL_BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", "2"))
DEFAULT_SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
DEFAULT_LR = float(os.environ.get("LR", "7e-5"))
DEFAULT_MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
DEFAULT_WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "1000"))
DEFAULT_AUX_PENALTY = float(os.environ.get("AUX_PENALTY", str(1 / 32)))
DEFAULT_N_BATCHES_TO_DEAD = int(os.environ.get("N_BATCHES_TO_DEAD", "1000"))
DEFAULT_WANDB_PROJECT = os.environ.get(
    "WANDB_PROJECT",
    "matryoshka-all-layers-gemmascope-4B-no-scaling",
)
DEFAULT_PERF_LOG_FREQ = int(os.environ.get("PERF_LOG_FREQ", "20000"))
DEFAULT_CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", "200000"))
DEFAULT_SAMPLE_COLLECTION_FREQ = int(os.environ.get("SAMPLE_COLLECTION_FREQ", "80000"))
DEFAULT_ACTIVATION_CACHE_SIZE = int(os.environ.get("ACTIVATION_CACHE_SIZE", "200000"))
DEFAULT_INPUT_SCALE = float(os.environ.get("INPUT_SCALE", "1.0"))
DEFAULT_OUTPUT_SCALE = float(os.environ.get("OUTPUT_SCALE", "1.0"))
DEFAULT_RUN_TAG = os.environ.get("RUN_TAG", "")
DEFAULT_CACHE_PREFIX = os.environ.get("CACHE_PREFIX", "cache_matryoshka")
DEFAULT_TOP_K_AUX = os.environ.get("TOP_K_AUX")
DEFAULT_MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "bfloat16")
DEFAULT_DTYPE = os.environ.get("DTYPE", "float32")


def parse_torch_dtype(name: str) -> torch.dtype:
    normalized = name.replace("torch.", "").strip().lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype {name!r}.")
    return mapping[normalized]


def slugify_tag(value: str, default: str = "default") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or default


def build_experiment_tag(layer: int, top_k: int, args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    dataset_tag = slugify_tag(args.dataset_path.split("/")[-1])
    token_tag = f"{args.num_tokens // 1_000_000}M"
    scale_tag = f"scale{args.input_scale:g}x"
    aux_tag = "auxoff" if args.aux_penalty <= 0.0 else "auxon"
    parts = [
        f"matryoshka_layer{layer}",
        f"k{top_k}",
        args.batchtopk_mode,
        token_tag,
        dataset_tag,
        scale_tag,
        aux_tag,
    ]
    custom_group_sizes = parse_int_list(args.group_sizes)
    if custom_group_sizes:
        parts.append("groups-" + "-".join(str(size) for size in custom_group_sizes))
    if args.run_tag:
        parts.append(slugify_tag(args.run_tag))
    return "_".join(parts)


def build_cache_path(layer: int, args: argparse.Namespace) -> str | None:
    if args.activation_cache_size <= 0:
        return None
    dataset_tag = slugify_tag(args.dataset_path.split("/")[-1])
    token_tag = f"{args.num_tokens // 1_000_000}M"
    suffix_parts = [dataset_tag, token_tag]
    if args.run_tag:
        suffix_parts.append(slugify_tag(args.run_tag))
    cache_name = f"{args.cache_prefix}_layer{layer}_{'_'.join(suffix_parts)}.pt"
    return str(Path("activation_cache") / cache_name)


def build_config_for_layer(layer: int, device: str, args: argparse.Namespace):
    cfg = get_default_cfg()
    dict_size = args.dict_size
    num_tokens = args.num_tokens
    top_k = args.top_k if args.top_k is not None else LAYER_TOP_K[layer]
    top_k_aux = args.top_k_aux if args.top_k_aux is not None else top_k * 8
    group_sizes = parse_int_list(args.group_sizes)
    if group_sizes:
        if sum(group_sizes) != dict_size:
            raise ValueError(
                f"--group-sizes must sum to --dict-size={dict_size}, got {sum(group_sizes)}."
            )
    else:
        group_sizes = [
            dict_size // 16,
            dict_size // 16,
            dict_size // 8,
            dict_size // 4,
            dict_size // 2,
        ]
    experiment_tag = build_experiment_tag(layer, top_k, args)

    cfg.update(
        {
            "model_name": "gemma-2-2b",
            "seed": args.seed,
            "joint_source_weight": args.joint_source_weight,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "dataset_shuffle_buffer": args.dataset_shuffle_buffer,
            "trust_remote_code": True,
            "layer": layer,
            "num_tokens": num_tokens,
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
            "sae_type": "matryoshka",
            "transcoder_type": "matryoshka",
            "top_k": top_k,
            "batchtopk_mode": args.batchtopk_mode,
            "top_k_aux": top_k_aux,
            "aux_penalty": args.aux_penalty,
            "n_batches_to_dead": args.n_batches_to_dead,
            "save_activation_samples": args.save_activation_samples,
            "sample_collection_freq": args.sample_collection_freq,
            "perf_log_freq": args.perf_log_freq,
            "checkpoint_freq": args.checkpoint_freq,
            "wandb_project": args.wandb_project,
            "experiment_description": experiment_tag,
            "results_base_dir": args.results_base_dir,
            "activation_cache_path": build_cache_path(layer, args),
            "activation_cache_size": args.activation_cache_size,
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
    cfg["name"] = experiment_tag
    return cfg


def parse_int_list(csv: str):
    if csv is None:
        return []
    csv = csv.strip()
    if csv == "":
        return []
    if csv.startswith("[") and csv.endswith("]"):
        csv = csv[1:-1]
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


def is_layer_already_trained(
    layer: int,
    num_tokens: int,
    batch_size: int,
    results_base_dir: str = "results",
) -> bool:
    expected_steps = int(num_tokens // batch_size)
    layer_dir = (
        Path(results_base_dir)
        / "training"
        / "matryoshka"
        / f"layer_{layer}"
        / str(expected_steps)
    )
    checkpoint_pattern = layer_dir.glob("*/checkpoints/final.pt")
    return any(checkpoint_pattern)


def train_single_layer(layer: int, device: str, args: argparse.Namespace):
    cfg = build_config_for_layer(layer, device, args)
    steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(
        f"[Layer {layer} | device={device}] dict={cfg['dict_size']} top_k={cfg['top_k']} "
        f"group_sizes={cfg['group_sizes']} "
        f"batchtopk_mode={cfg['batchtopk_mode']} steps≈{steps} "
        f"dataset={cfg['dataset_path']}[{cfg['dataset_split']}] "
        f"scale={cfg['input_scale']:g}x aux_penalty={cfg['aux_penalty']} "
        f"model_batch_size={cfg['model_batch_size']} seed={cfg['seed']} "
        f"results_base_dir={cfg['results_base_dir']} name={cfg['name']}"
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
        description="Train Matryoshka transcoders (batchtopk global/groupwise) for all Gemma-2-2B layers with per-layer L0 sparsity."
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
    parser.add_argument("--seed", type=int, default=49, help="Random seed for init and dataset shuffling.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="HuggingFace dataset path.",
    )
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT, help="Dataset split.")
    parser.add_argument(
        "--dataset-shuffle-buffer",
        type=int,
        default=10_000,
        help="Streaming dataset shuffle buffer. Use 0 to disable dataset shuffling.",
    )
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS, help="Total training tokens.")
    parser.add_argument("--dict-size", type=int, default=DEFAULT_DICT_SIZE, help="Dictionary size.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the per-layer L0 sparsity target. Defaults to the default2 layer map.",
    )
    parser.add_argument(
        "--group-sizes",
        type=str,
        default="",
        help="Optional comma-separated Matryoshka group sizes. Defaults to dict fractions 1/16,1/16,1/8,1/4,1/2.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Activation batch size.")
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=DEFAULT_MODEL_BATCH_SIZE,
        help="Number of token sequences processed by Gemma per activation-store fetch.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM, help="Gradient clipping norm.")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Warmup steps.")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default=DEFAULT_MODEL_DTYPE,
        help="Torch dtype for the Gemma model weights (e.g. float32, bfloat16).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        help="Torch dtype for transcoder activations and parameters (e.g. float32, bfloat16).",
    )
    parser.add_argument(
        "--top-k-aux",
        type=int,
        default=int(DEFAULT_TOP_K_AUX) if DEFAULT_TOP_K_AUX is not None else None,
        help="Auxiliary TopK budget. Defaults to top_k * 8.",
    )
    parser.add_argument("--aux-penalty", type=float, default=DEFAULT_AUX_PENALTY, help="Auxiliary loss weight.")
    parser.add_argument(
        "--joint-source-weight",
        type=float,
        default=0.0,
        help="Weight of the auxiliary per-prefix source (x) reconstruction loss. 0 disables the joint head.",
    )
    parser.add_argument(
        "--n-batches-to-dead",
        type=int,
        default=DEFAULT_N_BATCHES_TO_DEAD,
        help="Dead-feature threshold in batches.",
    )
    parser.add_argument("--input-scale", type=float, default=DEFAULT_INPUT_SCALE, help="Input scaling factor.")
    parser.add_argument("--output-scale", type=float, default=DEFAULT_OUTPUT_SCALE, help="Output scaling factor.")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="Weights & Biases project.")
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="results",
        help="Base directory for training artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Exact run name/experiment description override. Intended for single-layer launches.",
    )
    parser.add_argument(
        "--batchtopk-mode",
        type=str,
        choices=["global", "groupwise"],
        default="global",
        help="BatchTopK routing mode.",
    )
    parser.add_argument("--run-tag", type=str, default=DEFAULT_RUN_TAG, help="Optional run suffix for names/cache.")
    parser.add_argument("--cache-prefix", type=str, default=DEFAULT_CACHE_PREFIX, help="Activation cache prefix.")
    parser.add_argument(
        "--activation-cache-size",
        type=int,
        default=DEFAULT_ACTIVATION_CACHE_SIZE,
        help="Activation pairs to cache on disk per layer (0 disables cache).",
    )
    parser.add_argument("--perf-log-freq", type=int, default=DEFAULT_PERF_LOG_FREQ, help="Perf logging frequency.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQ,
        help="Checkpoint frequency.",
    )
    parser.add_argument(
        "--sample-collection-freq",
        type=int,
        default=DEFAULT_SAMPLE_COLLECTION_FREQ,
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

    model_cfg = get_model_config("gemma-2-2b")
    total_layers = int(model_cfg.get("n_layers", 26))

    explicit_layers = parse_int_list(args.layers)
    skip_layers = set(parse_int_list(args.skip_layers))

    all_layers = explicit_layers if explicit_layers else list(range(total_layers))
    all_layers = [l for l in all_layers if l not in skip_layers]

    num_tokens = args.num_tokens
    batch_size = args.batch_size
    already_trained = [
        l
        for l in all_layers
        if is_layer_already_trained(l, num_tokens, batch_size, args.results_base_dir)
    ]
    layers_to_train = [l for l in all_layers if l not in already_trained]

    print(f"Dataset: {args.dataset_path}[{args.dataset_split}]")
    print(f"Tokens: {args.num_tokens:,}")
    print(f"Devices: {devices}")
    print(f"Seed: {args.seed}")
    print(f"Run tag: {args.run_tag or '(none)'}")
    print(
        f"Scaling: input={args.input_scale:g}x output={args.output_scale:g}x | "
        f"aux_penalty={args.aux_penalty} | batchtopk_mode={args.batchtopk_mode} | "
        f"model_batch_size={args.model_batch_size} | "
        f"model_dtype={args.model_dtype} dtype={args.dtype}"
    )
    print(f"Skipping layers: {sorted(skip_layers)}")
    print(f"Already trained: {sorted(already_trained)}")
    print(f"Layers to train: {sorted(layers_to_train)}")
    effective_top_k = {
        layer: args.top_k if args.top_k is not None else LAYER_TOP_K[layer]
        for layer in layers_to_train
    }
    print(f"Per-layer top_k: {effective_top_k}")

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
            p = mp.Process(target=train_layers_on_device, args=(dev, assigned_layers, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
