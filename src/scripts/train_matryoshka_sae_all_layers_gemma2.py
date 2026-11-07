"""
Train Matryoshka SAEs on Gemma-2-2B residual stream (all layers).

Config matched to existing Gemma-2-2B Matryoshka transcoder for fair SAE-vs-Transcoder comparison.

Key differences from the transcoder:
  - Hook: resid_post (single-site SAE) vs mlp_in -> mlp_out (transcoder)
  - Input: subtracts b_dec (SAE centering) vs scales by sqrt(d_model) (transcoder)

All other hyperparameters (dict_size, k, dataset, lr, tokens) are matched.
"""

import os
import sys
import argparse
import gc
import multiprocessing as mp

import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.matryoshka_sae import MatryoshkaBatchTopKSAE
from src.model.transcoder_activation_store import TranscoderActivationsStore
from src.training.train_sae import train_sae
from src.utils.config import (
    get_default_cfg,
    post_init_cfg,
    create_sae_config,
    get_model_config,
    load_hooked_transformer,
)

MODEL_NAME = "gemma-2-2b"
DICT_SIZE = 18_432  # 8x expansion (matching transcoder)
TOP_K = 88  # matching transcoder k


def build_config_for_layer(layer: int, device: str, dataset_path: str):
    cfg = get_default_cfg()
    num_tokens = int(os.environ.get("NUM_TOKENS", str(100_000_000)))
    group_sizes = [
        DICT_SIZE // 16,   # 1152
        DICT_SIZE // 16,   # 1152
        DICT_SIZE // 8,    # 2304
        DICT_SIZE // 4,    # 4608
        DICT_SIZE // 2,    # 9216
    ]

    cfg.update(
        {
            "model_name": MODEL_NAME,
            "dataset_path": dataset_path,
            "trust_remote_code": True,
            "layer": layer,
            "num_tokens": num_tokens,
            "model_batch_size": 4,
            "batch_size": 4096,
            "seq_len": 128,
            "lr": 3e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "model_dtype": torch.bfloat16,
            "dtype": torch.bfloat16,
            "device": device,
            "buffer_device": "cpu",
            "scheduler_type": "warmup_decay",
            "warmup_steps": 1000,
            "warmup_start_ratio": 0.1,
            "min_lr_ratio": 0.01,
            "dict_size": DICT_SIZE,
            "group_sizes": group_sizes,
            "k": TOP_K,
            "top_k": TOP_K,
            "top_k_aux": TOP_K * 8,
            "aux_penalty": 1 / 32,
            "n_batches_to_dead": 20,
            "dead_steps_threshold": 20,
            "dead_feature_threshold": 1e-8,
            "threshold_beta": 0.99,
            "sae_mode": True,
            "trainer_class": "MatryoshkaBatchTopKTrainer",
            "sae_type": "matryoshka_batchtopk_sae",
            "activation_type": "batchtopk",
            "save_activation_samples": True,
            "sample_collection_freq": 2_000,
            "perf_log_freq": 500,
            "checkpoint_freq": 5_000,
            "transcoder_type": "matryoshka_sae_gemma2",
            "wandb_project": "matryoshka-sae-gemma2-all-layers",
            "experiment_description": f"matryoshka_sae_layer{layer}_k{TOP_K}",
            "activation_cache_path": None,
            "activation_cache_size": 0,
        }
    )

    cfg = create_sae_config(
        cfg,
        layer=layer,
        site="resid_post",
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


def is_layer_already_trained(layer: int, num_tokens: int, batch_size: int) -> bool:
    import json

    expected_steps = int(num_tokens // batch_size)
    layer_dir = Path("results/training/matryoshka_sae") / f"layer_{layer}" / str(expected_steps)
    for final_ckpt in layer_dir.glob("*/checkpoints/final.pt"):
        config_path = final_ckpt.parent.parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    saved_cfg = json.load(f)
                if (
                    saved_cfg.get("model_name") == MODEL_NAME
                    and saved_cfg.get("dict_size") == DICT_SIZE
                    and saved_cfg.get("sae_mode", False)
                ):
                    return True
            except Exception:
                pass
        else:
            return True
    return False


def train_single_layer(layer: int, device: str, dataset_path: str):
    cfg = build_config_for_layer(layer, device, dataset_path)
    steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(
        f"[Layer {layer} | device={device}] SAE dict={cfg['dict_size']} k={cfg['k']} "
        f"hook={cfg['hook_point']} steps≈{steps}"
    )

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    sae = MatryoshkaBatchTopKSAE(cfg).to(cfg["device"])
    train_sae(sae, activation_store, model, cfg)

    del sae, activation_store, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def partition_layers(layers, num_parts: int):
    parts = [[] for _ in range(num_parts)]
    for idx, layer in enumerate(layers):
        parts[idx % num_parts].append(layer)
    return parts


def train_layers_on_device(device: str, layers, dataset_path: str):
    if not layers:
        print(f"[{device}] No layers assigned, skipping.")
        return
    print(f"[{device}] Training SAE layers: {layers}")
    for layer in layers:
        train_single_layer(layer, device=device, dataset_path=dataset_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Matryoshka SAEs on Gemma-2-2B residual stream (all layers). "
        "Config matched to transcoder for fair comparison."
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
        default=os.environ.get("DATASET_PATH", "HuggingFaceFW/fineweb-edu"),
        help="HuggingFace dataset path.",
    )
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

    model_cfg = get_model_config(MODEL_NAME)
    total_layers = int(model_cfg.get("n_layers", 26))

    explicit_layers = parse_int_list(args.layers)
    skip_layers = set(parse_int_list(args.skip_layers))

    all_layers = explicit_layers if explicit_layers else list(range(total_layers))
    all_layers = [l for l in all_layers if l not in skip_layers]

    num_tokens = int(os.environ.get("NUM_TOKENS", str(100_000_000)))
    batch_size = 4096
    already_trained = [l for l in all_layers if is_layer_already_trained(l, num_tokens, batch_size)]
    layers_to_train = [l for l in all_layers if l not in already_trained]

    print(f"Model: {MODEL_NAME} | Mode: Matryoshka SAE (resid_post)")
    print(f"Dict size: {DICT_SIZE} | k: {TOP_K}")
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
            train_single_layer(layer, device=devices[0], dataset_path=args.dataset_path)
    else:
        mp.set_start_method("spawn", force=True)
        layer_partitions = partition_layers(layers_to_train, len(devices))
        processes = []
        for dev, assigned_layers in zip(devices, layer_partitions):
            print(f"Assigning layers {assigned_layers} to device {dev}")
            p = mp.Process(target=train_layers_on_device, args=(dev, assigned_layers, args.dataset_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
