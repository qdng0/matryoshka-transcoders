"""Train the default2 layer-6 width-8k Matryoshka transcoder ablation."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.scripts import train_matryoshka_all_layers as train_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Training device.")
    args = parser.parse_args()

    train_args = argparse.Namespace(
        dataset_path="monology/pile-uncopyrighted",
        dataset_split="train",
        num_tokens=500_000_000,
        dict_size=8192,
        group_sizes="",
        batch_size=4096,
        model_batch_size=4,
        seq_len=1024,
        lr=7e-5,
        max_grad_norm=1.0,
        warmup_steps=1000,
        model_dtype="float32",
        dtype="float32",
        top_k_aux=760,
        aux_penalty=0.03125,
        n_batches_to_dead=1000,
        input_scale=1.0,
        output_scale=1.0,
        wandb_project="default2-matryoshka",
        batchtopk_mode="global",
        run_tag="width8k-l6",
        cache_prefix="cache_matryoshka",
        activation_cache_size=0,
        perf_log_freq=20_000,
        checkpoint_freq=200_000,
        sample_collection_freq=80_000,
        save_activation_samples=True,
    )
    train_all.train_single_layer(6, device=args.device, args=train_args)


if __name__ == "__main__":
    main()
