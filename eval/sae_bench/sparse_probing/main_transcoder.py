"""Sparse probing evaluation for transcoders.

Wraps the vendored SAE Bench sparse probing implementation so it works
with our TranscoderAdapter (duck-type compatible with sae_lens.SAE).

Usage:
    python eval/sae_bench/sparse_probing/main_transcoder.py \
        --config_path path/to/config.json \
        --model_name gemma-2-2b

    python eval/sae_bench/sparse_probing/main_transcoder.py \
        --google_params_path path/to/params.npz \
        --model_name gemma-2-2b \
        --layer 13
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# Ensure vendor, repo, and eval roots are on path for local imports
REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_PARENT = Path(__file__).resolve().parents[2].parent
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "sae_bench_vendor"
for p in (VENDOR_ROOT, REPO_ROOT / "eval", REPO_ROOT, EVAL_PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sae_bench.common.transcoder_adapter import (  # noqa: E402
    TranscoderAdapter,
    load_google_transcoder_adapter,
    load_transcoder_adapter,
)
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig  # noqa: E402
from sae_bench.evals.sparse_probing.eval_output import (  # noqa: E402
    EVAL_TYPE_ID_SPARSE_PROBING,
    SparseProbingEvalOutput,
    SparseProbingLlmMetrics,
    SparseProbingMetricCategories,
    SparseProbingResultDetail,
    SparseProbingSaeMetrics,
)
from sae_bench.evals.sparse_probing.main import run_eval_single_sae  # noqa: E402
from sae_bench.sae_bench_utils import (  # noqa: E402
    activation_collection,
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)


def run_eval(
    config: SparseProbingEvalConfig,
    selected_transcoders: list[tuple[str, TranscoderAdapter]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
    artifacts_path: str = "artifacts",
):
    torch.set_grad_enabled(False)

    eval_instance_id = get_eval_uuid()
    sae_bench_commit_hash = get_sae_bench_version()

    output_path = os.path.join(output_path, EVAL_TYPE_ID_SPARSE_PROBING)
    os.makedirs(output_path, exist_ok=True)

    results_dict = {}
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for transcoder_name, transcoder in selected_transcoders:
        transcoder = transcoder.to(device=device, dtype=llm_dtype)

        result_filename = (
            f"{transcoder_name}_{config.model_name}_l{transcoder.cfg.hook_layer}.json"
        )
        sae_result_path = os.path.join(output_path, result_filename)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} as results already exist")
            continue

        artifacts_folder = os.path.join(
            artifacts_path,
            EVAL_TYPE_ID_SPARSE_PROBING,
            config.model_name,
            transcoder.cfg.hook_name,
        )

        # TranscoderAdapter is duck-type compatible with sae_lens.SAE:
        # has .encode(), .cfg.hook_layer, .cfg.hook_name, .dtype
        sparse_probing_results, per_dataset_class_results = run_eval_single_sae(
            config,
            transcoder,  # type: ignore[arg-type]
            model,
            device,
            artifacts_folder,
            save_activations,
        )

        # Build per-k accuracy dicts for aggregate metrics
        llm_metrics_kwargs: dict = {}
        sae_metrics_kwargs: dict = {}
        for key, val in sparse_probing_results.items():
            if isinstance(val, dict):
                continue
            if key.startswith("llm_"):
                llm_metrics_kwargs[key] = val
            elif key.startswith("sae_"):
                sae_metrics_kwargs[key] = val

        # Build per-dataset result details
        result_details = []
        for dataset_key, dataset_result in sparse_probing_results.items():
            if not isinstance(dataset_result, dict):
                continue
            result_details.append(
                SparseProbingResultDetail(
                    dataset_name=dataset_key.replace("_results", ""),
                    **dataset_result,
                )
            )

        eval_output = SparseProbingEvalOutput(
            eval_type_id=EVAL_TYPE_ID_SPARSE_PROBING,
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=SparseProbingMetricCategories(
                llm=SparseProbingLlmMetrics(**llm_metrics_kwargs),
                sae=SparseProbingSaeMetrics(**sae_metrics_kwargs),
            ),
            eval_result_details=result_details,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=transcoder_name,
            sae_lens_release_id=None,
            sae_lens_version="transcoder_adapter",
            sae_cfg_dict=transcoder.cfg.to_dict(),
        )

        eval_output.to_json_file(sae_result_path, indent=2)
        results_dict[transcoder_name] = asdict(eval_output)

        if clean_up_activations and artifacts_folder and os.path.exists(artifacts_folder):
            try:
                os.remove(artifacts_folder)
            except OSError:
                pass

        torch.cuda.empty_cache()

    return results_dict


def arg_parser():
    default_config = SparseProbingEvalConfig()
    parser = argparse.ArgumentParser(
        description="Run sparse probing evaluation for transcoders"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=default_config.dataset_names,
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--llm_batch_size", type=int, default=None, help="LLM batch size override"
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM",
    )
    parser.add_argument(
        "--random_seed", type=int, default=default_config.random_seed, help="Random seed"
    )
    parser.add_argument(
        "--sae_batch_size", type=int, default=None, help="SAE batch size override"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/evals/sparse_probing",
        help="Output folder",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to transcoder training config (json).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to transcoder checkpoint (.pt).",
    )
    parser.add_argument(
        "--google_params_path",
        type=str,
        default=None,
        help="Path to Google Gemma Scope params.npz.",
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="Hook layer override."
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--save_activations", action="store_true", help="Save activations to disk."
    )
    parser.add_argument(
        "--lower_vram_usage",
        action="store_true",
        help="Move the base model off GPU while encoding SAE/transcoder activations.",
    )
    return parser


def create_config_and_transcoders(
    args, device: str
) -> tuple[SparseProbingEvalConfig, list[tuple[str, TranscoderAdapter]]]:
    if not args.google_params_path and not args.config_path:
        raise ValueError("Provide --config_path or --google_params_path.")

    config = SparseProbingEvalConfig(
        model_name=args.model_name,
        dataset_names=args.dataset_names,
        random_seed=args.random_seed,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
            config.model_name
        ]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.sae_batch_size is not None:
        config.sae_batch_size = args.sae_batch_size
    if args.lower_vram_usage:
        config.lower_vram_usage = True

    selected_transcoders: list[tuple[str, TranscoderAdapter]] = []
    if args.google_params_path:
        if args.layer is None:
            raise ValueError("Please provide --layer when using --google_params_path.")
        name, adapter = load_google_transcoder_adapter(
            args.google_params_path,
            layer=args.layer,
            device=device,
            dtype=general_utils.str_to_dtype(config.llm_dtype),
        )
        selected_transcoders.append((name, adapter))
    else:
        name, adapter = load_transcoder_adapter(
            args.config_path,
            args.checkpoint_path,
            device=device,
            dtype=general_utils.str_to_dtype(config.llm_dtype),
        )
        selected_transcoders.append((name, adapter))

    if args.layer is not None:
        for _, adapter in selected_transcoders:
            adapter.cfg.hook_layer = args.layer

    print(f"Selected transcoders: {[name for name, _ in selected_transcoders]}")

    return config, selected_transcoders


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_transcoders = create_config_and_transcoders(args, device)
    os.makedirs(args.output_folder, exist_ok=True)

    results_dict = run_eval(
        config,
        selected_transcoders,
        device,
        args.output_folder,
        args.force_rerun,
        save_activations=args.save_activations,
    )

    end_time = time.time()
    print(f"Finished evaluation in {end_time - start_time:.2f} seconds")
