import argparse
import gc
import os
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sae_bench.absorption.eval_config import AbsorptionEvalConfig  # noqa: E402
from sae_bench.absorption.eval_output import (  # noqa: E402
    EVAL_TYPE_ID_ABSORPTION,
    AbsorptionEvalOutput,
    AbsorptionMeanMetrics,
    AbsorptionMetricCategories,
    AbsorptionResultDetail,
)
from sae_bench.absorption.feature_absorption import (  # noqa: E402
    run_feature_absortion_experiment,
)
from sae_bench.absorption.k_sparse_probing import run_k_sparse_probing_experiment  # noqa: E402
from sae_bench.common.transcoder_adapter import (  # noqa: E402
    TranscoderAdapter,
    load_transcoder_adapter,
    load_google_transcoder_adapter,
)

# ---------------------------------------------------------------------------
# Inlined utilities (previously from sae_bench.sae_bench_utils)
# ---------------------------------------------------------------------------
LLM_NAME_TO_BATCH_SIZE = {
    "pythia-70m-deduped": 512,
    "pythia-160m-deduped": 256,
    "gemma-2-2b": 32,
    "gemma-2-9b": 32,
    "gemma-2-2b-it": 32,
    "gemma-2-9b-it": 32,
}

LLM_NAME_TO_DTYPE = {
    "pythia-70m-deduped": "float32",
    "pythia-160m-deduped": "float32",
    "gemma-2-2b": "bfloat16",
    "gemma-2-2b-it": "bfloat16",
    "gemma-2-9b": "bfloat16",
    "gemma-2-9b-it": "bfloat16",
}


def str_to_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def setup_environment(seed: int = 42, cuda_device: str = "cuda") -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if cuda_device == "cpu" or not torch.cuda.is_available():
        return "cpu"

    if cuda_device.startswith("cuda:") and len(cuda_device) > 5:
        torch.cuda.set_device(int(cuda_device.split(":")[1]))
    elif cuda_device == "cuda":
        torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    return cuda_device


def _build_diagnostics(k_sparse_probing_results: pd.DataFrame, max_k_value: int) -> dict:
    """Build diagnostic metrics from k-sparse probing results for deeper analysis."""
    per_letter = {}
    for _, row in k_sparse_probing_results.iterrows():
        letter = row["letter"]
        entry = {
            "f1_probe": float(row.get("f1_probe", 0)),
            "recall_probe": float(row.get("recall_probe", 0)),
            "precision_probe": float(row.get("precision_probe", 0)),
            "num_split_features": int(row.get("num_split_features", 0)),
        }
        for k in [1, 3, 5, min(10, max_k_value)]:
            if k > max_k_value:
                continue
            col = f"f1_sparse_sae_{k}"
            if col in row.index:
                entry[f"f1_feature_k{k}"] = float(row[col])
            dec_col = f"cos_probe_sae_dec_k_{k}"
            if dec_col in row.index:
                cos_vals = row[dec_col]
                if hasattr(cos_vals, '__len__'):
                    entry[f"max_dec_cos_k{k}"] = float(max(cos_vals))
                else:
                    entry[f"max_dec_cos_k{k}"] = float(cos_vals)
        per_letter[letter] = entry

    # Summary across letters
    def _mean(key):
        vals = [v[key] for v in per_letter.values() if key in v]
        return float(sum(vals) / len(vals)) if vals else 0.0

    summary = {
        "mean_f1_probe": _mean("f1_probe"),
        "mean_f1_feature_k1": _mean("f1_feature_k1"),
        "mean_f1_feature_k5": _mean("f1_feature_k5"),
        "mean_max_dec_cos_k1": _mean("max_dec_cos_k1"),
        "num_letters_evaluated": len(per_letter),
    }
    return {"per_letter_probing": per_letter, "summary": summary}


def run_eval(
    config: AbsorptionEvalConfig,
    selected_transcoders: list[tuple[str, TranscoderAdapter]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
):
    torch.set_grad_enabled(True)
    """
    selected_transcoders is a list of tuples of (transcoder_name, TranscoderAdapter object)
    """

    if "gemma" not in config.model_name:
        print(
            "\n\n\nWARNING: We recommend running this eval on LLMS >= 2B parameters\n\n\n"
        )

    eval_instance_id = str(uuid.uuid4())
    sae_bench_commit_hash = "standalone"

    results_dict = {}

    llm_dtype = str_to_dtype(config.llm_dtype)

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for transcoder_name, transcoder in tqdm(
        selected_transcoders, desc="Running absorption evaluation on transcoders"
    ):
        transcoder = transcoder.to(device=device, dtype=llm_dtype)
        source_hook_point = transcoder.cfg.source_hook_point
        target_hook_point = transcoder.cfg.target_hook_point

        model_tag = config.model_name.replace("/", "_")
        result_filename = f"{transcoder_name}_{model_tag}_l{config.layer}.json"

        absorption_dir = os.path.join(output_path, "absorption")
        k_sparse_dir = os.path.join(output_path, "k_sparse_probing")
        os.makedirs(absorption_dir, exist_ok=True)
        os.makedirs(k_sparse_dir, exist_ok=True)

        sae_result_path = os.path.join(absorption_dir, result_filename)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} as results already exist")
            continue
        
        k_sparse_probing_results = run_k_sparse_probing_experiment(
            model=model,
            transcoder=transcoder,
            layer=config.layer,
            transcoder_name=transcoder_name,
            force=force_rerun,
            max_k_value=config.max_k_value,
            f1_jump_threshold=config.f1_jump_threshold,
            prompt_template=config.prompt_template,
            prompt_token_pos=config.prompt_token_pos,
            source_hook_point=source_hook_point,
            target_hook_point=target_hook_point,
            device=device,
            k_sparse_probe_l1_decay=config.k_sparse_probe_l1_decay,
            k_sparse_probe_batch_size=config.k_sparse_probe_batch_size,
            k_sparse_probe_num_epochs=config.k_sparse_probe_num_epochs,
            eval_batch_size=config.eval_k_sparse_probe_batch_size,
            precalc_k_sparse_probe_sae_acts=config.precalc_k_sparse_probe_sae_acts,
        )

        # Save k_sparse_probing_results in separate subdirectory
        k_sparse_probing_file = result_filename.replace(".json", "_k_sparse_probing.json")
        k_sparse_probing_path = os.path.join(k_sparse_dir, k_sparse_probing_file)
        k_sparse_probing_results.to_json(
            k_sparse_probing_path, orient="records", indent=4
        )
        
        # check we have enough first-letter features to evaluate absorption
        if (
            k_sparse_probing_results["f1_probe"] > config.min_GT_probe_f1
        ).sum() < config.min_feats_for_eval:
            print(
                f"\n\n\nCannot evaluate absorption due to insufficent first-letter features detected in model: {config.model_name}\n\n\n"
            )
            break

        raw_df = run_feature_absortion_experiment(
            model=model,
            transcoder=transcoder,
            layer=config.layer,
            sae_name=transcoder_name,
            force=force_rerun,
            max_k_value=config.max_k_value,
            feature_split_f1_jump_threshold=config.f1_jump_threshold,
            prompt_template=config.prompt_template,
            prompt_token_pos=config.prompt_token_pos,
            batch_size=config.llm_batch_size,
            source_hook_point=source_hook_point,
            target_hook_point=target_hook_point,
            device=device,
        )
        agg_df = _aggregate_results_df(raw_df)

        # aggregate results and produce the output
        mean_absorption_fractions = []
        full_absorption_rates = []
        num_split_features = []
        eval_result_details = []
        for _, row in agg_df.iterrows():
            letter = row["letter"]
            if (
                k_sparse_probing_results[
                    k_sparse_probing_results["letter"] == row["letter"]
                ]["f1_probe"].item()
                > config.min_GT_probe_f1
            ):
                mean_absorption_fractions.append(row["mean_absorption_fraction"])
                full_absorption_rates.append(row["full_absorption_rate"])
                num_split_features.append(row["num_split_feats"])
                eval_result_details.append(
                    AbsorptionResultDetail(
                        first_letter=letter,  # type: ignore
                        mean_absorption_fraction=row["mean_absorption_fraction"],  # type: ignore
                        full_absorption_rate=row["full_absorption_rate"],  # type: ignore
                        num_full_absorption=row["num_full_absorption"],  # type: ignore
                        num_probe_true_positives=row["num_probe_true_positives"],  # type: ignore
                        num_split_features=row["num_split_feats"],  # type: ignore
                    )
                )
        diagnostics = _build_diagnostics(k_sparse_probing_results, config.max_k_value)
        eval_output = AbsorptionEvalOutput(
            eval_type_id=EVAL_TYPE_ID_ABSORPTION,
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=AbsorptionMetricCategories(
                mean=AbsorptionMeanMetrics(
                    mean_absorption_fraction_score=statistics.mean(
                        mean_absorption_fractions
                    ),
                    mean_full_absorption_score=statistics.mean(full_absorption_rates),
                    mean_num_split_features=statistics.mean(num_split_features),
                    std_dev_absorption_fraction_score=statistics.stdev(
                        mean_absorption_fractions
                    ),
                    std_dev_full_absorption_score=statistics.stdev(
                        full_absorption_rates
                    ),
                    std_dev_num_split_features=statistics.stdev(num_split_features),
                )
            ),
            eval_result_details=eval_result_details,
            eval_result_unstructured=diagnostics,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=transcoder_name,
            sae_lens_release_id=None,
            sae_lens_version="transcoder_adapter",
            sae_cfg_dict=transcoder.cfg.to_dict(),
        )

        results_dict[transcoder_name] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


def _aggregate_results_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    agg_df = (
        df[["letter", "absorption_fraction", "is_full_absorption"]]
        .groupby(["letter"])
        .sum()
        .reset_index()
        .merge(
            df[["letter", "num_probe_true_positives", "split_feats"]]
            .groupby(["letter"])
            .agg(
                {
                    "num_probe_true_positives": "mean",
                    "split_feats": lambda x: x.iloc[
                        0
                    ],  # Take the first split_feats list for each letter
                }
            )
            .reset_index()
        )
    )
    agg_df["num_split_feats"] = agg_df["split_feats"].apply(len)
    agg_df["mean_absorption_fraction"] = (
        agg_df["absorption_fraction"] / agg_df["num_probe_true_positives"]
    )
    agg_df["num_full_absorption"] = agg_df["is_full_absorption"]
    agg_df["full_absorption_rate"] = (
        agg_df["num_full_absorption"] / agg_df["num_probe_true_positives"]
    )
    return agg_df


def arg_parser():
    default_config = AbsorptionEvalConfig()

    parser = argparse.ArgumentParser(description="Run absorption evaluation")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=default_config.random_seed,
        help="Random seed",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--f1_jump_threshold",
        type=float,
        default=default_config.f1_jump_threshold,
        help="F1 jump threshold",
    )
    parser.add_argument(
        "--max_k_value",
        type=int,
        default=default_config.max_k_value,
        help="Maximum k value",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=default_config.prompt_template,
        help="Prompt template",
    )
    parser.add_argument(
        "--prompt_token_pos",
        type=int,
        default=default_config.prompt_token_pos,
        help="Prompt token position",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="Path to transcoder training config (json). Required if not using --google_params_path.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to transcoder checkpoint (.pt). If omitted, uses last checkpoint in config directory.",
    )
    parser.add_argument(
        "--google_params_path",
        type=str,
        default=None,
        help="Path to Google Gemma Scope params.npz (uses Google transcoder loader).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Hook layer to evaluate. If omitted, inferred from config.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/absorption",
        help="Output folder",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to the transcoder name for cache/result filenames.",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )
    parser.add_argument(
        "--k_sparse_probe_l1_decay",
        type=float,
        default=default_config.k_sparse_probe_l1_decay,
        help="L1 decay for k-sparse probes.",
    )
    parser.add_argument(
        "--k_sparse_probe_batch_size",
        type=float,
        default=default_config.k_sparse_probe_batch_size,
        help="L1 decay for k-sparse probes.",
    )

    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="cuda",
        help="Torch device (e.g. cuda:0, cuda:1). Use two terminals or parallel jobs "
        "with cuda:0 and cuda:1 to load one Gemma copy per GPU.",
    )

    return parser


def create_config_and_transcoders(
    args, device: str
) -> tuple[AbsorptionEvalConfig, list[tuple[str, TranscoderAdapter]]]:
    config = AbsorptionEvalConfig(
        random_seed=args.random_seed,
        f1_jump_threshold=args.f1_jump_threshold,
        max_k_value=args.max_k_value,
        prompt_template=args.prompt_template,
        prompt_token_pos=args.prompt_token_pos,
        model_name=args.model_name,
        k_sparse_probe_l1_decay=args.k_sparse_probe_l1_decay,
        k_sparse_probe_batch_size=args.k_sparse_probe_batch_size,
        layer=args.layer if args.layer is not None else -1,
        config_path=str(Path(args.config_path).resolve()) if args.config_path else "",
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = LLM_NAME_TO_BATCH_SIZE.get(config.model_name, 32)

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = LLM_NAME_TO_DTYPE.get(config.model_name, "bfloat16")

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    selected_transcoders: list[tuple[str, TranscoderAdapter]] = []
    if args.google_params_path:
        if args.layer is None:
            raise ValueError("Please provide --layer when using --google_params_path.")
        name, adapter = load_google_transcoder_adapter(
            args.google_params_path,
            layer=args.layer,
            device=device,
            dtype=str_to_dtype(config.llm_dtype),
        )
        selected_transcoders.append((name, adapter))
        config.layer = args.layer
        config.config_path = str(Path(args.google_params_path).resolve())
    else:
        selected_transcoders.append(
            load_transcoder_adapter(
                args.config_path,
                args.checkpoint_path,
                device=device,
                dtype=str_to_dtype(config.llm_dtype),
            )
        )

    if args.tag:
        safe_tag = args.tag.strip().replace("/", "_").replace(" ", "_")
        if safe_tag:
            selected_transcoders = [
                (f"{name}_{safe_tag}", adapter)
                for name, adapter in selected_transcoders
            ]

    print(f"Selected transcoders: {[name for name, _ in selected_transcoders]}")
    if config.layer == -1:
        config.layer = selected_transcoders[0][1].cfg.hook_layer
    config.source_hook_point = selected_transcoders[0][1].cfg.source_hook_point
    config.target_hook_point = selected_transcoders[0][1].cfg.target_hook_point

    return config, selected_transcoders


if __name__ == "__main__":
    """
    python eval/sae_bench/absorption_refactor/main.py \
    --config_path path/to/config.json \
    --checkpoint_path path/to/checkpoints/final.pt \
    --model_name gemma-2-2b \
    --layer 12
    """
    args = arg_parser().parse_args()

    device = setup_environment(seed=args.random_seed, cuda_device=args.cuda_device)

    start_time = time.time()

    config, selected_transcoders = create_config_and_transcoders(args, device)
    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected transcoders
    results_dict = run_eval(
        config, selected_transcoders, device, args.output_folder, args.force_rerun
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time:.2f} seconds")
