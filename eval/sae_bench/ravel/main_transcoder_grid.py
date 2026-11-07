import argparse
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = Path(__file__).resolve().parents[2]
EVAL_PARENT = EVAL_ROOT.parent
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "sae_bench_vendor"
for p in (VENDOR_ROOT, REPO_ROOT / "eval", REPO_ROOT, EVAL_PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Install transcoder-aware MDBM / intervention shadows before the vendor
# ``run_eval_single_sae`` is imported. See the docstring of ``transcoder_mdbm.py``.
from eval.sae_bench.ravel import transcoder_mdbm as _tc_mdbm  # noqa: E402
from eval.sae_bench.ravel import transcoder_intervention as _tc_intervention  # noqa: E402

sys.modules["sae_bench.evals.ravel.mdbm"] = _tc_mdbm
sys.modules["sae_bench.evals.ravel.intervention"] = _tc_intervention

from sae_bench.common.transcoder_eval_adapter import (  # noqa: E402
    TranscoderEvalAdapter,
    load_google_transcoder_eval_adapter,
    load_transcoder_eval_adapter,
)
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig  # noqa: E402
from sae_bench.evals.ravel.eval_output import (  # noqa: E402
    EVAL_TYPE_ID_RAVEL,
    RAVELEvalOutput,
    RAVELMetricCategories,
    RAVELMetricResults,
)
from sae_bench.evals.ravel.main import run_eval_single_sae  # noqa: E402
from sae_bench.sae_bench_utils import (  # noqa: E402
    activation_collection,
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
)

LLM_NAME_MAP = {
    "gemma-2-2b": "google/gemma-2-2b",
    "pythia-160m-deduped": "EleutherAI/pythia-160m-deduped",
}


def run_eval(
    config: RAVELEvalConfig,
    selected_transcoders: list[tuple[str, TranscoderEvalAdapter]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    artifacts_path: str = "artifacts",
):
    eval_instance_id = get_eval_uuid()
    sae_bench_commit_hash = get_sae_bench_version()

    results_dict = {}
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    # config.model_name is already the full HF name (resolved in create_config_and_transcoders)
    model_kwargs = {"attn_implementation": "eager"} if "gemma" in config.model_name else {}
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device,
        torch_dtype=llm_dtype,
        **model_kwargs,
    )

    os.makedirs(output_path, exist_ok=True)

    for transcoder_name, transcoder in selected_transcoders:
        transcoder = transcoder.to(device=device, dtype=llm_dtype)
        result_filename = f"{transcoder_name}_{config.model_name.replace('/', '_')}_l{transcoder.cfg.hook_layer}.json"
        sae_result_path = os.path.join(output_path, result_filename)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} as results already exist")
            continue

        artifacts_folder = os.path.join(
            artifacts_path,
            EVAL_TYPE_ID_RAVEL,
            config.model_name,
            transcoder.cfg.hook_name,
        )

        eval_results, per_class_dict = run_eval_single_sae(
            config,
            transcoder,
            model,
            device,
            artifacts_folder,
        )

        eval_output = RAVELEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=RAVELMetricCategories(
                ravel=RAVELMetricResults(
                    disentanglement_score=eval_results["disentangle_score"],  # type: ignore
                    cause_score=eval_results["cause_score"],  # type: ignore
                    isolation_score=eval_results["isolation_score"],  # type: ignore
                )
            ),
            eval_result_details=[],
            eval_result_unstructured=per_class_dict,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=transcoder_name,
            sae_lens_release_id=None,
            sae_lens_version="transcoder_eval_adapter",
            sae_cfg_dict=transcoder.cfg.to_dict(),
        )

        eval_output.to_json_file(sae_result_path, indent=2)
        results_dict[transcoder_name] = asdict(eval_output)
        torch.cuda.empty_cache()

    return results_dict


def arg_parser():
    default_config = RAVELEvalConfig()
    parser = argparse.ArgumentParser(description="Run RAVEL evaluation for transcoders")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--llm_batch_size", type=int, default=None, help="LLM batch size override")
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, populated from activation_collection.",
    )
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--paper_release_defaults",
        action="store_true",
        help="Match the paper-release defaults where known (notably random_seed=68).",
    )
    parser.add_argument("--output_folder", type=str, default="results/evals/ravel", help="Output folder")
    parser.add_argument("--config_path", type=str, default=None, help="Path to transcoder training config (json).")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to transcoder checkpoint (.pt).")
    parser.add_argument("--google_params_path", type=str, default=None, help="Path to Google Gemma Scope params.npz.")
    parser.add_argument("--layer", type=int, default=None, help="Hook layer to evaluate. If omitted, inferred from config.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    return parser


def create_config_and_transcoders(args, device: str):
    if not args.google_params_path and not args.config_path:
        raise ValueError("Provide --config_path or --google_params_path.")

    model_key = args.model_name
    if model_key not in activation_collection.LLM_NAME_TO_BATCH_SIZE and "gemma-2-2b" in model_key:
        model_key = "gemma-2-2b"

    # Vendor RAVEL code calls AutoTokenizer.from_pretrained(config.model_name),
    # so the config must use the full HF model name.
    hf_model_name = LLM_NAME_MAP.get(args.model_name, args.model_name)
    config = RAVELEvalConfig(model_name=hf_model_name)

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[model_key] // 4

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[model_key]

    if args.random_seed is not None:
        config.random_seed = args.random_seed
    elif args.paper_release_defaults:
        config.random_seed = 68

    selected_transcoders = []
    target_dtype = general_utils.str_to_dtype(config.llm_dtype)
    if args.google_params_path:
        if args.layer is None:
            raise ValueError("Please provide --layer when using --google_params_path.")
        name, adapter = load_google_transcoder_eval_adapter(
            args.google_params_path,
            layer=args.layer,
            device=device,
            dtype=target_dtype,
            model_name=args.model_name,
        )
        selected_transcoders.append((name, adapter))
        config.config_path = str(Path(args.google_params_path).resolve())
    else:
        name, adapter = load_transcoder_eval_adapter(
            args.config_path,
            args.checkpoint_path,
            device=device,
            dtype=target_dtype,
        )
        selected_transcoders.append((name, adapter))
        config.config_path = str(Path(args.config_path).resolve())

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

    run_eval(
        config,
        selected_transcoders,
        device,
        args.output_folder,
        args.force_rerun,
    )

    end_time = time.time()
    print(f"Finished evaluation in {end_time - start_time:.2f} seconds")
