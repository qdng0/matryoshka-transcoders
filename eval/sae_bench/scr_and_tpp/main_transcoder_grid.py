import argparse
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = Path(__file__).resolve().parents[2]
EVAL_PARENT = EVAL_ROOT.parent
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "sae_bench_vendor"
for p in (VENDOR_ROOT, REPO_ROOT / "eval", REPO_ROOT, EVAL_PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sae_bench.common.transcoder_eval_adapter import (  # noqa: E402
    TranscoderEvalAdapter,
    load_google_transcoder_eval_adapter,
    load_transcoder_eval_adapter,
)
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig  # noqa: E402
from sae_bench.evals.scr_and_tpp.eval_output import (  # noqa: E402
    EVAL_TYPE_ID_SCR,
    EVAL_TYPE_ID_TPP,
    ScrEvalOutput,
    ScrMetricCategories,
    ScrMetrics,
    ScrResultDetail,
    TppEvalOutput,
    TppMetricCategories,
    TppMetrics,
    TppResultDetail,
)
from sae_bench.evals.scr_and_tpp.main import run_eval_single_sae  # noqa: E402
from sae_bench.scr_and_tpp.transcoder_scr_fix import (  # noqa: E402
    run_eval_single_sae_transcoder,
)
from sae_bench.sae_bench_utils import (  # noqa: E402
    activation_collection,
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
)


def run_eval(
    config: ScrAndTppEvalConfig,
    selected_transcoders: list[tuple[str, TranscoderEvalAdapter]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
    artifacts_path: str = "artifacts",
    transcoder_fix: bool = False,
):
    eval_instance_id = get_eval_uuid()
    sae_bench_commit_hash = get_sae_bench_version()

    eval_type = EVAL_TYPE_ID_SCR if config.perform_scr else EVAL_TYPE_ID_TPP
    output_path = os.path.join(output_path, eval_type)
    os.makedirs(output_path, exist_ok=True)

    results_dict = {}
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for transcoder_name, transcoder in selected_transcoders:
        transcoder = transcoder.to(device=device, dtype=llm_dtype)
        result_filename = f"{transcoder_name}_{config.model_name}_l{transcoder.cfg.hook_layer}.json"
        sae_result_path = os.path.join(output_path, result_filename)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} as results already exist")
            continue

        artifacts_folder = os.path.join(
            artifacts_path,
            eval_type,
            config.model_name,
            transcoder.cfg.hook_name,
        )

        if transcoder_fix:
            scr_or_tpp_results, per_dataset_class_results = run_eval_single_sae_transcoder(
                config, transcoder, model, device,
            )
        else:
            scr_or_tpp_results, per_dataset_class_results = run_eval_single_sae(
                config, transcoder, model, device, artifacts_folder, save_activations,
            )

        if eval_type == EVAL_TYPE_ID_SCR:
            eval_output = ScrEvalOutput(
                eval_type_id=eval_type,
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=ScrMetricCategories(
                    scr_metrics=ScrMetrics(
                        **{k: v for k, v in scr_or_tpp_results.items() if not isinstance(v, dict)}
                    )
                ),
                eval_result_details=[
                    ScrResultDetail(dataset_name=dataset_name, **result)
                    for dataset_name, result in scr_or_tpp_results.items()
                    if isinstance(result, dict)
                ],
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=transcoder_name,
                sae_lens_release_id=None,
                sae_lens_version="transcoder_eval_adapter",
                sae_cfg_dict=transcoder.cfg.to_dict(),
            )
        else:
            eval_output = TppEvalOutput(
                eval_type_id=eval_type,
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=TppMetricCategories(
                    tpp_metrics=TppMetrics(
                        **{k: v for k, v in scr_or_tpp_results.items() if not isinstance(v, dict)}
                    )
                ),
                eval_result_details=[
                    TppResultDetail(dataset_name=dataset_name, **result)
                    for dataset_name, result in scr_or_tpp_results.items()
                    if isinstance(result, dict)
                ],
                eval_result_unstructured=per_dataset_class_results,
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=transcoder_name,
                sae_lens_release_id=None,
                sae_lens_version="transcoder_eval_adapter",
                sae_cfg_dict=transcoder.cfg.to_dict(),
            )

        eval_output.to_json_file(sae_result_path, indent=2)
        results_dict[transcoder_name] = asdict(eval_output)

        if clean_up_activations and artifacts_folder and os.path.exists(artifacts_folder):
            import shutil
            shutil.rmtree(artifacts_folder)

        torch.cuda.empty_cache()

    return results_dict


def arg_parser():
    parser = argparse.ArgumentParser(description="Run SCR/TPP evaluation for transcoders")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=ScrAndTppEvalConfig().dataset_names,
        help="Datasets to evaluate",
    )
    parser.add_argument("--perform_scr", action="store_true", help="If set, run SCR; otherwise TPP")
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
    parser.add_argument("--sae_batch_size", type=int, default=None, help="SAE batch size override")
    parser.add_argument("--train_set_size", type=int, default=None, help="Train set size override per dataset.")
    parser.add_argument("--test_set_size", type=int, default=None, help="Test set size override per dataset.")
    parser.add_argument("--output_folder", type=str, default="results/evals/scr_and_tpp", help="Output folder")
    parser.add_argument("--config_path", type=str, default=None, help="Path to transcoder training config (json).")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to transcoder checkpoint (.pt).")
    parser.add_argument("--google_params_path", type=str, default=None, help="Path to Google Gemma Scope params.npz.")
    parser.add_argument("--layer", type=int, default=None, help="Hook layer to evaluate. If omitted, inferred from config.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument("--clean_up_activations", action="store_true", help="Delete saved activations after eval.")
    parser.add_argument(
        "--save_activations",
        action="store_false",
        help="Match vendored SAE Bench semantics: pass this flag to disable saving activations.",
    )
    parser.add_argument(
        "--transcoder_fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the transcoder-aware dual-hook SCR/TPP pipeline (default, used for the paper). "
            "--no-transcoder_fix runs the unmodified single-hook SAEBench code instead."
        ),
    )
    return parser


def create_config_and_transcoders(args, device: str):
    if not args.google_params_path and not args.config_path:
        raise ValueError("Provide --config_path or --google_params_path.")

    config = ScrAndTppEvalConfig(
        model_name=args.model_name,
        dataset_names=args.dataset_names,
        perform_scr=args.perform_scr,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed
    elif args.paper_release_defaults:
        config.random_seed = 68

    if args.sae_batch_size is not None:
        config.sae_batch_size = args.sae_batch_size
    if args.train_set_size is not None:
        config.train_set_size = args.train_set_size
    if args.test_set_size is not None:
        config.test_set_size = args.test_set_size

    target_dtype = general_utils.str_to_dtype(config.llm_dtype)
    selected_transcoders = []
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
        args.clean_up_activations,
        args.save_activations,
        transcoder_fix=args.transcoder_fix,
    )

    end_time = time.time()
    print(f"Finished evaluation in {end_time - start_time:.2f} seconds")
