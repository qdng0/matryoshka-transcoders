import argparse
import gc
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Ensure eval/ is on sys.path so `sae_bench` resolves when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Also add vendor path for base classes used by eval_output
VENDOR_ROOT = PROJECT_ROOT / "sae_bench_vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))

from sae_bench.common.transcoder_adapter import (
    TranscoderAdapter,
    load_google_transcoder_adapter,
    load_transcoder_adapter,
)
from sae_bench.loss_recovered.eval_config import LossRecoveredEvalConfig
from sae_bench.loss_recovered.eval_output import (
    EVAL_TYPE_ID_LOSS_RECOVERED,
    LossRecoveredEvalOutput,
    LossRecoveredMetricCategories,
    LossRecoveredMetrics,
)
from sae_bench.sae_bench_utils import (
    activation_collection,
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
)


def _text_batches(
    dataset_name: str, tokens_column: str, batch_size: int
) -> Iterable[list[str]]:
    ds_iter = iter(load_dataset(dataset_name, split="train", streaming=True))
    while True:
        batch: list[str] = []
        while len(batch) < batch_size:
            row = next(ds_iter)
            text = row[tokens_column]
            if isinstance(text, list):
                text = " ".join(map(str, text))
            batch.append(str(text))
        yield batch


def _token_batches(
    model: HookedTransformer,
    dataset_name: str,
    tokens_column: str,
    batch_size: int,
    context_size: int,
) -> Iterable[torch.Tensor]:
    for texts in _text_batches(dataset_name, tokens_column, batch_size):
        tokens = model.to_tokens(
            texts, truncate=True, prepend_bos=True, move_to_device=True
        )
        yield tokens[:, :context_size]


def _ignore_token_ids(model: HookedTransformer) -> set[int]:
    ids: set[int] = set()
    for tok in [
        getattr(model.tokenizer, "pad_token_id", None),
        getattr(model.tokenizer, "eos_token_id", None),
        getattr(model.tokenizer, "bos_token_id", None),
    ]:
        if tok is not None:
            ids.add(int(tok))
    return ids


def _apply_token_mask(
    values: torch.Tensor, tokens: torch.Tensor, ignore_tokens: set[int]
) -> torch.Tensor:
    if not ignore_tokens:
        return values.reshape(-1)

    mask = torch.ones_like(tokens, dtype=torch.bool)
    for tok in ignore_tokens:
        mask &= tokens != tok

    if values.shape[1] != mask.shape[1]:
        mask = mask[:, : values.shape[1]]

    mask = mask.to(values.device)
    return values[mask]


@torch.no_grad()
def _batch_losses(
    model: HookedTransformer,
    transcoder: TranscoderAdapter,
    tokens: torch.Tensor,
    ignore_tokens: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (baseline, ablated, transcoder) CE losses flattened."""
    # Baseline
    _, ce_without = model(tokens, return_type="both", loss_per_token=True)

    # Ablation hook
    def zero_ablate_hook(activations: torch.Tensor, hook):
        return torch.zeros_like(activations)

    _, ce_with_ablation = model.run_with_hooks(
        tokens,
        return_type="both",
        loss_per_token=True,
        fwd_hooks=[(transcoder.cfg.target_hook_point, zero_ablate_hook)],
    )

    cached_source: dict[str, torch.Tensor] = {}

    def cache_source_hook(activations: torch.Tensor, hook):
        cached_source["source"] = activations
        return activations

    def replace_target_hook(activations: torch.Tensor, hook):
        source_acts = cached_source.get("source")
        if source_acts is None:
            raise RuntimeError("Source activations not cached before target hook.")
        source_acts = source_acts.to(device=transcoder.device, dtype=transcoder.dtype)
        features = transcoder.encode(source_acts)
        recon = transcoder.decode(features).to(dtype=activations.dtype)
        recon = recon.to(device=activations.device)

        if ignore_tokens:
            mask = torch.ones_like(tokens, dtype=torch.bool)
            for tok in ignore_tokens:
                mask &= tokens != tok
            if recon.shape[:2] != mask.shape:
                mask = mask[:, : recon.shape[1]]
            recon = torch.where(mask[..., None].to(recon.device), recon, activations)

        return recon

    _, ce_with_transcoder = model.run_with_hooks(
        tokens,
        return_type="both",
        loss_per_token=True,
        fwd_hooks=[
            (transcoder.cfg.source_hook_point, cache_source_hook),
            (transcoder.cfg.target_hook_point, replace_target_hook),
        ],
    )

    ce_without = _apply_token_mask(ce_without, tokens, ignore_tokens)
    ce_with_ablation = _apply_token_mask(ce_with_ablation, tokens, ignore_tokens)
    ce_with_transcoder = _apply_token_mask(ce_with_transcoder, tokens, ignore_tokens)

    return ce_without, ce_with_ablation, ce_with_transcoder


def run_eval(
    config: LossRecoveredEvalConfig,
    selected_transcoders: list[tuple[str, TranscoderAdapter]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
):
    torch.set_grad_enabled(False)
    torch.manual_seed(config.random_seed)

    eval_instance_id = get_eval_uuid()
    sae_bench_commit_hash = get_sae_bench_version()

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )
    ignore_tokens = _ignore_token_ids(model)

    for transcoder_name, transcoder in tqdm(
        selected_transcoders, desc="Running loss recovered evaluation"
    ):
        transcoder = transcoder.to(device=device, dtype=llm_dtype)

        if config.layer == -1:
            config.layer = transcoder.cfg.hook_layer

        model_tag = config.model_name.replace("/", "_")
        result_filename = f"{transcoder_name}_{model_tag}_l{config.layer}.json"
        result_path = os.path.join(output_path, result_filename)

        if os.path.exists(result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} as results already exist")
            continue

        token_iter = _token_batches(
            model=model,
            dataset_name=config.dataset,
            tokens_column=config.tokens_column,
            batch_size=config.batch_size_prompts,
            context_size=config.context_size,
        )

        ce_without_sum = 0.0
        ce_with_ablation_sum = 0.0
        ce_with_transcoder_sum = 0.0
        total_tokens = 0

        for _ in range(config.n_batches):
            tokens = next(token_iter)
            ce_without, ce_with_ablation, ce_with_transcoder = _batch_losses(
                model, transcoder, tokens, ignore_tokens
            )

            ce_without_sum += ce_without.sum().item()
            ce_with_ablation_sum += ce_with_ablation.sum().item()
            ce_with_transcoder_sum += ce_with_transcoder.sum().item()
            total_tokens += ce_without.numel()

        ce_loss_without = ce_without_sum / max(1, total_tokens)
        ce_loss_with_ablation = ce_with_ablation_sum / max(1, total_tokens)
        ce_loss_with_transcoder = ce_with_transcoder_sum / max(1, total_tokens)

        denom = ce_loss_with_ablation - ce_loss_without
        if abs(denom) < config.stability_epsilon:
            loss_recovered_score = float("nan")
        else:
            loss_recovered_score = (ce_loss_with_ablation - ce_loss_with_transcoder) / denom

        eval_output = LossRecoveredEvalOutput(
            eval_type_id=EVAL_TYPE_ID_LOSS_RECOVERED,
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=LossRecoveredMetricCategories(
                performance=LossRecoveredMetrics(
                    loss_recovered_score=loss_recovered_score,
                    ce_loss_with_transcoder=ce_loss_with_transcoder,
                    ce_loss_without_transcoder=ce_loss_without,
                    ce_loss_with_ablation=ce_loss_with_ablation,
                )
            ),
            eval_result_details=[],
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=transcoder_name,
            sae_lens_release_id=None,
            sae_lens_version="transcoder_adapter",
            sae_cfg_dict=transcoder.cfg.to_dict(),
        )

        results_dict[transcoder_name] = asdict(eval_output)

        eval_output.to_json_file(result_path, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


def arg_parser():
    default_config = LossRecoveredEvalConfig()

    parser = argparse.ArgumentParser(description="Run loss recovered evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_config.dataset,
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--tokens_column",
        type=str,
        default=default_config.tokens_column,
        help="Column containing text in the dataset.",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=default_config.context_size,
        help="Context length to evaluate on.",
    )
    parser.add_argument(
        "--batch_size_prompts",
        type=int,
        default=default_config.batch_size_prompts,
        help="Batch size for prompts.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=default_config.random_seed,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=default_config.n_batches,
        help="Number of batches to evaluate.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/loss_recovered",
        help="Output folder for results.",
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
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, populated from activation_collection.",
    )
    parser.add_argument(
        "--stability_epsilon",
        type=float,
        default=default_config.stability_epsilon,
        help="Small value to avoid divide-by-zero in the loss recovered score.",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )

    return parser


def create_config_and_transcoders(
    args, device: str
) -> tuple[LossRecoveredEvalConfig, list[tuple[str, TranscoderAdapter]]]:
    if not args.google_params_path and not args.config_path:
        raise ValueError("Provide --config_path or --google_params_path.")

    config_path_str = (
        str(Path(args.google_params_path).resolve()) if args.google_params_path
        else str(Path(args.config_path).resolve())
    )
    config = LossRecoveredEvalConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        tokens_column=args.tokens_column,
        context_size=args.context_size,
        batch_size_prompts=args.batch_size_prompts,
        n_batches=args.n_batches,
        llm_dtype=args.llm_dtype if args.llm_dtype is not None else "float32",
        layer=args.layer if args.layer is not None else -1,
        config_path=config_path_str,
        stability_epsilon=args.stability_epsilon,
        random_seed=args.random_seed,
    )

    if args.llm_dtype is None:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

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
        config.layer = args.layer
        config.config_path = str(Path(args.google_params_path).resolve())
    else:
        selected_transcoders.append(
            load_transcoder_adapter(
                args.config_path,
                args.checkpoint_path,
                device=device,
                dtype=general_utils.str_to_dtype(config.llm_dtype),
            )
        )

    releases = set([name for name, _ in selected_transcoders])

    print(f"Selected transcoders: {releases}")

    for name, _ in selected_transcoders:
        print(f"Selected transcoder: {name}")

    if config.layer == -1:
        config.layer = selected_transcoders[0][1].cfg.hook_layer

    return config, selected_transcoders


if __name__ == "__main__":
    """
    python eval/sae_bench/loss_recovered/main.py \
    --config_path path/to/transcoder/config.json \
    --checkpoint_path path/to/checkpoints/final.pt \
    --model_name gemma-2-2b \
    --layer 8
    """
    args = arg_parser().parse_args()

    device = general_utils.setup_environment()
    torch.manual_seed(args.random_seed)

    start_time = time.time()

    config, selected_transcoders = create_config_and_transcoders(args, device)
    os.makedirs(args.output_folder, exist_ok=True)

    results_dict = run_eval(
        config, selected_transcoders, device, args.output_folder, args.force_rerun
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time:.2f} seconds")


