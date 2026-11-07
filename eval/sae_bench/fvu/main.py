"""FVU (Fraction of Variance Unexplained) evaluation for transcoders.

Measures reconstruction quality by computing FVU = MSE / Var(target).
Also reports L0 (average number of active features per token).

Usage:
    python eval/sae_bench/fvu/main.py \
        --config_path path/to/config.json \
        --model_name gemma-2-2b

    python eval/sae_bench/fvu/main.py \
        --google_params_path path/to/params.npz \
        --model_name gemma-2-2b \
        --layer 13
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Iterable
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sae_bench.common.transcoder_adapter import (  # noqa: E402
    TranscoderAdapter,
    load_google_transcoder_adapter,
    load_transcoder_adapter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_NAME_TO_DTYPE = {
    "gemma-2-2b": "bfloat16",
    "gemma-2-2b-it": "bfloat16",
    "gemma-2-9b": "bfloat16",
}


def str_to_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[s]


def setup_environment() -> str:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda"
    return "cpu"


def load_run_cfg(config_path: str | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tristate_flag(value: str, default: bool) -> bool:
    if value == "auto":
        return default
    return value == "true"


def resolve_token_settings(
    run_cfg: dict[str, Any] | None,
    *,
    prepend_bos: str,
    exclude_special_tokens_from_reconstruction: str,
    exclude_special_tokens_from_l0: str,
    match_training_tokens: bool,
) -> tuple[bool, bool, bool]:
    default_prepend_bos = True
    default_reconstruction_mask = False
    default_l0_mask = False

    if run_cfg is not None:
        default_prepend_bos = bool(run_cfg.get("prepend_bos", True))
        if match_training_tokens:
            match_mask = bool(run_cfg.get("exclude_special_tokens", True))
            default_reconstruction_mask = match_mask
            default_l0_mask = match_mask

    return (
        resolve_tristate_flag(prepend_bos, default_prepend_bos),
        resolve_tristate_flag(
            exclude_special_tokens_from_reconstruction, default_reconstruction_mask
        ),
        resolve_tristate_flag(exclude_special_tokens_from_l0, default_l0_mask),
    )


def iter_token_batches(
    model: HookedTransformer,
    dataset_name: str,
    batch_size: int,
    seq_len: int,
    text_column: str = "text",
    prepend_bos: bool = True,
) -> Iterable[torch.Tensor]:
    """Yield token batches from a streaming HF dataset."""
    dataset = iter(load_dataset(dataset_name, split="train", streaming=True))
    while True:
        texts = [str(next(dataset)[text_column]) for _ in range(batch_size)]
        tokens = model.to_tokens(
            texts, truncate=True, prepend_bos=prepend_bos, move_to_device=True
        )
        yield tokens[:, :seq_len]


def get_valid_token_mask(
    tokens: torch.Tensor,
    model: HookedTransformer,
) -> torch.Tensor:
    """Return a mask excluding BOS, EOS, and PAD positions."""
    mask = torch.ones_like(tokens, dtype=torch.bool)
    tokenizer = model.tokenizer
    for token_id in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if token_id is not None:
            mask &= tokens != int(token_id)
    return mask


# ---------------------------------------------------------------------------
# Core FVU computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_fvu(
    model: HookedTransformer,
    transcoder: TranscoderAdapter,
    n_batches: int = 50,
    n_reconstruction_batches: int | None = None,
    n_sparsity_batches: int | None = None,
    batch_size: int = 16,
    seq_len: int = 128,
    dataset_name: str = "monology/pile-uncopyrighted",
    text_column: str = "text",
    prepend_bos: bool = True,
    exclude_special_tokens_from_reconstruction: bool = False,
    exclude_special_tokens_from_l0: bool = False,
) -> dict:
    """Compute FVU and L0 for a transcoder on streaming text data."""
    dtype = transcoder.dtype
    source_hook = transcoder.cfg.source_hook_point
    target_hook = transcoder.cfg.target_hook_point
    layer = transcoder.cfg.hook_layer
    d_model = transcoder.W_dec.shape[-1]

    reconstruction_batches = (
        n_reconstruction_batches if n_reconstruction_batches is not None else n_batches
    )
    sparsity_batches = (
        n_sparsity_batches if n_sparsity_batches is not None else n_batches
    )

    # Running statistics for numerically stable FVU
    residual_sq = 0.0
    sparse_residual_sq = 0.0
    skip_residual_sq = 0.0
    residual_sum_x = torch.zeros(d_model, dtype=torch.float64)
    residual_sum_x2 = torch.zeros(d_model, dtype=torch.float64)
    sum_x = torch.zeros(d_model, dtype=torch.float64)
    sum_x2 = torch.zeros(d_model, dtype=torch.float64)
    n_tokens = 0
    l0_sum = 0.0
    l0_tokens = 0

    recon_batches = iter_token_batches(
        model=model,
        dataset_name=dataset_name,
        batch_size=batch_size,
        seq_len=seq_len,
        text_column=text_column,
        prepend_bos=prepend_bos,
    )

    for _ in tqdm(range(reconstruction_batches), desc="Computing FVU"):
        tokens = next(recon_batches)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[source_hook, target_hook],
                stop_at_layer=layer + 1,
            )

            source = cache[source_hook]
            target = cache[target_hook]

            source_flat = source.reshape(-1, source.shape[-1])
            target_flat = target.reshape(-1, target.shape[-1])

            if exclude_special_tokens_from_reconstruction:
                valid_mask = get_valid_token_mask(tokens, model).reshape(-1)
                source_flat = source_flat[valid_mask]
                target_flat = target_flat[valid_mask]

            source_flat = source_flat.to(dtype)
            target_flat = target_flat.to(dtype)

            features = transcoder.encode(source_flat)
            sparse_recon = transcoder.decode_sparse(features)
            if transcoder.has_skip:
                skip_recon = transcoder.skip(source_flat)
                recon = skip_recon + sparse_recon
            else:
                skip_recon = None
                recon = sparse_recon

            # Update running stats
            target_f = target_flat.float()
            sparse_f = sparse_recon.float()
            recon_f = recon.float()
            residual = target_f - recon_f

            residual_sq += residual.pow(2).sum().item()
            sum_x += target_f.sum(dim=0, dtype=torch.float64).cpu()
            sum_x2 += target_f.pow(2).sum(dim=0, dtype=torch.float64).cpu()
            n_tokens += target_f.shape[0]

            if skip_recon is not None:
                skip_f = skip_recon.float()
                residual_target_f = target_f - skip_f
                sparse_residual = residual_target_f - sparse_f
                sparse_residual_sq += sparse_residual.pow(2).sum().item()
                skip_residual_sq += residual_target_f.pow(2).sum().item()
                residual_sum_x += residual_target_f.sum(dim=0, dtype=torch.float64).cpu()
                residual_sum_x2 += residual_target_f.pow(2).sum(
                    dim=0, dtype=torch.float64
                ).cpu()

    sparsity_batches_iter = iter_token_batches(
        model=model,
        dataset_name=dataset_name,
        batch_size=batch_size,
        seq_len=seq_len,
        text_column=text_column,
        prepend_bos=prepend_bos,
    )
    for _ in tqdm(range(sparsity_batches), desc="Computing L0"):
        tokens = next(sparsity_batches_iter)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[source_hook],
                stop_at_layer=layer + 1,
            )
            source = cache[source_hook]
            source_flat = source.reshape(-1, source.shape[-1])
            if exclude_special_tokens_from_l0:
                valid_mask = get_valid_token_mask(tokens, model).reshape(-1)
                source_flat = source_flat[valid_mask]
            source_flat = source_flat.to(dtype)
            features = transcoder.encode(source_flat)

            l0_sum += (features > 0).float().sum().item()
            l0_tokens += features.shape[0]

    # Finalize
    if n_tokens == 0:
        raise ValueError("No reconstruction tokens were evaluated; check batch settings.")

    mean_per_dim = sum_x / n_tokens
    var_per_dim = sum_x2 / n_tokens - mean_per_dim**2
    mean_var = var_per_dim.mean().item()
    mse = residual_sq / (n_tokens * d_model)
    fvu = mse / (mean_var + 1e-8)
    l0 = l0_sum / max(1, l0_tokens)

    metrics = {
        "fvu": float(fvu),
        "mse": float(mse),
        "mean_variance": float(mean_var),
        "l0": float(l0),
        "n_tokens": int(n_tokens),
        "n_tokens_l0": int(l0_tokens),
        "prepend_bos": bool(prepend_bos),
        "exclude_special_tokens_from_reconstruction": bool(
            exclude_special_tokens_from_reconstruction
        ),
        "exclude_special_tokens_from_l0": bool(exclude_special_tokens_from_l0),
    }
    if transcoder.has_skip:
        residual_mean_per_dim = residual_sum_x / n_tokens
        residual_var_per_dim = residual_sum_x2 / n_tokens - residual_mean_per_dim**2
        residual_mean_var = residual_var_per_dim.mean().item()
        sparse_residual_mse = sparse_residual_sq / (n_tokens * d_model)
        skip_mse = skip_residual_sq / (n_tokens * d_model)
        metrics.update(
            {
                "fvu_y": float(fvu),
                "fvu_r": float(sparse_residual_mse / (residual_mean_var + 1e-8)),
                "fvu_skip": float(skip_mse / (mean_var + 1e-8)),
                "mse_r": float(sparse_residual_mse),
                "mse_skip": float(skip_mse),
                "residual_mean_variance": float(residual_mean_var),
            }
        )
    return metrics


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------
def run_eval(
    selected_transcoders: list[tuple[str, TranscoderAdapter]],
    model: HookedTransformer,
    output_path: str,
    n_batches: int = 50,
    n_reconstruction_batches: int | None = None,
    n_sparsity_batches: int | None = None,
    batch_size: int = 16,
    seq_len: int = 128,
    dataset_name: str = "monology/pile-uncopyrighted",
    text_column: str = "text",
    prepend_bos: bool = True,
    exclude_special_tokens_from_reconstruction: bool = False,
    exclude_special_tokens_from_l0: bool = False,
    force_rerun: bool = False,
) -> dict:
    os.makedirs(output_path, exist_ok=True)
    results = {}

    for transcoder_name, transcoder in selected_transcoders:
        result_filename = (
            f"{transcoder_name}_fvu_l{transcoder.cfg.hook_layer}.json"
        )
        result_path = os.path.join(output_path, result_filename)

        if os.path.exists(result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} (results exist)")
            with open(result_path) as f:
                results[transcoder_name] = json.load(f)
            continue

        print(f"\nEvaluating {transcoder_name}...")
        metrics = evaluate_fvu(
            model=model,
            transcoder=transcoder,
            n_batches=n_batches,
            n_reconstruction_batches=n_reconstruction_batches,
            n_sparsity_batches=n_sparsity_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            dataset_name=dataset_name,
            text_column=text_column,
            prepend_bos=prepend_bos,
            exclude_special_tokens_from_reconstruction=exclude_special_tokens_from_reconstruction,
            exclude_special_tokens_from_l0=exclude_special_tokens_from_l0,
        )

        output = {
            "transcoder_name": transcoder_name,
            "transcoder_cfg": transcoder.cfg.to_dict(),
            "metrics": metrics,
        }

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"  FVU: {metrics['fvu']:.4f}, L0: {metrics['l0']:.1f}, MSE: {metrics['mse']:.6f}")
        if "fvu_r" in metrics:
            print(
                f"  Skip-only FVU: {metrics['fvu_skip']:.4f}, "
                f"residual FVU: {metrics['fvu_r']:.4f}"
            )
        results[transcoder_name] = output

        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def arg_parser():
    parser = argparse.ArgumentParser(description="Run FVU evaluation for transcoders")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Path to transcoder training config (json).",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path to transcoder checkpoint (.pt).",
    )
    parser.add_argument(
        "--google_params_path", type=str, default=None,
        help="Path to Google Gemma Scope params.npz.",
    )
    parser.add_argument("--layer", type=int, default=None, help="Hook layer override.")
    parser.add_argument(
        "--n_batches", type=int, default=50,
        help="Legacy fallback batch count used when split reconstruction/L0 counts are not provided.",
    )
    parser.add_argument(
        "--n_reconstruction_batches", type=int, default=None,
        help="Number of batches used for reconstruction/FVU (FF-KV core uses 200).",
    )
    parser.add_argument(
        "--n_sparsity_batches", type=int, default=None,
        help="Number of batches used for L0 estimation. Defaults to --n_batches.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length.")
    parser.add_argument(
        "--dataset", type=str, default="monology/pile-uncopyrighted",
        help="HuggingFace dataset for evaluation tokens (default: The Pile).",
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Dataset column used for text.",
    )
    parser.add_argument(
        "--exclude_special_tokens_from_reconstruction",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Whether to exclude BOS/EOS/PAD from reconstruction stats. "
            "Use 'auto' to follow --match_training_tokens or fall back to legacy behavior."
        ),
    )
    parser.add_argument(
        "--exclude_special_tokens_from_l0",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Whether to exclude BOS/EOS/PAD from L0 counting. "
            "Use 'auto' to follow --match_training_tokens or fall back to legacy behavior."
        ),
    )
    parser.add_argument(
        "--prepend_bos",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Whether to prepend BOS during tokenization. "
            "Use 'auto' to read config.json when available, else default to legacy True."
        ),
    )
    parser.add_argument(
        "--match_training_tokens",
        action="store_true",
        help=(
            "When --config_path is provided, match prepend_bos and special-token masking "
            "to the training config for both reconstruction and L0."
        ),
    )
    parser.add_argument(
        "--llm_dtype", type=str, default=None,
        choices=[None, "float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--output_folder", type=str, default="results/evals/fvu",
        help="Output folder.",
    )
    parser.add_argument("--force_rerun", action="store_true")
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = setup_environment()
    run_cfg = load_run_cfg(args.config_path)

    (
        prepend_bos,
        exclude_special_tokens_from_reconstruction,
        exclude_special_tokens_from_l0,
    ) = resolve_token_settings(
        run_cfg,
        prepend_bos=args.prepend_bos,
        exclude_special_tokens_from_reconstruction=args.exclude_special_tokens_from_reconstruction,
        exclude_special_tokens_from_l0=args.exclude_special_tokens_from_l0,
        match_training_tokens=args.match_training_tokens,
    )

    llm_dtype_str = args.llm_dtype or LLM_NAME_TO_DTYPE.get(args.model_name, "bfloat16")
    llm_dtype = str_to_dtype(llm_dtype_str)

    start_time = time.time()

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name, device=device, dtype=llm_dtype
    )

    selected_transcoders: list[tuple[str, TranscoderAdapter]] = []
    if args.google_params_path:
        if args.layer is None:
            raise ValueError("Please provide --layer when using --google_params_path.")
        name, adapter = load_google_transcoder_adapter(
            args.google_params_path, layer=args.layer, device=device, dtype=llm_dtype
        )
        selected_transcoders.append((name, adapter))
    else:
        if args.config_path is None:
            raise ValueError("Provide --config_path or --google_params_path.")
        name, adapter = load_transcoder_adapter(
            args.config_path, args.checkpoint_path, device=device, dtype=llm_dtype
        )
        selected_transcoders.append((name, adapter))

    if args.layer is not None:
        for _, a in selected_transcoders:
            a.cfg.hook_layer = args.layer

    for name, adapter in selected_transcoders:
        adapter.to(device=device, dtype=llm_dtype)

    print(f"Selected transcoders: {[n for n, _ in selected_transcoders]}")
    print(
        "Token settings: "
        f"prepend_bos={prepend_bos}, "
        f"exclude_special_tokens_from_reconstruction={exclude_special_tokens_from_reconstruction}, "
        f"exclude_special_tokens_from_l0={exclude_special_tokens_from_l0}",
    )

    results = run_eval(
        selected_transcoders,
        model,
        args.output_folder,
        n_batches=args.n_batches,
        n_reconstruction_batches=args.n_reconstruction_batches,
        n_sparsity_batches=args.n_sparsity_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dataset_name=args.dataset,
        text_column=args.text_column,
        prepend_bos=prepend_bos,
        exclude_special_tokens_from_reconstruction=exclude_special_tokens_from_reconstruction,
        exclude_special_tokens_from_l0=exclude_special_tokens_from_l0,
        force_rerun=args.force_rerun,
    )

    print(f"\nFinished in {time.time() - start_time:.1f}s")

    # Some environments hit a CPython finalization crash after HF/datasets teardown.
    # Allow callers to opt into an immediate clean exit after results are written.
    if os.environ.get("FVU_FORCE_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
