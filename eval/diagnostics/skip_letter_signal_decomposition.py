#!/usr/bin/env python3
"""Decompose first-letter target-probe signal into skip and sparse terms."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sae_bench.absorption.vocab import LETTERS  # noqa: E402
from sae_bench.common.transcoder_adapter import load_transcoder_adapter  # noqa: E402


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def hook_id_for_layer(layer: int) -> str:
    return f"blocks_{layer}_hook_mlp_out"


def valid_eval_indices(probe_dir: Path, split: str, expected_len: int) -> np.ndarray:
    df_path = probe_dir / f"{split}_df.csv"
    if not df_path.exists():
        return np.arange(expected_len)

    indices = []
    with df_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            token = row.get("token", "")
            if not isinstance(token, str) or re.match(r"[\d<>]", token):
                continue
            indices.append(idx)
    if not indices:
        raise ValueError(f"No valid eval tokens found in {df_path}")
    if max(indices) >= expected_len:
        raise ValueError(f"{df_path} has indices outside cached activation data.")
    return np.asarray(indices, dtype=np.int64)


def load_probe_weights(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing target probe at {path}. Run absorption eval first.")
    probe = torch.load(path, map_location="cpu", weights_only=False)
    weights = probe.weights.detach().float()
    return weights / weights.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def compute_components(
    adapter: Any,
    x_source: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    skip_chunks = []
    sparse_chunks = []
    for start in range(0, x_source.shape[0], batch_size):
        source_batch = torch.from_numpy(x_source[start : start + batch_size]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        z = adapter.encode(source_batch, max_tokens=batch_size)
        skip = adapter.transcoder.compute_skip(source_batch).float()
        sparse = adapter.transcoder.decode_residual(z).float()
        skip_chunks.append(skip.cpu())
        sparse_chunks.append(sparse.cpu())
    return torch.cat(skip_chunks, dim=0), torch.cat(sparse_chunks, dim=0)


def direct_r2(true: torch.Tensor, pred: torch.Tensor) -> float:
    denom = true.var(unbiased=False).clamp_min(1e-12)
    return float((1.0 - (true - pred).pow(2).mean() / denom).item())


def corr_r2(true: torch.Tensor, pred: torch.Tensor) -> float:
    true_c = true - true.mean()
    pred_c = pred - pred.mean()
    denom = true_c.norm() * pred_c.norm()
    if float(denom) <= 1e-12:
        return float("nan")
    corr = true_c.dot(pred_c) / denom
    return float((corr * corr).item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--probe-cache-root",
        type=Path,
        default=Path("results/evals/absorption/cache/probes"),
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/skip_letter_signal_decomposition"),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    name, adapter = load_transcoder_adapter(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=torch.float32,
    )
    if not bool(getattr(adapter.transcoder, "has_skip_connection", False)):
        raise ValueError("The selected checkpoint is not a skip transcoder.")

    layer = args.layer if args.layer is not None else int(adapter.cfg.hook_layer)
    probe_dir = args.probe_cache_root / args.model_name / hook_id_for_layer(layer)
    data_path = probe_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing absorption probe data at {data_path}.")

    data = np.load(data_path)
    x_source = data[f"X_{args.split}_source"]
    x_target = data[f"X_{args.split}_target"]
    valid_indices = valid_eval_indices(probe_dir, args.split, x_source.shape[0])
    x_source = x_source[valid_indices]
    x_target = x_target[valid_indices]

    probe_weights = load_probe_weights(probe_dir / "probe.pth")
    skip, sparse = compute_components(
        adapter=adapter,
        x_source=x_source,
        batch_size=args.batch_size,
        device=device,
    )
    target = torch.from_numpy(x_target).float()

    target_scores = (target - target.mean(dim=0, keepdim=True)) @ probe_weights.T
    skip_scores = (skip - skip.mean(dim=0, keepdim=True)) @ probe_weights.T
    sparse_scores = (sparse - sparse.mean(dim=0, keepdim=True)) @ probe_weights.T

    rows = []
    for idx, letter in enumerate(LETTERS):
        true = target_scores[:, idx]
        skip_score = skip_scores[:, idx]
        sparse_score = sparse_scores[:, idx]
        var_skip = skip_score.var(unbiased=False)
        var_sparse = sparse_score.var(unbiased=False)
        rows.append(
            {
                "letter": letter,
                "n_tokens": int(target.shape[0]),
                "var_true": float(true.var(unbiased=False).item()),
                "var_skip": float(var_skip.item()),
                "var_sparse": float(var_sparse.item()),
                "skip_share": float(
                    (var_skip / (var_skip + var_sparse).clamp_min(1e-12)).item()
                ),
                "direct_r2_skip": direct_r2(true, skip_score),
                "direct_r2_sparse": direct_r2(true, sparse_score),
                "direct_r2_skip_plus_sparse": direct_r2(true, skip_score + sparse_score),
                "corr_r2_skip": corr_r2(true, skip_score),
                "corr_r2_sparse": corr_r2(true, sparse_score),
                "corr_r2_skip_plus_sparse": corr_r2(true, skip_score + sparse_score),
            }
        )

    output_dir = args.output_dir / name
    write_csv(output_dir / "skip_share_by_letter.csv", rows)
    summary = {
        "name": name,
        "config_path": str(args.config_path),
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else "",
        "layer": int(layer),
        "split": args.split,
        "num_filtered_tokens": int(target.shape[0]),
        "mean_skip_share": finite_mean(rows, "skip_share"),
        "mean_direct_r2_skip": finite_mean(rows, "direct_r2_skip"),
        "mean_direct_r2_sparse": finite_mean(rows, "direct_r2_sparse"),
        "mean_direct_r2_skip_plus_sparse": finite_mean(rows, "direct_r2_skip_plus_sparse"),
        "mean_corr_r2_skip": finite_mean(rows, "corr_r2_skip"),
        "mean_corr_r2_sparse": finite_mean(rows, "corr_r2_sparse"),
        "mean_corr_r2_skip_plus_sparse": finite_mean(rows, "corr_r2_skip_plus_sparse"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Wrote skip-share diagnostics to {output_dir} "
        f"(mean skip share={summary['mean_skip_share']:.3f})"
    )


if __name__ == "__main__":
    main()
