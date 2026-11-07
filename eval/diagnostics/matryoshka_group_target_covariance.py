"""Measure target-covariance structure by Matryoshka group.

For each trained Matryoshka transcoder, this diagnostic estimates

    C_i = || E[z_i (y - mean(y))] ||_2^2 / E[z_i^2]

on the cached first-letter absorption activations.  It also reports source-space
covariance, encoder/probe cosine, decoder/probe cosine, firing statistics, and
the group membership of k=1 first-letter sparse-probing features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
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


def parse_layers(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def hook_id_for_layer(layer: int) -> str:
    return f"blocks_{layer}_hook_mlp_out"


def valid_eval_indices(probe_dir: Path, split: str, expected_len: int) -> np.ndarray:
    """Match the absorption loader's token filter for probe evaluation."""
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
    max_idx = max(indices)
    if max_idx >= expected_len:
        raise ValueError(
            f"{df_path} contains index {max_idx}, but cached split has {expected_len} rows"
        )
    return np.asarray(indices, dtype=np.int64)


def group_for_feature(feature_idx: int, group_indices: list[int]) -> int:
    for group_idx in range(len(group_indices) - 1):
        if group_indices[group_idx] <= feature_idx < group_indices[group_idx + 1]:
            return group_idx
    raise ValueError(f"Feature {feature_idx} is outside Matryoshka group bounds")


def finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def finite_median(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def load_probe_weights(path: Path, device: torch.device) -> torch.Tensor | None:
    if not path.exists():
        return None
    probe = torch.load(path, map_location=device, weights_only=False)
    return probe.fc.weight.detach().to(device=device, dtype=torch.float32)


@torch.inference_mode()
def max_abs_probe_cosine(
    feature_vectors: torch.Tensor,
    probe_weights: torch.Tensor | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return max absolute cosine, signed cosine at max, and argmax letter index."""
    n_features = feature_vectors.shape[0]
    if probe_weights is None:
        return (
            np.full(n_features, np.nan, dtype=np.float32),
            np.full(n_features, np.nan, dtype=np.float32),
            np.full(n_features, -1, dtype=np.int64),
        )

    probe_normed = probe_weights / probe_weights.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    max_abs_chunks = []
    signed_chunks = []
    argmax_chunks = []
    for start in range(0, n_features, batch_size):
        vectors = feature_vectors[start : start + batch_size]
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos = vectors @ probe_normed.T
        abs_cos, argmax = cos.abs().max(dim=-1)
        signed = cos.gather(1, argmax[:, None]).squeeze(1)
        max_abs_chunks.append(abs_cos.cpu())
        signed_chunks.append(signed.cpu())
        argmax_chunks.append(argmax.cpu())

    return (
        torch.cat(max_abs_chunks).numpy(),
        torch.cat(signed_chunks).numpy(),
        torch.cat(argmax_chunks).numpy(),
    )


@torch.inference_mode()
def compute_covariance_metrics(
    adapter: Any,
    x_source: np.ndarray,
    x_target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    n_tokens = x_source.shape[0]
    d_sae = int(adapter.cfg.d_sae)
    d_source = x_source.shape[1]
    d_target = x_target.shape[1]

    source_mean = torch.from_numpy(x_source).float().mean(dim=0).to(device)
    target_mean = torch.from_numpy(x_target).float().mean(dim=0).to(device)

    source_cross = torch.zeros((d_sae, d_source), device=device, dtype=torch.float32)
    target_cross = torch.zeros((d_sae, d_target), device=device, dtype=torch.float32)
    z_sum = torch.zeros(d_sae, device=device, dtype=torch.float32)
    z2_sum = torch.zeros(d_sae, device=device, dtype=torch.float32)
    active_count = torch.zeros(d_sae, device=device, dtype=torch.float32)

    for start in range(0, n_tokens, batch_size):
        source_batch = torch.from_numpy(x_source[start : start + batch_size]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        target_batch = torch.from_numpy(x_target[start : start + batch_size]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        z = adapter.encode(source_batch, max_tokens=batch_size).float()

        source_centered = source_batch - source_mean
        target_centered = target_batch - target_mean

        source_cross.add_(z.T @ source_centered)
        target_cross.add_(z.T @ target_centered)
        z_sum.add_(z.sum(dim=0))
        z2_sum.add_(z.pow(2).sum(dim=0))
        active_count.add_((z > 0).sum(dim=0).float())

    mean_z2 = z2_sum / max(1, n_tokens)
    denominator = mean_z2.clamp_min(1e-12)
    source_expectation = source_cross / max(1, n_tokens)
    target_expectation = target_cross / max(1, n_tokens)

    source_covariance = source_expectation.pow(2).sum(dim=-1) / denominator
    target_covariance = target_expectation.pow(2).sum(dim=-1) / denominator

    return {
        "source_covariance": source_covariance.cpu().numpy(),
        "target_covariance": target_covariance.cpu().numpy(),
        "mean_activation": (z_sum / max(1, n_tokens)).cpu().numpy(),
        "mean_activation_sq": mean_z2.cpu().numpy(),
        "firing_rate": (active_count / max(1, n_tokens)).cpu().numpy(),
    }


def load_k1_sparse_roles(path: Path, group_indices: list[int]) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "available": False,
            "best_feature_counts": Counter(),
            "best_feature_rows": [],
            "split_feature_counts": Counter(),
        }

    rows = json.loads(path.read_text())
    best_feature_counts: Counter[int] = Counter()
    split_feature_counts: Counter[int] = Counter()
    best_feature_rows = []

    for row in rows:
        feats = row.get("sparse_sae_k_1_feats") or []
        if feats:
            feature = int(feats[0])
            group = group_for_feature(feature, group_indices)
            best_feature_counts[group] += 1
            best_feature_rows.append(
                {
                    "letter": row.get("letter"),
                    "feature": feature,
                    "group": group,
                    "f1_sparse_sae_1": float(row.get("f1_sparse_sae_1", float("nan"))),
                    "cos_probe_sae_enc_k_1": float(
                        (row.get("cos_probe_sae_enc_k_1") or [float("nan")])[0]
                    ),
                    "cos_probe_sae_dec_k_1": float(
                        (row.get("cos_probe_sae_dec_k_1") or [float("nan")])[0]
                    ),
                }
            )
        for feature in row.get("split_feats") or []:
            split_feature_counts[group_for_feature(int(feature), group_indices)] += 1

    return {
        "path": str(path),
        "available": True,
        "best_feature_counts": best_feature_counts,
        "best_feature_rows": best_feature_rows,
        "split_feature_counts": split_feature_counts,
    }


def summarize_groups(
    layer: int,
    group_sizes: list[int],
    feature_rows: list[dict[str, Any]],
    sparse_roles: dict[str, Any],
) -> list[dict[str, Any]]:
    total_features = sum(group_sizes)
    total_target_cov = sum(float(row["target_covariance"]) for row in feature_rows)
    total_source_cov = sum(float(row["source_covariance"]) for row in feature_rows)
    rows_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        rows_by_group[int(row["group"])].append(row)

    best_rows_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sparse_roles["best_feature_rows"]:
        best_rows_by_group[int(row["group"])].append(row)

    summaries = []
    start = 0
    for group_idx, group_size in enumerate(group_sizes):
        end = start + group_size
        rows = rows_by_group[group_idx]
        target_values = [float(row["target_covariance"]) for row in rows]
        source_values = [float(row["source_covariance"]) for row in rows]
        target_probe_values = [float(row["target_probe_max_abs_cosine"]) for row in rows]
        source_probe_values = [float(row["source_probe_max_abs_cosine"]) for row in rows]
        firing_values = [float(row["firing_rate"]) for row in rows]
        best_rows = best_rows_by_group[group_idx]

        group_target_sum = float(np.sum(target_values))
        group_source_sum = float(np.sum(source_values))
        dict_share = group_size / total_features
        target_share = group_target_sum / total_target_cov if total_target_cov > 0 else float("nan")
        source_share = group_source_sum / total_source_cov if total_source_cov > 0 else float("nan")
        summaries.append(
            {
                "layer": layer,
                "group": group_idx,
                "start_feature": start,
                "end_feature_exclusive": end,
                "group_size": group_size,
                "dict_share": dict_share,
                "mean_target_covariance": finite_mean(target_values),
                "median_target_covariance": finite_median(target_values),
                "sum_target_covariance": group_target_sum,
                "target_covariance_share": target_share,
                "target_covariance_share_over_dict_share": target_share / dict_share
                if np.isfinite(target_share) and dict_share > 0
                else float("nan"),
                "mean_source_covariance": finite_mean(source_values),
                "median_source_covariance": finite_median(source_values),
                "sum_source_covariance": group_source_sum,
                "source_covariance_share": source_share,
                "source_covariance_share_over_dict_share": source_share / dict_share
                if np.isfinite(source_share) and dict_share > 0
                else float("nan"),
                "mean_target_over_source_covariance": finite_mean(
                    [
                        float(row["target_over_source_covariance"])
                        for row in rows
                        if np.isfinite(float(row["target_over_source_covariance"]))
                    ]
                ),
                "mean_source_probe_max_abs_cosine": finite_mean(source_probe_values),
                "median_source_probe_max_abs_cosine": finite_median(source_probe_values),
                "mean_target_probe_max_abs_cosine": finite_mean(target_probe_values),
                "median_target_probe_max_abs_cosine": finite_median(target_probe_values),
                "mean_firing_rate": finite_mean(firing_values),
                "median_firing_rate": finite_median(firing_values),
                "k1_best_feature_count": int(sparse_roles["best_feature_counts"][group_idx]),
                "k1_best_feature_share": int(sparse_roles["best_feature_counts"][group_idx])
                / max(1, len(LETTERS)),
                "mean_k1_best_feature_f1": finite_mean(
                    [float(row["f1_sparse_sae_1"]) for row in best_rows]
                ),
                "mean_k1_best_feature_dec_cos": finite_mean(
                    [abs(float(row["cos_probe_sae_dec_k_1"])) for row in best_rows]
                ),
                "split_feature_count": int(sparse_roles["split_feature_counts"][group_idx]),
            }
        )
        start = end

    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def summarize_across_layers(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        rows_by_group[int(row["group"])].append(row)

    summaries = []
    for group_idx in sorted(rows_by_group):
        rows = rows_by_group[group_idx]
        summaries.append(
            {
                "group": group_idx,
                "num_layers": len(rows),
                "dict_share": finite_mean([float(row["dict_share"]) for row in rows]),
                "mean_target_covariance_share": finite_mean(
                    [float(row["target_covariance_share"]) for row in rows]
                ),
                "mean_target_covariance_share_over_dict_share": finite_mean(
                    [
                        float(row["target_covariance_share_over_dict_share"])
                        for row in rows
                    ]
                ),
                "mean_source_covariance_share": finite_mean(
                    [float(row["source_covariance_share"]) for row in rows]
                ),
                "mean_source_covariance_share_over_dict_share": finite_mean(
                    [
                        float(row["source_covariance_share_over_dict_share"])
                        for row in rows
                    ]
                ),
                "mean_target_probe_max_abs_cosine": finite_mean(
                    [float(row["mean_target_probe_max_abs_cosine"]) for row in rows]
                ),
                "mean_source_probe_max_abs_cosine": finite_mean(
                    [float(row["mean_source_probe_max_abs_cosine"]) for row in rows]
                ),
                "mean_firing_rate": finite_mean(
                    [float(row["mean_firing_rate"]) for row in rows]
                ),
                "total_k1_best_feature_count": int(
                    sum(int(row["k1_best_feature_count"]) for row in rows)
                ),
                "mean_k1_best_feature_count_per_layer": finite_mean(
                    [float(row["k1_best_feature_count"]) for row in rows]
                ),
                "mean_k1_best_feature_f1": finite_mean(
                    [float(row["mean_k1_best_feature_f1"]) for row in rows]
                ),
                "mean_k1_best_feature_dec_cos": finite_mean(
                    [float(row["mean_k1_best_feature_dec_cos"]) for row in rows]
                ),
                "total_split_feature_count": int(
                    sum(int(row["split_feature_count"]) for row in rows)
                ),
                "mean_split_feature_count_per_layer": finite_mean(
                    [float(row["split_feature_count"]) for row in rows]
                ),
            }
        )
    return summaries


def process_layer(args: argparse.Namespace, layer: int, device: torch.device) -> dict[str, Any]:
    hook_id = hook_id_for_layer(layer)
    probe_dir = args.probe_cache_root / args.model_name / hook_id
    data_path = probe_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing absorption cache: {data_path}")

    config_path = args.training_root / str(layer) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Matryoshka config: {config_path}")

    print(f"[layer {layer}] loading Matryoshka checkpoint from {config_path}", flush=True)
    _, adapter = load_transcoder_adapter(
        config_path=config_path,
        checkpoint_path=None,
        device=device,
        dtype=torch.float32,
    )
    group_sizes = [int(size) for size in adapter.transcoder.group_sizes]
    group_indices = [int(idx) for idx in adapter.transcoder.group_indices]

    print(f"[layer {layer}] loading cached source/target activations", flush=True)
    data = np.load(data_path)
    split = args.split
    x_source = data[f"X_{split}_source"]
    x_target = data[f"X_{split}_target"]
    valid_indices = valid_eval_indices(probe_dir, split, x_source.shape[0])
    x_source = x_source[valid_indices]
    x_target = x_target[valid_indices]

    print(
        f"[layer {layer}] estimating covariance over {x_source.shape[0]} filtered tokens",
        flush=True,
    )
    cov_metrics = compute_covariance_metrics(
        adapter=adapter,
        x_source=x_source,
        x_target=x_target,
        device=device,
        batch_size=args.batch_size,
    )

    target_probe_path = probe_dir / "probe.pth"
    source_probe_path = (
        args.source_probe_root / args.model_name / hook_id / "source_probe.pth"
    )
    target_probe_weights = load_probe_weights(target_probe_path, device)
    source_probe_weights = load_probe_weights(source_probe_path, device)

    w_enc = adapter.transcoder.W_enc.detach().to(device=device, dtype=torch.float32).T
    w_dec = adapter.transcoder.W_dec.detach().to(device=device, dtype=torch.float32)

    print(f"[layer {layer}] computing probe cosines", flush=True)
    source_abs_cos, source_signed_cos, source_argmax = max_abs_probe_cosine(
        w_enc, source_probe_weights, args.cosine_batch_size
    )
    target_abs_cos, target_signed_cos, target_argmax = max_abs_probe_cosine(
        w_dec, target_probe_weights, args.cosine_batch_size
    )

    k_sparse_path = args.k_sparse_root / "matryoshka" / str(layer) / "result.json"
    sparse_roles = load_k1_sparse_roles(k_sparse_path, group_indices)

    feature_rows = []
    for feature_idx in range(int(adapter.cfg.d_sae)):
        group = group_for_feature(feature_idx, group_indices)
        source_cov = float(cov_metrics["source_covariance"][feature_idx])
        target_cov = float(cov_metrics["target_covariance"][feature_idx])
        if source_cov > 0 and np.isfinite(source_cov):
            target_over_source = target_cov / source_cov
        else:
            target_over_source = float("nan")
        feature_rows.append(
            {
                "layer": layer,
                "feature": feature_idx,
                "group": group,
                "target_covariance": target_cov,
                "source_covariance": source_cov,
                "target_over_source_covariance": target_over_source,
                "mean_activation": float(cov_metrics["mean_activation"][feature_idx]),
                "mean_activation_sq": float(cov_metrics["mean_activation_sq"][feature_idx]),
                "firing_rate": float(cov_metrics["firing_rate"][feature_idx]),
                "source_probe_max_abs_cosine": float(source_abs_cos[feature_idx]),
                "source_probe_signed_cosine": float(source_signed_cos[feature_idx]),
                "source_probe_argmax_letter": LETTERS[int(source_argmax[feature_idx])]
                if int(source_argmax[feature_idx]) >= 0
                else "",
                "target_probe_max_abs_cosine": float(target_abs_cos[feature_idx]),
                "target_probe_signed_cosine": float(target_signed_cos[feature_idx]),
                "target_probe_argmax_letter": LETTERS[int(target_argmax[feature_idx])]
                if int(target_argmax[feature_idx]) >= 0
                else "",
            }
        )

    group_summary = summarize_groups(layer, group_sizes, feature_rows, sparse_roles)
    top_features = sorted(
        feature_rows, key=lambda row: float(row["target_covariance"]), reverse=True
    )[: args.top_n_features]

    layer_dir = args.output_dir / f"layer_{layer}"
    write_csv(layer_dir / "feature_metrics.csv", feature_rows)
    write_csv(layer_dir / "group_summary.csv", group_summary)
    write_csv(layer_dir / "top_target_covariance_features.csv", top_features)
    write_csv(layer_dir / "k1_best_letter_features.csv", sparse_roles["best_feature_rows"])
    save_json(
        layer_dir / "summary.json",
        {
            "layer": layer,
            "config_path": str(config_path),
            "data_path": str(data_path),
            "target_probe_path": str(target_probe_path),
            "source_probe_path": str(source_probe_path),
            "k_sparse_path": str(k_sparse_path),
            "k_sparse_available": bool(sparse_roles["available"]),
            "num_filtered_tokens": int(x_source.shape[0]),
            "group_sizes": group_sizes,
            "group_summary": group_summary,
        },
    )

    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "layer": layer,
        "num_filtered_tokens": int(x_source.shape[0]),
        "group_sizes": group_sizes,
        "group_summary": group_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure target-covariance and probe-alignment by Matryoshka group."
    )
    parser.add_argument("--layers", default="0,6,12,18,24")
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("results/default2/training/matryoshka"),
    )
    parser.add_argument(
        "--probe-cache-root",
        type=Path,
        default=Path("results/evals/absorption/cache/probes"),
    )
    parser.add_argument(
        "--source-probe-root",
        type=Path,
        default=Path("results/diagnostics/source_target_letter_probe/source_probes"),
    )
    parser.add_argument(
        "--k-sparse-root",
        type=Path,
        default=Path("results/default2/k_sparse_probing"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/matryoshka_group_target_covariance"),
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cosine-batch-size", type=int, default=4096)
    parser.add_argument("--top-n-features", type=int, default=200)
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_group_rows = []
    layer_summaries = []

    for layer in parse_layers(args.layers):
        layer_result = process_layer(args, layer, device)
        layer_summaries.append(layer_result)
        all_group_rows.extend(layer_result["group_summary"])

    write_csv(args.output_dir / "group_summary.csv", all_group_rows)
    aggregate_rows = summarize_across_layers(all_group_rows)
    write_csv(args.output_dir / "group_summary_by_group.csv", aggregate_rows)
    save_json(
        args.output_dir / "summary.json",
        {
            "layers": parse_layers(args.layers),
            "device": str(device),
            "split": args.split,
            "layer_summaries": layer_summaries,
            "group_summary_by_group": aggregate_rows,
        },
    )
    print(f"Wrote group summaries to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
