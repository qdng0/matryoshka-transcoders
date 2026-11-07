"""Compare first-letter probe geometry in transcoder source and target spaces.

This diagnostic reuses the SAE Bench absorption probe cache. The cached target
probe is trained on MLP-output activations, while this script trains a matched
source-space probe on the cached MLP-input activations. It then reports per-letter
F1/AUC and normalized mean-shift strength in both spaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.functional import one_hot


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sae_bench.absorption.probing import LinearProbe, train_multi_probe  # noqa: E402
from sae_bench.absorption.vocab import LETTERS  # noqa: E402


def parse_layers(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def hook_id_for_layer(layer: int) -> str:
    return f"blocks_{layer}_hook_mlp_out"


def load_npz_array(np_data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in np_data.files:
        raise KeyError(f"{key!r} not present in cached data.npz")
    return np_data[key]


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


def load_target_probe(probe_dir: Path, device: torch.device) -> LinearProbe:
    probe_path = probe_dir / "probe.pth"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"Missing target probe at {probe_path}. Run absorption eval first."
        )
    probe = torch.load(probe_path, map_location=device, weights_only=False)
    return probe.to(device=device, dtype=torch.float32).eval()


def train_or_load_source_probe(
    source_probe_path: Path,
    x_train_source: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_epochs: int,
    lr: float,
    force: bool,
    show_progress: bool,
) -> LinearProbe:
    if source_probe_path.exists() and not force:
        probe = torch.load(source_probe_path, map_location=device, weights_only=False)
        return probe.to(device=device, dtype=torch.float32).eval()

    source_probe_path.parent.mkdir(parents=True, exist_ok=True)
    x_train = torch.from_numpy(x_train_source).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    y_train_one_hot = one_hot(y_train_tensor, num_classes=len(LETTERS))

    probe = train_multi_probe(
        x_train=x_train,
        y_train=y_train_one_hot,
        num_probes=len(LETTERS),
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=1e-4,
        show_progress=show_progress,
        verbose=False,
        device=device,
    )
    probe = probe.to(device=device, dtype=torch.float32).eval()
    torch.save(probe.cpu(), source_probe_path)
    return probe.to(device=device, dtype=torch.float32).eval()


@torch.inference_mode()
def probe_logits(
    probe: LinearProbe,
    x_eval: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, x_eval.shape[0], batch_size):
        batch = torch.from_numpy(x_eval[start : start + batch_size]).float().to(device)
        chunks.append(probe(batch).float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def per_letter_probe_metrics(logits: np.ndarray, labels: np.ndarray, prefix: str) -> list[dict]:
    rows = []
    for letter_idx, letter in enumerate(LETTERS):
        y_true = labels == letter_idx
        scores = logits[:, letter_idx]
        y_pred = scores > 0.0
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "letter": letter,
                f"{prefix}_auc": auc,
                f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                f"{prefix}_precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                f"{prefix}_recall": float(
                    recall_score(y_true, y_pred, zero_division=0)
                ),
            }
        )
    return rows


def normalized_mean_shifts(
    x_eval: np.ndarray,
    labels: np.ndarray,
    prefix: str,
) -> list[dict]:
    x = torch.from_numpy(x_eval).float()
    labels_t = torch.from_numpy(labels).long()
    mean = x.mean(dim=0)
    centered = x - mean
    rms_centered_norm = centered.pow(2).sum(dim=-1).mean().sqrt().item()
    denom = max(rms_centered_norm, 1e-12)

    rows = []
    for letter_idx, letter in enumerate(LETTERS):
        mask = labels_t == letter_idx
        if not bool(mask.any()):
            shift = float("nan")
            count = 0
        else:
            delta = x[mask].mean(dim=0) - mean
            shift = float(delta.norm().item() / denom)
            count = int(mask.sum().item())
        rows.append(
            {
                "letter": letter,
                f"{prefix}_mean_shift_norm": shift,
                f"{prefix}_positive_count": count,
            }
        )
    return rows


def merge_rows(*row_sets: list[dict]) -> list[dict]:
    merged = {letter: {"letter": letter} for letter in LETTERS}
    for rows in row_sets:
        for row in rows:
            merged[row["letter"]].update(row)
    return [merged[letter] for letter in LETTERS]


def add_ratios(rows: list[dict]) -> None:
    for row in rows:
        source_shift = float(row.get("source_mean_shift_norm", float("nan")))
        target_shift = float(row.get("target_mean_shift_norm", float("nan")))
        if np.isfinite(source_shift) and source_shift > 0:
            row["rho_target_over_source_shift"] = target_shift / source_shift
        else:
            row["rho_target_over_source_shift"] = float("nan")
        source_f1 = float(row.get("source_f1", float("nan")))
        target_f1 = float(row.get("target_f1", float("nan")))
        row["target_minus_source_f1"] = target_f1 - source_f1


def mean_finite(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def summarize_layer(layer: int, rows: list[dict], source_probe_path: Path, target_probe_path: Path) -> dict:
    return {
        "layer": layer,
        "mean_source_f1": mean_finite([float(row["source_f1"]) for row in rows]),
        "mean_target_f1": mean_finite([float(row["target_f1"]) for row in rows]),
        "mean_target_minus_source_f1": mean_finite(
            [float(row["target_minus_source_f1"]) for row in rows]
        ),
        "mean_source_auc": mean_finite([float(row["source_auc"]) for row in rows]),
        "mean_target_auc": mean_finite([float(row["target_auc"]) for row in rows]),
        "mean_source_shift_norm": mean_finite(
            [float(row["source_mean_shift_norm"]) for row in rows]
        ),
        "mean_target_shift_norm": mean_finite(
            [float(row["target_mean_shift_norm"]) for row in rows]
        ),
        "mean_rho_target_over_source_shift": mean_finite(
            [float(row["rho_target_over_source_shift"]) for row in rows]
        ),
        "num_letters_target_f1_gt_source_f1": int(
            sum(float(row["target_f1"]) > float(row["source_f1"]) for row in rows)
        ),
        "num_letters_rho_lt_1": int(
            sum(float(row["rho_target_over_source_shift"]) < 1.0 for row in rows)
        ),
        "source_probe_path": str(source_probe_path),
        "target_probe_path": str(target_probe_path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "letter" in fieldnames:
        fieldnames.remove("letter")
        fieldnames = ["letter", *fieldnames]
    if "layer" in fieldnames:
        fieldnames.remove("layer")
        fieldnames = ["layer", *fieldnames]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_layer(args: argparse.Namespace, layer: int, device: torch.device) -> dict:
    hook_id = hook_id_for_layer(layer)
    probe_dir = Path(args.probe_cache_root) / args.model_name / hook_id
    data_path = probe_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing probe cache at {data_path}")

    output_dir = Path(args.output_dir)
    source_probe_path = (
        output_dir / "source_probes" / args.model_name / hook_id / "source_probe.pth"
    )
    target_probe_path = probe_dir / "probe.pth"

    print(f"[layer {layer}] loading cached arrays from {data_path}", flush=True)
    with np.load(data_path) as np_data:
        y_train = load_npz_array(np_data, "y_train").astype(np.int64)
        y_test = load_npz_array(np_data, "y_test").astype(np.int64)
        x_train_source = load_npz_array(np_data, "X_train_source")
        x_test_source = load_npz_array(np_data, "X_test_source")
        x_test_target = load_npz_array(np_data, "X_test_target")
    if args.filter_eval_tokens:
        eval_idx = valid_eval_indices(probe_dir, "test", len(y_test))
        y_test = y_test[eval_idx]
        x_test_source = x_test_source[eval_idx]
        x_test_target = x_test_target[eval_idx]

    print(f"[layer {layer}] training/loading source probe", flush=True)
    source_probe = train_or_load_source_probe(
        source_probe_path=source_probe_path,
        x_train_source=x_train_source,
        y_train=y_train,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.source_epochs,
        lr=args.lr,
        force=args.force_source_probe,
        show_progress=args.show_progress,
    )
    del x_train_source

    print(f"[layer {layer}] loading target probe", flush=True)
    target_probe = load_target_probe(probe_dir, device)

    print(f"[layer {layer}] evaluating probes", flush=True)
    source_logits = probe_logits(source_probe, x_test_source, device, args.eval_batch_size)
    target_logits = probe_logits(target_probe, x_test_target, device, args.eval_batch_size)

    rows = merge_rows(
        per_letter_probe_metrics(source_logits, y_test, "source"),
        per_letter_probe_metrics(target_logits, y_test, "target"),
        normalized_mean_shifts(x_test_source, y_test, "source"),
        normalized_mean_shifts(x_test_target, y_test, "target"),
    )
    add_ratios(rows)
    for row in rows:
        row["layer"] = layer

    layer_summary = summarize_layer(layer, rows, source_probe_path, target_probe_path)

    write_csv(output_dir / f"layer_{layer}_per_letter.csv", rows)
    write_json(output_dir / f"layer_{layer}_summary.json", layer_summary)

    print(
        "[layer {layer}] mean source F1={source:.4f}, target F1={target:.4f}, "
        "mean rho={rho:.4f}".format(
            layer=layer,
            source=layer_summary["mean_source_f1"],
            target=layer_summary["mean_target_f1"],
            rho=layer_summary["mean_rho_target_over_source_shift"],
        ),
        flush=True,
    )
    return layer_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="12", help="Comma-separated layer list.")
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument(
        "--probe-cache-root",
        default=str(REPO_ROOT / "results/evals/absorption/cache/probes"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/diagnostics/source_target_letter_probe"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--force-source-probe", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--no-filter-eval-tokens",
        dest="filter_eval_tokens",
        action="store_false",
        help="Disable the absorption-compatible eval token filter.",
    )
    parser.set_defaults(filter_eval_tokens=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    layers = parse_layers(args.layers)
    print(f"Running source-vs-target probe diagnostic on {device}: layers={layers}")

    summaries = []
    for layer in layers:
        summaries.append(run_layer(args, layer, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    summary_rows = [dict(summary) for summary in summaries]
    write_csv(output_dir / "summary.csv", summary_rows)
    write_json(output_dir / "summary.json", summary_rows)

    print("\nSummary")
    for summary in summaries:
        print(
            "L{layer}: source_f1={source:.4f} target_f1={target:.4f} "
            "target-source={delta:+.4f} rho={rho:.4f} rho<1 letters={rho_lt}/26".format(
                layer=summary["layer"],
                source=summary["mean_source_f1"],
                target=summary["mean_target_f1"],
                delta=summary["mean_target_minus_source_f1"],
                rho=summary["mean_rho_target_over_source_shift"],
                rho_lt=summary["num_letters_rho_lt_1"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
