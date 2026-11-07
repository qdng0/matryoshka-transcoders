#!/usr/bin/env python3
"""Fit a fixed affine ridge skip for source->target transcoders.

The saved artifact contains A and c for row-vector inference:

    skip = x @ A + c

It also records the selected ridge alpha by heldout skip-only FVU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.transcoder_activation_store import TranscoderActivationsStore  # noqa: E402
from src.scripts.train_matryoshka_all_layers import parse_torch_dtype  # noqa: E402
from src.utils.config import (  # noqa: E402
    create_transcoder_config,
    get_default_cfg,
    get_hook_name,
    load_hooked_transformer,
    post_init_cfg,
)


@dataclass
class MomentStats:
    d_in: int
    d_out: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self):
        self.n = 0
        self.sum_x = torch.zeros(self.d_in, device=self.device, dtype=self.dtype)
        self.sum_y = torch.zeros(self.d_out, device=self.device, dtype=self.dtype)
        self.sum_xx = torch.zeros(
            self.d_in, self.d_in, device=self.device, dtype=self.dtype
        )
        self.sum_xy = torch.zeros(
            self.d_in, self.d_out, device=self.device, dtype=self.dtype
        )
        self.sum_y_sq = torch.zeros((), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).to(device=self.device, dtype=self.dtype)
        y = y.reshape(-1, y.shape[-1]).to(device=self.device, dtype=self.dtype)
        if x.shape[0] != y.shape[0]:
            raise ValueError("Source and target batches have different token counts.")
        self.n += int(x.shape[0])
        self.sum_x.add_(x.sum(dim=0))
        self.sum_y.add_(y.sum(dim=0))
        self.sum_xx.add_(x.T @ x)
        self.sum_xy.add_(x.T @ y)
        self.sum_y_sq.add_(y.pow(2).sum())

    def centered(self) -> dict[str, torch.Tensor]:
        if self.n <= 0:
            raise ValueError("No tokens were accumulated.")
        n = float(self.n)
        mu_x = self.sum_x / n
        mu_y = self.sum_y / n
        exx = self.sum_xx / n
        exy = self.sum_xy / n
        cxx = exx - torch.outer(mu_x, mu_x)
        cxy = exy - torch.outer(mu_x, mu_y)
        return {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "exx": exx,
            "exy": exy,
            "cxx": cxx,
            "cxy": cxy,
            "ey_sq": self.sum_y_sq / n,
        }

    def heldout_fvu(self, A: torch.Tensor, c: torch.Tensor) -> dict[str, float]:
        stats = self.centered()
        ex = stats["mu_x"]
        ey = stats["mu_y"]
        exx = stats["exx"]
        exy = stats["exy"]
        ey_sq = stats["ey_sq"]

        y_dot_xa = (exy * A).sum()
        xa_sq = ((exx @ A) * A).sum()
        c_dot_y = c.dot(ey)
        c_dot_xa = c.dot(ex @ A)
        c_sq = c.dot(c)
        residual_sq = ey_sq - 2 * y_dot_xa - 2 * c_dot_y + xa_sq + 2 * c_dot_xa + c_sq
        target_sq_centered = ey_sq - ey.dot(ey)
        fvu = residual_sq / target_sq_centered.clamp_min(1e-12)
        mse = residual_sq / self.d_out
        variance = target_sq_centered / self.d_out
        return {
            "fvu_skip": float(fvu.detach().cpu()),
            "mse_skip": float(mse.detach().cpu()),
            "target_variance": float(variance.detach().cpu()),
        }


def parse_float_list(csv: str) -> list[float]:
    return [float(part.strip()) for part in csv.split(",") if part.strip()]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def resolve_fold_ln(args: argparse.Namespace) -> bool:
    if args.fold_ln is not None:
        return bool(args.fold_ln)
    return "gemma" in args.model_name.lower()


def build_cfg(args: argparse.Namespace) -> dict:
    cfg = get_default_cfg()
    cfg.update(
        {
            "model_name": args.model_name,
            "seed": args.seed,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "dataset_shuffle_buffer": args.dataset_shuffle_buffer,
            "trust_remote_code": True,
            "fold_ln": resolve_fold_ln(args),
            "layer": args.layer,
            "model_batch_size": args.model_batch_size,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_batches_in_buffer": args.num_batches_in_buffer,
            "model_dtype": parse_torch_dtype(args.model_dtype),
            "dtype": parse_torch_dtype(args.dtype),
            "device": resolve_device(args.device),
            "buffer_device": args.buffer_device,
            "activation_cache_path": None,
            "activation_cache_size": 0,
            "prepend_bos": args.prepend_bos,
        }
    )
    cfg = create_transcoder_config(
        cfg,
        source_layer=args.layer,
        target_layer=args.layer,
        source_site=args.source_site,
        target_site=args.target_site,
    )
    if args.source_hook:
        cfg["source_hook_point"] = args.source_hook
    if args.target_hook:
        cfg["target_hook_point"] = args.target_hook
    cfg = post_init_cfg(cfg)
    return cfg


@torch.no_grad()
def collect_stats(
    activation_store: TranscoderActivationsStore,
    n_tokens: int,
    *,
    d_in: int,
    d_out: int,
    stats_device: torch.device,
    stats_dtype: torch.dtype,
    label: str,
) -> MomentStats:
    stats = MomentStats(d_in=d_in, d_out=d_out, device=stats_device, dtype=stats_dtype)
    pbar = tqdm(total=n_tokens, desc=label, unit="tok")
    while stats.n < n_tokens:
        source, target = activation_store.next_batch()
        remaining = n_tokens - stats.n
        if source.shape[0] > remaining:
            source = source[:remaining]
            target = target[:remaining]
        stats.update(source, target)
        pbar.update(int(source.shape[0]))
    pbar.close()
    return stats


def fit_ridge(
    train_stats: MomentStats,
    heldout_stats: MomentStats,
    alphas: list[float],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    centered = train_stats.centered()
    cxx = centered["cxx"]
    cxy = centered["cxy"]
    mu_x = centered["mu_x"]
    mu_y = centered["mu_y"]
    d_in = cxx.shape[0]
    trace_scale = cxx.diag().mean().clamp_min(1e-12)
    eye = torch.eye(d_in, device=cxx.device, dtype=cxx.dtype)

    alpha_metrics = []
    best = None
    for alpha in alphas:
        ridge_lambda = float(alpha) * trace_scale
        A = torch.linalg.solve(cxx + ridge_lambda * eye, cxy)
        c = mu_y - mu_x @ A
        metrics = heldout_stats.heldout_fvu(A, c)
        metrics.update(
            {
                "alpha": float(alpha),
                "lambda": float(ridge_lambda.detach().cpu()),
            }
        )
        alpha_metrics.append(metrics)
        if best is None or metrics["fvu_skip"] < best["metrics"]["fvu_skip"]:
            best = {"A": A, "c": c, "metrics": metrics}

    if best is None:
        raise ValueError("No ridge alphas were provided.")

    metadata = {
        "selected_alpha": best["metrics"]["alpha"],
        "selected_lambda": best["metrics"]["lambda"],
        "heldout_fvu_skip": best["metrics"]["fvu_skip"],
        "ridge_alphas": alpha_metrics,
        "trace_cxx_over_d": float(trace_scale.detach().cpu()),
        "n_train_tokens": int(train_stats.n),
        "n_heldout_tokens": int(heldout_stats.n),
    }
    return best["A"], best["c"], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="gemma-2-2b")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source-site", type=str, default="mlp_in")
    parser.add_argument("--target-site", type=str, default="mlp_out")
    parser.add_argument("--source-hook", type=str, default="")
    parser.add_argument("--target-hook", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=0)
    parser.add_argument("--n-tokens", type=int, default=50_000_000)
    parser.add_argument(
        "--n-heldout-tokens",
        type=int,
        default=None,
        help="Heldout activation tokens for alpha selection. Defaults to min(5M, n_tokens/5).",
    )
    parser.add_argument("--ridge-alphas", type=str, default="1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stats-device", type=str, default="auto")
    parser.add_argument(
        "--stats-dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
    )
    parser.add_argument("--model-dtype", type=str, default="float32")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-batches-in-buffer", type=int, default=10)
    parser.add_argument("--buffer-device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--fold-ln", dest="fold_ln", action="store_true", default=None)
    parser.add_argument("--no-fold-ln", dest="fold_ln", action="store_false")
    parser.add_argument("--save-covariances", action="store_true")
    args = parser.parse_args()

    if args.source_hook and not args.source_site:
        args.source_site = "mlp_in"
    if args.target_hook and not args.target_site:
        args.target_site = "mlp_out"

    cfg = build_cfg(args)
    stats_device_str = resolve_device(args.stats_device)
    stats_device = torch.device(stats_device_str)
    stats_dtype = parse_torch_dtype(args.stats_dtype)
    heldout_tokens = args.n_heldout_tokens
    if heldout_tokens is None:
        heldout_tokens = max(args.batch_size, min(5_000_000, args.n_tokens // 5))

    print(
        "Fitting ridge skip: "
        f"model={cfg['model_name']} layer={args.layer} "
        f"{cfg['source_hook_point']} -> {cfg['target_hook_point']} "
        f"train_tokens={args.n_tokens:,} heldout_tokens={heldout_tokens:,} "
        f"device={cfg['device']} stats_device={stats_device}"
    )

    model = load_hooked_transformer(cfg)
    activation_store = TranscoderActivationsStore(model, cfg)
    d_in = int(cfg.get("source_act_size", cfg["act_size"]))
    d_out = int(cfg.get("target_act_size", cfg["act_size"]))

    train_stats = collect_stats(
        activation_store,
        args.n_tokens,
        d_in=d_in,
        d_out=d_out,
        stats_device=stats_device,
        stats_dtype=stats_dtype,
        label="train moments",
    )
    heldout_stats = collect_stats(
        activation_store,
        heldout_tokens,
        d_in=d_in,
        d_out=d_out,
        stats_device=stats_device,
        stats_dtype=stats_dtype,
        label="heldout moments",
    )

    A, c, metadata = fit_ridge(
        train_stats=train_stats,
        heldout_stats=heldout_stats,
        alphas=parse_float_list(args.ridge_alphas),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "A": A.detach().cpu().float(),
        "c": c.detach().cpu().float(),
        "x_mean": train_stats.centered()["mu_x"].detach().cpu().float(),
        "y_mean": train_stats.centered()["mu_y"].detach().cpu().float(),
        "source_hook_point": cfg["source_hook_point"],
        "target_hook_point": cfg["target_hook_point"],
        "layer": int(args.layer),
        "model_name": cfg["model_name"],
        "dataset_path": cfg["dataset_path"],
        "dataset_split": cfg["dataset_split"],
        "dataset_shuffle_buffer": cfg.get("dataset_shuffle_buffer", 0),
        "metrics": metadata,
    }
    if args.save_covariances:
        centered = train_stats.centered()
        artifact["Cxx"] = centered["cxx"].detach().cpu().float()
        artifact["Cxy"] = centered["cxy"].detach().cpu().float()

    torch.save(artifact, out_dir / "ridge_skip.pt")
    metrics = {
        **metadata,
        "source_hook_point": cfg["source_hook_point"],
        "target_hook_point": cfg["target_hook_point"],
        "layer": int(args.layer),
        "model_name": cfg["model_name"],
        "dataset_path": cfg["dataset_path"],
        "dataset_split": cfg["dataset_split"],
        "ridge_skip_path": str((out_dir / "ridge_skip.pt").resolve()),
    }
    with (out_dir / "ridge_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        "Saved ridge skip to "
        f"{out_dir / 'ridge_skip.pt'} | alpha={metadata['selected_alpha']} "
        f"heldout FVU={metadata['heldout_fvu_skip']:.4f}"
    )

    if os.environ.get("RIDGE_SKIP_FORCE_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
