#!/usr/bin/env python3
"""Rerun RAVEL + SCR on the default2 grid.

This helper dispatches each cell to the transcoder evaluation pipelines:

  * ``eval/sae_bench/ravel/main_transcoder_grid.py`` -- the shadow MDBM
    installed at module load time keeps interventions in the target-hook space.
  * ``eval/sae_bench/scr_and_tpp/main_transcoder_grid.py --perform_scr
    --transcoder_fix`` -- routes through the dual-hook implementation in
    ``transcoder_scr_fix.run_eval_single_sae_transcoder``.

Results land under ``results/default2/{eval}/{type}/{layer}/``.

Usage
-----
    # Sanity: just one cell, both evals
    python eval/rerun_default2_evals.py --cells matryoshka:24

    # Full sweep, RAVEL only
    python eval/rerun_default2_evals.py --evals ravel

    # Full sweep, both evals
    python eval/rerun_default2_evals.py

    # Re-run a cell that already has a result.json
    python eval/rerun_default2_evals.py --cells matryoshka:24 --force_rerun

    # Full SCR sweep on two GPUs (two concurrent eval subprocesses)
    python eval/rerun_default2_evals.py --evals scr --cuda_devices 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ARCHITECTURES = ["matryoshka", "relu", "jumprelu", "topk", "batchtopk"]
LAYERS = [0, 6, 12, 18, 24]
EVALS = ["ravel", "scr"]
MODEL_NAME = "gemma-2-2b"

RAVEL_SCRIPT = REPO_ROOT / "eval/sae_bench/ravel/main_transcoder_grid.py"
SCR_SCRIPT = REPO_ROOT / "eval/sae_bench/scr_and_tpp/main_transcoder_grid.py"


def cell_paths(arch: str, layer: int) -> tuple[Path, Path]:
    """Return (config_path, checkpoint_path) for a default2 cell."""
    base = REPO_ROOT / "results/default2/training" / arch / str(layer)
    return base / "config.json", base / "checkpoints" / "final.pt"


def output_dir(eval_name: str, arch: str, layer: int) -> Path:
    return REPO_ROOT / "results/default2" / eval_name / arch / str(layer)


def existing_result(eval_dir: Path) -> Path | None:
    """Locate the result file inside ``eval_dir``.

    Both RAVEL and SCR grid scripts emit a single JSON whose name embeds
    the transcoder identifier. We surface that file plus a normalised
    ``result.json`` symlink to match the rest of the default2 layout.
    """
    if not eval_dir.exists():
        return None
    direct = eval_dir / "result.json"
    if direct.exists():
        return direct
    candidates = sorted(eval_dir.glob("*.json"))
    return candidates[0] if candidates else None


def normalise_result_filename(eval_dir: Path) -> Path | None:
    """After a run, ensure ``result.json`` exists alongside the long-named file.

    Mirrors the convention used elsewhere in ``results/default2/``.
    """
    result_link = eval_dir / "result.json"
    if result_link.exists():
        return result_link
    candidates = sorted(
        p for p in eval_dir.glob("*.json") if p.name != "result.json"
    )
    if not candidates:
        return None
    src = candidates[-1]  # most recently produced
    shutil.copy2(src, result_link)
    return result_link


def build_ravel_cmd(
    config_path: Path, checkpoint_path: Path, output_folder: Path
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(RAVEL_SCRIPT),
        "--config_path",
        str(config_path),
        "--checkpoint_path",
        str(checkpoint_path),
        "--model_name",
        MODEL_NAME,
        "--output_folder",
        str(output_folder),
        "--force_rerun",
    ]
    return cmd


def build_scr_cmd(
    config_path: Path,
    checkpoint_path: Path,
    output_folder: Path,
    sae_batch_size: int | None = None,
    llm_batch_size: int | None = None,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(SCR_SCRIPT),
        "--config_path",
        str(config_path),
        "--checkpoint_path",
        str(checkpoint_path),
        "--model_name",
        MODEL_NAME,
        "--output_folder",
        str(output_folder),
        "--perform_scr",
        "--transcoder_fix",
        "--force_rerun",
    ]
    if sae_batch_size is not None:
        cmd.extend(["--sae_batch_size", str(sae_batch_size)])
    if llm_batch_size is not None:
        cmd.extend(["--llm_batch_size", str(llm_batch_size)])
    return cmd


def run_one_cell(
    eval_name: str,
    arch: str,
    layer: int,
    force_rerun: bool,
    log_root: Path,
    cuda_visible_devices: str | None = None,
    sae_batch_size: int | None = None,
    llm_batch_size: int | None = None,
) -> dict:
    config_path, checkpoint_path = cell_paths(arch, layer)
    if not config_path.exists() or not checkpoint_path.exists():
        return {
            "cell": f"{arch}:{layer}",
            "eval": eval_name,
            "status": "skip_missing_checkpoint",
            "elapsed_s": 0.0,
        }

    out_dir = output_dir(eval_name, arch, layer)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force_rerun and (out_dir / "result.json").exists():
        return {
            "cell": f"{arch}:{layer}",
            "eval": eval_name,
            "status": "skip_already_done",
            "elapsed_s": 0.0,
            "result": str(out_dir / "result.json"),
        }

    if eval_name == "ravel":
        # RAVEL writes into <output_folder>/*.json directly.
        # Point it at the per-cell directory so we get a clean per-cell tree.
        cmd = build_ravel_cmd(config_path, checkpoint_path, out_dir)
    elif eval_name == "scr":
        # SCR appends an "scr/" sub-dir under output_folder, so we
        # supply the parent (per-cell dir) and let it create scr/ inside.
        cmd = build_scr_cmd(
            config_path,
            checkpoint_path,
            out_dir,
            sae_batch_size=sae_batch_size,
            llm_batch_size=llm_batch_size,
        )
    else:
        raise ValueError(f"Unknown eval: {eval_name}")

    log_dir = log_root / eval_name / arch / str(layer)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\n[{eval_name}] {arch}:{layer} -> {out_dir}")
    print(f"  log: {log_path}")
    print(f"  cmd: {' '.join(cmd)}")

    run_env = os.environ.copy()
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    start = time.time()
    with log_path.open("w") as logf:
        logf.write(f"# {datetime.now().isoformat()} {' '.join(cmd)}\n\n")
        if cuda_visible_devices is not None:
            logf.write(f"# CUDA_VISIBLE_DEVICES={cuda_visible_devices}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=run_env if cuda_visible_devices is not None else None,
        )
    elapsed = time.time() - start

    # SCR writes into <output_folder>/scr/*.json. Hoist that
    # one level up so the per-cell folder layout matches RAVEL's.
    if eval_name == "scr":
        scr_subdir = out_dir / "scr"
        if scr_subdir.exists():
            for f in scr_subdir.glob("*.json"):
                target = out_dir / f.name
                shutil.move(str(f), str(target))
            try:
                scr_subdir.rmdir()
            except OSError:
                pass

    result_path = normalise_result_filename(out_dir)

    status = (
        "ok"
        if proc.returncode == 0 and result_path is not None
        else f"fail_rc={proc.returncode}_no_result={result_path is None}"
    )

    print(f"  done in {elapsed/60:.1f} min, status={status}")
    return {
        "cell": f"{arch}:{layer}",
        "eval": eval_name,
        "status": status,
        "elapsed_s": elapsed,
        "result": str(result_path) if result_path else None,
        "log": str(log_path),
    }


def parse_cells(spec: str | None) -> list[tuple[str, int]]:
    if not spec:
        return [(a, l) for a in ARCHITECTURES for l in LAYERS]
    cells: list[tuple[str, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Bad cell spec '{tok}', expected '<arch>:<layer>'")
        arch, layer = tok.split(":", 1)
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch}")
        cells.append((arch, int(layer)))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help=(
            "Comma-separated cell list, e.g. 'matryoshka:24,relu:0'. "
            "Default: all 25 cells."
        ),
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        choices=EVALS,
        default=EVALS,
        help="Which evals to run per cell.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-run cells whose result.json already exists.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/default2/_rerun_summary.json",
        help="Where to write the per-cell run summary.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help=(
            "Comma-separated physical GPU indices passed to each eval subprocess "
            "via CUDA_VISIBLE_DEVICES. Multiple entries enable concurrent jobs "
            "(one subprocess per device), e.g. '0,1'. Default: inherit environment "
            "(sequential jobs, no override)."
        ),
    )
    parser.add_argument(
        "--sae_batch_size",
        type=int,
        default=None,
        help="Optional SCR SAE batch size override. Useful for low-VRAM cells.",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Optional SCR LLM batch size override.",
    )
    args = parser.parse_args()

    cells = parse_cells(args.cells)
    cuda_devices = (
        [d.strip() for d in args.cuda_devices.split(",") if d.strip()]
        if args.cuda_devices
        else []
    )

    jobs: list[tuple[str, str, int]] = []
    for arch, layer in cells:
        for eval_name in args.evals:
            jobs.append((eval_name, arch, layer))

    print(
        f"Running {len(jobs)} job(s) "
        f"({len(args.evals)} eval(s) x {len(cells)} cell(s))"
        + (
            f", cuda_devices={cuda_devices} (max {len(cuda_devices)} parallel)"
            if cuda_devices
            else ", sequential (inherit CUDA)"
        ),
        flush=True,
    )
    log_root = REPO_ROOT / "logs/rerun_default2_evals"
    log_root.mkdir(parents=True, exist_ok=True)
    cuda_queue: queue.Queue[str] | None = None
    if len(cuda_devices) > 1:
        cuda_queue = queue.Queue()
        for cuda_device in cuda_devices:
            cuda_queue.put(cuda_device)

    overall_start = time.time()
    summary_path = REPO_ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lock = threading.Lock()
    summary: list[dict | None] = [None] * len(jobs)

    def persist_summary() -> None:
        with summary_path.open("w") as f:
            json.dump(
                {
                    "started": datetime.fromtimestamp(
                        overall_start
                    ).isoformat(),
                    "elapsed_s": time.time() - overall_start,
                    "cuda_devices": cuda_devices or None,
                    "results": summary,
                },
                f,
                indent=2,
            )

    def run_job(job_index: int, eval_name: str, arch: str, layer: int) -> None:
        cuda_dev = None
        if cuda_queue is not None:
            cuda_dev = cuda_queue.get()
        elif cuda_devices:
            cuda_dev = cuda_devices[0]
        try:
            entry = run_one_cell(
                eval_name=eval_name,
                arch=arch,
                layer=layer,
                force_rerun=args.force_rerun,
                log_root=log_root,
                cuda_visible_devices=cuda_dev,
                sae_batch_size=args.sae_batch_size,
                llm_batch_size=args.llm_batch_size,
            )
        finally:
            if cuda_queue is not None and cuda_dev is not None:
                cuda_queue.put(cuda_dev)
        with summary_lock:
            summary[job_index] = entry
            persist_summary()

    if len(cuda_devices) <= 1:
        cuda_dev = cuda_devices[0] if cuda_devices else None
        for i, (eval_name, arch, layer) in enumerate(jobs):
            run_job(i, eval_name, arch, layer)
    else:
        with ThreadPoolExecutor(max_workers=len(cuda_devices)) as pool:
            futures = {
                pool.submit(run_job, i, eval_name, arch, layer): i
                for i, (eval_name, arch, layer) in enumerate(jobs)
            }
            for fut in as_completed(futures):
                fut.result()

    total_elapsed = time.time() - overall_start
    ok = sum(1 for s in summary if s["status"] == "ok")
    skipped = sum(
        1
        for s in summary
        if s["status"].startswith("skip_")
    )
    failed = len(summary) - ok - skipped

    print(
        f"\nDone in {total_elapsed/60:.1f} min: "
        f"{ok} ok, {skipped} skipped, {failed} failed"
    )
    print(f"Summary: {REPO_ROOT / args.summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
