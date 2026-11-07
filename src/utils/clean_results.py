"""Utilities for saving training runs."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def _make_json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to strings."""
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


class CleanResultsManager:
    """Helper class for writing training artifacts to disk."""

    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_dir(
        self,
        model_name: str,
        layer: int,
        steps: int,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create experiment directory: results/training/[model_type]/layer_{layer}/{steps}/{date}/"""
        # Determine model type from config
        model_type = self._get_model_type(config) if config else "unknown"
        
        # New structure: results/training/[model_type]/layer_{layer}/{steps}/{date}/
        type_dir = self.base_dir / "training" / model_type
        layer_dir = type_dir / f"layer_{layer}"
        date_str = datetime.now().strftime("%b-%d")
        
        # Add l0_coef suffix for jumprelu runs to distinguish different l0_coef values
        run_suffix = ""
        if config:
            if config.get("activation_type") == "jumprelu":
                l0_coef = config.get("act_l0_coef")
                if l0_coef and l0_coef != 0:
                    l0_str = f"{l0_coef:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                    run_suffix = f"_l0-{l0_str}"
            elif config.get("activation_type") == "batchtopk" and not config.get("sae_mode", False):
                batchtopk_mode = config.get("batchtopk_mode", "global")
                run_suffix = f"-{batchtopk_mode}"

        description_suffix = ""
        if description:
            sanitized_description = re.sub(r"[^A-Za-z0-9._-]+", "-", description.strip()).strip("-")
            if sanitized_description:
                description_suffix = f"_{sanitized_description}"

        run_dir = layer_dir / str(steps) / f"{date_str}{run_suffix}{description_suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        return run_dir
    
    def _get_model_type(self, config: Dict[str, Any]) -> str:
        """Determine model type from config."""
        if config.get("sae_mode", False):
            if config.get("trainer_class") == "MatryoshkaBatchTopKTrainer":
                return "matryoshka_sae"
            return config.get("sae_type", "sae")

        explicit_type = config.get("transcoder_type")
        if explicit_type:
            return explicit_type
        group_sizes = config.get("group_sizes", [])
        
        # Matryoshka has multiple groups
        if isinstance(group_sizes, list) and len(group_sizes) > 1:
            return "matryoshka"
        
        # Baseline uses activation_type
        activation_type = config.get("activation_type") or "batchtopk"
        return activation_type

    def save_checkpoint(
        self,
        experiment_dir: Path,
        model: torch.nn.Module,
        config: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
    ) -> None:
        """Persist model state, config, and metrics to the run directory."""

        checkpoints_dir = experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        checkpoint_name = "final.pt" if is_final else f"step_{step}.pt"
        checkpoint_path = checkpoints_dir / checkpoint_name
        torch.save(model.state_dict(), checkpoint_path)

        config_copy = _make_json_serializable(config)
        if "created_at" not in config_copy:
            config_copy["created_at"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        if is_final:
            config_copy["completed_at"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        else:
            config_copy.pop("completed_at", None)

        config_path = experiment_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_copy, f, indent=2)

        if metrics:
            metrics_copy = _make_json_serializable(metrics)
            if is_final:
                metrics_path = experiment_dir / "metrics.json"
            else:
                interim_dir = experiment_dir / "metrics"
                interim_dir.mkdir(exist_ok=True)
                metrics_path = interim_dir / f"step_{step}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics_copy, f, indent=2)

    def list_experiments(self, model_type: Optional[str] = None, layer: Optional[int] = None) -> Dict[str, Any]:
        """Return a dictionary describing saved runs.
        
        New structure: results/training/{model_type}/layer_{X}/{steps}/{date}/
        """
        experiments: Dict[str, Any] = {}
        
        training_dir = self.base_dir / "training"
        if not training_dir.exists():
            return experiments

        for type_dir in training_dir.iterdir():
            if not type_dir.is_dir():
                continue
            if model_type and type_dir.name != model_type:
                continue

            type_runs: Dict[int, Any] = {}

            for layer_dir in type_dir.iterdir():
                if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                    continue
                try:
                    layer_num = int(layer_dir.name.replace("layer_", ""))
                except ValueError:
                    continue
                if layer is not None and layer_num != layer:
                    continue

                runs = []
                for steps_dir in sorted(layer_dir.iterdir()):
                    if not steps_dir.is_dir():
                        continue
                    
                    # Each steps dir can have multiple date subdirs
                    for run_dir in sorted(steps_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue

                        runs.append(
                            {
                                "steps": steps_dir.name,
                                "date": run_dir.name,
                                "path": str(run_dir),
                                "has_final_checkpoint": (run_dir / "checkpoints" / "final.pt").exists(),
                                "has_metrics": (run_dir / "metrics.json").exists(),
                            }
                        )

                if runs:
                    type_runs[layer_num] = runs

            if type_runs:
                experiments[type_dir.name] = type_runs

        return experiments
