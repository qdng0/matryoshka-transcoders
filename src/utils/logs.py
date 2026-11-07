import wandb
import torch


def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)


def log_wandb(output, step, wandb_run, index=None):
    """Log scalar metrics to wandb."""
    log_dict = {
        k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
        for k, v in output.items()
        if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.dim() == 0)
    }
    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}
    wandb_run.log(log_dict, step=step)
