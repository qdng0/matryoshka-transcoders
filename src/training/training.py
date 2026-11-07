import torch
import tqdm
import math
import wandb

from src.utils.logs import init_wandb, log_wandb
from src.utils.activation_samples import ActivationSampleCollector
from src.utils.clean_results import CleanResultsManager
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(step, warmup_steps, warmup_start_ratio):
    if warmup_steps <= 0:
        return 1.0
    denom = max(1, warmup_steps - 1)
    warmup_progress = min(1.0, step / denom)
    cosine_warmup = 0.5 * (1 - math.cos(math.pi * warmup_progress))
    return warmup_start_ratio + (1 - warmup_start_ratio) * cosine_warmup


def warmup_linear(step, warmup_steps, warmup_start_ratio):
    if warmup_steps <= 0:
        return 1.0
    denom = max(1, warmup_steps - 1)
    warmup_progress = min(1.0, step / denom)
    return warmup_start_ratio + (1 - warmup_start_ratio) * warmup_progress


def warmup_cosine_decay(step, warmup_steps, total_steps, min_lr_ratio, warmup_start_ratio):
    if step < warmup_steps:
        return warmup_cosine(step, warmup_steps, warmup_start_ratio)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay


def warmup_linear_decay(
    step,
    warmup_steps,
    total_steps,
    decay_start_fraction,
    final_lr_ratio,
    warmup_start_ratio,
):
    if step < warmup_steps:
        return warmup_linear(step, warmup_steps, warmup_start_ratio)

    decay_start = int(total_steps * decay_start_fraction)
    if step < decay_start:
        return 1.0

    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return final_lr_ratio + (1.0 - final_lr_ratio) * max(0.0, 1.0 - progress)


def create_scheduler(optimizer, cfg, num_batches):
    """Create learning rate scheduler based on config."""
    scheduler_type = cfg.get("scheduler_type", "warmup_decay")
    warmup_steps = int(cfg.get("warmup_steps", 1000) or 0)
    min_lr_ratio = cfg.get("min_lr_ratio", 0.01)
    warmup_start_ratio = cfg.get("warmup_start_ratio", 0.1)
    decay_start_fraction = float(cfg.get("decay_start_fraction", 0.8))
    final_lr_ratio = float(cfg.get("final_lr_ratio", min_lr_ratio))

    if scheduler_type == "warmup_decay":
        lr_fn = lambda step: warmup_cosine_decay(
            step,
            warmup_steps,
            num_batches,
            min_lr_ratio,
            warmup_start_ratio,
        )
    elif scheduler_type == "warmup_only":
        lr_fn = lambda step: warmup_cosine(step, warmup_steps, warmup_start_ratio)
    elif scheduler_type == "warmup_linear_decay":
        lr_fn = lambda step: warmup_linear_decay(
            step,
            warmup_steps,
            num_batches,
            decay_start_fraction,
            final_lr_ratio,
            warmup_start_ratio,
        )
    else:
        return None

    return LambdaLR(optimizer, lr_fn)


def log_dead_features(transcoder, cfg, wandb_run, step):
    """Log dead feature statistics to wandb."""
    dead_mask = transcoder.num_batches_not_active >= cfg.get("n_batches_to_dead", 20)
    dead_count = int(dead_mask.sum().item())
    total_features = int(transcoder.num_batches_not_active.numel())
    wandb_run.log({
        "dead_features_count": dead_count,
        "dead_features_percentage": dead_count / max(1, total_features) * 100,
    }, step=step)
    return dead_count


def save_training_checkpoint(results_manager, experiment_dir, transcoder, cfg, step, 
                              transcoder_output, dead_count, is_final=False):
    """Save model checkpoint with metrics."""
    metrics = {
        "loss": float(transcoder_output['loss'].detach().cpu()),
        "l2_loss": float(transcoder_output['l2_loss'].detach().cpu()),
        "l0_norm": float(transcoder_output['l0_norm'].detach().cpu()),
        "fvu": float(transcoder_output['fvu'].detach().cpu()),
        "dead_features": dead_count,
    }
    if "aux_loss" in transcoder_output:
        metrics["aux_loss"] = float(transcoder_output["aux_loss"].detach().cpu())
    if "ghost_grad_loss" in transcoder_output:
        metrics["ghost_grad_loss"] = float(
            transcoder_output["ghost_grad_loss"].detach().cpu()
        )
    if "l1_loss" in transcoder_output:
        metrics["l1_loss"] = float(transcoder_output["l1_loss"].detach().cpu())
    if "l0_loss" in transcoder_output:
        metrics["l0_loss"] = float(transcoder_output["l0_loss"].detach().cpu())
        
    checkpoint_cfg = dict(cfg)
    if is_final:
        checkpoint_cfg["completed_steps"] = step
    
    results_manager.save_checkpoint(
        experiment_dir=experiment_dir,
        model=transcoder,
        config=checkpoint_cfg,
        step=step,
        metrics=metrics,
        is_final=is_final
    )
    return metrics


def train_transcoder(transcoder, activation_store, model, cfg):
    num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
    trainable_params = [p for p in transcoder.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable transcoder parameters found.")
    optimizer = torch.optim.Adam(trainable_params, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = create_scheduler(optimizer, cfg, num_batches)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    l0_coef = float(cfg.get("act_l0_coef", 0.0) or 0.0)
    l0_warmup_steps = int(cfg.get("act_l0_warmup_steps", 0) or 0)
    freeze_decoder = bool(cfg.get("freeze_decoder", False))
    if scheduler is not None:
        scheduler.step(0)

    wandb_run = init_wandb(cfg)
    results_manager = CleanResultsManager(base_dir=cfg.get("results_base_dir", "results"))
    experiment_dir = results_manager.create_experiment_dir(
        model_name=cfg["model_name"],
        layer=cfg["layer"],
        steps=num_batches,
        description=cfg.get("experiment_description", ""),
        config=cfg
    )

    sample_collector = None
    if cfg.get("save_activation_samples", False):
        sample_collector = ActivationSampleCollector(
            max_samples_per_feature=cfg.get("max_samples_per_feature", 100),
            context_size=cfg.get("sample_context_size", 20),
            storage_threshold=cfg.get("sample_activation_threshold", 0.1)
        )

    pbar = tqdm.trange(num_batches)
    for i in pbar:
        transcoder.training_step = i
        if l0_coef and l0_warmup_steps > 0:
            warmup_ratio = min(1.0, i / max(1, l0_warmup_steps))
            transcoder.config["act_l0_coef_current"] = l0_coef * warmup_ratio
        else:
            transcoder.config.pop("act_l0_coef_current", None)

        source_batch, target_batch = activation_store.next_batch()
        transcoder_output = transcoder(source_batch, target_batch)
        log_wandb(transcoder_output, i, wandb_run)
        dead_count = log_dead_features(transcoder, cfg, wandb_run, i)
        
        if scheduler is not None:
            wandb_run.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=i)

        if i % cfg["perf_log_freq"] == 0:
            log_transcoder_performance(wandb_run, i, activation_store, transcoder)

        if i % cfg["checkpoint_freq"] == 0:
            save_training_checkpoint(results_manager, experiment_dir, transcoder, cfg, 
                                    i, transcoder_output, dead_count, is_final=False)

        if sample_collector and i % cfg.get("sample_collection_freq", 1000) == 0:
            source_acts, batch_tokens = activation_store.get_batch_with_tokens()
            with torch.no_grad():
                sparse_features = transcoder.encode(source_acts)
            sample_collector.collect_batch_samples(sparse_features, batch_tokens, activation_store.tokenizer)

        pbar.set_description(
            f"Loss: {transcoder_output['loss'].item():.3f}, "
            f"L0: {transcoder_output['l0_norm'].item():.0f}, "
            f"FVU: {transcoder_output['fvu'].item():.3f}, "
            f"Dead: {dead_count}"
        )

        transcoder_output["loss"].backward()
        if not freeze_decoder:
            transcoder.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        if not freeze_decoder:
            transcoder.renorm_decoder_weights()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step(i + 1)

    final_dead_mask = transcoder.num_batches_not_active >= cfg.get("n_batches_to_dead", 20)
    final_dead_count = int(final_dead_mask.sum().item())
    save_training_checkpoint(results_manager, experiment_dir, transcoder, cfg, 
                             num_batches, transcoder_output, final_dead_count, is_final=True)

    if sample_collector:
        samples_dir = experiment_dir / "activation_samples"
        samples_dir.mkdir(exist_ok=True)
        sample_collector.save_samples(
            str(samples_dir),
            top_k_features=cfg.get("top_features_to_save", 100),
            samples_per_feature=cfg.get("samples_per_feature_to_save", 10)
        )

    wandb_run.finish()


def log_transcoder_performance(wandb_run, step, activation_store, transcoder):
    """Log detailed performance metrics on a fresh batch."""
    source_batch, target_batch = activation_store.next_batch()
    with torch.no_grad():
        output = transcoder(source_batch, target_batch)
        reconstruction = output["transcoder_out"]
        wandb_run.log({
            "performance/mse": (reconstruction - target_batch).pow(2).mean().item(),
            "performance/mae": (reconstruction - target_batch).abs().mean().item(),
            "performance/cosine_similarity": torch.nn.functional.cosine_similarity(
                reconstruction.flatten(), target_batch.flatten(), dim=0
            ).item(),
            "performance/fvu": output["fvu"].item(),
        }, step=step)
