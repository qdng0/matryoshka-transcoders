import torch
import tqdm
import wandb

from src.training.training import create_scheduler
from src.utils.logs import init_wandb, log_wandb
from src.utils.activation_samples import ActivationSampleCollector
from src.utils.clean_results import CleanResultsManager


def _dead_feature_threshold(cfg):
    return int(cfg.get("dead_steps_threshold", cfg.get("n_batches_to_dead", 20)))


def log_dead_features(sae, cfg, wandb_run, step):
    """Log dead feature statistics to wandb."""
    dead_mask = sae.num_batches_not_active >= _dead_feature_threshold(cfg)
    dead_count = int(dead_mask.sum().item())
    total_features = int(sae.num_batches_not_active.numel())
    wandb_run.log(
        {
            "dead_features_count": dead_count,
            "dead_features_percentage": dead_count / max(1, total_features) * 100,
        },
        step=step,
    )
    return dead_count


def save_training_checkpoint(
    results_manager,
    experiment_dir,
    sae,
    cfg,
    step,
    sae_output,
    dead_count,
    is_final=False,
):
    """Save model checkpoint with metrics."""
    metrics = {
        "loss": float(sae_output["loss"].detach().cpu()),
        "l2_loss": float(sae_output["l2_loss"].detach().cpu()),
        "l0_norm": float(sae_output["l0_norm"].detach().cpu()),
        "fvu": float(sae_output["fvu"].detach().cpu()),
        "dead_features": dead_count,
    }

    checkpoint_cfg = dict(cfg)
    if is_final:
        checkpoint_cfg["completed_steps"] = step

    results_manager.save_checkpoint(
        experiment_dir=experiment_dir,
        model=sae,
        config=checkpoint_cfg,
        step=step,
        metrics=metrics,
        is_final=is_final,
    )
    return metrics


def log_sae_performance(wandb_run, step, activation_store, sae):
    """Log detailed performance metrics on a fresh batch."""
    source_batch, _ = activation_store.next_batch()
    with torch.no_grad():
        output = sae(source_batch)
        reconstruction = output["sae_out"]
        wandb_run.log(
            {
                "performance/mse": (reconstruction - source_batch).pow(2).mean().item(),
                "performance/mae": (reconstruction - source_batch).abs().mean().item(),
                "performance/cosine_similarity": torch.nn.functional.cosine_similarity(
                    reconstruction.flatten(), source_batch.flatten(), dim=0
                ).item(),
                "performance/fvu": output["fvu"].item(),
            },
            step=step,
        )


def train_sae(sae, activation_store, model, cfg):
    num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
    )
    scheduler = create_scheduler(optimizer, cfg, num_batches)
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    if scheduler is not None:
        scheduler.step(0)

    wandb_run = init_wandb(cfg)
    results_manager = CleanResultsManager()
    experiment_dir = results_manager.create_experiment_dir(
        model_name=cfg["model_name"],
        layer=cfg["layer"],
        steps=num_batches,
        description=cfg.get("experiment_description", ""),
        config=cfg,
    )

    sample_collector = None
    if cfg.get("save_activation_samples", False):
        sample_collector = ActivationSampleCollector(
            max_samples_per_feature=cfg.get("max_samples_per_feature", 100),
            context_size=cfg.get("sample_context_size", 20),
            storage_threshold=cfg.get("sample_activation_threshold", 0.1),
        )

    pbar = tqdm.trange(num_batches)
    for i in pbar:
        source_batch, _ = activation_store.next_batch()
        sae_output = sae(source_batch)
        log_wandb(sae_output, i, wandb_run)
        dead_count = log_dead_features(sae, cfg, wandb_run, i)

        if scheduler is not None:
            wandb_run.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=i)

        if i % cfg["perf_log_freq"] == 0:
            log_sae_performance(wandb_run, i, activation_store, sae)

        if i % cfg["checkpoint_freq"] == 0:
            save_training_checkpoint(
                results_manager,
                experiment_dir,
                sae,
                cfg,
                i,
                sae_output,
                dead_count,
                is_final=False,
            )

        if sample_collector and i % cfg.get("sample_collection_freq", 1000) == 0:
            source_acts, batch_tokens = activation_store.get_batch_with_tokens()
            with torch.no_grad():
                sparse_features = sae.encode(source_acts)
            sample_collector.collect_batch_samples(
                sparse_features, batch_tokens, activation_store.tokenizer
            )

        pbar.set_description(
            f"Loss: {sae_output['loss'].item():.3f}, "
            f"L0: {sae_output['l0_norm'].item():.0f}, "
            f"FVU: {sae_output['fvu'].item():.3f}, "
            f"Dead: {dead_count}"
        )

        sae_output["loss"].backward()
        sae.make_decoder_weights_and_grad_unit_norm()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step(i + 1)

    final_dead_mask = sae.num_batches_not_active >= _dead_feature_threshold(cfg)
    final_dead_count = int(final_dead_mask.sum().item())
    save_training_checkpoint(
        results_manager,
        experiment_dir,
        sae,
        cfg,
        num_batches,
        sae_output,
        final_dead_count,
        is_final=True,
    )

    if sample_collector:
        samples_dir = experiment_dir / "activation_samples"
        samples_dir.mkdir(exist_ok=True)
        sample_collector.save_samples(
            str(samples_dir),
            top_k_features=cfg.get("top_features_to_save", 100),
            samples_per_feature=cfg.get("samples_per_feature_to_save", 10),
        )

    wandb_run.finish()
