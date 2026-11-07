import copy
import os
import random

import numpy as np
import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer


def set_global_seed(seed):
    """Seed every RNG that affects training, not just torch's CPU generator.

    Without this, two runs with different `cfg["seed"]` only differ in weight
    init and within-buffer DataLoader shuffle: the streaming dataset, numpy,
    and python `random` are unseeded, so the model sees the same activations
    in the same order across seeds. That makes a multi-seed comparison weaker
    than it appears.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_MODEL_CONFIGS = {
    "gemma-2-2b": {"act_size": 2304, "n_layers": 26},
    "meta-llama/Llama-3.2-1B": {"act_size": 2048, "n_layers": 16},
    "meta-llama/Llama-3.2-3B": {"act_size": 3072, "n_layers": 28},
    "meta-llama/Llama-3.1-8B": {"act_size": 4096, "n_layers": 32},
}

_MODEL_CONFIG_DEFAULT = {"act_size": 2304, "n_layers": 26}


def get_model_config(model_name):
    return _MODEL_CONFIGS.get(model_name, _MODEL_CONFIG_DEFAULT)


def get_hook_name(site, layer, model_name):
    model_lower = model_name.lower()
    if "gemma" in model_lower or "llama" in model_lower:
        if site in ("ln2_normalized", "mlp_in"):
            return f"blocks.{layer}.ln2.hook_normalized"
        elif site == "mlp_out":
            return f"blocks.{layer}.hook_mlp_out"
    return utils.get_act_name(site, layer)


def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "model_name": "gemma-2-2b",
        "layer": 8,
        "site": "mlp_in",
        "hook_point": None,
        "act_size": 2304,
        "dict_size": 18432,
        "device": "cuda",
        "dtype": torch.bfloat16,
        "model_dtype": torch.float32,
        "fold_ln": True,

        "lr": 3e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "max_grad_norm": 1.0,

        "scheduler_type": "warmup_decay",
        "warmup_steps": 1000,
        "warmup_start_ratio": 0.1,
        "min_lr_ratio": 0.01,

        "batch_size": 4096,
        "num_tokens": int(1e9),
        "seq_len": 128,
        "model_batch_size": 32,
        "num_batches_in_buffer": 10,
        "dataset_path": "HuggingFaceFW/fineweb-edu",
        "buffer_device": "cpu",
        "dataset_shuffle_buffer": 10_000,
        "activation_cache_path": None,
        "activation_cache_size": 0,
        "activation_cache_refresh_steps": 0,
        "activation_cache_refresh_ratio": 0.0,
        "activation_cache_save_every": 0,
        "prepend_bos": False,

        "sae_type": "topk",
        "sae_mode": False,
        "trainer_class": "",
        "k": 96,
        "group_fractions": [0.125, 0.125, 0.25, 0.5],
        "dead_steps_threshold": 20,
        "threshold_beta": 0.99,
        "top_k": 96,
        "top_k_aux": 512,
        "aux_penalty": 1/32,
        "ghost_grad_penalty": 0.0,
        "ghost_grad_target_norm_ratio": 0.5,
        "ghost_grad_max_pre": 10.0,
        "ghost_grad_rescale_loss": True,
        "n_batches_to_dead": 20,
        "dead_feature_threshold": 1e-8, 
        "jumprelu_threshold_init": 1e-3,
        "jumprelu_kde_bandwidth": 1e-3,
        "act_l0_warmup_steps": 10_000,
        "wandb_project": "matryoshka_transcoder",
        "perf_log_freq": 1000,
        "checkpoint_freq": 10000,

        "save_activation_samples": False,
        "sample_collection_freq": 1000,
        "max_samples_per_feature": 100,
        "sample_context_size": 20,
        "sample_activation_threshold": 0.1,
        "top_features_to_save": 100,
        "samples_per_feature_to_save": 10,
    }
    return post_init_cfg(default_cfg)


def post_init_cfg(cfg):
    model_config = get_model_config(cfg["model_name"])
    cfg["act_size"] = model_config["act_size"]
    if cfg.get("sae_mode", False):
        k = int(cfg.get("k", cfg.get("top_k", 0)))
        cfg["k"] = k
        cfg["top_k"] = k
        hook_point = cfg.get("hook_point") or cfg.get("source_hook_point", "sae")
        trainer_name = cfg.get("trainer_class") or cfg.get("sae_type", "sae")
        cfg["name"] = (
            f"{cfg['model_name']}_{hook_point}_{cfg['dict_size']}_{trainer_name}_{k}_{cfg['lr']}"
        )
    else:
        cfg["name"] = (
            f"{cfg['model_name']}_{cfg.get('source_hook_point', 'sae')}_"
            f"{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
        )
    return cfg


def create_transcoder_config(
    base_cfg, 
    source_layer, 
    target_layer, 
    source_site="mlp_in", 
    target_site="mlp_out",
    num_tokens=None,
    context_size=None,
    description=None
):
    """Create transcoder config with optional eval-specific overrides."""
    cfg = copy.deepcopy(base_cfg)
    
    if num_tokens is not None:
        cfg["num_tokens"] = int(num_tokens)
    if context_size is not None:
        cfg["seq_len"] = int(context_size)
    if description is not None:
        cfg["experiment_description"] = description
    
    cfg.setdefault("model_batch_size", 32)
    cfg.setdefault("num_batches_in_buffer", 4)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("dtype", torch.bfloat16)
    
    model_name = cfg.get("model_name", "gemma-2-2b")
    cfg["source_layer"] = source_layer
    cfg["target_layer"] = target_layer
    cfg["layer"] = source_layer
    cfg["source_site"] = source_site
    cfg["target_site"] = target_site
    cfg["source_hook_point"] = get_hook_name(source_site, source_layer, model_name)
    cfg["target_hook_point"] = get_hook_name(target_site, target_layer, model_name)
    return cfg


def create_sae_config(
    base_cfg,
    layer,
    site="resid_post",
    num_tokens=None,
    context_size=None,
    description=None,
):
    """Create SAE config for single-hook autoencoding runs."""
    cfg = copy.deepcopy(base_cfg)

    if num_tokens is not None:
        cfg["num_tokens"] = int(num_tokens)
    if context_size is not None:
        cfg["seq_len"] = int(context_size)
    if description is not None:
        cfg["experiment_description"] = description

    cfg.setdefault("model_batch_size", 32)
    cfg.setdefault("num_batches_in_buffer", 4)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("dtype", torch.bfloat16)

    model_name = cfg.get("model_name", "gemma-2-2b")
    hook_point = get_hook_name(site, layer, model_name)

    k = int(cfg.get("k", cfg.get("top_k", 96)))
    cfg["sae_mode"] = True
    cfg["k"] = k
    cfg["top_k"] = k
    cfg["layer"] = layer
    cfg["site"] = site
    cfg["source_layer"] = layer
    cfg["target_layer"] = layer
    cfg["source_site"] = site
    cfg["target_site"] = site
    cfg["hook_point"] = hook_point
    cfg["source_hook_point"] = hook_point
    cfg["target_hook_point"] = hook_point

    return cfg


def load_hooked_transformer(cfg):
    """Load a HookedTransformer with configurable weight folding."""
    model_name = cfg["model_name"]
    dtype = cfg.get("model_dtype", cfg.get("dtype"))
    use_from_pretrained_no_processing = bool(
        cfg.get("use_from_pretrained_no_processing", False)
    )
    model_from_pretrained_kwargs = cfg.get("model_from_pretrained_kwargs") or {}
    fold_ln = cfg.get("fold_ln")
    if fold_ln is None:
        fold_ln = "gemma" in model_name.lower()

    device = cfg.get("device")
    if use_from_pretrained_no_processing:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name=model_name,
            device=device,
            dtype=dtype,
            **model_from_pretrained_kwargs,
        )
        return model
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        fold_ln=fold_ln,
        device=device,
    )
    return model


def compute_fvu(x_original, x_reconstruct, eps=1e-8):
    x = x_original.float()
    x_hat = x_reconstruct.float()

    # Ensure shape [num_samples, num_features]
    x = x.reshape(-1, x.shape[-1])
    x_hat = x_hat.reshape(-1, x_hat.shape[-1])

    residual_mse = (x - x_hat).pow(2).mean()
    centered = x - x.mean(dim=0, keepdim=True)
    total_variance = centered.pow(2).mean()

    return residual_mse / (total_variance + eps)
