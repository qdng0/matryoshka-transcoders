"""Transcoder-aware SCR/TPP evaluation.

The vanilla SAE Bench SCR code assumes encode(X)->decode(X) operates in the
same vector space (SAE: resid -> resid).  For transcoders the mapping is
source_hook -> target_hook (MLP_in -> MLP_out), so:

  * Probes must be trained on **target** activations (MLP output).
  * Feature effects must use encoder(source) but probe·decoder in target space.
  * Ablation error = target - decode(encode(source)), NOT source - decode(source).

This module provides drop-in replacements for the affected vendor functions.
"""

from __future__ import annotations

import einops
import torch
from tqdm import tqdm

from sae_bench.evals.scr_and_tpp import dataset_creation
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.main import (
    COLUMN2_VALS_LOOKUP,
    get_probe_test_accuracy,
    get_scr_plotting_dict,
    create_tpp_plotting_dict,
    select_top_n_features,
)
from sae_bench.evals.sparse_probing import probe_training
from sae_bench.sae_bench_utils import (
    activation_collection,
    dataset_info,
    dataset_utils,
)


# ---------------------------------------------------------------------------
# Dual-hook activation collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_llm_activations_dual(
    tokens: torch.Tensor,
    model,
    batch_size: int,
    layer: int,
    source_hook: str,
    target_hook: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect activations at both source and target hooks in a single forward pass.

    Returns CPU tensors to avoid OOM when storing both source and target.
    """
    all_source = []
    all_target = []

    for i in tqdm(range(0, len(tokens), batch_size), desc="Collecting dual activations"):
        tokens_BL = tokens[i : i + batch_size]

        _, cache = model.run_with_cache(
            tokens_BL,
            stop_at_layer=layer + 1,
            names_filter=[source_hook, target_hook],
        )
        source_BLD = cache[source_hook]
        target_BLD = cache[target_hook]

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = activation_collection.get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
            source_BLD = source_BLD * attn_mask_BL[:, :, None]
            target_BLD = target_BLD * attn_mask_BL[:, :, None]

        # Store on CPU to avoid OOM (2x activations)
        all_source.append(source_BLD.cpu())
        all_target.append(target_BLD.cpu())

    return torch.cat(all_source, dim=0), torch.cat(all_target, dim=0)


def get_all_llm_activations_dual(
    tokenized_inputs_dict: dict,
    model,
    batch_size: int,
    layer: int,
    source_hook: str,
    target_hook: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collect dual-hook activations for all classes."""
    all_source = {}
    all_target = {}

    for class_name in tokenized_inputs_dict:
        tokens = tokenized_inputs_dict[class_name]["input_ids"]
        src, tgt = get_llm_activations_dual(
            tokens, model, batch_size, layer,
            source_hook, target_hook,
            mask_bos_pad_eos_tokens,
        )
        all_source[class_name] = src
        all_target[class_name] = tgt

    return all_source, all_target


# ---------------------------------------------------------------------------
# Node effects (GPU-aware, handles CPU activations)
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_effects_per_class_transcoder(
    sae,
    probe: probe_training.Probe,
    class_idx: str,
    precomputed_acts: dict[str, torch.Tensor],
    perform_scr: bool,
    sae_batch_size: int,
) -> torch.Tensor:
    """Compute per-feature effects. Handles CPU activations by batching to GPU."""
    device = sae.device
    inputs_train_BLD, labels_train_B = probe_training.prepare_probe_data(
        precomputed_acts, class_idx, perform_scr,
    )
    dtype = inputs_train_BLD.dtype

    running_sum_pos_F = torch.zeros(
        sae.W_dec.data.shape[0], dtype=torch.float32, device=device,
    )
    running_sum_neg_F = torch.zeros(
        sae.W_dec.data.shape[0], dtype=torch.float32, device=device,
    )
    count_pos = 0
    count_neg = 0

    for i in range(0, inputs_train_BLD.shape[0], sae_batch_size):
        activation_batch_BLD = inputs_train_BLD[i : i + sae_batch_size].to(device)
        labels_batch_B = labels_train_B[i : i + sae_batch_size]

        activations_BL = einops.reduce(activation_batch_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum").to(torch.float32)

        f_BLF = sae.encode(activation_batch_BLD)
        f_BLF = f_BLF * nonzero_acts_BL[:, :, None]

        average_sae_acts_BF = (
            einops.reduce(f_BLF, "B L F -> B F", "sum").to(torch.float32)
            / nonzero_acts_B[:, None]
        )

        pos_mask = labels_batch_B == dataset_info.POSITIVE_CLASS_LABEL
        neg_mask = labels_batch_B == dataset_info.NEGATIVE_CLASS_LABEL

        running_sum_pos_F += einops.reduce(
            average_sae_acts_BF[pos_mask], "B F -> F", "sum",
        )
        running_sum_neg_F += einops.reduce(
            average_sae_acts_BF[neg_mask], "B F -> F", "sum",
        )
        count_pos += pos_mask.sum().item()
        count_neg += neg_mask.sum().item()

    average_pos_sae_acts_F = (
        running_sum_pos_F / count_pos if count_pos > 0 else running_sum_pos_F
    )
    average_neg_sae_acts_F = (
        running_sum_neg_F / count_neg if count_neg > 0 else running_sum_neg_F
    )

    average_acts_F = (average_pos_sae_acts_F - average_neg_sae_acts_F).to(dtype)
    probe_weight_D = probe.net.weight.to(dtype=dtype, device=device)
    decoder_weight_DF = sae.W_dec.data.T.to(dtype=dtype, device=device)
    dot_prod_F = (probe_weight_D @ decoder_weight_DF).squeeze()

    if not perform_scr:
        average_acts_F.clamp_(min=0.0)

    effects_F = (average_acts_F * dot_prod_F).to(dtype=torch.float32)
    if perform_scr:
        effects_F = effects_F.abs()

    return effects_F


def get_all_node_effects_transcoder(
    sae,
    probes: dict,
    chosen_class_indices: list[str],
    perform_scr: bool,
    indirect_effect_acts: dict[str, torch.Tensor],
    sae_batch_size: int,
) -> dict[str, torch.Tensor]:
    node_effects = {}
    for ablated_class_idx in chosen_class_indices:
        node_effects[ablated_class_idx] = get_effects_per_class_transcoder(
            sae, probes[ablated_class_idx], ablated_class_idx,
            indirect_effect_acts, perform_scr, sae_batch_size,
        )
    return node_effects


# ---------------------------------------------------------------------------
# Transcoder-aware ablation
# ---------------------------------------------------------------------------
def ablated_precomputed_activations_transcoder(
    source_acts_BLD: torch.Tensor,
    target_acts_BLD: torch.Tensor,
    sae,
    to_ablate: torch.Tensor,
    sae_batch_size: int,
) -> torch.Tensor:
    """Ablation in target (output) space for transcoders.

    error = target - decode(encode(source))         [both in target space]
    modified = decode(encode_ablated(source)) + error

    Activations may be on CPU; batches are moved to GPU for computation.
    """
    device = sae.device
    all_acts_list_BD = []

    for i in range(0, source_acts_BLD.shape[0], sae_batch_size):
        source_batch = source_acts_BLD[i : i + sae_batch_size].to(device)
        target_batch = target_acts_BLD[i : i + sae_batch_size].to(device)
        dtype = target_batch.dtype

        # Mask for attention (zero-padded tokens)
        activations_BL = einops.reduce(target_batch, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        # Encode source, decode to target space
        f_BLF = sae.encode(source_batch)
        x_hat_BLD = sae.decode(f_BLF)

        # Error in TARGET space (not source)
        error_BLD = target_batch - x_hat_BLD

        # Ablate and reconstruct
        f_BLF[..., to_ablate] = 0.0
        modified_acts_BLD = sae.decode(f_BLF) + error_BLD

        # Average over sequence (masked)
        probe_acts_BD = (
            einops.reduce(modified_acts_BLD, "B L D -> B D", "sum")
            / nonzero_acts_B[:, None]
        )
        all_acts_list_BD.append(probe_acts_BD.cpu())

    return torch.cat(all_acts_list_BD, dim=0)


def perform_feature_ablations_transcoder(
    probes: dict,
    sae,
    sae_batch_size: int,
    all_source_test_BLD: dict[str, torch.Tensor],
    all_target_test_BLD: dict[str, torch.Tensor],
    node_effects: dict[str, torch.Tensor],
    top_n_values: list[int],
    chosen_classes: list[str],
    probe_batch_size: int,
    perform_scr: bool,
) -> dict:
    """Feature ablation using dual-hook activations for transcoders."""
    device = sae.device
    ablated_class_accuracies = {}
    for ablated_class_name in chosen_classes:
        ablated_class_accuracies[ablated_class_name] = {}
        for top_n in top_n_values:
            selected_features_F = select_top_n_features(
                node_effects[ablated_class_name], top_n, ablated_class_name,
            )
            test_acts_ablated = {}
            for evaluated_class_name in all_target_test_BLD.keys():
                # ablation returns CPU; move to probe device for accuracy eval
                ablated = ablated_precomputed_activations_transcoder(
                    all_source_test_BLD[evaluated_class_name],
                    all_target_test_BLD[evaluated_class_name],
                    sae,
                    selected_features_F,
                    sae_batch_size,
                )
                test_acts_ablated[evaluated_class_name] = ablated.to(device)

            ablated_class_accuracies[ablated_class_name][top_n] = (
                get_probe_test_accuracy(
                    probes,
                    chosen_classes,
                    test_acts_ablated,
                    probe_batch_size,
                    perform_scr,
                )
            )
    return ablated_class_accuracies


# ---------------------------------------------------------------------------
# Dataset activation collection (dual-hook)
# ---------------------------------------------------------------------------
def get_dataset_activations_transcoder(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
    model,
    llm_batch_size: int,
    layer: int,
    source_hook: str,
    target_hook: str,
    device: str,
    chosen_classes: list[str],
    column1_vals: tuple[str, str] | None = None,
    column2_vals: tuple[str, str] | None = None,
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor],
    dict[str, torch.Tensor], dict[str, torch.Tensor],
]:
    """Collect activations at both source and target hooks.

    Returns (source_train, target_train, source_test, target_test).
    """
    train_data, test_data = dataset_creation.get_train_test_data(
        dataset_name,
        config.perform_scr,
        config.train_set_size,
        config.test_set_size,
        config.random_seed,
        column1_vals,
        column2_vals,
    )

    if not config.perform_scr:
        train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
        test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    train_data = dataset_utils.tokenize_data_dictionary(
        train_data, model.tokenizer, config.context_length, device,
    )
    test_data = dataset_utils.tokenize_data_dictionary(
        test_data, model.tokenizer, config.context_length, device,
    )

    source_train, target_train = get_all_llm_activations_dual(
        train_data, model, llm_batch_size, layer,
        source_hook, target_hook, mask_bos_pad_eos_tokens=True,
    )
    source_test, target_test = get_all_llm_activations_dual(
        test_data, model, llm_batch_size, layer,
        source_hook, target_hook, mask_bos_pad_eos_tokens=True,
    )

    return source_train, target_train, source_test, target_test


# ---------------------------------------------------------------------------
# Top-level eval (replaces run_eval_single_dataset for transcoders)
# ---------------------------------------------------------------------------
def run_eval_single_dataset_transcoder(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
    sae,
    model,
    layer: int,
    source_hook: str,
    target_hook: str,
    device: str | torch.device,
    column1_vals: tuple[str, str] | None = None,
) -> tuple[dict, dict[str, float]]:
    """SCR/TPP evaluation for a single dataset using dual-hook transcoder logic."""
    device_str = str(device)
    column2_vals = COLUMN2_VALS_LOOKUP[dataset_name]

    if not config.perform_scr:
        chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]
    else:
        chosen_classes = list(dataset_info.PAIRED_CLASS_KEYS.keys())

    # Collect dual-hook activations
    source_train, target_train, source_test, target_test = (
        get_dataset_activations_transcoder(
            dataset_name, config, model, config.llm_batch_size,
            layer, source_hook, target_hook, device_str,
            chosen_classes, column1_vals, column2_vals,
        )
    )

    # Train probes on TARGET activations (MLP output space)
    # Move to device for probe training
    meaned_target_train = activation_collection.create_meaned_model_activations(target_train)
    meaned_target_test = activation_collection.create_meaned_model_activations(target_test)
    for k in meaned_target_train:
        meaned_target_train[k] = meaned_target_train[k].to(device_str)
    for k in meaned_target_test:
        meaned_target_test[k] = meaned_target_test[k].to(device_str)

    torch.set_grad_enabled(True)
    llm_probes, _ = probe_training.train_probe_on_activations(
        meaned_target_train,
        meaned_target_test,
        select_top_k=None,
        use_sklearn=False,
        batch_size=config.probe_train_batch_size,
        epochs=config.probe_epochs,
        lr=config.probe_lr,
        perform_scr=config.perform_scr,
        early_stopping_patience=config.early_stopping_patience,
        l1_penalty=config.probe_l1_penalty,
    )
    torch.set_grad_enabled(False)

    # Ensure test acts on same device as probes for accuracy evaluation
    meaned_target_test_gpu = {
        k: v.to(device_str) for k, v in meaned_target_test.items()
    }
    llm_test_accuracies = get_probe_test_accuracy(
        llm_probes, chosen_classes, meaned_target_test_gpu,
        config.probe_test_batch_size, config.perform_scr,
    )

    # Node effects: encode SOURCE acts, use probes trained on TARGET
    # (probe_weight @ W_dec.T works because both are in target space)
    sae_node_effects = get_all_node_effects_transcoder(
        sae, llm_probes, chosen_classes, config.perform_scr,
        source_train,  # SOURCE acts for encoding
        config.sae_batch_size,
    )

    # Ablation with dual-hook error correction
    ablated_class_accuracies = perform_feature_ablations_transcoder(
        llm_probes, sae, config.sae_batch_size,
        source_test, target_test,
        sae_node_effects, config.n_values, chosen_classes,
        config.probe_test_batch_size, config.perform_scr,
    )

    return ablated_class_accuracies, llm_test_accuracies


def run_eval_single_sae_transcoder(
    config: ScrAndTppEvalConfig,
    sae,
    model,
    device: str,
) -> tuple[dict[str, float | dict[str, float]], dict]:
    """Top-level transcoder-aware SCR/TPP eval (replaces run_eval_single_sae)."""
    import random

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    layer = sae.cfg.hook_layer
    source_hook = sae.cfg.input_hook_name
    target_hook = sae.cfg.output_hook_name

    dataset_results = {}
    per_dataset_class_results = {}
    averaging_names = []

    for dataset_name in config.dataset_names:
        if config.perform_scr:
            column1_vals_list = config.column1_vals_lookup[dataset_name]
            for column1_vals in column1_vals_list:
                run_name = f"{dataset_name}_scr_{column1_vals[0]}_{column1_vals[1]}"
                raw_results, llm_clean_accs = run_eval_single_dataset_transcoder(
                    dataset_name, config, sae, model, layer,
                    source_hook, target_hook, device, column1_vals,
                )
                processed_results = get_scr_plotting_dict(raw_results, llm_clean_accs)
                dataset_results[f"{run_name}_results"] = processed_results
                averaging_names.append(run_name)
        else:
            run_name = f"{dataset_name}_tpp"
            raw_results, llm_clean_accs = run_eval_single_dataset_transcoder(
                dataset_name, config, sae, model, layer,
                source_hook, target_hook, device,
            )
            processed_results, per_class_results = create_tpp_plotting_dict(
                raw_results, llm_clean_accs,
            )
            dataset_results[f"{run_name}_results"] = processed_results
            per_dataset_class_results[dataset_name] = per_class_results
            averaging_names.append(run_name)

    from sae_bench.sae_bench_utils import general_utils
    results_dict = general_utils.average_results_dictionaries(
        dataset_results, averaging_names,
    )
    results_dict.update(dataset_results)

    return results_dict, per_dataset_class_results
