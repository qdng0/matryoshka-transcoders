"""Transcoder-aware generation-time intervention.

Shadows ``sae_bench.evals.ravel.intervention``. The only behavioural changes
versus the vendor version are:

- source activations are captured at the transcoder's source site
  (``mlp_in`` / ``mlp_out``) rather than always at ``resid_post``,
- the intervention hook is registered on the MLP submodule (for ``mlp_in`` /
  ``mlp_out`` sites) so the transcoder replaces the MLP contribution instead
  of overwriting the whole residual stream,
- the ``resid_post`` fallback preserves the original vendor behaviour so a
  plain residual-stream SAE sanity run still works.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sae_bench.evals.ravel.instance import (
    Prompt,
    RAVELFilteredDataset,
    evaluate_completion,
)

# Local, transcoder-aware MDBM and hook-site helpers. When this module is
# installed under ``sae_bench.evals.ravel.intervention`` via sys.modules the
# relative-style imports below still resolve because
# ``eval/sae_bench/ravel/`` is on sys.path.
from eval.sae_bench.ravel import transcoder_mdbm as _mdbm_mod
from eval.sae_bench.ravel.transcoder_mdbm import (
    _gather_activations,
    _get_decoder_layer,
    _get_mlp_submodule,
    _hook_site_for,
)

# Re-export MDBM under the vendor name for type hints and downstream imports.
MDBM = _mdbm_mod.MDBM


def get_different_attribute_prompt(
    base_prompt: Prompt, source_prompts: list[Prompt]
) -> Prompt:
    different_prompts = [
        p for p in source_prompts if p.attribute_label != base_prompt.attribute_label
    ]
    if not different_prompts:
        raise ValueError(
            f"No prompts with different attribute label found for "
            f"{base_prompt.attribute_label}"
        )
    return random.choice(different_prompts)


def sample_prompts_by_attribute(
    dataset: RAVELFilteredDataset, attribute: str, n_samples: int
):
    all_prompts = dataset.get_prompts_by_attribute(attribute)
    if len(all_prompts) < n_samples:
        print(
            f"Warning: Not enough prompts for attribute {attribute} for intervention. "
            f"Returning {len(all_prompts)} instead of {n_samples} prompts."
        )
        return all_prompts, all_prompts

    selected_prompts = random.sample(all_prompts, n_samples)
    return all_prompts, selected_prompts


def get_prompt_pairs(
    dataset: RAVELFilteredDataset,
    base_attribute: str,
    source_attribute: str,
    n_interventions: int,
):
    all_base_prompts, base_prompts = sample_prompts_by_attribute(
        dataset, base_attribute, n_interventions
    )

    if base_attribute != source_attribute:
        _, source_prompts = sample_prompts_by_attribute(
            dataset, source_attribute, n_interventions
        )
    else:
        all_source_prompts = all_base_prompts
        source_prompts = []
        for p in base_prompts:
            source_prompts.append(get_different_attribute_prompt(p, all_source_prompts))

    min_length = min(len(base_prompts), len(source_prompts))
    return base_prompts[:min_length], source_prompts[:min_length]


@torch.no_grad()
def generate_batched_interventions(
    model: PreTrainedModel,
    mdbm: "MDBM",
    tokenizer: PreTrainedTokenizerBase,
    val_loader: torch.utils.data.DataLoader,
    max_new_tokens: int = 8,
) -> tuple[float, float]:
    iso_scores: list[float] = []
    cause_scores: list[float] = []

    hook_site = _hook_site_for(str(mdbm.sae.cfg.hook_name))

    for batch in tqdm(val_loader, desc="Generating with interventions"):
        (
            base_encoding_BL,
            source_encoding_BL,
            base_pos_B,
            source_pos_B,
            base_pred_B,
            source_pred_B,
            base_text_str,
            base_label_str,
        ) = batch

        source_rep_BD = _gather_activations(
            model,
            mdbm.layer_intervened,
            hook_site,
            source_encoding_BL,
            source_pos_B,
        )

        intervention_hook = mdbm.create_intervention_hook(
            source_rep_BD,
            base_pos_B,
            training_mode=False,
        )

        if hook_site in ("mlp_in", "mlp_out"):
            target_module = _get_mlp_submodule(model, mdbm.layer_intervened)
        else:
            target_module = _get_decoder_layer(model, mdbm.layer_intervened)
        handle = target_module.register_forward_hook(intervention_hook)

        attention_mask = base_encoding_BL.get("attention_mask", None)
        generate_kwargs = dict(
            input_ids=base_encoding_BL["input_ids"].to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask.to(model.device)
        # GPT-2 / Llama / Pythia don't have a pad token by default; following
        # FF-KV, set it to EOS so generate() doesn't warn or misbehave.
        tokenizer_name = getattr(tokenizer, "name_or_path", "")
        if any(t in tokenizer_name.lower() for t in ("llama", "gpt", "pythia")):
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

        try:
            output_ids = model.generate(**generate_kwargs)  # type: ignore
        finally:
            handle.remove()

        generated_ids = output_ids[:, -max_new_tokens:]
        generated_strings = tokenizer.batch_decode(generated_ids)

        for base_text, base_label, generated_string, base_pred, source_pred in zip(
            base_text_str, base_label_str, generated_strings, base_pred_B, source_pred_B
        ):
            if base_pred == source_pred:
                iso_scores.append(
                    evaluate_completion(base_text, base_label, generated_string)
                )
            else:
                cause_scores.append(
                    evaluate_completion(base_text, base_label, generated_string)
                )

    iso_score = float(np.mean(iso_scores)) if iso_scores else 0.0
    cause_score = float(np.mean(cause_scores)) if cause_scores else 0.0
    return iso_score, cause_score


__all__ = [
    "MDBM",
    "get_different_attribute_prompt",
    "sample_prompts_by_attribute",
    "get_prompt_pairs",
    "generate_batched_interventions",
]
