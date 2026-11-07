"""Transcoder-aware MDBM module.

Shadows ``sae_bench.evals.ravel.mdbm`` so the vendor ``run_eval_single_sae``
uses a transcoder-aware intervention hook.

The vendor implementation hooks the entire decoder layer at ``hook_resid_post``
and assumes ``encode``/``decode`` live in the same space. For a transcoder
(``mlp_in`` -> ``mlp_out``) that assumption fails in two ways: ``encode`` gets
fed ``resid_post`` (wrong input space, OOD magnitudes) and ``decode`` writes
an ``mlp_out``-sized vector back into the ``resid_post`` slot (wrong output
space, destroys attention + prior residual at that position). This file ports
the transcoder-aware hook from the FF-KV SAEBench fork
(https://github.com/muyo8692/ff-kv-sae, arXiv:2510.22332) and applies a
target-space fix so the real MLP output flows through at every
non-intervention position.

The module name/API matches the vendor ``mdbm`` so it can be installed via
``sys.modules['sae_bench.evals.ravel.mdbm'] = <this module>`` before importing
``sae_bench.evals.ravel.main``.

Branches on ``sae.cfg.hook_name``:
- ``"ln2"``         : transcoder with source ``ln2.hook_normalized`` (mlp_in).
                     Encode mlp_in, decode to an mlp_out estimate, replace the
                     mlp submodule's output only at ``base_pos``.
- ``"mlp_out"``     : SAE trained on raw MLP output. Replace mlp module output
                     only at ``base_pos``.
- anything else     : fall back to the vendor behaviour (hook the full decoder
                     layer, treat it as ``resid_post``). Preserved so a plain
                     residual-stream SAE sanity run still works.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.mdas import MDAS


# ---------------------------------------------------------------------------
# Hook-site helpers (inlined from FF-KV's activation_collection so the shadow
# is self-contained and doesn't require patching the vendor utils module).
# ---------------------------------------------------------------------------
def _get_decoder_layer(model: PreTrainedModel, layer_num: int) -> torch.nn.Module:
    arch = model.config.architectures[0]  # type: ignore
    if arch in ("Gemma2ForCausalLM", "LlamaForCausalLM"):
        return model.model.layers[layer_num]  # type: ignore
    if arch == "GPT2LMHeadModel":
        return model.transformer.h[layer_num]  # type: ignore
    if arch == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer_num]  # type: ignore
    raise ValueError(f"Unsupported architecture: {arch}")


def _get_mlp_submodule(model: PreTrainedModel, layer_num: int) -> torch.nn.Module:
    block = _get_decoder_layer(model, layer_num)
    for name in ("mlp", "feed_forward"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError(
        f"Could not locate MLP submodule on {type(block).__name__}"
    )


def _hook_site_for(hook_name: str) -> str:
    """Return ``'mlp_out'``, ``'mlp_in'``, or ``'resid'`` for a hook name."""
    if "mlp_out" in hook_name:
        return "mlp_out"
    if "ln2" in hook_name or "mlp_in" in hook_name:
        return "mlp_in"
    return "resid"


@torch.no_grad()
def _gather_activations(
    model: PreTrainedModel,
    target_layer: int,
    site: str,
    inputs: BatchEncoding,
    source_pos_B: torch.Tensor,
) -> torch.Tensor:
    """Capture activations at a transcoder hook site for a batch.

    ``site`` must be one of ``'mlp_in'``, ``'mlp_out'``, ``'resid'``.
    Returns a ``(batch, d_model)`` tensor indexed by ``source_pos_B``.
    """
    acts_BLD: torch.Tensor | None = None

    if site == "mlp_in":
        module = _get_mlp_submodule(model, target_layer)

        def _hook(_mod, m_inputs, m_outputs):
            nonlocal acts_BLD
            acts_BLD = m_inputs[0]
            return m_outputs
    elif site == "mlp_out":
        module = _get_mlp_submodule(model, target_layer)

        def _hook(_mod, m_inputs, m_outputs):
            nonlocal acts_BLD
            acts_BLD = m_outputs if not isinstance(m_outputs, tuple) else m_outputs[0]
            return m_outputs
    else:
        module = _get_decoder_layer(model, target_layer)

        def _hook(_mod, m_inputs, m_outputs):
            nonlocal acts_BLD
            acts_BLD = m_outputs[0] if isinstance(m_outputs, tuple) else m_outputs
            return m_outputs

    attention_mask = inputs.get("attention_mask", None)
    handle = module.register_forward_hook(_hook)
    try:
        _ = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=(
                attention_mask.to(model.device)
                if attention_mask is not None
                else None
            ),
        )
    finally:
        handle.remove()

    assert acts_BLD is not None
    batch_range = list(range(acts_BLD.shape[0]))
    return acts_BLD[batch_range, source_pos_B, :].clone()


# ---------------------------------------------------------------------------
# MDBM with a transcoder-aware intervention hook.
# ---------------------------------------------------------------------------
class MDBM(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: RAVELEvalConfig,
        sae: Any,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer_intervened = sae.cfg.hook_layer
        self.hook_site = _hook_site_for(str(sae.cfg.hook_name))
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(sae.cfg.d_sae, device=model.device, dtype=torch.float32),
            requires_grad=True,
        )
        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature: float = 1.0

    # -- hook factory -------------------------------------------------------
    def create_intervention_hook(
        self,
        source_rep_BD: torch.Tensor,
        base_pos_B: torch.Tensor,
        training_mode: bool = False,
        add_error: bool = False,
    ):
        """Build a forward hook that patches transcoder features at ``base_pos``.

        Applies a binary (eval) / sigmoid (train) mask to swap selected
        features from ``source_rep`` into the base prompt at ``base_pos``.
        The crucial correctness property: every token position other than
        ``base_pos`` keeps whatever activation the real model produced. This
        is what makes the intervention "surgical" instead of "layer-wide
        identity + one patched token".
        """

        def intervention_hook(module, inputs, outputs):
            # Resolve "where do I read features from" and "where do I write".
            if self.hook_site == "mlp_out":
                feature_source_BLD = outputs  # raw MLP output
                write_target = outputs
                is_tuple_out = isinstance(outputs, tuple)
                if is_tuple_out:
                    feature_source_BLD = outputs[0]
                    write_target = outputs[0]
            elif self.hook_site == "mlp_in":
                feature_source_BLD = inputs[0]  # mlp_in (pre-MLP layernormed)
                write_target = outputs  # real mlp_out; we only replace base_pos
                is_tuple_out = isinstance(outputs, tuple)
                if is_tuple_out:
                    write_target = outputs[0]
            else:
                # resid_post fallback (vendor semantics)
                assert isinstance(outputs, tuple), (
                    "Expected a tuple output when hooking the full decoder layer"
                )
                feature_source_BLD = outputs[0]
                write_target = outputs[0]
                is_tuple_out = True

            # KV-cache generation path: single-token forward, intervention
            # already applied in the prompt pass.
            if feature_source_BLD.shape[1] == 1:
                return outputs

            batch_range = list(range(feature_source_BLD.shape[0]))

            with torch.no_grad():
                source_act_BF = self.sae.encode(source_rep_BD)
                base_feature_BD = feature_source_BLD[batch_range, base_pos_B, :]
                base_act_BF = self.sae.encode(base_feature_BD)

            if not training_mode:
                mask_values_F = (self.binary_mask > 0).to(
                    dtype=self.binary_mask.dtype
                )
            else:
                mask_values_F = torch.sigmoid(self.binary_mask / self.temperature)

            modified_act_BF = (
                1 - mask_values_F
            ) * base_act_BF + mask_values_F * source_act_BF

            modified_resid_BD = self.sae.decode(
                modified_act_BF.to(dtype=source_rep_BD.dtype)
            )

            if add_error:
                error_BD = base_feature_BD - self.sae.decode(base_act_BF)
                modified_resid_BD = modified_resid_BD + error_BD

            # Target-space fix: start from the real module output and only
            # overwrite at base_pos. In the original FF-KV ``ln2`` branch this
            # started from ``inputs[0]`` (mlp_in) instead, which collapsed the
            # MLP to ~identity at every non-intervention position.
            new_output_BLD = write_target.clone()
            new_output_BLD[batch_range, base_pos_B, :] = modified_resid_BD

            if is_tuple_out:
                return (new_output_BLD, *outputs[1:])
            return new_output_BLD

        return intervention_hook

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        base_encoding_BL,
        source_encoding_BL,
        base_pos_B,
        source_pos_B,
        training_mode: bool = False,
    ):
        with torch.no_grad():
            source_rep_BD = _gather_activations(
                self.model,
                self.layer_intervened,
                self.hook_site,
                source_encoding_BL,
                source_pos_B,
            )

        intervention_hook = self.create_intervention_hook(
            source_rep_BD,
            base_pos_B,
            training_mode,
        )

        if self.hook_site in ("mlp_in", "mlp_out"):
            target_module = _get_mlp_submodule(self.model, self.layer_intervened)
        else:
            target_module = _get_decoder_layer(self.model, self.layer_intervened)

        attention_mask = base_encoding_BL.get("attention_mask", None)
        handle = target_module.register_forward_hook(intervention_hook)
        try:
            logits_BV = self.model(
                input_ids=base_encoding_BL["input_ids"].to(self.model.device),
                attention_mask=(
                    attention_mask.to(self.model.device)
                    if attention_mask is not None
                    else None
                ),
                logits_to_keep=1,
            ).logits[:, -1, :]
        finally:
            handle.remove()

        predicted_B = logits_BV.argmax(dim=-1)
        predicted_text = [self.tokenizer.decode(pid) for pid in predicted_B]

        return logits_BV, predicted_text


# ---------------------------------------------------------------------------
# Loss / metric helpers (unchanged from the vendor / FF-KV version).
# ---------------------------------------------------------------------------
def compute_loss(intervened_logits_BV, target_attr_B):
    loss = F.cross_entropy(intervened_logits_BV, target_attr_B)
    predictions_B = intervened_logits_BV.argmax(dim=-1)
    accuracy = (predictions_B == target_attr_B).float().mean()
    return loss, accuracy


def get_cause_isolation_score_estimates(
    intervened_logits_BV, source_pred_B, base_pred_B
):
    predictions_B = intervened_logits_BV.argmax(dim=-1)

    is_isolation = base_pred_B == source_pred_B
    is_cause = ~is_isolation

    cause_count = is_cause.sum().item()
    isolation_count = is_isolation.sum().item()

    cause_correct = ((predictions_B == source_pred_B) & is_cause).sum().item()
    isolation_correct = (
        (predictions_B == base_pred_B) & is_isolation
    ).sum().item()

    cause_score = cause_correct / cause_count if cause_count > 0 else 0.0
    isolation_score = (
        isolation_correct / isolation_count if isolation_count > 0 else 0.0
    )

    return cause_score, isolation_score, cause_count, isolation_count


@torch.no_grad()
def get_validation_loss(mdbm: "MDBM", val_loader: torch.utils.data.DataLoader):
    mdbm.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_batch_count = 0
    val_cause_score = 0.0
    val_isolation_score = 0.0
    total_cause_count = 0
    total_isolation_count = 0

    for batch in val_loader:
        (
            base_encodings_BL,
            source_encodings_BL,
            base_pos_B,
            source_pos_B,
            base_pred_B,
            source_pred_B,
            _base_text_str,
            _base_label_str,
        ) = batch
        intervened_logits_BV, _ = mdbm(
            base_encodings_BL, source_encodings_BL, base_pos_B, source_pos_B
        )
        loss, accuracy = compute_loss(intervened_logits_BV, source_pred_B)

        cause_score, isolation_score, cause_count, isolation_count = (
            get_cause_isolation_score_estimates(
                intervened_logits_BV, source_pred_B, base_pred_B
            )
        )

        val_loss += loss.item()
        val_accuracy += accuracy.item()
        val_cause_score += cause_score * cause_count
        val_isolation_score += isolation_score * isolation_count
        total_cause_count += cause_count
        total_isolation_count += isolation_count
        val_batch_count += 1

    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
    avg_val_accuracy = val_accuracy / val_batch_count if val_batch_count > 0 else 0.0
    avg_val_cause_score = (
        val_cause_score / total_cause_count if total_cause_count > 0 else 0.0
    )
    avg_val_isolation_score = (
        val_isolation_score / total_isolation_count
        if total_isolation_count > 0
        else 0.0
    )

    return avg_val_loss, avg_val_accuracy, avg_val_cause_score, avg_val_isolation_score


# ---------------------------------------------------------------------------
# Training loop (unchanged from vendor/FF-KV aside from using our MDBM class).
# ---------------------------------------------------------------------------
def train_mdbm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: RAVELEvalConfig,
    sae: Any,
    train_loader,
    val_loader,
    verbose: bool = False,
    train_mdas: bool = False,
) -> "MDBM":
    initial_temperature = 1.0
    final_temperature = 1e-4
    temperature_schedule = torch.logspace(
        torch.log10(torch.tensor(initial_temperature)),
        torch.log10(torch.tensor(final_temperature)),
        config.num_epochs * len(train_loader),
        device=model.device,
        dtype=model.dtype,
    )

    if train_mdas:
        mdbm = MDAS(model, tokenizer, config, sae).to(model.device)
        optimizer = torch.optim.Adam(
            [mdbm.binary_mask, mdbm.transform_matrix], lr=config.learning_rate
        )
        orthonormal = True
    else:
        mdbm = MDBM(model, tokenizer, config, sae).to(model.device)
        optimizer = torch.optim.Adam([mdbm.binary_mask], lr=config.learning_rate)
        orthonormal = False

    if verbose:
        (
            initial_val_loss,
            initial_val_accuracy,
            initial_val_cause_score,
            initial_val_isolation_score,
        ) = get_validation_loss(mdbm, val_loader)  # type: ignore
        print(
            f"Initial validation loss: {initial_val_loss:.4f}, "
            f"accuracy: {initial_val_accuracy:.4f}, "
            f"cause score: {initial_val_cause_score:.4f}, "
            f"isolation score: {initial_val_isolation_score:.4f}"
        )

    for epoch in range(config.num_epochs):
        mdbm.train()
        train_loss = 0.0
        train_accuracy = 0.0
        batch_count = 0
        log_count = 0

        gc.collect()
        torch.cuda.empty_cache()

        for batch in train_loader:
            mdbm.temperature = temperature_schedule[
                epoch * len(train_loader) + batch_count
            ].item()
            (
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                _base_pred_B,
                source_pred_B,
                _base_text_str,
                _base_label_str,
            ) = batch

            optimizer.zero_grad()

            intervened_logits_BV, _ = mdbm(
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                training_mode=True,
            )
            loss, accuracy = compute_loss(intervened_logits_BV, source_pred_B)

            loss.backward()
            optimizer.step()
            if train_mdas and orthonormal:
                with torch.no_grad():
                    Q, _R = torch.linalg.qr(mdbm.transform_matrix, mode="reduced")
                    det = torch.det(Q)
                    if det < 0:
                        Q[:, 0] = -Q[:, 0]
                    mdbm.transform_matrix[...] = Q  # type: ignore

            train_loss += loss.item()
            train_accuracy += accuracy.item()
            batch_count += 1
            log_count += 1

            if log_count % 20 == 0 and verbose:
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"Train Loss: {train_loss / log_count:.4f}, "
                    f"Train Accuracy: {train_accuracy / log_count:.4f}"
                )
                train_loss = 0
                train_accuracy = 0
                log_count = 0

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0.0
        avg_train_accuracy = train_accuracy / batch_count if batch_count > 0 else 0.0

        if verbose:
            (
                avg_val_loss,
                avg_val_accuracy,
                avg_val_cause_score,
                avg_val_isolation_score,
            ) = get_validation_loss(mdbm, val_loader)  # type: ignore
            percent_above_zero = (mdbm.binary_mask > 0).float().mean().item()
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {avg_train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {avg_val_accuracy:.4f}, "
                f"Percent above zero: {percent_above_zero:.4f}, "
                f"Val Cause Score: {avg_val_cause_score:.4f}, "
                f"Val Isolation Score: {avg_val_isolation_score:.4f}"
            )

    return mdbm  # type: ignore


__all__ = [
    "MDBM",
    "compute_loss",
    "get_cause_isolation_score_estimates",
    "get_validation_loss",
    "train_mdbm",
]
