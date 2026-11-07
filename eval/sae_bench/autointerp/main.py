"""Autointerp evaluation for transcoders.

Adapted from SAE Bench's autointerp evaluation. Self-contained implementation
that does not depend on sae_bench_vendor or sae_lens.

Usage:
    python eval/sae_bench/autointerp/main.py \
        --config_path path/to/config.json \
        --model_name gemma-2-2b

    python eval/sae_bench/autointerp/main.py \
        --google_params_path path/to/params.npz \
        --model_name gemma-2-2b \
        --layer 12
"""

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

import einops
import torch
from datasets import load_dataset
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sae_bench.common.transcoder_adapter import (  # noqa: E402
    TranscoderAdapter,
    load_google_transcoder_adapter,
    load_transcoder_adapter,
)

from sae_bench.autointerp.eval_config import AutoInterpEvalConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_NAME_TO_BATCH_SIZE = {
    "pythia-70m-deduped": 512,
    "pythia-160m-deduped": 256,
    "gemma-2-2b": 32,
    "gemma-2-9b": 32,
    "gemma-2-2b-it": 32,
    "gemma-2-9b-it": 32,
}

LLM_NAME_TO_DTYPE = {
    "pythia-70m-deduped": "float32",
    "pythia-160m-deduped": "float32",
    "gemma-2-2b": "bfloat16",
    "gemma-2-2b-it": "bfloat16",
    "gemma-2-9b": "bfloat16",
    "gemma-2-9b-it": "bfloat16",
}

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


def get_autointerp_max_workers() -> int:
    return max(1, int(os.getenv("LOCAL_AUTOINTERP_MAX_WORKERS", "2")))


def get_autointerp_api_max_retries() -> int:
    return max(0, int(os.getenv("LOCAL_AUTOINTERP_API_MAX_RETRIES", "20")))


def get_autointerp_api_backoff_base_seconds() -> float:
    return max(
        0.1, float(os.getenv("LOCAL_AUTOINTERP_API_BACKOFF_BASE_SECONDS", "1.0"))
    )


def get_autointerp_api_timeout_seconds() -> float:
    return max(5.0, float(os.getenv("LOCAL_AUTOINTERP_API_TIMEOUT_SECONDS", "60.0")))


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def str_to_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def setup_environment() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    return device


# ---------------------------------------------------------------------------
# Indexing utilities (from SAE Bench indexing_utils)
# ---------------------------------------------------------------------------
def get_k_largest_indices(
    x: Tensor, k: int, buffer: int = 0, no_overlap: bool = False
) -> Tensor:
    x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if (row, col) not in seen:
                unique_indices.append((row, col))
                for offset in range(-buffer, buffer + 1):
                    seen.add((row, col + offset))
            if len(unique_indices) == k:
                break
        rows, cols = torch.tensor(
            unique_indices, dtype=torch.int64, device=x.device
        ).unbind(dim=-1)

    return torch.stack((rows, cols), dim=1)[:k]


def get_iw_sample_indices(
    x: Tensor, k: int, buffer: int = 0, use_squared_values: bool = True
) -> Tensor:
    x = x[:, buffer:-buffer]
    if use_squared_values:
        x = x.pow(2)
    probabilities = x.flatten() / x.sum()
    indices = torch.multinomial(probabilities, k, replacement=False)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(x: Tensor, indices: Tensor, buffer: int = 0) -> Tensor:
    rows, cols = indices.unbind(dim=-1)
    rows = einops.repeat(rows, "k -> k buf", buf=buffer * 2 + 1)
    cols = einops.repeat(cols, "k -> k buf", buf=buffer * 2 + 1) + torch.arange(
        -buffer, buffer + 1, device=cols.device
    )
    return x[rows, cols]


# ---------------------------------------------------------------------------
# Dataset utilities (simplified from SAE Bench dataset_utils)
# ---------------------------------------------------------------------------
def load_and_tokenize_dataset(
    dataset_name: str, ctx_len: int, num_tokens: int, tokenizer: Any
) -> Tensor:
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    total_chars = 0
    texts: list[str] = []
    for row in dataset:
        text = row["text"]  # type: ignore
        if len(text) > 100:
            texts.append(text)
            total_chars += len(text)
            if total_chars > num_tokens * 5:
                break

    full_text = tokenizer.eos_token.join(texts)
    n_chunks = 20
    chunk_len = (len(full_text) - 1) // n_chunks + 1
    chunks = [full_text[i * chunk_len : (i + 1) * chunk_len] for i in range(n_chunks)]
    tokens = tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()
    tokens = tokens[tokens != tokenizer.pad_token_id]
    tokens = tokens[: num_tokens + ctx_len + 1]

    n_batches = len(tokens) // ctx_len
    tokens = tokens[: n_batches * ctx_len]
    tokens = einops.rearrange(tokens, "(b s) -> b s", b=n_batches, s=ctx_len)
    tokens[:, 0] = tokenizer.bos_token_id
    return tokens


# ---------------------------------------------------------------------------
# Activation collection for transcoders
# ---------------------------------------------------------------------------
@torch.no_grad()
def _get_bos_pad_eos_mask(tokens: Tensor, tokenizer: Any) -> Tensor:
    mask = (
        (tokens == tokenizer.pad_token_id)
        | (tokens == tokenizer.eos_token_id)
        | (tokens == tokenizer.bos_token_id)
    )
    return ~mask


@torch.no_grad()
def collect_transcoder_activations(
    tokens: Tensor,
    model: HookedTransformer,
    transcoder: TranscoderAdapter,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: list[int] | None = None,
    activation_dtype: torch.dtype | None = None,
) -> Tensor:
    all_acts = []
    for i in tqdm(
        range(0, tokens.shape[0], batch_size), desc="Collecting transcoder activations"
    ):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD = cache[hook_name]
        acts_BLF = transcoder.encode(resid_BLD)

        if selected_latents is not None:
            acts_BLF = acts_BLF[:, :, selected_latents]

        if mask_bos_pad_eos_tokens:
            attn_mask = _get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask = torch.ones_like(tokens_BL, dtype=torch.bool)
        acts_BLF = acts_BLF * attn_mask.to(device=acts_BLF.device)[:, :, None]

        if activation_dtype is not None:
            acts_BLF = acts_BLF.to(dtype=activation_dtype)
        all_acts.append(acts_BLF)

    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def get_feature_activation_sparsity(
    tokens: Tensor,
    model: HookedTransformer,
    transcoder: TranscoderAdapter,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> Tensor:
    device = transcoder.device
    running_sum = torch.zeros(
        transcoder.W_dec.shape[0], dtype=torch.float32, device=device
    )
    total_tokens = 0

    for i in tqdm(
        range(0, tokens.shape[0], batch_size), desc="Computing feature sparsity"
    ):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD = cache[hook_name]
        acts_BLF = transcoder.encode(resid_BLD)
        acts_BLF = (acts_BLF > 0).to(dtype=torch.float32)

        if mask_bos_pad_eos_tokens:
            attn_mask = _get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask = torch.ones_like(tokens_BL, dtype=torch.bool)
        attn_mask = attn_mask.to(device=acts_BLF.device)
        acts_BLF = acts_BLF * attn_mask[:, :, None]
        total_tokens += attn_mask.sum().item()
        running_sum += einops.reduce(acts_BLF, "B L F -> F", "sum")

    return running_sum / total_tokens


# ---------------------------------------------------------------------------
# AutoInterp core classes
# ---------------------------------------------------------------------------
class Example:
    """Data for a single example sequence."""

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model: HookedTransformer,
    ):
        self.toks = toks
        self.str_toks = model.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)
            )
            .replace("�", "")
            .replace("\n", "↵")
        )


class Examples:
    """Collection of example sequences with display and shuffling."""

    def __init__(self, examples: list[Example], shuffle: bool = False):
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(
                self.examples, key=lambda x: max(x.acts), reverse=True
            )

    def display(self, predictions: list[int] | None = None) -> str:
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        "Y" if ex.is_active else "",
                        "Y" if i + 1 in predictions else "",
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class AutoInterp:
    """Generates explanations and scores for transcoder latents using an LLM judge."""

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: HookedTransformer,
        transcoder: TranscoderAdapter,
        tokenized_dataset: Tensor,
        sparsity: Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.transcoder = transcoder
        self.tokenized_dataset = tokenized_dataset
        self.device = device
        self.api_key = api_key
        self.api_client = OpenAI(
            api_key=self.api_key,
            max_retries=0,
            timeout=get_autointerp_api_timeout_seconds(),
        )

        if cfg.override_latents is not None:
            self.latents = cfg.override_latents
        else:
            sparsity_counts = sparsity * cfg.total_tokens
            alive_latents = (
                torch.nonzero(sparsity_counts > cfg.dead_latent_threshold)
                .squeeze(1)
                .tolist()
            )
            if len(alive_latents) < cfg.n_latents:
                self.latents = alive_latents
                print(
                    f"\nWARNING: Found only {len(alive_latents)} alive latents, "
                    f"less than {cfg.n_latents}\n"
                )
            else:
                self.latents = random.sample(alive_latents, k=cfg.n_latents)
        self.n_latents = len(self.latents)

    async def run(
        self, explanations_override: dict[int, str] | None = None
    ) -> dict[int, dict[str, Any]]:
        if explanations_override is None:
            explanations_override = {}

        generation_examples, scoring_examples = self.gather_data()
        latents_with_data = sorted(generation_examples.keys())
        n_dead = self.n_latents - len(latents_with_data)
        if n_dead > 0:
            print(
                f"Found data for {len(latents_with_data)}/{self.n_latents} "
                f"alive latents; {n_dead} dead"
            )

        max_workers = get_autointerp_max_workers()
        semaphore = asyncio.Semaphore(max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            async def run_bounded_single_feature(latent: int):
                async with semaphore:
                    return await self.run_single_feature(
                        executor,
                        latent,
                        generation_examples[latent],
                        scoring_examples[latent],
                        explanations_override.get(latent, None),
                    )

            tasks = [run_bounded_single_feature(latent) for latent in latents_with_data]
            results: dict[int, dict[str, Any]] = {}
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Calling API (gen & scoring)",
            ):
                result = await future
                if result:
                    results[result["latent"]] = result

        return results

    async def run_single_feature(
        self,
        executor: ThreadPoolExecutor,
        latent: int,
        generation_examples: Examples,
        scoring_examples: Examples,
        explanation_override: str | None = None,
    ) -> dict[str, Any] | None:
        gen_prompts = self.get_generation_prompts(generation_examples)
        (explanation_raw,), logs = await asyncio.get_event_loop().run_in_executor(
            executor,
            self.get_api_response,
            gen_prompts,
            self.cfg.max_tokens_in_explanation,
        )
        explanation = explanation_raw.split("activates on")[-1].rstrip(".").strip()
        results: dict[str, Any] = {
            "latent": latent,
            "explanation": explanation,
            "logs": f"Generation phase\n{logs}\n{generation_examples.display()}",
        }

        if self.cfg.scoring:
            scoring_prompts = self.get_scoring_prompts(
                explanation=explanation_override or explanation,
                scoring_examples=scoring_examples,
            )
            (predictions_raw,), logs = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.get_api_response,
                scoring_prompts,
                self.cfg.max_tokens_in_prediction,
            )
            predictions = self._parse_predictions(predictions_raw)
            if predictions is None:
                return None
            score = self._score_predictions(predictions, scoring_examples)
            results |= {
                "predictions": predictions,
                "correct seqs": [
                    i
                    for i, ex in enumerate(scoring_examples, start=1)
                    if ex.is_active
                ],
                "score": score,
                "logs": results["logs"]
                + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
            }

        return results

    @staticmethod
    def _parse_predictions(predictions: str) -> list[int] | None:
        parts = (
            predictions.strip()
            .rstrip(".")
            .replace("and", ",")
            .replace("None", "")
            .split(",")
        )
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return []
        if not all(p.isdigit() for p in parts):
            return None
        return [int(p) for p in parts]

    @staticmethod
    def _score_predictions(
        predictions: list[int], scoring_examples: Examples
    ) -> float:
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct = [ex.is_active for ex in scoring_examples]
        return sum(c == cc for c, cc in zip(classifications, correct)) / len(
            classifications
        )

    def get_api_response(
        self, messages: Messages, max_tokens: int, n_completions: int = 1
    ) -> tuple[list[str], str]:
        max_retries = get_autointerp_api_max_retries()
        base_backoff_seconds = get_autointerp_api_backoff_base_seconds()

        for attempt in range(max_retries + 1):
            try:
                result = self.api_client.chat.completions.create(
                    model=self.cfg.openai_model,
                    messages=messages,  # type: ignore
                    n=n_completions,
                    max_tokens=max_tokens,
                    stream=False,
                )
                response = [
                    choice.message.content.strip() for choice in result.choices
                ]  # type: ignore
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
                if attempt >= max_retries:
                    raise
                sleep_seconds = min(base_backoff_seconds * (2**attempt), 60.0)
                sleep_seconds += random.uniform(0.0, 0.5)
                print(
                    f"OpenAI request failed, retrying in {sleep_seconds:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_seconds)
        logs = tabulate(
            [
                m.values()
                for m in messages + [{"role": "assistant", "content": response[0]}]
            ],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )
        return response, logs

    def get_generation_prompts(self, generation_examples: Examples) -> Messages:
        assert len(generation_examples) > 0, "No generation examples found"

        examples_as_str = "\n".join(
            f"{i + 1}. {ex.to_str(mark_toks=True)}"
            for i, ex in enumerate(generation_examples)
        )

        system = (
            "We're studying neurons in a neural network. Each neuron activates on "
            "some particular word/words/substring/concept in a short document. The "
            "activating words in each document are indicated with << ... >>. We will "
            "give you a list of documents on which the neuron activates, in order from "
            "most strongly activating to least strongly activating. Look at the parts "
            "of the document the neuron activates for and summarize in a single sentence "
            "what the neuron is activating on. Try not to be overly specific in your "
            "explanation. Note that some neurons will activate only on specific words or "
            "substrings, but others will activate on most/all words in a sentence "
            "provided that sentence contains some particular concept. Your explanation "
            "should cover most or all activating words (for example, don't give an "
            "explanation which is specific to a single word if all words in a sentence "
            "cause the neuron to activate). Pay attention to things like the "
            "capitalization and punctuation of the activating words or concepts, if that "
            "seems relevant. Keep the explanation as short and simple as possible, "
            "limited to 20 words or less. Omit punctuation and formatting. You should "
            "avoid giving long lists of words."
        )
        if self.cfg.use_demos_in_explanation:
            system += (
                ' Some examples: "This neuron activates on the word \'knows\' in '
                'rhetorical questions", and "This neuron activates on verbs related '
                'to decision-making and preferences", and "This neuron activates on '
                'the substring \'Ent\' at the start of words", and "This neuron '
                'activates on text about government economic policy".'
            )
        else:
            system += (
                ' Your response should be in the form "This neuron activates on...".'
            )

        user = f"The activating documents are given below:\n\n{examples_as_str}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def get_scoring_prompts(
        self, explanation: str, scoring_examples: Examples
    ) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join(
            f"{i + 1}. {ex.to_str(mark_toks=False)}"
            for i, ex in enumerate(scoring_examples)
        )

        example_response = sorted(
            random.sample(
                range(1, 1 + self.cfg.n_ex_for_scoring),
                k=self.cfg.n_correct_for_scoring,
            )
        )
        example_response_str = ", ".join(str(i) for i in example_response)

        system = (
            f"We're studying neurons in a neural network. Each neuron activates on "
            f"some particular word/words/substring/concept in a short document. You "
            f"will be given a short explanation of what this neuron activates for, and "
            f"then be shown {self.cfg.n_ex_for_scoring} example sequences in random "
            f"order. You will have to return a comma-separated list of the examples "
            f"where you think the neuron should activate at least once, on ANY of the "
            f"words or substrings in the document. For example, your response might "
            f'look like "{example_response_str}". Try not to be overly specific in '
            f"your interpretation of the explanation. If you think there are no "
            f"examples where the neuron will activate, you should just respond with "
            f'"None". You should include nothing else in your response other than '
            f'comma-separated numbers or the word "None" - this is important.'
        )
        user = (
            f"Here is the explanation: this neuron fires on {explanation}.\n\n"
            f"Here are the examples:\n\n{examples_as_str}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def gather_data(self) -> tuple[dict[int, Examples], dict[int, Examples]]:
        dataset_size, seq_len = self.tokenized_dataset.shape
        layer = self.transcoder.cfg.hook_layer
        hook_name = self.transcoder.cfg.source_hook_point

        acts = collect_transcoder_activations(
            self.tokenized_dataset,
            self.model,
            self.transcoder,
            self.cfg.llm_batch_size,
            layer,
            hook_name,
            mask_bos_pad_eos_tokens=True,
            selected_latents=self.latents,
            activation_dtype=torch.bfloat16,
        )

        generation_examples: dict[int, Examples] = {}
        scoring_examples: dict[int, Examples] = {}

        for i, latent in tqdm(
            enumerate(self.latents), desc="Collecting examples for LLM judge"
        ):
            # (1/3) Random examples (no activation values needed)
            rand_indices = torch.stack(
                [
                    torch.randint(
                        0, dataset_size, (self.cfg.n_random_ex_for_scoring,)
                    ),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer
            )

            # (2/3) Top-activating examples
            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=self.cfg.buffer
            )
            top_values = index_with_buffer(
                acts[..., i], top_indices, buffer=self.cfg.buffer
            )
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Importance-weighted examples (disjoint from top)
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(
                acts[..., i] >= threshold, 0.0, acts[..., i]
            )
            if (
                acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6
            ):
                continue

            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
            )
            iw_toks = index_with_buffer(
                self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer
            )
            iw_values = index_with_buffer(
                acts[..., i], iw_indices, buffer=self.cfg.buffer
            )

            # Split top and IW examples between generation and scoring
            rand_top = torch.randperm(self.cfg.n_top_ex)
            top_gen = rand_top[: self.cfg.n_top_ex_for_generation]
            top_score = rand_top[self.cfg.n_top_ex_for_generation :]
            rand_iw = torch.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen = rand_iw[: self.cfg.n_iw_sampled_ex_for_generation]
            iw_score = rand_iw[self.cfg.n_iw_sampled_ex_for_generation :]

            def _make(
                toks: Tensor, act_vals: Tensor | None = None
            ) -> list[Example]:
                if act_vals is None:
                    act_vals = torch.zeros_like(toks).float()
                return [
                    Example(
                        toks=t,
                        acts=a,
                        act_threshold=act_threshold,
                        model=self.model,
                    )
                    for t, a in zip(toks.tolist(), act_vals.tolist())
                ]

            generation_examples[latent] = Examples(
                _make(top_toks[top_gen], top_values[top_gen])
                + _make(iw_toks[iw_gen], iw_values[iw_gen]),
            )
            scoring_examples[latent] = Examples(
                _make(top_toks[top_score], top_values[top_score])
                + _make(iw_toks[iw_score], iw_values[iw_score])
                + _make(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------
def run_eval_single_transcoder(
    config: AutoInterpEvalConfig,
    transcoder: TranscoderAdapter,
    model: HookedTransformer,
    device: str,
    artifacts_folder: str,
    api_key: str,
) -> dict[int, dict[str, Any]]:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    model_tag = config.model_name.replace("/", "_")
    tokens_filename = (
        f"{model_tag}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    )
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = torch.load(tokens_path).to(device)
    else:
        tokenized_dataset = load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            model.tokenizer,
        ).to(device)
        torch.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset: {tokenized_dataset.shape}")

    layer = transcoder.cfg.hook_layer
    hook_name = transcoder.cfg.source_hook_point

    sparsity = get_feature_activation_sparsity(
        tokenized_dataset,
        model,
        transcoder,
        config.llm_batch_size,
        layer,
        hook_name,
        mask_bos_pad_eos_tokens=True,
    )

    autointerp = AutoInterp(
        cfg=config,
        model=model,
        transcoder=transcoder,
        tokenized_dataset=tokenized_dataset,
        sparsity=sparsity,
        api_key=api_key,
        device=device,
    )
    return asyncio.run(autointerp.run())


def run_eval(
    config: AutoInterpEvalConfig,
    selected_transcoders: list[tuple[str, TranscoderAdapter]],
    device: str,
    api_key: str,
    output_path: str,
    force_rerun: bool = False,
    save_logs_path: str | None = None,
    artifacts_path: str = "artifacts",
) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    os.makedirs(output_path, exist_ok=True)
    results_dict: dict[str, Any] = {}

    llm_dtype = str_to_dtype(config.llm_dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for transcoder_name, transcoder in tqdm(
        selected_transcoders, desc="Running autointerp evaluation"
    ):
        transcoder = transcoder.to(device=device, dtype=llm_dtype)

        model_tag = config.model_name.replace("/", "_")
        result_filename = (
            f"{transcoder_name}_{model_tag}_l{transcoder.cfg.hook_layer}"
            f"_autointerp.json"
        )
        sae_result_path = os.path.join(output_path, result_filename)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {transcoder_name} as results already exist")
            continue

        artifacts_folder = os.path.join(artifacts_path, "autointerp")

        per_latent = run_eval_single_transcoder(
            config, transcoder, model, device, artifacts_folder, api_key
        )

        if save_logs_path is not None and per_latent:
            headers = [
                "latent",
                "explanation",
                "predictions",
                "correct seqs",
                "score",
            ]
            rows = [
                [per_latent[lat].get(h) for h in headers] for lat in per_latent
            ]
            logs = (
                f"Summary table:\n"
                f"{tabulate(rows, headers=headers, tablefmt='simple_outline')}"
            )
            worst = min(
                per_latent.values(),
                key=lambda x: x.get("score", float("inf")),
            )
            best = max(
                per_latent.values(),
                key=lambda x: x.get("score", float("-inf")),
            )
            logs += (
                f"\n\nWorst: idx {worst['latent']}, "
                f"score={worst.get('score')}\n{worst.get('logs', '')}"
            )
            logs += (
                f"\n\nBest: idx {best['latent']}, "
                f"score={best.get('score')}\n{best.get('logs', '')}"
            )
            with open(save_logs_path, "a") as f:
                f.write(logs)

        all_scores = [r["score"] for r in per_latent.values() if "score" in r]
        if not all_scores:
            print(f"WARNING: No valid scores for {transcoder_name}")
            continue

        scores_t = torch.tensor(all_scores)
        config_dict = {f.name: getattr(config, f.name) for f in dc_fields(config)}

        eval_output = {
            "eval_type": "autointerp",
            "eval_config": config_dict,
            "eval_id": str(uuid.uuid4()),
            "datetime": datetime.now().isoformat(),
            "transcoder_name": transcoder_name,
            "transcoder_cfg": transcoder.cfg.to_dict(),
            "metrics": {
                "autointerp_score": scores_t.mean().item(),
                "autointerp_std_dev": scores_t.std().item(),
                "n_latents_scored": len(all_scores),
            },
            "per_latent_results": {
                str(k): {key: val for key, val in v.items() if key != "logs"}
                for k, v in per_latent.items()
            },
        }

        with open(sae_result_path, "w") as f:
            json.dump(eval_output, f, indent=2, default=str)
        print(f"Results saved to {sae_result_path}")
        print(
            f"  Score: {scores_t.mean().item():.4f} +/- {scores_t.std().item():.4f}"
        )

        results_dict[transcoder_name] = eval_output

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def arg_parser():
    parser = argparse.ArgumentParser(
        description="Run autointerp evaluation for transcoders"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to transcoder training config (json)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to transcoder checkpoint (.pt). If omitted, uses last "
        "checkpoint in config directory.",
    )
    parser.add_argument(
        "--google_params_path",
        type=str,
        default=None,
        help="Path to Google Gemma Scope params.npz",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Hook layer. If omitted, inferred from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. 'cuda', 'cuda:1', 'cpu'). If omitted, uses CUDA when available.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/evals/autointerp",
        help="Output folder",
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="artifacts",
        help="Path for cached artifacts (tokenized dataset, etc.)",
    )
    parser.add_argument(
        "--api_key_path",
        type=str,
        default="openai_api_key.txt",
        help="Path to file containing OpenAI API key",
    )
    parser.add_argument(
        "--api_key_env_var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing OpenAI API key if --api_key_path is absent.",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4.1-nano",
        help="OpenAI model for explanations and scoring",
    )
    parser.add_argument(
        "--n_latents",
        type=int,
        default=1000,
        help="Number of latents to interpret",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=2_000_000,
        help="Total tokens for dataset",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="LLM batch size",
    )
    parser.add_argument(
        "--llm_context_size",
        type=int,
        default=None,
        help="Context length per sequence (default: 128). Use 64 to reduce memory.",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM",
    )
    parser.add_argument(
        "--random_seed", type=int, default=68, help="Random seed"
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--save_logs_path",
        type=str,
        default=None,
        help="Path to append human-readable logs",
    )
    return parser


def create_config_and_transcoders(
    args, device: str
) -> tuple[AutoInterpEvalConfig, list[tuple[str, TranscoderAdapter]], str]:
    config = AutoInterpEvalConfig(
        model_name=args.model_name,
        random_seed=args.random_seed,
        openai_model=args.openai_model,
        n_latents=args.n_latents,
        total_tokens=args.total_tokens,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = LLM_NAME_TO_BATCH_SIZE.get(config.model_name, 32)

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = LLM_NAME_TO_DTYPE.get(config.model_name, "bfloat16")

    if args.llm_context_size is not None:
        config.llm_context_size = args.llm_context_size

    selected_transcoders: list[tuple[str, TranscoderAdapter]] = []
    if args.google_params_path:
        if args.layer is None:
            raise ValueError(
                "Please provide --layer when using --google_params_path."
            )
        name, adapter = load_google_transcoder_adapter(
            args.google_params_path,
            layer=args.layer,
            device=device,
            dtype=str_to_dtype(config.llm_dtype),
        )
        selected_transcoders.append((name, adapter))
        config.config_path = str(Path(args.google_params_path).resolve())
        config.layer = args.layer
    else:
        if args.config_path is None:
            raise ValueError("Provide --config_path or --google_params_path.")
        name, adapter = load_transcoder_adapter(
            args.config_path,
            args.checkpoint_path,
            device=device,
            dtype=str_to_dtype(config.llm_dtype),
        )
        selected_transcoders.append((name, adapter))
        config.config_path = str(Path(args.config_path).resolve())

    if args.layer is not None:
        for _, adapter in selected_transcoders:
            adapter.cfg.hook_layer = args.layer
        config.layer = args.layer
    elif config.layer == -1:
        config.layer = selected_transcoders[0][1].cfg.hook_layer

    config.source_hook_point = selected_transcoders[0][1].cfg.source_hook_point
    config.target_hook_point = selected_transcoders[0][1].cfg.target_hook_point

    if os.path.exists(args.api_key_path):
        with open(args.api_key_path) as f:
            api_key = f.read().strip()
    else:
        api_key = os.getenv(args.api_key_env_var, "").strip()
        if not api_key:
            raise FileNotFoundError(
                f"OpenAI API key file not found at {args.api_key_path}, and "
                f"{args.api_key_env_var} is not set."
            )

    print(f"Selected transcoders: {[n for n, _ in selected_transcoders]}")
    return config, selected_transcoders, api_key


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = args.device if args.device is not None else setup_environment()

    start_time = time.time()

    config, selected_transcoders, api_key = create_config_and_transcoders(
        args, device
    )
    os.makedirs(args.output_folder, exist_ok=True)

    results_dict = run_eval(
        config,
        selected_transcoders,
        device,
        api_key,
        args.output_folder,
        args.force_rerun,
        args.save_logs_path,
        args.artifacts_path,
    )

    print(f"Finished evaluation in {time.time() - start_time:.2f} seconds")
