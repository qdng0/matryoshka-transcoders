import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import load_dataset


class TranscoderActivationsStore:
    """Activation store for training transcoders with paired source/target activations."""
    
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.trust_remote_code = cfg.get("trust_remote_code", False)
        raw_dataset = load_dataset(
            cfg["dataset_path"],
            split=cfg.get("dataset_split", "train"),
            streaming=True,
            trust_remote_code=self.trust_remote_code,
        )
        # Make the data stream depend on cfg["seed"]. Without this, two seeds
        # see the same Pile tokens in the same order; only weight init and
        # within-buffer shuffle differ, which understates seed variance.
        shuffle_buffer = int(cfg.get("dataset_shuffle_buffer", 10_000) or 0)
        if shuffle_buffer > 0:
            raw_dataset = raw_dataset.shuffle(
                seed=int(cfg.get("seed", 49)),
                buffer_size=shuffle_buffer,
            )
        self.dataset = iter(raw_dataset)
        
        # Source and target hook points
        self.source_hook_point = cfg["source_hook_point"]
        self.target_hook_point = cfg["target_hook_point"]
        self.source_act_size = cfg.get("source_act_size", cfg["act_size"])
        self.target_act_size = cfg.get("target_act_size", cfg["act_size"])
        
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.buffer_device = cfg.get("buffer_device", "cpu")
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.tokens_column = cfg.get("tokens_column", cfg.get("text_column", "text"))
        self.config = cfg
        self.prepend_bos = bool(cfg.get("prepend_bos", False))
        self.tokenizer = model.tokenizer
        self.exclude_special_tokens = cfg.get("exclude_special_tokens", True)
        self.special_token_ids = self._get_special_token_ids()
        self.dtype = cfg["dtype"]
        self.act_size = cfg["act_size"]
        self.stop_at_layer = cfg.get("target_layer", cfg["layer"]) + 1
        self.cache_path = cfg.get("activation_cache_path")
        self.cache_size = int(cfg.get("activation_cache_size", 0) or 0)
        self.cache_refresh_steps = int(cfg.get("activation_cache_refresh_steps", 0) or 0)
        self.cache_refresh_ratio = float(cfg.get("activation_cache_refresh_ratio", 0.0) or 0.0)
        self.cache_save_every = int(cfg.get("activation_cache_save_every", 0) or 0)
        self.cache_source = None
        self.cache_target = None
        self.cache_batches_seen = 0
        self.cache_refresh_count = 0
        self._maybe_load_or_build_cache()

    def _get_special_token_ids(self):
        if self.exclude_special_tokens in (False, None):
            return []
        if isinstance(self.exclude_special_tokens, (list, tuple)):
            return [int(token_id) for token_id in self.exclude_special_tokens]

        token_ids = []
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None and token_id != -1:
                token_ids.append(int(token_id))
        return token_ids

    def _special_token_mask(self, batch_tokens: torch.Tensor):
        if not self.special_token_ids:
            return None
        mask = torch.ones_like(batch_tokens, dtype=torch.bool)
        for token_id in self.special_token_ids:
            mask &= batch_tokens != token_id
        return mask

    def _cache_metadata(self):
        return {
            "model_name": self.config.get("model_name"),
            "source_hook_point": self.source_hook_point,
            "target_hook_point": self.target_hook_point,
            "source_act_size": self.source_act_size,
            "target_act_size": self.target_act_size,
            "seq_len": self.context_size,
            "dataset_path": self.config.get("dataset_path"),
            "dataset_split": self.config.get("dataset_split", "train"),
            "tokens_column": self.tokens_column,
            "prepend_bos": self.prepend_bos,
            "exclude_special_tokens": self.exclude_special_tokens,
            "special_token_ids": self.special_token_ids,
            "fold_ln": self.config.get("fold_ln"),
            "cache_size": self.cache_size,
        }

    def _validate_cache_metadata(self, meta):
        if not meta:
            raise ValueError(
                "Activation cache missing metadata. Delete the cache file "
                "or rebuild with the current configuration."
            )
        current = self._cache_metadata()
        mismatches = []
        for key, value in current.items():
            if meta.get(key) != value:
                mismatches.append((key, meta.get(key), value))
        if mismatches:
            details = ", ".join(
                f"{key}: cache={old!r} vs cfg={new!r}" for key, old, new in mismatches
            )
            raise ValueError(f"Activation cache metadata mismatch: {details}")

    def _flatten_and_mask(self, source_acts, target_acts, batch_tokens):
        source_dim = source_acts.shape[-1]
        target_dim = target_acts.shape[-1]
        if source_dim != self.source_act_size or target_dim != self.target_act_size:
            raise ValueError(
                f"Activation sizes do not match config: "
                f"source={source_dim} (cfg {self.source_act_size}), "
                f"target={target_dim} (cfg {self.target_act_size})."
            )
        source_flat = source_acts.reshape(-1, source_dim)
        target_flat = target_acts.reshape(-1, target_dim)
        mask = self._special_token_mask(batch_tokens)
        if mask is not None:
            flat_mask = mask.reshape(-1)
            source_flat = source_flat[flat_mask]
            target_flat = target_flat[flat_mask]
        return source_flat, target_flat

    def _maybe_load_or_build_cache(self):
        if not self.cache_path or self.cache_size <= 0:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        if os.path.exists(self.cache_path):
            cache = torch.load(self.cache_path, map_location="cpu")
            self._validate_cache_metadata(cache.get("meta"))
            self.cache_source = cache["source"]
            self.cache_target = cache["target"]
            if self.cache_source.shape[0] != self.cache_target.shape[0]:
                raise ValueError("Cache source/target have different lengths.")
            self.cache_size = int(self.cache_source.shape[0])
            return

        self._build_cache(self.cache_size)

    def _build_cache(self, cache_size):
        all_source = []
        all_target = []
        total = 0
        while total < cache_size:
            batch_tokens = self.get_batch_tokens()
            source_acts, target_acts = self.get_paired_activations(batch_tokens)
            source_flat, target_flat = self._flatten_and_mask(source_acts, target_acts, batch_tokens)
            all_source.append(source_flat.cpu())
            all_target.append(target_flat.cpu())
            total += source_flat.shape[0]

        source_cache = torch.cat(all_source, dim=0)[:cache_size]
        target_cache = torch.cat(all_target, dim=0)[:cache_size]
        torch.save(
            {"source": source_cache, "target": target_cache, "meta": self._cache_metadata()},
            self.cache_path,
        )
        self.cache_source = source_cache
        self.cache_target = target_cache

    def _maybe_refresh_cache(self):
        if (
            self.cache_source is None
            or self.cache_target is None
            or self.cache_refresh_steps <= 0
            or self.cache_refresh_ratio <= 0.0
        ):
            return
        if self.cache_batches_seen % self.cache_refresh_steps != 0:
            return

        replace_count = int(self.cache_size * self.cache_refresh_ratio)
        replace_count = max(1, min(replace_count, self.cache_size))
        all_source = []
        all_target = []
        total = 0
        while total < replace_count:
            batch_tokens = self.get_batch_tokens()
            source_acts, target_acts = self.get_paired_activations(batch_tokens)
            source_flat, target_flat = self._flatten_and_mask(source_acts, target_acts, batch_tokens)
            all_source.append(source_flat.cpu())
            all_target.append(target_flat.cpu())
            total += source_flat.shape[0]

        source_new = torch.cat(all_source, dim=0)[:replace_count]
        target_new = torch.cat(all_target, dim=0)[:replace_count]
        idx = torch.randint(0, self.cache_size, (replace_count,))
        self.cache_source[idx] = source_new
        self.cache_target[idx] = target_new
        self.cache_refresh_count += 1
        if self.cache_save_every > 0 and self.cache_refresh_count % self.cache_save_every == 0:
            torch.save(
                {"source": self.cache_source, "target": self.cache_target, "meta": self._cache_metadata()},
                self.cache_path,
            )

    def get_batch_tokens(self):
        """Generate a batch of tokens from the dataset"""
        token_chunks = []
        total_tokens = 0
        target_tokens = self.model_batch_size * self.context_size
        first_chunk = True
        while total_tokens < target_tokens:
            batch = next(self.dataset)
            if isinstance(batch[self.tokens_column], str):
                prepend_bos = self.prepend_bos and first_chunk
                tokens = self.model.to_tokens(
                    batch[self.tokens_column],
                    truncate=True,
                    move_to_device=True,
                    prepend_bos=prepend_bos,
                ).squeeze(0)
            else:
                tokens = torch.as_tensor(
                    batch[self.tokens_column],
                    dtype=torch.long,
                    device=self.device,
                )
            tokens = tokens.reshape(-1)
            token_chunks.append(tokens)
            total_tokens += tokens.numel()
            first_chunk = False
        
        token_tensor = torch.cat(token_chunks, dim=0)[:target_tokens]
        
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_paired_activations(self, batch_tokens: torch.Tensor):
        """Get activations from both source and target hook points."""
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.source_hook_point, self.target_hook_point],
                stop_at_layer=self.stop_at_layer,
            )

        source_acts = cache[self.source_hook_point].to(dtype=self.dtype)
        target_acts = cache[self.target_hook_point].to(dtype=self.dtype)

        return source_acts, target_acts

    def _fill_buffer(self):
        """Fill the activation buffer with paired source-target activations."""
        if self.cache_source is not None and self.cache_target is not None:
            total = self.cache_source.shape[0]
            buffer_size = self.model_batch_size * self.context_size * self.num_batches_in_buffer
            idx = torch.randint(0, total, (buffer_size,))
            source_buffer = self.cache_source[idx].to(self.buffer_device, dtype=self.dtype)
            target_buffer = self.cache_target[idx].to(self.buffer_device, dtype=self.dtype)
            return source_buffer, target_buffer

        all_source_acts = []
        all_target_acts = []
        
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            source_acts, target_acts = self.get_paired_activations(batch_tokens)
            source_flat, target_flat = self._flatten_and_mask(source_acts, target_acts, batch_tokens)
            all_source_acts.append(source_flat.to(self.buffer_device, dtype=self.dtype))
            all_target_acts.append(target_flat.to(self.buffer_device, dtype=self.dtype))
        
        return (
            torch.cat(all_source_acts, dim=0),
            torch.cat(all_target_acts, dim=0)
        )
    
    def _get_dataloader(self):
        """Create a DataLoader from the current buffer"""
        dataset = TensorDataset(self.source_buffer, self.target_buffer)
        return DataLoader(
            dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True,
            pin_memory=self.device != "cpu" and self.buffer_device == "cpu",
        )

    def next_batch(self):
        """Get the next batch of paired activations."""
        self.cache_batches_seen += 1
        self._maybe_refresh_cache()
        try:
            source_batch, target_batch = next(self.dataloader_iter)
        except (StopIteration, AttributeError):
            self.source_buffer, self.target_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            source_batch, target_batch = next(self.dataloader_iter)

        if self.buffer_device != self.device:
            source_batch = source_batch.to(self.device, dtype=self.dtype, non_blocking=True)
            target_batch = target_batch.to(self.device, dtype=self.dtype, non_blocking=True)
        return source_batch, target_batch

    def get_batch_with_tokens(self):
        """Get a fresh batch of activations with corresponding tokens for sample collection."""
        batch_tokens = self.get_batch_tokens()
        source_acts, target_acts = self.get_paired_activations(batch_tokens)
        mask = self._special_token_mask(batch_tokens)
        if mask is not None:
            source_acts = source_acts.clone()
            source_acts[~mask] = 0
        return source_acts, batch_tokens
