"""Activation sample collector for feature interpretability."""

import torch
import numpy as np
from collections import defaultdict
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import heapq


@dataclass
class ActivationSample:
    """Single activation sample with context."""
    feature_idx: int
    activation_value: float
    tokens: List[int]
    text: str
    position: int
    context_tokens: List[int]
    context_text: str
    
    def __lt__(self, other):
        return self.activation_value > other.activation_value


class ActivationSampleCollector:
    """Collects top-k activation samples per feature for interpretability."""
    
    def __init__(
        self,
        max_samples_per_feature: int = 100,
        context_size: int = 20,
        storage_threshold: float = 0.1
    ):
        self.max_samples = max_samples_per_feature
        self.context_size = context_size
        self.storage_threshold = storage_threshold
        self.feature_samples: Dict[int, List[ActivationSample]] = defaultdict(list)
        self.total_samples_seen = 0
        self.total_samples_stored = 0
        self.feature_activation_counts = defaultdict(int)
        self.feature_max_activations = defaultdict(float)
        
    def collect_batch_samples(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        tokenizer,
        feature_indices: Optional[List[int]] = None
    ):
        """Collect high-activation samples from a batch."""
        if activations.dim() == 2:
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            activations = activations.reshape(batch_size, seq_len, -1)
        
        batch_size, seq_len, n_features = activations.shape
        
        # Move to CPU for processing
        activations_cpu = activations.detach().cpu()
        tokens_cpu = tokens.detach().cpu()
        
        # Determine which features to track
        if feature_indices is None:
            feature_indices = range(n_features)
        
        # Process each feature
        for feature_idx in feature_indices:
            feature_acts = activations_cpu[:, :, feature_idx]  # [batch, seq_len]
            
            # Find positions above threshold
            above_threshold = feature_acts > self.storage_threshold
            if not above_threshold.any():
                continue
            
            batch_indices, seq_indices = torch.where(above_threshold)
            act_values = feature_acts[batch_indices, seq_indices]
            
            self.total_samples_seen += len(act_values)
            self.feature_activation_counts[feature_idx] += len(act_values)
            
            max_act = act_values.max().item()
            if max_act > self.feature_max_activations[feature_idx]:
                self.feature_max_activations[feature_idx] = max_act
            for batch_idx, seq_idx, act_val in zip(
                batch_indices.tolist(), 
                seq_indices.tolist(), 
                act_values.tolist()
            ):
                start_idx = max(0, seq_idx - self.context_size // 2)
                end_idx = min(seq_len, seq_idx + self.context_size // 2 + 1)
                context_tokens = tokens_cpu[batch_idx, start_idx:end_idx].tolist()
                activating_tokens = tokens_cpu[batch_idx, max(0, seq_idx-1):seq_idx+1].tolist()
                try:
                    context_text = tokenizer.decode(context_tokens)
                    text = tokenizer.decode(activating_tokens)
                except Exception as e:
                    context_text = f"<decode error: {e}>"
                    text = f"<decode error: {e}>"
                
                sample = ActivationSample(
                    feature_idx=feature_idx,
                    activation_value=act_val,
                    tokens=activating_tokens,
                    text=text,
                    position=seq_idx,
                    context_tokens=context_tokens,
                    context_text=context_text
                )
                self._add_sample(feature_idx, sample)
    
    def _add_sample(self, feature_idx: int, sample: ActivationSample):
        """Add sample to feature's heap, maintaining max_samples size."""
        heap = self.feature_samples[feature_idx]
        
        if len(heap) < self.max_samples:
            heapq.heappush(heap, (sample.activation_value, sample))
            self.total_samples_stored += 1
        else:
            min_val = heap[0][0]
            if sample.activation_value > min_val:
                heapq.heapreplace(heap, (sample.activation_value, sample))
    
    def get_top_samples(self, feature_idx: int, k: Optional[int] = None) -> List[ActivationSample]:
        """Get top-k samples for a feature, sorted by activation (highest first)."""
        if feature_idx not in self.feature_samples:
            return []
        
        samples = [sample for _, sample in self.feature_samples[feature_idx]]
        samples.sort(key=lambda x: x.activation_value, reverse=True)
        return samples[:k] if k is not None else samples
    
    def get_feature_statistics(self, feature_idx: int) -> Dict:
        """Get statistics for a specific feature."""
        samples = self.get_top_samples(feature_idx)
        activations = [s.activation_value for s in samples] if samples else []
        
        return {
            'feature_idx': feature_idx,
            'num_samples': len(samples),
            'max_activation': max(activations) if activations else 0.0,
            'mean_activation': float(np.mean(activations)) if activations else 0.0,
            'std_activation': float(np.std(activations)) if activations else 0.0,
            'min_activation': min(activations) if activations else 0.0,
            'activation_count': self.feature_activation_counts.get(feature_idx, 0),
            'all_time_max': self.feature_max_activations.get(feature_idx, 0.0)
        }
    
    def get_all_statistics(self) -> Dict:
        """Get overall statistics about sample collection."""
        return {
            'total_samples_seen': self.total_samples_seen,
            'total_samples_stored': self.total_samples_stored,
            'num_features_tracked': len(self.feature_samples),
            'avg_samples_per_feature': (
                self.total_samples_stored / max(1, len(self.feature_samples))
            ),
            'features_with_samples': sorted(list(self.feature_samples.keys()))
        }
    
    def save_samples(self, save_dir: str, top_k_features: Optional[int] = None, samples_per_feature: int = 10):
        """Save collected samples to disk for analysis."""
        os.makedirs(save_dir, exist_ok=True)
        if top_k_features is not None:
            # Sort features by max activation
            sorted_features = sorted(
                self.feature_samples.keys(),
                key=lambda f: self.feature_max_activations[f],
                reverse=True
            )[:top_k_features]
        else:
            sorted_features = sorted(self.feature_samples.keys())
        
        for feature_idx in sorted_features:
            samples = self.get_top_samples(feature_idx, k=samples_per_feature)
            stats = self.get_feature_statistics(feature_idx)
            
            feature_data = {
                'statistics': stats,
                'top_samples': [asdict(s) for s in samples]
            }
            
            filepath = os.path.join(save_dir, f"feature_{feature_idx:05d}_samples.json")
            with open(filepath, 'w') as f:
                json.dump(feature_data, f, indent=2)
        
        overall_stats = self.get_all_statistics()
        overall_stats['top_features_by_activation'] = [
            {
                'feature_idx': f,
                'max_activation': self.feature_max_activations[f],
                'num_samples': len(self.feature_samples[f])
            }
            for f in sorted(
                self.feature_samples.keys(),
                key=lambda f: self.feature_max_activations[f],
                reverse=True
            )[:50]
        ]
        
        summary_path = os.path.join(save_dir, "collection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        print(f"\nSaved activation samples to {save_dir}")
        print(f"  Features saved: {len(sorted_features)}")
        print(f"  Samples per feature: {samples_per_feature}")
        print(f"  Total samples stored: {self.total_samples_stored}")
    
    def load_samples(self, save_dir: str):
        """Load previously saved samples from disk."""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory not found: {save_dir}")
        
        for filename in os.listdir(save_dir):
            if filename.startswith("feature_") and filename.endswith("_samples.json"):
                filepath = os.path.join(save_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                feature_idx = data['statistics']['feature_idx']
                for sample_dict in data['top_samples']:
                    sample = ActivationSample(**sample_dict)
                    self._add_sample(feature_idx, sample)
        
        print(f"Loaded samples from {save_dir}")
        print(f"  Features loaded: {len(self.feature_samples)}")
    
    def clear(self):
        """Clear all stored samples."""
        self.feature_samples.clear()
        self.total_samples_seen = 0
        self.total_samples_stored = 0
        self.feature_activation_counts.clear()
        self.feature_max_activations.clear()

