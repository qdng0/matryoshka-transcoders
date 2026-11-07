from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class LossRecoveredEvalConfig:
    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name for the LLM used during evaluation.",
    )
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed for dataset shuffling and reproducibility.",
    )
    dataset: str = Field(
        default="Skylion007/openwebtext",
        title="Dataset",
        description="Dataset to draw evaluation text from.",
    )
    tokens_column: str = Field(
        default="text",
        title="Tokens Column",
        description="Column in the dataset containing text to tokenize.",
    )
    context_size: int = Field(
        default=128,
        title="Context Length",
        description="Context length for evaluation prompts.",
    )
    batch_size_prompts: int = Field(
        default=16,
        title="Batch Size Prompts",
        description="Number of prompts per evaluation batch.",
    )
    n_batches: int = Field(
        default=10,
        title="Number of Batches",
        description="Number of batches to evaluate for loss recovered.",
    )
    llm_dtype: str = Field(
        default="float32",
        title="LLM Data Type",
        description="Data type for LLM forward passes.",
    )
    layer: int = Field(
        default=0,
        title="Hook Layer",
        description="Layer index for the transcoder hooks.",
    )
    config_path: str = Field(
        default="",
        title="Config Path",
        description="Path to the transcoder training config used for this eval.",
    )
    stability_epsilon: float = Field(
        default=1e-6,
        title="Stability Epsilon",
        description="Small value to avoid divide-by-zero in the loss recovered score.",
    )


