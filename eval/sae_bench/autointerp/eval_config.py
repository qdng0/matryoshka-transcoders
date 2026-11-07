from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class AutoInterpEvalConfig:
    """Configuration for autointerp evaluation on transcoders."""

    model_name: str = Field(default="")
    n_latents: int = Field(default=1000)
    override_latents: list[int] | None = Field(default=None)
    dead_latent_threshold: float = Field(default=15)
    random_seed: int = Field(default=68)
    dataset_name: str = Field(default="monology/pile-uncopyrighted")
    llm_context_size: int = Field(default=128)
    llm_batch_size: int | None = Field(default=None)
    llm_dtype: str = Field(default="")
    openai_model: str = Field(default="gpt-4.1-nano")

    buffer: int = Field(default=10)
    no_overlap: bool = Field(default=True)
    act_threshold_frac: float = Field(default=0.01)
    total_tokens: int = Field(default=2_000_000)
    scoring: bool = Field(default=True)
    max_tokens_in_explanation: int = Field(default=30)
    use_demos_in_explanation: bool = Field(default=True)

    n_top_ex_for_generation: int = Field(default=10)
    n_iw_sampled_ex_for_generation: int = Field(default=5)
    n_top_ex_for_scoring: int = Field(default=2)
    n_random_ex_for_scoring: int = Field(default=10)
    n_iw_sampled_ex_for_scoring: int = Field(default=2)

    # Transcoder-specific
    config_path: str = Field(default="")
    layer: int = Field(default=-1)
    source_hook_point: str = Field(default="")
    target_hook_point: str = Field(default="")

    @property
    def n_top_ex(self):
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def max_tokens_in_prediction(self) -> int:
        return 2 * self.n_ex_for_scoring + 5

    @property
    def n_ex_for_generation(self) -> int:
        return self.n_top_ex_for_generation + self.n_iw_sampled_ex_for_generation

    @property
    def n_ex_for_scoring(self) -> int:
        return (
            self.n_top_ex_for_scoring
            + self.n_random_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_iw_sampled_ex(self) -> int:
        return self.n_iw_sampled_ex_for_generation + self.n_iw_sampled_ex_for_scoring

    @property
    def n_correct_for_scoring(self) -> int:
        return self.n_top_ex_for_scoring + self.n_iw_sampled_ex_for_scoring
