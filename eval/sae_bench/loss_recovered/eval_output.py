from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.loss_recovered.eval_config import LossRecoveredEvalConfig

EVAL_TYPE_ID_LOSS_RECOVERED = "loss_recovered"


@dataclass
class LossRecoveredMetrics(BaseMetrics):
    loss_recovered_score: float = Field(
        title="Loss Recovered",
        description="Fraction of ablation-induced loss recovered by the transcoder.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    ce_loss_with_transcoder: float = Field(
        title="CE Loss with Transcoder",
        description="Cross-entropy loss when using the transcoder reconstruction.",
    )
    ce_loss_without_transcoder: float = Field(
        title="CE Loss without Transcoder",
        description="Baseline cross-entropy loss with no intervention.",
    )
    ce_loss_with_ablation: float = Field(
        title="CE Loss with Ablation",
        description="Cross-entropy loss when the activation is zero-ablated.",
    )


@dataclass
class LossRecoveredMetricCategories(BaseMetricCategories):
    performance: LossRecoveredMetrics = Field(
        title="Performance",
        description="Performance preservation metrics.",
    )


@dataclass(config=ConfigDict(title="Loss Recovered"))
class LossRecoveredEvalOutput(
    BaseEvalOutput[
        LossRecoveredEvalConfig,
        LossRecoveredMetricCategories,
        BaseResultDetail,
    ]
):
    """
    Loss recovered evaluation adapted from SAE Bench core for transcoders.
    """

    eval_config: LossRecoveredEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: LossRecoveredMetricCategories
    eval_result_details: list[BaseResultDetail] = Field(
        default_factory=list,
        title="Details",
        description="No per-feature details for loss recovered.",
    )
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_LOSS_RECOVERED,
        title="Eval Type ID",
        description="The type of the evaluation",
    )


