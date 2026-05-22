"""
SOE Agent - Speech Oral Evaluation (pronunciation scoring).
Wraps the existing SOEService for use in the multi-agent pipeline.
"""
import logging

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services.tencent.soe import soe_service

logger = logging.getLogger(__name__)


class SOEAgent(BaseAgent):
    """Agent for pronunciation scoring using Tencent Cloud SOE."""

    @property
    def name(self) -> str:
        return "soe"

    async def execute(self, context: EvaluationContext) -> AgentResult:
        """
        Run SOE on audio data and populate context with results.

        Reads from context:
            - audio_data: Raw audio bytes
            - ref_text: Reference text for evaluation
            - eval_mode: Evaluation mode (0-8)
            - score_coeff: Score coefficient (1.0-4.0)
            - server_type: 0=Chinese, 1=English

        Writes to context:
            - soe_result: Full SOE result
            - scores_data: Parsed scores
            - low_score_words: Words with accuracy < 90
            - statistics_data: Word statistics
        """
        audio_data = context.audio_data
        if not audio_data:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="No audio data provided"
            )

        result = await soe_service.evaluate_audio(
            audio_data=audio_data,
            ref_text=context.ref_text,
            eval_mode=context.eval_mode,
            score_coeff=context.score_coeff,
            server_type=context.server_type
        )

        # Populate context
        context.soe_result = result
        context.scores_data = result.get("scores", {})
        context.low_score_words = result.get("low_score_words", [])
        context.statistics_data = result.get("statistics", {})

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "scores": context.scores_data,
                "statistics": context.statistics_data,
                "low_score_count": len(context.low_score_words),
            }
        )
