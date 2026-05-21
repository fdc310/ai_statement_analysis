"""
Report Agent - Aggregates results from other agents into final evaluation report.
Uses LLM for report generation.
"""
import logging
from typing import Optional

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services.agents.prompts.common import extract_json, build_low_score_words_table
from app.services.monitoring.token_tracker import token_tracker
from app.services.agents.prompts.evaluation import (
    basic_evaluation_system_prompt,
    basic_evaluation_user_prompt,
    extended_evaluation_system_prompt,
    extended_evaluation_user_prompt,
    simple_report_system_prompt,
    simple_report_user_prompt,
    full_report_system_prompt,
    full_report_user_prompt,
)
from app.services.agents.prompts.story_reading import (
    story_reading_system_prompt,
    story_reading_user_prompt,
)
from app.services.agents.prompts.tongue_twister import (
    tongue_twister_system_prompt,
    tongue_twister_user_prompt,
    article_reading_system_prompt,
    article_reading_user_prompt,
)
from app.services.agents.prompts.opinion_statement import (
    opinion_statement_system_prompt,
    opinion_statement_user_prompt,
)
from app.services.agents.prompts.impromptu_reaction import (
    impromptu_reaction_system_prompt,
    impromptu_reaction_user_prompt,
)

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """Agent for generating final evaluation reports."""

    @property
    def name(self) -> str:
        return "report"

    async def execute(self, context: EvaluationContext) -> AgentResult:
        """
        Generate final evaluation report from all agent results.

        Reads from context:
            - speech_text: Transcribed text
            - scores_data: SOE scores
            - low_score_words: Low-scoring words
            - statistics_data: Word statistics
            - speech_rate: Calculated speech rate
            - All agent results

        Writes to context:
            - evaluation_report: Final report (Markdown or JSON)
        """
        from app.services import get_llm_service

        llm = get_llm_service()
        eval_type = context.request.get("eval_type", "basic")
        report_format = context.request.get("report_format", "markdown")

        # Build prompts based on evaluation type and format
        system_prompt = self._get_system_prompt(eval_type, report_format, context)
        user_prompt = self._build_user_prompt(context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await llm.chat(
            messages,
            temperature=0.3,
            status_callback=context.request.get("_llm_status_callback"),
        )
        content = result.get("content", "")

        # Record token usage
        usage = result.get("usage", {})
        if usage:
            await token_tracker.record_usage(
                provider=llm.provider_name,
                model=llm.model_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                agent_name=self.name,
            )

        # Parse JSON from LLM response (handles markdown code blocks)
        report_data = extract_json(content)
        if not report_data:
            # If JSON parsing fails, store raw content as fallback
            report_data = {"raw_report": content}

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"report": report_data},
            token_usage=result.get("usage"),
        )

    def _get_system_prompt(self, eval_type: str, report_format: str, context: EvaluationContext) -> str:
        """Get system prompt based on evaluation type and format."""
        has_topic = bool(context.request.get("topic"))
        language = context.language

        if eval_type == "opinion_statement":
            return opinion_statement_system_prompt(language=language, has_topic=has_topic)
        elif eval_type == "impromptu_reaction":
            return impromptu_reaction_system_prompt(language=language, has_scenario=has_topic)
        elif eval_type == "story_reading":
            return story_reading_system_prompt()
        elif eval_type == "tongue_twister":
            return tongue_twister_system_prompt()
        elif eval_type == "article_reading":
            return article_reading_system_prompt()
        elif report_format == "json":
            return simple_report_system_prompt(language=language)
        else:
            return full_report_system_prompt(language=language, has_topic=has_topic)

    def _build_user_prompt(self, context: EvaluationContext) -> str:
        """Build user prompt with all available data."""
        eval_type = context.request.get("eval_type", "basic")
        report_format = context.request.get("report_format", "markdown")
        has_topic = bool(context.request.get("topic"))

        # Use extracted prompt builders for specific eval types
        if eval_type == "opinion_statement":
            return opinion_statement_user_prompt(
                speech_text=context.speech_text or "",
                speech_scores=context.scores_data or {},
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                statistics=context.statistics_data,
                topic=context.request.get("topic"),
                speech_rate=context.speech_rate,
                audio_duration=context.audio_duration,
                language=context.language,
            )
        elif eval_type == "impromptu_reaction":
            return impromptu_reaction_user_prompt(
                speech_text=context.speech_text or "",
                speech_scores=context.scores_data or {},
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                statistics=context.statistics_data,
                scenario=context.request.get("topic"),
                speech_rate=context.speech_rate,
                audio_duration=context.audio_duration,
                language=context.language,
            )
        elif eval_type == "story_reading":
            return story_reading_user_prompt(
                speech_text=context.speech_text or "",
                reference_text=context.request.get("reference_text"),
                speech_scores=context.scores_data,
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                statistics=context.statistics_data,
                speech_rate=context.speech_rate,
                audio_duration=context.audio_duration,
                language=context.language,
            )
        elif eval_type == "tongue_twister":
            return tongue_twister_user_prompt(
                speech_text=context.speech_text or "",
                reference_text=context.request.get("reference_text"),
                speech_scores=context.scores_data,
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                language=context.language,
            )
        elif eval_type == "article_reading":
            return article_reading_user_prompt(
                speech_text=context.speech_text or "",
                reference_text=context.request.get("reference_text"),
                speech_scores=context.scores_data,
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                statistics=context.statistics_data,
                speech_rate=context.speech_rate,
                audio_duration=context.audio_duration,
                language=context.language,
            )
        elif report_format == "json":
            return simple_report_user_prompt(
                speech_text=context.speech_text or "",
                speech_scores=context.scores_data,
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                statistics=context.statistics_data,
                speech_rate=context.speech_rate,
                audio_duration=context.audio_duration,
                language=context.language,
            )
        else:
            return full_report_user_prompt(
                speech_text=context.speech_text or "",
                speech_scores=context.scores_data,
                word_info_list=context.word_info_list,
                low_score_words=context.low_score_words,
                statistics=context.statistics_data,
                speech_rate=context.speech_rate,
                audio_duration=context.audio_duration,
                language=context.language,
                reference_text=context.ref_text,
            )
