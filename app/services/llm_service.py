"""
Unified LLM service supporting multiple providers via Provider Registry.
"""
import logging
from typing import Awaitable, Callable, Optional, AsyncGenerator

from app.core.config import settings
from app.services.llm.registry import ProviderRegistry
from app.services.llm.base import ChatResponse
from app.services.llm.limiter import llm_limiter
from app.services.agents.prompts.common import extract_json
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
from app.services.agents.prompts.story_reading import (
    story_reading_system_prompt,
    story_reading_user_prompt,
)
from app.services.agents.prompts.text_analysis import (
    text_structure_system_prompt,
    text_structure_user_prompt,
    sentence_interpretation_system_prompt,
    sentence_interpretation_user_prompt,
)

logger = logging.getLogger(__name__)

StatusCallback = Callable[[dict], Awaitable[None]]


class LLMService:
    """Unified LLM service supporting multiple providers."""

    def __init__(self, provider: Optional[str] = None, **kwargs):
        self._provider_name = provider or settings.llm_provider
        self._provider = ProviderRegistry.get_provider(self._provider_name, **kwargs)
        logger.info(f"Initialized LLMService with provider: {self._provider_name}")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider.name

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None,
        status_callback: Optional[StatusCallback] = None
    ) -> dict:
        """Generate chat completion."""
        result = await llm_limiter.run(
            lambda: self._provider.chat(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                timeout=timeout
            ),
            provider=self.provider_name,
            operation_name="chat_stream_collect" if stream else "chat",
            status_callback=status_callback,
        )
        return result.model_dump()

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[float] = None,
        status_callback: Optional[StatusCallback] = None
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming."""
        async def _operation():
            async for chunk in self._provider.chat_stream(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            ):
                yield chunk

        async for chunk in llm_limiter.stream(
            _operation,
            provider=self.provider_name,
            operation_name="chat_stream",
            status_callback=status_callback,
        ):
            yield chunk

    async def chat_multimodal(
        self,
        audio_url: str,
        messages: list[dict],
        system_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        status_callback: Optional[StatusCallback] = None
    ) -> dict:
        """Chat with multimodal model using audio input directly."""
        result = await llm_limiter.run(
            lambda: self._provider.chat_multimodal(
                audio_url=audio_url,
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                model=model,
                timeout=timeout
            ),
            provider=self.provider_name,
            operation_name="chat_multimodal",
            status_callback=status_callback,
        )
        return result.model_dump()

    async def generate_evaluation(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None
    ) -> dict:
        """Generate speech evaluation report in JSON format."""
        system_prompt = basic_evaluation_system_prompt()
        user_prompt = basic_evaluation_user_prompt(
            speech_text, speech_scores, custom_prompt,
            low_score_words, statistics
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def generate_evaluation_extended(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None,
        topic: Optional[str] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None
    ) -> dict:
        """Generate extended speech evaluation report with topic relevance and speech rate analysis."""
        system_prompt = extended_evaluation_system_prompt(has_topic=topic is not None)
        user_prompt = extended_evaluation_user_prompt(
            speech_text, speech_scores, custom_prompt,
            low_score_words, statistics, topic, speech_rate, audio_duration
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def generate_simple_report_json(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Generate simple report (JSON format)."""
        system_prompt = simple_report_system_prompt(language=language)
        user_prompt = simple_report_user_prompt(
            speech_text, speech_scores, low_score_words,
            speech_rate, audio_duration
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def generate_full_report_json(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None,
        topic: Optional[str] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Generate full report (JSON format)."""
        system_prompt = full_report_system_prompt(language=language)
        user_prompt = full_report_user_prompt(
            speech_text, speech_scores, low_score_words,
            statistics, topic, speech_rate, audio_duration
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def analyze_text_structure(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> dict:
        """Analyze text structure."""
        system_prompt = text_structure_system_prompt()
        user_prompt = text_structure_user_prompt(text, custom_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def analyze_tongue_twister(
        self,
        text: str,
        language: str = "zh"
    ) -> dict:
        """Analyze tongue twister pronunciation."""
        system_prompt = tongue_twister_system_prompt(language=language)
        user_prompt = tongue_twister_user_prompt(text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def analyze_sentence_interpretation(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> dict:
        """Analyze sentence for reading interpretation."""
        system_prompt = sentence_interpretation_system_prompt()
        user_prompt = sentence_interpretation_user_prompt(text, custom_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def analyze_story_reading(
        self,
        speech_text: str,
        story_text: str,
        word_info_list: Optional[list] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Analyze story reading performance."""
        system_prompt = story_reading_system_prompt(language=language)
        user_prompt = story_reading_user_prompt(
            speech_text, story_text, word_info_list, audio_duration
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}

    async def analyze_tongue_twister_reading(
        self,
        speech_text: str,
        tongue_twister_text: str,
        word_info_list: Optional[list] = None,
        low_score_words: Optional[list] = None,
        scores_data: Optional[dict] = None,
        statistics_data: Optional[dict] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh",
        eval_type: str = "tongue_twister"
    ) -> dict:
        """Analyze reading performance for tongue twisters or articles."""
        if eval_type == "article":
            system_prompt = article_reading_system_prompt(language=language)
            user_prompt = article_reading_user_prompt(
                speech_text, tongue_twister_text, word_info_list,
                low_score_words, scores_data, statistics_data, audio_duration
            )
        else:
            system_prompt = tongue_twister_system_prompt(language=language)
            user_prompt = tongue_twister_user_prompt(
                speech_text, tongue_twister_text, word_info_list,
                low_score_words, scores_data, statistics_data, audio_duration
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}
