"""
Unified LLM service supporting multiple providers (Hunyuan, OpenAI, etc.)
All providers delegate to HunyuanService which handles backend switching internally.
"""
import logging
from typing import Optional, AsyncGenerator
from enum import Enum

from app.services.tencent.hunyuan import HunyuanService
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    HUNYUAN = "hunyuan"
    OPENAI = "openai"


class LLMService:
    """Unified LLM service supporting multiple providers."""

    def __init__(
        self,
        provider: str = "hunyuan",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self.provider = LLMProvider(provider.lower())
        self.timeout = timeout or settings.hunyuan_timeout

        if self.provider == LLMProvider.OPENAI:
            self.model = model or settings.openai_model
            self._client = HunyuanService(
                backend="openai",
                model=self.model,
                api_key=api_key or settings.openai_api_key,
                base_url=base_url or settings.openai_base_url,
                timeout=self.timeout
            )
        else:
            self.model = model or settings.hunyuan_model
            self._client = HunyuanService(
                model=self.model,
                timeout=self.timeout
            )

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> dict:
        """Generate chat completion."""
        return await self._client.chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            timeout=timeout or self.timeout
        )

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming."""
        async for chunk in self._client.chat_stream(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout or self.timeout
        ):
            yield chunk

    async def generate_evaluation(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None
    ) -> str:
        """Generate speech evaluation report."""
        return await self._client.generate_evaluation(
            speech_text=speech_text,
            speech_scores=speech_scores,
            custom_prompt=custom_prompt,
            low_score_words=low_score_words,
            statistics=statistics
        )

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
    ) -> str:
        """Generate extended speech evaluation report."""
        return await self._client.generate_evaluation_extended(
            speech_text=speech_text,
            speech_scores=speech_scores,
            custom_prompt=custom_prompt,
            low_score_words=low_score_words,
            statistics=statistics,
            topic=topic,
            speech_rate=speech_rate,
            audio_duration=audio_duration
        )

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
        return await self._client.generate_simple_report_json(
            speech_text=speech_text,
            speech_scores=speech_scores,
            low_score_words=low_score_words,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            language=language
        )

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
        return await self._client.generate_full_report_json(
            speech_text=speech_text,
            speech_scores=speech_scores,
            low_score_words=low_score_words,
            statistics=statistics,
            topic=topic,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            language=language
        )

    async def analyze_text_structure(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Analyze text structure."""
        return await self._client.analyze_text_structure(
            text=text,
            custom_prompt=custom_prompt
        )

    async def analyze_tongue_twister(
        self,
        text: str,
        language: str = "zh"
    ) -> str:
        """Analyze tongue twister pronunciation."""
        return await self._client.analyze_tongue_twister(
            text=text,
            language=language
        )

    async def analyze_sentence_interpretation(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Analyze sentence for reading interpretation."""
        return await self._client.analyze_sentence_interpretation(
            text=text,
            custom_prompt=custom_prompt
        )

    async def analyze_story_reading(
        self,
        speech_text: str,
        story_text: str,
        word_info_list: Optional[list] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Analyze story reading performance."""
        return await self._client.analyze_story_reading(
            speech_text=speech_text,
            story_text=story_text,
            word_info_list=word_info_list,
            audio_duration=audio_duration,
            language=language
        )

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
        return await self._client.analyze_tongue_twister_reading(
            speech_text=speech_text,
            tongue_twister_text=tongue_twister_text,
            word_info_list=word_info_list,
            low_score_words=low_score_words,
            scores_data=scores_data,
            statistics_data=statistics_data,
            audio_duration=audio_duration,
            language=language,
            eval_type=eval_type
        )
