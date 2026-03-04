"""
Unified LLM service supporting multiple providers (Hunyuan, OpenAI, etc.)
"""
import logging
from typing import Optional, AsyncGenerator
from enum import Enum

from openai import AsyncOpenAI

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
        
        if self.provider == LLMProvider.HUNYUAN:
            self._hunyuan_client = HunyuanService(
                model=model or settings.hunyuan_model,
                timeout=self.timeout
            )
            self._openai_client = None
            self.model = model or settings.hunyuan_model
        elif self.provider == LLMProvider.OPENAI:
            self._openai_client = AsyncOpenAI(
                api_key=api_key or settings.openai_api_key,
                base_url=base_url or settings.openai_base_url,
                timeout=self.timeout
            )
            self._hunyuan_client = None
            self.model = model or settings.openai_model
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> dict:
        """Generate chat completion."""
        timeout = timeout or self.timeout
        
        if self.provider == LLMProvider.HUNYUAN:
            logger.info(f"Using Hunyuan provider, model={self.model}")
            return await self._hunyuan_client.chat(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                timeout=timeout
            )
        elif self.provider == LLMProvider.OPENAI:
            logger.info(f"Using OpenAI provider, model={self.model}")
            return await self._chat_openai(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                timeout=timeout
            )

    async def _chat_openai(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        stream: bool,
        timeout: float
    ) -> dict:
        """Chat using OpenAI client."""
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                timeout=timeout
            )
            
            if stream:
                content_parts = []
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content_parts.append(chunk.choices[0].delta.content)
                
                return {
                    "content": "".join(content_parts),
                    "usage": {},
                    "raw_response": None
                }
            else:
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "raw_response": response.model_dump()
                }
                
        except Exception as e:
            error_msg = f"OpenAI API error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming."""
        timeout = timeout or self.timeout
        
        if self.provider == LLMProvider.HUNYUAN:
            async for chunk in self._hunyuan_client.chat_stream(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            ):
                yield chunk
        elif self.provider == LLMProvider.OPENAI:
            logger.info(f"Using OpenAI stream provider, model={self.model}")
            try:
                response = await self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    timeout=timeout
                )
                
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
            except Exception as e:
                error_msg = f"OpenAI stream error: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

    async def generate_evaluation(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None
    ) -> str:
        """Generate speech evaluation report."""
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.generate_evaluation(
                speech_text=speech_text,
                speech_scores=speech_scores,
                custom_prompt=custom_prompt,
                low_score_words=low_score_words,
                statistics=statistics
            )
        else:
            raise NotImplementedError(f"generate_evaluation not implemented for {self.provider}")

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
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.generate_evaluation_extended(
                speech_text=speech_text,
                speech_scores=speech_scores,
                custom_prompt=custom_prompt,
                low_score_words=low_score_words,
                statistics=statistics,
                topic=topic,
                speech_rate=speech_rate,
                audio_duration=audio_duration
            )
        else:
            raise NotImplementedError(f"generate_evaluation_extended not implemented for {self.provider}")

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
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.generate_simple_report_json(
                speech_text=speech_text,
                speech_scores=speech_scores,
                low_score_words=low_score_words,
                speech_rate=speech_rate,
                audio_duration=audio_duration,
                language=language
            )
        else:
            raise NotImplementedError(f"generate_simple_report_json not implemented for {self.provider}")

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
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.generate_full_report_json(
                speech_text=speech_text,
                speech_scores=speech_scores,
                low_score_words=low_score_words,
                statistics=statistics,
                topic=topic,
                speech_rate=speech_rate,
                audio_duration=audio_duration,
                language=language
            )
        else:
            raise NotImplementedError(f"generate_full_report_json not implemented for {self.provider}")

    async def analyze_text_structure(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Analyze text structure."""
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.analyze_text_structure(
                text=text,
                custom_prompt=custom_prompt
            )
        else:
            raise NotImplementedError(f"analyze_text_structure not implemented for {self.provider}")

    async def analyze_tongue_twister(
        self,
        text: str,
        language: str = "zh"
    ) -> str:
        """Analyze tongue twister pronunciation."""
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.analyze_tongue_twister(
                text=text,
                language=language
            )
        else:
            raise NotImplementedError(f"analyze_tongue_twister not implemented for {self.provider}")

    async def analyze_sentence_interpretation(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Analyze sentence for reading interpretation."""
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.analyze_sentence_interpretation(
                text=text,
                custom_prompt=custom_prompt
            )
        else:
            raise NotImplementedError(f"analyze_sentence_interpretation not implemented for {self.provider}")

    async def analyze_story_reading(
        self,
        speech_text: str,
        story_text: str,
        word_info_list: Optional[list] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Analyze story reading performance."""
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.analyze_story_reading(
                speech_text=speech_text,
                story_text=story_text,
                word_info_list=word_info_list,
                audio_duration=audio_duration,
                language=language
            )
        else:
            raise NotImplementedError(f"analyze_story_reading not implemented for {self.provider}")

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
        if self.provider == LLMProvider.HUNYUAN:
            return await self._hunyuan_client.analyze_tongue_twister_reading(
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
        else:
            raise NotImplementedError(f"analyze_tongue_twister_reading not implemented for {self.provider}")
