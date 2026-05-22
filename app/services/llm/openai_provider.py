"""
OpenAI-compatible LLM provider.
Supports OpenAI, Azure OpenAI, Ollama, vLLM, and any OpenAI-compatible API.
"""
import logging
from typing import Optional, AsyncGenerator

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider, ChatResponse

logger = logging.getLogger(__name__)


def _completion_limit_kwargs() -> dict:
    if settings.llm_max_tokens <= 0:
        return {}
    return {"max_tokens": settings.llm_max_tokens}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        multimodal_model: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._model = model or settings.openai_model
        self._multimodal_model = multimodal_model or settings.openai_multimodal_model
        self._timeout = timeout or settings.llm_timeout

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout
        )

    @property
    def name(self) -> str:
        return "openai"

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        """Convert internal format (role/content) to OpenAI format."""
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.get("role", msg.get("Role", "")),
                "content": msg.get("content", msg.get("Content", ""))
            })
        return converted

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> ChatResponse:
        """Generate chat completion."""
        messages = self._convert_messages(messages)
        timeout = timeout or self._timeout

        logger.info(f"Starting OpenAI chat request with model={self._model}, messages_count={len(messages)}, timeout={timeout}")

        try:
            if stream:
                logger.info("Using OpenAI streaming mode")
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    timeout=timeout,
                    **_completion_limit_kwargs()
                )
                content_parts = []
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_parts.append(chunk.choices[0].delta.content)
                result = ChatResponse(
                    content="".join(content_parts),
                    usage={},
                    raw_response=None
                )
                logger.info(f"OpenAI stream completed, content_length={len(result.content)}")
                return result
            else:
                logger.info(f"Waiting for OpenAI response (timeout={timeout}s)...")
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    timeout=timeout,
                    **_completion_limit_kwargs()
                )
                result = ChatResponse(
                    content=response.choices[0].message.content,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    raw_response=response.model_dump()
                )
                logger.info(f"OpenAI chat completed, content_length={len(result.content)}, tokens={result.usage.get('total_tokens', 0)}")
                return result

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
        messages = self._convert_messages(messages)
        timeout = timeout or self._timeout

        logger.info(f"Starting OpenAI stream request with model={self._model}, messages_count={len(messages)}, timeout={timeout}")

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                timeout=timeout,
                **_completion_limit_kwargs()
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            logger.info("OpenAI stream request completed")

        except Exception as e:
            error_msg = f"OpenAI stream error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    async def chat_multimodal(
        self,
        audio_url: str,
        messages: list[dict],
        system_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> ChatResponse:
        """Chat with multimodal model using audio input directly."""
        timeout = timeout or self._timeout
        use_model = model or self._multimodal_model

        logger.info(f"Starting OpenAI multimodal chat request with model={use_model}, audio_url={audio_url[:80]}...")

        # Build multimodal messages
        multimodal_messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history (text only)
        for msg in messages:
            role = msg.get("role", msg.get("Role", ""))
            content = msg.get("content", msg.get("Content", ""))
            if role and content:
                multimodal_messages.append({"role": role, "content": content})

        # Add current user message with audio
        multimodal_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "请根据这段音频回复我。"},
                {"type": "input_audio", "input_audio": {"data": audio_url, "format": "wav"}}
            ]
        })

        try:
            logger.info(f"Waiting for OpenAI multimodal response (timeout={timeout}s)...")
            response = await self._client.chat.completions.create(
                model=use_model,
                messages=multimodal_messages,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                timeout=timeout,
                **_completion_limit_kwargs()
            )
            result = ChatResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response.model_dump()
            )
            logger.info(f"OpenAI multimodal completed, content_length={len(result.content)}")
            return result

        except Exception as e:
            error_msg = f"OpenAI multimodal error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise
