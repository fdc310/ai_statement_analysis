"""
Anthropic Claude LLM provider.
"""
import logging
from typing import Optional, AsyncGenerator

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider, ChatResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        multimodal_model: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.anthropic_model
        self._multimodal_model = multimodal_model or settings.anthropic_multimodal_model
        self._timeout = timeout or settings.llm_timeout

        self._client = AsyncAnthropic(
            api_key=self._api_key,
            timeout=self._timeout
        )

    @property
    def name(self) -> str:
        return "anthropic"

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """
        Convert internal format to Anthropic format.
        Returns (system_prompt, messages) since Anthropic uses system as a top-level param.
        """
        system_prompt = None
        converted = []

        for msg in messages:
            role = msg.get("role", msg.get("Role", ""))
            content = msg.get("content", msg.get("Content", ""))

            if role == "system":
                system_prompt = content
            else:
                # Anthropic requires alternating user/assistant messages
                converted.append({"role": role, "content": content})

        return system_prompt, converted

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> ChatResponse:
        """Generate chat completion."""
        timeout = timeout or self._timeout
        system_prompt, messages = self._convert_messages(messages)

        logger.info(f"Starting Anthropic chat request with model={self._model}, messages_count={len(messages)}, timeout={timeout}")

        try:
            if stream:
                logger.info("Using Anthropic streaming mode")
                content_parts = []
                async with self._client.messages.stream(
                    model=self._model,
                    max_tokens=4096,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_prompt or "",
                    messages=messages
                ) as stream_response:
                    async for text in stream_response.text_stream:
                        content_parts.append(text)
                result = ChatResponse(
                    content="".join(content_parts),
                    usage={},
                    raw_response=None
                )
                logger.info(f"Anthropic stream completed, content_length={len(result.content)}")
                return result
            else:
                logger.info(f"Waiting for Anthropic response (timeout={timeout}s)...")
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_prompt or "",
                    messages=messages
                )
                result = ChatResponse(
                    content=response.content[0].text,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    },
                    raw_response=response.model_dump()
                )
                logger.info(f"Anthropic chat completed, content_length={len(result.content)}, tokens={result.usage.get('total_tokens', 0)}")
                return result

        except Exception as e:
            error_msg = f"Anthropic API error: {type(e).__name__}: {e}"
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
        timeout = timeout or self._timeout
        system_prompt, messages = self._convert_messages(messages)

        logger.info(f"Starting Anthropic stream request with model={self._model}, messages_count={len(messages)}, timeout={timeout}")

        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=4096,
                temperature=temperature,
                top_p=top_p,
                system=system_prompt or "",
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text
            logger.info("Anthropic stream request completed")

        except Exception as e:
            error_msg = f"Anthropic stream error: {type(e).__name__}: {e}"
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

        logger.info(f"Starting Anthropic multimodal chat request with model={use_model}, audio_url={audio_url[:80]}...")

        # Build multimodal messages
        # Anthropic uses system as a top-level param
        multimodal_messages = []

        # Add conversation history (text only)
        for msg in messages:
            role = msg.get("role", msg.get("Role", ""))
            content = msg.get("content", msg.get("Content", ""))
            if role and content and role != "system":
                multimodal_messages.append({"role": role, "content": content})

        # Add current user message with audio
        # Anthropic supports base64-encoded media in content blocks
        import base64
        import httpx

        # Download audio and encode as base64
        async with httpx.AsyncClient(timeout=30.0) as client:
            audio_response = await client.get(audio_url)
            audio_bytes = audio_response.content
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        multimodal_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "请根据这段音频回复我。"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                }
            ]
        })

        try:
            logger.info(f"Waiting for Anthropic multimodal response (timeout={timeout}s)...")
            response = await self._client.messages.create(
                model=use_model,
                max_tokens=4096,
                temperature=temperature,
                top_p=top_p,
                system=system_prompt,
                messages=multimodal_messages
            )
            result = ChatResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                raw_response=response.model_dump()
            )
            logger.info(f"Anthropic multimodal completed, content_length={len(result.content)}")
            return result

        except Exception as e:
            error_msg = f"Anthropic multimodal error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise
