"""
Tencent Cloud Hunyuan LLM provider (native SDK).
"""
import json
import logging
import asyncio
import base64
from typing import Optional, AsyncGenerator

import httpx
from tencentcloud.hunyuan.v20230901 import hunyuan_client_async, models
from tencentcloud.common.exception import TencentCloudSDKException

from app.core.config import settings
from app.services.llm.base import BaseLLMProvider, ChatResponse
from app.services.tencent.base import TencentCloudClient

logger = logging.getLogger(__name__)


class TencentProvider(BaseLLMProvider, TencentCloudClient):
    """Tencent Cloud Hunyuan LLM provider using native SDK."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: Optional[str] = None,
        multimodal_model: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        TencentCloudClient.__init__(self, secret_id, secret_key, "hunyuan.tencentcloudapi.com")
        self._model = model or settings.tencent_model
        self._multimodal_model = multimodal_model or settings.tencent_multimodal_model
        self._timeout = timeout or settings.llm_timeout

    @property
    def name(self) -> str:
        return "tencent"

    def _create_async_client(self) -> hunyuan_client_async.HunyuanClient:
        """Create a new async Hunyuan client for each request."""
        return hunyuan_client_async.HunyuanClient(
            self._get_credential(), "ap-guangzhou", self._get_client_profile()
        )

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        """Convert internal format (role/content) to Tencent format (Role/Content)."""
        converted = []
        for msg in messages:
            converted.append({
                "Role": msg.get("role", msg.get("Role", "")),
                "Content": msg.get("content", msg.get("Content", ""))
            })
        return converted

    def _parse_chat_result(self, result: dict) -> ChatResponse:
        """Parse chat completion result."""
        data = result.get("Response", result)
        choices = data.get("Choices", [])
        content = ""
        if choices:
            message = choices[0].get("Message", {})
            content = message.get("Content", "")
        usage = data.get("Usage", {})

        return ChatResponse(
            content=content,
            usage={
                "prompt_tokens": usage.get("PromptTokens", 0),
                "completion_tokens": usage.get("CompletionTokens", 0),
                "total_tokens": usage.get("TotalTokens", 0)
            },
            raw_response=result
        )

    async def _handle_stream_response(self, response) -> ChatResponse:
        """Handle streaming response and collect full content."""
        logger.info("Processing stream response")
        content_parts = []
        try:
            async for event in response:
                data = json.loads(event["data"])
                if "Choices" in data and len(data["Choices"]) > 0:
                    delta = data["Choices"][0].get("Delta", {})
                    content = delta.get("Content", "")
                    if content:
                        content_parts.append(content)

            result = ChatResponse(
                content="".join(content_parts),
                usage={},
                raw_response=None
            )
            logger.info(f"Stream response processed, content_length={len(result.content)}")
            return result
        except Exception as e:
            logger.error(f"Error processing stream response: {e}", exc_info=True)
            raise

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
        messages = self._convert_messages(messages)

        logger.info(f"Starting Tencent chat request with model={self._model}, messages_count={len(messages)}, timeout={timeout}")

        try:
            client = self._create_async_client()
            req = models.ChatCompletionsRequest()
            params = {
                "Model": self._model,
                "Messages": messages,
                "Temperature": temperature,
                "TopP": top_p,
                "Stream": stream
            }
            req.from_json_string(json.dumps(params))

            async with client:
                if stream:
                    logger.info("Using streaming mode")
                    response = await asyncio.wait_for(
                        client.ChatCompletions(req),
                        timeout=timeout
                    )
                    result = await self._handle_stream_response(response)
                    logger.info(f"Stream completed, content_length={len(result.content)}")
                    return result
                else:
                    logger.info(f"Waiting for response (timeout={timeout}s)...")
                    response = await asyncio.wait_for(
                        client.ChatCompletions(req),
                        timeout=timeout
                    )
                    result = json.loads(response.to_json_string())
                    parsed = self._parse_chat_result(result)
                    logger.info(f"Chat completed, content_length={len(parsed.content)}, tokens={parsed.usage.get('total_tokens', 0)}")
                    return parsed

        except asyncio.TimeoutError:
            error_msg = f"Chat request timeout after {timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
        except TencentCloudSDKException as e:
            error_msg = f"Tencent Cloud SDK error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Chat request failed: {type(e).__name__}: {e}"
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
        messages = self._convert_messages(messages)

        logger.info(f"Starting Tencent stream request with model={self._model}, messages_count={len(messages)}, timeout={timeout}")

        try:
            client = self._create_async_client()
            req = models.ChatCompletionsRequest()
            params = {
                "Model": self._model,
                "Messages": messages,
                "Temperature": temperature,
                "TopP": top_p,
                "Stream": True
            }
            req.from_json_string(json.dumps(params))

            async with client:
                response = await asyncio.wait_for(
                    client.ChatCompletions(req),
                    timeout=timeout
                )
                async for event in response:
                    data = json.loads(event["data"])
                    if "Choices" in data and len(data["Choices"]) > 0:
                        delta = data["Choices"][0].get("Delta", {})
                        content = delta.get("Content", "")
                        if content:
                            yield content

            logger.info("Stream request completed")

        except asyncio.TimeoutError:
            error_msg = f"Stream request timeout after {timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
        except TencentCloudSDKException as e:
            error_msg = f"Tencent Cloud SDK error in stream: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Stream request failed: {type(e).__name__}: {e}"
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

        logger.info(f"Starting Tencent multimodal chat request with model={use_model}, audio_url={audio_url[:80]}...")

        # Download audio and encode as base64
        async with httpx.AsyncClient(timeout=30.0) as client:
            audio_response = await client.get(audio_url)
            audio_bytes = audio_response.content
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Build Hunyuan format messages with audio
        hunyuan_messages = [
            {"Role": "system", "Content": system_prompt}
        ]
        for msg in messages:
            role = msg.get("role", msg.get("Role", ""))
            content = msg.get("content", msg.get("Content", ""))
            if role and content:
                hunyuan_messages.append({"Role": role, "Content": content})

        hunyuan_messages.append({
            "Role": "user",
            "Content": [
                {"type": "text", "text": "请根据这段音频回复我。"},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
            ]
        })

        try:
            client = self._create_async_client()
            req = models.ChatCompletionsRequest()
            params = {
                "Model": use_model,
                "Messages": hunyuan_messages,
                "Temperature": temperature,
                "TopP": top_p,
                "Stream": False
            }
            req.from_json_string(json.dumps(params))

            async with client:
                logger.info(f"Waiting for multimodal response (timeout={timeout}s)...")
                response = await asyncio.wait_for(
                    client.ChatCompletions(req),
                    timeout=timeout
                )
                result = json.loads(response.to_json_string())
                parsed = self._parse_chat_result(result)
                logger.info(f"Multimodal chat completed, content_length={len(parsed.content)}")
                return parsed

        except asyncio.TimeoutError:
            error_msg = f"Multimodal chat request timeout after {timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Multimodal chat request failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise
