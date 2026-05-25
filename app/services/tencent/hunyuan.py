"""
Tencent Cloud Hunyuan LLM service with async support.
Prompts have been extracted to app.services.agents.prompts for modular maintenance.
"""
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator

from tencentcloud.hunyuan.v20230901 import hunyuan_client_async, models
from tencentcloud.common.exception import TencentCloudSDKException
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.tencent.base import TencentCloudClient
from app.services.agents.prompts.common import extract_json
from app.services.agents.prompts.evaluation import (
    simple_report_system_prompt,
    simple_report_user_prompt,
    full_report_system_prompt,
    full_report_user_prompt,
)
from app.services.agents.prompts.tongue_twister import (
    tongue_twister_system_prompt,
    tongue_twister_user_prompt,
    tongue_twister_reading_system_prompt,
    tongue_twister_reading_user_prompt,
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


class HunyuanService(TencentCloudClient):
    """Tencent Cloud Hunyuan LLM service with async support."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        backend: str = "native",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        super().__init__(secret_id, secret_key, "hunyuan.tencentcloudapi.com")
        self.backend = backend
        self.model = model or getattr(settings, "hunyuan_model", settings.tencent_model)
        self.timeout = timeout or getattr(settings, "hunyuan_timeout", settings.llm_timeout)
        self._openai_client = None
        if self.backend == "openai":
            self._openai_client = AsyncOpenAI(
                api_key=api_key or settings.openai_api_key,
                base_url=base_url or settings.openai_base_url,
                timeout=self.timeout
            )

    def _create_async_client(self) -> hunyuan_client_async.HunyuanClient:
        """Create a new async Hunyuan client for each request."""
        return hunyuan_client_async.HunyuanClient(
            self._get_credential(), "ap-guangzhou", self._get_client_profile()
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> dict:
        """Generate chat completion (async)."""
        timeout = timeout or self.timeout
        logger.info(f"Starting chat request with model={self.model}, backend={self.backend}, messages_count={len(messages)}, timeout={timeout}")

        if self.backend == "openai":
            return await self._chat_openai(messages, temperature, top_p, stream, timeout)

        try:
            client = self._create_async_client()

            req = models.ChatCompletionsRequest()
            params = {
                "Model": self.model,
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
                    logger.info(f"Stream completed, content_length={len(result.get('content', ''))}")
                    return result
                else:
                    logger.info(f"Waiting for response (timeout={timeout}s)...")
                    response = await asyncio.wait_for(
                        client.ChatCompletions(req),
                        timeout=timeout
                    )
                    result = json.loads(response.to_json_string())
                    parsed = self._parse_chat_result(result)
                    logger.info(f"Chat completed, content_length={len(parsed.get('content', ''))}, tokens={parsed.get('usage', {}).get('total_tokens', 0)}")
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
        """Generate chat completion with streaming (async generator)."""
        timeout = timeout or self.timeout
        logger.info(f"Starting stream request with model={self.model}, backend={self.backend}, messages_count={len(messages)}, timeout={timeout}")

        if self.backend == "openai":
            openai_messages = self._convert_messages_to_openai(messages)
            try:
                response = await self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    timeout=timeout
                )
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                logger.info("OpenAI stream request completed")
            except Exception as e:
                error_msg = f"OpenAI stream error: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                raise
            return

        try:
            client = self._create_async_client()

            req = models.ChatCompletionsRequest()
            params = {
                "Model": self.model,
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
    ) -> dict:
        """
        Chat with multimodal model using audio input directly.

        Args:
            audio_url: URL of the audio file to process
            messages: Conversation history (text only)
            system_prompt: System prompt for the conversation
            temperature: Sampling temperature
            top_p: Top-p sampling
            model: Override model name (uses config default if None)
            timeout: Request timeout

        Returns:
            dict with content, usage, and raw_response
        """
        timeout = timeout or self.timeout
        tencent_multimodal_model = getattr(
            settings,
            "hunyuan_multimodal_model",
            settings.tencent_multimodal_model,
        )
        use_model = model or (settings.openai_multimodal_model if self.backend == "openai" else tencent_multimodal_model)
        logger.info(f"Starting multimodal chat request with model={use_model}, backend={self.backend}, audio_url={audio_url[:80]}...")

        # Build multimodal messages
        # System prompt as first message
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

        if self.backend == "openai":
            return await self._chat_openai_multimodal(multimodal_messages, use_model, temperature, top_p, timeout)

        # Native Hunyuan multimodal - send audio as base64 in content
        import base64
        import httpx

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
                logger.info(f"Multimodal chat completed, content_length={len(parsed.get('content', ''))}")
                return parsed

        except asyncio.TimeoutError:
            error_msg = f"Multimodal chat request timeout after {timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Multimodal chat request failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    async def _chat_openai_multimodal(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        top_p: float,
        timeout: float
    ) -> dict:
        """Chat using OpenAI-compatible API with multimodal input."""
        try:
            logger.info(f"Waiting for OpenAI multimodal response (timeout={timeout}s)...")
            response = await self._openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                timeout=timeout
            )
            result = {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "raw_response": response.model_dump()
            }
            logger.info(f"OpenAI multimodal completed, content_length={len(result['content'])}")
            return result
        except Exception as e:
            error_msg = f"OpenAI multimodal error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    async def _handle_stream_response(self, response) -> dict:
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

            result = {
                "content": "".join(content_parts),
                "usage": {},
                "raw_response": None
            }
            logger.info(f"Stream response processed, content_length={len(result['content'])}")
            return result
        except Exception as e:
            logger.error(f"Error processing stream response: {e}", exc_info=True)
            raise

    def _parse_chat_result(self, result: dict) -> dict:
        """Parse chat completion result. Handles both direct and Response-wrapped formats."""
        data = result.get("Response", result)

        choices = data.get("Choices", [])
        content = ""
        if choices:
            message = choices[0].get("Message", {})
            content = message.get("Content", "")

        usage = data.get("Usage", {})

        logger.info(f"API response parsed, content_length={len(content)}, "
                     f"tokens={usage.get('TotalTokens', 0)}")
        if content:
            preview = content[:500] + ("..." if len(content) > 500 else "")
            logger.debug(f"API raw content: {preview}")

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("PromptTokens", 0),
                "completion_tokens": usage.get("CompletionTokens", 0),
                "total_tokens": usage.get("TotalTokens", 0)
            },
            "raw_response": result
        }

    @staticmethod
    def _extract_json(content: str):
        """
        Extract JSON from AI response content.
        Handles: markdown code blocks (```json ... ```), plain JSON, extra text around JSON.
        """
        import re

        preview = content[:500] + ("..." if len(content) > 500 else "")
        logger.info(f"Extracting JSON from content, length={len(content)}")
        logger.debug(f"Raw content: {preview}")

        # Try to extract from markdown code block first
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', content)
        if code_block_match:
            try:
                result = json.loads(code_block_match.group(1).strip())
                logger.info("JSON extracted from markdown code block")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse markdown code block content as JSON: {e}")

        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                result = json.loads(json_match.group())
                logger.info("JSON extracted from object pattern")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse extracted object as JSON: {e}")

        # Try to find JSON array
        array_match = re.search(r'\[[\s\S]*\]', content)
        if array_match:
            try:
                result = json.loads(array_match.group())
                logger.info("JSON extracted from array pattern")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse extracted array as JSON: {e}")

        # Last resort: try parsing the whole content
        logger.warning("No JSON pattern matched, attempting to parse raw content")
        return json.loads(content)

    @staticmethod
    def _convert_messages_to_openai(messages: list[dict]) -> list[dict]:
        """Convert Hunyuan format (Role/Content) to OpenAI format (role/content)."""
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.get("Role", msg.get("role", "")),
                "content": msg.get("Content", msg.get("content", ""))
            })
        return converted

    async def _chat_openai(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        stream: bool,
        timeout: float
    ) -> dict:
        """Chat using OpenAI-compatible API."""
        messages = self._convert_messages_to_openai(messages)
        try:
            if stream:
                logger.info("Using OpenAI streaming mode")
                response = await self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    timeout=timeout
                )
                content_parts = []
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content_parts.append(chunk.choices[0].delta.content)
                result = {
                    "content": "".join(content_parts),
                    "usage": {},
                    "raw_response": None
                }
                logger.info(f"OpenAI stream completed, content_length={len(result['content'])}")
                return result
            else:
                logger.info(f"Waiting for OpenAI response (timeout={timeout}s)...")
                response = await self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    timeout=timeout
                )
                result = {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "raw_response": response.model_dump()
                }
                logger.info(f"OpenAI chat completed, content_length={len(result['content'])}, tokens={result['usage'].get('total_tokens', 0)}")
                return result
        except Exception as e:
            error_msg = f"OpenAI API error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    # ─── Evaluation methods (using extracted prompts) ───────────────────────

    async def generate_evaluation(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None
    ) -> str:
        """Generate speech evaluation report in Markdown format (async)."""
        system_prompt = """你是一个专业的语音演讲评测专家。你的任务是根据用户提供的语音转文字内容和语音评分数据，生成一份详细的演讲评测报告。

你必须严格按照以下Markdown格式输出评测报告，不要添加任何额外的格式或内容：

# 评分

## 逻辑完整性评分
* 综合评分 [综合评分分数]
1. 逻辑性 [逻辑性分数]
2. 流畅度 [流畅度分数]
3. 语速 [语速分数]

## 结构可视化

### 论点

1. [论点1]
2. [论点2]
3. [论点3]

### 结论

* [结论要点]

## 优点

* [优点1]
* [优点2]
* [优点3]

## 改进意见

* [改进意见1]
* [改进意见2]
* [改进意见3]

评分规则：
1. 逻辑完整性评分：根据演讲的论证逻辑、因果关系、论据支撑等方面评分，每项满分100分
2. 结构可视化：提取演讲的主要论点和结论，清晰展示论证结构
3. 优点：指出演讲中表现出色的地方
4. 改进意见：给出具体可行的改进建议

注意：
- 必须严格按照上述Markdown格式输出，结论是结构可视化的子标题(###)
- 评分要客观公正，有理有据
- 改进意见要具体、可操作
- 结合语音评分数据（发音准确度、流利度等）进行综合评价"""

        user_prompt = f"""请根据以下信息生成演讲评测报告：

## 语音转文字内容

{speech_text}

## 语音评分数据

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分
- 总体评分: {speech_scores.get('overall_score', 0)}分
"""

        if statistics:
            user_prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0)}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words:
            user_prompt += """
## 发音待改进的字词（准确度<90分）

| 字词 | 准确度 | 流利度 |
|------|--------|--------|
"""
            for word in low_score_words[:20]:
                user_prompt += f"| {word.get('word', '')} | {word.get('accuracy', 0)} | {word.get('fluency', 0)} |\n"

        if custom_prompt:
            user_prompt += f"""
## 额外评测要求

{custom_prompt}
"""

        user_prompt += """
请严格按照系统提示中指定的Markdown格式生成评测报告。在改进意见中，请特别关注发音待改进的字词。"""

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

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
        """Generate extended speech evaluation report with topic relevance and speech rate analysis."""
        system_prompt = """你是一个专业的语音演讲评测专家。你的任务是根据用户提供的语音转文字内容和语音评分数据，生成一份详细的演讲评测报告。

你必须严格按照以下Markdown格式输出评测报告，不要添加任何额外的格式或内容：

# 评分

## 逻辑完整性评分
* 综合评分 [综合评分分数]
1. 逻辑性 [逻辑性分数]
2. 流畅度 [流畅度分数]
3. 语速 [语速分数]"""

        if topic is not None:
            system_prompt += """
4. 贴题性 [贴题性分数]"""

        system_prompt += """

## 结构可视化

### 论点

1. [论点1]
2. [论点2]
3. [论点3]

### 结论

* [结论要点]

## 优点

* [优点1]
* [优点2]
* [优点3]

## 改进意见

* [改进意见1]
* [改进意见2]
* [改进意见3]

评分规则：
1. 逻辑完整性评分：根据演讲的论证逻辑、因果关系、论据支撑等方面评分，每项满分100分
2. 结构可视化：提取演讲的主要论点和结论，清晰展示论证结构
3. 优点：指出演讲中表现出色的地方
4. 改进意见：给出具体可行的改进建议

语速评分标准（中文）：
- 120-180字/分钟：优秀（90-100分）
- 100-120 或 180-200字/分钟：良好（70-89分）
- 80-100 或 200-220字/分钟：一般（50-69分）
- 低于80 或 高于220字/分钟：较差（0-49分）

语速评分标准（英文）：
- 100-150词/分钟：优秀（90-100分）
- 80-100 或 150-180词/分钟：良好（70-89分）
- 60-80 或 180-200词/分钟：一般（50-69分）
- 低于60 或 高于200词/分钟：较差（0-49分）"""

        if topic is not None:
            system_prompt += """

贴题性评分标准：
- 内容与主题高度相关，论点紧扣主题：90-100分
- 内容基本围绕主题，偶有偏离：70-89分
- 内容部分相关，有明显跑题：50-69分
- 内容与主题关联度低：0-49分"""

        system_prompt += """

注意：
- 必须严格按照上述Markdown格式输出，结论是结构可视化的子标题(###)
- 评分要客观公正，有理有据
- 改进意见要具体、可操作
- 结合语音评分数据（发音准确度、流利度等）进行综合评价"""

        user_prompt = f"""请根据以下信息生成演讲评测报告：

## 语音转文字内容

{speech_text}

## 语音评分数据

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分
- 总体评分: {speech_scores.get('overall_score', 0)}分
"""

        if speech_rate is not None:
            user_prompt += f"""
## 语速信息

- 语速: {speech_rate} 字/分钟（或词/分钟）
- 音频时长: {(audio_duration or 0):.1f} 秒
"""

        if topic:
            user_prompt += f"""
## 演讲主题

主题：{topic}

请分析演讲内容与该主题的贴题性，并在评分中体现。"""

        if statistics:
            user_prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words:
            user_prompt += """
## 发音待改进的字词（准确度<90分）

| 字词 | 准确度 | 流利度 |
|------|--------|--------|
"""
            for word in low_score_words[:20]:
                user_prompt += f"| {word.get('word', '')} | {word.get('accuracy', 0)} | {word.get('fluency', 0)} |\n"

        if custom_prompt:
            user_prompt += f"""
## 额外评测要求

{custom_prompt}
"""

        user_prompt += """
请严格按照系统提示中指定的Markdown格式生成评测报告。"""

        if topic:
            user_prompt += "请特别关注内容与主题的贴题性分析。"

        if low_score_words:
            user_prompt += "在改进意见中，请特别关注发音待改进的字词。"

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

    async def generate_simple_report_json(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Generate simple report in JSON format."""
        system_prompt = simple_report_system_prompt(language=language)
        user_prompt = simple_report_user_prompt(
            speech_text, speech_scores, low_score_words,
            speech_rate, audio_duration, language
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        try:
            return extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}, content={content[:300]}")
            return {
                "speech_rate": {
                    "rate": speech_rate or 0,
                    "score": 0,
                    "level": "未知",
                    "suggestion": "无法解析AI响应"
                },
                "weak_paragraphs": [],
                "overall_suggestion": content
            }

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
        """Generate full report in JSON format."""
        system_prompt = full_report_system_prompt(language=language, has_topic=topic is not None)
        user_prompt = full_report_user_prompt(
            speech_text, speech_scores, low_score_words, statistics,
            topic, speech_rate, audio_duration, language
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        try:
            return extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}, content={content[:300]}")
            return {
                "logic_completeness": {
                    "overall_score": 0,
                    "logic_score": 0,
                    "fluency_score": 0,
                    "speech_rate_score": 0,
                    "speech_rate_value": speech_rate or 0,
                    "speech_rate_level": "未知",
                    "speech_rate_suggestion": "无法解析AI响应"
                },
                "structure_visualization": {"arguments": [], "conclusion": ""},
                "speech_rate_evaluation": {"score": 0, "rate_value": speech_rate or 0, "level": "未知", "analysis": "无法解析AI响应", "suggestion": ""},
                "content_perspective": {"score": 0, "topic_relevance": "", "depth": "", "coverage": "", "suggestion": "无法解析AI响应"},
                "logic_structure": {"score": 0, "organization": "", "coherence": "", "reasoning": "", "suggestion": "无法解析AI响应"},
                "expression_wording": {"score": 0, "vocabulary_level": "", "expression_style": "", "highlights": [], "suggestion": "无法解析AI响应"},
                "strengths": [],
                "improvements": [content],
                "weak_paragraphs": []
            }

    async def analyze_text_structure(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Analyze text structure: core ideas, logical structure, key points."""
        system_prompt = text_structure_system_prompt()
        user_prompt = text_structure_user_prompt(text=text, custom_prompt=custom_prompt)

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

    async def analyze_tongue_twister(
        self,
        text: str,
        language: str = "zh"
    ) -> str:
        """Analyze tongue twister pronunciation key points."""
        # Use the tongue twister analysis prompt from the prompt module
        system_prompt = tongue_twister_system_prompt()
        user_prompt = f"""请分析以下绕口令的发音要点：

## 绕口令内容

{text}

请严格按照JSON格式输出分析结果，重点分析核心音素、声学特征差异和发音技巧。"""

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

    async def analyze_sentence_interpretation(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Analyze sentence for reading interpretation."""
        system_prompt = sentence_interpretation_system_prompt()
        user_prompt = sentence_interpretation_user_prompt(speech_text=text)

        if custom_prompt:
            user_prompt += f"\n\n## 额外分析要求\n\n{custom_prompt}"

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

    async def analyze_story_reading(
        self,
        speech_text: str,
        story_text: str,
        word_info_list: Optional[list] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """Analyze story reading performance."""
        system_prompt = story_reading_system_prompt()
        user_prompt = story_reading_user_prompt(
            speech_text=speech_text,
            reference_text=story_text,
            word_info_list=word_info_list,
            audio_duration=audio_duration,
            language=language,
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        try:
            return extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}, content={content[:300]}")
            return {
                "structure_analysis": {"opening": "无法解析", "development": "无法解析", "climax": "无法解析", "ending": "无法解析", "overall_assessment": "无法解析AI响应"},
                "logic_analysis": {"time_jumps": 0, "causal_errors": 0, "missing_events": 0, "logical_contradictions": 0, "overall_assessment": "无法解析AI响应"},
                "fluency_analysis": {"long_pauses_count": 0, "long_pauses": [], "repetition_count": 0, "filler_words_count": 0, "sentence_completion_rate": 0, "overall_assessment": "无法解析AI响应"},
                "event_distribution": {"events": [], "transition_time": "无法解析", "overall_assessment": "无法解析AI响应"},
                "improvements": ["无法解析AI响应，请稍后重试"],
                "overall_score": {"score": 0, "level": "需改进", "comment": "无法解析AI响应"}
            }

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
        scores_data = scores_data or {}
        statistics_data = statistics_data or {}

        if eval_type == "article":
            system_prompt = article_reading_system_prompt()
            user_prompt = article_reading_user_prompt(
                speech_text=speech_text,
                reference_text=tongue_twister_text,
                speech_scores=scores_data,
                statistics=statistics_data,
                word_info_list=word_info_list,
                low_score_words=low_score_words,
                language=language,
            )
        else:
            system_prompt = tongue_twister_reading_system_prompt()
            user_prompt = tongue_twister_reading_user_prompt(
                speech_text=speech_text,
                reference_text=tongue_twister_text,
                speech_scores=scores_data,
                low_score_words=low_score_words,
                statistics=statistics_data,
                word_info_list=word_info_list,
                language=language,
            )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        try:
            return extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}, content={content[:300]}")
            if eval_type == "article":
                return self._default_article_result()
            return self._default_tongue_twister_result()

    async def generate_opinion_statement_report(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None,
        topic: Optional[str] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        word_info_list: Optional[list] = None,
        language: str = "zh"
    ) -> dict:
        """Generate one-minute opinion statement evaluation report (JSON format)."""
        system_prompt = opinion_statement_system_prompt(language=language, has_topic=topic is not None)
        user_prompt = opinion_statement_user_prompt(
            speech_text=speech_text,
            speech_scores=speech_scores,
            word_info_list=word_info_list,
            low_score_words=low_score_words,
            statistics=statistics,
            topic=topic,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            language=language,
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        try:
            return extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}, content={content[:300]}")
            return self._default_opinion_statement_result(audio_duration)

    async def generate_impromptu_reaction_report(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None,
        scenario: Optional[str] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        word_info_list: Optional[list] = None,
        language: str = "zh"
    ) -> dict:
        """Generate impromptu reaction evaluation report (JSON format)."""
        system_prompt = impromptu_reaction_system_prompt(language=language, has_scenario=scenario is not None)
        user_prompt = impromptu_reaction_user_prompt(
            speech_text=speech_text,
            speech_scores=speech_scores,
            word_info_list=word_info_list,
            low_score_words=low_score_words,
            statistics=statistics,
            scenario=scenario,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            language=language,
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        try:
            return extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}, content={content[:300]}")
            return self._default_impromptu_reaction_result(audio_duration)

    # ─── Default result structures ──────────────────────────────────────────

    def _default_tongue_twister_result(self) -> dict:
        """Default return structure for tongue twister evaluation."""
        return {
            "strengths": [],
            "improvements": {
                "extra_words": {"count": 0, "words": [], "description": "无法解析AI响应"},
                "pronunciation_issues": []
            },
            "fluency_analysis": {
                "overall_fluency": "无法解析",
                "long_pauses": [],
                "rhythm_assessment": "无法解析AI响应",
                "speed_assessment": "无法解析AI响应"
            },
            "overall_assessment": "无法解析AI响应，请稍后重试",
            "practice_suggestions": []
        }

    def _default_article_result(self) -> dict:
        """Default return structure for article reading evaluation."""
        return {
            "improvements": {
                "extra_words": {"count": 0, "words": [], "description": "无法解析AI响应"},
                "wrong_words": [],
                "pronunciation_issues": []
            },
            "fluency_analysis": {
                "score": 0,
                "overall_fluency": "无法解析",
                "interruptions": [],
                "repeated_reads": [],
                "stutters": []
            },
            "speech_rate_analysis": {
                "overall_rate": 0,
                "rate_level": "无法解析",
                "standard_range": "180-240字/分钟",
                "segment_rates": [],
                "fast_segments": [],
                "slow_segments": [],
                "suggestion": "无法解析AI响应"
            },
            "pause_analysis": {
                "proper_pauses": 0,
                "improper_pauses": [],
                "missed_pauses": [],
                "overall_assessment": "无法解析AI响应"
            },
            "overall_assessment": "无法解析AI响应，请稍后重试",
            "practice_suggestions": []
        }

    def _default_opinion_statement_result(self, audio_duration=None) -> dict:
        """Default return structure for opinion statement evaluation."""
        return {
            "viewpoint_analysis": {
                "has_clear_viewpoint": False,
                "viewpoint_summary": "无法解析AI响应",
                "opening_type": "未知",
                "opening_quote": "",
                "evasion_signals": [],
                "score": 0,
                "assessment": "无法解析AI响应"
            },
            "structure_completeness": {
                "score": 0,
                "has_viewpoint": False,
                "has_reason": False,
                "has_example": False,
                "has_summary": False,
                "structure_pattern": "无法解析",
                "ideal_pattern": "观点→理由→举例→总结",
                "missing_parts": [],
                "assessment": "无法解析AI响应"
            },
            "logic_clarity": {
                "score": 0,
                "logic_jumps": [],
                "contradictions": [],
                "argument_piling": {"detected": False, "description": "无法解析AI响应"},
                "reasoning_chain": "",
                "assessment": "无法解析AI响应"
            },
            "time_rhythm": {
                "score": 0,
                "total_duration_seconds": audio_duration or 0,
                "duration_level": "未知",
                "first_half_rate": 0,
                "second_half_rate": 0,
                "rate_change": "未知",
                "panic_acceleration": False,
                "time_allocation": {"opening_seconds": 0, "body_seconds": 0, "closing_seconds": 0, "assessment": "无法解析AI响应"},
                "assessment": "无法解析AI响应"
            },
            "expression_redundancy": {
                "score": 0,
                "filler_words": [],
                "total_filler_count": 0,
                "filler_ratio": "无法解析",
                "redundant_expressions": [],
                "effective_content_ratio": "无法解析",
                "assessment": "无法解析AI响应"
            },
            "overall_scores": {
                "overall_score": 0,
                "viewpoint_score": 0,
                "structure_score": 0,
                "logic_score": 0,
                "fluency_score": 0,
                "speech_rate_score": 0,
                "expression_score": 0,
                "time_rhythm_score": 0,
                "pronunciation_accuracy": 0,
                "pronunciation_fluency": 0,
                "pronunciation_completion": 0,
                "suggested_score": 0,
                "speech_rate_value": 0,
                "speech_rate_level": "未知",
                "speech_rate_suggestion": "",
                "level": "需改进",
                "one_sentence_comment": "无法解析AI响应"
            },
            "structure_visualization": {"arguments": [], "conclusion": ""},
            "strengths": [],
            "improvements": ["无法解析AI响应，请稍后重试"],
            "practice_tips": []
        }

    def _default_impromptu_reaction_result(self, audio_duration=None) -> dict:
        """Default return structure for impromptu reaction evaluation."""
        return {
            "reaction_speed": {
                "first_word_time_ms": 0,
                "opening_speed": "未知",
                "panic_signals": False,
                "thinking_pauses": [],
                "assessment": "无法解析AI响应"
            },
            "structure_formation": {
                "formed_in_15s": False,
                "structure_signal": "无法解析",
                "structure_pattern": "未知",
                "has_opening": False,
                "has_body": False,
                "has_closing": False,
                "assessment": "无法解析AI响应"
            },
            "content_relevance": {
                "topic_relevance": "未知",
                "is_mere_repetition": False,
                "repetition_ratio": "0%",
                "has_original_response": False,
                "on_topic": False,
                "topic_drift": False,
                "off_topic_parts": [],
                "content_depth": "未知",
                "relevance_description": "无法解析AI响应",
                "assessment": "无法解析AI响应"
            },
            "logic_coherence": {
                "coherence_level": "未知",
                "logic_jumps": [],
                "transition_quality": "无法解析AI响应",
                "assessment": "无法解析AI响应"
            },
            "expression_redundancy": {
                "filler_words": [],
                "total_filler_count": 0,
                "filler_ratio": "无法解析",
                "redundancy_level": "未知",
                "effective_content_ratio": "无法解析",
                "assessment": "无法解析AI响应"
            },
            "overall_scores": {
                "overall_score": 0,
                "reaction_speed_score": 0,
                "content_relevance_score": 0,
                "logic_coherence_score": 0,
                "fluency_score": 0,
                "expression_score": 0,
                "structure_score": 0,
                "pronunciation_accuracy": 0,
                "pronunciation_fluency": 0,
                "pronunciation_completion": 0,
                "suggested_score": 0,
                "speech_rate_value": 0,
                "speech_rate_level": "未知",
                "level": "需改进",
                "one_sentence_comment": "无法解析AI响应"
            },
            "structure_visualization": {"key_points": [], "conclusion": ""},
            "strengths": [],
            "improvements": ["无法解析AI响应，请稍后重试"],
            "next_action": "无法解析AI响应"
        }


# Singleton instance - backend determined by LLM_PROVIDER config
if settings.llm_provider == "openai":
    hunyuan_service = HunyuanService(
        backend="openai",
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=getattr(settings, "hunyuan_timeout", settings.llm_timeout)
    )
else:
    hunyuan_service = HunyuanService()
