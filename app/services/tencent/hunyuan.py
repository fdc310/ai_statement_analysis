"""
Tencent Cloud Hunyuan LLM service with async support.
"""
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator

from tencentcloud.hunyuan.v20230901 import hunyuan_client_async, models
from tencentcloud.common.exception import TencentCloudSDKException

from app.core.config import settings
from app.services.tencent.base import TencentCloudClient

logger = logging.getLogger(__name__)


class HunyuanService(TencentCloudClient):
    """Tencent Cloud Hunyuan LLM service with async support."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        super().__init__(secret_id, secret_key, "hunyuan.tencentcloudapi.com")
        self.model = model or settings.hunyuan_model
        self.timeout = timeout or settings.hunyuan_timeout

    def _create_async_client(self) -> hunyuan_client_async.HunyuanClient:
        """Create a new async Hunyuan client for each request."""
        # Create new client each time since async with closes the client after use
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
        logger.info(f"Starting chat request with model={self.model}, messages_count={len(messages)}, timeout={timeout}")
        
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
        logger.info(f"Starting stream request with model={self.model}, messages_count={len(messages)}, timeout={timeout}")
        
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
        """Parse chat completion result."""
        choices = result.get("Choices", [])
        content = ""
        if choices:
            message = choices[0].get("Message", {})
            content = message.get("Content", "")

        usage = result.get("Usage", {})

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("PromptTokens", 0),
                "completion_tokens": usage.get("CompletionTokens", 0),
                "total_tokens": usage.get("TotalTokens", 0)
            },
            "raw_response": result
        }

    async def generate_evaluation(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None
    ) -> str:
        """Generate speech evaluation report in Markdown format (async)."""
        system_prompt = self._build_evaluation_system_prompt()
        user_prompt = self._build_evaluation_user_prompt(
            speech_text, speech_scores, custom_prompt,
            low_score_words, statistics
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

    def _build_evaluation_system_prompt(self) -> str:
        """Build the system prompt for evaluation."""
        return """你是语音演讲评测专家。严格按照以下Markdown格式输出评测报告：

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

每项满分100分。结论是结构可视化的子标题(###)。评分客观公正，改进意见具体可操作，结合语音评分数据综合评价。"""

    def _build_evaluation_user_prompt(
        self,
        speech_text: str,
        speech_scores: dict,
        custom_prompt: Optional[str] = None,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None
    ) -> str:
        """Build the user prompt for evaluation."""
        prompt = f"""请根据以下信息生成演讲评测报告：

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
            prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0)}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words and len(low_score_words) > 0:
            prompt += """
## 发音待改进的字词（准确度<90分）

| 字词 | 准确度 | 流利度 |
|------|--------|--------|
"""
            for word in low_score_words[:20]:
                prompt += f"| {word.get('word', '')} | {word.get('accuracy', 0)} | {word.get('fluency', 0)} |\n"

        if custom_prompt:
            prompt += f"""
## 额外评测要求

{custom_prompt}
"""

        prompt += """
严格按照系统提示的Markdown格式输出。"""

        return prompt

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
        """
        Generate extended speech evaluation report with topic relevance and speech rate analysis.

        Args:
            speech_text: Transcribed speech text
            speech_scores: Speech evaluation scores
            custom_prompt: Optional custom evaluation prompt
            low_score_words: List of words with low scores
            statistics: Evaluation statistics
            topic: Speech topic for relevance analysis (None for free speech)
            speech_rate: Speech rate (chars/min or words/min)
            audio_duration: Audio duration in seconds
        """
        system_prompt = self._build_extended_system_prompt(topic is not None)
        user_prompt = self._build_extended_user_prompt(
            speech_text, speech_scores, custom_prompt,
            low_score_words, statistics, topic, speech_rate, audio_duration
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result["content"]

    def _build_extended_system_prompt(self, has_topic: bool) -> str:
        """Build the system prompt for extended evaluation."""
        base_prompt = """你是语音演讲评测专家。严格按照以下Markdown格式输出评测报告：

# 评分

## 逻辑完整性评分
* 综合评分 [综合评分分数]
1. 逻辑性 [逻辑性分数]
2. 流畅度 [流畅度分数]
3. 语速 [语速分数]"""

        if has_topic:
            base_prompt += """
4. 贴题性 [贴题性分数]"""

        base_prompt += """

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

每项满分100分。语速标准：中文120-180优秀(90-100), 100-120/180-200良好(70-89), 80-100/200-220一般(50-69), 其他较差(0-49)。
英文：100-150优秀, 80-100/150-180良好, 60-80/180-200一般, 其他较差。"""

        if has_topic:
            base_prompt += """
贴题性：高度相关90-100, 基本围绕70-89, 部分相关50-69, 关联度低0-49。"""

        base_prompt += """
结论是结构可视化的子标题(###)。评分客观公正，改进意见具体可操作，结合语音评分数据综合评价。"""

        return base_prompt

    def _build_extended_user_prompt(
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
        """Build the user prompt for extended evaluation."""
        prompt = f"""请根据以下信息生成演讲评测报告：

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
            prompt += f"""
## 语速信息

- 语速: {speech_rate} 字/分钟（或词/分钟）
- 音频时长: {audio_duration:.1f} 秒
"""

        if topic:
            prompt += f"""
## 演讲主题

主题：{topic}

请分析演讲内容与该主题的贴题性，并在评分中体现。
"""

        if statistics:
            prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words and len(low_score_words) > 0:
            prompt += """
## 发音待改进的字词（准确度<90分）

| 字词 | 准确度 | 流利度 |
|------|--------|--------|
"""
            for word in low_score_words[:20]:
                prompt += f"| {word.get('word', '')} | {word.get('accuracy', 0)} | {word.get('fluency', 0)} |\n"

        if custom_prompt:
            prompt += f"""
## 额外评测要求

{custom_prompt}
"""

        prompt += """
严格按照系统提示的Markdown格式输出。"""

        if topic:
            prompt += "请特别关注内容与主题的贴题性分析。"

        if low_score_words:
            prompt += "在改进意见中，请特别关注发音待改进的字词。"

        return prompt

    async def generate_simple_report_json(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> dict:
        """
        生成简洁报告（JSON格式）
        包含：语速评分、语速评价、低分段落分析
        """
        system_prompt = self._build_simple_report_system_prompt(language)
        user_prompt = self._build_simple_report_user_prompt(
            speech_text, speech_scores, low_score_words,
            speech_rate, audio_duration, language
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        # 解析 JSON
        try:
            # 尝试从响应中提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结构
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

    def _build_simple_report_system_prompt(self, language: str) -> str:
        """构建简洁报告的系统提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"
        return f"""你是语音演讲评测专家，生成简洁评测报告。只输出纯JSON：

{{
    "speech_rate": {{
        "rate": <语速数值>,
        "score": <语速评分0-100>,
        "level": "<优秀/良好/一般/较差>",
        "suggestion": "<语速建议，简短一句话>"
    }},
    "weak_paragraphs": [
        {{
            "paragraph_index": <段落索引从1开始>,
            "content": "<段落内容摘要，不超过50字>",
            "low_score_words": [
                {{"word": "<字词>", "accuracy": <准确度分数>}}
            ],
            "suggestion": "<针对该段落的改进建议>"
        }}
    ],
    "overall_suggestion": "<整体简短建议，不超过100字>"
}}

语速评分标准（{rate_unit}）：
- 中文：120-180优秀(90-100), 100-120/180-200良好(70-89), 80-100/200-220一般(50-69), 其他较差(0-49)
- 英文：100-150优秀(90-100), 80-100/150-180良好(70-89), 60-80/180-200一般(50-69), 其他较差(0-49)

weak_paragraphs只含有低分字词的段落（最多3个），无低分字词则为空数组。
只输出纯JSON，不要添加markdown代码块标记。"""

    def _build_simple_report_user_prompt(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> str:
        """构建简洁报告的用户提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"
        prompt = f"""请分析以下语音内容并生成简洁报告：

## 语音转文字内容

{speech_text}

## 语速信息

- 语速: {speech_rate or 0} {rate_unit}
- 音频时长: {audio_duration or 0:.1f} 秒
"""

        if low_score_words and len(low_score_words) > 0:
            prompt += """
## 低分字词列表

"""
            for word in low_score_words[:20]:
                prompt += f"- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分\n"

        prompt += """
请根据低分字词的位置，判断哪些段落读的不太好，并给出改进建议。
严格按照系统提示的JSON格式输出。"""

        return prompt

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
        """
        生成完整报告（JSON格式）
        包含：语速、内容角度、逻辑与结构、表达与用词
        """
        system_prompt = self._build_full_report_system_prompt(language, topic is not None)
        user_prompt = self._build_full_report_user_prompt(
            speech_text, speech_scores, low_score_words, statistics,
            topic, speech_rate, audio_duration, language
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        # 解析 JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结构
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
                "structure_visualization": {
                    "arguments": [],
                    "conclusion": ""
                },
                "speech_rate_evaluation": {
                    "score": 0,
                    "rate_value": speech_rate or 0,
                    "level": "未知",
                    "analysis": "无法解析AI响应",
                    "suggestion": ""
                },
                "content_perspective": {
                    "score": 0,
                    "topic_relevance": "",
                    "depth": "",
                    "coverage": "",
                    "suggestion": "无法解析AI响应"
                },
                "logic_structure": {
                    "score": 0,
                    "organization": "",
                    "coherence": "",
                    "reasoning": "",
                    "suggestion": "无法解析AI响应"
                },
                "expression_wording": {
                    "score": 0,
                    "vocabulary_level": "",
                    "expression_style": "",
                    "highlights": [],
                    "suggestion": "无法解析AI响应"
                },
                "strengths": [],
                "improvements": [content],
                "weak_paragraphs": []
            }

    def _build_full_report_system_prompt(self, language: str, has_topic: bool) -> str:
        """构建完整报告的系统提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"

        topic_field = ""
        if has_topic:
            topic_field = '"topic_relevance_score": <贴题性评分0-100>,'

        return f"""你是语音演讲评测专家，生成完整评测报告。只输出纯JSON：

{{
    "logic_completeness": {{
        "overall_score": <综合评分0-100，加权：内容角度20%+逻辑结构20%+表达用词20%+流畅度15%+语速15%+口头禅控制10%>,
        "logic_score": <逻辑性评分0-100>,
        "fluency_score": <流畅度评分0-100>,
        "speech_rate_score": <语速评分0-100>,
        {topic_field}
        "speech_rate_value": <语速数值>,
        "speech_rate_level": "<优秀/良好/一般/较差>",
        "speech_rate_suggestion": "<语速建议>"
    }},
    "structure_visualization": {{
        "arguments": ["<论点1>", "<论点2>", "<论点3>"],
        "conclusion": "<结论要点>"
    }},
    "speech_rate_evaluation": {{
        "score": <语速评分0-100>,
        "rate_value": <语速数值>,
        "level": "<优秀/良好/一般/较差>",
        "analysis": "<语速分析，描述语速快慢对表达的影响>",
        "suggestion": "<语速改进建议>"
    }},
    "content_perspective": {{
        "score": <内容角度评分0-100>,
        "topic_relevance": "<贴题性分析，如无主题则分析内容主旨>",
        "depth": "<内容深度分析>",
        "coverage": "<内容覆盖面分析>",
        "suggestion": "<内容改进建议>"
    }},
    "logic_structure": {{
        "score": <逻辑结构评分0-100>,
        "organization": "<整体结构分析>",
        "coherence": "<连贯性分析>",
        "reasoning": "<论证逻辑分析>",
        "suggestion": "<逻辑结构改进建议>"
    }},
    "expression_wording": {{
        "score": <表达用词评分0-100>,
        "vocabulary_level": "<用词水平分析>",
        "expression_style": "<表达风格分析>",
        "highlights": ["<表达亮点1>", "<表达亮点2>"],
        "suggestion": "<表达用词改进建议>"
    }},
    "verbal_habits": {{
        "filler_words": [
            {{"word": "<口头禅/填充词>", "count": <出现次数>, "example_context": "<出现的上下文示例>"}}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<口头禅占比描述，如每分钟X次>",
        "impact_assessment": "<口头禅对表达效果的影响分析>",
        "suggestion": "<减少口头禅的具体建议>"
    }},
    "main_issues": [
        {{
            "issue_type": "<问题类型>",
            "description": "<问题具体描述>",
            "impact_level": "<高/中/低>",
            "example": "<具体例子>",
            "suggested_fix": "<具体改进建议>"
        }}
    ],
    "strengths": ["<优点1>", "<优点2>", "<优点3>"],
    "improvements": ["<改进意见1>", "<改进意见2>", "<改进意见3>"],
    "weak_paragraphs": [
        {{
            "paragraph_index": <段落索引从1开始>,
            "content": "<段落内容摘要>",
            "low_score_words": [
                {{"word": "<字词>", "accuracy": <准确度分数>}}
            ],
            "suggestion": "<改进建议>"
        }}
    ]
}}

语速评分标准（{rate_unit}）：
- 中文：120-180优秀(90-100), 100-120/180-200良好(70-89), 80-100/200-220一般(50-69), 其他较差(0-49)
- 英文：100-150优秀(90-100), 80-100/150-180良好(70-89), 60-80/180-200一般(50-69), 其他较差(0-49)

口头禅识别：嗯/啊/那个/就是/然后/对吧/这个等，评分：无(90-100), 偶有(70-89), 较频繁(50-69), 严重(0-49)。
优点每条不少于15字，要具体引用原文。weak_paragraphs只含有低分字词的段落。
只输出纯JSON，所有评分0-100分。"""

    def _build_full_report_user_prompt(
        self,
        speech_text: str,
        speech_scores: dict,
        low_score_words: Optional[list] = None,
        statistics: Optional[dict] = None,
        topic: Optional[str] = None,
        speech_rate: Optional[float] = None,
        audio_duration: Optional[float] = None,
        language: str = "zh"
    ) -> str:
        """构建完整报告的用户提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"
        prompt = f"""请分析以下语音内容并生成完整报告：

## 语音转文字内容

{speech_text}

## 语音评分数据

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分

## 语速信息

- 语速: {speech_rate or 0} {rate_unit}
- 音频时长: {audio_duration or 0:.1f} 秒
"""

        if topic:
            prompt += f"""
## 演讲主题

主题：{topic}
请分析演讲内容与该主题的贴题性。
"""

        if statistics:
            prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words and len(low_score_words) > 0:
            prompt += """
## 低分字词列表

"""
            for word in low_score_words[:20]:
                prompt += f"- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分, 流利度{word.get('fluency', 0)}分\n"

        prompt += """
严格按照系统提示的JSON格式输出。"""

        return prompt


    async def analyze_text_structure(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze text structure: core ideas, logical structure, key points.

        Args:
            text: Text content to analyze
            custom_prompt: Optional custom analysis requirements

        Returns:
            Analysis result in Markdown format
        """
        system_prompt = """你是文本分析专家，提取核心思想和逻辑结构。只输出纯JSON：

{{
    "core_idea": "文本的核心思想/主旨，用一两句话概括",
    "key_points": [
        {{
            "title": "要点标题",
            "content": "要点详细内容",
            "importance": "高/中/低"
        }}
    ],
    "logical_structure": {{
        "type": "结构类型（如：总分总、递进式、并列式、对比式、因果式等）",
        "description": "对逻辑结构的简要说明",
        "outline": [
            {{
                "level": 1,
                "title": "一级标题/段落主题",
                "summary": "该部分的简要概括",
                "sub_points": [
                    {{
                        "level": 2,
                        "title": "二级要点",
                        "summary": "要点说明"
                    }}
                ]
            }}
        ]
    }},
    "arguments": [
        {{
            "claim": "论点/观点",
            "evidence": "支撑论据",
            "reasoning": "论证逻辑"
        }}
    ],
    "conclusion": "结论或总结",
    "writing_style": "写作风格特点",
    "suggestions": [
        "改进建议1",
        "改进建议2"
    ]
}}

某些部分在文本中不明显时可标注为null或空数组。分析要客观中立。
只输出纯JSON，不要添加markdown代码块标记。"""

        user_prompt = f"""请分析以下文本的核心思想和逻辑结构：

## 待分析文本

{text}
"""

        if custom_prompt:
            user_prompt += f"""
## 额外分析要求

{custom_prompt}
"""

        user_prompt += """
请严格按照JSON格式输出分析结果。"""

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
        """
        Analyze tongue twister pronunciation key points.

        Args:
            text: Tongue twister text to analyze
            language: Language code ('zh' for Chinese, 'en' for English)

        Returns:
            Analysis result in JSON format
        """
        if language == "zh":
            system_prompt = """你是语音学和发音教学专家，分析绕口令发音要点。只输出纯JSON：

{{
    "title": "绕口令标题/主题",
    "difficulty": "难度等级（简单/中等/困难/专家）",
    "core_phonemes": [
        {{
            "phoneme": "音素（如：b、p、m、f等）",
            "pinyin": "对应拼音",
            "ipa": "国际音标",
            "description": "发音描述",
            "articulation": {{
                "manner": "发音方式（如：爆破音、摩擦音、鼻音等）",
                "place": "发音部位（如：双唇、舌尖、舌根等）",
                "voicing": "清浊（清音/浊音）"
            }},
            "examples": ["包含该音素的字词示例"]
        }}
    ],
    "acoustic_features": [
        {{
            "feature": "声学特征名称",
            "description": "特征描述",
            "key_difference": "与易混淆音的关键差异",
            "measurement": "可量化的声学指标（如：VOT、F1/F2频率等）"
        }}
    ],
    "confusion_pairs": [
        {{
            "pair": ["音素1", "音素2"],
            "difference": "区分要点",
            "common_errors": "常见错误",
            "practice_tip": "练习建议"
        }}
    ],
    "pronunciation_tips": [
        {{
            "tip": "发音提示",
            "target_sounds": ["针对的音素"],
            "technique": "具体技巧",
            "practice_method": "练习方法"
        }}
    ],
    "rhythm_pattern": {{
        "beat_count": "节拍数",
        "stress_pattern": "重音模式",
        "pause_points": ["建议停顿位置"],
        "speed_suggestion": "建议语速"
    }},
    "practice_sequence": [
        {{
            "step": 1,
            "focus": "练习重点",
            "content": "练习内容",
            "repetitions": "建议重复次数"
        }}
    ],
    "annotated_text": "带音素标注的文本（用[]标注核心音素）"
}}

基于语音学原理分析，建议要具体可操作。只输出纯JSON，不要添加markdown代码块标记。"""
        else:
            system_prompt = """You are a phonetics and pronunciation teaching expert. Analyze tongue twister pronunciation key points. Only output JSON:

{{
    "title": "Tongue twister title/theme",
    "difficulty": "Difficulty level (Easy/Medium/Hard/Expert)",
    "core_phonemes": [
        {{
            "phoneme": "Phoneme (e.g., /p/, /b/, /θ/, /ð/)",
            "ipa": "IPA symbol",
            "description": "Pronunciation description",
            "articulation": {{
                "manner": "Manner of articulation (e.g., plosive, fricative, nasal)",
                "place": "Place of articulation (e.g., bilabial, alveolar, velar)",
                "voicing": "Voiced/Voiceless"
            }},
            "examples": ["Example words containing this phoneme"]
        }}
    ],
    "acoustic_features": [
        {{
            "feature": "Acoustic feature name",
            "description": "Feature description",
            "key_difference": "Key difference from similar sounds",
            "measurement": "Measurable acoustic indicators (e.g., VOT, F1/F2 frequency)"
        }}
    ],
    "confusion_pairs": [
        {{
            "pair": ["phoneme1", "phoneme2"],
            "difference": "Key distinction",
            "common_errors": "Common mistakes",
            "practice_tip": "Practice suggestion"
        }}
    ],
    "pronunciation_tips": [
        {{
            "tip": "Pronunciation tip",
            "target_sounds": ["Target phonemes"],
            "technique": "Specific technique",
            "practice_method": "Practice method"
        }}
    ],
    "rhythm_pattern": {{
        "beat_count": "Number of beats",
        "stress_pattern": "Stress pattern",
        "pause_points": ["Suggested pause positions"],
        "speed_suggestion": "Suggested speed"
    }},
    "practice_sequence": [
        {{
            "step": 1,
            "focus": "Practice focus",
            "content": "Practice content",
            "repetitions": "Suggested repetitions"
        }}
    ],
    "annotated_text": "Text with phoneme annotations (mark core phonemes with [])"
}}

Analysis based on phonetic principles. Only output pure JSON, no markdown code blocks."""

        user_prompt = f"""请分析以下绕口令的发音要点：

{text}

严格按照JSON格式输出。"""

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
        """
        Analyze sentence for reading interpretation.

        Args:
            text: Sentence to analyze
            custom_prompt: Optional custom analysis requirements

        Returns:
            Analysis result in JSON format
        """
        system_prompt = """你是语文朗读指导专家，分析句子并提供朗读指导。只输出纯JSON：

{{
    "center_content": "<句子的中心内容/主旨>",
    "reading_points": [
        "<朗读重点1>",
        "<朗读重点2>",
        "<朗读重点3>"
    ],
    "reading_notes": [
        "<注意事项1>",
        "<注意事项2>",
        "<注意事项3>"
    ]
}}

朗读重点：需重读/强调的关键词或短语（最多3-5个）。
注意事项：语气、停顿、语速、情感等要点（最多3-5条），每条不超过20字。
只输出纯JSON，不要添加markdown代码块标记。"""

        user_prompt = f"""请分析以下句子的朗读要点：

## 待分析句子

{text}
"""

        if custom_prompt:
            user_prompt += f"""
## 额外分析要求

{custom_prompt}
"""

        user_prompt += """
请严格按照JSON格式输出分析结果。"""

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
        """
        Analyze story reading performance.

        Args:
            speech_text: Transcribed speech text
            story_text: Reference story text
            word_info_list: List of word-level timestamp data from ASR
            audio_duration: Audio duration in seconds
            language: Language code ('zh' for Chinese, 'en' for English)

        Returns:
            Analysis result in JSON format with:
            - structure_analysis: Structure completeness analysis
            - logic_analysis: Logic coherence analysis
            - fluency_analysis: Language fluency analysis
            - event_distribution: Event time distribution
            - improvements: Suggestions for improvement
        """
        # Build timestamp context
        timestamp_info = ""
        if word_info_list and len(word_info_list) > 0:
            timestamp_info = "\n## 词级别时间戳信息\n\n| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |\n|------|-------------|-------------|-------------|\n"
            for w in word_info_list[:100]:  # Limit to first 100 words
                timestamp_info += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

        if audio_duration:
            timestamp_info += f"\n总音频时长: {audio_duration:.1f} 秒\n"

        system_prompt = """你是故事阅读评测专家，评估结构完整性、逻辑连贯性、语言流畅度和事件分布。只输出纯JSON：

{
    "structure_analysis": {
        "opening": "<开头情况：有/无，简短描述>",
        "development": "<发展情况>",
        "climax": "<高潮情况：有/无，简短描述>",
        "ending": "<结尾情况：有/无/仓促，简短描述>",
        "overall_assessment": "<整体结构评价>"
    },
    "logic_analysis": {
        "time_jumps": <时间跳跃次数>,
        "causal_errors": <因果错误次数>,
        "missing_events": <事件遗漏次数>,
        "logical_contradictions": <逻辑矛盾次数>,
        "overall_assessment": "<整体逻辑评价>"
    },
    "fluency_analysis": {
        "long_pauses_count": <长停顿(>3秒)次数>,
        "long_pauses": [
            {
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "position_time_ms": <停顿发生的时间点毫秒>
            }
        ],
        "repetition_count": <重复修正次数>,
        "filler_words_count": <填充词次数>,
        "sentence_completion_rate": <句子完整度0-100>,
        "overall_assessment": "<整体流畅度评价>"
    },
    "event_distribution": {
        "events": [
            {
                "name": "<事件名称>",
                "start_time_ms": <开始时间毫秒>,
                "end_time_ms": <结束时间毫秒>,
                "duration_seconds": <持续时间秒>,
                "assessment": "<该事件评价>"
            }
        ],
        "transition_time": "<过渡时间描述>",
        "overall_assessment": "<整体事件分布评价>"
    },
    "improvements": [
        "<改进建议1>",
        "<改进建议2>",
        "<改进建议3>"
    ]
}

分析规则：
- 长停顿：相邻词间隔>3000ms，记录前后词语、时长和时间点
- 填充词：如"啊"、"呃"、"那个"、"这个"、"嗯"等
- 事件分布根据时间戳分析，无明确划分则按内容合理划分
- 无法从时间戳分析某些指标时给出合理推断
只输出纯JSON，不要添加markdown代码块标记。"""

        user_prompt = f"""请分析以下用户的故事阅读表现：

## 原始故事文本

{story_text}

## 用户阅读内容

{speech_text}
{timestamp_info}

请严格按照JSON格式输出分析结果。"""

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        # Parse JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            # Return default structure if parsing fails
            return {
                "structure_analysis": {
                    "opening": "无法解析",
                    "development": "无法解析",
                    "climax": "无法解析",
                    "ending": "无法解析",
                    "overall_assessment": "无法解析AI响应"
                },
                "logic_analysis": {
                    "time_jumps": 0,
                    "causal_errors": 0,
                    "missing_events": 0,
                    "logical_contradictions": 0,
                    "overall_assessment": "无法解析AI响应"
                },
                "fluency_analysis": {
                    "long_pauses_count": 0,
                    "long_pauses": [],
                    "repetition_count": 0,
                    "filler_words_count": 0,
                    "sentence_completion_rate": 0,
                    "overall_assessment": "无法解析AI响应"
                },
                "event_distribution": {
                    "events": [],
                    "transition_time": "无法解析",
                    "overall_assessment": "无法解析AI响应"
                },
                "improvements": ["无法解析AI响应，请稍后重试"]
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
        """
        Analyze reading performance for tongue twisters or articles.

        Args:
            speech_text: Transcribed speech text from ASR
            tongue_twister_text: Original reference text
            word_info_list: Word-level timestamp data from ASR
            low_score_words: Low score words from SOE evaluation
            scores_data: SOE pronunciation scores
            statistics_data: SOE evaluation statistics
            audio_duration: Audio duration in seconds
            language: Language code
            eval_type: Evaluation type - "tongue_twister" or "article"

        Returns:
            Analysis result with strengths, improvements, fluency analysis
        """
        scores_data = scores_data or {}
        statistics_data = statistics_data or {}

        # Build timestamp context
        timestamp_info = ""
        if word_info_list and len(word_info_list) > 0:
            timestamp_info = "\n## 词级别时间戳信息\n\n| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |\n|------|-------------|-------------|-------------|\n"
            for w in word_info_list[:100]:
                timestamp_info += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

        if audio_duration:
            timestamp_info += f"\n总音频时长: {audio_duration:.1f} 秒\n"

        # Build low score words context
        low_score_info = ""
        if low_score_words and len(low_score_words) > 0:
            low_score_info = "\n## 发音待改进的字词（SOE评测低分）\n\n| 字词 | 准确度 | 流利度 |\n|------|--------|--------|\n"
            for word in low_score_words[:30]:
                low_score_info += f"| {word.get('word', '')} | {word.get('accuracy', 0)} | {word.get('fluency', 0)} |\n"

        # Calculate speech rate
        speech_rate_info = ""
        if audio_duration and audio_duration > 0 and statistics_data.get('total_words', 0) > 0:
            speech_rate = statistics_data['total_words'] / audio_duration * 60
            speech_rate_info = f"\n语速: {speech_rate:.0f} 字/分钟\n"

        # Select prompt based on eval_type
        if eval_type == "article":
            system_prompt = self._build_article_reading_system_prompt()
            user_prompt = self._build_article_reading_user_prompt(
                speech_text, tongue_twister_text, scores_data,
                statistics_data, low_score_info, timestamp_info, speech_rate_info
            )
        else:
            system_prompt = self._build_tongue_twister_system_prompt()
            user_prompt = self._build_tongue_twister_user_prompt(
                speech_text, tongue_twister_text, scores_data,
                statistics_data, low_score_info, timestamp_info
            )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        # Parse JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            if eval_type == "article":
                return self._default_article_result()
            return self._default_tongue_twister_result()

    def _build_tongue_twister_system_prompt(self) -> str:
        """构建绕口令评测的系统提示词"""
        return """你是绕口令语音评测专家，对比原始文本和实际朗读内容评估表现。只输出纯JSON：

{
    "strengths": [
        "<优势1：具体描述发音、节奏、流畅度等亮点，每条不少于15字>",
        "<优势2>",
        "<优势3>"
    ],
    "improvements": {
        "extra_words": {
            "count": <多读字词数量>,
            "words": ["<多读的字词1>", "<多读的字词2>"],
            "description": "<对多读情况的简要说明>"
        },
        "missed_words": {
            "count": <漏读字词数量>,
            "words": ["<漏读的字词1>", "<漏读的字词2>"],
            "description": "<对漏读情况的简要说明>"
        },
        "pronunciation_issues": [
            {
                "word": "<发音有问题的字词>",
                "accuracy_score": <SOE准确度评分>,
                "issue_description": "<具体发音问题描述，如声母/韵母/声调问题>",
                "correct_pronunciation": "<正确的发音要领>",
                "practice_tip": "<针对性练习建议>"
            }
        ]
    },
    "fluency_analysis": {
        "overall_fluency": "<整体流畅度评价：优秀/良好/一般/较差>",
        "long_pauses": [
            {
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "suggestion": "<针对该停顿的建议>"
            }
        ],
        "rhythm_assessment": "<节奏评价>",
        "speed_assessment": "<语速评价>"
    },
    "overall_assessment": "<综合评价，50-100字>",
    "practice_suggestions": [
        "<练习建议1：具体可操作>",
        "<练习建议2>",
        "<练习建议3>"
    ]
}

分析规则：
- 多读/漏读：逐字对比原文和朗读内容
- 发音问题：基于SOE低分字词(accuracy<90分)分析声母/韵母/声调
- 长停顿：相邻词间隔>2000ms（绕口令标准更严格）
- 如果某项无问题，保留字段给出正面描述（如count为0）
只输出纯JSON，不要添加markdown代码块标记。"""

    def _build_tongue_twister_user_prompt(
        self, speech_text: str, ref_text: str, scores_data: dict,
        statistics_data: dict, low_score_info: str, timestamp_info: str
    ) -> str:
        """构建绕口令评测的用户提示词"""
        user_prompt = f"""请分析以下用户朗读绕口令的表现：

## 绕口令原文

{ref_text}

## 用户实际朗读内容（ASR识别结果）

{speech_text}

## SOE语音评测评分

- 发音准确度: {scores_data.get('pronunciation_accuracy', 0)}分
- 发音流利度: {scores_data.get('pronunciation_fluency', 0)}分
- 发音完整度: {scores_data.get('pronunciation_completion', 0)}分
- 综合建议分: {scores_data.get('suggested_score', 0)}分
"""

        if statistics_data:
            user_prompt += f"""
## 评分统计

- 总字数: {statistics_data.get('total_words', 0)}
- 平均准确度: {statistics_data.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics_data.get('low_score_count', 0)}个
"""

        user_prompt += low_score_info
        user_prompt += timestamp_info

        user_prompt += """
严格按照系统提示的JSON格式输出。"""
        return user_prompt

    def _build_article_reading_system_prompt(self) -> str:
        """构建文章朗读评测的系统提示词"""
        return """你是文章朗读评测专家，从流畅度、语速、断句停顿、读错漏字等维度评估。只输出纯JSON：

{
    "strengths": [
        "<优势1：具体描述朗读亮点，每条不少于15字>",
        "<优势2>",
        "<优势3>"
    ],
    "improvements": {
        "extra_words": {
            "count": <多读字词数量>,
            "words": ["<多读的字词1>", "<多读的字词2>"],
            "description": "<对多读情况的简要说明>"
        },
        "missed_words": {
            "count": <漏读字词数量>,
            "words": ["<漏读的字词1>", "<漏读的字词2>"],
            "description": "<对漏读情况的简要说明>"
        },
        "wrong_words": [
            {
                "original": "<原文字词>",
                "actual": "<实际读成的字词>",
                "position": "<大致位置描述>"
            }
        ],
        "pronunciation_issues": [
            {
                "word": "<发音有问题的字词>",
                "accuracy_score": <SOE准确度评分>,
                "issue_description": "<具体发音问题描述>",
                "correct_pronunciation": "<正确的发音要领>",
                "practice_tip": "<针对性练习建议>"
            }
        ]
    },
    "fluency_analysis": {
        "score": <流畅度评分0-100>,
        "overall_fluency": "<整体流畅度评价：优秀/良好/一般/较差>",
        "interruptions": [
            {
                "position": "<中断发生的位置描述>",
                "before_word": "<中断前的词语>",
                "after_word": "<中断后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "type": "<类型：异常停顿/重复读/卡壳>"
            }
        ],
        "repeated_reads": [
            {
                "word": "<被重复读的词语或句段>",
                "position": "<位置描述>",
                "count": <重复次数>
            }
        ],
        "stutters": [
            "<明显卡壳的位置和内容描述>"
        ]
    },
    "speech_rate_analysis": {
        "overall_rate": <整体语速，字/分钟>,
        "rate_level": "<偏快/适中/偏慢>",
        "standard_range": "180-240字/分钟",
        "segment_rates": [
            {
                "segment": "<段落描述>",
                "rate": <该段语速>,
                "assessment": "<该段语速评价>"
            }
        ],
        "fast_segments": ["<语速过快的位置描述>"],
        "slow_segments": ["<语速过慢的位置描述>"],
        "suggestion": "<语速改进建议>"
    },
    "pause_analysis": {
        "proper_pauses": <在标点/语义边界处正确停顿的次数>,
        "improper_pauses": [
            {
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "context": "<该停顿所在的句子>",
                "issue": "<问题描述>"
            }
        ],
        "missed_pauses": [
            {
                "position": "<应该停顿但没有停顿的位置>",
                "context": "<所在句子>",
                "suggestion": "<建议>"
            }
        ],
        "overall_assessment": "<整体断句停顿评价>"
    },
    "overall_assessment": "<综合评价，80-150字>",
    "practice_suggestions": [
        "<练习建议1：具体可操作>",
        "<练习建议2>",
        "<练习建议3>"
    ]
}

分析规则：
- 异常停顿：相邻词间隔>1500ms（非标点处）；流畅度评分：无中断90-100, 偶有停顿70-89, 多处中断50-69, 严重卡顿0-49
- 标准朗读语速：140-240字/分钟；过快>280, 过慢<120
- 断句以原文标点和语义边界为基准
- 逐字对比原文和ASR转写，识别漏读、误读、多读
- 发音问题基于SOE低分字词数据
- 如果某项无问题，保留字段给出正面描述
只输出纯JSON，不要添加markdown代码块标记。"""

    def _build_article_reading_user_prompt(
        self, speech_text: str, ref_text: str, scores_data: dict,
        statistics_data: dict, low_score_info: str, timestamp_info: str,
        speech_rate_info: str
    ) -> str:
        """构建文章朗读评测的用户提示词"""
        user_prompt = f"""请分析以下用户朗读文章的表现：

## 文章原文

{ref_text}

## 用户实际朗读内容（ASR识别结果）

{speech_text}

## SOE语音评测评分

- 发音准确度: {scores_data.get('pronunciation_accuracy', 0)}分
- 发音流利度: {scores_data.get('pronunciation_fluency', 0)}分
- 发音完整度: {scores_data.get('pronunciation_completion', 0)}分
- 综合建议分: {scores_data.get('suggested_score', 0)}分
{speech_rate_info}"""

        if statistics_data:
            user_prompt += f"""
## 评分统计

- 总字数: {statistics_data.get('total_words', 0)}
- 平均准确度: {statistics_data.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics_data.get('low_score_count', 0)}个
"""

        user_prompt += low_score_info
        user_prompt += timestamp_info

        user_prompt += """
严格按照系统提示的JSON格式输出。"""
        return user_prompt

    def _default_tongue_twister_result(self) -> dict:
        """绕口令评测的默认返回结构"""
        return {
            "strengths": [],
            "improvements": {
                "extra_words": {"count": 0, "words": [], "description": "无法解析AI响应"},
                "missed_words": {"count": 0, "words": [], "description": "无法解析AI响应"},
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
        """文章朗读评测的默认返回结构"""
        return {
            "strengths": [],
            "improvements": {
                "extra_words": {"count": 0, "words": [], "description": "无法解析AI响应"},
                "missed_words": {"count": 0, "words": [], "description": "无法解析AI响应"},
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
        """
        生成一分钟观点陈述评测报告（JSON格式）

        评测维度：观点明确性、结构完整度、逻辑清晰度、时间节奏、表达冗余度

        Args:
            speech_text: 语音转写文本
            speech_scores: SOE评分数据
            low_score_words: 低分字词列表
            statistics: 评测统计数据
            topic: 观点陈述的题目/话题
            speech_rate: 语速（字/分钟）
            audio_duration: 音频时长（秒）
            word_info_list: ASR词级时间戳数据
            language: 语言

        Returns:
            JSON格式的评测报告
        """
        system_prompt = self._build_opinion_statement_system_prompt(language, topic is not None)
        user_prompt = self._build_opinion_statement_user_prompt(
            speech_text, speech_scores, low_score_words, statistics,
            topic, speech_rate, audio_duration, word_info_list, language
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        # 解析 JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            return self._default_opinion_statement_result(audio_duration)

    def _build_opinion_statement_system_prompt(self, language: str, has_topic: bool) -> str:
        """构建一分钟观点陈述评测的系统提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"

        topic_field = ""
        if has_topic:
            topic_field = '"topic_relevance_score": <贴题性评分0-100>,'

        return f"""你是观点陈述评测专家，针对"一分钟观点陈述"场景进行评测。只输出纯JSON：

{{
    "viewpoint_analysis": {{
        "has_clear_viewpoint": <是否有明确观点，true/false>,
        "viewpoint_summary": "<用一句话概括核心观点，若无则写'未提出明确观点'>",
        "opening_type": "<开头类型：直接亮明观点/渐进引入/回避式开头/模糊开头>",
        "opening_quote": "<开头原文前30字>",
        "evasion_signals": ["<回避性表达，如'我觉得这个问题比较复杂'等>"],
        "score": <观点明确性评分0-100>,
        "assessment": "<观点表达评价>"
    }},
    "structure_completeness": {{
        "score": <结构完整度评分0-100>,
        "has_viewpoint": <是否有观点环节，true/false>,
        "has_reason": <是否有理由论证，true/false>,
        "has_example": <是否有举例支撑，true/false>,
        "has_summary": <是否有总结收尾，true/false>,
        "structure_pattern": "<实际结构模式描述，如'观点→理由→总结（缺少举例）'>",
        "ideal_pattern": "观点→理由→举例→总结",
        "missing_parts": ["<缺失的结构部分>"],
        "assessment": "<结构完整度评价>"
    }},
    "logic_clarity": {{
        "score": <逻辑清晰度评分0-100>,
        "logic_jumps": [
            {{
                "from_point": "<跳跃前的内容要点>",
                "to_point": "<跳跃后的内容要点>",
                "description": "<跳跃描述>"
            }}
        ],
        "contradictions": [
            {{
                "statement_a": "<矛盾表述A>",
                "statement_b": "<矛盾表述B>",
                "description": "<矛盾分析>"
            }}
        ],
        "argument_piling": {{
            "detected": <是否存在论据堆砌（只罗列不论证），true/false>,
            "description": "<堆砌情况描述>"
        }},
        "reasoning_chain": "<论证链条描述，如'观点A←因为B←例如C←所以A'>",
        "assessment": "<逻辑清晰度评价>"
    }},
    "time_rhythm": {{
        "score": <时间节奏评分0-100>,
        "total_duration_seconds": <总时长秒>,
        "duration_level": "<时间判定：严重超时/略微超时/适中/偏短/过短>",
        "first_half_rate": <前半段语速({rate_unit})>,
        "second_half_rate": <后半段语速({rate_unit})>,
        "rate_change": "<语速变化：加速/减速/平稳>",
        "panic_acceleration": <后半段是否存在慌张加速，true/false>,
        "time_allocation": {{
            "opening_seconds": <开头部分用时秒>,
            "body_seconds": <主体论述用时秒>,
            "closing_seconds": <收尾部分用时秒>,
            "assessment": "<时间分配评价>"
        }},
        "assessment": "<时间节奏评价>"
    }},
    "expression_redundancy": {{
        "score": <表达精炼度评分0-100>,
        "filler_words": [
            {{"word": "<口头禅/填充词>", "count": <出现次数>, "example_context": "<出现的上下文示例>"}}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<废话比例描述，如每分钟X次口头禅>",
        "redundant_expressions": [
            {{
                "expression": "<冗余表达原文>",
                "issue": "<问题描述，如重复啰嗦/无意义修饰/空泛套话>",
                "suggestion": "<精简建议>"
            }}
        ],
        "effective_content_ratio": "<有效内容占比估算，如80%>",
        "assessment": "<表达冗余度评价>"
    }},
    "overall_scores": {{
        "overall_score": <综合评分0-100，加权：观点明确性20%+逻辑清晰度20%+表达精炼度15%+流畅度15%+语速10%+结构完整度10%+时间节奏10%>,
        "viewpoint_score": <观点明确性评分0-100>,
        "structure_score": <结构完整度评分0-100>,
        "logic_score": <逻辑清晰度评分0-100>,
        "fluency_score": <流畅度评分0-100，基于SOE发音流利度数据>,
        "speech_rate_score": <语速评分0-100>,
        "expression_score": <表达精炼度评分0-100>,
        "time_rhythm_score": <时间节奏评分0-100>,
        {topic_field}
        "pronunciation_accuracy": <SOE发音准确度原始分>,
        "pronunciation_fluency": <SOE发音流利度原始分>,
        "pronunciation_completion": <SOE发音完整度原始分>,
        "suggested_score": <SOE综合建议分>,
        "speech_rate_value": <语速数值>,
        "speech_rate_level": "<优秀/良好/一般/较差>",
        "speech_rate_suggestion": "<语速建议>",
        "level": "<等级：优秀(85-100)/良好(70-84)/一般(55-69)/需改进(0-54)>",
        "one_sentence_comment": "<一句话点评，不超过30字>"
    }},
    "structure_visualization": {{
        "arguments": ["<论点1>", "<论点2>", "<论点3>"],
        "conclusion": "<结论要点>"
    }},
    "strengths": ["<优点1，不少于15字>", "<优点2>", "<优点3>"],
    "improvements": ["<改进建议1，具体可操作>", "<改进建议2>", "<改进建议3>"],
    "practice_tips": [
        {{
            "dimension": "<针对的维度，如观点表达/结构组织/逻辑论证/时间控制/语言精炼>",
            "tip": "<具体练习方法>"
        }}
    ]
}}

评分标准：
- 观点明确性(20%): 90-100开门见山观点鲜明, 70-89有观点但不够直接, 50-69观点模糊, 0-49无明确观点
- 逻辑清晰度(20%): 90-100论证清晰无矛盾, 70-89偶有跳跃, 50-69明显跳跃或堆砌, 0-49逻辑混乱
- 表达精炼度(15%): 90-100无口头禅有效内容>90%, 70-89偶有口头禅, 50-69较多冗余60-80%, 0-49大量废话
- 流畅度(15%): 基于SOE发音流利度评定
- 语速(10%): 中文120-180优秀, 100-120/180-200良好; 英文100-150优秀, 80-100/150-180良好
- 结构完整度(10%): 90-100四要素完整, 70-89缺一个, 50-69缺两个, 0-49无明显结构
- 时间节奏(10%): 90-100在50-65秒节奏均匀, 70-89在45-70秒, 50-69在30-45或70-80秒, 0-49<30或>80秒

SOE原始分直接填入评测返回数值。
回避式开头识别：如"这个问题比较复杂"、"要看情况"等均属回避。
常见口头禅："然后"、"就是"、"其实"、"那个"、"嗯"、"啊"、"这个"、"对吧"、"反正"、"所以说"。

只输出纯JSON，所有评分0-100分，分析要引用原文。"""

    def _build_opinion_statement_user_prompt(
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
    ) -> str:
        """构建一分钟观点陈述评测的用户提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"
        prompt = f"""请评测以下一分钟观点陈述：

## 语音转文字内容

{speech_text}

## 语音评分数据（SOE）

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分

## 时间与语速信息

- 音频时长: {audio_duration or 0:.1f} 秒
- 语速: {speech_rate or 0} {rate_unit}
"""

        if topic:
            prompt += f"""
## 陈述题目

题目：{topic}
请分析陈述内容与该题目的贴题性。
"""

        if statistics:
            prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words and len(low_score_words) > 0:
            prompt += """
## 低分字词列表

"""
            for word in low_score_words[:20]:
                prompt += f"- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分, 流利度{word.get('fluency', 0)}分\n"

        # 添加词级时间戳信息
        if word_info_list and len(word_info_list) > 0:
            prompt += "\n## 词级别时间戳信息（ASR识别）\n\n| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |\n|------|-------------|-------------|-------------|\n"
            for w in word_info_list[:150]:
                prompt += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

            # 计算前半段和后半段的分界
            if audio_duration and audio_duration > 0:
                half_time_ms = int(audio_duration * 1000 / 2)
                prompt += f"\n前半段/后半段分界时间点: {half_time_ms}ms ({audio_duration/2:.1f}秒)\n"
                prompt += "请根据时间戳数据分别计算前半段和后半段的语速，判断是否存在后半段慌张加速。\n"

        prompt += """
严格按照系统提示的JSON格式输出。"""

        return prompt

    def _default_opinion_statement_result(self, audio_duration=None) -> dict:
        """一分钟观点陈述评测的默认返回结构"""
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
                "time_allocation": {
                    "opening_seconds": 0,
                    "body_seconds": 0,
                    "closing_seconds": 0,
                    "assessment": "无法解析AI响应"
                },
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
            "structure_visualization": {
                "arguments": [],
                "conclusion": ""
            },
            "strengths": [],
            "improvements": ["无法解析AI响应，请稍后重试"],
            "practice_tips": []
        }

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
        """
        生成即兴反应评测报告（JSON格式）

        评测维度：反应速度、内容相关性、结构形成、逻辑连贯度、表达冗余度

        Args:
            speech_text: 语音转写文本
            speech_scores: SOE评分数据
            low_score_words: 低分字词列表
            statistics: 评测统计数据
            scenario: 即兴反应场景/题目
            speech_rate: 语速（字/分钟）
            audio_duration: 音频时长（秒）
            word_info_list: ASR词级时间戳数据
            language: 语言

        Returns:
            JSON格式的评测报告
        """
        system_prompt = self._build_impromptu_reaction_system_prompt(language, scenario is not None)
        user_prompt = self._build_impromptu_reaction_user_prompt(
            speech_text, speech_scores, low_score_words, statistics,
            scenario, speech_rate, audio_duration, word_info_list, language
        )

        messages = [
            {"Role": "system", "Content": system_prompt},
            {"Role": "user", "Content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]

        # 解析 JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            return self._default_impromptu_reaction_result(audio_duration)

    def _build_impromptu_reaction_system_prompt(self, language: str, has_scenario: bool) -> str:
        """构建即兴反应评测的系统提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"

        scenario_field = ""
        if has_scenario:
            scenario_field = '"scenario_relevance_score": <切题性评分0-100>,'

        return f"""你是即兴反应评测专家。根据发言转写和词级时间戳进行结构化评测，只输出纯JSON：

{{
    "reaction_speed": {{
        "first_word_time_ms": <第一个词出现的时间戳毫秒>,
        "opening_speed": "<起步判断：果断开口/犹豫拖延/大量填充词起手>",
        "panic_signals": <是否存在明显慌乱(如语速突变、结巴、大量"嗯""啊")，true/false>,
        "thinking_pauses": [
            {{
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "position_time_ms": <停顿发生的时间点毫秒>
            }}
        ],
        "assessment": "<起步反应速度与情绪表现的详细评价>"
    }},
    "structure_formation": {{
        "formed_in_15s": <是否在开场(约前15秒)内建立主线结构，true/false>,
        "structure_signal": "<结构信号词，如'我会从两个方面说'、'首先其次'等，若无则写'无明确结构信号'>",
        "structure_pattern": "<实际表现出的结构，如'总分总'、'并列式'、'无序散发'>",
        "has_opening": <是否有开头，true/false>,
        "has_body": <是否有主体论述，true/false>,
        "has_closing": <是否有结尾，true/false>,
        "assessment": "<结构形成速度和清晰度的犀利评价>"
    }},
    "content_relevance": {{
        "topic_relevance": "<切题度判定：紧扣主题/略微偏题/完全跑题>",
        "on_topic": <是否切题，true/false>,
        "topic_drift": <是否跑题，true/false>,
        "off_topic_parts": ["<跑题的部分内容>"],
        "relevance_description": "<相关性描述，分析回答是否紧扣场景>",
        "assessment": "<内容相关性评价>"
    }},
    "logic_coherence": {{
        "coherence_level": "<连贯程度：流畅连贯/基本连贯/偶有跳跃/逻辑混乱>",
        "logic_jumps": [
            {{
                "from_point": "<跳跃前的内容>",
                "to_point": "<跳跃后的内容>",
                "description": "<思维跳跃或话题中断的具体表现>"
            }}
        ],
        "transition_quality": "<过渡质量评价>",
        "assessment": "<逻辑连贯性与切题度的犀利评价>"
    }},
    "expression_redundancy": {{
        "filler_words": [
            {{"word": "<嗯/啊/然后/就是说等口头禅>", "count": <出现次数>}}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<废话比例描述>",
        "redundancy_level": "<冗余度判定：极低/正常/偏高/极高>",
        "effective_content_ratio": "<有效内容占比估算>",
        "assessment": "<表达流畅度及填充词比例的犀利评价>"
    }},
    "overall_scores": {{
        "overall_score": <综合评分0-100，加权：反应速度25%+内容相关性25%+逻辑连贯度20%+流畅度15%+表达精炼度10%+结构形成5%>,
        "reaction_speed_score": <反应速度评分0-100>,
        "content_relevance_score": <内容相关性评分0-100>,
        "logic_coherence_score": <逻辑连贯度评分0-100>,
        "fluency_score": <流畅度评分0-100，基于SOE发音流利度数据>,
        "expression_score": <表达精炼度评分0-100>,
        "structure_score": <结构形成评分0-100>,
        {scenario_field}
        "pronunciation_accuracy": <SOE发音准确度原始分>,
        "pronunciation_fluency": <SOE发音流利度原始分>,
        "pronunciation_completion": <SOE发音完整度原始分>,
        "suggested_score": <SOE综合建议分>,
        "speech_rate_value": <语速数值>,
        "speech_rate_level": "<优秀/良好/一般/较差>",
        "level": "<等级：优秀(85-100)/良好(70-84)/一般(55-69)/需改进(0-54)>",
        "one_sentence_comment": "<一句话总结，不超过30字>"
    }},
    "structure_visualization": {{
        "key_points": ["<要点1>", "<要点2>", "<要点3>"],
        "conclusion": "<结论或总结>"
    }},
    "strengths": ["<优点1，不少于15字>", "<优点2>"],
    "improvements": ["<改进建议1，具体可操作>", "<改进建议2>"],
    "next_action": "<给出唯一且最具操作性的改进建议>"
}}

评分标准：
- 反应速度(25%): 90-100果断开口(<500ms), 70-89短暂思考(500-1500ms), 50-69明显犹豫(1500-3000ms), 0-49长时间沉默(>3000ms)或慌乱
- 内容相关性(25%): 90-100紧扣场景, 70-89基本切题, 50-69部分相关有跑题, 0-49严重跑题
- 逻辑连贯度(20%): 90-100流畅无跳跃, 70-89偶有小跳跃, 50-69跳跃明显, 0-49逻辑混乱
- 流畅度(15%): 基于SOE发音流利度数据评定
- 表达精炼度(10%): 90-100无口头禅, 70-89偶有口头禅, 50-69较多冗余, 0-49大量废话
- 结构形成(5%): 90-100前15秒建立主线, 70-89结构较慢, 50-69结构模糊, 0-49全程无序

要求：客观犀利，引用原文，改进建议具体可执行，反应速度基于时间戳分析。
只输出纯JSON，所有评分0-100分。"""

    def _build_impromptu_reaction_user_prompt(
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
    ) -> str:
        """构建即兴反应评测的用户提示词"""
        rate_unit = "字/分钟" if language == "zh" else "词/分钟"
        prompt = f"""请评测以下即兴反应表现：

## 语音转文字内容

{speech_text}

## 语音评分数据（SOE）

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分

## 时间与语速信息

- 音频时长: {audio_duration or 0:.1f} 秒
- 语速: {speech_rate or 0} {rate_unit}
"""

        if scenario:
            prompt += f"""
## 即兴反应场景/题目

场景：{scenario}
请分析回答内容与该场景的相关性和切题程度。
"""

        if statistics:
            prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

        if low_score_words and len(low_score_words) > 0:
            prompt += """
## 低分字词列表

"""
            for word in low_score_words[:20]:
                prompt += f"- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分, 流利度{word.get('fluency', 0)}分\n"

        # 添加词级时间戳信息
        if word_info_list and len(word_info_list) > 0:
            prompt += "\n## 词级别时间戳信息（ASR识别）\n\n| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |\n|------|-------------|-------------|-------------|\n"
            for w in word_info_list[:150]:
                prompt += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

            if word_info_list:
                first_word_time = word_info_list[0].get('begin_time', 0)
                prompt += f"\n第一个词出现时间: {first_word_time}ms\n"
                prompt += "请根据时间戳数据分析反应速度（开口前停顿）和思考停顿位置。\n"

        prompt += """
严格按照系统提示的JSON格式输出。"""

        return prompt

    def _default_impromptu_reaction_result(self, audio_duration=None) -> dict:
        """即兴反应评测的默认返回结构"""
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
                "on_topic": False,
                "topic_drift": False,
                "off_topic_parts": [],
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
            "structure_visualization": {
                "key_points": [],
                "conclusion": ""
            },
            "strengths": [],
            "improvements": ["无法解析AI响应，请稍后重试"],
            "next_action": "无法解析AI响应"
        }


# Singleton instance
hunyuan_service = HunyuanService()

