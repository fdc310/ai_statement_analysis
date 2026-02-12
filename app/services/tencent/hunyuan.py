"""
Tencent Cloud Hunyuan LLM service with async support.
"""
import json
from typing import Optional, AsyncGenerator

from tencentcloud.hunyuan.v20230901 import hunyuan_client_async, models

from app.core.config import settings
from app.services.tencent.base import TencentCloudClient


class HunyuanService(TencentCloudClient):
    """Tencent Cloud Hunyuan LLM service with async support."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        super().__init__(secret_id, secret_key, "hunyuan.tencentcloudapi.com")
        self.model = model or settings.hunyuan_model

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
        stream: bool = False
    ) -> dict:
        """Generate chat completion (async)."""
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
                response = await client.ChatCompletions(req)
                return await self._handle_stream_response(response)
            else:
                response = await client.ChatCompletions(req)
                result = json.loads(response.to_json_string())
                return self._parse_chat_result(result)

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming (async generator)."""
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
            response = await client.ChatCompletions(req)
            async for event in response:
                data = json.loads(event["data"])
                if "Choices" in data and len(data["Choices"]) > 0:
                    delta = data["Choices"][0].get("Delta", {})
                    content = delta.get("Content", "")
                    if content:
                        yield content

    async def _handle_stream_response(self, response) -> dict:
        """Handle streaming response and collect full content."""
        content_parts = []
        async for event in response:
            data = json.loads(event["data"])
            if "Choices" in data and len(data["Choices"]) > 0:
                delta = data["Choices"][0].get("Delta", {})
                content = delta.get("Content", "")
                if content:
                    content_parts.append(content)

        return {
            "content": "".join(content_parts),
            "usage": {},
            "raw_response": None
        }

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
        return """你是一个专业的语音演讲评测专家。你的任务是根据用户提供的语音转文字内容和语音评分数据，生成一份详细的演讲评测报告。

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
请严格按照系统提示中指定的Markdown格式生成评测报告。在改进意见中，请特别关注发音待改进的字词。"""

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
        base_prompt = """你是一个专业的语音演讲评测专家。你的任务是根据用户提供的语音转文字内容和语音评分数据，生成一份详细的演讲评测报告。

你必须严格按照以下Markdown格式输出评测报告，不要添加任何额外的格式或内容：

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

        if has_topic:
            base_prompt += """

贴题性评分标准：
- 内容与主题高度相关，论点紧扣主题：90-100分
- 内容基本围绕主题，偶有偏离：70-89分
- 内容部分相关，有明显跑题：50-69分
- 内容与主题关联度低：0-49分"""

        base_prompt += """

注意：
- 必须严格按照上述Markdown格式输出，结论是结构可视化的子标题(###)
- 评分要客观公正，有理有据
- 改进意见要具体、可操作
- 结合语音评分数据（发音准确度、流利度等）进行综合评价"""

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
请严格按照系统提示中指定的Markdown格式生成评测报告。"""

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
        return f"""你是一个专业的语音演讲评测专家。你的任务是生成一份简洁的评测报告。

你必须严格按照以下JSON格式输出，不要添加任何额外内容，只输出JSON：

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
- 中文：120-180优秀(90-100分)，100-120或180-200良好(70-89分)，80-100或200-220一般(50-69分)，其他较差(0-49分)
- 英文：100-150优秀(90-100分)，80-100或150-180良好(70-89分)，60-80或180-200一般(50-69分)，其他较差(0-49分)

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- weak_paragraphs只包含有低分字词的段落，最多3个
- 如果没有低分字词，weak_paragraphs为空数组"""

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

        return f"""你是一个专业的语音演讲评测专家。你的任务是生成一份完整的评测报告。

你必须严格按照以下JSON格式输出，不要添加任何额外内容，只输出JSON：

{{
    "logic_completeness": {{
        "overall_score": <综合评分0-100>,
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
- 中文：120-180优秀(90-100分)，100-120或180-200良好(70-89分)，80-100或200-220一般(50-69分)，其他较差(0-49分)
- 英文：100-150优秀(90-100分)，80-100或150-180良好(70-89分)，60-80或180-200一般(50-69分)，其他较差(0-49分)

评分规则：
1. 逻辑完整性评分：综合评分是各项的加权平均
2. 语速评价：根据语速数值评分，分析语速对表达效果的影响
3. 内容角度：分析内容的贴题性、深度和覆盖面
4. 逻辑与结构：分析演讲的组织结构、连贯性和论证逻辑
5. 表达与用词：分析用词水平、表达风格和亮点
6. 优点：要详细描述演讲的亮点，每条优点不少于20字，要具体说明体现在哪里

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- weak_paragraphs只包含有低分字词的段落
- 论点和结论要从演讲内容中提取
- 各维度的分析要具体、有针对性
- strengths优点部分要写得详细具体，每条不少于15字，突出演讲者的闪光点"""

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
请根据以上信息生成评测报告，包含：
1. 逻辑完整性评分（综合评分、逻辑性、流畅度、语速）
2. 结构可视化（论点、结论）
3. 语速评价（评分、分析、建议）
4. 内容角度（贴题性、深度、覆盖面）
5. 逻辑与结构（组织结构、连贯性、论证逻辑）
6. 表达与用词（用词水平、表达风格、亮点）
7. 优点
8. 改进意见（如果涉及发音问题，请列出准确度低于60分的低分字词作为具体示例）
9. 低分段落分析

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
        system_prompt = """你是一个专业的文本分析专家。你的任务是分析用户提供的文本，提取其核心思想和逻辑结构。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

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

分析要求：
1. 核心思想要精准概括，抓住文本的中心主旨
2. 关键要点要分点列出，标注重要程度
3. 逻辑结构要清晰展示文本的组织方式
4. 论点论据要分析清楚论证过程
5. 改进建议要具体可行

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 如果某些部分在文本中不明显，可以标注为null或空数组
- 分析要客观中立，基于文本内容"""

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
            system_prompt = """你是一个专业的语音学和发音教学专家。你的任务是分析绕口令的发音要点，帮助用户更好地练习发音。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

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

分析要求：
1. 准确识别绕口令中的核心音素和难点
2. 详细解释声学特征的关键差异
3. 找出容易混淆的音素对
4. 提供实用的发音技巧和练习方法
5. 设计合理的练习顺序

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要基于语音学原理
- 建议要具体可操作"""
        else:
            system_prompt = """You are an expert in phonetics and pronunciation teaching. Your task is to analyze tongue twister pronunciation key points to help users practice pronunciation.

You must output the analysis result in the following JSON format, do not add any extra content, only output JSON:

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

Requirements:
1. Accurately identify core phonemes and difficulty points
2. Explain acoustic feature differences in detail
3. Identify easily confused phoneme pairs
4. Provide practical pronunciation tips and practice methods
5. Design a reasonable practice sequence

Note:
- Only output pure JSON, do not add markdown code block markers
- Analysis should be based on phonetic principles"""

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
        """
        Analyze sentence for reading interpretation.

        Args:
            text: Sentence to analyze
            custom_prompt: Optional custom analysis requirements

        Returns:
            Analysis result in JSON format
        """
        system_prompt = """你是一个专业的语文朗读指导专家。你的任务是分析给定的句子，提供朗读指导建议。

你必须严格按照以下JSON格式输出，不要添加任何额外内容，只输出JSON：

{{
    "center_content": "<句子的中心内容/主旨，用简短语言概括>",
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

分析要求：
1. 中心内容：准确把握句子的核心含义和主旨
2. 朗读重点：找出需要重读、强调的关键词或短语，最多3-5个
3. 注意事项：指出朗读时的语气、停顿、语速、情感等要点，最多3-5个
4. 每个要点要简练，每个不超过20字
5. 注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要基于句子的语言特征"""

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

        system_prompt = """你是一个专业的故事阅读评测专家。你的任务是分析用户的故事阅读表现，评估其结构完整性、逻辑连贯性、语言流畅度和事件分布情况。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

{
    "structure_analysis": {
        "opening": "<开头情况：有/无，简短描述>",
        "development": "<发展情况：描述事件发展过程>",
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

分析要求：
1. 结构完整性：分析故事是否有完整的开头、发展、高潮、结尾
2. 逻辑连贯性：分析是否存在时间跳跃、因果错误、事件遗漏、逻辑矛盾
3. 语言流畅度：基于时间戳数据分析长停顿、重复修正、填充词使用情况
4. 事件分布：根据时间戳分析各事件的时长和分布
5. 待改进：给出具体可行的改进建议

时间戳分析规则：
- 长停顿：相邻词语间隔超过3000ms（3秒），需要记录每处长停顿的前后词语、停顿时长和发生时间点
- 重复修正：相同或相似词语在短时间内重复出现
- 填充词：如"啊"、"呃"、"那个"、"这个"、"嗯"等

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 如果无法从时间戳数据中分析某些指标，给出合理推断
- 改进建议要具体、可操作
- 事件分布要根据时间戳分析，如果没有明确事件划分，根据内容合理划分"""

        user_prompt = f"""请分析以下用户的故事阅读表现：

## 原始故事文本

{story_text}

## 用户阅读内容

{speech_text}
{timestamp_info}

请严格按照JSON格式输出分析结果，包含结构完整性、逻辑连贯性、语言流畅度、事件分布和待改进建议。"""

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
        language: str = "zh"
    ) -> dict:
        """
        Analyze tongue twister reading performance.

        Args:
            speech_text: Transcribed speech text from ASR
            tongue_twister_text: Original tongue twister text
            word_info_list: Word-level timestamp data from ASR
            low_score_words: Low score words from SOE evaluation
            scores_data: SOE pronunciation scores
            statistics_data: SOE evaluation statistics
            audio_duration: Audio duration in seconds
            language: Language code

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

        system_prompt = """你是一个专业的绕口令语音评测专家。你的任务是分析用户朗读绕口令的语音表现，通过对比原始绕口令文本和实际朗读内容，评估优势和待改进之处。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

{
    "strengths": [
        "<优势1：具体描述用户在发音、节奏、流畅度等方面的亮点，每条不少于15字>",
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
        "rhythm_assessment": "<节奏评价：绕口令的节奏感是否把握得当>",
        "speed_assessment": "<语速评价：是否适合该绕口令的难度>"
    },
    "overall_assessment": "<综合评价，50-100字，概括整体朗读表现>",
    "practice_suggestions": [
        "<练习建议1：具体可操作的改进方法>",
        "<练习建议2>",
        "<练习建议3>"
    ]
}

分析规则：
1. 多读(extra_words)判断：将实际朗读文本与绕口令原文逐字对比，找出朗读中有但原文中没有的字词
2. 漏读(missed_words)判断：找出原文中有但朗读中缺少的字词
3. 发音问题(pronunciation_issues)：基于SOE评测的低分字词(accuracy<90分)，分析具体的发音问题
4. 流畅度分析：基于词级时间戳，分析停顿（相邻词间隔>2000ms为长停顿）、节奏和语速
5. 优势：要从完成度、发音准确性、流畅度、节奏感等多角度寻找亮点
6. 练习建议：要针对具体问题给出可操作的练习方法

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 多读和漏读的判断要精确，逐字对比
- 发音问题要结合SOE低分数据，给出具体的声母/韵母/声调分析
- 如果某项没有问题，保留字段但给出正面描述（如extra_words的count为0）
- 绕口令的停顿标准比普通阅读更严格，使用2000ms作为长停顿阈值"""

        user_prompt = f"""请分析以下用户朗读绕口令的表现：

## 绕口令原文

{tongue_twister_text}

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
请严格按照JSON格式输出分析结果。重点分析：
1. 逐字对比原文和朗读内容，找出多读和漏读
2. 结合SOE低分字词数据分析具体发音问题
3. 基于时间戳分析流畅度和节奏
4. 给出具体的优势和改进建议"""

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


# Singleton instance
hunyuan_service = HunyuanService()
