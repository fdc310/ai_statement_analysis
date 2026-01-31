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

你必须严格按照以下JSON格式输出分析结果：

```json
{
    "core_idea": "文本的核心思想/主旨，用一两句话概括",
    "key_points": [
        {
            "title": "要点标题",
            "content": "要点详细内容",
            "importance": "高/中/低"
        }
    ],
    "logical_structure": {
        "type": "结构类型（如：总分总、递进式、并列式、对比式、因果式等）",
        "description": "对逻辑结构的简要说明",
        "outline": [
            {
                "level": 1,
                "title": "一级标题/段落主题",
                "summary": "该部分的简要概括",
                "sub_points": [
                    {
                        "level": 2,
                        "title": "二级要点",
                        "summary": "要点说明"
                    }
                ]
            }
        ]
    },
    "arguments": [
        {
            "claim": "论点/观点",
            "evidence": "支撑论据",
            "reasoning": "论证逻辑"
        }
    ],
    "conclusion": "结论或总结",
    "writing_style": "写作风格特点",
    "suggestions": [
        "改进建议1",
        "改进建议2"
    ]
}
```

分析要求：
1. 核心思想要精准概括，抓住文本的中心主旨
2. 关键要点要分点列出，标注重要程度
3. 逻辑结构要清晰展示文本的组织方式
4. 论点论据要分析清楚论证过程
5. 改进建议要具体可行

注意：
- 必须严格按照JSON格式输出
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

你必须严格按照以下JSON格式输出分析结果：

```json
{
    "title": "绕口令标题/主题",
    "difficulty": "难度等级（简单/中等/困难/专家）",
    "core_phonemes": [
        {
            "phoneme": "音素（如：b、p、m、f等）",
            "pinyin": "对应拼音",
            "ipa": "国际音标",
            "description": "发音描述",
            "articulation": {
                "manner": "发音方式（如：爆破音、摩擦音、鼻音等）",
                "place": "发音部位（如：双唇、舌尖、舌根等）",
                "voicing": "清浊（清音/浊音）"
            },
            "examples": ["包含该音素的字词示例"]
        }
    ],
    "acoustic_features": [
        {
            "feature": "声学特征名称",
            "description": "特征描述",
            "key_difference": "与易混淆音的关键差异",
            "measurement": "可量化的声学指标（如：VOT、F1/F2频率等）"
        }
    ],
    "confusion_pairs": [
        {
            "pair": ["音素1", "音素2"],
            "difference": "区分要点",
            "common_errors": "常见错误",
            "practice_tip": "练习建议"
        }
    ],
    "pronunciation_tips": [
        {
            "tip": "发音提示",
            "target_sounds": ["针对的音素"],
            "technique": "具体技巧",
            "practice_method": "练习方法"
        }
    ],
    "rhythm_pattern": {
        "beat_count": "节拍数",
        "stress_pattern": "重音模式",
        "pause_points": ["建议停顿位置"],
        "speed_suggestion": "建议语速"
    },
    "practice_sequence": [
        {
            "step": 1,
            "focus": "练习重点",
            "content": "练习内容",
            "repetitions": "建议重复次数"
        }
    ],
    "annotated_text": "带音素标注的文本（用[]标注核心音素）"
}
```

分析要求：
1. 准确识别绕口令中的核心音素和难点
2. 详细解释声学特征的关键差异
3. 找出容易混淆的音素对
4. 提供实用的发音技巧和练习方法
5. 设计合理的练习顺序

注意：
- 必须严格按照JSON格式输出
- 分析要基于语音学原理
- 建议要具体可操作"""
        else:
            system_prompt = """You are an expert in phonetics and pronunciation teaching. Your task is to analyze tongue twister pronunciation key points to help users practice pronunciation.

You must output the analysis result in the following JSON format:

```json
{
    "title": "Tongue twister title/theme",
    "difficulty": "Difficulty level (Easy/Medium/Hard/Expert)",
    "core_phonemes": [
        {
            "phoneme": "Phoneme (e.g., /p/, /b/, /θ/, /ð/)",
            "ipa": "IPA symbol",
            "description": "Pronunciation description",
            "articulation": {
                "manner": "Manner of articulation (e.g., plosive, fricative, nasal)",
                "place": "Place of articulation (e.g., bilabial, alveolar, velar)",
                "voicing": "Voiced/Voiceless"
            },
            "examples": ["Example words containing this phoneme"]
        }
    ],
    "acoustic_features": [
        {
            "feature": "Acoustic feature name",
            "description": "Feature description",
            "key_difference": "Key difference from similar sounds",
            "measurement": "Measurable acoustic indicators (e.g., VOT, F1/F2 frequency)"
        }
    ],
    "confusion_pairs": [
        {
            "pair": ["phoneme1", "phoneme2"],
            "difference": "Key distinction",
            "common_errors": "Common mistakes",
            "practice_tip": "Practice suggestion"
        }
    ],
    "pronunciation_tips": [
        {
            "tip": "Pronunciation tip",
            "target_sounds": ["Target phonemes"],
            "technique": "Specific technique",
            "practice_method": "Practice method"
        }
    ],
    "rhythm_pattern": {
        "beat_count": "Number of beats",
        "stress_pattern": "Stress pattern",
        "pause_points": ["Suggested pause positions"],
        "speed_suggestion": "Suggested speed"
    },
    "practice_sequence": [
        {
            "step": 1,
            "focus": "Practice focus",
            "content": "Practice content",
            "repetitions": "Suggested repetitions"
        }
    ],
    "annotated_text": "Text with phoneme annotations (mark core phonemes with [])"
}
```

Requirements:
1. Accurately identify core phonemes and difficulty points
2. Explain acoustic feature differences in detail
3. Identify easily confused phoneme pairs
4. Provide practical pronunciation tips and practice methods
5. Design a reasonable practice sequence"""

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


# Singleton instance
hunyuan_service = HunyuanService()
