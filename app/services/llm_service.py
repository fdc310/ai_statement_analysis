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

    @property
    def model_name(self) -> str:
        """Get the configured provider model name when available."""
        return getattr(self._provider, "_model", "unknown")

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
    ) -> str:
        """Generate speech evaluation report in Markdown format."""
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
        system_prompt = tongue_twister_system_prompt()
        user_prompt = tongue_twister_user_prompt(
            speech_text=text,
            language="zh",
        )

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
        system_prompt = story_reading_system_prompt()
        user_prompt = story_reading_user_prompt(
            speech_text=speech_text,
            reference_text=story_text,
            word_info_list=word_info_list,
            audio_duration=audio_duration,
            language="zh",
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
            system_prompt = article_reading_system_prompt()
            user_prompt = article_reading_user_prompt(
                speech_text=speech_text,
                reference_text=tongue_twister_text,
                speech_scores=scores_data,
                word_info_list=word_info_list,
                low_score_words=low_score_words,
                statistics=statistics_data,
                audio_duration=audio_duration,
                language="zh",
            )
        else:
            system_prompt = tongue_twister_system_prompt()
            user_prompt = tongue_twister_user_prompt(
                speech_text=speech_text,
                reference_text=tongue_twister_text,
                speech_scores=scores_data,
                word_info_list=word_info_list,
                low_score_words=low_score_words,
                language="zh",
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        content = result["content"]
        report_data = extract_json(content)
        return report_data or {"raw_report": content}
