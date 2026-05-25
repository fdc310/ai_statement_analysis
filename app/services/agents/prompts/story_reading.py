"""
Story reading evaluation prompts.
"""
from typing import Optional


def story_reading_system_prompt() -> str:
    """System prompt for story reading evaluation."""
    return """你是一名专业的故事朗读评测专家。请根据原故事文本、用户朗读内容、时间戳和基础语音信息，
输出严格的 JSON 结果，用于评估故事结构还原、逻辑连贯、流畅度、事件分布和改进建议。

只输出 JSON，不要输出 Markdown、解释文字或代码块。输出结构必须严格符合下面格式：
{
    "structure_analysis": {
        "opening": "<开头部分是否完整、是否自然进入故事>",
        "development": "<发展部分是否连贯、关键过程是否覆盖>",
        "climax": "<高潮部分是否出现、是否清晰>",
        "ending": "<结尾部分是否完整、是否收束>",
        "overall_assessment": "<对故事结构完整性的总结>"
    },
    "logic_analysis": {
        "time_jumps": <明显时间跳跃次数>,
        "causal_errors": <因果关系错误次数>,
        "missing_events": <关键事件遗漏数量>,
        "logical_contradictions": <逻辑矛盾数量>,
        "overall_assessment": "<对逻辑连贯性的总结>"
    },
    "fluency_analysis": {
        "long_pauses_count": <长停顿次数>,
        "long_pauses": [
            {
                "before_word": "<停顿前词语>",
                "after_word": "<停顿后词语>",
                "pause_duration_ms": <停顿毫秒数>,
                "impact": "<该停顿对表达的影响>"
            }
        ],
        "repetition_count": <重复或自我修正次数>,
        "filler_words_count": <口头填充词次数>,
        "sentence_completion_rate": <句子完整率0-100>,
        "overall_assessment": "<对流畅度和表达自然度的总结>"
    },
    "event_distribution": {
        "events": [
            {
                "name": "<事件名称>",
                "start_time_ms": <开始时间>,
                "end_time_ms": <结束时间>,
                "duration_seconds": <持续秒数>,
                "assessment": "<该事件篇幅是否合适>"
            }
        ],
        "transition_time": "<事件切换整体评价>",
        "overall_assessment": "<对事件分布是否均衡的总结>"
    },
    "improvements": [
        "<具体可执行的改进建议1>",
        "<具体可执行的改进建议2>"
    ],
    "overall_score": {
        "score": <0-100整数>,
        "level": "<优秀/良好/一般/需改进>",
        "comment": "<综合结论>"
    }
}

评估要求：
1. 结构分析关注开头、发展、高潮、结尾四部分是否完整。
2. 逻辑分析关注时间顺序、因果关系、关键事件遗漏和前后矛盾。
3. 流畅度分析结合时间戳识别长停顿、重复、自我修正、填充词和句子完整度。
4. 事件分布要结合讲述时长判断某个事件是否过长、过短或缺失。
5. 改进建议必须具体，不要写空泛鼓励。
6. 综合评分必须和前面分析保持一致，不能前面问题很多却给很高分。"""


def story_reading_user_prompt(
    speech_text: str,
    reference_text: Optional[str] = None,
    speech_scores: Optional[dict] = None,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh",
) -> str:
    """User prompt for story reading evaluation."""
    del language  # Story reading currently uses a fixed Chinese evaluation contract.

    speech_scores = speech_scores or {}
    statistics = statistics or {}

    prompt = f"""请分析以下用户的故事朗读表现。

## 用户朗读内容
{speech_text}
"""

    if reference_text:
        prompt += f"""

## 原故事文本
{reference_text}
"""

    if speech_scores:
        prompt += f"""

## 基础语音评分
- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}
- 综合建议分: {speech_scores.get('suggested_score', 0)}
"""

    if speech_rate is not None or audio_duration is not None:
        prompt += f"""

## 语速信息
- 语速: {speech_rate or 0} 字/分钟
- 音频时长: {audio_duration or 0:.1f} 秒
"""

    if statistics:
        prompt += f"""

## 统计信息
- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}
- 低分字数: {statistics.get('low_score_count', 0)}
"""

    if word_info_list:
        prompt += """

## 词级时间戳
| 词语 | 开始(ms) | 结束(ms) | 时长(ms) |
|------|----------|----------|----------|
"""
        for word in word_info_list[:100]:
            prompt += (
                f"| {word.get('word', '')} | {word.get('begin_time', 0)} | "
                f"{word.get('end_time', 0)} | {word.get('duration', 0)} |\n"
            )

    if low_score_words:
        prompt += """

## 低分词语
"""
        for word in low_score_words[:20]:
            prompt += (
                f"- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}，"
                f"流利度{word.get('fluency', 0)}\n"
            )

    prompt += """

请严格按照指定 JSON 结构输出结果，重点分析：
1. 故事结构是否完整，是否包含开头、发展、高潮、结尾。
2. 讲述是否存在时间跳跃、因果错误、事件遗漏或逻辑矛盾。
3. 朗读是否流畅，是否有长停顿、重复、自我修正、填充词。
4. 事件分布是否均衡，是否有某段讲太长、太短或缺失。
5. 给出清晰、具体、可执行的改进建议。
"""

    return prompt
