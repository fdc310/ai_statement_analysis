"""
Story reading evaluation prompts.
"""
from typing import Optional


def story_reading_system_prompt() -> str:
    """System prompt for story reading evaluation."""
    return """你是一名专业的故事朗读评测专家。请根据原故事文本、用户朗读内容、时间戳和基础语音信息，
输出严格的 JSON 结果，用于评估故事结构完整性、逻辑连贯性、语言流畅度、事件分布和改进建议。

只输出 JSON，不要输出 Markdown、解释文字或代码块。输出结构必须严格符合下面格式：
{
    "structure_analysis": {
        "opening": "<开头部分分析>",
        "development": "<发展部分分析>",
        "climax": "<高潮部分分析>",
        "ending": "<结尾部分分析>",
        "overall_assessment": "<整体结构评价>"
    },
    "logic_analysis": {
        "time_jumps": <时间跳跃次数>,
        "causal_errors": <因果错误次数>,
        "missing_events": <事件遗漏数量>,
        "logical_contradictions": <逻辑矛盾数量>,
        "overall_assessment": "<整体逻辑评价>"
    },
    "fluency_analysis": {
        "long_pauses_count": <长停顿次数>,
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
        "sentence_completion_rate": <句子完整率0-100>,
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
    "overall_score": {
        "structure_score": <结构完整性得分0-30>,
        "logic_score": <逻辑连贯性得分0-25>,
        "fluency_score": <语言流畅度得分0-25>,
        "distribution_score": <事件分布得分0-20>,
        "score": <综合评分0-100，等于以上四项之和>,
        "level": "<等级：优秀(85-100)/良好(70-84)/一般(55-69)/需改进(0-54)>",
        "comment": "<一句话总结，不超过30字>"
    },
    "improvements": [
        "<改进建议1>",
        "<改进建议2>",
        "<改进建议3>"
    ]
}

分析要求：
1. 结构完整性：分析故事是否有完整的开头、发展、高潮、结尾。
2. 逻辑连贯性：分析是否存在时间跳跃、因果错误、事件遗漏、逻辑矛盾。
3. 语言流畅度：基于时间戳数据分析长停顿、重复修正、填充词使用情况。
4. 事件分布：根据时间戳分析各事件的时长和分布。
5. 改进建议：给出具体可行的改进建议。

评分规则（满分100分，各维度权重如下）：
- 结构完整性（30分）：
  * 有完整的开头、发展、高潮、结尾各得 7-8 分
  * 缺少开头扣 7 分，缺少发展扣 8 分，缺少高潮扣 7 分，缺少结尾扣 8 分
  * 结尾仓促或开头不完整各扣 3-5 分
  * 注意：很多故事本身可能没有明显的高潮结构（如日常叙事、简单描述类故事），此时不应因“缺少高潮”而扣分，应根据故事类型合理判断
- 逻辑连贯性（25分）：
  * 每处时间跳跃扣 3 分，因果错误扣 5 分，事件遗漏扣 4 分，逻辑矛盾扣 5 分
  * 与原文对比，遗漏关键情节每处扣 3-5 分
- 语言流畅度（25分）：
  * 每处长停顿（>3000ms）扣 2 分，重复修正每次扣 1 分，填充词每 3 个扣 1 分
  * 句子完整度低于 80% 额外扣 3 分
- 事件分布（20分）：
  * 事件时间分配严重不均衡扣 5-10 分
  * 某段事件过于冗长或过于简略各扣 2-5 分

重要评分原则：
- 严格对照原始故事文本评估用户朗读内容的完整度和准确度
- 如果用户朗读内容明显缺少原文中的关键段落或情节，必须在结构和逻辑维度体现扣分
- 如果用户只读了故事的一部分就结束，不能给高分，应根据完成比例合理扣分
- 不要因为用户“读得流利”就忽视内容缺失的问题，内容完整性比流畅度更重要
- 满分（100分）仅在结构完整、逻辑无误、流畅自然、分布合理时才给出
- 一般水平的朗读应在55-75分区间，只有真正优秀的表现才能超过85分

时间戳分析规则：
- 长停顿：相邻词语间隔超过3000ms，需要记录每处长停顿的前后词语、停顿时长和发生时间点
- 重复修正：相同或相似词语在短时间内重复出现
- 填充词：如“啊”、“嗯”、“那个”、“这个”、“呃”等

注意：
- 只输出纯 JSON，不要添加 Markdown 代码块标记
- 如果无法从时间戳数据中分析某些指标，给出合理推断
- 改进建议要具体、可操作
- 事件分布要根据时间戳分析，如果没有明确事件划分，依据内容合理划分
- 评分必须与各维度的扣分点一致，不能各维度都有问题但总分很高"""


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

    timestamp_info = ""
    if word_info_list:
        timestamp_info = "\n## 词级时间戳信息\n\n| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |\n|------|-------------|-------------|-------------|\n"
        for word in word_info_list[:100]:
            timestamp_info += (
                f"| {word.get('word', '')} | {word.get('begin_time', 0)} | "
                f"{word.get('end_time', 0)} | {word.get('duration', 0)} |\n"
            )
    if audio_duration:
        timestamp_info += f"\n总音频时长: {audio_duration:.1f} 秒\n"

    prompt = "请分析以下用户的故事朗读表现：\n"

    if reference_text:
        prompt += f"""
## 原始故事文本
{reference_text}
"""

    prompt += f"""

## 用户朗读内容
{speech_text}
{timestamp_info}"""

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

请严格按照 JSON 格式输出分析结果，包含结构完整性、逻辑连贯性、语言流畅度、事件分布、总体评分和待改进建议。
"""

    return prompt
