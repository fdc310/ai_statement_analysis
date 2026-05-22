"""
Story reading evaluation prompts.
Extracted from HunyuanService for modular maintenance.
"""
from typing import Optional


def story_reading_system_prompt() -> str:
    """System prompt for story reading evaluation."""
    return """你是一个专业的朗读评测专家。你的任务是评测用户的故事朗读，分析其结构完整性、逻辑连贯性、语言流畅度和事件分布。

你必须严格按照以下JSON格式输出评测结果，不要添加任何额外内容，只输出JSON：

{
    "structure_completeness": {
        "score": <结构完整性评分0-30>,
        "has_beginning": <是否有开头 true/false>,
        "has_development": <是否有发展 true/false>,
        "has_climax": <是否有高潮 true/false>,
        "has_ending": <是否有结尾 true/false>,
        "analysis": "<结构完整性分析>"
    },
    "logic_coherence": {
        "score": <逻辑连贯性评分0-25>,
        "causality": "<因果关系分析>",
        "character_consistency": "<角色一致性分析>",
        "plot_logic": "<情节逻辑分析>",
        "analysis": "<逻辑连贯性分析>"
    },
    "language_fluency": {
        "score": <语言流畅度评分0-25>,
        "pronunciation": "<发音分析>",
        "intonation": "<语调分析>",
        "rhythm": "<节奏分析>",
        "expression": "<表现力分析>",
        "analysis": "<语言流畅度分析>"
    },
    "event_distribution": {
        "score": <事件分布评分0-20>,
        "event_count": <事件数量>,
        "coverage": "<事件覆盖度分析>",
        "balance": "<事件分布均衡度分析>",
        "analysis": "<事件分布分析>"
    },
    "overall_score": <总分0-100>,
    "strengths": ["<优点1>", "<优点2>"],
    "improvements": ["<改进1>", "<改进2>"],
    "suggestions": ["<具体建议1>", "<具体建议2>"]
}

评分标准：
1. 结构完整性(30分)：开头、发展、高潮、结尾四要素齐全得满分
2. 逻辑连贯性(25分)：因果关系清晰、角色一致、情节合理得高分
3. 语言流畅度(25分)：发音准确、语调自然、节奏恰当、表现力强得高分
4. 事件分布(20分)：事件数量适中、覆盖全面、分布均衡得高分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分（结构完整性满分30，其他满分25/25/20）
- 总分 = 各维度分数之和（满分100）"""


def story_reading_user_prompt(
    speech_text: str,
    reference_text: Optional[str] = None,
    speech_scores: dict = None,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh"
) -> str:
    """User prompt for story reading evaluation."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"
    prompt = f"""请评测以下故事朗读：

## 朗读内容

{speech_text}
"""

    if reference_text:
        prompt += f"""
## 原文参考

{reference_text}
"""

    if speech_scores:
        prompt += f"""
## 语音评分数据

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分
"""

    prompt += f"""
## 语速信息

- 语速: {speech_rate or 0} {rate_unit}
- 音频时长: {audio_duration or 0:.1f} 秒
"""

    if statistics:
        prompt += f"""
## 评分统计

- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

    if word_info_list and len(word_info_list) > 0:
        prompt += """
## 字词时间戳

| 字词 | 开始时间(ms) | 结束时间(ms) | 时长(ms) |
|------|-------------|-------------|----------|
"""
        for w in word_info_list[:50]:
            prompt += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

    if low_score_words and len(low_score_words) > 0:
        prompt += """
## 低分字词

"""
        for word in low_score_words[:10]:
            prompt += f"- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分\n"

    prompt += """
请严格按照JSON格式输出评测结果。特别注意分析：
1. 故事结构完整性（开头、发展、高潮、结尾）
2. 逻辑连贯性（因果关系、角色一致性）
3. 语言流畅度（发音、语调、节奏、表现力）
4. 事件分布（事件数量、覆盖度、均衡度）"""

    return prompt
