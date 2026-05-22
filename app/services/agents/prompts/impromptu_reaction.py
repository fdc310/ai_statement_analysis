"""
Impromptu reaction evaluation prompts.
Extracted from HunyuanService for modular maintenance.
"""
from typing import Optional


def impromptu_reaction_system_prompt(language: str = "zh", has_scenario: bool = False) -> str:
    """System prompt for impromptu reaction evaluation."""
    return """你是一个专业的即兴反应评测专家。你的任务是评测用户的即兴口语反应，分析其反应速度、内容相关性、逻辑连贯度、流畅度和结构形成。

你必须严格按照以下JSON格式输出评测结果，不要添加任何额外内容，只输出JSON：

{
    "reaction_speed": {
        "score": <反应速度评分0-100>,
        "first_word_time": <首词时间毫秒>,
        "thinking_pauses": <思考停顿次数>,
        "panic_signals": ["<恐慌信号1>"],
        "analysis": "<反应速度分析>"
    },
    "content_relevance": {
        "score": <内容相关性评分0-100>,
        "is_effective_response": <是否有效回应 true/false>,
        "is_mere_repetition": <是否只是重复问题 true/false>,
        "topic_coverage": "<内容覆盖度分析>",
        "analysis": "<内容相关性分析>"
    },
    "logic_coherence": {
        "score": <逻辑连贯度评分0-100>,
        "logic_jumps": ["<逻辑跳跃点1>"],
        "transition_quality": "<过渡质量分析>",
        "analysis": "<逻辑连贯度分析>"
    },
    "fluency": {
        "score": <流畅度评分0-100>,
        "soe_fluency": <SOE流畅度数据>,
        "hesitation_count": <犹豫次数>,
        "analysis": "<流畅度分析>"
    },
    "expression_conciseness": {
        "score": <表达精炼度评分0-100>,
        "filler_words": ["<口头禅1>"],
        "redundant_expressions": ["<冗余表达1>"],
        "analysis": "<表达精炼度分析>"
    },
    "structure_formation": {
        "score": <结构形成评分0-100>,
        "has_clear_structure": <是否形成清晰结构 true/false>,
        "structure_type": "<结构类型>",
        "first_15s_structure": <前15秒是否形成结构 true/false>,
        "analysis": "<结构形成分析>"
    },
    "overall_score": <综合评分0-100，加权计算：反应速度25%+内容相关性25%+逻辑连贯度20%+流畅度15%+表达精炼度10%+结构形成5%>,
    "strengths": ["<优点1>", "<优点2>"],
    "improvements": ["<改进1>", "<改进2>"],
    "suggestions": ["<具体建议1>", "<具体建议2>"]
}

评分标准：
1. 反应速度(25%)：首词时间<1秒优秀(90-100)，1-2秒良好(70-89)，2-3秒一般(50-69)，>3秒较差(0-49)
2. 内容相关性(25%)：有效回应得高分，仅重复问题最多20分
3. 逻辑连贯度(20%)：论证有条理、过渡自然得高分
4. 流畅度(15%)：基于SOE流畅度数据评分
5. 表达精炼度(10%)：无冗余表达得高分
6. 结构形成(5%)：前15秒内形成清晰结构得满分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- 有效回应：针对问题给出实质性回答，而非简单重复或回避
- 恐慌信号：语速突然加快、大量重复、无意义填充词"""


def impromptu_reaction_user_prompt(
    speech_text: str,
    speech_scores: dict,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    scenario: Optional[str] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh"
) -> str:
    """User prompt for impromptu reaction evaluation."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"
    prompt = f"""请评测以下即兴反应：

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

    if scenario:
        prompt += f"""
## 场景/问题

{scenario}
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
1. 首词时间（从问题结束到第一个字的时间）
2. 是否有效回应（而非简单重复问题）
3. 逻辑跳跃和过渡质量
4. 前15秒是否形成清晰结构
5. 恐慌信号（语速突然加快、大量重复等）"""

    return prompt
