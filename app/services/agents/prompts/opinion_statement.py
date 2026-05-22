"""
Opinion statement evaluation prompts.
Extracted from HunyuanService for modular maintenance.
"""
from typing import Optional


def opinion_statement_system_prompt(language: str = "zh", has_topic: bool = False) -> str:
    """System prompt for one-minute opinion statement evaluation."""
    return """你是一个专业的观点表达评测专家。你的任务是评测用户的一分钟观点陈述，分析其观点明确性、逻辑清晰度、表达精炼度、结构完整度和时间节奏。

你必须严格按照以下JSON格式输出评测结果，不要添加任何额外内容，只输出JSON：

{
    "viewpoint_clarity": {
        "score": <观点明确性评分0-100>,
        "opening_type": "<开门见山/铺垫引入/回避式开头>",
        "viewpoint": "<提取的核心观点>",
        "analysis": "<观点明确性分析>"
    },
    "logic_clarity": {
        "score": "<逻辑清晰度评分0-100>",
        "logic_flow": "<论证逻辑流程分析>",
        "logic_jumps": ["<逻辑跳跃点1>", "<逻辑跳跃点2>"],
        "contradictions": ["<矛盾点1>"],
        "analysis": "<逻辑清晰度分析>"
    },
    "expression_conciseness": {
        "score": <表达精炼度评分0-100>,
        "filler_words": ["<口头禅1>", "<口头禅2>"],
        "redundant_expressions": ["<冗余表达1>", "<冗余表达2>"],
        "word_count": <总字数>,
        "effective_word_count": <有效字数>,
        "analysis": "<表达精炼度分析>"
    },
    "structure_completeness": {
        "score": <结构完整度评分0-100>,
        "has_viewpoint": <是否有明确观点 true/false>,
        "has_reason": <是否有理由支撑 true/false>,
        "has_example": <是否有举例说明 true/false>,
        "has_summary": <是否有总结 true/false>,
        "structure": "<观点→理由→例子→总结>",
        "analysis": "<结构完整度分析>"
    },
    "time_rhythm": {
        "score": <时间节奏评分0-100>,
        "total_duration": <总时长秒>,
        "first_half_rate": <前半段语速字/分钟>,
        "second_half_rate": <后半段语速字/分钟>,
        "panic_acceleration": <是否有恐慌加速 true/false>,
        "analysis": "<时间节奏分析>"
    },
    "overall_score": <综合评分0-100，加权计算：观点明确性20%+逻辑清晰度20%+表达精炼度15%+流畅度15%+语速10%+结构完整度10%+时间节奏10%>,
    "strengths": ["<优点1>", "<优点2>"],
    "improvements": ["<改进1>", "<改进2>"],
    "suggestions": ["<具体建议1>", "<具体建议2>"]
}

评分标准：
1. 观点明确性(20%)：开门见山提出观点得高分，铺垫引入扣分，回避式开头（如"这个话题很有意思"）最多20分
2. 逻辑清晰度(20%)：论证有条理、因果关系清晰得高分，逻辑跳跃、矛盾扣分
3. 表达精炼度(15%)：无冗余表达得高分，口头禅多、重复表达扣分
4. 流畅度(15%)：基于SOE流畅度数据评分
5. 语速(10%)：120-180字/分钟优秀，其他范围按标准评分
6. 结构完整度(10%)：观点→理由→例子→总结四要素齐全得满分
7. 时间节奏(10%)：语速均匀、无恐慌加速得高分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- 回避式开头包括："这个话题很有意思"、"说到这个"、"我觉得吧"等不直接切入主题的开头
- 恐慌加速：后半段语速明显快于前半段（>30%）"""


def opinion_statement_user_prompt(
    speech_text: str,
    speech_scores: dict,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    topic: Optional[str] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh"
) -> str:
    """User prompt for opinion statement evaluation."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"
    prompt = f"""请评测以下一分钟观点陈述：

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
1. 开头类型（是否开门见山）
2. 逻辑跳跃和矛盾
3. 冗余表达和口头禅
4. 结构完整性（观点→理由→例子→总结）
5. 时间节奏（前后半段语速对比）"""

    return prompt
