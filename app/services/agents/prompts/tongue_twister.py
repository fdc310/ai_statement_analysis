"""
Tongue twister analysis prompts.
Extracted from HunyuanService for modular maintenance.
"""
from typing import Optional


def tongue_twister_system_prompt() -> str:
    """System prompt for tongue twister analysis."""
    return """你是一个专业的绕口令发音分析专家。你的任务是分析用户的绕口令朗读，从音素层面分析发音问题。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

{
    "core_phonemes": [
        {
            "phoneme": "<核心音素>",
            "occurrences": <出现次数>,
            "accuracy": <准确度0-100>,
            "common_errors": ["<常见错误1>"]
        }
    ],
    "articulation_analysis": {
        "manner": "<发音方式分析（塞音/擦音/塞擦音/鼻音/边音）>",
        "place": "<发音部位分析（双唇/唇齿/舌尖/舌根等）>",
        "voicing": "<清浊分析>",
        "issues": ["<发音问题1>", "<发音问题2>"]
    },
    "acoustic_features": {
        "rhythm_pattern": "<节奏模式分析>",
        "stress_pattern": "<重音模式分析>",
        "speed_consistency": "<语速一致性分析>",
        "analysis": "<声学特征分析>"
    },
    "confusion_pairs": [
        {
            "pair": ["<混淆音1>", "<混淆音2>"],
            "frequency": <混淆频率>,
            "example": "<混淆示例>"
        }
    ],
    "rhythm_analysis": {
        "syllable_count": <音节数>,
        "pause_positions": ["<停顿位置1>"],
        "rhythm_score": <节奏评分0-100>,
        "analysis": "<节奏分析>"
    },
    "practice_sequence": [
        {
            "focus": "<练习重点>",
            "exercise": "<练习方法>",
            "difficulty": "<难度等级>"
        }
    ],
    "overall_score": <综合评分0-100>,
    "strengths": ["<优点1>", "<优点2>"],
    "improvements": ["<改进1>", "<改进2>"]
}

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- 核心音素：绕口令中容易混淆的音素
- 混淆_pairs：如平翘舌(z/zh)、前后鼻音(an/ang)等"""


def tongue_twister_user_prompt(
    speech_text: str,
    reference_text: Optional[str] = None,
    speech_scores: dict = None,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    language: str = "zh"
) -> str:
    """User prompt for tongue twister analysis."""
    prompt = f"""请分析以下绕口令朗读：

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
请严格按照JSON格式输出分析结果。特别注意分析：
1. 核心音素（绕口令中容易混淆的音素）
2. 发音方式和部位分析
3. 混淆音对（如平翘舌、前后鼻音）
4. 节奏模式和停顿位置
5. 针对性的练习建议

重要提醒：
- 完整度评估应通过对比"朗读内容"与"原文参考"来判断
- 对比两者是否一致，是否有遗漏、错字或多余内容
- 不要使用"语音评分数据"中的"发音完整度"分数（该分数可能不准确）"""

    return prompt


def article_reading_system_prompt() -> str:
    """System prompt for article reading analysis."""
    return """你是一个专业的朗读评测专家。你的任务是分析用户的朗读，从流畅度、语速、停顿和错字等方面进行评测。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

{
    "fluency": {
        "score": <流畅度评分0-100>,
        "interruptions": <中断次数>,
        "repeated_reads": <重复朗读次数>,
        "stutters": <口吃次数>,
        "analysis": "<流畅度分析>"
    },
    "speech_rate": {
        "overall_rate": <整体语速字/分钟>,
        "segment_rates": [
            {"segment": "<段落>", "rate": <语速>}
        ],
        "consistency": "<语速一致性分析>",
        "analysis": "<语速分析>"
    },
    "pause_analysis": {
        "proper_pauses": <合理停顿次数>,
        "improper_pauses": <不当停顿次数>,
        "missed_pauses": <遗漏停顿次数>,
        "pause_positions": ["<停顿位置1>"],
        "analysis": "<停顿分析>"
    },
    "wrong_words": [
        {
            "expected": "<预期字词>",
            "actual": "<实际读出>",
            "position": "<位置>"
        }
    ],
    "overall_score": <综合评分0-100>,
    "improvements": ["<改进1>", "<改进2>"]
}

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- 中断：不自然的停顿或重新开始
- 不当停顿：在不该停顿的地方停顿"""


def article_reading_user_prompt(
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
    """User prompt for article reading analysis."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"
    prompt = f"""请分析以下朗读：

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
请严格按照JSON格式输出分析结果。特别注意分析：
1. 流畅度（中断、重复朗读、口吃）
2. 语速分析（整体语速、分段语速、一致性）
3. 停顿分析（合理停顿、不当停顿、遗漏停顿）
4. 错字检测（与原文对比）

重要提醒：
- 完整度评估应通过对比"朗读内容"与"原文参考"来判断
- 对比两者是否一致，是否有遗漏、错字或多余内容
- 不要使用"语音评分数据"中的"发音完整度"分数（该分数可能不准确）"""

    return prompt
