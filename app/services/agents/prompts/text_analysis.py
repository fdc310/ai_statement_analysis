"""
Text analysis prompts.
Extracted from HunyuanService for modular maintenance.
"""
from typing import Optional


def text_structure_system_prompt() -> str:
    """System prompt for text structure analysis."""
    return """你是一个专业的文本分析专家。你的任务是分析用户提供的文本，提取其核心思想和逻辑结构。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

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


def text_structure_user_prompt(
    text: str,
    custom_prompt: Optional[str] = None
) -> str:
    """User prompt for text structure analysis."""
    prompt = f"""请分析以下文本的核心思想和逻辑结构：

## 待分析文本

{text}
"""

    if custom_prompt:
        prompt += f"""
## 额外分析要求

{custom_prompt}
"""

    prompt += """
请严格按照JSON格式输出分析结果。"""

    return prompt


def sentence_interpretation_system_prompt() -> str:
    """System prompt for sentence interpretation analysis."""
    return """你是一个专业的朗读指导专家。你的任务是分析用户朗读的句子，提供朗读指导建议。

你必须严格按照以下JSON格式输出分析结果，不要添加任何额外内容，只输出JSON：

{
    "sentence_analysis": {
        "structure": "<句子结构分析>",
        "meaning": "<句子含义>",
        "key_words": ["<关键词1>", "<关键词2>"],
        "emotion": "<情感基调>"
    },
    "reading_guidance": {
        "pause_positions": ["<停顿位置1>", "<停顿位置2>"],
        "stress_words": ["<重读词1>", "<重读词2>"],
        "intonation": "<语调建议>",
        "speed": "<语速建议>",
        "emotion_expression": "<情感表达建议>"
    },
    "common_mistakes": [
        {
            "mistake": "<常见错误>",
            "correction": "<正确读法>",
            "tip": "<避免技巧>"
        }
    ],
    "practice_suggestions": [
        "<练习建议1>",
        "<练习建议2>"
    ],
    "overall_score": <综合评分0-100>,
    "strengths": ["<优点1>", "<优点2>"],
    "improvements": ["<改进1>", "<改进2>"]
}

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- 停顿位置用标点符号或词语位置表示
- 重读词用加粗标记"""


def sentence_interpretation_user_prompt(
    speech_text: str,
    reference_text: Optional[str] = None,
    speech_scores: dict = None,
    language: str = "zh"
) -> str:
    """User prompt for sentence interpretation analysis."""
    prompt = f"""请分析以下朗读并提供指导：

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

    prompt += """
请严格按照JSON格式输出分析结果。特别注意分析：
1. 句子结构和含义
2. 停顿位置和重读词
3. 语调和语速建议
4. 情感表达建议
5. 常见错误和避免技巧"""

    return prompt
