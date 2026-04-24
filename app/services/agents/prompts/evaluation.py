"""
Evaluation report prompts.
Extracted from HunyuanService for modular maintenance.
Contains: basic, extended, simple JSON, and full JSON report prompts.
"""
from typing import Optional


# ============================================================
# Basic Evaluation Report (JSON format)
# ============================================================

def basic_evaluation_system_prompt(language: str = "zh") -> str:
    """System prompt for basic evaluation report."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"
    return f"""你是一个专业的语音演讲评测专家。你的任务是根据用户提供的语音转文字内容和语音评分数据，生成一份详细的演讲评测报告。

你必须严格按照以下JSON格式输出评测报告，不要添加任何额外内容，只输出JSON：

{{
    "logic_completeness": {{
        "overall_score": <综合评分0-100>,
        "logic_score": <逻辑性评分0-100>,
        "fluency_score": <流畅度评分0-100>,
        "speech_rate_score": <语速评分0-100>,
        "speech_rate_value": <语速数值>,
        "speech_rate_level": "<优秀/良好/一般/较差>",
        "speech_rate_suggestion": "<语速建议>"
    }},
    "structure_visualization": {{
        "arguments": ["<论点1>", "<论点2>", "<论点3>"],
        "conclusion": "<结论要点>"
    }},
    "strengths": ["<优点1>", "<优点2>", "<优点3>"],
    "improvements": ["<改进意见1>", "<改进意见2>", "<改进意见3>"]
}}

语速评分标准（{rate_unit}）：
- 中文：120-180优秀(90-100分)，100-120或180-200良好(70-89分)，80-100或200-220一般(50-69分)，其他较差(0-49分)
- 英文：100-150优秀(90-100分)，80-100或150-180良好(70-89分)，60-80或180-200一般(50-69分)，其他较差(0-49分)

评分规则：
1. 综合评分：根据逻辑性、流畅度、语速等维度加权计算
2. 结构可视化：提取演讲的主要论点和结论
3. 优点：指出演讲中表现出色的地方
4. 改进意见：给出具体可行的改进建议

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 评分要客观公正，有理有据
- 结合语音评分数据（发音准确度、流利度等）进行综合评价"""


def basic_evaluation_user_prompt(
    speech_text: str,
    speech_scores: dict,
    custom_prompt: Optional[str] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None
) -> str:
    """User prompt for basic evaluation report."""
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
请严格按照系统提示中指定的JSON格式生成评测报告。在改进意见中，请特别关注发音待改进的字词。"""

    return prompt


# ============================================================
# Extended Evaluation Report (JSON format with topic + speech rate)
# ============================================================

def extended_evaluation_system_prompt(has_topic: bool, language: str = "zh") -> str:
    """System prompt for extended evaluation report."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"

    topic_field = ""
    if has_topic:
        topic_field = '"topic_relevance_score": <贴题性评分0-100>,'

    base_prompt = f"""你是一个专业的语音演讲评测专家。你的任务是根据用户提供的语音转文字内容和语音评分数据，生成一份详细的演讲评测报告。

你必须严格按照以下JSON格式输出评测报告，不要添加任何额外内容，只输出JSON：

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
    "strengths": ["<优点1>", "<优点2>", "<优点3>"],
    "improvements": ["<改进意见1>", "<改进意见2>", "<改进意见3>"]
}}

语速评分标准（{rate_unit}）：
- 中文：120-180优秀(90-100分)，100-120或180-200良好(70-89分)，80-100或200-220一般(50-69分)，其他较差(0-49分)
- 英文：100-150优秀(90-100分)，80-100或150-180良好(70-89分)，60-80或180-200一般(50-69分)，其他较差(0-49分)"""

    if has_topic:
        base_prompt += """

贴题性评分标准：
- 内容与主题高度相关，论点紧扣主题：90-100分
- 内容基本围绕主题，偶有偏离：70-89分
- 内容部分相关，有明显跑题：50-69分
- 内容与主题关联度低：0-49分"""

    base_prompt += """

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 评分要客观公正，有理有据
- 结合语音评分数据（发音准确度、流利度等）进行综合评价"""

    return base_prompt


def extended_evaluation_user_prompt(
    speech_text: str,
    speech_scores: dict,
    custom_prompt: Optional[str] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    topic: Optional[str] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None
) -> str:
    """User prompt for extended evaluation report."""
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
- 音频时长: {audio_duration or 0:.1f} 秒
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
请严格按照系统提示中指定的JSON格式生成评测报告。"""

    if topic:
        prompt += "请特别关注内容与主题的贴题性分析。"

    if low_score_words:
        prompt += "在改进意见中，请特别关注发音待改进的字词。"

    return prompt


# ============================================================
# Simple Report (JSON format)
# ============================================================

def simple_report_system_prompt(language: str = "zh") -> str:
    """System prompt for simple JSON report."""
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


def simple_report_user_prompt(
    speech_text: str,
    speech_scores: dict,
    low_score_words: Optional[list] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh"
) -> str:
    """User prompt for simple JSON report."""
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


# ============================================================
# Full Report (JSON format with detailed dimensions)
# ============================================================

def full_report_system_prompt(language: str = "zh", has_topic: bool = False) -> str:
    """System prompt for full JSON report."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"

    topic_field = ""
    if has_topic:
        topic_field = '"topic_relevance_score": <贴题性评分0-100>,'

    return f"""你是一个专业的语音演讲评测专家。你的任务是生成一份完整的评测报告。

你必须严格按照以下JSON格式输出，不要添加任何额外内容，只输出JSON：

{{
    "logic_completeness": {{
        "overall_score": <综合评分0-100，由以下维度加权计算：内容角度20%+逻辑结构20%+表达用词20%+流畅度15%+语速15%+口头禅控制10%>,
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
    "verbal_habits": {{
        "filler_words": [
            {{"word": "<口头禅/填充词>", "count": <出现次数>, "example_context": "<出现的上下文示例>"}}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<口头禅占比描述，如每分钟X次>",
        "impact_assessment": "<口头禅对表达效果的影响分析>",
        "suggestion": "<减少口头禅的具体建议>"
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
1. 综合评分：由多维度加权计算，权重为：内容角度(20%) + 逻辑结构(20%) + 表达用词(20%) + 流畅度(15%) + 语速(15%) + 口头禅控制(10%)
2. 语速评价：根据语速数值评分，分析语速对表达效果的影响
3. 内容角度：分析内容的贴题性、深度和覆盖面
4. 逻辑与结构：分析演讲的组织结构、连贯性和论证逻辑
5. 表达与用词：分析用词水平、表达风格和亮点
6. 口头禅分析：识别"嗯"、"啊"、"那个"、"就是"、"然后"、"对吧"、"这个"等口头禅和填充词，统计出现次数，分析对表达的影响
7. 口头禅评分标准：无口头禅(90-100分)，偶尔出现不影响表达(70-89分)，较频繁影响流畅度(50-69分)，严重干扰表达(0-49分)
8. 优点：要详细描述演讲的亮点，每条优点不少于20字，要具体说明体现在哪里

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- weak_paragraphs只包含有低分字词的段落
- 论点和结论要从演讲内容中提取
- 各维度的分析要具体、有针对性
- strengths优点部分要写得详细具体，每条不少于15字，突出演讲者的闪光点"""


def full_report_user_prompt(
    speech_text: str,
    speech_scores: dict,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    topic: Optional[str] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh"
) -> str:
    """User prompt for full JSON report."""
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
