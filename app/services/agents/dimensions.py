"""
Evaluation dimension definitions.
Each dimension defines a focused LLM evaluation task with unified JSON output.
"""
import json
from typing import Optional
from app.services.agents.base_agent import EvaluationContext
from app.services.agents.prompts.common import build_word_info_table, build_low_score_words_table


def _base_data_block(ctx: EvaluationContext, current_dim: str = None) -> str:
    """Build the common data block shared by all dimension prompts."""
    lang = ctx.language or "zh"
    rate_unit = "字/分钟" if lang == "zh" else "词/分钟"
    scores = ctx.scores_data or {}

    parts = [f"## 语音转文字\n\n{ctx.speech_text or '(空)'}"]

    if ctx.ref_text:
        parts.append(f"## 参考原文\n\n{ctx.ref_text}")

    if scores:
        parts.append(f"""## 发音评分

- 准确度: {scores.get('pronunciation_accuracy', 0)}分
- 流利度: {scores.get('pronunciation_fluency', 0)}分
- 完整度: {scores.get('pronunciation_completion', 0)}分
- 综合分: {scores.get('suggested_score', 0)}分""")

    if ctx.speech_rate:
        parts.append(f"""## 语速信息

- 语速: {ctx.speech_rate} {rate_unit}
- 音频时长: {ctx.audio_duration or 0:.1f}秒""")

    stats = ctx.statistics_data or {}
    if stats:
        parts.append(f"""## 统计数据

- 总字数: {stats.get('total_words', 0)}
- 平均准确度: {stats.get('average_accuracy', 0)}分
- 低分字数: {stats.get('low_score_count', 0)}个""")

    low_words = ctx.low_score_words or []
    if low_words:
        parts.append("## 低分字词\n")
        for w in low_words[:15]:
            parts.append(f"- {w.get('word', '')}: 准确度{w.get('accuracy', 0)}分, 流利度{w.get('fluency', 0)}分")

    word_table = build_word_info_table(ctx.word_info_list or [])
    if word_table:
        parts.append(f"## 词语时间戳\n\n{word_table}")

    topic = ctx.request.get("topic") or ""
    if topic:
        parts.append(f"## 演讲主题\n\n{topic}")

    # 添加已完成维度的分析结果（跳过当前维度，避免循环引用）
    for dim_name in ["content", "logic", "expression", "verbal_habits"]:
        if dim_name == current_dim:
            continue
        result = ctx.get_agent_result(f"dim_{dim_name}")
        if result and result.success:
            dim_data = result.data
            if isinstance(dim_data, dict):
                parts.append(f"## {dim_name} 维度分析\n\n{json.dumps(dim_data, ensure_ascii=False, indent=2)}")

    return "\n\n".join(parts)


# ============================================================
# Dimension: Speech Rate (语速评估)
# ============================================================

def speech_rate_system_prompt(ctx: EvaluationContext) -> str:
    lang = ctx.language or "zh"
    rate_unit = "字/分钟" if lang == "zh" else "词/分钟"

    if lang == "zh":
        rate_standards = """语速评分标准（字/分钟）：
- 优秀(90-100分): 120-180字/分钟，语速适中，节奏舒适
- 良好(70-89分): 100-120或180-200字/分钟，略有偏快或偏慢
- 一般(50-69分): 80-100或200-220字/分钟，明显偏快或偏慢
- 较差(0-49分): <80或>220字/分钟，严重影响理解"""
    else:
        rate_standards = """Speech rate scoring (words/min):
- Excellent (90-100): 100-150 words/min, natural and comfortable pace
- Good (70-89): 80-100 or 150-180 words/min, slightly fast or slow
- Fair (50-69): 60-80 or 180-200 words/min, noticeably fast or slow
- Poor (0-49): <60 or >200 words/min, hinders comprehension"""

    return f"""你是语速评估专家。根据提供的语音数据全面评估语速。

输出JSON格式：
{{
    "score": <0-100>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<语速分析>",
    "suggestion": "<改进建议>",
    "details": {{
        "rate_value": <语速数值{rate_unit}>,
        "first_half_rate": <前半段语速>,
        "second_half_rate": <后半段语速>,
        "rate_change": "<语速变化趋势：稳定/逐渐加快/逐渐减慢/波动较大>",
        "panic_acceleration": <是否有恐慌加速 true/false>,
        "impact_on_expression": "<语速对表达效果的影响分析>"
    }}
}}

{rate_standards}

分析要点：
1. 整体语速是否适中
2. 前半段/后半段语速变化（恐慌加速检测：后半段语速明显快于前半段>30%）
3. 语速对表达效果的影响（过快影响理解，过慢影响流畅感）
4. 语速变化趋势分析

只输出纯JSON，不要添加markdown代码块标记。"""


def speech_rate_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请评估以下语音的语速：

{_base_data_block(ctx, "speech_rate")}

严格按JSON格式输出。"""


# ============================================================
# Dimension: Content (内容角度)
# ============================================================

def content_system_prompt(ctx: EvaluationContext) -> str:
    has_topic = bool(ctx.request.get("topic"))

    topic_analysis = ""
    if has_topic:
        topic_analysis = """
贴题性评分标准：
- 高度相关(90-100分): 内容与主题完全契合，论点紧扣主题
- 基本相关(70-89分): 内容基本围绕主题，偶有偏离
- 部分相关(50-69分): 内容部分相关，有明显跑题
- 关联度低(0-49分): 内容与主题关联度很低"""

    return f"""你是内容分析专家。分析演讲内容的质量和深度。

输出JSON格式：
{{
    "score": <0-100>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<内容分析>",
    "suggestion": "<改进建议>",
    "details": {{
        "main_point": "<核心主旨概括，用1-2句话>",
        "depth": "<内容深度分析：论述是否深入、有见地>",
        "coverage": "<内容覆盖面分析：是否全面、有遗漏>",
        "topic_relevance": "<贴题性分析>"{"，评分0-100" if has_topic else ""}
    }}
}}

内容评分标准：
- 优秀(90-100分): 主旨明确，论述深入，有独到见解，覆盖全面
- 良好(70-89分): 主旨清晰，论述较深入，覆盖较全面
- 一般(50-69分): 主旨基本明确，论述一般，有遗漏
- 较差(0-49分): 主旨不清，论述肤浅，覆盖不全
{topic_analysis}

只输出纯JSON，不要添加markdown代码块标记。"""


def content_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下语音内容的质量：

{_base_data_block(ctx, "content")}

严格按JSON格式输出。"""


# ============================================================
# Dimension: Logic (逻辑结构)
# ============================================================

def logic_system_prompt(ctx: EvaluationContext) -> str:
    return """你是逻辑结构分析专家。分析演讲的组织结构和逻辑性。

输出JSON格式：
{
    "score": <0-100>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<逻辑分析>",
    "suggestion": "<改进建议>",
    "details": {
        "organization": "<整体结构分析：是否有清晰的开头、主体、结尾>",
        "coherence": "<连贯性分析：段落之间是否衔接自然>",
        "reasoning": "<论证逻辑分析：论点是否有论据支撑>",
        "arguments": ["<论点1>", "<论点2>"],
        "conclusion": "<结论要点>",
        "logic_jumps": ["<逻辑跳跃点1>"],
        "contradictions": ["<矛盾点1>"]
    }
}

逻辑评分标准：
- 优秀(90-100分): 结构清晰，逻辑严密，论证有力，衔接自然
- 良好(70-89分): 结构较清晰，逻辑较严密，有少量跳跃
- 一般(50-69分): 结构基本清晰，有明显逻辑跳跃
- 较差(0-49分): 结构混乱，逻辑不清，论证无力

分析要点：
1. 整体结构是否清晰（开头→主体→结尾）
2. 段落之间的衔接是否自然
3. 论点是否有充分的论据支撑
4. 是否存在逻辑跳跃或矛盾

只输出纯JSON，不要添加markdown代码块标记。"""


def logic_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下语音内容的逻辑结构：

{_base_data_block(ctx, "logic")}

严格按JSON格式输出。"""


# ============================================================
# Dimension: Expression (表达用词)
# ============================================================

def expression_system_prompt(ctx: EvaluationContext) -> str:
    return """你是表达用词分析专家。分析演讲的用词水平和表达风格。

输出JSON格式：
{
    "score": <0-100>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<表达分析>",
    "suggestion": "<改进建议>",
    "details": {
        "vocabulary_level": "<用词水平分析：词汇丰富度、准确性>",
        "expression_style": "<表达风格：正式/口语化/生动/平实>",
        "highlights": ["<表达亮点1，具体说明好在哪里>", "<表达亮点2>"],
        "word_accuracy": "<用词准确性分析>",
        "rhetoric_usage": "<修辞手法使用情况>"
    }
}

表达用词评分标准：
- 优秀(90-100分): 用词精准丰富，表达生动有力，有修辞亮点
- 良好(70-89分): 用词较准确，表达较流畅，有亮点
- 一般(50-69分): 用词基本准确，表达一般
- 较差(0-49分): 用词不当，表达平淡或混乱

分析要点：
1. 词汇是否丰富、准确
2. 表达风格是否适合场景
3. 是否有精彩的表达亮点
4. 修辞手法的使用情况

只输出纯JSON，不要添加markdown代码块标记。"""


def expression_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下语音的表达与用词：

{_base_data_block(ctx, "expression")}

严格按JSON格式输出。"""


# ============================================================
# Dimension: Verbal Habits (口头禅)
# ============================================================

def verbal_habits_system_prompt(ctx: EvaluationContext) -> str:
    return """你是口头禅分析专家。识别演讲中的口头禅和填充词。

输出JSON格式：
{
    "score": <0-100>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<口头禅分析>",
    "suggestion": "<改进建议>",
    "details": {
        "filler_words": [
            {
                "word": "<口头禅/填充词>",
                "count": <出现次数>,
                "example_context": "<出现的上下文示例>"
            }
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<口头禅占比描述，如每分钟X次>",
        "impact_assessment": "<对表达效果的影响分析>",
        "most_frequent": "<出现最多的口头禅>"
    }
}

口头禅评分标准：
- 优秀(90-100分): 无口头禅或极少，表达干净流畅
- 良好(70-89分): 偶尔出现，不影响整体表达
- 一般(50-69分): 较频繁，影响流畅度
- 较差(0-49分): 严重干扰表达，影响听众理解

常见口头禅列表：
- 中文："嗯"、"啊"、"那个"、"就是"、"然后"、"对吧"、"这个"、"那么"、"其实"、"基本上"
- 英文："um"、"uh"、"like"、"you know"、"basically"、"actually"、"so"、"right"

只输出纯JSON，不要添加markdown代码块标记。"""


def verbal_habits_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下语音中的口头禅和填充词：

{_base_data_block(ctx, "verbal_habits")}

识别口头禅和填充词，统计出现次数，分析对表达的影响。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Strengths (优点)
# ============================================================

def strengths_system_prompt(ctx: EvaluationContext) -> str:
    return """你是演讲优点分析专家。找出演讲的亮点和优势。

输出JSON格式：
{
    "score": <0-100>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<整体正面评价>",
    "suggestion": "<如何保持和发扬优点>",
    "details": {
        "strengths": [
            {
                "category": "<优点类别：内容/逻辑/表达/发音/其他>",
                "description": "<优点描述，不少于15字>",
                "example": "<具体例子或体现之处>"
            }
        ],
        "overall_positive": "<整体正面评价>"
    }
}

优点评分标准：
- 优秀(90-100分): 多个突出亮点，表现全面优秀
- 良好(70-89分): 有明显优点，整体表现良好
- 一般(50-69分): 有少量优点，表现中规中矩
- 较差(0-49分): 优点较少，需要全面改进

分析要点：
1. 内容方面的优点（主旨明确、论述深入等）
2. 逻辑方面的优点（结构清晰、论证有力等）
3. 表达方面的优点（用词精准、风格生动等）
4. 发音方面的优点（发音准确、流畅自然等）

每条优点要求：
- 不少于15字
- 说明具体体现在哪里
- 最好有具体例子

只输出纯JSON，不要添加markdown代码块标记。"""


def strengths_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请找出以下语音演讲的优点和亮点：

{_base_data_block(ctx, "strengths")}

每条优点不少于15字，要具体说明体现在哪里，突出演讲者的闪光点。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Issues & Improvements (问题与改进)
# ============================================================

def issues_system_prompt(ctx: EvaluationContext) -> str:
    has_ref_text = bool(ctx.ref_text)
    completeness_note = ""
    if has_ref_text:
        completeness_note = """
重要提醒：
- 完整度评估应通过对比"语音转文字"内容与"参考原文"来判断
- 对比两者是否一致，是否有遗漏、错字或多余内容
- 不要使用"发音评分"中的"完整度"分数（该分数可能不准确）
- 不要基于"音频时长"和"总字数"自行推断完整度问题
- 例如：不要说"音频时长X秒，总字数Y字，理论朗读时间应更长"这类话"""

    return f"""你是演讲问题分析专家。找出主要问题并给出改进建议。

输出JSON格式：
{{
    "score": <0-100，问题越少分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<问题分析>",
    "suggestion": "<改进建议>",
    "details": {{
        "main_issues": [
            {{
                "issue_type": "<问题类型：发音/逻辑/内容/表达/完整度>",
                "description": "<问题具体描述>",
                "impact_level": "<高/中/低>",
                "example": "<具体例子>",
                "suggested_fix": "<具体改进建议>"
            }}
        ],
        "improvements": ["<改进建议1>", "<改进建议2>", "<改进建议3>"],
        "low_score_words_analysis": "<低分字词整体分析>"
    }}
}}

问题评分标准：
- 优秀(90-100分): 几乎无问题，表现优秀
- 良好(70-89分): 有少量小问题，不影响整体
- 一般(50-69分): 有一些明显问题，需要改进
- 较差(0-49分): 问题较多，需要重点改进
{completeness_note}

只输出纯JSON，不要添加markdown代码块标记。"""


def issues_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下语音演讲的主要问题和改进方向：

{_base_data_block(ctx, "issues")}

如果涉及发音问题，请结合低分字词给出具体分析。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Weak Paragraphs (弱段落)
# ============================================================

def weak_paragraphs_system_prompt(ctx: EvaluationContext) -> str:
    return """你是段落质量分析专家。找出读得不好的段落并分析原因。

输出JSON格式：
{
    "score": <0-100，弱段落越少分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<段落质量分析>",
    "suggestion": "<整体改进建议>",
    "details": {
        "weak_paragraphs": [
            {
                "paragraph_index": <段落索引从1开始>,
                "content": "<段落内容摘要，不超过50字>",
                "low_score_words": [{"word": "<字词>", "accuracy": <准确度分数>}],
                "issue": "<该段落的主要问题>",
                "suggestion": "<针对该段落的改进建议>"
            }
        ],
        "overall_suggestion": "<整体改进建议>"
    }
}

弱段落评分标准：
- 优秀(90-100分): 几乎无弱段落，整体表现优秀
- 良好(70-89分): 少量弱段落，不影响整体
- 一般(50-69分): 有一些弱段落，需要针对性改进
- 较差(0-49分): 多个弱段落，需要全面改进

分析要点：
1. 结合低分字词定位弱段落
2. 分析每个弱段落的具体问题
3. 给出针对性的改进建议
4. 最多找出3个最弱的段落

只输出纯JSON，不要添加markdown代码块标记。"""


def weak_paragraphs_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下语音中读得不太好的段落：

{_base_data_block(ctx, "weak_paragraphs")}

结合低分字词，找出哪些段落读得不太好，并给出改进建议。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Tongue Twister - Completeness (绕口令完整度：多读/漏读)
# ============================================================

def tw_completeness_system_prompt(ctx: EvaluationContext) -> str:
    return """你是绕口令完整度分析专家。逐字对比原文和朗读内容，找出多读和漏读。

输出JSON格式：
{
    "score": <0-100，完整度越高分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<完整度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "extra_words": {
            "count": <多读字词数量>,
            "words": ["<多读的字词1>", "<多读的字词2>"],
            "description": "<对多读情况的简要说明>"
        },
        "missed_words": {
            "count": <漏读字词数量>,
            "words": ["<漏读的字词1>", "<漏读的字词2>"],
            "description": "<对漏读情况的简要说明>"
        },
        "accuracy_rate": <完整度数字，如95.5，不要加%符号>
    }
}

分析规则：
1. 多读判断：将实际朗读文本与绕口令原文逐字对比，找出朗读中有但原文中没有的字词
2. 漏读判断：找出原文中有但朗读中缺少的字词
3. 完整度 = (总字数 - 多读数 - 漏读数) / 总字数 × 100%

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 多读和漏读的判断要精确，逐字对比"""


def tw_completeness_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下绕口令朗读的完整度：

{_base_data_block(ctx, "tw_completeness")}

逐字对比原文和朗读内容，找出多读和漏读的字词。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Tongue Twister - Pronunciation (绕口令发音问题)
# ============================================================

def tw_pronunciation_system_prompt(ctx: EvaluationContext) -> str:
    return """你是绕口令发音分析专家。基于SOE低分字词数据，分析具体的发音问题。

输出JSON格式：
{
    "score": <0-100，发音问题越少分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<发音分析>",
    "suggestion": "<改进建议>",
    "details": {
        "pronunciation_issues": [
            {
                "word": "<发音有问题的字词>",
                "accuracy_score": <SOE准确度评分>,
                "issue_description": "<具体发音问题描述，如声母/韵母/声调问题>",
                "correct_pronunciation": "<正确的发音要领>",
                "practice_tip": "<针对性练习建议>"
            }
        ],
        "total_issues_count": <发音问题总数>,
        "most_difficult_sounds": ["<最难发的音1>", "<最难发的音2>"]
    }
}

分析规则：
1. 发音问题：基于SOE评测的低分字词(accuracy<90分)，分析具体的发音问题
2. 问题类型：声母问题、韵母问题、声调问题、混淆音问题（如平翘舌z/zh、前后鼻音an/ang）
3. 给出正确的发音要领和针对性练习建议

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 发音问题要结合SOE低分数据，给出具体的声母/韵母/声调分析"""


def tw_pronunciation_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下绕口令朗读的发音问题：

{_base_data_block(ctx, "tw_pronunciation")}

结合SOE低分字词数据分析具体发音问题。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Tongue Twister - Fluency (绕口令流畅度)
# ============================================================

def tw_fluency_system_prompt(ctx: EvaluationContext) -> str:
    return """你是绕口令流畅度分析专家。基于词级时间戳分析流畅度、节奏和语速。

输出JSON格式：
{
    "score": <0-100，流畅度越高分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<流畅度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "overall_fluency": "<整体流畅度评价：优秀/良好/一般/较差>",
        "long_pauses": [
            {
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "suggestion": "<针对该停顿的建议>"
            }
        ],
        "rhythm_assessment": "<节奏评价：绕口令的节奏感是否把握得当>",
        "speed_assessment": "<语速评价：是否适合该绕口令的难度>"
    }
}

分析规则：
1. 流畅度分析：基于词级时间戳，分析停顿（相邻词间隔>2000ms为长停顿）、节奏和语速
2. 绕口令的停顿标准比普通阅读更严格，使用2000ms作为长停顿阈值
3. 节奏感：绕口令需要有节奏感，不能太平淡也不能太急促
4. 语速：绕口令需要适当语速，过快容易出错，过慢失去节奏感

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 如果没有长停顿，long_pauses为空数组"""


def tw_fluency_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下绕口令朗读的流畅度：

{_base_data_block(ctx, "tw_fluency")}

基于时间戳分析流畅度、节奏和语速。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Tongue Twister - Strengths (绕口令优势)
# ============================================================

def tw_strengths_system_prompt(ctx: EvaluationContext) -> str:
    return """你是绕口令优点分析专家。找出朗读中的亮点和优势。

输出JSON格式：
{
    "score": <0-100，优点越多分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<整体正面评价>",
    "suggestion": "<如何保持和发扬优点>",
    "details": {
        "strengths": [
            {
                "category": "<优点类别：发音/完整度/流畅度/节奏/其他>",
                "description": "<优点描述，不少于15字>",
                "example": "<具体例子或体现之处>"
            }
        ],
        "overall_positive": "<整体正面评价>"
    }
}

分析要点：
1. 完整度方面的优点（多读/漏读情况好）
2. 发音方面的优点（发音准确、清晰）
3. 流畅度方面的优点（停顿少、节奏好）
4. 节奏感方面的优点（绕口令节奏把握得当）

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 每条优点不少于15字，要具体说明体现在哪里"""


def tw_strengths_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请找出以下绕口令朗读的优点和亮点：

{_base_data_block(ctx, "tw_strengths")}

从完整度、发音、流畅度、节奏等角度找出优点。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Opinion Statement - Viewpoint (一分钟观点陈述-观点明确性)
# ============================================================

def op_viewpoint_system_prompt(ctx: EvaluationContext) -> str:
    return """你是观点表达分析专家。分析陈述者的观点是否明确、开头是否直接。

输出JSON格式：
{
    "score": <0-100，观点越明确分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<观点表达分析>",
    "suggestion": "<改进建议>",
    "details": {
        "has_clear_viewpoint": <是否有明确观点，true/false>,
        "viewpoint_summary": "<用一句话概括核心观点，若无明确观点则写'未提出明确观点'>",
        "opening_type": "<开头类型：直接亮明观点/渐进引入/回避式开头/模糊开头>",
        "opening_quote": "<开头原文前30字>",
        "evasion_signals": ["<回避性表达，如'我觉得这个问题比较复杂'、'这个要看情况'>"],
        "assessment": "<观点表达评价，分析是否开门见山、观点是否鲜明>"
    }
}

评分标准：
- 优秀(90-100分): 开门见山，观点鲜明有力
- 良好(70-89分): 有明确观点但表述不够直接
- 一般(50-69分): 观点模糊，需要听者推断
- 较差(0-49分): 没有明确观点，全程回避或模棱两可

回避式开头识别规则：
- "我觉得这个问题比较复杂" → 回避
- "这个要从多个角度来看" → 回避
- "关于这个话题其实很多人都讨论过" → 回避
- "我认为XX是对的/XX是最重要的" → 直接亮明观点（正面示例）

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def op_viewpoint_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下一分钟观点陈述的观点明确性：

{_base_data_block(ctx, "op_viewpoint")}

重点分析：是否有清晰观点？开头是否直接？是否存在回避式表达？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Opinion Statement - Structure (一分钟观点陈述-结构完整度)
# ============================================================

def op_structure_system_prompt(ctx: EvaluationContext) -> str:
    return """你是结构分析专家。分析陈述的结构完整性，是否包含观点→理由→举例→总结。

输出JSON格式：
{
    "score": <0-100，结构越完整分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<结构完整度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "has_viewpoint": <是否有观点环节，true/false>,
        "has_reason": <是否有理由论证，true/false>,
        "has_example": <是否有举例支撑，true/false>,
        "has_summary": <是否有总结收尾，true/false>,
        "structure_pattern": "<实际结构模式描述，如'观点→理由→总结（缺少举例）'>",
        "ideal_pattern": "观点→理由→举例→总结",
        "missing_parts": ["<缺失的结构部分>"],
        "assessment": "<结构完整度评价>"
    }
}

评分标准：
- 优秀(90-100分): 观点→理由→举例→总结 四要素完整
- 良好(70-89分): 缺少一个要素但整体连贯
- 一般(50-69分): 缺少两个要素，结构松散
- 较差(0-49分): 无明显结构，意识流表达

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def op_structure_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下一分钟观点陈述的结构完整度：

{_base_data_block(ctx, "op_structure")}

重点分析：是否包含观点→理由→举例→总结的完整结构？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Opinion Statement - Logic (一分钟观点陈述-逻辑清晰度)
# ============================================================

def op_logic_system_prompt(ctx: EvaluationContext) -> str:
    return """你是逻辑分析专家。分析陈述的逻辑清晰度，是否存在跳跃、矛盾、论据堆砌。

输出JSON格式：
{
    "score": <0-100，逻辑越清晰分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<逻辑清晰度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "logic_jumps": [
            {
                "from_point": "<跳跃前的内容要点>",
                "to_point": "<跳跃后的内容要点>",
                "description": "<跳跃描述>"
            }
        ],
        "contradictions": [
            {
                "statement_a": "<矛盾表述A>",
                "statement_b": "<矛盾表述B>",
                "description": "<矛盾分析>"
            }
        ],
        "argument_piling": {
            "detected": <是否存在论据堆砌（只罗列不论证），true/false>,
            "description": "<堆砌情况描述>"
        },
        "reasoning_chain": "<论证链条描述，如'观点A←因为B←例如C←所以A'>",
        "assessment": "<逻辑清晰度评价>"
    }
}

评分标准：
- 优秀(90-100分): 论证链清晰，因果关系明确，无矛盾
- 良好(70-89分): 整体逻辑通顺，偶有小跳跃
- 一般(50-69分): 存在明显逻辑跳跃或论据堆砌
- 较差(0-49分): 逻辑混乱，自相矛盾

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def op_logic_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下一分钟观点陈述的逻辑清晰度：

{_base_data_block(ctx, "op_logic")}

重点分析：是否存在逻辑跳跃、矛盾、论据堆砌？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Opinion Statement - Time Rhythm (一分钟观点陈述-时间节奏)
# ============================================================

def op_time_rhythm_system_prompt(ctx: EvaluationContext) -> str:
    return """你是时间节奏分析专家。分析陈述的时间分配和语速变化。

输出JSON格式：
{
    "score": <0-100，时间节奏越合理分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<时间节奏分析>",
    "suggestion": "<改进建议>",
    "details": {
        "total_duration_seconds": <总时长秒>,
        "duration_level": "<时间判定：严重超时/略微超时/适中/偏短/过短>",
        "first_half_rate": <前半段语速(字/分钟)>,
        "second_half_rate": <后半段语速(字/分钟)>,
        "rate_change": "<语速变化：加速/减速/平稳>",
        "panic_acceleration": <后半段是否存在慌张加速，true/false>,
        "time_allocation": {
            "opening_seconds": <开头部分用时秒>,
            "body_seconds": <主体论述用时秒>,
            "closing_seconds": <收尾部分用时秒>,
            "assessment": "<时间分配评价>"
        },
        "assessment": "<时间节奏评价>"
    }
}

评分标准：
- 优秀(90-100分): 50-65秒，节奏均匀，收尾从容
- 良好(70-89分): 45-70秒，节奏基本稳定
- 一般(50-69分): 30-45秒或70-80秒，节奏有波动
- 较差(0-49分): <30秒或>80秒，后半段明显加速/草草收场

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 利用词级时间戳分析前后半段语速变化
- 如果音频时长为0或未提供，给出合理推断"""


def op_time_rhythm_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下一分钟观点陈述的时间节奏：

{_base_data_block(ctx, "op_time_rhythm")}

重点分析：时间分配是否合理？前后半段语速变化？是否存在慌张加速？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Opinion Statement - Expression (一分钟观点陈述-表达精炼度)
# ============================================================

def op_expression_system_prompt(ctx: EvaluationContext) -> str:
    return """你是表达精炼度分析专家。分析陈述的口头禅、冗余表达和有效内容占比。

输出JSON格式：
{
    "score": <0-100，表达越精炼分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<表达精炼度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "filler_words": [
            {"word": "<口头禅/填充词>", "count": <出现次数>, "example_context": "<出现的上下文示例>"}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<废话比例描述，如每分钟X次口头禅>",
        "redundant_expressions": [
            {
                "expression": "<冗余表达原文>",
                "issue": "<问题描述，如重复啰嗦/无意义修饰/空泛套话>",
                "suggestion": "<精简建议>"
            }
        ],
        "effective_content_ratio": "<有效内容占比估算，如80%>",
        "assessment": "<表达冗余度评价>"
    }
}

评分标准：
- 优秀(90-100分): 无口头禅，语言干练，有效内容占比>90%
- 良好(70-89分): 偶有口头禅，表达基本精炼
- 一般(50-69分): 较多口头禅或冗余表达，有效内容60-80%
- 较差(0-49分): 大量废话，口头禅严重干扰表达

常见口头禅列表：
"然后"、"就是"、"其实"、"那个"、"嗯"、"啊"、"这个"、"对吧"、"反正"、"所以说"、"怎么说呢"

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def op_expression_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下一分钟观点陈述的表达精炼度：

{_base_data_block(ctx, "op_expression")}

重点分析：口头禅频率？废话比例？表达是否精炼？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Impromptu Reaction - Reaction Speed (即兴反应-反应速度)
# ============================================================

def ir_reaction_speed_system_prompt(ctx: EvaluationContext) -> str:
    return """你是反应速度分析专家。分析即兴反应的起步速度和情绪表现。

输出JSON格式：
{
    "score": <0-100，反应越快分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<反应速度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "first_word_time_ms": <第一个词出现的时间戳毫秒>,
        "opening_speed": "<起步判断：果断开口/犹豫拖延/大量填充词起手>",
        "panic_signals": <是否存在明显慌乱(如语速突变、结巴、大量"嗯""啊")，true/false>,
        "thinking_pauses": [
            {
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "position_time_ms": <停顿发生的时间点毫秒>
            }
        ],
        "assessment": "<起步反应速度与情绪表现的详细评价>"
    }
}

评分标准：
- 优秀(90-100分): 果断开口(<500ms)，无慌乱信号，思考停顿少
- 良好(70-89分): 短暂思考(500-1500ms)，停顿适度，情绪稳定
- 一般(50-69分): 明显犹豫(1500-3000ms)或大量填充词起手，停顿较多
- 较差(0-49分): 长时间沉默(>3000ms)或明显慌乱(语速突变、频繁结巴)

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 反应速度分析需基于时间戳数据"""


def ir_reaction_speed_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下即兴反应的反应速度：

{_base_data_block(ctx, "ir_reaction_speed")}

重点分析：开口时间、慌乱信号、思考停顿。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Impromptu Reaction - Structure (即兴反应-结构形成)
# ============================================================

def ir_structure_system_prompt(ctx: EvaluationContext) -> str:
    return """你是结构形成分析专家。分析即兴反应的结构形成速度和清晰度。

输出JSON格式：
{
    "score": <0-100，结构越清晰分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<结构形成分析>",
    "suggestion": "<改进建议>",
    "details": {
        "formed_in_15s": <是否在开场(约前15秒)内建立主线结构，true/false>,
        "structure_signal": "<结构信号词，如'我会从两个方面说'、'首先其次'等，若无则写'无明确结构信号'>",
        "structure_pattern": "<实际表现出的结构，如'总分总'、'并列式'、'无序散发'>",
        "has_opening": <是否有开头，true/false>,
        "has_body": <是否有主体论述，true/false>,
        "has_closing": <是否有结尾，true/false>,
        "assessment": "<结构形成速度和清晰度的评价>"
    }
}

评分标准：
- 优秀(90-100分): 前15秒内建立主线，结构信号明确，开头-主体-结尾完整
- 良好(70-89分): 有基本结构，但形成较慢或不够清晰
- 一般(50-69分): 结构模糊，无明确信号词
- 较差(0-49分): 无明显结构，全程无序散发

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 结构形成速度重点看前15秒
- 音频时长<10秒的发言，结构分上限40分"""


def ir_structure_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下即兴反应的结构形成：

{_base_data_block(ctx, "ir_structure")}

重点分析：前15秒是否建立主线？结构信号词？开头-主体-结尾完整性？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Impromptu Reaction - Content Relevance (即兴反应-内容相关性)
# ============================================================

def ir_content_relevance_system_prompt(ctx: EvaluationContext) -> str:
    return """你是内容相关性分析专家。分析即兴反应是否切题、是否有实质性回应。

输出JSON格式：
{
    "score": <0-100，越切题分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<内容相关性分析>",
    "suggestion": "<改进建议>",
    "details": {
        "topic_relevance": "<切题度判定：紧扣主题/略微偏题/完全跑题/复述题目未作回应>",
        "is_mere_repetition": <用户是否只是复述/朗读了场景题目而非做出回应，true/false>,
        "repetition_ratio": "<与场景题目的文字重叠比例估算，如'90%'、'30%'、'0%'>",
        "has_original_response": <是否包含用户自己的原创回应内容（观点、共情、建议等），true/false>,
        "on_topic": <是否切题，true/false>,
        "topic_drift": <是否跑题，true/false>,
        "off_topic_parts": ["<跑题的部分内容>"],
        "content_depth": "<内容深度：有独到见解/有基本论述/内容单薄/几乎无内容>",
        "relevance_description": "<相关性描述，分析回答是否紧扣场景，是否有实质性回应>",
        "assessment": "<内容相关性评价，如果是复述题目必须明确指出>"
    }
}

评分标准：
- 优秀(90-100分): 紧扣场景，有实质性原创回应，内容有深度和独到见解
- 良好(70-89分): 基本切题，有自己的回应但深度一般
- 一般(50-69分): 部分相关但内容单薄，或有明显跑题
- 较差(0-49分): 严重跑题或答非所问，内容空洞

核心评测原则：
1. 区分"回应"与"复述"：即兴反应的本质是对场景/题目做出自己的回应
2. 有效回应：用自己的话对场景做出反应、评价、共情、建议、延伸等
3. 无效复述：只是重复、朗读或转述场景题目本身，没有自己的观点
4. 如果用户只是复述了场景题目本身，该项最高不超过20分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def ir_content_relevance_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下即兴反应的内容相关性：

{_base_data_block(ctx, "ir_content_relevance")}

重点分析：是否切题？是回应还是复述题目？是否有实质性内容？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Impromptu Reaction - Logic (即兴反应-逻辑连贯度)
# ============================================================

def ir_logic_system_prompt(ctx: EvaluationContext) -> str:
    return """你是逻辑连贯度分析专家。分析即兴反应的思维连贯性和过渡质量。

输出JSON格式：
{
    "score": <0-100，逻辑越连贯分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<逻辑连贯度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "coherence_level": "<连贯程度：流畅连贯/基本连贯/偶有跳跃/逻辑混乱/内容不足无法判断>",
        "logic_jumps": [
            {
                "from_point": "<跳跃前的内容>",
                "to_point": "<跳跃后的内容>",
                "description": "<思维跳跃或话题中断的具体表现>"
            }
        ],
        "transition_quality": "<过渡质量评价>",
        "assessment": "<逻辑连贯性与切题度的评价>"
    }
}

评分标准：
- 优秀(90-100分): 逻辑流畅，论点递进清晰，过渡自然，无跳跃
- 良好(70-89分): 基本连贯，偶有小跳跃
- 一般(50-69分): 连贯性一般，跳跃明显或话题中断
- 较差(0-49分): 逻辑混乱或内容过少无法体现逻辑

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def ir_logic_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下即兴反应的逻辑连贯度：

{_base_data_block(ctx, "ir_logic")}

重点分析：思维是否连贯？过渡是否自然？是否有跳跃或中断？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Impromptu Reaction - Expression (即兴反应-表达精炼度)
# ============================================================

def ir_expression_system_prompt(ctx: EvaluationContext) -> str:
    return """你是表达精炼度分析专家。分析即兴反应的口头禅和冗余表达。

输出JSON格式：
{
    "score": <0-100，表达越精炼分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<表达精炼度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "filler_words": [
            {"word": "<嗯/啊/然后/就是说等口头禅>", "count": <出现次数>}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<废话比例描述>",
        "redundancy_level": "<冗余度判定：极低/正常/偏高/极高>",
        "effective_content_ratio": "<有效内容占比估算>",
        "assessment": "<表达流畅度及填充词比例的评价>"
    }
}

评分标准：
- 优秀(90-100分): 无口头禅，表达干练，每句话都有信息量
- 良好(70-89分): 偶有口头禅，基本精炼
- 一般(50-69分): 较多口头禅或冗余表达
- 较差(0-49分): 大量废话，严重干扰

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 分析要具体，引用原文内容"""


def ir_expression_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下即兴反应的表达精炼度：

{_base_data_block(ctx, "ir_expression")}

重点分析：口头禅频率？冗余度？有效内容占比？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Story Reading - Structure (小故事-结构分析)
# ============================================================

def sr_structure_system_prompt(ctx: EvaluationContext) -> str:
    return """你是故事结构分析专家。分析故事阅读的结构完整性。

输出JSON格式：
{
    "score": <0-100，结构越完整分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<结构分析>",
    "suggestion": "<改进建议>",
    "details": {
        "opening": "<开头情况：有/无，简短描述>",
        "development": "<发展情况：描述事件发展过程>",
        "climax": "<高潮情况：有/无，简短描述>",
        "ending": "<结尾情况：有/无/仓促，简短描述>",
        "overall_assessment": "<整体结构评价>"
    }
}

评分标准（满分30分）：
- 有完整的开头、发展、高潮、结尾各得7-8分
- 缺少开头扣7分，缺少发展扣8分，缺少高潮扣8分，缺少结尾扣7分
- 结尾仓促或开头不完整各扣3-5分
- 注意：很多故事本身可能没有明显的高潮结构（如日常叙事、简单描述类故事），此时不应因"缺少高潮"而扣分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 严格对照原始故事文本评估结构完整性"""


def sr_structure_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下故事阅读的结构：

{_base_data_block(ctx, "sr_structure")}

重点分析：开头、发展、高潮、结尾是否完整？
严格按JSON格式输出。"""


# ============================================================
# Dimension: Story Reading - Logic (小故事-逻辑分析)
# ============================================================

def sr_logic_system_prompt(ctx: EvaluationContext) -> str:
    return """你是故事逻辑分析专家。分析故事阅读的逻辑连贯性。

输出JSON格式：
{
    "score": <0-100，逻辑越连贯分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<逻辑分析>",
    "suggestion": "<改进建议>",
    "details": {
        "time_jumps": <时间跳跃次数>,
        "causal_errors": <因果错误次数>,
        "missing_events": <事件遗漏次数>,
        "logical_contradictions": <逻辑矛盾次数>,
        "overall_assessment": "<整体逻辑评价>"
    }
}

评分标准（满分25分）：
- 每处时间跳跃扣3分，因果错误扣4分，事件遗漏扣3分，逻辑矛盾扣5分
- 与原文对比，遗漏关键情节每处扣3-5分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 严格对照原始故事文本评估逻辑连贯性"""


def sr_logic_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下故事阅读的逻辑：

{_base_data_block(ctx, "sr_logic")}

重点分析：时间跳跃、因果错误、事件遗漏、逻辑矛盾。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Story Reading - Fluency (小故事-流畅度分析)
# ============================================================

def sr_fluency_system_prompt(ctx: EvaluationContext) -> str:
    return """你是故事流畅度分析专家。基于时间戳分析故事阅读的流畅度。

输出JSON格式：
{
    "score": <0-100，流畅度越高分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<流畅度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "long_pauses_count": <长停顿(>3秒)次数>,
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
        "sentence_completion_rate": <句子完整度0-100>,
        "overall_assessment": "<整体流畅度评价>"
    }
}

评分标准（满分25分）：
- 每处长停顿(>3秒)扣2分，重复修正每次扣1分，填充词每3个扣1分
- 句子完整度低于80%额外扣5分

时间戳分析规则：
- 长停顿：相邻词语间隔超过3000ms（3秒）
- 重复修正：相同或相似词语在短时间内重复出现
- 填充词：如"啊"、"呃"、"那个"、"这个"、"嗯"等

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 利用词级时间戳精确分析中断和停顿"""


def sr_fluency_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下故事阅读的流畅度：

{_base_data_block(ctx, "sr_fluency")}

基于词级时间戳分析长停顿、重复修正、填充词使用情况。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Story Reading - Event Distribution (小故事-事件分布)
# ============================================================

def sr_event_distribution_system_prompt(ctx: EvaluationContext) -> str:
    return """你是事件分布分析专家。分析故事阅读中各事件的时间分配。

输出JSON格式：
{
    "score": <0-100，分布越合理分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<事件分布分析>",
    "suggestion": "<改进建议>",
    "details": {
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
    }
}

评分标准（满分20分）：
- 事件时间分配严重不均匀扣5-10分
- 某段事件过于冗长或过于简略各扣3-5分

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 事件分布要根据时间戳分析
- 如果没有明确事件划分，根据内容合理划分"""


def sr_event_distribution_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下故事阅读的事件分布：

{_base_data_block(ctx, "sr_event_distribution")}

基于时间戳分析各事件的时长和分布是否合理。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Article Reading - Completeness (文章朗读完整度)
# ============================================================

def ar_completeness_system_prompt(ctx: EvaluationContext) -> str:
    return """你是文章朗读完整度分析专家。逐字对比原文和朗读内容，找出多读、漏读和读错。

输出JSON格式：
{
    "score": <0-100，完整度越高分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<完整度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "extra_words": {
            "count": <多读字词数量>,
            "words": ["<多读的字词1>", "<多读的字词2>"],
            "description": "<对多读情况的简要说明>"
        },
        "missed_words": {
            "count": <漏读字词数量>,
            "words": ["<漏读的字词1>", "<漏读的字词2>"],
            "description": "<对漏读情况的简要说明>"
        },
        "wrong_words": [
            {
                "original": "<原文字词>",
                "actual": "<实际读成的字词>",
                "position": "<大致位置描述，如第几段第几句>"
            }
        ],
        "accuracy_rate": <完整度数字，如95.5，不要加%符号>
    }
}

分析规则：
1. 多读判断：将实际朗读文本与原文逐字对比，找出朗读中有但原文中没有的字词
2. 漏读判断：找出原文中有但朗读中缺少的字词
3. 读错判断：找出实际读出的字词与原文不一致的地方
4. 完整度 = (总字数 - 多读数 - 漏读数 - 读错数) / 总字数 × 100%

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 判断要精确，逐字对比"""


def ar_completeness_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下文章朗读的完整度：

{_base_data_block(ctx, "ar_completeness")}

逐字对比原文和朗读内容，找出多读、漏读和读错的字词。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Article Reading - Pronunciation (文章朗读发音问题)
# ============================================================

def ar_pronunciation_system_prompt(ctx: EvaluationContext) -> str:
    return """你是文章朗读发音分析专家。基于SOE低分字词数据，分析具体的发音问题。

输出JSON格式：
{
    "score": <0-100，发音问题越少分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<发音分析>",
    "suggestion": "<改进建议>",
    "details": {
        "pronunciation_issues": [
            {
                "word": "<发音有问题的字词>",
                "accuracy_score": <SOE准确度评分>,
                "issue_description": "<具体发音问题描述>",
                "correct_pronunciation": "<正确的发音要领>",
                "practice_tip": "<针对性练习建议>"
            }
        ],
        "total_issues_count": <发音问题总数>,
        "most_difficult_sounds": ["<最难发的音1>", "<最难发的音2>"]
    }
}

分析规则：
1. 发音问题：基于SOE评测的低分字词(accuracy<90分)，分析具体的发音问题
2. 问题类型：声母问题、韵母问题、声调问题、混淆音问题
3. 给出正确的发音要领和针对性练习建议

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 发音问题要结合SOE低分数据，给出具体的声母/韵母/声调分析"""


def ar_pronunciation_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下文章朗读的发音问题：

{_base_data_block(ctx, "ar_pronunciation")}

结合SOE低分字词数据分析具体发音问题。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Article Reading - Fluency (文章朗读流畅度)
# ============================================================

def ar_fluency_system_prompt(ctx: EvaluationContext) -> str:
    return """你是文章朗读流畅度分析专家。基于词级时间戳分析流畅度。

输出JSON格式：
{
    "score": <0-100，流畅度越高分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<流畅度分析>",
    "suggestion": "<改进建议>",
    "details": {
        "overall_fluency": "<整体流畅度评价：优秀/良好/一般/较差>",
        "interruptions": [
            {
                "position": "<中断发生的位置描述，如第几段>",
                "before_word": "<中断前的词语>",
                "after_word": "<中断后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "type": "<类型：异常停顿/重复读/卡壳>"
            }
        ],
        "repeated_reads": [
            {
                "word": "<被重复读的词语或句段>",
                "position": "<位置描述>",
                "count": <重复次数>
            }
        ],
        "stutters": ["<明显卡壳的位置和内容描述>"]
    }
}

分析规则：
1. 异常停顿：相邻词间隔>1500ms（非标点处）判定为异常停顿
2. 重复读：相同或相似词语在短时间内重复出现
3. 卡壳：在非停顿位置出现明显的犹豫或断续
4. 流畅度评分标准：无明显中断(90-100分)，偶有停顿不影响理解(70-89分)，多处中断影响流畅度(50-69分)，严重卡顿(0-49分)

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 流畅度的中断要精确定位到具体位置"""


def ar_fluency_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下文章朗读的流畅度：

{_base_data_block(ctx, "ar_fluency")}

基于时间戳分析流畅度，识别异常停顿、重复读、卡壳。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Article Reading - Pause (文章朗读停顿分析)
# ============================================================

def ar_pause_system_prompt(ctx: EvaluationContext) -> str:
    return """你是文章朗读停顿分析专家。分析断句停顿是否合理。

输出JSON格式：
{
    "score": <0-100，停顿越合理分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<停顿分析>",
    "suggestion": "<改进建议>",
    "details": {
        "proper_pauses": <在标点/语义边界处正确停顿的次数>,
        "improper_pauses": [
            {
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "context": "<该停顿所在的句子>",
                "issue": "<问题描述，如：停顿打断了语义结构>"
            }
        ],
        "missed_pauses": [
            {
                "position": "<应该停顿但没有停顿的位置>",
                "context": "<所在句子>",
                "suggestion": "<建议>"
            }
        ],
        "overall_assessment": "<整体断句停顿评价>"
    }
}

分析规则：
1. 以原文标点和语义边界为基准
2. 判断用户停顿是否出现在正确位置（标点处、语义边界处）
3. 如果停顿打断了语义结构，明确指出问题
4. 找出应该停顿但没有停顿的位置

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 断句分析要结合原文标点判断停顿合理性"""


def ar_pause_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请分析以下文章朗读的停顿：

{_base_data_block(ctx, "ar_pause")}

判断停顿是否在标点/语义边界处，指出打断语义的不当停顿。
严格按JSON格式输出。"""


# ============================================================
# Dimension: Article Reading - Strengths (文章朗读优势)
# ============================================================

def ar_strengths_system_prompt(ctx: EvaluationContext) -> str:
    return """你是文章朗读优点分析专家。找出朗读中的亮点和优势。

输出JSON格式：
{
    "score": <0-100，优点越多分越高>,
    "level": "<优秀/良好/一般/较差>",
    "analysis": "<整体正面评价>",
    "suggestion": "<如何保持和发扬优点>",
    "details": {
        "strengths": [
            {
                "category": "<优点类别：发音/完整度/流畅度/停顿/语速/其他>",
                "description": "<优点描述，不少于15字>",
                "example": "<具体例子或体现之处>"
            }
        ],
        "overall_positive": "<整体正面评价>"
    }
}

分析要点：
1. 完整度方面的优点（多读/漏读/读错情况好）
2. 发音方面的优点（发音准确、清晰）
3. 流畅度方面的优点（停顿少、无卡壳）
4. 停顿方面的优点（断句合理、节奏好）
5. 语速方面的优点（语速适中、稳定）

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 每条优点不少于15字，要具体说明体现在哪里"""


def ar_strengths_user_prompt(ctx: EvaluationContext) -> str:
    return f"""请找出以下文章朗读的优点和亮点：

{_base_data_block(ctx, "ar_strengths")}

从完整度、发音、流畅度、停顿、语速等角度找出优点。
严格按JSON格式输出。"""


# ============================================================
# Dimension registry
# ============================================================

# Maps dimension name -> (system_prompt_fn, user_prompt_fn)
DIMENSION_REGISTRY = {
    # 通用维度
    "speech_rate": (speech_rate_system_prompt, speech_rate_user_prompt),
    "content": (content_system_prompt, content_user_prompt),
    "logic": (logic_system_prompt, logic_user_prompt),
    "expression": (expression_system_prompt, expression_user_prompt),
    "verbal_habits": (verbal_habits_system_prompt, verbal_habits_user_prompt),
    "strengths": (strengths_system_prompt, strengths_user_prompt),
    "issues": (issues_system_prompt, issues_user_prompt),
    "weak_paragraphs": (weak_paragraphs_system_prompt, weak_paragraphs_user_prompt),
    # 绕口令专用维度
    "tw_completeness": (tw_completeness_system_prompt, tw_completeness_user_prompt),
    "tw_pronunciation": (tw_pronunciation_system_prompt, tw_pronunciation_user_prompt),
    "tw_fluency": (tw_fluency_system_prompt, tw_fluency_user_prompt),
    "tw_strengths": (tw_strengths_system_prompt, tw_strengths_user_prompt),
    # 文章朗读专用维度
    "ar_completeness": (ar_completeness_system_prompt, ar_completeness_user_prompt),
    "ar_pronunciation": (ar_pronunciation_system_prompt, ar_pronunciation_user_prompt),
    "ar_fluency": (ar_fluency_system_prompt, ar_fluency_user_prompt),
    "ar_pause": (ar_pause_system_prompt, ar_pause_user_prompt),
    "ar_strengths": (ar_strengths_system_prompt, ar_strengths_user_prompt),
    # 一分钟观点陈述专用维度
    "op_viewpoint": (op_viewpoint_system_prompt, op_viewpoint_user_prompt),
    "op_structure": (op_structure_system_prompt, op_structure_user_prompt),
    "op_logic": (op_logic_system_prompt, op_logic_user_prompt),
    "op_time_rhythm": (op_time_rhythm_system_prompt, op_time_rhythm_user_prompt),
    "op_expression": (op_expression_system_prompt, op_expression_user_prompt),
    # 即兴反应专用维度
    "ir_reaction_speed": (ir_reaction_speed_system_prompt, ir_reaction_speed_user_prompt),
    "ir_structure": (ir_structure_system_prompt, ir_structure_user_prompt),
    "ir_content_relevance": (ir_content_relevance_system_prompt, ir_content_relevance_user_prompt),
    "ir_logic": (ir_logic_system_prompt, ir_logic_user_prompt),
    "ir_expression": (ir_expression_system_prompt, ir_expression_user_prompt),
    # 小故事专用维度
    "sr_structure": (sr_structure_system_prompt, sr_structure_user_prompt),
    "sr_logic": (sr_logic_system_prompt, sr_logic_user_prompt),
    "sr_fluency": (sr_fluency_system_prompt, sr_fluency_user_prompt),
    "sr_event_distribution": (sr_event_distribution_system_prompt, sr_event_distribution_user_prompt),
}

# Pipeline -> list of dimension names (补全缺失维度)
PIPELINE_DIMENSIONS = {
    "basic_evaluation": ["speech_rate", "logic", "expression", "issues", "strengths"],
    "extended_evaluation": ["speech_rate", "content", "logic", "expression", "verbal_habits", "issues", "strengths", "weak_paragraphs"],
    "opinion_statement": ["speech_rate", "op_viewpoint", "op_structure", "op_logic", "op_time_rhythm", "op_expression"],
    "impromptu_reaction": ["speech_rate", "ir_reaction_speed", "ir_structure", "ir_content_relevance", "ir_logic", "ir_expression"],
    "story_reading": ["speech_rate", "sr_structure", "sr_logic", "sr_fluency", "sr_event_distribution"],
    # 绕口令：拆分成4个细粒度维度并行运行，提高速度
    "tongue_twister_reading": ["speech_rate", "tw_completeness", "tw_pronunciation", "tw_fluency", "tw_strengths"],
    "article_reading": ["speech_rate", "ar_completeness", "ar_pronunciation", "ar_fluency", "ar_pause", "ar_strengths"],
}
