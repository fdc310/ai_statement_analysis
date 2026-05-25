"""
Prompts for tongue-twister and article-reading analysis.
"""
from typing import Optional


def tongue_twister_system_prompt() -> str:
    """System prompt for pure tongue-twister text analysis."""
    return """你是一个专业的语音学和发音教学专家。你的任务是分析绕口令的发音要点，帮助用户更好地练习发音。

你必须严格按照以下 JSON 格式输出，不要添加任何额外内容，只输出 JSON：
{
    "title": "绕口令标题/主题",
    "difficulty": "难度等级（简单/中等/困难/专家）",
    "core_phonemes": [
        {
            "phoneme": "音素（如：b、p、zh、ch）",
            "pinyin": "对应拼音",
            "ipa": "国际音标",
            "description": "发音描述",
            "articulation": {
                "manner": "发音方式",
                "place": "发音部位",
                "voicing": "清音/浊音"
            },
            "examples": ["包含该音素的字词示例"]
        }
    ],
    "acoustic_features": [
        {
            "feature": "声学特征名称",
            "description": "特征描述",
            "key_difference": "与相似音的关键差异",
            "measurement": "可量化指标（如 VOT、F1/F2）"
        }
    ],
    "confusion_pairs": [
        {
            "pair": ["音素1", "音素2"],
            "difference": "区分要点",
            "common_errors": "常见错误",
            "practice_tip": "练习建议"
        }
    ],
    "pronunciation_tips": [
        {
            "tip": "发音提示",
            "target_sounds": ["目标音素"],
            "technique": "具体技巧",
            "practice_method": "练习方法"
        }
    ],
    "rhythm_pattern": {
        "beat_count": "节拍数",
        "stress_pattern": "重音模式",
        "pause_points": ["建议停顿位置"],
        "speed_suggestion": "建议语速"
    },
    "practice_sequence": [
        {
            "step": 1,
            "focus": "练习重点",
            "content": "练习内容",
            "repetitions": "建议重复次数"
        }
    ],
    "annotated_text": "带标注的绕口令文本（用 [] 标注核心音素）"
}

要求：
1. 准确识别绕口令中的核心音素和难点
2. 解释容易混淆音之间的关键区别
3. 给出可执行的发音技巧和练习步骤
4. 分析节奏和停顿建议
5. 只输出纯 JSON，不要输出 markdown 代码块"""


def tongue_twister_user_prompt(
    speech_text: str,
    reference_text: Optional[str] = None,
    speech_scores: Optional[dict] = None,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    language: str = "zh"
) -> str:
    """User prompt for pure tongue-twister text analysis."""
    return f"""请分析以下绕口令的发音要点：

## 绕口令内容
{speech_text}

请严格按照 JSON 格式输出分析结果，重点分析核心音素、混淆音对、发音技巧、节奏模式和练习顺序。"""


def tongue_twister_reading_system_prompt() -> str:
    """System prompt for tongue-twister reading evaluation."""
    return """你是一个专业的绕口令语音评测专家。你的任务是分析用户朗读绕口令的语音表现，通过对比原始绕口令文本和实际朗读内容，评估优势和待改进之处。

你必须严格按照以下 JSON 格式输出，不要添加任何额外内容，只输出 JSON：
{
    "strengths": [
        "优点1",
        "优点2"
    ],
    "improvements": {
        "extra_words": {
            "count": 0,
            "words": [],
            "description": "多读情况描述"
        },
        "missed_words": {
            "count": 0,
            "words": [],
            "description": "漏读情况描述"
        },
        "pronunciation_issues": [
            {
                "word": "存在问题的字词",
                "accuracy_score": 0,
                "issue_description": "具体发音问题",
                "correct_pronunciation": "正确发音说明",
                "practice_tip": "练习建议"
            }
        ]
    },
    "fluency_analysis": {
        "overall_fluency": "优秀/良好/一般/需改进",
        "long_pauses": [
            {
                "before_word": "停顿前的词",
                "after_word": "停顿后的词",
                "pause_duration_ms": 0,
                "position_time_ms": 0
            }
        ],
        "rhythm_assessment": "节奏评价",
        "speed_assessment": "语速评价"
    },
    "overall_assessment": "综合评价",
    "practice_suggestions": [
        "练习建议1",
        "练习建议2",
        "练习建议3"
    ]
}

评测规则：
1. 多读判断：将实际朗读文本与绕口令原文逐字对比，找出朗读中有但原文中没有的字词
2. 漏读判断：将实际朗读文本与绕口令原文逐字对比，找出原文中有但朗读中遗漏的字词
3. 发音问题：基于 SOE 低分字词（accuracy < 90 分）分析具体问题
4. 长停顿：绕口令的停顿标准比普通阅读更严格，使用 2000ms 作为长停顿阈值
5. 节奏感：绕口令需要有节奏感，不能过于平淡，也不能一味抢快

注意：
- 只输出纯 JSON，不要输出 markdown 代码块
- 不要凭空捏造不存在的错误
- 漏读/多读必须以原文与朗读文本对比为准"""


def tongue_twister_reading_user_prompt(
    speech_text: str,
    reference_text: str,
    speech_scores: Optional[dict] = None,
    statistics: Optional[dict] = None,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    language: str = "zh"
) -> str:
    """User prompt for tongue-twister reading evaluation."""
    prompt = f"""请分析以下用户朗读绕口令的表现：

## 绕口令原文
{reference_text}

## 用户实际朗读内容（ASR 识别结果）
{speech_text}
"""

    if speech_scores:
        prompt += f"""
## SOE 语音评测评分
- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分
"""

    if statistics:
        prompt += f"""
## 评分统计
- 总字数: {statistics.get('total_words', 0)}
- 平均准确度: {statistics.get('average_accuracy', 0):.1f}分
- 低分字数: {statistics.get('low_score_count', 0)}个
"""

    if low_score_words:
        prompt += """
## 低分字词列表
"""
        for word in low_score_words[:20]:
            prompt += f"\n- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分, 流利度{word.get('fluency', 0)}分"
        prompt += "\n"

    if word_info_list:
        prompt += """
## 词级时间戳信息（ASR）

| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |
|------|-------------|-------------|-------------|
"""
        for w in word_info_list[:150]:
            prompt += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

    prompt += """
请严格按照系统提示中的 JSON 结构输出，重点分析：
1. 优点：指出朗读中表现好的地方
2. 待提升：多读、漏读、以及具体发音问题
3. 流畅度：长停顿、节奏感、语速是否适合绕口令
4. 综合评价：概括整体表现
5. 练习建议：给出具体可执行的改进方法"""

    return prompt


def article_reading_system_prompt() -> str:
    """System prompt for article-reading evaluation."""
    return """你是一个专业的文章朗读评测专家。你的任务是分析用户朗读文章的语音表现，从流畅度、语速、断句停顿、读错漏字等多个维度进行评估。

你必须严格按照以下 JSON 格式输出，不要添加任何额外内容，只输出 JSON：
{
    "strengths": ["优点1", "优点2"],
    "improvements": {
        "extra_words": {
            "count": 0,
            "words": [],
            "description": "多读情况描述"
        },
        "missed_words": {
            "count": 0,
            "words": [],
            "description": "漏读情况描述"
        },
        "wrong_words": [
            {
                "expected": "原文",
                "actual": "误读",
                "position": "位置说明",
                "description": "错误描述"
            }
        ],
        "pronunciation_issues": [
            {
                "word": "存在问题的字词",
                "accuracy_score": 0,
                "issue_description": "具体发音问题",
                "correct_pronunciation": "正确发音说明",
                "practice_tip": "练习建议"
            }
        ]
    },
    "fluency_analysis": {
        "score": 0,
        "overall_fluency": "优秀/良好/一般/需改进",
        "interruptions": [],
        "repeated_reads": [],
        "stutters": []
    },
    "speech_rate_analysis": {
        "overall_rate": 0,
        "rate_level": "偏快/适中/偏慢",
        "standard_range": "180-240字/分钟",
        "segment_rates": [],
        "fast_segments": [],
        "slow_segments": [],
        "suggestion": "语速建议"
    },
    "pause_analysis": {
        "proper_pauses": 0,
        "improper_pauses": [],
        "missed_pauses": [],
        "overall_assessment": "断句停顿评价"
    },
    "overall_assessment": "综合评价",
    "practice_suggestions": ["练习建议1", "练习建议2", "练习建议3"]
}

注意：
- 只输出纯 JSON，不要输出 markdown 代码块
- 断句、停顿和错词判断必须结合原文"""


def article_reading_user_prompt(
    speech_text: str,
    reference_text: Optional[str] = None,
    speech_scores: Optional[dict] = None,
    word_info_list: Optional[list] = None,
    low_score_words: Optional[list] = None,
    statistics: Optional[dict] = None,
    speech_rate: Optional[float] = None,
    audio_duration: Optional[float] = None,
    language: str = "zh"
) -> str:
    """User prompt for article-reading evaluation."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"
    prompt = f"""请分析以下用户朗读文章的表现：

## 文章原文
{reference_text or ""}

## 用户实际朗读内容（ASR 识别结果）
{speech_text}
"""

    if speech_scores:
        prompt += f"""
## SOE 语音评测评分
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

    if low_score_words:
        prompt += """
## 低分字词列表
"""
        for word in low_score_words[:20]:
            prompt += f"\n- {word.get('word', '')}: 准确度{word.get('accuracy', 0)}分, 流利度{word.get('fluency', 0)}分"
        prompt += "\n"

    if word_info_list:
        prompt += """
## 词级时间戳信息（ASR）

| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |
|------|-------------|-------------|-------------|
"""
        for w in word_info_list[:150]:
            prompt += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

    prompt += """
请严格按照系统提示中的 JSON 结构输出，重点分析：
1. 优点：指出朗读中表现好的地方
2. 待提升：多读、漏读、读错字以及具体发音问题
3. 流畅度：中断、重复读、卡壳等情况
4. 语速：整体和分段语速是否合适
5. 停顿：断句和停顿是否自然
6. 练习建议：给出具体可执行的改进方法"""

    return prompt
