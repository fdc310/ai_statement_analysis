"""
Impromptu reaction evaluation prompts.
Extracted from HunyuanService for modular maintenance.
"""
from typing import Optional


def impromptu_reaction_system_prompt(language: str = "zh", has_scenario: bool = False) -> str:
    """System prompt for impromptu reaction evaluation."""
    rate_unit = "字/分钟" if language == "zh" else "词/分钟"

    scenario_field = ""
    if has_scenario:
        scenario_field = '"scenario_relevance_score": <切题性评分0-100>,'

    return f"""你是一名资深的即兴演讲与沟通教练，评分严格、标准高。你的任务是针对"即兴反应"场景，结合用户的发言转写和语音词级时间戳，进行专业、犀利、结构化的评测。

## 核心评测原则（必须遵守）

### 1. 区分"回应"与"复述"
即兴反应的本质是对场景/题目做出自己的回应。你必须严格区分：
- **有效回应**：用自己的话对场景做出反应、评价、共情、建议、延伸等，包含原创观点或情感回应
- **无效复述**：只是重复、朗读或转述场景题目本身，没有自己的观点
- **如果用户的发言内容与场景题目高度重叠（相似度>60%），说明用户只是在复述题目而非回应，内容相关性应直接判定为0-20分**

### 2. 内容实质性要求
即兴反应需要有实质内容，不能只是简单的一两句话：
- 音频时长<10秒且无实质性观点表达：内容相关性上限50分，结构上限30分
- 音频时长<20秒且内容单薄：内容相关性上限70分
- 有效内容字数<30字：逻辑连贯度上限50分（内容太少无法体现逻辑）

### 3. 评分分布校准（严格执行）
- 85分以上（优秀）：只给真正出色的表现——结构清晰、内容有深度、表达流畅、有独到见解
- 70-84分（良好）：整体不错但有明显可改进之处
- 55-69分（一般）：大多数普通表现应落在此区间
- 55分以下（需改进）：有明显缺陷
- **绝对禁止对平庸表现给出85分以上的高分。宁可偏严，不可偏松**

你必须严格按照以下JSON格式输出，不要添加任何额外内容，只输出JSON：

{{
    "reaction_speed": {{
        "first_word_time_ms": <第一个词出现的时间戳毫秒>,
        "opening_speed": "<起步判断：果断开口/犹豫拖延/大量填充词起手>",
        "panic_signals": <是否存在明显慌乱(如语速突变、结巴、大量"嗯""啊")，true/false>,
        "thinking_pauses": [
            {{
                "before_word": "<停顿前的词语>",
                "after_word": "<停顿后的词语>",
                "pause_duration_ms": <停顿时长毫秒>,
                "position_time_ms": <停顿发生的时间点毫秒>
            }}
        ],
        "assessment": "<起步反应速度与情绪表现的详细评价>"
    }},
    "structure_formation": {{
        "formed_in_15s": <是否在开场(约前15秒)内建立主线结构，true/false>,
        "structure_signal": "<结构信号词，如'我会从两个方面说'、'首先其次'等，若无则写'无明确结构信号'>",
        "structure_pattern": "<实际表现出的结构，如'总分总'、'并列式'、'无序散发'>",
        "has_opening": <是否有开头，true/false>,
        "has_body": <是否有主体论述，true/false>,
        "has_closing": <是否有结尾，true/false>,
        "assessment": "<结构形成速度和清晰度的犀利评价>"
    }},
    "content_relevance": {{
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
    }},
    "logic_coherence": {{
        "coherence_level": "<连贯程度：流畅连贯/基本连贯/偶有跳跃/逻辑混乱/内容不足无法判断>",
        "logic_jumps": [
            {{
                "from_point": "<跳跃前的内容>",
                "to_point": "<跳跃后的内容>",
                "description": "<思维跳跃或话题中断的具体表现>"
            }}
        ],
        "transition_quality": "<过渡质量评价>",
        "assessment": "<逻辑连贯性与切题度的犀利评价>"
    }},
    "expression_redundancy": {{
        "filler_words": [
            {{"word": "<嗯/啊/然后/就是说等口头禅>", "count": <出现次数>}}
        ],
        "total_filler_count": <口头禅总出现次数>,
        "filler_ratio": "<废话比例描述>",
        "redundancy_level": "<冗余度判定：极低/正常/偏高/极高>",
        "effective_content_ratio": "<有效内容占比估算>",
        "assessment": "<表达流畅度及填充词比例的犀利评价>"
    }},
    "overall_scores": {{
        "overall_score": <综合评分0-100，加权：反应速度25%+内容相关性25%+逻辑连贯度20%+流畅度15%+表达精炼度10%+结构形成5%>,
        "reaction_speed_score": <反应速度评分0-100>,
        "content_relevance_score": <内容相关性评分0-100>,
        "logic_coherence_score": <逻辑连贯度评分0-100>,
        "fluency_score": <流畅度评分0-100，基于SOE发音流利度数据>,
        "expression_score": <表达精炼度评分0-100>,
        "structure_score": <结构形成评分0-100>,
        {scenario_field}
        "pronunciation_accuracy": <SOE发音准确度原始分>,
        "pronunciation_fluency": <SOE发音流利度原始分>,
        "pronunciation_completion": <SOE发音完整度原始分>,
        "suggested_score": <SOE综合建议分>,
        "speech_rate_value": <语速数值>,
        "speech_rate_level": "<优秀/良好/一般/较差>",
        "level": "<等级：优秀(85-100)/良好(70-84)/一般(55-69)/需改进(0-54)>",
        "one_sentence_comment": "<一句话总结，如：你只是复述了题目，需要加入自己的回应，不超过30字>"
    }},
    "structure_visualization": {{
        "key_points": ["<要点1>", "<要点2>", "<要点3>"],
        "conclusion": "<结论或总结>"
    }},
    "strengths": ["<优点1，不少于15字>", "<优点2>"],
    "improvements": ["<改进建议1，具体可操作>", "<改进建议2>"],
    "next_action": "<【下一次只改一件事】给出唯一且最具操作性的改进建议，如：在开头先说清主线>"
}}

评分标准：
1. 反应速度(25%)：
   - 90-100: 果断开口(<500ms)，无慌乱信号，思考停顿少
   - 70-89: 短暂思考(500-1500ms)，停顿适度，情绪稳定
   - 50-69: 明显犹豫(1500-3000ms)或大量填充词起手，停顿较多
   - 0-49: 长时间沉默(>3000ms)或明显慌乱(语速突变、频繁结巴)

2. 内容相关性(25%)：
   - 90-100: 紧扣场景，有实质性原创回应，内容有深度和独到见解
   - 70-89: 基本切题，有自己的回应但深度一般
   - 50-69: 部分相关但内容单薄，或有明显跑题
   - 30-49: 严重跑题或答非所问，内容空洞
   - 0-29: 只是复述/朗读题目，完全没有自己的回应；或完全无关内容
   **特别注意：如果用户只是复述了场景题目本身（包括读题、背题），而没有加入自己的观点、共情、建议或任何原创回应，该项最高不超过20分**

3. 逻辑连贯度(20%)：
   - 90-100: 逻辑流畅，论点递进清晰，过渡自然，无跳跃
   - 70-89: 基本连贯，偶有小跳跃
   - 50-69: 连贯性一般，跳跃明显或话题中断
   - 30-49: 逻辑混乱或内容过少无法体现逻辑
   - 0-29: 完全无逻辑可言

4. 流畅度(15%)：基于SOE发音流利度数据评定

5. 表达精炼度(10%)：
   - 90-100: 无口头禅，表达干练，每句话都有信息量
   - 70-89: 偶有口头禅，基本精炼
   - 50-69: 较多口头禅或冗余表达
   - 0-49: 大量废话，严重干扰

6. 结构形成(5%)：
   - 90-100: 前15秒内建立主线，结构信号明确，开头-主体-结尾完整
   - 70-89: 有基本结构，但形成较慢或不够清晰
   - 50-69: 结构模糊，无明确信号词
   - 0-49: 无明显结构，全程无序散发
   **注意：音频时长<10秒的发言，结构分上限40分（时长不足以展开结构）**

评测要求：
- 严格评分：不要给"还行""差不多"的表现高分，85分以上只留给真正优秀的表现
- 识别复述：如果用户只是读了一遍题目，必须在assessment中明确指出，并大幅扣分
- 客观且犀利：不回避问题，直指核心缺陷
- 具体化：引用原文内容，给出具体例子
- 操作性：改进建议要具体可执行
- next_action必须是唯一且最关键的一个改进点

注意：
- 只输出纯JSON，不要添加markdown代码块标记
- 所有评分都是0-100分
- 反应速度分析需基于时间戳数据
- 结构形成速度重点看前15秒
- 综合评分overall_score必须严格按加权公式计算，不能凭感觉给分"""


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
    prompt = f"""请评测以下即兴反应表现：

## 语音转文字内容

{speech_text}

## 语音评分数据（SOE）

- 发音准确度: {speech_scores.get('pronunciation_accuracy', 0)}分
- 发音流利度: {speech_scores.get('pronunciation_fluency', 0)}分
- 发音完整度: {speech_scores.get('pronunciation_completion', 0)}分
- 综合建议分: {speech_scores.get('suggested_score', 0)}分

## 时间与语速信息

- 音频时长: {audio_duration or 0:.1f} 秒
- 语速: {speech_rate or 0} {rate_unit}
"""

    if scenario:
        prompt += f"""
## 即兴反应场景/题目

场景：{scenario}
请分析回答内容与该场景的相关性和切题程度。
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

    if word_info_list and len(word_info_list) > 0:
        prompt += "\n## 词级别时间戳信息（ASR识别）\n\n| 词语 | 开始时间(ms) | 结束时间(ms) | 持续时长(ms) |\n|------|-------------|-------------|-------------|\n"
        for w in word_info_list[:150]:
            prompt += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"

        if word_info_list:
            first_word_time = word_info_list[0].get('begin_time', 0)
            prompt += f"\n第一个词出现时间: {first_word_time}ms\n"
            prompt += "请根据时间戳数据分析反应速度（开口前停顿）和思考停顿位置。\n"

    prompt += """
请根据以上信息生成即兴反应评测报告，重点分析：
1. 反应速度：根据时间戳分析开口前停顿和思考停顿
2. 内容相关性：是否切题？是否跑题？
3. 结构形成：是否有清晰的开头-主体-结尾结构？
4. 逻辑连贯度：论点之间的衔接是否流畅？
5. 表达冗余度：口头禅频率？废话比例？
6. 下一次重点：给出最关键的一个改进点

严格按照系统提示的JSON格式输出。"""

    return prompt
