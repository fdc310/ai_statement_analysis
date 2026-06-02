"""
Prompt templates for scenario dialogue report generation.
Supports: interview, office_work, business_social, custom, daily, customer_service.
"""
from typing import Optional


# ── Scene-specific evaluation dimensions ──────────────────────────────────

SCENE_EVAL_DIMENSIONS = {
    "interview": {
        "name": "求职面试",
        "sub_types": ["应届求职", "社会招聘", "考公考编"],
        "dimensions": [
            "对话亮点：发言条理、主次顺序、是否分点/抓重点、有无逻辑混乱、答非所问，向面试官提问的水平，是否体现思考而非无效问题",
            "岗位匹配度：个人经历、能力、自我介绍是否贴合岗位要求，有无突出核心优势",
            "问题应答质量：专业问题、情景题、压力面、优缺点/离职原因等高频题回答质量，避坑能力",
            "礼仪与职业形象：开场问候、离场致谢、称呼使用，公考/体制类额外考核规矩感、稳重感",
            "改进建议：针对问题给出可落地的话术/行为改进方案",
        ],
    },
    "office_work": {
        "name": "职场办公",
        "sub_types": ["工作汇报", "升职加薪", "离职跳槽"],
        "dimensions": [
            "对话亮点：发言条理、主次顺序、是否分点/抓重点、有无逻辑混乱、答非所问",
            "内容价值（汇报重点）：数据/成果/问题/计划是否清晰，是否抓领导关注点，汇报详略是否得当",
            "诉求表达（加薪/升职）：理由是否充分（业绩、价值、贡献），语气是否不卑不亢，诉求是否合理、表述委婉有度",
            "职场情商与分寸：说话尺度、立场站位，是否抱怨、情绪化，面对拒绝/质疑的沟通姿态",
            "向上沟通适配度：话术是否适配上下级关系，是否懂得换位思考领导视角",
            "改进建议：针对问题给出可落地的话术/行为改进方案",
        ],
    },
    "business_social": {
        "name": "商务社交",
        "sub_types": ["销售沟通", "商务洽谈", "商务社交"],
        "dimensions": [
            "对话亮点：发言条理、主次顺序、是否分点/抓重点、有无逻辑混乱、答非所问",
            "需求挖掘能力：能否主动询问、捕捉对方真实需求，不单向自说自话",
            "价值传递：产品/方案/合作优势讲解是否清晰，能否给到对方利益点",
            "谈判与博弈：价格、条件、合作条款沟通，让步节奏、底线把控、议价能力",
            "商务礼仪：称呼、寒暄、开场收尾、场合适配话术，正式场合是否专业得体",
            "关系维护意识：闲聊破冰、氛围营造、后续邀约/跟进意识，社交亲和力",
            "改进建议：针对问题给出可落地的话术/行为改进方案",
        ],
    },
    "custom": {
        "name": "自定义",
        "sub_types": [],
        "dimensions": [
            "维度评分（1-10分）：说服力、共情力、应变力",
            "对话亮点：发言条理、主次顺序、是否分点/抓重点、有无逻辑混乱、答非所问",
            "改进建议：针对问题给出可落地的话术/行为改进方案",
        ],
    },
    "daily": {
        "name": "日常对话",
        "sub_types": [],
        "dimensions": [
            "对话亮点：发言条理、表达是否自然流畅、有互动感",
            "沟通能力：是否有效回应话题、有无跑题、表达是否清晰",
            "改进建议：针对问题给出可落地的话术/行为改进方案",
        ],
    },
    "customer_service": {
        "name": "客服场景",
        "sub_types": [],
        "dimensions": [
            "对话亮点：发言条理、是否有效回应需求、表达是否清晰",
            "服务态度：是否礼貌、有耐心，问题解决是否到位",
            "改进建议：针对问题给出可落地的话术/行为改进方案",
        ],
    },
}

# Default dimensions for unknown scenes
_DEFAULT_DIMENSIONS = {
    "name": "通用对话",
    "sub_types": [],
    "dimensions": [
        "对话亮点：发言条理、表达是否清晰、有无逻辑混乱",
        "沟通能力：是否切题、表达是否有效",
        "改进建议：针对问题给出可落地的话术/行为改进方案",
    ],
}


def _get_scene_config(scene: str) -> dict:
    """Get scene configuration, falling back to default."""
    return SCENE_EVAL_DIMENSIONS.get(scene, _DEFAULT_DIMENSIONS)


# ── Summary prompt (20-30 chars) ──────────────────────────────────────────

def scenario_summary_system_prompt() -> str:
    """System prompt for generating a short 20-30 character summary."""
    return """你是一位对话评估专家。你的任务是对一段情景对话进行简短概括。

要求：
- 用一句中文概括用户在对话中的整体表现
- 严格控制在20-30个字以内
- 突出核心优点或主要问题
- 言简意赅，直击要点

你必须严格返回以下JSON格式，不要返回任何其他内容：
{"summary": "<20-30字概括>"}"""


def scenario_summary_user_prompt(
    scene: str,
    messages: list[dict],
    blood_history: list[dict],
    final_hp: int,
    initial_hp: int,
) -> str:
    """Build user prompt for short summary generation."""
    scene_config = _get_scene_config(scene)
    scene_name = scene_config["name"]

    # Build dialogue history
    dialogue_lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "AI"
        dialogue_lines.append(f"{role}：{m.get('content', '')}")
    dialogue_text = "\n".join(dialogue_lines)

    # Build blood history summary
    blood_lines = []
    for h in blood_history:
        blood_lines.append(f"血量变化 {h.get('delta', +0)}（{h.get('reason', '')}），当前 {h.get('hp', 0)}/100")
    blood_text = "\n".join(blood_lines) if blood_lines else "未启用血量机制"

    return f"""情景类型：{scene_name}
初始血量：{initial_hp}/100，最终血量：{final_hp}/100

【血量变化记录】
{blood_text}

【对话记录】
{dialogue_text}

请根据以上对话内容，用20-30字概括用户在本次{scene_name}中的整体表现。"""


# ── Full report prompt ────────────────────────────────────────────────────

def scenario_report_system_prompt(scene: str) -> str:
    """System prompt for generating the full evaluation report."""
    scene_config = _get_scene_config(scene)
    scene_name = scene_config["name"]
    dimensions = scene_config["dimensions"]
    sub_types = scene_config.get("sub_types", [])

    dimensions_text = "\n".join(f"- {d}" for d in dimensions)
    sub_types_text = f"子类型：{'、'.join(sub_types)}" if sub_types else ""

    return f"""你是一位专业的对话评估专家。你的任务是对一段{scene_name}情景对话进行全面评估分析。
{sub_types_text}

评估维度：
{dimensions_text}

分析要求：
1. 分析要言简意赅，直击问题
2. 对话亮点要具体引用用户原话
3. 改进建议要可落地、可操作
4. 更优话术示例要贴合{scene_name}场景
5. 如果有血量变化记录，结合血量变化分析用户表现的起伏

你必须严格返回以下JSON格式，不要返回任何其他内容：
{{
  "scene": "{scene_name}",
  "overall_score": <1-10分整数>,
  "summary": "<50字以内的整体评价>",
  "dimensions": [
    {{"name": "<维度名称>", "score": <1-10分>, "comment": "<该维度的具体评价>"}}
  ],
  "highlights": ["<亮点1>", "<亮点2>"],
  "improvements": ["<改进建议1>", "<改进建议2>"],
  "better_examples": ["<更优话术示例1>", "<更优话术示例2>"]
}}"""


def scenario_report_user_prompt(
    scene: str,
    messages: list[dict],
    blood_history: list[dict],
    final_hp: int,
    initial_hp: int,
) -> str:
    """Build user prompt for full report generation."""
    scene_config = _get_scene_config(scene)
    scene_name = scene_config["name"]

    # Build dialogue history
    dialogue_lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "AI"
        dialogue_lines.append(f"{role}：{m.get('content', '')}")
    dialogue_text = "\n".join(dialogue_lines)

    # Build blood history
    blood_lines = []
    for i, h in enumerate(blood_history, 1):
        blood_lines.append(
            f"第{i}次：{h.get('delta', +0):+d}（{h.get('reason', '')}）→ 血量 {h.get('hp', 0)}/100"
        )
    blood_text = "\n".join(blood_lines) if blood_lines else "未启用血量机制"

    return f"""情景类型：{scene_name}
初始血量：{initial_hp}/100，最终血量：{final_hp}/100

【血量变化记录】
{blood_text}

【完整对话记录】
{dialogue_text}

请根据以上对话内容和血量变化记录，对用户在本次{scene_name}中的表现进行全面评估。"""
