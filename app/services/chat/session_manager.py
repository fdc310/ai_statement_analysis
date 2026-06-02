"""
Chat session manager for voice chat conversations.
Manages server-side conversation history and scene switching.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

# Maximum number of messages to keep in session history
_MAX_SESSION_MESSAGES = 50

# Scene presets — 场景 + 子类型 → system prompt
VOICE_CHAT_SCENE_PROMPTS = {
    # ── 求职面试 ──
    "interview": """你是一位经验丰富的面试官。你的任务是模拟面试场景，对用户进行面试评估。
请根据用户的回答进行追问，评价其表达能力、逻辑思维和专业水平。
回复要简洁专业，每次追问一个方向。""",
    "interview:campus": """你是一位校园招聘面试官，正在面试一位应届毕业生。
请围绕校园经历、实习经验、专业能力、职业规划等方面提问。
关注点：学生是否有清晰的职业方向，实习/项目经历是否扎实，表达是否自信有条理。
回复要简洁专业，每次追问一个方向。""",
    "interview:social": """你是一位社会招聘面试官，正在面试一位有工作经验的求职者。
请围绕过往工作经历、项目成果、离职原因、岗位匹配度等方面提问。
关注点：候选人是否有核心竞争力，经验是否匹配岗位，离职原因是否合理。
回复要简洁专业，每次追问一个方向。""",
    "interview:civil": """你是一位公务员/事业单位面试考官，正在进行结构化面试。
请围绕综合分析能力、组织协调能力、应急处理能力、人际关系处理等方面提问。
关注点：考生是否具备公职人员的思维格局，回答是否稳重、有条理、符合体制内表达规范。
回复要严谨规范，像真实的考公面试一样。""",
    # ── 职场办公 ──
    "office_work": """你是一位资深的职场导师/领导。你的任务是模拟职场办公场景，与用户进行工作汇报、升职加薪或离职跳槽相关的对话。
请根据用户的回答进行追问或回应，评价其职场沟通能力、情商与分寸感。
回复要专业务实，像真实的上下级对话一样。""",
    "office_work:report": """你是一位部门领导，下属正在向你进行工作汇报。
请围绕汇报内容进行追问或反馈，关注：数据是否清晰、成果是否有亮点、问题是否有解决方案、计划是否可行。
关注点：汇报是否抓重点、逻辑是否清晰、是否站在领导视角思考问题。
回复要像真实的领导回应，简洁有力，偶尔提出质疑。""",
    "office_work:promotion": """你是一位部门领导，下属正在向你提出升职加薪的诉求。
请根据其陈述进行回应，可以追问业绩数据、表达认可或提出疑虑。
关注点：理由是否充分、语气是否不卑不亢、诉求是否合理、是否懂得换位思考。
回复要像真实的领导回应，既不轻易答应也不直接打压。""",
    "office_work:resignation": """你是一位部门领导，下属正在向你提出离职。
请根据其陈述进行回应，可以挽留、追问原因或讨论交接事宜。
关注点：离职原因表述是否得体、有无吐槽原公司、交接态度是否专业。
回复要像真实的领导回应，体现格局和关怀。""",
    # ── 商务社交 ──
    "business_social": """你是一位资深的商务人士。你的任务是模拟商务社交场景，与用户进行销售沟通、商务洽谈或社交对话。
请根据用户的回答进行互动，评价其商务礼仪、需求挖掘和谈判能力。
回复要专业得体，像真实的商务场合对话一样。""",
    "business_social:sales": """你是一位潜在客户，对方是一位销售人员正在向你推销产品/服务。
请根据对方的介绍进行回应，可以提出需求、表示疑虑或询问细节。
关注点：对方是否主动挖掘你的需求、产品介绍是否清晰、能否给出利益点、处理异议是否得当。
回复要像真实的客户，时而感兴趣时而犹豫。""",
    "business_social:deal": """你是一位合作方负责人，对方正在与你进行商务洽谈/合作谈判。
请根据对方的提案进行回应，讨论合作条件、价格、条款等。
关注点：对方的谈判技巧、让步节奏、底线把控、方案是否对双方有利。
回复要像真实的商务谈判，精明但不失礼。""",
    "business_social:networking": """你是一位行业同行/潜在合作伙伴，对方正在与你进行商务社交/破冰交流。
请根据对方的话题进行互动，聊行业、资源、合作机会等。
关注点：对方的社交亲和力、破冰能力、关系维护意识、后续跟进意识。
回复要像真实的社交场合，友好但有距离感。""",
    # ── 自定义 ──
    "custom": """你是一位友善且专业的对话伙伴。请和用户进行自然流畅的对话，根据用户设定的主题展开交流。
回复要有深度、有逻辑，像一位专业的沟通顾问一样。""",
    # ── 兼容旧场景 ──
    "daily": """你是一位友善的日常对话伙伴。你的任务是和用户进行轻松自然的日常对话。
使用口语化的表达，像朋友一样聊天，可以聊生活、工作、兴趣爱好等话题。
回复要自然亲切，像真人对话一样。""",
    "customer_service": """你是一位专业的客服人员。你的任务是处理用户的咨询和问题。
请耐心倾听用户的需求，提供准确的解答和建议。
回复要礼貌专业，体现服务意识。""",
}

DEFAULT_VOICE_CHAT_PROMPT = """你是一位友善的对话伙伴。请和用户进行自然流畅的对话。
回复要简洁自然，像真人对话一样。"""


class ChatSession(BaseModel):
    """Server-side chat session state."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scene: str = ""
    system_prompt: str = ""
    mode: str = "traditional"  # "traditional" | "multimodal"
    voice_type: int = 101001
    messages: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    # Blood bar state
    hp: int = 100
    enable_blood_bar: bool = False
    blood_history: list[dict] = Field(default_factory=list)
    # End state
    is_ended: bool = False
    report_data: Optional[dict] = None


class ChatSessionManager:
    """
    In-memory chat session manager.

    Manages server-side conversation history and scene switching.
    Sessions expire after chat_session_ttl seconds of inactivity.
    """

    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    def _resolve_scene_prompt(self, scene: Optional[str], custom_prompt: Optional[str]) -> str:
        """Resolve system prompt from scene or custom prompt."""
        if custom_prompt:
            return custom_prompt
        if scene and scene in VOICE_CHAT_SCENE_PROMPTS:
            return VOICE_CHAT_SCENE_PROMPTS[scene]
        return DEFAULT_VOICE_CHAT_PROMPT

    async def create_session(
        self,
        scene: Optional[str] = None,
        system_prompt: Optional[str] = None,
        mode: str = "traditional",
        voice_type: int = 101001,
        enable_blood_bar: bool = False,
        initial_hp: int = 100,
    ) -> ChatSession:
        """Create a new chat session."""
        async with self._lock:
            resolved_prompt = self._resolve_scene_prompt(scene, system_prompt)
            session = ChatSession(
                scene=scene or "",
                system_prompt=resolved_prompt,
                mode=mode,
                voice_type=voice_type,
                enable_blood_bar=enable_blood_bar,
                hp=initial_hp,
            )
            self._sessions[session.session_id] = session
            logger.info(f"Created chat session {session.session_id}, scene={scene}, mode={mode}")
            return session

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID. Returns None if not found or expired."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            # Check expiration
            ttl = settings.chat_session_ttl
            if (datetime.now() - session.last_active).total_seconds() > ttl:
                del self._sessions[session_id]
                logger.info(f"Session {session_id} expired")
                return None

            session.last_active = datetime.now()
            return session

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        scene: Optional[str] = None,
        system_prompt: Optional[str] = None,
        mode: str = "traditional",
        voice_type: int = 101001,
        enable_blood_bar: bool = False,
        initial_hp: int = 100,
    ) -> ChatSession:
        """Get existing session or create new one."""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session

        return await self.create_session(
            scene, system_prompt, mode, voice_type,
            enable_blood_bar, initial_hp,
        )

    async def update_scene(
        self,
        session_id: str,
        scene: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[ChatSession]:
        """Update session scene (session-level switch)."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if scene is not None:
                session.scene = scene
            if system_prompt is not None:
                session.system_prompt = system_prompt
            elif scene is not None:
                session.system_prompt = self._resolve_scene_prompt(scene, None)

            session.last_active = datetime.now()
            logger.info(f"Session {session_id} scene updated to {scene}")
            return session

    async def update_hp(
        self,
        session_id: str,
        delta: int,
        reason: str = "",
    ) -> Optional[dict]:
        """Update session HP and record blood bar change. Returns blood bar state dict."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.hp = max(0, session.hp + delta)
            record = {
                "delta": delta,
                "hp": session.hp,
                "reason": reason,
                "game_over": session.hp <= 0,
            }
            session.blood_history.append(record)
            session.last_active = datetime.now()
            return record

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to session history, trimming oldest if over limit."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.append({"role": role, "content": content})
                # Trim oldest messages if over limit (keep most recent)
                if len(session.messages) > _MAX_SESSION_MESSAGES:
                    session.messages = session.messages[-_MAX_SESSION_MESSAGES:]
                session.last_active = datetime.now()

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        async with self._lock:
            self._sessions.pop(session_id, None)
            logger.info(f"Deleted chat session {session_id}")

    async def end_session(self, session_id: str, report_data: dict) -> None:
        """Mark a session as ended and store report data."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_ended = True
                session.report_data = report_data
                session.last_active = datetime.now()
                logger.info(f"Session {session_id} ended with report")

    async def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        async with self._lock:
            ttl = settings.chat_session_ttl
            now = datetime.now()
            expired = [
                sid for sid, s in self._sessions.items()
                if (now - s.last_active).total_seconds() > ttl
            ]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired chat sessions")
            return len(expired)


# Singleton
chat_session_manager = ChatSessionManager()
