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

# Scene presets
VOICE_CHAT_SCENE_PROMPTS = {
    "interview": """你是一位经验丰富的面试官。你的任务是模拟面试场景，对用户进行面试评估。
请根据用户的回答进行追问，评价其表达能力、逻辑思维和专业水平。
回复要简洁专业，每次追问一个方向。""",
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
    ) -> ChatSession:
        """Create a new chat session."""
        async with self._lock:
            resolved_prompt = self._resolve_scene_prompt(scene, system_prompt)
            session = ChatSession(
                scene=scene or "",
                system_prompt=resolved_prompt,
                mode=mode,
                voice_type=voice_type,
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
    ) -> ChatSession:
        """Get existing session or create new one."""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session

        return await self.create_session(scene, system_prompt, mode, voice_type)

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
