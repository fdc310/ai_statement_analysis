# Services module - re-export from tencent submodule
from app.services.tencent import (
    ASRService, asr_service,
    SOEService, soe_service,
)
from app.services.llm_service import LLMService
from app.services.tasks import TaskManager, task_manager, TaskExecutor, task_executor
from app.services.monitoring import TokenTracker, token_tracker
from app.services.agents import BaseAgent, AgentResult, EvaluationContext
from app.core.config import settings


def get_llm_service(provider: str = None, **kwargs) -> LLMService:
    """Get LLM service instance with configured provider."""
    return LLMService(provider=provider or settings.llm_provider, **kwargs)


def get_tts_service(provider: str = None, **kwargs):
    """Get TTS service instance with configured provider."""
    from app.services.tts.tts_service import TTSService
    return TTSService(provider_name=provider, **kwargs)


# Backward-compatible alias: tts_service delegates to the new provider system
tts_service = get_tts_service()

__all__ = [
    "ASRService", "asr_service",
    "SOEService", "soe_service",
    "tts_service",
    "LLMService",
    "get_llm_service",
    "get_tts_service",
    "TaskManager", "task_manager",
    "TaskExecutor", "task_executor",
    "TokenTracker", "token_tracker",
    "BaseAgent", "AgentResult", "EvaluationContext",
]
