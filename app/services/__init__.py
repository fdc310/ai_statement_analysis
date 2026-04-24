# Services module - re-export from tencent submodule
from app.services.tencent import (
    ASRService, asr_service,
    SOEService, soe_service,
    TTSService, tts_service
)
from app.services.llm_service import LLMService
from app.services.tasks import TaskManager, task_manager, TaskExecutor, task_executor
from app.services.monitoring import TokenTracker, token_tracker
from app.services.agents import BaseAgent, AgentResult, EvaluationContext
from app.core.config import settings


def get_llm_service(provider: str = None, **kwargs) -> LLMService:
    """Get LLM service instance with configured provider."""
    return LLMService(provider=provider or settings.llm_provider, **kwargs)


__all__ = [
    "ASRService", "asr_service",
    "SOEService", "soe_service",
    "TTSService", "tts_service",
    "LLMService",
    "get_llm_service",
    "TaskManager", "task_manager",
    "TaskExecutor", "task_executor",
    "TokenTracker", "token_tracker",
    "BaseAgent", "AgentResult", "EvaluationContext",
]
