# Services module - re-export from tencent submodule
from app.services.tencent import (
    ASRService, asr_service,
    SOEService, soe_service,
    HunyuanService, hunyuan_service
)
from app.services.llm_service import LLMService, LLMProvider
from app.core.config import settings


def get_llm_service() -> LLMService:
    """获取配置的 LLM 服务"""
    provider = getattr(settings, 'llm_provider', 'hunyuan')
    
    if provider == "openai":
        return LLMService(
            provider="openai",
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
    else:
        return LLMService(
            provider="hunyuan",
            model=settings.hunyuan_model
        )


__all__ = [
    "ASRService", "asr_service",
    "SOEService", "soe_service",
    "HunyuanService", "hunyuan_service",
    "LLMService", "LLMProvider",
    "get_llm_service"
]
