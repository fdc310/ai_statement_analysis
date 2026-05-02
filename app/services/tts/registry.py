"""
TTS provider registry.
"""
import logging
from typing import Optional

from app.services.tts.base import BaseTTSProvider

logger = logging.getLogger(__name__)


class TTSProviderRegistry:
    """TTS provider registry."""

    _providers: dict[str, type[BaseTTSProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseTTSProvider]):
        """Register a provider class."""
        cls._providers[name] = provider_class
        logger.info(f"Registered TTS provider: {name}")

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseTTSProvider:
        """Get a provider instance by name."""
        provider_class = cls._providers.get(name)
        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown TTS provider: {name}. Available: {available}")
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())


def _register_builtin_providers():
    """Register built-in providers."""
    from app.services.tts.providers.tencent_provider import TencentTTSProvider
    TTSProviderRegistry.register("tencent", TencentTTSProvider)

    try:
        from app.services.tts.providers.volcengine_provider import VolcengineTTSProvider
        TTSProviderRegistry.register("volcengine", VolcengineTTSProvider)
    except ImportError:
        logger.info("Volcengine TTS provider not available (missing dependencies)")

    try:
        from app.services.tts.providers.xiaomi_provider import XiaomiTTSProvider
        TTSProviderRegistry.register("xiaomi", XiaomiTTSProvider)
    except ImportError:
        logger.info("Xiaomi TTS provider not available (missing dependencies)")

    try:
        from app.services.tts.providers.minimax_provider import MinimaxTTSProvider
        TTSProviderRegistry.register("minimax", MinimaxTTSProvider)
    except ImportError:
        logger.info("Minimax TTS provider not available (missing dependencies)")


# Register built-in providers on module import
_register_builtin_providers()
