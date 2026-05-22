"""
LLM provider registry.
"""
import logging
from typing import Optional

from app.services.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """LLM provider registry."""

    _providers: dict[str, type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLMProvider]):
        """Register a provider class."""
        cls._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseLLMProvider:
        """Get a provider instance by name."""
        provider_class = cls._providers.get(name)
        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown LLM provider: {name}. Available: {available}")
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())


def _register_builtin_providers():
    """Register built-in providers."""
    from app.services.llm.openai_provider import OpenAIProvider
    from app.services.llm.tencent_provider import TencentProvider

    ProviderRegistry.register("openai", OpenAIProvider)
    ProviderRegistry.register("tencent", TencentProvider)

    # Anthropic is optional - only register if installed
    try:
        from app.services.llm.anthropic_provider import AnthropicProvider
        ProviderRegistry.register("anthropic", AnthropicProvider)
    except ImportError:
        logger.info("Anthropic provider not available (anthropic package not installed)")


# Register built-in providers on module import
_register_builtin_providers()
