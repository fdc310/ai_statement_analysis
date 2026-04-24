"""
Abstract base class and response models for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Unified LLM response format."""
    content: str
    usage: dict = Field(default_factory=dict)  # {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw_response: Optional[dict] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> ChatResponse:
        """Generate chat completion."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming."""
        ...

    @abstractmethod
    async def chat_multimodal(
        self,
        audio_url: str,
        messages: list[dict],
        system_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> ChatResponse:
        """Chat with multimodal model using audio input directly."""
        ...
