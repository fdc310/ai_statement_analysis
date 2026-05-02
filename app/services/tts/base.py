"""
Abstract base class and response models for TTS providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator
from pydantic import BaseModel, Field


class TTSResponse(BaseModel):
    """Unified TTS response format."""
    audio_data: bytes
    audio_chunks: list[bytes] = Field(default_factory=list)
    content_type: str = "audio/mpeg"
    usage: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class BaseTTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> TTSResponse:
        """Synthesize text to speech and return complete audio."""
        ...

    async def synthesize_stream(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to speech and yield audio chunks.

        Default implementation: calls synthesize() then yields chunks.
        Providers with native streaming should override this.
        """
        result = await self.synthesize(text, voice_type, codec, sample_rate, speed, volume, **kwargs)
        for chunk in result.audio_chunks:
            yield chunk

    def create_stream_session(
        self,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ):
        """Create a persistent stream session for ws_chat.

        Returns a BaseTTSStreamSession. Providers that don't support
        persistent connections will raise NotImplementedError.
        """
        from app.services.tts.stream_session import BaseTTSStreamSession
        raise NotImplementedError(f"{self.name} does not support persistent stream sessions")
