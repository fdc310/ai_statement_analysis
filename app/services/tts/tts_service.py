"""
TTS service facade — delegates to the configured provider.
"""
import logging
from typing import Optional, AsyncGenerator

from app.core.config import settings
from app.services.tts.base import BaseTTSProvider, TTSResponse
from app.services.tts.registry import TTSProviderRegistry
from app.services.tts.stream_session import BaseTTSStreamSession

logger = logging.getLogger(__name__)


class TTSService:
    """TTS service facade that delegates to a registered provider."""

    def __init__(self, provider_name: Optional[str] = None):
        self._provider: BaseTTSProvider = TTSProviderRegistry.get_provider(
            provider_name or settings.tts_provider
        )
        logger.info(f"TTS service initialized with provider: {self._provider.name}")

    @property
    def provider(self) -> BaseTTSProvider:
        return self._provider

    async def synthesize(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> bytes:
        """Synthesize text to speech, return complete audio bytes."""
        result = await self._provider.synthesize(
            text=text,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            **kwargs,
        )
        return result.audio_data

    async def synthesize_and_upload(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> dict:
        """Synthesize and upload to S3 storage."""
        from app.services.s3_storage import s3_storage

        audio_data = await self.synthesize(
            text=text,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            **kwargs,
        )

        upload_result = s3_storage.upload_tts_audio(
            audio_data=audio_data,
            codec=codec,
            text=text,
            subfolder="tts",
        )

        return upload_result

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
        """Synthesize text to speech and yield audio chunks."""
        async for chunk in self._provider.synthesize_stream(
            text=text,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            **kwargs,
        ):
            yield chunk

    def create_stream_session(
        self,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> BaseTTSStreamSession:
        """Create a persistent stream session (for ws_chat)."""
        return self._provider.create_stream_session(
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            **kwargs,
        )
