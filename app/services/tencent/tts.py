# -*- coding: utf-8 -*-
"""
Tencent Cloud TTS (Text-to-Speech) service using WebSocket SDK.
"""
import asyncio
import sys
import os
import threading
from typing import Optional, AsyncGenerator

# Add SDK path to sys.path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from tts.speech_synthesizer_ws import SpeechSynthesizer, SpeechSynthesisListener

from app.core.config import settings


class StreamingSynthesisListener(SpeechSynthesisListener):
    """Listener that collects audio data for streaming."""

    def __init__(self):
        self.audio_data = bytes()
        self.is_complete = False
        self.error = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def on_synthesis_start(self, session_id):
        super().on_synthesis_start(session_id)
        self.audio_data = bytes()
        self.is_complete = False
        self.error = None

    def on_synthesis_end(self):
        super().on_synthesis_end()
        with self._lock:
            self.is_complete = True
        self._event.set()

    def on_audio_result(self, audio_bytes):
        super().on_audio_result(audio_bytes)
        with self._lock:
            self.audio_data += audio_bytes
        self._event.set()

    def on_text_result(self, response):
        super().on_text_result(response)

    def on_synthesis_fail(self, response):
        super().on_synthesis_fail(response)
        with self._lock:
            self.error = f"TTS failed: code={response.get('code')}, message={response.get('message')}"
            self.is_complete = True
        self._event.set()

    def get_audio_data(self) -> bytes:
        """Get collected audio data."""
        with self._lock:
            data = self.audio_data
            self.audio_data = bytes()
            return data


class TTSService:
    """Tencent Cloud TTS service for text-to-speech conversion."""

    # Common voice types
    VOICE_TYPES = {
        # Chinese female voices
        "zh_female_1": 101001,  # 智瑜 - 通用女声
        "zh_female_2": 101002,  # 智聆 - 通用女声
        "zh_female_3": 101003,  # 智美 - 客服女声
        "zh_female_4": 101004,  # 智云 - 通用女声
        # Chinese male voices
        "zh_male_1": 101005,    # 智华 - 通用男声
        "zh_male_2": 101006,    # 智龙 - 新闻男声
        "zh_male_3": 101007,    # 智明 - 新闻男声
        # English voices
        "en_female_1": 101050,  # WeJack - 英文女声
        "en_male_1": 101051,    # WeRose - 英文男声
    }

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        appid: Optional[str] = None
    ):
        self.secret_id = secret_id or settings.tencent_secret_id
        self.secret_key = secret_key or settings.tencent_secret_key
        self.appid = appid or settings.tencent_appid

    def _create_synthesizer(
        self,
        listener: SpeechSynthesisListener,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0
    ) -> SpeechSynthesizer:
        """Create a speech synthesizer instance."""
        credential = Credential(self.secret_id, self.secret_key)
        synthesizer = SpeechSynthesizer(int(self.appid), credential, listener)
        synthesizer.set_voice_type(voice_type)
        synthesizer.set_codec(codec)
        synthesizer.set_sample_rate(sample_rate)
        synthesizer.set_speed(speed)
        synthesizer.set_volume(volume)
        return synthesizer

    async def synthesize(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0
    ) -> bytes:
        """
        Synthesize text to speech and return complete audio data.

        Args:
            text: Text to synthesize
            voice_type: Voice type ID (default: 101001 - 智瑜)
            codec: Audio format - "mp3" or "pcm" (default: "mp3")
            sample_rate: Sample rate - 8000 or 16000 (default: 16000)
            speed: Speech speed, range -2.0 to 6.0 (default: 1.0)
            volume: Volume adjustment in dB, range -10.0 to 10.0 (default: 0.0)

        Returns:
            Audio data as bytes
        """
        listener = StreamingSynthesisListener()
        synthesizer = self._create_synthesizer(
            listener, voice_type, codec, sample_rate, speed, volume
        )
        synthesizer.set_text(text)

        # Run synthesis in thread pool
        def run_synthesis():
            synthesizer.start()
            synthesizer.wait()

        await asyncio.to_thread(run_synthesis)

        if listener.error:
            raise Exception(listener.error)

        return listener.audio_data

    async def synthesize_stream(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        chunk_interval: float = 0.05,
        timeout: float = 300.0
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech and yield audio chunks as they become available.

        Args:
            text: Text to synthesize
            voice_type: Voice type ID (default: 101001 - 智瑜)
            codec: Audio format - "mp3" or "pcm" (default: "mp3")
            sample_rate: Sample rate - 8000 or 16000 (default: 16000)
            speed: Speech speed, range -2.0 to 6.0 (default: 1.0)
            volume: Volume adjustment in dB, range -10.0 to 10.0 (default: 0.0)
            chunk_interval: Interval between chunk yields in seconds (default: 0.05)
            timeout: Maximum time to wait for synthesis completion (default: 300s)

        Yields:
            Audio data chunks as bytes
        """
        import time
        start_time = time.time()

        listener = StreamingSynthesisListener()
        synthesizer = self._create_synthesizer(
            listener, voice_type, codec, sample_rate, speed, volume
        )
        synthesizer.set_text(text)

        # Start synthesis in background thread (daemon=True, auto cleanup)
        def run_synthesis():
            synthesizer.start()
            synthesizer.wait()

        synthesis_thread = threading.Thread(target=run_synthesis, daemon=True)
        synthesis_thread.start()

        # Yield audio chunks as they become available (non-blocking)
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                raise Exception(f"TTS synthesis timeout after {timeout} seconds")

            # Use asyncio.sleep instead of blocking wait
            await asyncio.sleep(chunk_interval)

            # Get available audio data
            chunk = listener.get_audio_data()
            if chunk:
                yield chunk

            # Check if synthesis is complete
            with listener._lock:
                if listener.is_complete:
                    # Get any remaining data
                    remaining = listener.get_audio_data()
                    if remaining:
                        yield remaining
                    if listener.error:
                        raise Exception(listener.error)
                    return  # Use return instead of break for cleaner exit


# Default service instance
tts_service = TTSService()
