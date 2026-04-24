# -*- coding: utf-8 -*-
"""
Tencent Cloud TTS (Text-to-Speech) service using WebSocket SDK.
Uses FlowingSpeechSynthesizer for streaming synthesis without character limits.
"""
import sys
import os
import queue
import re
import threading
import time
from typing import Optional, AsyncGenerator

from app.core.thread_pool import ThreadPool

# Add SDK path to sys.path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from tts.flowing_speech_synthesizer import FlowingSpeechSynthesizer, FlowingSpeechSynthesisListener

from app.core.config import settings
from app.services.s3_storage import s3_storage


class FlowingListener(FlowingSpeechSynthesisListener):
    """Listener for flowing speech synthesis — collects audio and supports streaming."""

    def __init__(self):
        self.audio_data = bytes()
        self.is_complete = False
        self.error = None
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._audio_queue = queue.Queue()

    def on_synthesis_start(self, session_id):
        pass

    def on_synthesis_end(self):
        with self._lock:
            self.is_complete = True
        self._audio_queue.put(None)  # sentinel
        self._done_event.set()

    def on_audio_result(self, audio_bytes):
        with self._lock:
            self.audio_data += audio_bytes
        self._audio_queue.put(audio_bytes)

    def on_text_result(self, response):
        pass

    def on_synthesis_fail(self, response):
        with self._lock:
            self.error = f"TTS failed: code={response.get('code')}, message={response.get('message')}"
            self.is_complete = True
        self._audio_queue.put(None)  # sentinel
        self._done_event.set()


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
        listener: FlowingSpeechSynthesisListener,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0
    ) -> FlowingSpeechSynthesizer:
        """Create a flowing speech synthesizer instance."""
        credential = Credential(self.secret_id, self.secret_key)
        synthesizer = FlowingSpeechSynthesizer(int(self.appid), credential, listener)
        synthesizer.set_voice_type(voice_type)
        synthesizer.set_codec(codec)
        synthesizer.set_sample_rate(sample_rate)
        synthesizer.set_speed(speed)
        synthesizer.set_volume(volume)
        return synthesizer

    def _split_text(self, text: str, max_chunk_size: int = 150) -> list:
        """
        Split text into chunks at sentence boundaries to avoid per-message limits.
        Falls back to comma/pause punctuation for long sentences.
        """
        # First split at sentence-ending punctuation
        parts = re.split(r'(?<=[。！？\n])', text)
        chunks = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) <= max_chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current)
                # Long single sentence: split further at commas/pauses
                if len(part) > max_chunk_size:
                    sub_parts = re.split(r'(?<=[，、；])', part)
                    current = ""
                    for sub in sub_parts:
                        if not sub:
                            continue
                        if len(current) + len(sub) <= max_chunk_size:
                            current += sub
                        else:
                            if current:
                                chunks.append(current)
                            current = sub
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks if chunks else [text]

    async def synthesize(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        ready_timeout_ms: int = 5000
    ) -> bytes:
        """
        Synthesize text to speech using flowing synthesis (no character limit).

        Args:
            text: Text to synthesize
            voice_type: Voice type ID (default: 101001 - 智瑜)
            codec: Audio format - "mp3" or "pcm" (default: "mp3")
            sample_rate: Sample rate - 8000 or 16000 (default: 16000)
            speed: Speech speed, range -2.0 to 6.0 (default: 1.0)
            volume: Volume adjustment in dB, range -10.0 to 10.0 (default: 0.0)
            ready_timeout_ms: Max wait for WebSocket ready state in ms (default: 5000)

        Returns:
            Complete audio data as bytes
        """
        listener = FlowingListener()
        synthesizer = self._create_synthesizer(
            listener, voice_type, codec, sample_rate, speed, volume
        )
        chunks = self._split_text(text)

        def run_synthesis():
            synthesizer.start()
            if not synthesizer.wait_ready(ready_timeout_ms):
                raise Exception("TTS WebSocket connection not ready within timeout")
            for chunk in chunks:
                synthesizer.process(chunk)
            synthesizer.complete()
            synthesizer.wait()

        await ThreadPool.run(run_synthesis)

        if listener.error:
            raise Exception(listener.error)

        return listener.audio_data

    async def synthesize_and_upload(
        self,
        text: str,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0
    ) -> dict:
        """
        Synthesize text to speech and upload to S3 storage.

        Args:
            text: Text to synthesize
            voice_type: Voice type ID (default: 101001 - 智瑜)
            codec: Audio format - "mp3" or "pcm" (default: "mp3")
            sample_rate: Sample rate - 8000 or 16000 (default: 16000)
            speed: Speech speed, range -2.0 to 6.0 (default: 1.0)
            volume: Volume adjustment in dB, range -10.0 to 10.0 (default: 0.0)

        Returns:
            Dict with:
                - success: bool
                - url: str (S3 URL if successful)
                - object_key: str (S3 object key if successful)
                - error: str (if failed)
        """
        audio_data = await self.synthesize(
            text=text,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume
        )

        upload_result = s3_storage.upload_tts_audio(
            audio_data=audio_data,
            codec=codec,
            text=text,
            subfolder="tts"
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
        ready_timeout_ms: int = 5000,
        timeout: float = 300.0
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech and yield audio chunks as they are produced.

        Args:
            text: Text to synthesize
            voice_type: Voice type ID (default: 101001 - 智瑜)
            codec: Audio format - "mp3" or "pcm" (default: "mp3")
            sample_rate: Sample rate - 8000 or 16000 (default: 16000)
            speed: Speech speed, range -2.0 to 6.0 (default: 1.0)
            volume: Volume adjustment in dB, range -10.0 to 10.0 (default: 0.0)
            ready_timeout_ms: Max wait for WebSocket ready state in ms (default: 5000)
            timeout: Maximum total synthesis time in seconds (default: 300)

        Yields:
            Audio data chunks as bytes
        """
        listener = FlowingListener()
        synthesizer = self._create_synthesizer(
            listener, voice_type, codec, sample_rate, speed, volume
        )
        chunks = self._split_text(text)

        def run_synthesis():
            synthesizer.start()
            if not synthesizer.wait_ready(ready_timeout_ms):
                raise Exception("TTS WebSocket connection not ready within timeout")
            for chunk in chunks:
                synthesizer.process(chunk)
            synthesizer.complete()
            synthesizer.wait()

        synthesis_thread = threading.Thread(target=run_synthesis, daemon=True)
        synthesis_thread.start()

        # Drain the audio queue, yielding chunks as they arrive
        import time
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise Exception(f"TTS synthesis timeout after {timeout} seconds")
            try:
                chunk = listener._audio_queue.get(timeout=0.05)
                if chunk is None:  # sentinel — synthesis ended
                    if listener.error:
                        raise Exception(listener.error)
                    return
                yield chunk
            except queue.Empty:
                continue


# Default service instance
tts_service = TTSService()
