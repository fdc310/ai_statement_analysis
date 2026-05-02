"""
Tencent Cloud TTS provider — wraps the existing FlowingSpeechSynthesizer SDK.
"""
import logging
import queue
import re
import threading
from typing import Optional, AsyncGenerator

from app.core.config import settings
from app.core.sdk_path import SDK_PATH  # noqa: F401
from app.core.thread_pool import ThreadPool
from app.services.tts.base import BaseTTSProvider, TTSResponse
from app.services.tts.stream_session import BaseTTSStreamSession

from common.credential import Credential
from tts.flowing_speech_synthesizer import FlowingSpeechSynthesizer, FlowingSpeechSynthesisListener

logger = logging.getLogger(__name__)


class _FlowingListener(FlowingSpeechSynthesisListener):
    """Listener that collects audio data and supports streaming via queue."""

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
        self._audio_queue.put(None)
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
        self._audio_queue.put(None)
        self._done_event.set()


class _TencentStreamSession(BaseTTSStreamSession):
    """Persistent TTS stream session using Tencent FlowingSpeechSynthesizer."""

    def __init__(
        self,
        appid: str,
        secret_id: str,
        secret_key: str,
        voice_type: int,
        codec: str,
        sample_rate: int,
        speed: float,
        volume: float,
    ):
        self._appid = appid
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._voice_type = voice_type
        self._codec = codec
        self._sample_rate = sample_rate
        self._speed = speed
        self._volume = volume

        self._listener = _FlowingListener()
        self._synthesizer: Optional[FlowingSpeechSynthesizer] = None
        self._error: Optional[str] = None

    def start(self) -> None:
        credential = Credential(self._secret_id, self._secret_key)
        self._synthesizer = FlowingSpeechSynthesizer(int(self._appid), credential, self._listener)
        self._synthesizer.set_voice_type(self._voice_type)
        self._synthesizer.set_codec(self._codec)
        self._synthesizer.set_sample_rate(self._sample_rate)
        self._synthesizer.set_speed(self._speed)
        self._synthesizer.set_volume(self._volume)
        self._synthesizer.start()

    def wait_ready(self, timeout_ms: int) -> bool:
        if not self._synthesizer:
            return False
        return self._synthesizer.wait_ready(timeout_ms)

    def process(self, text: str) -> None:
        if self._synthesizer:
            self._synthesizer.process(text)

    def complete(self) -> None:
        if self._synthesizer:
            self._synthesizer.complete()

    def wait(self) -> None:
        if self._synthesizer:
            self._synthesizer.wait()
        if self._listener.error:
            self._error = self._listener.error

    def get_audio_chunks(self) -> list[bytes]:
        chunks = []
        while True:
            try:
                chunk = self._listener._audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                chunks.append(chunk)
            except queue.Empty:
                break
        return chunks

    @property
    def error(self) -> Optional[str]:
        return self._error or self._listener.error


class TencentTTSProvider(BaseTTSProvider):
    """Tencent Cloud TTS provider using FlowingSpeechSynthesizer SDK."""

    def __init__(self, **kwargs):
        self._secret_id = kwargs.get("secret_id") or settings.tencent_secret_id
        self._secret_key = kwargs.get("secret_key") or settings.tencent_secret_key
        self._appid = kwargs.get("appid") or settings.tencent_appid

    @property
    def name(self) -> str:
        return "tencent"

    def _create_synthesizer(
        self,
        listener: FlowingSpeechSynthesisListener,
        voice_type: int,
        codec: str,
        sample_rate: int,
        speed: float,
        volume: float,
    ) -> FlowingSpeechSynthesizer:
        credential = Credential(self._secret_id, self._secret_key)
        synthesizer = FlowingSpeechSynthesizer(int(self._appid), credential, listener)
        synthesizer.set_voice_type(voice_type)
        synthesizer.set_codec(codec)
        synthesizer.set_sample_rate(sample_rate)
        synthesizer.set_speed(speed)
        synthesizer.set_volume(volume)
        return synthesizer

    @staticmethod
    def _split_text(text: str, max_chunk_size: int = 150) -> list:
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
        **kwargs,
    ) -> TTSResponse:
        ready_timeout_ms = kwargs.get("ready_timeout_ms", 5000)
        listener = _FlowingListener()
        synthesizer = self._create_synthesizer(listener, voice_type, codec, sample_rate, speed, volume)
        text_chunks = self._split_text(text)

        def run():
            synthesizer.start()
            if not synthesizer.wait_ready(ready_timeout_ms):
                raise Exception("TTS WebSocket connection not ready within timeout")
            for chunk in text_chunks:
                synthesizer.process(chunk)
            synthesizer.complete()
            synthesizer.wait()

        await ThreadPool.run(run)

        if listener.error:
            raise Exception(listener.error)

        return TTSResponse(
            audio_data=listener.audio_data,
            audio_chunks=[listener.audio_data],
            content_type="audio/mpeg" if codec == "mp3" else "audio/pcm",
        )

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
        ready_timeout_ms = kwargs.get("ready_timeout_ms", 5000)
        timeout = kwargs.get("timeout", 300.0)
        listener = _FlowingListener()
        synthesizer = self._create_synthesizer(listener, voice_type, codec, sample_rate, speed, volume)
        text_chunks = self._split_text(text)

        def run():
            synthesizer.start()
            if not synthesizer.wait_ready(ready_timeout_ms):
                raise Exception("TTS WebSocket connection not ready within timeout")
            for chunk in text_chunks:
                synthesizer.process(chunk)
            synthesizer.complete()
            synthesizer.wait()

        import time
        synthesis_thread = threading.Thread(target=run, daemon=True)
        synthesis_thread.start()

        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise Exception(f"TTS synthesis timeout after {timeout} seconds")
            try:
                chunk = listener._audio_queue.get(timeout=0.05)
                if chunk is None:
                    if listener.error:
                        raise Exception(listener.error)
                    return
                yield chunk
            except queue.Empty:
                continue

    def create_stream_session(
        self,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> BaseTTSStreamSession:
        return _TencentStreamSession(
            appid=self._appid,
            secret_id=self._secret_id,
            secret_key=self._secret_key,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
        )
