"""
Volcengine (火山引擎) TTS provider.

Uses the Volcengine TTS WebSocket API for streaming synthesis.
Requires: volcengine-python-sdk or direct WebSocket connection.
"""
import asyncio
import json
import logging
import struct
import threading
from typing import Optional, AsyncGenerator

import httpx

from app.core.config import settings
from app.services.tts.base import BaseTTSProvider, TTSResponse
from app.services.tts.stream_session import BaseTTSStreamSession

logger = logging.getLogger(__name__)

# Volcengine TTS API endpoint
_VOLCENGINE_TTS_URL = "https://openspeech.bytedance.com/api/v1/tts"


class _VolcengineStreamSession(BaseTTSStreamSession):
    """Persistent stream session for Volcengine TTS via HTTP chunked requests."""

    def __init__(
        self,
        appid: str,
        access_token: str,
        voice_type: int,
        codec: str,
        sample_rate: int,
        speed: float,
        volume: float,
    ):
        self._appid = appid
        self._access_token = access_token
        self._voice_type = voice_type
        self._codec = codec
        self._sample_rate = sample_rate
        self._speed = speed
        self._volume = volume

        self._text_buffer = ""
        self._audio_chunks: list[bytes] = []
        self._error: Optional[str] = None
        self._ready = False

    def start(self) -> None:
        self._ready = True

    def wait_ready(self, timeout_ms: int) -> bool:
        return self._ready

    def process(self, text: str) -> None:
        """Buffer text for synthesis on complete()."""
        self._text_buffer += text

    def complete(self) -> None:
        """Synthesize all buffered text in one request."""
        if not self._text_buffer.strip():
            return
        try:
            import httpx as _httpx
            headers = {
                "Authorization": f"Bearer; {self._access_token}",
                "Content-Type": "application/json",
            }
            # Map voice_type to Volcengine voice name
            voice_name = _map_voice_type(self._voice_type)
            codec = "mp3" if self._codec == "mp3" else "wav"
            payload = {
                "app": {"appid": self._appid, "token": self._access_token},
                "user": {"uid": "ws_chat_user"},
                "audio": {
                    "voice_type": voice_name,
                    "encoding": codec,
                    "speed_ratio": self._speed,
                    "volume_ratio": self._volume,
                    "sample_rate": self._sample_rate,
                },
                "request": {
                    "reqid": f"stream_{id(self)}",
                    "text": self._text_buffer,
                    "operation": "query",
                },
            }
            with _httpx.Client(timeout=60.0) as client:
                resp = client.post(_VOLCENGINE_TTS_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 3000:
                    self._error = f"Volcengine TTS error: {data.get('message', 'unknown')}"
                    return
                audio_b64 = data.get("data", "")
                if audio_b64:
                    import base64
                    self._audio_chunks.append(base64.b64decode(audio_b64))
        except Exception as e:
            self._error = f"Volcengine TTS error: {e}"

    def wait(self) -> None:
        pass

    def get_audio_chunks(self) -> list[bytes]:
        return self._audio_chunks

    @property
    def error(self) -> Optional[str]:
        return self._error


def _map_voice_type(voice_type: int) -> str:
    """Map numeric voice_type to Volcengine voice name."""
    # Default mapping; extend as needed
    _VOICE_MAP = {
        101001: "zh_female_cancan",
        101002: "zh_female_shuangkuai",
        101005: "zh_male_chunhou",
    }
    return _VOICE_MAP.get(voice_type, "zh_female_cancan")


class VolcengineTTSProvider(BaseTTSProvider):
    """Volcengine (火山引擎) TTS provider."""

    def __init__(self, **kwargs):
        self._appid = kwargs.get("appid") or settings.volcengine_tts_appid
        self._access_token = kwargs.get("access_token") or settings.volcengine_tts_access_token

    @property
    def name(self) -> str:
        return "volcengine"

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
        import base64
        voice_name = _map_voice_type(voice_type)
        audio_codec = "mp3" if codec == "mp3" else "wav"

        headers = {
            "Authorization": f"Bearer; {self._access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "app": {"appid": self._appid, "token": self._access_token},
            "user": {"uid": "api_user"},
            "audio": {
                "voice_type": voice_name,
                "encoding": audio_codec,
                "speed_ratio": speed,
                "volume_ratio": volume,
                "sample_rate": sample_rate,
            },
            "request": {
                "reqid": f"tts_{id(text)}",
                "text": text,
                "operation": "query",
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_VOLCENGINE_TTS_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 3000:
            raise Exception(f"Volcengine TTS error: {data.get('message', 'unknown')}")

        audio_b64 = data.get("data", "")
        audio_data = base64.b64decode(audio_b64) if audio_b64 else b""

        return TTSResponse(
            audio_data=audio_data,
            audio_chunks=[audio_data],
            content_type="audio/mpeg" if codec == "mp3" else "audio/wav",
        )

    def create_stream_session(
        self,
        voice_type: int = 101001,
        codec: str = "mp3",
        sample_rate: int = 16000,
        speed: float = 1.0,
        volume: float = 0.0,
        **kwargs,
    ) -> BaseTTSStreamSession:
        return _VolcengineStreamSession(
            appid=self._appid,
            access_token=self._access_token,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
        )
