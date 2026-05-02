"""
Xiaomi TTS provider.

Uses Xiaomi's TTS HTTP API for speech synthesis.
"""
import base64
import logging
from typing import Optional

import httpx

from app.core.config import settings
from app.services.tts.base import BaseTTSProvider, TTSResponse

logger = logging.getLogger(__name__)

# Xiaomi TTS API endpoint
_XIAOMI_TTS_URL = "https://open.xiaomi.com/tts/v1/synthesize"


def _map_voice_type(voice_type: int) -> str:
    """Map numeric voice_type to Xiaomi voice name."""
    _VOICE_MAP = {
        101001: "female-tianmei",
        101002: "female-qn-jingying",
        101005: "male-qn-qingse",
    }
    return _VOICE_MAP.get(voice_type, "female-tianmei")


class XiaomiTTSProvider(BaseTTSProvider):
    """Xiaomi TTS provider using HTTP API."""

    def __init__(self, **kwargs):
        self._appid = kwargs.get("appid") or settings.xiaomi_tts_appid
        self._api_key = kwargs.get("api_key") or settings.xiaomi_tts_api_key

    @property
    def name(self) -> str:
        return "xiaomi"

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
        voice_name = _map_voice_type(voice_type)
        audio_codec = "mp3" if codec == "mp3" else "wav"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "app_id": self._appid,
            "voice": voice_name,
            "text": text,
            "audio": {
                "encoding": audio_codec,
                "sample_rate": sample_rate,
                "speed": speed,
                "volume": volume,
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_XIAOMI_TTS_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"Xiaomi TTS error: {data.get('message', 'unknown')}")

        audio_b64 = data.get("audio", "")
        audio_data = base64.b64decode(audio_b64) if audio_b64 else b""

        return TTSResponse(
            audio_data=audio_data,
            audio_chunks=[audio_data],
            content_type="audio/mpeg" if codec == "mp3" else "audio/wav",
        )
