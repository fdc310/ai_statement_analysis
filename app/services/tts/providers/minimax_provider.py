"""
Minimax TTS provider.

Uses Minimax's TTS HTTP API (t2a_v2) for speech synthesis.
"""
import base64
import logging
from typing import Optional

import httpx

from app.core.config import settings
from app.services.tts.base import BaseTTSProvider, TTSResponse

logger = logging.getLogger(__name__)

# Minimax TTS API endpoint
_MINIMAX_TTS_URL = "https://api.minimax.chat/v1/t2a_v2"


def _map_voice_type(voice_type: int) -> str:
    """Map numeric voice_type to Minimax voice ID."""
    _VOICE_MAP = {
        101001: "female-tianmei",
        101002: "female-shaonv",
        101005: "male-qn-qingse",
        101006: "male-qn-jingying",
    }
    return _VOICE_MAP.get(voice_type, "female-tianmei")


class MinimaxTTSProvider(BaseTTSProvider):
    """Minimax TTS provider using HTTP API."""

    def __init__(self, **kwargs):
        self._api_key = kwargs.get("api_key") or settings.minimax_tts_api_key
        self._group_id = kwargs.get("group_id") or settings.minimax_tts_group_id

    @property
    def name(self) -> str:
        return "minimax"

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
        voice_id = _map_voice_type(voice_type)
        audio_codec = "mp3" if codec == "mp3" else "wav"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "speech-01-turbo",
            "text": text,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": volume,
            },
            "audio_setting": {
                "audio_encoding": audio_codec,
                "sample_rate": sample_rate,
            },
        }

        url = f"{_MINIMAX_TTS_URL}?GroupId={self._group_id}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("base_resp", {}).get("status_code", -1) != 0:
            err_msg = data.get("base_resp", {}).get("status_msg", "unknown")
            raise Exception(f"Minimax TTS error: {err_msg}")

        audio_hex = data.get("audio_file", "")
        audio_data = bytes.fromhex(audio_hex) if audio_hex else b""
        # Fallback: try base64 if hex decode fails
        if not audio_data and audio_hex:
            try:
                audio_data = base64.b64decode(audio_hex)
            except Exception:
                pass

        return TTSResponse(
            audio_data=audio_data,
            audio_chunks=[audio_data],
            content_type="audio/mpeg" if codec == "mp3" else "audio/wav",
        )
