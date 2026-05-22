"""
Tencent Cloud ASR (Automatic Speech Recognition) service using Flash Recognizer SDK.
"""
import asyncio
import ipaddress
import json
import socket
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.core.thread_pool import ThreadPool
from app.core.sdk_path import SDK_PATH  # noqa: F401 — ensures SDK is on sys.path

from common.credential import Credential
from asr.flash_recognizer import FlashRecognizer, FlashRecognitionRequest

from app.core.config import settings
from app.services.tencent.audio import convert_audio_to_wav


class ASRService:
    """Tencent Cloud ASR service for speech-to-text conversion using Flash Recognizer SDK."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        appid: Optional[str] = None
    ):
        self.secret_id = secret_id or settings.tencent_secret_id
        self.secret_key = secret_key or settings.tencent_secret_key
        self.appid = appid or settings.tencent_appid

    async def download_audio(self, url: str) -> bytes:
        """Download audio file from URL asynchronously."""
        await self._validate_download_url(url)
        max_bytes = max(1, settings.audio_download_max_bytes)

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = None
                    if declared_size and declared_size > max_bytes:
                        raise ValueError(f"Audio file is too large: {content_length} bytes")

                chunks = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Audio file exceeds limit: {max_bytes} bytes")
                    chunks.append(chunk)

        return b"".join(chunks)

    async def _validate_download_url(self, url: str) -> None:
        """Validate remote audio URL to reduce SSRF risk."""
        parsed = urlparse(str(url))
        if parsed.scheme not in ("http", "https"):
            raise ValueError("audio_url must use http or https")
        if not parsed.hostname:
            raise ValueError("audio_url host is required")

        if settings.audio_download_allow_private:
            return

        host = parsed.hostname

        def _resolve_host() -> list[str]:
            infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
            return list({info[4][0] for info in infos})

        addresses = await asyncio.to_thread(_resolve_host)
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise ValueError(f"audio_url resolves to a blocked address: {address}")

    async def convert_audio(self, audio_data: bytes) -> bytes:
        """
        Convert audio to standard format: 16kHz, 16bit, mono, WAV.

        Args:
            audio_data: Input audio bytes (any format)

        Returns:
            Converted WAV audio bytes
        """
        return await convert_audio_to_wav(
            audio_data,
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )

    def _sync_recognize(
        self,
        audio_data: bytes,
        engine_type: str,
        word_info: int = 0
    ) -> dict:
        """Synchronous recognition using Flash Recognizer SDK."""
        import logging
        logger = logging.getLogger(__name__)

        credential = Credential(self.secret_id, self.secret_key)
        recognizer = FlashRecognizer(self.appid, credential)

        # Create recognition request
        req = FlashRecognitionRequest(engine_type)
        req.set_voice_format("wav")
        req.set_filter_dirty(0)
        req.set_filter_modal(0)
        req.set_filter_punc(0)
        req.set_convert_num_mode(1)
        req.set_word_info(word_info)  # 0: no timestamp, 1: word timestamp (no punctuation), 2: word timestamp (with punctuation)
        req.set_first_channel_only(1)

        logger.info(f"ASR: audio_data size = {len(audio_data)} bytes, word_info = {word_info}")

        # Execute recognition
        result_data = recognizer.recognize(req, audio_data)
        result = json.loads(result_data)

        logger.info(f"ASR result: code={result.get('code')}, message={result.get('message')}")
        if result.get('flash_result'):
            for i, ch in enumerate(result['flash_result']):
                logger.info(f"ASR channel {i}: text='{ch.get('text', '')}'")

        return result

    async def recognize_audio(
        self,
        audio_data: bytes,
        engine_type: str = "16k_zh",
        voice_format: str = "wav",
        word_info: int = 0
    ) -> dict:
        """
        Recognize speech in audio using Flash Recognizer API.

        Audio will be automatically converted to 16kHz, 16bit, mono WAV format.

        Args:
            audio_data: Audio file bytes (any format supported by ffmpeg).
            engine_type: Recognition engine type (16k_zh, 16k_en, etc.)
            voice_format: Ignored - audio will be converted to WAV.
            word_info: Word level timestamp. 0: no timestamp, 1: word timestamp (no punctuation), 2: word timestamp (with punctuation).

        Returns:
            Recognition result dict with 'text', 'word_info_list', and 'raw_response'.
        """
        # Convert audio to standard format: 16kHz, 16bit, mono, WAV
        audio_data = await self.convert_audio(audio_data)

        # Run sync recognition in centralized thread pool
        result = await ThreadPool.run(
            self._sync_recognize,
            audio_data,
            engine_type,
            word_info
        )

        # Check for errors
        code = result.get("code", 0)
        if code != 0:
            raise Exception(f"ASR failed: {result.get('message', 'Unknown error')}")

        # Extract text from flash_result
        text = ""
        word_info_list = []
        flash_result = result.get("flash_result", [])
        if flash_result:
            # Get text from first channel
            text = flash_result[0].get("text", "")
            # Extract word info if available
            if "sentence_list" in flash_result[0]:
                for sentence in flash_result[0]["sentence_list"]:
                    if "word_list" in sentence:
                        for word in sentence["word_list"]:
                            word_info_list.append({
                                "word": word.get("word", ""),
                                "begin_time": word.get("begin_time", 0),
                                "end_time": word.get("end_time", 0),
                                "duration": word.get("end_time", 0) - word.get("begin_time", 0)
                            })

        return {
            "text": text,
            "word_info_list": word_info_list,
            "raw_response": result
        }


# Singleton instance
asr_service = ASRService()
