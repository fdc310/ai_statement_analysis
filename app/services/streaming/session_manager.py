"""
WebSocket session manager for streaming audio evaluation.
Manages the lifecycle of a streaming evaluation session.
"""
import asyncio
import logging
import uuid
from typing import Optional, Callable, Awaitable

from pydantic import BaseModel, Field

from app.services.streaming.asr_stream import StreamingASR
from app.services.streaming.soe_stream import StreamingSOE
from app.services.streaming.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)


class StreamConfig(BaseModel):
    """Configuration for a streaming session."""
    language: str = "zh"
    ref_text: str = ""
    eval_mode: int = 3
    score_coeff: float = 1.0
    server_type: int = 0
    word_info: int = 1
    enable_asr: bool = True
    enable_soe: bool = True


class StreamResult(BaseModel):
    """Result from a streaming session."""
    session_id: str
    asr_result: Optional[dict] = None
    soe_result: Optional[dict] = None
    speech_text: str = ""
    scores_data: dict = Field(default_factory=dict)
    audio_url: Optional[str] = None


class StreamingSession:
    """
    Manages a streaming audio evaluation session.

    Handles:
    - ASR (real-time speech-to-text)
    - SOE (real-time pronunciation scoring)
    - Audio buffering and chunking
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        config: Optional[StreamConfig] = None,
        on_asr_partial: Optional[Callable] = None,
        on_soe_intermediate: Optional[Callable] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or StreamConfig()
        self._asr: Optional[StreamingASR] = None
        self._soe: Optional[StreamingSOE] = None
        self._buffer = AudioBuffer()
        self._started = False
        self._finished = False

        # Callbacks for real-time updates
        self._on_asr_partial = on_asr_partial
        self._on_soe_intermediate = on_soe_intermediate

    async def start(self) -> None:
        """Start the streaming session."""
        if self._started:
            return

        logger.info(f"Streaming session {self.session_id} starting")

        # Start ASR if enabled
        if self.config.enable_asr:
            self._asr = StreamingASR()
            await self._asr.start_recognition(
                engine_type="16k_zh" if self.config.language == "zh" else "16k_en",
                word_info=self.config.word_info
            )

        # Start SOE if enabled
        if self.config.enable_soe:
            self._soe = StreamingSOE()
            await self._soe.start_evaluation(
                ref_text=self.config.ref_text,
                eval_mode=self.config.eval_mode,
                score_coeff=self.config.score_coeff,
                server_type=self.config.server_type
            )

        self._started = True
        logger.info(f"Streaming session {self.session_id} started")

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        """
        Feed audio data to the session.

        Args:
            pcm_chunk: Raw PCM 16kHz 16bit mono audio bytes
        """
        if not self._started:
            raise RuntimeError("Session not started")

        # Buffer the audio
        await self._buffer.append(pcm_chunk)

        # Feed to ASR
        if self._asr:
            try:
                await self._asr.feed_audio(pcm_chunk)
            except Exception as e:
                logger.error(f"ASR feed error: {e}")

        # Feed to SOE (SOE expects WAV format, so we need to convert)
        # For streaming, we send raw PCM and let SOE handle it
        if self._soe:
            try:
                await self._soe.feed_audio(pcm_chunk)
            except Exception as e:
                logger.error(f"SOE feed error: {e}")

    async def finish(self) -> StreamResult:
        """
        Finish the session and return results.

        Returns:
            StreamResult with ASR and SOE results
        """
        if self._finished:
            raise RuntimeError("Session already finished")

        logger.info(f"Streaming session {self.session_id} finishing")

        # Mark buffer as final
        self._buffer.set_final()

        # Stop ASR and get final result
        asr_result = None
        speech_text = ""
        if self._asr:
            try:
                asr_result = await self._asr.stop_recognition()
                if asr_result and "error" not in asr_result:
                    # Extract text from result
                    if "result" in asr_result:
                        speech_text = asr_result["result"].get("text", "")
            except Exception as e:
                logger.error(f"ASR stop error: {e}")

        # Stop SOE and get final result
        soe_result = None
        scores_data = {}
        if self._soe:
            try:
                soe_result = await self._soe.stop_evaluation()
                if soe_result and "error" not in soe_result:
                    # Parse SOE result
                    from app.services.tencent.soe import SOEService
                    soe_service = SOEService()
                    parsed = soe_service.parse_evaluation_result(soe_result)
                    scores_data = parsed.get("scores", {})
            except Exception as e:
                logger.error(f"SOE stop error: {e}")

        # Upload audio to S3
        audio_url = None
        try:
            audio_data = await self._buffer.get_full_buffer()
            if audio_data:
                from app.services.s3_storage import s3_storage
                upload_result = s3_storage.upload_bytes(
                    data=audio_data,
                    object_name=f"streaming/{self.session_id}.pcm",
                    content_type="audio/pcm",
                    subfolder="streaming",
                )
                if upload_result.get("success"):
                    audio_url = upload_result.get("url")
                    logger.info(f"Streaming session {self.session_id} audio uploaded: {audio_url}")
                else:
                    logger.warning(f"Streaming session {self.session_id} audio upload failed: {upload_result.get('error')}")
        except Exception as e:
            logger.error(f"Streaming session {self.session_id} audio upload error: {e}")

        self._finished = True

        result = StreamResult(
            session_id=self.session_id,
            asr_result=asr_result,
            soe_result=soe_result,
            speech_text=speech_text,
            scores_data=scores_data,
            audio_url=audio_url,
        )

        logger.info(f"Streaming session {self.session_id} completed")
        return result

    async def get_asr_results(self):
        """Get ASR results as they arrive."""
        if self._asr:
            async for event in self._asr.get_results():
                yield event

    async def get_soe_results(self):
        """Get SOE results as they arrive."""
        if self._soe:
            async for event in self._soe.get_intermediate_results():
                yield event
