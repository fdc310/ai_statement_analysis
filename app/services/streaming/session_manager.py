"""
WebSocket session manager for streaming audio evaluation.
Manages the lifecycle of a streaming evaluation session.
"""
import asyncio
import logging
import string
import uuid
from typing import Optional, Callable, Awaitable

from pydantic import BaseModel, Field

from app.services.streaming.asr_stream import StreamingASR
from app.services.streaming.soe_stream import StreamingSOE
from app.services.streaming.audio_buffer import AudioBuffer
from app.schemas.streaming import StreamConfig

logger = logging.getLogger(__name__)

# PCM 16kHz 16bit mono: 32000 bytes/sec
BYTES_PER_SEC = 32000

# Chinese punctuation for speech rate calculation
_ZH_PUNCTUATION = set(string.punctuation + '。，！？、；：""''（）【】《》…—')


class StreamResult(BaseModel):
    """Result from a streaming session."""
    session_id: str
    asr_result: Optional[dict] = None
    soe_result: Optional[dict] = None
    speech_text: str = ""
    scores_data: dict = Field(default_factory=dict)
    audio_url: Optional[str] = None
    word_info_list: list = Field(default_factory=list)
    low_score_words: list = Field(default_factory=list)
    statistics_data: dict = Field(default_factory=dict)
    speech_rate: Optional[float] = None
    audio_duration: Optional[float] = None
    eval_type: str = "basic_evaluation"


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

        # Background tasks for forwarding intermediate results
        self._asr_consumer_task: Optional[asyncio.Task] = None
        self._soe_consumer_task: Optional[asyncio.Task] = None

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
                word_info=self.config.word_info,
                enable_timestamps=self.config.enable_timestamps,
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

        # Start background consumers for real-time callbacks
        if self._on_asr_partial and self._asr:
            self._asr_consumer_task = asyncio.create_task(self._consume_asr_results())
        if self._on_soe_intermediate and self._soe:
            self._soe_consumer_task = asyncio.create_task(self._consume_soe_results())

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

    async def _consume_asr_results(self) -> None:
        """Background task: consume ASR queue and forward partial results to callback."""
        try:
            async for event in self._asr.get_results():
                if event["type"] == "partial":
                    # Extract text from ASR partial result
                    result = event.get("data", {})
                    if isinstance(result, dict):
                        result_data = result.get("result", {})
                        text = ""
                        if isinstance(result_data, dict):
                            text = result_data.get("voice_text_str", "")
                        if text:
                            try:
                                await self._on_asr_partial({"data": {"text": text}})
                            except Exception as e:
                                logger.error(f"ASR partial callback error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ASR consumer error: {e}")

    async def _consume_soe_results(self) -> None:
        """Background task: consume SOE queue and forward intermediate results to callback."""
        try:
            async for event in self._soe.get_intermediate_results():
                if event["type"] == "intermediate":
                    # Extract scores from SOE intermediate result
                    data = event.get("data", {})
                    if isinstance(data, dict):
                        # Parse intermediate SOE result for scores
                        pron = data.get("pronunciation_assessment", {})
                        if isinstance(pron, dict):
                            scores = {
                                "pronunciation_accuracy": pron.get("accuracy", 0),
                                "pronunciation_fluency": pron.get("fluency", 0),
                                "pronunciation_completion": pron.get("completeness", 0),
                                "suggested_score": pron.get("overall_score", 0),
                            }
                            try:
                                await self._on_soe_intermediate({"data": {"scores": scores}})
                            except Exception as e:
                                logger.error(f"SOE intermediate callback error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SOE consumer error: {e}")

    async def finish(self) -> StreamResult:
        """
        Finish the session and return results.

        Returns:
            StreamResult with ASR, SOE results, and derived data
        """
        if self._finished:
            raise RuntimeError("Session already finished")

        logger.info(f"Streaming session {self.session_id} finishing")

        # Cancel background consumers
        for task in (self._asr_consumer_task, self._soe_consumer_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Mark buffer as final
        self._buffer.set_final()

        # Stop ASR and get final result
        asr_result = None
        speech_text = ""
        word_info_list = []
        if self._asr:
            try:
                asr_result = await self._asr.stop_recognition()
                if asr_result and "error" not in asr_result:
                    speech_text = asr_result.get("accumulated_text", "")
                    # Extract word-level timestamps from ASR result
                    result_data = asr_result.get("result", {})
                    word_info_list = result_data.get("WordList", [])
                    if not word_info_list:
                        word_info_list = asr_result.get("word_info_list", [])
            except Exception as e:
                logger.error(f"ASR stop error: {e}")

        # Stop SOE and get final result
        soe_result = None
        scores_data = {}
        low_score_words = []
        statistics_data = {}
        if self._soe:
            try:
                soe_result = await self._soe.stop_evaluation()
                if soe_result and "error" not in soe_result:
                    from app.services.tencent.soe import SOEService
                    soe_service = SOEService()
                    parsed = soe_service.parse_evaluation_result(soe_result)
                    scores_data = parsed.get("scores", {})
                    low_score_words = parsed.get("low_score_words", [])
                    statistics_data = parsed.get("statistics", {})
            except Exception as e:
                logger.error(f"SOE stop error: {e}")

        # Get audio buffer (once, reused for duration + S3 upload)
        audio_data = await self._buffer.get_full_buffer()

        # Calculate audio duration from buffer size
        audio_duration = None
        if audio_data:
            audio_duration = round(len(audio_data) / BYTES_PER_SEC, 2)

        # Calculate speech rate (chars/min for Chinese, words/min for English)
        speech_rate = None
        if audio_duration and audio_duration > 0 and speech_text:
            if self.config.language == "zh":
                char_count = len([c for c in speech_text if c not in _ZH_PUNCTUATION and not c.isspace()])
            else:
                char_count = len(speech_text.split())
            if char_count > 0:
                speech_rate = round(char_count / (audio_duration / 60), 1)

        # Upload audio to S3
        audio_url = None
        try:
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
            word_info_list=word_info_list,
            low_score_words=low_score_words,
            statistics_data=statistics_data,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            eval_type=self.config.eval_type,
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
