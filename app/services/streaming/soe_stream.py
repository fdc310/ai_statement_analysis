"""
Real-time SOE via SpeakingAssessment WebSocket (native streaming mode).
Uses rec_mode=0 (streaming) with 200ms chunk rate limiting.
"""
import asyncio
import sys
import os
import logging
import threading
import time
from typing import Optional, AsyncGenerator

# Add SDK path to sys.path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from soe.speaking_assessment import SpeakingAssessment, SpeakingAssessmentListener

from app.core.config import settings
from app.core.thread_pool import ThreadPool

logger = logging.getLogger(__name__)

# 16kHz 16bit mono WAV: 1 second = 32000 bytes
BYTES_PER_SEC = 32000
CHUNK_DURATION = 0.2  # 200ms per chunk
STREAM_CHUNK_SIZE = int(BYTES_PER_SEC * CHUNK_DURATION)  # 6400 bytes


class StreamingSOE:
    """
    Real-time SOE using Tencent Cloud SpeakingAssessment WebSocket.

    Uses native streaming mode (rec_mode=0) with rate limiting.
    """

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        appid: Optional[str] = None
    ):
        self.secret_id = secret_id or settings.tencent_secret_id
        self.secret_key = secret_key or settings.tencent_secret_key
        self.appid = appid or settings.tencent_appid
        self._recognizer: Optional[SpeakingAssessment] = None
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._completed = threading.Event()
        self._last_send_time: float = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start_evaluation(
        self,
        ref_text: str = "",
        eval_mode: int = 3,
        score_coeff: float = 1.0,
        server_type: int = 0
    ) -> str:
        """
        Start a real-time evaluation session.

        Args:
            ref_text: Reference text for evaluation
            eval_mode: Evaluation mode (0-8)
            score_coeff: Score coefficient (1.0-4.0)
            server_type: 0=Chinese, 1=English
        """
        self._completed.clear()
        self._result_queue = asyncio.Queue()
        self._last_send_time = 0
        self._loop = asyncio.get_running_loop()

        engine_type = "16k_zh" if server_type == 0 else "16k_en"
        credential = Credential(self.secret_id, self.secret_key)

        loop = self._loop

        class EvalListener(SpeakingAssessmentListener):
            def __init__(self, queue, completed_event):
                self._queue = queue
                self._completed = completed_event

            def on_recognition_start(self, response):
                logger.debug("SOE Stream: recognition started")

            def on_intermediate_result(self, response):
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "intermediate", "data": response}
                    )
                except RuntimeError:
                    pass

            def on_recognition_complete(self, response):
                logger.info("SOE Stream: recognition complete")
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "complete", "data": response}
                    )
                    loop.call_soon_threadsafe(self._completed.set)
                except RuntimeError:
                    pass

            def on_fail(self, response):
                logger.error(f"SOE Stream: failed, {response}")
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "error", "data": response}
                    )
                    loop.call_soon_threadsafe(self._completed.set)
                except RuntimeError:
                    pass

        listener = EvalListener(self._result_queue, self._completed)

        def create_recognizer():
            self._recognizer = SpeakingAssessment(
                self.appid, credential, engine_type, listener
            )
            self._recognizer.set_text_mode(0)
            self._recognizer.set_ref_text(ref_text)
            self._recognizer.set_eval_mode(eval_mode)
            self._recognizer.set_keyword("")
            self._recognizer.set_sentence_info_enabled(1)
            self._recognizer.set_voice_format(1)  # WAV format
            self._recognizer.set_rec_mode(0)  # Streaming mode
            self._recognizer.score_coeff = score_coeff
            self._recognizer.start()

        await ThreadPool.run(create_recognizer)
        logger.info("SOE Stream: evaluation started (streaming mode)")
        return "started"

    async def feed_audio(self, wav_chunk: bytes) -> None:
        """
        Feed audio data to the evaluator with rate limiting.

        Args:
            wav_chunk: WAV audio chunk (16kHz 16bit mono)
        """
        if not self._recognizer:
            raise RuntimeError("Evaluation not started")

        # Rate limiting: ensure 200ms between sends
        now = time.time()
        elapsed = now - self._last_send_time
        if elapsed < CHUNK_DURATION:
            await asyncio.sleep(CHUNK_DURATION - elapsed)

        def write_chunk():
            self._recognizer.write(wav_chunk)

        await ThreadPool.run(write_chunk)
        self._last_send_time = time.time()

    async def stop_evaluation(self) -> dict:
        """Stop evaluation and wait for final result."""
        if not self._recognizer:
            return {"error": "Evaluation not started"}

        def stop():
            self._recognizer.stop()

        await ThreadPool.run(stop)

        # Wait for completion
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._completed.wait, timeout=60),
                timeout=60
            )
        except asyncio.TimeoutError:
            logger.warning("SOE Stream: timeout waiting for completion")

        # Drain results
        final_result = {}
        while not self._result_queue.empty():
            event = self._result_queue.get_nowait()
            if event["type"] == "complete":
                final_result = event["data"]
            elif event["type"] == "error":
                return {"error": event["data"]}

        return final_result

    async def get_intermediate_results(self) -> AsyncGenerator[dict, None]:
        """Yield intermediate evaluation results as they arrive."""
        while not self._completed.is_set() or not self._result_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=1.0
                )
                yield event
                if event["type"] in ("complete", "error"):
                    return
            except asyncio.TimeoutError:
                continue
