"""
Real-time ASR via SpeechRecognizer WebSocket.
Wraps the bundled Tencent Cloud Speech SDK SpeechRecognizer.
"""
import asyncio
import sys
import os
import logging
import threading
from typing import Optional, AsyncGenerator

# Add SDK path to sys.path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from asr.speech_recognizer import SpeechRecognizer, SpeechRecognitionListener

from app.core.config import settings
from app.core.thread_pool import ThreadPool

logger = logging.getLogger(__name__)


class StreamingASR:
    """
    Real-time ASR using Tencent Cloud SpeechRecognizer WebSocket.

    The SDK uses synchronous websocket-client, so all operations
    are wrapped in threads via ThreadPool.
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
        self._recognizer: Optional[SpeechRecognizer] = None
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._completed = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._accumulated_text = [""]  # [text] — list for mutability across threads

    async def start_recognition(
        self,
        engine_type: str = "16k_zh",
        word_info: int = 1
    ) -> str:
        """
        Start a real-time recognition session.

        Args:
            engine_type: Recognition engine type
            word_info: Word info level (0=none, 1=words, 2=words+punctuation)

        Returns:
            Session ID
        """
        self._completed.clear()
        self._result_queue = asyncio.Queue()
        self._accumulated_text = [""]
        self._loop = asyncio.get_running_loop()

        credential = Credential(self.secret_id, self.secret_key)

        loop = self._loop

        class StreamListener(SpeechRecognitionListener):
            def __init__(self, queue, completed_event, accumulated_text_ref):
                self._queue = queue
                self._completed = completed_event
                self._accumulated_text = accumulated_text_ref

            def on_sentence_begin(self, response):
                logger.debug(f"ASR Stream: sentence begin")
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "sentence_begin", "data": response}
                    )
                except RuntimeError:
                    pass

            def on_recognition_result_change(self, response):
                logger.debug(f"ASR Stream: result change")
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "partial", "data": response}
                    )
                except RuntimeError:
                    pass

            def on_sentence_end(self, response):
                logger.debug(f"ASR Stream: sentence end")
                # Accumulate text from sentence end result
                result = response.get("result", {})
                if isinstance(result, dict):
                    text = result.get("voice_text_str", "")
                    if text:
                        self._accumulated_text[0] += text
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "sentence_end", "data": response}
                    )
                except RuntimeError:
                    pass

            def on_recognition_complete(self, response):
                logger.info("ASR Stream: recognition complete")
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "complete", "data": response}
                    )
                    loop.call_soon_threadsafe(self._completed.set)
                except RuntimeError:
                    pass

            def on_fail(self, response):
                logger.error(f"ASR Stream: failed, {response}")
                try:
                    loop.call_soon_threadsafe(
                        self._queue.put_nowait,
                        {"type": "error", "data": response}
                    )
                    loop.call_soon_threadsafe(self._completed.set)
                except RuntimeError:
                    pass

        listener = StreamListener(self._result_queue, self._completed, self._accumulated_text)

        def create_recognizer():
            self._recognizer = SpeechRecognizer(
                self.appid, credential, engine_type, listener
            )
            self._recognizer.set_voice_format(1)  # raw PCM (linear16)
            self._recognizer.set_word_info(word_info)
            self._recognizer.set_filter_dirty(0)
            self._recognizer.set_filter_modal(0)
            self._recognizer.set_filter_punc(0)
            self._recognizer.set_convert_num_mode(1)
            self._recognizer.start()

        await ThreadPool.run(create_recognizer)
        logger.info("ASR Stream: recognition started")
        return "started"

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        """
        Feed audio data to the recognizer.

        Args:
            pcm_chunk: Raw PCM 16kHz 16bit mono audio bytes
        """
        if not self._recognizer:
            raise RuntimeError("Recognition not started")

        def write_chunk():
            self._recognizer.write(pcm_chunk)

        await ThreadPool.run(write_chunk)

    async def stop_recognition(self) -> dict:
        """Stop recognition and wait for final result."""
        if not self._recognizer:
            return {"error": "Recognition not started"}

        def stop():
            self._recognizer.stop()

        await ThreadPool.run(stop)

        # Wait for completion (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._completed.wait, timeout=30),
                timeout=30
            )
        except asyncio.TimeoutError:
            logger.warning("ASR Stream: timeout waiting for completion")

        # Drain remaining results
        final_result = {}
        while not self._result_queue.empty():
            event = self._result_queue.get_nowait()
            if event["type"] == "complete":
                final_result = event["data"]
            elif event["type"] == "error":
                return {"error": event["data"]}

        # Inject accumulated text (on_recognition_complete has no text)
        final_result["accumulated_text"] = self._accumulated_text[0]
        return final_result

    async def get_results(self) -> AsyncGenerator[dict, None]:
        """Yield recognition results as they arrive."""
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
