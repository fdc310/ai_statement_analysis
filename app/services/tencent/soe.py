"""
Tencent Cloud SOE (Smart Oral Evaluation) service using WebSocket SDK.
Supports fallback: recording mode -> streaming mode -> error.
"""
import sys
import os
import threading
import time
import logging
from typing import Optional

from app.core.thread_pool import ThreadPool

# Add SDK path to sys.path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from soe.speaking_assessment import SpeakingAssessment, SpeakingAssessmentListener

from app.core.config import settings
from app.services.tencent.audio import convert_audio_to_wav

logger = logging.getLogger(__name__)

# 16kHz 16bit mono WAV: 1秒 = 32000字节
BYTES_PER_SEC = 32000
CHUNK_DURATION = 0.2  # 200ms per chunk for streaming mode
STREAM_CHUNK_SIZE = int(BYTES_PER_SEC * CHUNK_DURATION)  # 6400 bytes


class SOEService:
    """Tencent Cloud SOE service for speech evaluation using WebSocket SDK."""

    def __init__(
        self,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        appid: Optional[str] = None
    ):
        self.secret_id = secret_id or settings.tencent_secret_id
        self.secret_key = secret_key or settings.tencent_secret_key
        self.appid = appid or settings.tencent_appid

    async def convert_audio(self, audio_data: bytes) -> bytes:
        """Convert audio to standard format: 16kHz, 16bit, mono, WAV."""
        return await convert_audio_to_wav(
            audio_data,
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )

    def _wait_for_open(self, recognizer, timeout: float = 10.0) -> Optional[str]:
        """Wait for WebSocket to open. Returns error string or None on success."""
        wait_time = 0
        while recognizer.status == 1:  # STARTED
            time.sleep(0.1)
            wait_time += 0.1
            if wait_time > timeout:
                return f"Connection timeout after {wait_time:.1f}s"
        if recognizer.status != 2:  # OPENED
            return f"Connection not opened, status={recognizer.status}"
        return None

    def _build_recognizer(self, audio_data: bytes, ref_text: str, eval_mode: int,
                          score_coeff: float, server_type: int, rec_mode: int,
                          listener: SpeakingAssessmentListener) -> SpeakingAssessment:
        """Create and configure a SpeakingAssessment recognizer."""
        engine_type = "16k_zh" if server_type == 0 else "16k_en"
        recognizer = SpeakingAssessment(
            self.appid, Credential(self.secret_id, self.secret_key), engine_type, listener
        )
        recognizer.set_text_mode(0)
        recognizer.set_ref_text(ref_text)
        recognizer.set_eval_mode(eval_mode)
        recognizer.set_keyword("")
        recognizer.set_sentence_info_enabled(1)
        recognizer.set_voice_format(1)  # 1=wav format
        recognizer.set_rec_mode(rec_mode)
        recognizer.score_coeff = score_coeff
        return recognizer

    def _send_audio(self, recognizer, audio_data: bytes, chunk_size: int, interval: float) -> Optional[str]:
        """Send audio data. Returns error string or None on success."""
        total_size = len(audio_data)
        send_start = time.time()
        for i in range(0, total_size, chunk_size):
            if recognizer.status != 2:  # OPENED
                return f"Connection lost during send, status={recognizer.status}, sent={i}/{total_size}"
            chunk = audio_data[i:i + chunk_size]
            recognizer.write(chunk)
            sent = min(i + chunk_size, total_size)
            if sent == total_size or sent % (BYTES_PER_SEC * 3) == 0:
                logger.info(f"SOE: sent {sent}/{total_size} bytes ({sent * 100 // total_size}%)")
            if interval > 0:
                time.sleep(interval)
        send_elapsed = time.time() - send_start
        logger.info(f"SOE: audio data sent in {send_elapsed:.2f}s")
        return None

    def _wait_result(self, completed_event: threading.Event, result_holder: dict,
                     recognizer, timeout: float = 60.0) -> dict:
        """Wait for recognition result. Returns SDK response dict or error dict."""
        if not completed_event.wait(timeout=timeout):
            return {"error": "Evaluation timeout"}
        if result_holder["error"]:
            return {"error": result_holder["error"]}
        return result_holder["result"] or {}

    def _do_evaluate(self, audio_data: bytes, ref_text: str, eval_mode: int,
                     score_coeff: float, server_type: int, rec_mode: int,
                     chunk_size: int, send_interval: float, mode_name: str) -> dict:
        """Run a single evaluation attempt with the given rec_mode."""
        completed_event = threading.Event()
        result_holder = {"result": None, "error": None}

        class EvalListener(SpeakingAssessmentListener):
            def on_recognition_start(self, response):
                logger.info(f"SOE [{mode_name}]: recognition started, voice_id={response.get('voice_id')}")

            def on_intermediate_result(self, response):
                result_holder["result"] = response

            def on_recognition_complete(self, response):
                logger.info(f"SOE [{mode_name}]: recognition complete")
                result_holder["result"] = response
                completed_event.set()

            def on_fail(self, response):
                logger.error(f"SOE [{mode_name}]: failed, code={response.get('code')}, message={response.get('message')}")
                result_holder["error"] = response
                completed_event.set()

        listener = EvalListener()
        recognizer = self._build_recognizer(audio_data, ref_text, eval_mode, score_coeff, server_type, rec_mode, listener)

        logger.info(f"SOE [{mode_name}]: starting, audio_size={len(audio_data)} bytes, rec_mode={rec_mode}")

        try:
            recognizer.start()

            err = self._wait_for_open(recognizer)
            if err:
                logger.error(f"SOE [{mode_name}]: {err}")
                return {"error": err}

            err = self._send_audio(recognizer, audio_data, chunk_size, send_interval)
            if err:
                logger.error(f"SOE [{mode_name}]: {err}")
                return {"error": err}

            recognizer.stop()
            logger.info(f"SOE [{mode_name}]: stop() called, waiting for result...")

            return self._wait_result(completed_event, result_holder, recognizer)

        except Exception as e:
            logger.exception(f"SOE [{mode_name}]: exception: {e}")
            try:
                recognizer.ws.close()
            except:
                pass
            return {"error": str(e)}

    def _sync_evaluate(self, audio_data: bytes, ref_text: str, eval_mode: int,
                       score_coeff: float, server_type: int) -> dict:
        """
        Evaluate with fallback: recording mode -> streaming mode -> error.
        - Recording mode (rec_mode=1): single packet, fast but fails on large audio.
        - Streaming mode (rec_mode=0): chunked send with rate limiting, slower but reliable.
        """
        total_size = len(audio_data)
        logger.info(f"SOE: starting evaluation, audio_size={total_size} bytes, eval_mode={eval_mode}")

        # Attempt 1: Recording mode (rec_mode=1), send all at once
        logger.info("SOE: attempt 1 - recording mode (rec_mode=1)")
        result = self._do_evaluate(
            audio_data, ref_text, eval_mode, score_coeff, server_type,
            rec_mode=1,
            chunk_size=total_size,  # send all at once
            send_interval=0,
            mode_name="recording"
        )

        # Check if recording mode succeeded (SDK response has no "error" key on success)
        if "error" not in result:
            return result

        err = result["error"]
        error_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.warning(f"SOE: recording mode failed ({error_msg}), falling back to streaming mode")

        # Attempt 2: Streaming mode (rec_mode=0), chunked with rate limiting
        logger.info("SOE: attempt 2 - streaming mode (rec_mode=0)")
        result = self._do_evaluate(
            audio_data, ref_text, eval_mode, score_coeff, server_type,
            rec_mode=0,
            chunk_size=STREAM_CHUNK_SIZE,
            send_interval=CHUNK_DURATION,
            mode_name="streaming"
        )

        if "error" not in result:
            return result

        # Both modes failed
        err = result["error"]
        error_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.error(f"SOE: all attempts failed, last error: {error_msg}")
        return result

    async def evaluate_audio(
        self,
        audio_data: bytes,
        ref_text: str = "",
        eval_mode: int = 0,
        score_coeff: float = 1.0,
        server_type: int = 0
    ) -> dict:
        """
        Evaluate speech audio using WebSocket SDK.

        Args:
            audio_data: Audio file bytes (any format supported by ffmpeg).
            ref_text: Reference text for evaluation.
            eval_mode: Evaluation mode (0=word, 1=sentence, 2=paragraph, 3=free speech)
            score_coeff: Score coefficient (1.0-4.0), 1.0 for children, 4.0 for adults
            server_type: 0=Chinese, 1=English

        Returns:
            Evaluation result dict with scores and details.
        """
        # Convert audio to standard format
        audio_data = await self.convert_audio(audio_data)

        # Run sync evaluation in centralized thread pool
        result = await ThreadPool.run(
            self._sync_evaluate,
            audio_data,
            ref_text,
            eval_mode,
            score_coeff,
            server_type
        )

        if "error" in result:
            err = result["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise Exception(msg)

        return self.parse_evaluation_result(result)

    def parse_evaluation_result(self, result: dict) -> dict:
        """Parse and structure the evaluation result from WebSocket SDK."""
        soe_result = result.get("result", {})

        pron_accuracy = soe_result.get("PronAccuracy", 0)
        pron_fluency = soe_result.get("PronFluency", 0)
        pron_completion = soe_result.get("PronCompletion", 0)
        suggested_score = soe_result.get("SuggestedScore", 0)

        words = []
        low_score_words = []

        for word_info in soe_result.get("Words", []):
            word = word_info.get("Word", "")
            accuracy = word_info.get("PronAccuracy", 0)
            fluency = word_info.get("PronFluency", 0)

            word_data = {
                "word": word,
                "reference_word": word_info.get("ReferenceWord", ""),
                "pron_accuracy": round(accuracy, 2),
                "pron_fluency": round(fluency, 2) if isinstance(fluency, float) and fluency <= 1 else round(fluency, 2),
                "phone_infos": word_info.get("PhoneInfos", [])
            }
            words.append(word_data)

            if accuracy < 90:
                low_score_words.append({
                    "word": word,
                    "accuracy": round(accuracy, 2),
                    "fluency": round(fluency * 100, 2) if fluency <= 1 else round(fluency, 2)
                })

        sentence_infos = []
        for sent in soe_result.get("SentenceInfoSet", []):
            sentence_infos.append({
                "text": sent.get("SentenceId", ""),
                "pron_accuracy": sent.get("PronAccuracy", 0),
                "pron_fluency": sent.get("PronFluency", 0),
                "pron_completion": sent.get("PronCompletion", 0),
                "suggested_score": sent.get("SuggestedScore", 0)
            })

        total_words = len(words)
        avg_accuracy = sum(w["pron_accuracy"] for w in words) / total_words if total_words > 0 else 0
        fluency_score = pron_fluency * 100 if pron_fluency <= 1 else pron_fluency

        return {
            "session_id": result.get("voice_id", ""),
            "scores": {
                "pronunciation_accuracy": round(pron_accuracy, 2),
                "pronunciation_fluency": round(fluency_score, 2),
                "pronunciation_completion": round(pron_completion, 2) if pron_completion >= 0 else -1,
                "suggested_score": round(suggested_score, 2),
                "overall_score": round(suggested_score, 2)
            },
            "statistics": {
                "total_words": total_words,
                "average_accuracy": round(avg_accuracy, 2),
                "low_score_count": len(low_score_words)
            },
            "words": words,
            "low_score_words": low_score_words,
            "sentences": sentence_infos,
            "raw_response": result
        }

    async def evaluate_free_speech(
        self,
        audio_data: bytes,
        server_type: int = 0
    ) -> dict:
        """Evaluate free speech without reference text."""
        return await self.evaluate_audio(
            audio_data=audio_data,
            ref_text="",
            eval_mode=3,
            server_type=server_type
        )


# Singleton instance
soe_service = SOEService()
