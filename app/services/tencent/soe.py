"""
Tencent Cloud SOE (Smart Oral Evaluation) service using WebSocket SDK.
"""
import asyncio
import sys
import os
from typing import Optional

# Add SDK path to sys.path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from soe.speaking_assessment import SpeakingAssessment, SpeakingAssessmentListener

from app.core.config import settings
from app.services.tencent.audio import convert_audio_to_wav


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

    def _sync_evaluate(
        self,
        audio_data: bytes,
        ref_text: str,
        eval_mode: int,
        score_coeff: float,
        server_type: int
    ) -> dict:
        """Synchronous evaluation using WebSocket SDK."""
        import threading
        import time

        # Use threading.Event instead of asyncio.Event for sync context
        completed_event = threading.Event()
        result_holder = {"result": None, "error": None}

        class SyncListener(SpeakingAssessmentListener):
            def on_recognition_start(self, response):
                pass

            def on_intermediate_result(self, response):
                result_holder["result"] = response

            def on_recognition_complete(self, response):
                result_holder["result"] = response
                completed_event.set()

            def on_fail(self, response):
                result_holder["error"] = response
                completed_event.set()

        listener = SyncListener()
        credential = Credential(self.secret_id, self.secret_key)

        # Engine type: 16k_zh for Chinese, 16k_en for English
        engine_type = "16k_zh" if server_type == 0 else "16k_en"

        recognizer = SpeakingAssessment(
            self.appid, credential, engine_type, listener
        )

        # Configure evaluation parameters based on API docs
        recognizer.set_text_mode(0)  # 0=普通文本
        recognizer.set_ref_text(ref_text)
        recognizer.set_eval_mode(eval_mode)  # 3=自由说模式
        recognizer.set_keyword("")
        recognizer.set_sentence_info_enabled(1)  # 输出断句中间结果
        recognizer.set_voice_format(1)  # 1=wav format
        recognizer.set_rec_mode(1)  # 1=录音识别模式，一次性发送音频
        recognizer.score_coeff = score_coeff  # 评价苛刻指数 1.0-4.0

        try:
            # Start WebSocket connection
            recognizer.start()

            # Wait for connection to be opened (status changes from STARTED to OPENED)
            wait_time = 0
            while recognizer.status == 1:  # STARTED = 1
                time.sleep(0.1)
                wait_time += 0.1
                if wait_time > 10:
                    try:
                        recognizer.ws.close()
                    except:
                        pass
                    return {"error": "Connection timeout"}

            # Check if connection is open
            if recognizer.status != 2:  # OPENED = 2
                try:
                    recognizer.ws.close()
                except:
                    pass
                return {"error": f"Connection failed, status: {recognizer.status}"}

            # Send all audio data at once in recording mode
            recognizer.write(audio_data)

            # Call stop() to send end message - SDK's stop() sends {"type": "end"}
            recognizer.stop()

            # Wait for result with timeout
            if not completed_event.wait(timeout=60):
                return {"error": "Evaluation timeout"}

        except Exception as e:
            try:
                recognizer.ws.close()
            except:
                pass
            return {"error": str(e)}

        if result_holder["error"]:
            return {"error": result_holder["error"].get("message", "Unknown error")}

        return result_holder["result"] or {}

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

        # Run sync evaluation in thread pool
        result = await asyncio.to_thread(
            self._sync_evaluate,
            audio_data,
            ref_text,
            eval_mode,
            score_coeff,
            server_type
        )

        if "error" in result:
            raise Exception(result["error"])

        return self._parse_evaluation_result(result)

    def _parse_evaluation_result(self, result: dict) -> dict:
        """Parse and structure the evaluation result from WebSocket SDK."""
        # The result structure from WebSocket SDK is different
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
