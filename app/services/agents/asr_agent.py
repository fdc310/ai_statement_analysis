"""
ASR Agent - Speech-to-text conversion.
Wraps the existing ASRService for use in the multi-agent pipeline.
"""
import logging
from typing import Optional

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services.tencent.asr import asr_service

logger = logging.getLogger(__name__)


class ASRAgent(BaseAgent):
    """Agent for speech-to-text conversion using Tencent Cloud ASR."""

    @property
    def name(self) -> str:
        return "asr"

    async def execute(self, context: EvaluationContext) -> AgentResult:
        """
        Run ASR on audio data and populate context with results.

        Reads from context:
            - audio_data: Raw audio bytes
            - language: Language code (zh/en)

        Writes to context:
            - speech_text: Transcribed text
            - word_info_list: Word-level timestamps
        """
        audio_data = context.audio_data
        if not audio_data:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="No audio data provided"
            )

        # Determine engine type from language
        engine_type = "16k_zh" if context.language == "zh" else "16k_en"

        # Determine word_info level based on request
        word_info = context.request.get("word_info", 1)  # Default: word timestamps

        result = await asr_service.recognize_audio(
            audio_data=audio_data,
            engine_type=engine_type,
            word_info=word_info
        )

        # Populate context
        context.speech_text = result.get("text", "")
        context.word_info_list = result.get("word_info_list", [])

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "text": context.speech_text,
                "word_count": len(context.word_info_list),
            }
        )
