"""
Standalone agent endpoints.
Each agent can be called independently for specific tasks.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form

from app.schemas.base import BaseResponse
from app.services.agents.orchestrator import orchestrator
from app.services.agents.asr_agent import ASRAgent
from app.services.agents.soe_agent import SOEAgent
from app.services.agents.content_agent import ContentAnalysisAgent
from app.services.agents.fluency_agent import FluencyAnalysisAgent
from app.services.agents.report_agent import ReportAgent
from app.services.agents.base_agent import EvaluationContext
from app.services.tasks.manager import task_manager
from app.services.tasks.executor import task_executor
from app.services.tencent.asr import asr_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/asr", response_model=BaseResponse)
async def run_asr_agent(
    audio_url: Optional[str] = Form(None),
    language: str = Form(default="zh"),
    word_info: int = Form(default=1),
    callback_url: Optional[str] = Form(None),
):
    """Run ASR agent independently on audio input."""
    msg_id = str(uuid.uuid4())

    try:
        # Get audio data
        if audio_url:
            audio_data = await asr_service.download_audio(audio_url)
        else:
            raise HTTPException(status_code=400, detail="audio_url is required")

        # Create context
        context = EvaluationContext({
            "language": language,
            "word_info": word_info,
        })
        context.audio_data = audio_data

        # Run agent
        agent = ASRAgent()
        result = await agent._run(context)

        if callback_url:
            from app.services.tasks.callback import callback_dispatcher
            await callback_dispatcher.send_success(callback_url, msg_id, result.data)

        return BaseResponse(
            success=result.success,
            message="ASR completed" if result.success else result.error,
            message_id=msg_id,
            error=result.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        return BaseResponse(
            success=False,
            message="ASR failed",
            message_id=msg_id,
            error=str(e),
        )


@router.post("/soe", response_model=BaseResponse)
async def run_soe_agent(
    audio_url: Optional[str] = Form(None),
    ref_text: str = Form(default=""),
    eval_mode: int = Form(default=3),
    score_coeff: float = Form(default=1.0),
    server_type: int = Form(default=0),
    callback_url: Optional[str] = Form(None),
):
    """Run SOE agent independently on audio input."""
    msg_id = str(uuid.uuid4())

    try:
        # Get audio data
        if audio_url:
            audio_data = await asr_service.download_audio(audio_url)
        else:
            raise HTTPException(status_code=400, detail="audio_url is required")

        # Create context
        context = EvaluationContext({
            "ref_text": ref_text,
            "eval_mode": eval_mode,
            "score_coeff": score_coeff,
            "server_type": server_type,
        })
        context.audio_data = audio_data

        # Run agent
        agent = SOEAgent()
        result = await agent._run(context)

        if callback_url:
            from app.services.tasks.callback import callback_dispatcher
            await callback_dispatcher.send_success(callback_url, msg_id, result.data)

        return BaseResponse(
            success=result.success,
            message="SOE completed" if result.success else result.error,
            message_id=msg_id,
            error=result.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        return BaseResponse(
            success=False,
            message="SOE failed",
            message_id=msg_id,
            error=str(e),
        )


@router.post("/content", response_model=BaseResponse)
async def run_content_agent(
    speech_text: str = Form(...),
    eval_type: str = Form(default="basic"),
    topic: str = Form(default=""),
    custom_prompt: str = Form(default=""),
    callback_url: Optional[str] = Form(None),
):
    """Run content analysis agent independently on text."""
    msg_id = str(uuid.uuid4())

    try:
        # Create context
        context = EvaluationContext({
            "eval_type": eval_type,
            "topic": topic,
            "custom_prompt": custom_prompt,
        })
        context.speech_text = speech_text

        # Run agent
        agent = ContentAnalysisAgent()
        result = await agent._run(context)

        if callback_url:
            from app.services.tasks.callback import callback_dispatcher
            await callback_dispatcher.send_success(callback_url, msg_id, result.data)

        return BaseResponse(
            success=result.success,
            message="Content analysis completed" if result.success else result.error,
            message_id=msg_id,
            error=result.error,
        )

    except Exception as e:
        return BaseResponse(
            success=False,
            message="Content analysis failed",
            message_id=msg_id,
            error=str(e),
        )


@router.post("/fluency", response_model=BaseResponse)
async def run_fluency_agent(
    speech_text: str = Form(...),
    word_info_list: str = Form(default="[]"),
    audio_duration: float = Form(default=0),
    callback_url: Optional[str] = Form(None),
):
    """Run fluency analysis agent independently on text."""
    import json
    msg_id = str(uuid.uuid4())

    try:
        # Parse word_info_list
        try:
            words = json.loads(word_info_list)
        except json.JSONDecodeError:
            words = []

        # Create context
        context = EvaluationContext({})
        context.speech_text = speech_text
        context.word_info_list = words
        context.audio_duration = audio_duration

        # Run agent
        agent = FluencyAnalysisAgent()
        result = await agent._run(context)

        if callback_url:
            from app.services.tasks.callback import callback_dispatcher
            await callback_dispatcher.send_success(callback_url, msg_id, result.data)

        return BaseResponse(
            success=result.success,
            message="Fluency analysis completed" if result.success else result.error,
            message_id=msg_id,
            error=result.error,
        )

    except Exception as e:
        return BaseResponse(
            success=False,
            message="Fluency analysis failed",
            message_id=msg_id,
            error=str(e),
        )


@router.post("/report", response_model=BaseResponse)
async def run_report_agent(
    speech_text: str = Form(...),
    scores_data: str = Form(default="{}"),
    low_score_words: str = Form(default="[]"),
    statistics_data: str = Form(default="{}"),
    speech_rate: float = Form(default=0),
    report_format: str = Form(default="markdown"),
    custom_prompt: str = Form(default=""),
    callback_url: Optional[str] = Form(None),
):
    """Run report generation agent independently."""
    import json
    msg_id = str(uuid.uuid4())

    try:
        # Parse JSON fields
        try:
            scores = json.loads(scores_data)
        except json.JSONDecodeError:
            scores = {}
        try:
            low_words = json.loads(low_score_words)
        except json.JSONDecodeError:
            low_words = []
        try:
            stats = json.loads(statistics_data)
        except json.JSONDecodeError:
            stats = {}

        # Create context
        context = EvaluationContext({
            "report_format": report_format,
            "custom_prompt": custom_prompt,
        })
        context.speech_text = speech_text
        context.scores_data = scores
        context.low_score_words = low_words
        context.statistics_data = stats
        context.speech_rate = speech_rate

        # Run agent
        agent = ReportAgent()
        result = await agent._run(context)

        if callback_url:
            from app.services.tasks.callback import callback_dispatcher
            await callback_dispatcher.send_success(callback_url, msg_id, result.data)

        return BaseResponse(
            success=result.success,
            message="Report generated" if result.success else result.error,
            message_id=msg_id,
            error=result.error,
        )

    except Exception as e:
        return BaseResponse(
            success=False,
            message="Report generation failed",
            message_id=msg_id,
            error=str(e),
        )
