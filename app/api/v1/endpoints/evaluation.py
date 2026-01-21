# -*- coding: utf-8 -*-
"""
Speech evaluation endpoints with async callback support and AES signature authentication.
Signature is passed via X-Signature header.

These endpoints accept evaluation tasks and return immediately with a message_id.
Results are sent to the callback_url when processing completes.
"""
import uuid
import asyncio
import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, BackgroundTasks
from typing import Optional

import httpx

from app.core.config import settings
from app.core.security import aes_service
from app.schemas.evaluation import (
    EvaluationRequest,
    EvaluationAcceptedResponse,
    EvaluationCallbackData,
    SpeechScores,
    EvaluationStatistics,
    WordScore,
    SignatureRequest,
    SignatureResponse
)
from app.services.tencent import asr_service, soe_service, hunyuan_service

router = APIRouter()


def verify_signature(signature: Optional[str]) -> None:
    """Verify AES signature from header, raise HTTPException if invalid."""
    if not signature:
        raise HTTPException(status_code=401, detail="Missing X-Signature header")

    success, message = aes_service.verify_signature(
        signature,
        max_age_seconds=settings.request_expire_seconds
    )
    if not success:
        raise HTTPException(status_code=401, detail=message)


async def send_callback(callback_url: str, data: EvaluationCallbackData) -> None:
    """Send evaluation results to callback URL."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                callback_url,
                json=data.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            print(f"[CALLBACK] Sent to {callback_url}, status: {response.status_code}")
    except Exception as e:
        print(f"[CALLBACK ERROR] Failed to send to {callback_url}: {str(e)}")


async def process_evaluation_task(
    message_id: str,
    callback_url: str,
    audio_data: bytes,
    language: str,
    ref_text: str,
    custom_prompt: Optional[str]
) -> None:
    """Background task to process evaluation and send callback."""
    try:
        # Determine engine type based on language
        engine_type = "16k_zh" if language == "zh" else "16k_en"
        server_type = 0 if language == "zh" else 1

        # Run ASR and SOE in parallel
        asr_result, soe_result = await asyncio.gather(
            asr_service.recognize_audio(audio_data, engine_type),
            soe_service.evaluate_audio(
                audio_data,
                ref_text or "",
                3 if not ref_text else 1,
                1.0,
                server_type
            )
        )

        # Extract results
        speech_text = asr_result.get("text", "")
        scores_data = soe_result.get("scores", {})
        low_score_words_data = soe_result.get("low_score_words", [])
        statistics_data = soe_result.get("statistics", {})

        speech_scores = SpeechScores(
            pronunciation_accuracy=scores_data.get("pronunciation_accuracy", 0),
            pronunciation_fluency=scores_data.get("pronunciation_fluency", 0),
            pronunciation_completion=scores_data.get("pronunciation_completion", 0),
            suggested_score=scores_data.get("suggested_score", 0),
            overall_score=scores_data.get("overall_score", 0)
        )

        statistics = EvaluationStatistics(
            total_words=statistics_data.get("total_words", 0),
            average_accuracy=statistics_data.get("average_accuracy", 0),
            low_score_count=statistics_data.get("low_score_count", 0)
        )

        low_score_words = [
            WordScore(
                word=w.get("word", ""),
                accuracy=w.get("accuracy", 0),
                fluency=w.get("fluency", 0)
            )
            for w in low_score_words_data
        ]

        # Generate AI evaluation report
        evaluation_report = await hunyuan_service.generate_evaluation(
            speech_text,
            scores_data,
            custom_prompt,
            low_score_words_data,
            statistics_data
        )

        # Send success callback
        callback_data = EvaluationCallbackData(
            message_id=message_id,
            success=True,
            message="Evaluation completed successfully",
            speech_text=speech_text,
            speech_scores=speech_scores,
            statistics=statistics,
            low_score_words=low_score_words,
            evaluation_report=evaluation_report
        )
        await send_callback(callback_url, callback_data)

    except Exception as e:
        # Send error callback
        callback_data = EvaluationCallbackData(
            message_id=message_id,
            success=False,
            message="Evaluation failed",
            error=str(e)
        )
        await send_callback(callback_url, callback_data)


@router.post("/signature", response_model=SignatureResponse)
async def generate_signature(request: SignatureRequest) -> SignatureResponse:
    """
    Generate AES signature for API authentication.

    Pass the AES key to generate a signature containing current timestamp.
    The signature should be included in the X-Signature header for other API calls.

    Request body:
    ```json
    {
        "aes_key": "your_aes_key"
    }
    ```

    Response:
    ```json
    {
        "success": true,
        "signature": "base64_encoded_aes_signature",
        "timestamp": 1234567890,
        "expires_in": 300
    }
    ```
    """
    try:
        # Verify the provided key matches configured key
        if request.aes_key != settings.aes_key:
            return SignatureResponse(
                success=False,
                error="Invalid AES key"
            )

        signature, timestamp = aes_service.generate_signature()

        return SignatureResponse(
            success=True,
            signature=signature,
            timestamp=timestamp,
            expires_in=settings.request_expire_seconds
        )
    except Exception as e:
        return SignatureResponse(
            success=False,
            error=str(e)
        )


@router.post("/analyze", response_model=EvaluationAcceptedResponse)
async def evaluate_speech(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> EvaluationAcceptedResponse:
    """
    Submit speech evaluation task (async with callback).

    The task is processed in the background. Results will be sent to the callback_url
    when processing completes.

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/audio.mp3",
        "language": "zh",
        "ref_text": "optional reference text",
        "custom_prompt": "optional custom evaluation prompt",
        "message_id": "optional-custom-id",
        "callback_url": "https://your-server.com/callback"
    }
    ```

    **Response** (immediate):
    ```json
    {
        "success": true,
        "message": "Task accepted",
        "message_id": "uuid-or-custom-id"
    }
    ```

    **Callback data** (sent to callback_url when complete):
    ```json
    {
        "message_id": "uuid-or-custom-id",
        "success": true,
        "message": "Evaluation completed successfully",
        "speech_text": "transcribed text",
        "speech_scores": {...},
        "statistics": {...},
        "low_score_words": [...],
        "evaluation_report": "markdown report"
    }
    ```
    """
    # Verify signature from header
    verify_signature(x_signature)

    # Validate input
    if not request.audio_url and not request.audio_path:
        raise HTTPException(
            status_code=400,
            detail="Either audio_url or audio_path must be provided"
        )

    # Generate message_id if not provided
    msg_id = request.message_id or str(uuid.uuid4())
    callback_url = str(request.callback_url)

    try:
        # Get audio data
        if request.audio_url:
            audio_data = await asr_service.download_audio(request.audio_url)
        else:
            if not os.path.exists(request.audio_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Audio file not found: {request.audio_path}"
                )
            audio_data = await asyncio.to_thread(
                lambda: open(request.audio_path, "rb").read()
            )

        # Add background task
        background_tasks.add_task(
            process_evaluation_task,
            msg_id,
            callback_url,
            audio_data,
            request.language,
            request.ref_text or "",
            request.custom_prompt
        )

        return EvaluationAcceptedResponse(
            success=True,
            message="Task accepted, results will be sent to callback URL",
            message_id=msg_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to accept task: {str(e)}"
        )


@router.post("/analyze/upload", response_model=EvaluationAcceptedResponse)
async def evaluate_speech_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to evaluate"),
    language: str = Form("zh", description="Language: 'zh' for Chinese, 'en' for English"),
    ref_text: Optional[str] = Form("", description="Reference text for evaluation"),
    custom_prompt: Optional[str] = Form(None, description="Custom prompt for AI evaluation"),
    message_id: Optional[str] = Form(None, description="Message ID for tracking (auto-generated if not provided)"),
    callback_url: str = Form(..., description="Callback URL to receive evaluation results"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> EvaluationAcceptedResponse:
    """
    Submit speech evaluation task from uploaded file (async with callback).

    The task is processed in the background. Results will be sent to the callback_url
    when processing completes.

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Form data**:
    - file: The audio file (any format supported by ffmpeg)
    - language: 'zh' or 'en' (default: 'zh')
    - ref_text: Optional reference text
    - custom_prompt: Optional custom prompt for AI evaluation
    - message_id: Optional message ID (auto-generated if not provided)
    - callback_url: URL to receive results (required)

    **Response** (immediate):
    ```json
    {
        "success": true,
        "message": "Task accepted",
        "message_id": "uuid-or-custom-id"
    }
    ```
    """
    # Verify signature from header
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = message_id or str(uuid.uuid4())

    try:
        # Read uploaded file
        audio_data = await file.read()

        if len(audio_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty"
            )

        # Add background task
        background_tasks.add_task(
            process_evaluation_task,
            msg_id,
            callback_url,
            audio_data,
            language,
            ref_text or "",
            custom_prompt
        )

        return EvaluationAcceptedResponse(
            success=True,
            message="Task accepted, results will be sent to callback URL",
            message_id=msg_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to accept task: {str(e)}"
        )
