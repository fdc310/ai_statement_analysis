# -*- coding: utf-8 -*-
"""
Speech evaluation endpoints with async callback support and AES signature authentication.
Signature is passed via X-Signature header.

These endpoints:
1. Immediately return ASR + SOE results (speech_text, speech_scores, etc.)
2. Generate AI report in background and send to callback_url when complete
"""
import uuid
import asyncio
import os
import string
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
    SignatureResponse,
    ReportRequest,
    ReportResponse,
    TextAnalysisRequest,
    TextAnalysisResponse,
    TongueTwisterRequest,
    TongueTwisterResponse
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
    """Send AI report to callback URL."""
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


async def generate_report_task(
    message_id: str,
    callback_url: str,
    speech_text: str,
    scores_data: dict,
    custom_prompt: Optional[str],
    low_score_words_data: list,
    statistics_data: dict
) -> None:
    """Background task to generate AI report and send callback."""
    try:
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
            message="AI report generated successfully",
            evaluation_report=evaluation_report
        )
        await send_callback(callback_url, callback_data)

    except Exception as e:
        # Send error callback
        callback_data = EvaluationCallbackData(
            message_id=message_id,
            success=False,
            message="AI report generation failed",
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
    Evaluate speech and get SOE scores immediately, AI report via callback.

    **Workflow**:
    1. Immediately processes ASR + SOE and returns results
    2. AI report is generated in background
    3. When AI report is ready, it's sent to callback_url

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

    **Response** (immediate, includes SOE scores):
    ```json
    {
        "success": true,
        "message": "Evaluation completed, AI report will be sent to callback URL",
        "message_id": "uuid",
        "speech_text": "transcribed text",
        "speech_scores": {
            "pronunciation_accuracy": 85.5,
            "pronunciation_fluency": 90.2,
            ...
        },
        "statistics": {...},
        "low_score_words": [...]
    }
    ```

    **Callback data** (sent to callback_url when AI report is ready):
    ```json
    {
        "message_id": "uuid",
        "success": true,
        "message": "AI report generated successfully",
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

        # Determine engine type based on language
        engine_type = "16k_zh" if request.language == "zh" else "16k_en"
        server_type = 0 if request.language == "zh" else 1

        # Run ASR and SOE in parallel (synchronously wait for results)
        asr_result, soe_result = await asyncio.gather(
            asr_service.recognize_audio(audio_data, engine_type),
            soe_service.evaluate_audio(
                audio_data,
                request.ref_text or "",
                3 if not request.ref_text else 1,
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

        # Add background task for AI report generation
        background_tasks.add_task(
            generate_report_task,
            msg_id,
            callback_url,
            speech_text,
            scores_data,
            request.custom_prompt,
            low_score_words_data,
            statistics_data
        )

        return EvaluationAcceptedResponse(
            success=True,
            message="Evaluation completed, AI report will be sent to callback URL",
            message_id=msg_id,
            speech_text=speech_text,
            speech_scores=speech_scores,
            statistics=statistics,
            low_score_words=low_score_words
        )

    except HTTPException:
        raise
    except Exception as e:
        return EvaluationAcceptedResponse(
            success=False,
            message="Evaluation failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/analyze/upload", response_model=EvaluationAcceptedResponse)
async def evaluate_speech_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to evaluate"),
    language: str = Form("zh", description="Language: 'zh' for Chinese, 'en' for English"),
    ref_text: Optional[str] = Form("", description="Reference text for evaluation"),
    custom_prompt: Optional[str] = Form(None, description="Custom prompt for AI evaluation"),
    message_id: Optional[str] = Form(None, description="Message ID for tracking (auto-generated if not provided)"),
    callback_url: str = Form(..., description="Callback URL to receive AI report"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> EvaluationAcceptedResponse:
    """
    Evaluate uploaded audio and get SOE scores immediately, AI report via callback.

    **Workflow**:
    1. Immediately processes ASR + SOE and returns results
    2. AI report is generated in background
    3. When AI report is ready, it's sent to callback_url

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Form data**:
    - file: The audio file (any format supported by ffmpeg)
    - language: 'zh' or 'en' (default: 'zh')
    - ref_text: Optional reference text
    - custom_prompt: Optional custom prompt for AI evaluation
    - message_id: Optional message ID (auto-generated if not provided)
    - callback_url: URL to receive AI report (required)

    **Response** (immediate, includes SOE scores):
    ```json
    {
        "success": true,
        "message": "Evaluation completed, AI report will be sent to callback URL",
        "message_id": "uuid",
        "speech_text": "transcribed text",
        "speech_scores": {...},
        "statistics": {...},
        "low_score_words": [...]
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

        # Determine engine type based on language
        engine_type = "16k_zh" if language == "zh" else "16k_en"
        server_type = 0 if language == "zh" else 1

        # Run ASR and SOE in parallel (synchronously wait for results)
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

        # Add background task for AI report generation
        background_tasks.add_task(
            generate_report_task,
            msg_id,
            callback_url,
            speech_text,
            scores_data,
            custom_prompt,
            low_score_words_data,
            statistics_data
        )

        return EvaluationAcceptedResponse(
            success=True,
            message="Evaluation completed, AI report will be sent to callback URL",
            message_id=msg_id,
            speech_text=speech_text,
            speech_scores=speech_scores,
            statistics=statistics,
            low_score_words=low_score_words
        )

    except HTTPException:
        raise
    except Exception as e:
        return EvaluationAcceptedResponse(
            success=False,
            message="Evaluation failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> ReportResponse:
    """
    Generate AI evaluation report from pre-existing SOE scores (synchronous).

    This endpoint receives SOE evaluation results and generates an AI report directly.
    If speech_text is not provided, ASR will be called to transcribe the audio.

    **Features**:
    - Receives full SOE result data
    - Optional ASR transcription (if speech_text not provided)
    - Topic relevance analysis (if topic provided)
    - Speech rate calculation (if audio_duration provided)

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/audio.mp3",
        "speech_text": "可选，不传则自动ASR识别",
        "soe_result": {
            "SuggestedScore": 85.5,
            "PronAccuracy": 90.2,
            "PronFluency": 88.0,
            "PronCompletion": 95.0,
            "Words": [...]
        },
        "audio_duration": 60.5,
        "topic": "环境保护",
        "custom_prompt": "optional custom evaluation prompt",
        "message_id": "optional-custom-id",
        "language": "zh"
    }
    ```

    **Response**:
    ```json
    {
        "success": true,
        "message": "Report generated successfully",
        "message_id": "uuid",
        "audio_url": "https://example.com/audio.mp3",
        "speech_text": "转写文本",
        "speech_rate": 180.5,
        "evaluation_report": "# AI Evaluation Report\\n..."
    }
    ```
    """
    # Verify signature from header
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        speech_text = request.speech_text

        # If speech_text not provided, call ASR
        if not speech_text:
            # Download audio and run ASR
            audio_data = await asr_service.download_audio(audio_url)
            engine_type = "16k_zh" if request.language == "zh" else "16k_en"
            asr_result = await asr_service.recognize_audio(audio_data, engine_type)
            speech_text = asr_result.get("text", "")

        # Calculate speech rate if audio_duration provided
        speech_rate = None
        if request.audio_duration and request.audio_duration > 0 and speech_text:
            # Calculate characters/words per minute
            if request.language == "zh":
                # Chinese: count characters (excluding spaces and punctuation)
                punctuation = string.punctuation + '。，！？、；：""''（）【】《》…—'
                char_count = len([c for c in speech_text if c not in punctuation and not c.isspace()])
            else:
                # English: count words
                char_count = len(speech_text.split())

            # Convert to per minute
            speech_rate = round(char_count / (request.audio_duration / 60), 1)

        # Extract scores from SOE result (direct result data, no nesting)
        result_data = request.soe_result

        scores_data = {
            "pronunciation_accuracy": result_data.get("PronAccuracy", 0),
            "pronunciation_fluency": result_data.get("PronFluency", 0),
            "pronunciation_completion": result_data.get("PronCompletion", 0),
            "suggested_score": result_data.get("SuggestedScore", 0),
            "overall_score": result_data.get("SuggestedScore", 0)
        }

        # Extract low score words
        low_score_words_data = []
        words = result_data.get("Words", [])
        for word in words:
            accuracy = word.get("PronAccuracy", 100)
            if accuracy < 90:
                low_score_words_data.append({
                    "word": word.get("Word", ""),
                    "accuracy": accuracy,
                    "fluency": word.get("PronFluency", 0)
                })

        # Calculate statistics
        statistics_data = {
            "total_words": len(words),
            "average_accuracy": sum(w.get("PronAccuracy", 0) for w in words) / len(words) if words else 0,
            "low_score_count": len(low_score_words_data)
        }

        # Generate AI evaluation report with extended features
        evaluation_report = await hunyuan_service.generate_evaluation_extended(
            speech_text=speech_text,
            speech_scores=scores_data,
            custom_prompt=request.custom_prompt,
            low_score_words=low_score_words_data,
            statistics=statistics_data,
            topic=request.topic,
            speech_rate=speech_rate,
            audio_duration=request.audio_duration
        )

        return ReportResponse(
            success=True,
            message="Report generated successfully",
            message_id=msg_id,
            audio_url=audio_url,
            speech_text=speech_text,
            speech_rate=speech_rate,
            evaluation_report=evaluation_report
        )

    except Exception as e:
        return ReportResponse(
            success=False,
            message="Report generation failed",
            message_id=msg_id,
            audio_url=audio_url,
            error=str(e)
        )


@router.post("/report/upload", response_model=ReportResponse)
async def generate_report_upload(
    file: UploadFile = File(..., description="音频文件"),
    soe_result: str = Form(..., description="SOE评测返回的result数据(JSON字符串)"),
    speech_text: Optional[str] = Form(None, description="语音转写文本，不传则自动调用ASR识别"),
    audio_duration: Optional[float] = Form(None, description="音频时长（秒），用于计算语速"),
    topic: Optional[str] = Form(None, description="演讲主题，用于分析内容贴题性。不传则为自由说模式"),
    custom_prompt: Optional[str] = Form(None, description="自定义AI评测提示词"),
    message_id: Optional[str] = Form(None, description="消息ID，不传则自动生成UUID"),
    language: str = Form("zh", description="语言：'zh'中文，'en'英文"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> ReportResponse:
    """
    Upload audio file and generate AI evaluation report from SOE scores (synchronous).

    This endpoint receives an uploaded audio file and SOE evaluation results,
    then generates an AI report directly. If speech_text is not provided,
    ASR will be called to transcribe the audio.

    **Features**:
    - Upload audio file directly
    - Receives SOE result data as JSON string
    - Optional ASR transcription (if speech_text not provided)
    - Topic relevance analysis (if topic provided)
    - Speech rate calculation (if audio_duration provided)

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Form data**:
    - file: 音频文件
    - soe_result: SOE评测结果(JSON字符串)，如 {"SuggestedScore": 85.5, "PronAccuracy": 90.2, ...}
    - speech_text: 可选，语音转写文本
    - audio_duration: 可选，音频时长（秒）
    - topic: 可选，演讲主题
    - custom_prompt: 可选，自定义评测提示词
    - message_id: 可选，消息ID
    - language: 语言，默认 "zh"

    **Response**:
    ```json
    {
        "success": true,
        "message": "Report generated successfully",
        "message_id": "uuid",
        "audio_url": "",
        "speech_text": "转写文本",
        "speech_rate": 180.5,
        "evaluation_report": "# AI Evaluation Report\\n..."
    }
    ```
    """
    import json
    import re

    # Verify signature from header
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = message_id or str(uuid.uuid4())

    try:
        # Parse soe_result JSON string
        try:
            result_data = json.loads(soe_result)
        except json.JSONDecodeError:
            return ReportResponse(
                success=False,
                message="Invalid soe_result JSON format",
                message_id=msg_id,
                audio_url="",
                error="soe_result must be a valid JSON string"
            )

        # Read uploaded file
        audio_data = await file.read()
        if len(audio_data) == 0:
            return ReportResponse(
                success=False,
                message="Audio file is empty",
                message_id=msg_id,
                audio_url="",
                error="Uploaded audio file is empty"
            )

        text = speech_text

        # If speech_text not provided, call ASR
        if not text:
            engine_type = "16k_zh" if language == "zh" else "16k_en"
            asr_result = await asr_service.recognize_audio(audio_data, engine_type)
            text = asr_result.get("text", "")

        # Calculate speech rate if audio_duration provided
        speech_rate = None
        if audio_duration and audio_duration > 0 and text:
            # Calculate characters/words per minute
            if language == "zh":
                # Chinese: count characters (excluding spaces and punctuation)
                punctuation = string.punctuation + '。，！？、；：""''（）【】《》…—'
                char_count = len([c for c in text if c not in punctuation and not c.isspace()])
            else:
                # English: count words
                char_count = len(text.split())

            # Convert to per minute
            speech_rate = round(char_count / (audio_duration / 60), 1)

        scores_data = {
            "pronunciation_accuracy": result_data.get("PronAccuracy", 0),
            "pronunciation_fluency": result_data.get("PronFluency", 0),
            "pronunciation_completion": result_data.get("PronCompletion", 0),
            "suggested_score": result_data.get("SuggestedScore", 0),
            "overall_score": result_data.get("SuggestedScore", 0)
        }

        # Extract low score words
        low_score_words_data = []
        words = result_data.get("Words", [])
        for word in words:
            accuracy = word.get("PronAccuracy", 100)
            if accuracy < 90:
                low_score_words_data.append({
                    "word": word.get("Word", ""),
                    "accuracy": accuracy,
                    "fluency": word.get("PronFluency", 0)
                })

        # Calculate statistics
        statistics_data = {
            "total_words": len(words),
            "average_accuracy": sum(w.get("PronAccuracy", 0) for w in words) / len(words) if words else 0,
            "low_score_count": len(low_score_words_data)
        }

        # Generate AI evaluation report with extended features
        evaluation_report = await hunyuan_service.generate_evaluation_extended(
            speech_text=text,
            speech_scores=scores_data,
            custom_prompt=custom_prompt,
            low_score_words=low_score_words_data,
            statistics=statistics_data,
            topic=topic,
            speech_rate=speech_rate,
            audio_duration=audio_duration
        )

        return ReportResponse(
            success=True,
            message="Report generated successfully",
            message_id=msg_id,
            audio_url="",
            speech_text=text,
            speech_rate=speech_rate,
            evaluation_report=evaluation_report
        )

    except Exception as e:
        return ReportResponse(
            success=False,
            message="Report generation failed",
            message_id=msg_id,
            audio_url="",
            error=str(e)
        )


@router.post("/text-analysis", response_model=TextAnalysisResponse)
async def analyze_text_structure(
    request: TextAnalysisRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> TextAnalysisResponse:
    """
    Analyze text structure: core ideas, logical structure, key points.

    This endpoint analyzes the provided text and extracts:
    - Core idea / main theme
    - Key points with importance levels
    - Logical structure (type and outline)
    - Arguments (claims, evidence, reasoning)
    - Conclusion
    - Writing style
    - Improvement suggestions

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Request body**:
    ```json
    {
        "text": "待分析的文本内容...",
        "custom_prompt": "可选的自定义分析要求",
        "message_id": "可选的消息ID"
    }
    ```

    **Response**:
    ```json
    {
        "success": true,
        "message": "Analysis completed successfully",
        "message_id": "uuid",
        "analysis_result": "{...JSON格式的分析结果...}"
    }
    ```

    **Analysis Result Structure**:
    ```json
    {
        "core_idea": "核心思想",
        "key_points": [
            {"title": "要点标题", "content": "内容", "importance": "高/中/低"}
        ],
        "logical_structure": {
            "type": "结构类型",
            "description": "结构说明",
            "outline": [...]
        },
        "arguments": [
            {"claim": "论点", "evidence": "论据", "reasoning": "论证"}
        ],
        "conclusion": "结论",
        "writing_style": "写作风格",
        "suggestions": ["建议1", "建议2"]
    }
    ```
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())

    try:
        analysis_result = await hunyuan_service.analyze_text_structure(
            text=request.text,
            custom_prompt=request.custom_prompt
        )

        return TextAnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            message_id=msg_id,
            analysis_result=analysis_result
        )

    except Exception as e:
        return TextAnalysisResponse(
            success=False,
            message="Analysis failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/tongue-twister", response_model=TongueTwisterResponse)
async def analyze_tongue_twister(
    request: TongueTwisterRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> TongueTwisterResponse:
    """
    Analyze tongue twister pronunciation key points.

    This endpoint analyzes tongue twisters and provides:
    - Core phonemes and their articulation details
    - Acoustic feature differences between similar sounds
    - Confusion pairs and how to distinguish them
    - Pronunciation tips and techniques
    - Practice sequence recommendations

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Request body**:
    ```json
    {
        "text": "八百标兵奔北坡，炮兵并排北边跑",
        "language": "zh",
        "message_id": "可选的消息ID"
    }
    ```

    **Response**:
    ```json
    {
        "success": true,
        "message": "Analysis completed successfully",
        "message_id": "uuid",
        "tongue_twister": "原文",
        "analysis_result": "{...JSON格式的分析结果...}"
    }
    ```

    **Analysis Result Structure (Chinese)**:
    ```json
    {
        "title": "绕口令标题",
        "difficulty": "难度等级",
        "core_phonemes": [
            {
                "phoneme": "音素",
                "pinyin": "拼音",
                "ipa": "国际音标",
                "description": "发音描述",
                "articulation": {
                    "manner": "发音方式",
                    "place": "发音部位",
                    "voicing": "清浊"
                },
                "examples": ["示例字词"]
            }
        ],
        "acoustic_features": [
            {
                "feature": "声学特征",
                "description": "特征描述",
                "key_difference": "关键差异",
                "measurement": "声学指标"
            }
        ],
        "confusion_pairs": [
            {
                "pair": ["音素1", "音素2"],
                "difference": "区分要点",
                "common_errors": "常见错误",
                "practice_tip": "练习建议"
            }
        ],
        "pronunciation_tips": [
            {
                "tip": "发音提示",
                "target_sounds": ["目标音素"],
                "technique": "技巧",
                "practice_method": "练习方法"
            }
        ],
        "rhythm_pattern": {
            "beat_count": "节拍数",
            "stress_pattern": "重音模式",
            "pause_points": ["停顿位置"],
            "speed_suggestion": "建议语速"
        },
        "practice_sequence": [
            {
                "step": 1,
                "focus": "练习重点",
                "content": "练习内容",
                "repetitions": "重复次数"
            }
        ],
        "annotated_text": "带[音素标注]的文本"
    }
    ```
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())

    try:
        analysis_result = await hunyuan_service.analyze_tongue_twister(
            text=request.text,
            language=request.language
        )

        return TongueTwisterResponse(
            success=True,
            message="Analysis completed successfully",
            message_id=msg_id,
            tongue_twister=request.text,
            analysis_result=analysis_result
        )

    except Exception as e:
        return TongueTwisterResponse(
            success=False,
            message="Analysis failed",
            message_id=msg_id,
            tongue_twister=request.text,
            error=str(e)
        )
