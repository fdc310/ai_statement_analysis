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
import base64
from pathlib import Path
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
    ReportType,
    TextAnalysisRequest,
    TextAnalysisResponse,
    TongueTwisterRequest,
    TongueTwisterResponse,
    SentenceInterpretationRequest,
    SentenceInterpretationResponse,
    StoryReadingRequest,
    StoryReadingResponse,
    TongueTwisterReadingRequest,
    TongueTwisterReadingResponse,
    VoiceChatRequest,
    VoiceTextChatRequest,
    VoiceChatResponse,
    OpinionStatementRequest,
    OpinionStatementResponse,
    ImpromptuReactionRequest,
    ImpromptuReactionResponse
)
from app.services.tencent import asr_service, soe_service, tts_service
from app.services.tencent.hunyuan import hunyuan_service
from app.services.tencent.audio import get_audio_duration
from app.services.chat.session_manager import chat_session_manager

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


async def read_allowed_audio_path(audio_path: str) -> bytes:
    """Read server-local audio only from the configured AUDIO_LOCAL_ROOT."""
    root_value = settings.audio_local_root.strip()
    if not root_value:
        raise HTTPException(
            status_code=400,
            detail="audio_path is disabled. Configure AUDIO_LOCAL_ROOT to enable local audio reads."
        )

    root = Path(root_value)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    target = Path(audio_path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="audio_path is outside AUDIO_LOCAL_ROOT")

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    def _read_file(path: Path) -> bytes:
        with path.open("rb") as f:
            return f.read()

    return await asyncio.to_thread(_read_file, target)


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
            audio_data = await read_allowed_audio_path(request.audio_path)

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

        if not speech_text or not speech_text.strip():
            return EvaluationAcceptedResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                error="ASR returned empty text"
            )

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

        if not speech_text or not speech_text.strip():
            return EvaluationAcceptedResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                error="ASR returned empty text"
            )

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
    根据SOE评测结果生成AI评测报告（同步接口）。

    接收SOE评测结果数据，直接生成AI报告。如果未提供speech_text，将自动调用ASR进行转写。

    **功能特性**:
    - 接收完整的SOE评测结果数据
    - 可选ASR转写（如果未提供speech_text）
    - 主题贴题性分析（如果提供了topic）
    - 语速计算（如果提供了audio_duration）
    - 支持简洁报告和完整报告两种类型

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/audio.mp3",
        "speech_text": "可选，不传则自动ASR识别",
        "soe_result": {
            "SuggestedScore": 85.5,
            "PronAccuracy": 90.2,
            "PronFluency": 0.88,
            "PronCompletion": -1,
            "Words": [
                {"Word": "你", "PronAccuracy": 95.5, "PronFluency": 0.92},
                {"Word": "好", "PronAccuracy": 88.2, "PronFluency": 0.85}
            ]
        },
        "audio_duration": 60.5,
        "topic": "环境保护",
        "custom_prompt": "可选，自定义评测提示词",
        "message_id": "可选，消息ID",
        "language": "zh",
        "report_type": "full"
    }
    ```

    **参数说明**:
    - `audio_url`: 音频文件URL（必填）
    - `speech_text`: 语音转写文本，不传则自动调用ASR识别
    - `soe_result`: SOE评测返回的result数据（必填），包含SuggestedScore、PronAccuracy、Words等字段
    - `audio_duration`: 音频时长（秒），用于计算语速
    - `topic`: 演讲主题，用于分析内容贴题性。不传则为自由说模式
    - `custom_prompt`: 自定义AI评测提示词
    - `message_id`: 消息ID，不传则自动生成UUID
    - `language`: 语言，'zh'中文，'en'英文，默认'zh'
    - `report_type`: 报告类型，'simple'简洁报告，'full'完整报告，默认'full'

    **Response - 简洁报告 (report_type="simple")**:
    ```json
    {
        "success": true,
        "message": "Report generated successfully",
        "message_id": "uuid",
        "audio_url": "https://example.com/audio.mp3",
        "speech_text": "转写文本",
        "speech_rate": 180.5,
        "evaluation_report": {
            "speech_rate": {
                "rate": 180.5,
                "score": 85,
                "level": "良好",
                "suggestion": "语速适中，建议保持..."
            },
            "weak_paragraphs": [
                {
                    "paragraph_index": 1,
                    "content": "段落内容...",
                    "low_score_words": [
                        {"word": "好", "accuracy": 88.2}
                    ],
                    "suggestion": "建议加强..."
                }
            ],
            "overall_suggestion": "整体建议..."
        }
    }
    ```

    **Response - 完整报告 (report_type="full")**:
    ```json
    {
        "success": true,
        "message": "Report generated successfully",
        "message_id": "uuid",
        "audio_url": "https://example.com/audio.mp3",
        "speech_text": "转写文本",
        "speech_rate": 180.5,
        "evaluation_report": {
            "logic_completeness": {
                "overall_score": 85,
                "logic_score": 82,
                "fluency_score": 88,
                "speech_rate_score": 85,
                "topic_relevance_score": 90,
                "speech_rate_value": 180.5,
                "speech_rate_level": "良好",
                "speech_rate_suggestion": "语速建议..."
            },
            "structure_visualization": {
                "arguments": ["论点1", "论点2", "论点3"],
                "conclusion": "结论要点..."
            },
            "speech_rate_evaluation": {
                "score": 85,
                "rate_value": 180.5,
                "level": "良好",
                "analysis": "语速分析...",
                "suggestion": "语速建议..."
            },
            "content_perspective": {
                "score": 88,
                "topic_relevance": "贴题性分析...",
                "depth": "内容深度分析...",
                "coverage": "内容覆盖面分析...",
                "suggestion": "内容改进建议..."
            },
            "logic_structure": {
                "score": 82,
                "organization": "整体结构分析...",
                "coherence": "连贯性分析...",
                "reasoning": "论证逻辑分析...",
                "suggestion": "逻辑结构改进建议..."
            },
            "expression_wording": {
                "score": 86,
                "vocabulary_level": "用词水平分析...",
                "expression_style": "表达风格分析...",
                "highlights": ["表达亮点1", "表达亮点2"],
                "suggestion": "表达用词改进建议..."
            },
            "strengths": ["优点1", "优点2", "优点3"],
            "improvements": ["改进意见1", "改进意见2"],
            "weak_paragraphs": [
                {
                    "paragraph_index": 1,
                    "content": "段落内容...",
                    "low_score_words": [{"word": "好", "accuracy": 88.2}],
                    "suggestion": "建议加强..."
                }
            ]
        }
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

        if not speech_text or not speech_text.strip():
            return ReportResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                audio_url=audio_url,
                error="ASR returned empty text"
            )

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

        # 根据 report_type 生成不同格式的报告
        if request.report_type == ReportType.simple:
            # 简洁报告：语速评分 + 低分段落分析
            evaluation_report = await hunyuan_service.generate_simple_report_json(
                speech_text=speech_text,
                speech_scores=scores_data,
                low_score_words=low_score_words_data,
                speech_rate=speech_rate,
                audio_duration=request.audio_duration,
                language=request.language
            )
        else:
            # 完整报告：语速 + 内容角度 + 逻辑与结构 + 表达与用词
            evaluation_report = await hunyuan_service.generate_full_report_json(
                speech_text=speech_text,
                speech_scores=scores_data,
                low_score_words=low_score_words_data,
                statistics=statistics_data,
                topic=request.topic,
                speech_rate=speech_rate,
                audio_duration=request.audio_duration,
                language=request.language
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
    report_type: str = Form("full", description="报告类型：'simple'简洁报告，'full'完整报告"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> ReportResponse:
    """
    上传音频文件并根据SOE评测结果生成AI评测报告（同步接口）。

    接收上传的音频文件和SOE评测结果，直接生成AI报告。如果未提供speech_text，将自动调用ASR进行转写。

    **功能特性**:
    - 直接上传音频文件
    - 接收SOE评测结果数据（JSON字符串格式）
    - 可选ASR转写（如果未提供speech_text）
    - 主题贴题性分析（如果提供了topic）
    - 语速计算（如果提供了audio_duration）
    - 支持简洁报告和完整报告两种类型

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Form data**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | file | File | 是 | 音频文件（支持ffmpeg支持的所有格式） |
    | soe_result | string | 是 | SOE评测结果JSON字符串，如 `{"SuggestedScore": 85.5, "PronAccuracy": 90.2, "Words": [...]}` |
    | speech_text | string | 否 | 语音转写文本，不传则自动调用ASR识别 |
    | audio_duration | float | 否 | 音频时长（秒），用于计算语速 |
    | topic | string | 否 | 演讲主题，用于分析内容贴题性。不传则为自由说模式 |
    | custom_prompt | string | 否 | 自定义AI评测提示词 |
    | message_id | string | 否 | 消息ID，不传则自动生成UUID |
    | language | string | 否 | 语言，'zh'中文，'en'英文，默认'zh' |
    | report_type | string | 否 | 报告类型，'simple'简洁报告，'full'完整报告，默认'full' |

    **Response - 简洁报告 (report_type="simple")**:
    ```json
    {
        "success": true,
        "message": "Report generated successfully",
        "message_id": "uuid",
        "audio_url": "",
        "speech_text": "转写文本",
        "speech_rate": 180.5,
        "evaluation_report": {
            "speech_rate": {
                "rate": 180.5,
                "score": 85,
                "level": "良好",
                "suggestion": "语速适中，建议保持..."
            },
            "weak_paragraphs": [
                {
                    "paragraph_index": 1,
                    "content": "段落内容...",
                    "low_score_words": [{"word": "好", "accuracy": 88.2}],
                    "suggestion": "建议加强..."
                }
            ],
            "overall_suggestion": "整体建议..."
        }
    }
    ```

    **Response - 完整报告 (report_type="full")**:
    ```json
    {
        "success": true,
        "message": "Report generated successfully",
        "message_id": "uuid",
        "audio_url": "",
        "speech_text": "转写文本",
        "speech_rate": 180.5,
        "evaluation_report": {
            "logic_completeness": {
                "overall_score": 85,
                "logic_score": 82,
                "fluency_score": 88,
                "speech_rate_score": 85,
                "topic_relevance_score": 90,
                "speech_rate_value": 180.5,
                "speech_rate_level": "良好",
                "speech_rate_suggestion": "语速建议..."
            },
            "structure_visualization": {
                "arguments": ["论点1", "论点2", "论点3"],
                "conclusion": "结论要点..."
            },
            "speech_rate_evaluation": {
                "score": 85,
                "rate_value": 180.5,
                "level": "良好",
                "analysis": "语速分析...",
                "suggestion": "语速建议..."
            },
            "content_perspective": {
                "score": 88,
                "topic_relevance": "贴题性分析...",
                "depth": "内容深度分析...",
                "coverage": "内容覆盖面分析...",
                "suggestion": "内容改进建议..."
            },
            "logic_structure": {
                "score": 82,
                "organization": "整体结构分析...",
                "coherence": "连贯性分析...",
                "reasoning": "论证逻辑分析...",
                "suggestion": "逻辑结构改进建议..."
            },
            "expression_wording": {
                "score": 86,
                "vocabulary_level": "用词水平分析...",
                "expression_style": "表达风格分析...",
                "highlights": ["表达亮点1", "表达亮点2"],
                "suggestion": "表达用词改进建议..."
            },
            "strengths": ["优点1", "优点2", "优点3"],
            "improvements": ["改进意见1", "改进意见2"],
            "weak_paragraphs": [
                {
                    "paragraph_index": 1,
                    "content": "段落内容...",
                    "low_score_words": [{"word": "好", "accuracy": 88.2}],
                    "suggestion": "建议加强..."
                }
            ]
        }
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

        if not text or not text.strip():
            return ReportResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                audio_url="",
                error="ASR returned empty text"
            )

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

        # 根据 report_type 生成不同格式的报告
        if report_type == "simple":
            # 简洁报告：语速评分 + 低分段落分析
            evaluation_report = await hunyuan_service.generate_simple_report_json(
                speech_text=text,
                speech_scores=scores_data,
                low_score_words=low_score_words_data,
                speech_rate=speech_rate,
                audio_duration=audio_duration,
                language=language
            )
        else:
            # 完整报告：语速 + 内容角度 + 逻辑与结构 + 表达与用词
            evaluation_report = await hunyuan_service.generate_full_report_json(
                speech_text=text,
                speech_scores=scores_data,
                low_score_words=low_score_words_data,
                statistics=statistics_data,
                topic=topic,
                speech_rate=speech_rate,
                audio_duration=audio_duration,
                language=language
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
    文本结构分析接口 - 提取核心思想、逻辑结构和关键要点。

    本接口对提供的文本进行深度分析，提取：
    - 核心思想 / 主旨
    - 关键要点（附带重要性级别）
    - 逻辑结构（类型和大纲）
    - 论点论据（主张、证据、论证逻辑）
    - 结论
    - 写作风格
    - 改进建议

    **功能特性**:
    - 支持中英文文本分析
    - 智能提取文本核心主旨
    - 自动识别逻辑结构类型（总分总、递进式、并列式、对比式、因果式等）
    - 提供多级大纲结构
    - 给出具体可操作的改进建议

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "text": "待分析的文本内容...",
        "custom_prompt": "可选的自定义分析要求",
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | text | string | 是 | 待分析的文本内容，长度范围：10-50000字符 |
    | custom_prompt | string | 否 | 自定义分析要求，可指定关注的分析维度 |
    | message_id | string | 否 | 消息ID，不传则自动生成UUID |

    **Response**:
    ```json
    {
        "success": true,
        "message": "Analysis completed successfully",
        "message_id": "uuid",
        "analysis_result": "{...JSON格式的分析结果...}"
    }
    ```

    **Analysis Result Structure** (纯JSON格式，非markdown代码块):
    ```json
    {
        "core_idea": "文本的核心思想/主旨，用一两句话概括",
        "key_points": [
            {
                "title": "要点标题",
                "content": "要点详细内容",
                "importance": "高/中/低"
            }
        ],
        "logical_structure": {
            "type": "结构类型（如：总分总、递进式、并列式、对比式、因果式等）",
            "description": "对逻辑结构的简要说明",
            "outline": [
                {
                    "level": 1,
                    "title": "一级标题/段落主题",
                    "summary": "该部分的简要概括",
                    "sub_points": [
                        {
                            "level": 2,
                            "title": "二级要点",
                            "summary": "要点说明"
                        }
                    ]
                }
            ]
        },
        "arguments": [
            {
                "claim": "论点/观点",
                "evidence": "支撑论据",
                "reasoning": "论证逻辑"
            }
        ],
        "conclusion": "结论或总结",
        "writing_style": "写作风格特点",
        "suggestions": [
            "改进建议1",
            "改进建议2"
        ]
    }
    ```

    **注意**:
    - `analysis_result` 为JSON字符串格式，需要再次解析为JSON对象
    - 如果某些部分在文本中不明显，可能返回为null或空数组
    - 分析结果基于AI模型生成，可能存在一定的主观性
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
    绕口令发音分析接口 - 提供音素发音要点、声学特征差异和练习建议。

    本接口分析绕口令并提供：
    - 核心音素及其发音部位、方式详解
    - 相似音素间的声学特征差异
    - 易混淆音素对比及区分方法
    - 发音技巧和练习方法
    - 节奏模式和练习序列建议

    **功能特性**:
    - 支持中英文绕口令分析
    - 基于语音学原理的音素分析
    - 提供发音部位、方式、清浊等详细参数
    - 声学指标（如VOT、F1/F2频率）分析
    - 设计科学的练习顺序
    - 节奏和停顿建议

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "text": "八百标兵奔北坡，炮兵并排北边跑",
        "language": "zh",
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | text | string | 是 | 绕口令文本，长度范围：2-5000字符 |
    | language | string | 否 | 语言：'zh'中文，'en'英文，默认'zh' |
    | message_id | string | 否 | 消息ID，不传则自动生成UUID |

    **Response**:
    ```json
    {
        "success": true,
        "message": "Analysis completed successfully",
        "message_id": "uuid",
        "tongue_twister": "八百标兵奔北坡，炮兵并排北边跑",
        "analysis_result": "{...JSON格式的分析结果...}"
    }
    ```

    **Analysis Result Structure - 中文 (纯JSON格式，非markdown代码块)**:
    ```json
    {
        "title": "绕口令标题/主题",
        "difficulty": "难度等级（简单/中等/困难/专家）",
        "core_phonemes": [
            {
                "phoneme": "音素（如：b、p、m、f等）",
                "pinyin": "对应拼音",
                "ipa": "国际音标",
                "description": "发音描述",
                "articulation": {
                    "manner": "发音方式（如：爆破音、摩擦音、鼻音等）",
                    "place": "发音部位（如：双唇、舌尖、舌根等）",
                    "voicing": "清浊（清音/浊音）"
                },
                "examples": ["包含该音素的字词示例"]
            }
        ],
        "acoustic_features": [
            {
                "feature": "声学特征名称",
                "description": "特征描述",
                "key_difference": "与易混淆音的关键差异",
                "measurement": "可量化的声学指标（如：VOT、F1/F2频率等）"
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
                "target_sounds": ["针对的音素"],
                "technique": "具体技巧",
                "practice_method": "练习方法"
            }
        ],
        "rhythm_pattern": {
            "beat_count": "节拍数",
            "stress_pattern": "重音模式",
            "pause_points": ["建议停顿位置"],
            "speed_suggestion": "建议语速"
        },
        "practice_sequence": [
            {
                "step": 1,
                "focus": "练习重点",
                "content": "练习内容",
                "repetitions": "建议重复次数"
            }
        ],
        "annotated_text": "带音素标注的文本（用[]标注核心音素）"
    }
    ```

    **Analysis Result Structure - English (纯JSON格式，非markdown代码块)**:
    ```json
    {
        "title": "Tongue twister title/theme",
        "difficulty": "Difficulty level (Easy/Medium/Hard/Expert)",
        "core_phonemes": [
            {
                "phoneme": "Phoneme (e.g., /p/, /b/, /θ/, /ð/)",
                "ipa": "IPA symbol",
                "description": "Pronunciation description",
                "articulation": {
                    "manner": "Manner of articulation (e.g., plosive, fricative, nasal)",
                    "place": "Place of articulation (e.g., bilabial, alveolar, velar)",
                    "voicing": "Voiced/Voiceless"
                },
                "examples": ["Example words containing this phoneme"]
            }
        ],
        "acoustic_features": [
            {
                "feature": "Acoustic feature name",
                "description": "Feature description",
                "key_difference": "Key difference from similar sounds",
                "measurement": "Measurable acoustic indicators (e.g., VOT, F1/F2 frequency)"
            }
        ],
        "confusion_pairs": [
            {
                "pair": ["phoneme1", "phoneme2"],
                "difference": "Key distinction",
                "common_errors": "Common mistakes",
                "practice_tip": "Practice suggestion"
            }
        ],
        "pronunciation_tips": [
            {
                "tip": "Pronunciation tip",
                "target_sounds": ["Target phonemes"],
                "technique": "Specific technique",
                "practice_method": "Practice method"
            }
        ],
        "rhythm_pattern": {
            "beat_count": "Number of beats",
            "stress_pattern": "Stress pattern",
            "pause_points": ["Suggested pause positions"],
            "speed_suggestion": "Suggested speed"
        },
        "practice_sequence": [
            {
                "step": 1,
                "focus": "Practice focus",
                "content": "Practice content",
                "repetitions": "Suggested repetitions"
            }
        ],
        "annotated_text": "Text with phoneme annotations (mark core phonemes with [])"
    }
    ```

    **注意**:
    - `analysis_result` 为JSON字符串格式，需要再次解析为JSON对象
    - 发音部位和方式的术语基于国际语音学标准
    - VOT (Voice Onset Time) 是衡量塞音清浊的重要声学指标
    - F1/F2 指元音的第一、第二共振峰频率，用于区分不同元音
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


@router.post("/sentence-interpretation", response_model=SentenceInterpretationResponse)
async def analyze_sentence_interpretation(
    request: SentenceInterpretationRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> SentenceInterpretationResponse:
    """
    句子解读接口 - 提供中心内容、朗读重点和注意事项。

    本接口对提供的句子进行深度分析，提取：
    - 中心内容：句子的核心思想/主旨
    - 朗读重点：需要重读、强调的关键词或短语
    - 注意事项：朗读时的语气、停顿、语速、情感等要点

    **功能特性**:
    - 支持中英文句子分析
    - 智能提取朗读要点和注意事项
    - 可自定义分析要求

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "text": "待解读的句子内容...",
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
        "interpretation": {
            "center_content": "句子的中心内容",
            "reading_points": ["朗读重点1", "朗读重点2", "朗读重点3"],
            "reading_notes": ["注意事项1", "注意事项2", "注意事项3"]
        }
    }
    ```

    **Interpretation Result Structure** (JSON格式):
    ```json
    {
        "center_content": "句子的中心内容/主旨，用简短语言概括",
        "reading_points": [
            "朗读重点1",
            "朗读重点2",
            "朗读重点3"
        ],
        "reading_notes": [
            "注意事项1",
            "注意事项2",
            "注意事项3"
        ]
    }
    ```
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())

    try:
        analysis_result = await hunyuan_service.analyze_sentence_interpretation(
            text=request.text,
            custom_prompt=request.custom_prompt
        )

        # 解析 JSON
        import json
        import re
        try:
            json_match = re.search(r'\{[\s\S]*\}', analysis_result)
            if json_match:
                interpretation_data = json.loads(json_match.group())
            else:
                interpretation_data = json.loads(analysis_result)
        except json.JSONDecodeError:
            interpretation_data = {
                "center_content": "无法解析AI响应",
                "reading_points": [],
                "reading_notes": []
            }

        return SentenceInterpretationResponse(
            success=True,
            message="Analysis completed successfully",
            message_id=msg_id,
            interpretation=interpretation_data
        )

    except HTTPException:
        raise
    except Exception as e:
        return SentenceInterpretationResponse(
            success=False,
            message="Analysis failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/story-reading", response_model=StoryReadingResponse)
async def analyze_story_reading(
    request: StoryReadingRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> StoryReadingResponse:
    """
    故事阅读评测接口 - 分析用户围绕小故事的阅读表现。

    本接口通过ASR识别音频内容（带时间戳），对比原始故事文本，分析：
    - 结构完整性：开头、发展、高潮、结尾
    - 逻辑连贯性：时间跳跃、因果错误、事件遗漏、逻辑矛盾
    - 语言流畅度：长停顿、重复修正、填充词、句子完整度
    - 事件分布：各事件的时间位置和时长
    - 综合评分：基于各维度的综合打分
    - 待改进建议

    **功能特性**:
    - ASR带时间戳识别（WordInfo=1）
    - 对比原始故事文本分析内容完整性
    - 基于时间戳分析语言流畅度
    - 分析事件分布和时间分配

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/audio.mp3",
        "story_text": "从前有个小孩叫小明，他每天去河边钓鱼...",
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | audio_url | string | 是 | 音频文件URL |
    | story_text | string | 是 | 短故事文本，用户要围绕此故事发挥 |
    | message_id | string | 否 | 消息ID，不传则自动生成UUID |

    **Response**:
    ```json
    {
        "success": true,
        "message": "Analysis completed successfully",
        "message_id": "uuid",

        "structure_analysis": {
            "opening": "有",
            "development": "事件1+事件2",
            "climax": "无",
            "ending": "有但仓促",
            "overall_assessment": "结构基本完整，但缺少高潮，结尾仓促"
        },
        "logic_analysis": {
            "time_jumps": 0,
            "causal_errors": 0,
            "missing_events": 1,
            "logical_contradictions": 0,
            "overall_assessment": "逻辑连贯，遗漏了一个次要事件"
        },
        "fluency_analysis": {
            "long_pauses_count": 2,
            "repetition_count": 3,
            "filler_words_count": 15,
            "sentence_completion_rate": 85,
            "overall_assessment": "流畅度良好，有一些停顿和填充词"
        },
        "event_distribution": {
            "events": [
                {
                    "name": "事件1",
                    "start_time_ms": 0,
                    "end_time_ms": 22000,
                    "duration_seconds": 22,
                    "assessment": "主题明确"
                },
                {
                    "name": "事件2",
                    "start_time_ms": 22000,
                    "end_time_ms": 50000,
                    "duration_seconds": 28,
                    "assessment": "细节过多"
                },
                {
                    "name": "过渡",
                    "start_time_ms": 50000,
                    "end_time_ms": 55000,
                    "duration_seconds": 5,
                    "assessment": ""
                },
                {
                    "name": "结尾",
                    "start_time_ms": 55000,
                    "end_time_ms": 60000,
                    "duration_seconds": 5,
                    "assessment": "过短"
                }
            ],
            "transition_time": "5秒",
            "overall_assessment": "事件2描述过长，结尾仓促"
        },
        "improvements": [
            "事件2描述过长(28秒)",
            "缺少事件3",
            "结尾仓促，无总结"
        ],
        "overall_score": {
            "score": 72,
            "level": "良好",
            "comment": "故事结构完整但高潮不足，结尾仓促"
        },
        "asr_data": {
            "text": "识别的完整文本",
            "word_info_list": [
                {"word": "从前", "begin_time": 0, "end_time": 500, "duration": 500}
            ]
        },
        "error": null
    }
    ```
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # Download audio
        audio_data = await asr_service.download_audio(audio_url)

        # Recognize with word_info=1 to get timestamps
        asr_result = await asr_service.recognize_audio(
            audio_data,
            engine_type="16k_zh",
            word_info=1
        )

        speech_text = asr_result.get("text", "")
        word_info_list = asr_result.get("word_info_list", [])

        if not speech_text or not speech_text.strip():
            return StoryReadingResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                error="ASR returned empty text"
            )

        # Calculate audio duration from audio file
        audio_duration = await get_audio_duration(audio_data)
        if audio_duration is None and word_info_list:
            audio_duration = max(w.get("end_time", 0) for w in word_info_list) / 1000

        # Analyze story reading with Hunyuan
        analysis_result = await hunyuan_service.analyze_story_reading(
            speech_text=speech_text,
            story_text=request.story_text,
            word_info_list=word_info_list,
            audio_duration=audio_duration,
            language="zh"
        )

        return StoryReadingResponse(
            success=True,
            message="Analysis completed successfully",
            message_id=msg_id,
            structure_analysis=analysis_result.get("structure_analysis"),
            logic_analysis=analysis_result.get("logic_analysis"),
            fluency_analysis=analysis_result.get("fluency_analysis"),
            event_distribution=analysis_result.get("event_distribution"),
            improvements=analysis_result.get("improvements", []),
            overall_score=analysis_result.get("overall_score"),
            asr_data={
                "text": speech_text,
                "word_info_list": word_info_list
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return StoryReadingResponse(
            success=False,
            message="Analysis failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/tongue-twister-reading", response_model=TongueTwisterReadingResponse)
async def evaluate_tongue_twister_reading(
    request: TongueTwisterReadingRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> TongueTwisterReadingResponse:
    """
    绕口令/文章朗读评测接口 - 分析用户朗读的语音表现。

    本接口通过ASR识别音频内容（带时间戳），结合SOE发音评分，
    再由混元AI综合分析优势和待提升之处。

    支持两种评测类型：
    - **tongue_twister**（默认）：绕口令评测，侧重发音准确性、节奏感
    - **article**：文章朗读评测，侧重流畅度评分、语速分析、断句停顿、读错字

    **处理流程**:
    1. 下载音频文件
    2. ASR语音识别（带时间戳）与 SOE发音评测 并行执行
    3. 混元AI根据评测类型综合分析

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/audio.mp3",
        "tongue_twister_text": "八百标兵奔北坡，炮兵并排北边跑",
        "eval_type": "tongue_twister",
        "score_coeff": 1.0,
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 默认值 | 说明 |
    |--------|------|------|--------|------|
    | audio_url | string | 是 | - | 音频文件URL |
    | tongue_twister_text | string | 是 | - | 原文文本（绕口令或文章） |
    | eval_type | string | 否 | tongue_twister | 评测类型：tongue_twister / article |
    | score_coeff | float | 否 | 1.0 | SOE评分苛刻指数：1.0(宽松) 2.0(标准) 4.0(严格) |
    | message_id | string | 否 | 自动生成 | 消息ID |

    **Response 主要字段**:
    | 字段 | 说明 |
    |------|------|
    | speech_scores | SOE评分（准确度、流利度、完整度） |
    | statistics | 评测统计（总字数、平均准确度、低分字数） |
    | soe_words | SOE逐字评分详情 |
    | low_score_words | 低分字词列表（准确度<90分） |
    | soe_sentences | SOE句子级评分 |
    | soe_data | SOE完整原始数据 |
    | strengths | AI分析的优势列表 |
    | improvements | 待提升（多读/发音问题，article模式额外含读错字） |
    | fluency_analysis | 流畅度分析（article模式含评分、中断、重复读、卡壳） |
    | speech_rate_analysis | 语速分析（仅article模式，含分段语速） |
    | pause_analysis | 断句停顿分析（仅article模式） |
    | practice_suggestions | 练习建议 |
    | asr_data | ASR完整数据（text + word_info_list时间戳） |
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # Download audio
        audio_data = await asr_service.download_audio(audio_url)

        # Determine SOE eval_mode based on ref_text length
        # ref_text <= 120: eval_mode=2 (paragraph mode)
        # ref_text > 120: eval_mode=3 (free speech mode)
        ref_text = request.tongue_twister_text
        if len(ref_text) > 120:
            soe_eval_mode = 3
        else:
            soe_eval_mode = 2

        # Run ASR (with timestamps) and SOE in parallel
        asr_result, soe_result = await asyncio.gather(
            asr_service.recognize_audio(
                audio_data,
                engine_type="16k_zh",
                word_info=1
            ),
            soe_service.evaluate_audio(
                audio_data,
                ref_text=ref_text,
                eval_mode=soe_eval_mode,
                score_coeff=request.score_coeff,
                server_type=0
            )
        )

        # Extract ASR results
        speech_text = asr_result.get("text", "")
        word_info_list = asr_result.get("word_info_list", [])

        if not speech_text or not speech_text.strip():
            return TongueTwisterReadingResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                error="ASR returned empty text"
            )

        # Extract SOE results
        scores_data = soe_result.get("scores", {})
        low_score_words_data = soe_result.get("low_score_words", [])
        statistics_data = soe_result.get("statistics", {})
        soe_words_data = soe_result.get("words", [])
        soe_sentences_data = soe_result.get("sentences", [])

        # Calculate audio duration from audio file
        audio_duration = await get_audio_duration(audio_data)
        if audio_duration is None and word_info_list:
            audio_duration = max(w.get("end_time", 0) for w in word_info_list) / 1000

        # Build typed score objects for response
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

        # Call Hunyuan for AI analysis
        analysis_result = await hunyuan_service.analyze_tongue_twister_reading(
            speech_text=speech_text,
            tongue_twister_text=request.tongue_twister_text,
            word_info_list=word_info_list,
            low_score_words=low_score_words_data,
            scores_data=scores_data,
            statistics_data=statistics_data,
            audio_duration=audio_duration,
            language="zh",
            eval_type=request.eval_type
        )

        return TongueTwisterReadingResponse(
            success=True,
            message="Analysis completed successfully",
            message_id=msg_id,
            speech_scores=speech_scores,
            statistics=statistics,
            soe_words=soe_words_data,
            low_score_words=low_score_words_data,
            soe_sentences=soe_sentences_data,
            soe_data=soe_result,
            strengths=analysis_result.get("strengths", []),
            improvements=analysis_result.get("improvements"),
            fluency_analysis=analysis_result.get("fluency_analysis"),
            overall_assessment=analysis_result.get("overall_assessment"),
            practice_suggestions=analysis_result.get("practice_suggestions", []),
            speech_rate_analysis=analysis_result.get("speech_rate_analysis"),
            pause_analysis=analysis_result.get("pause_analysis"),
            asr_data={
                "text": speech_text,
                "word_info_list": word_info_list
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return TongueTwisterReadingResponse(
            success=False,
            message="Analysis failed",
            message_id=msg_id,
            error=str(e)
        )



@router.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    request: VoiceChatRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> VoiceChatResponse:
    """
    语音对话接口 - 支持情景对话（面试、日常对话等）。

    支持两种模式：
    - **traditional**: ASR转文字 → LLM对话 → TTS语音
    - **multimodal**: 多模态模型直接处理音频 → TTS语音

    **场景设定优先级**: 自定义system_prompt > 预设scene > 默认通用对话

    **预设场景**:
    | scene | 说明 |
    |-------|------|
    | interview | 面试官角色，追问评价 |
    | daily | 日常对话伙伴，口语化交流 |
    | customer_service | 客服人员，处理咨询 |

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/user-speech.mp3",
        "session_id": null,
        "mode": "traditional",
        "messages": null,
        "system_prompt": null,
        "scene": "interview",
        "voice_type": 101001,
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 默认值 | 说明 |
    |--------|------|------|--------|------|
    | audio_url | string | 是 | - | 用户语音文件URL |
    | session_id | string | 否 | null | 会话ID，传入则复用服务端对话历史 |
    | mode | string | 否 | traditional | traditional(ASR+LLM) / multimodal(多模态模型) |
    | messages | array | 否 | null | 对话历史（兼容旧模式，优先使用服务端会话） |
    | system_prompt | string | 否 | null | 自定义系统提示词，优先级高于scene |
    | scene | string | 否 | null | 预设场景：interview/daily/customer_service |
    | voice_type | int | 否 | 101001 | TTS音色：101001(智瑜-女) 101005(智华-男) |
    | message_id | string | 否 | 自动生成 | 消息ID |

    **多轮对话使用说明**:
    - 首轮：不传session_id，服务端自动创建会话并返回session_id
    - 后续轮：传入session_id，服务端自动管理对话历史
    - 场景切换：调用 POST /evaluation/voice-chat/scene 接口
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # 1. Get or create session
        session = await chat_session_manager.get_or_create_session(
            session_id=request.session_id,
            scene=request.scene,
            system_prompt=request.system_prompt,
            mode=request.mode,
            voice_type=request.voice_type,
        )

        # 2. Process based on mode
        user_text = None
        asr_data = None

        if session.mode == "multimodal":
            # Multimodal mode: send audio directly to model
            chat_result = await hunyuan_service.chat_multimodal(
                audio_url=audio_url,
                messages=session.messages,
                system_prompt=session.system_prompt,
                temperature=0.7,
            )
        else:
            # Traditional mode: ASR → text → LLM
            audio_data = await asr_service.download_audio(audio_url)
            asr_result = await asr_service.recognize_audio(
                audio_data,
                engine_type="16k_zh",
                word_info=0
            )
            user_text = asr_result.get("text", "")

            if not user_text or not user_text.strip():
                return VoiceChatResponse(
                    success=False,
                    message="音频内容为空，未识别到有效语音",
                    message_id=msg_id,
                    session_id=session.session_id,
                    error="ASR returned empty text"
                )

            # Build messages for LLM
            hunyuan_messages = [{"role": "user", "content": user_text}]
            chat_result = await hunyuan_service.chat(
                [{"role": "system", "content": session.system_prompt}] + session.messages + hunyuan_messages,
                temperature=0.7,
            )
            asr_data = {"text": user_text}

        assistant_text = chat_result.get("content", "")

        # 3. Update session history
        await chat_session_manager.append_message(session.session_id, "user", user_text or "[audio]")
        await chat_session_manager.append_message(session.session_id, "assistant", assistant_text)

        # 4. TTS: convert AI response to audio
        audio_bytes = await tts_service.synthesize(
            text=assistant_text,
            voice_type=session.voice_type,
            codec="mp3"
        )
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return VoiceChatResponse(
            success=True,
            message="Chat completed successfully",
            message_id=msg_id,
            session_id=session.session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            audio_base64=audio_base64,
            asr_data=asr_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        return VoiceChatResponse(
            success=False,
            message="Chat failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/voice-chat/text", response_model=VoiceChatResponse)
async def voice_text_chat(
    request: VoiceTextChatRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> VoiceChatResponse:
    """
    Text-only scene chat endpoint.

    This endpoint uses the same scene/session context as /voice-chat, but skips
    ASR and accepts user text directly.
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())
    user_text = request.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        session = await chat_session_manager.get_or_create_session(
            session_id=request.session_id,
            scene=request.scene,
            system_prompt=request.system_prompt,
            mode="text",
            voice_type=request.voice_type,
        )

        history = list(session.messages)
        if not history and request.messages:
            history = [
                {"role": item.role, "content": item.content}
                for item in request.messages
                if item.role and item.content
            ]

        messages = (
            [{"role": "system", "content": session.system_prompt}]
            + history
            + [{"role": "user", "content": user_text}]
        )

        chat_result = await hunyuan_service.chat(
            messages,
            temperature=0.7,
        )
        assistant_text = chat_result.get("content", "")

        await chat_session_manager.append_message(session.session_id, "user", user_text)
        await chat_session_manager.append_message(session.session_id, "assistant", assistant_text)

        audio_base64 = None
        if request.enable_tts and assistant_text:
            audio_bytes = await tts_service.synthesize(
                text=assistant_text,
                voice_type=session.voice_type,
                codec="mp3"
            )
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return VoiceChatResponse(
            success=True,
            message="Text chat completed successfully",
            message_id=msg_id,
            session_id=session.session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            audio_base64=audio_base64,
            asr_data=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        return VoiceChatResponse(
            success=False,
            message="Text chat failed",
            message_id=msg_id,
            error=str(e)
        )


@router.post("/voice-chat/scene")
async def switch_voice_chat_scene(
    session_id: str = Form(..., description="会话ID"),
    scene: str = Form(..., description="新场景：interview/daily/customer_service"),
    system_prompt: Optional[str] = Form(None, description="自定义系统提示词（优先级高于scene）"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
):
    """
    会话级场景切换。

    切换指定会话的对话场景，后续对话自动使用新场景的系统提示词。

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Form data**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | session_id | string | 是 | 会话ID |
    | scene | string | 是 | 新场景：interview/daily/customer_service |
    | system_prompt | string | 否 | 自定义系统提示词，优先级高于scene |
    """
    verify_signature(x_signature)

    session = await chat_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    updated = await chat_session_manager.update_scene(
        session_id=session_id,
        scene=scene,
        system_prompt=system_prompt,
    )

    if not updated:
        raise HTTPException(status_code=404, detail="Failed to update session")

    return {
        "success": True,
        "session_id": session_id,
        "scene": updated.scene,
        "system_prompt": updated.system_prompt,
    }


@router.post("/opinion-statement", response_model=OpinionStatementResponse)
async def generate_opinion_statement_report(
    request: OpinionStatementRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> OpinionStatementResponse:
    """
    一分钟观点陈述评测（同步接口）。

    传入音频URL，接口自动进行ASR语音识别和SOE发音评测，
    再由混元AI针对"一分钟观点陈述"场景生成专项评测报告。

    **处理流程**:
    1. 下载音频文件
    2. ASR语音识别（带时间戳）与 SOE发音评测（eval_mode=3自由说）并行执行
    3. 混元AI综合分析观点陈述表现（含词级时间戳用于时间节奏分析）

    **评测维度**:
    - 观点明确性（20%）：是否开门见山、观点是否鲜明，识别回避式开头
    - 逻辑清晰度（20%）：是否存在逻辑跳跃、矛盾、论据堆砌
    - 表达精炼度（15%）：口头禅频率、废话比例、有效内容占比
    - 流畅度（15%）：基于SOE发音流利度数据
    - 语速（10%）：语速是否在合理区间
    - 结构完整度（10%）：观点→理由→举例→总结 四要素
    - 时间节奏（10%）：时长是否合适、前后半段语速变化、是否慌张加速

    **请求参数**:
    | 参数 | 类型 | 必填 | 默认值 | 说明 |
    |------|------|------|--------|------|
    | audio_url | string | 是 | - | 音频文件URL |
    | ref_text | string | 否 | 空 | 参考文本，用于SOE对照。不传/空→eval_mode=3自由说；≤120字→eval_mode=2段落模式 |
    | topic | string | 否 | - | 陈述题目/话题，传入后会分析贴题性并在overall_scores中返回topic_relevance_score |
    | score_coeff | float | 否 | 1.0 | SOE评分苛刻指数：1.0(宽松) ~ 4.0(严格) |
    | language | string | 否 | zh | 语言：zh中文、en英文 |
    | message_id | string | 否 | 自动UUID | 消息ID |

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Response 主要字段**:
    | 字段 | 说明 |
    |------|------|
    | speech_text | ASR语音转写文本 |
    | speech_rate | 语速（字/分钟） |
    | speech_scores | SOE评分（pronunciation_accuracy/fluency/completion/suggested_score） |
    | statistics | 评测统计（total_words/average_accuracy/low_score_count） |
    | low_score_words | 低分字词列表（准确度<90分） |
    | evaluation_report | AI评测报告JSON，结构见下方 |

    **evaluation_report 结构**:
    ```json
    {
        "viewpoint_analysis": {
            "has_clear_viewpoint": true,
            "viewpoint_summary": "核心观点概括",
            "opening_type": "直接亮明观点/渐进引入/回避式开头/模糊开头",
            "opening_quote": "开头原文前30字",
            "evasion_signals": ["回避性表达列表"],
            "score": 85,
            "assessment": "观点表达评价"
        },
        "structure_completeness": {
            "score": 80,
            "has_viewpoint": true, "has_reason": true,
            "has_example": false, "has_summary": true,
            "structure_pattern": "观点→理由→总结（缺少举例）",
            "missing_parts": ["举例支撑"],
            "assessment": "结构完整度评价"
        },
        "logic_clarity": {
            "score": 75,
            "logic_jumps": [...],
            "contradictions": [...],
            "argument_piling": {"detected": false, "description": "..."},
            "reasoning_chain": "论证链条描述",
            "assessment": "逻辑清晰度评价"
        },
        "time_rhythm": {
            "score": 70,
            "total_duration_seconds": 58.5,
            "duration_level": "适中",
            "first_half_rate": 180, "second_half_rate": 220,
            "rate_change": "加速",
            "panic_acceleration": false,
            "time_allocation": {"opening_seconds": 5, "body_seconds": 45, "closing_seconds": 8},
            "assessment": "时间节奏评价"
        },
        "expression_redundancy": {
            "score": 65,
            "filler_words": [{"word": "然后", "count": 5}],
            "total_filler_count": 8,
            "filler_ratio": "每分钟8次",
            "redundant_expressions": [...],
            "effective_content_ratio": "75%",
            "assessment": "表达冗余度评价"
        },
        "overall_scores": {
            "overall_score": 78,
            "viewpoint_score": 85, "structure_score": 80, "logic_score": 75,
            "fluency_score": 80, "speech_rate_score": 75,
            "expression_score": 65, "time_rhythm_score": 70,
            "pronunciation_accuracy": 88.5, "pronunciation_fluency": 82.3,
            "pronunciation_completion": 95.0, "suggested_score": 85.0,
            "speech_rate_value": 195,
            "speech_rate_level": "良好",
            "level": "良好",
            "one_sentence_comment": "观点鲜明但举例不足"
        },
        "structure_visualization": {
            "arguments": ["论点1", "论点2"],
            "conclusion": "结论要点"
        },
        "strengths": ["优点1", "优点2"],
        "improvements": ["改进建议1", "改进建议2"],
        "practice_tips": [{"dimension": "结构组织", "tip": "练习方法"}]
    }
    ```
    """
    # Verify signature from header
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # 1. Download audio
        audio_data = await asr_service.download_audio(audio_url)

        # 2. Run ASR (with timestamps) and SOE in parallel
        engine_type = "16k_zh" if request.language == "zh" else "16k_en"
        soe_ref_text = request.ref_text or ""

        asr_result, soe_result = await asyncio.gather(
            asr_service.recognize_audio(
                audio_data,
                engine_type=engine_type,
                word_info=1
            ),
            soe_service.evaluate_audio(
                audio_data,
                ref_text=soe_ref_text,
                eval_mode=3,
                score_coeff=request.score_coeff,
                server_type=0
            )
        )

        # 3. Extract ASR results
        speech_text = asr_result.get("text", "")
        word_info_list = asr_result.get("word_info_list", [])

        if not speech_text or not speech_text.strip():
            return OpinionStatementResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                audio_url=audio_url,
                error="ASR returned empty text"
            )

        # 4. Extract SOE results
        scores_data = soe_result.get("scores", {})
        low_score_words_data = soe_result.get("low_score_words", [])
        statistics_data = soe_result.get("statistics", {})

        # Build typed score objects for response
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

        # 5. Calculate audio duration and speech rate
        real_audio_duration = await get_audio_duration(audio_data)
        audio_duration = real_audio_duration
        if audio_duration is None and word_info_list:
            audio_duration = max(w.get("end_time", 0) for w in word_info_list) / 1000
        speech_rate = None

        # Only calculate speech rate with real audio duration (not word-timestamp fallback)
        if real_audio_duration and real_audio_duration > 0 and speech_text:
            if request.language == "zh":
                punctuation = string.punctuation + '。，！？、；：""''（）【】《》…—'
                char_count = len([c for c in speech_text if c not in punctuation and not c.isspace()])
            else:
                char_count = len(speech_text.split())
            speech_rate = round(char_count / (real_audio_duration / 60), 1)

        # 6. Generate opinion statement report via Hunyuan
        evaluation_report = await hunyuan_service.generate_opinion_statement_report(
            speech_text=speech_text,
            speech_scores=scores_data,
            low_score_words=low_score_words_data,
            statistics=statistics_data,
            topic=request.topic,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            word_info_list=word_info_list,
            language=request.language
        )

        return OpinionStatementResponse(
            success=True,
            message="Opinion statement report generated successfully",
            message_id=msg_id,
            audio_url=audio_url,
            speech_text=speech_text,
            speech_rate=speech_rate,
            speech_scores=speech_scores,
            statistics=statistics,
            low_score_words=[
                WordScore(word=w.get("word", ""), accuracy=w.get("accuracy", 0), fluency=w.get("fluency", 0))
                for w in low_score_words_data
            ] if low_score_words_data else None,
            evaluation_report=evaluation_report
        )

    except HTTPException:
        raise
    except Exception as e:
        return OpinionStatementResponse(
            success=False,
            message="Opinion statement report generation failed",
            message_id=msg_id,
            audio_url=audio_url,
            error=str(e)
        )

@router.post("/impromptu-reaction", response_model=ImpromptuReactionResponse)
async def evaluate_impromptu_reaction(
    request: ImpromptuReactionRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> ImpromptuReactionResponse:
    """
    即兴反应评测接口（同步接口）。

    传入音频URL和场景/题目，接口自动进行ASR语音识别和SOE发音评测（eval_mode=3自由说），
    再由混元AI针对"即兴反应"场景进行犀利、结构化的专项评测。

    **处理流程**:
    1. 下载音频文件
    2. ASR语音识别（带时间戳）与 SOE发音评测（eval_mode=3自由说）并行执行
    3. 混元AI综合分析即兴反应表现（含词级时间戳用于反应速度分析）

    **评测维度**:
    - 反应速度：开口时间（基于时间戳）、慌乱信号、思考停顿
    - 结构形成：前15秒内是否建立主线、结构信号词识别
    - 内容相关性：是否切题、跑题部分识别
    - 逻辑连贯度：思维跳跃、过渡质量
    - 表达冗余度：口头禅统计、冗余度等级、有效内容占比

    **请求参数**:
    | 参数 | 类型 | 必填 | 默认值 | 说明 |
    |------|------|------|--------|------|
    | audio_url | string | 是 | - | 音频文件URL |
    | scenario | string | 是 | - | 即兴反应的触发情境/题目 |
    | score_coeff | float | 否 | 3.5 | SOE评分苛刻指数：1.0(宽松) ~ 4.0(严格) |
    | language | string | 否 | zh | 语言：zh中文、en英文 |
    | message_id | string | 否 | 自动UUID | 消息ID |

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Response 主要字段**:
    | 字段 | 说明 |
    |------|------|
    | speech_text | ASR语音转写文本 |
    | speech_rate | 语速（字/分钟） |
    | speech_scores | SOE评分（pronunciation_accuracy/fluency/completion/suggested_score） |
    | statistics | 评测统计（total_words/average_accuracy/low_score_count） |
    | low_score_words | 低分字词列表（准确度<90分） |
    | evaluation_report | AI评测报告JSON，结构见下方 |

    **evaluation_report 结构**:
    ```json
    {
        "reaction_speed": {
            "first_word_time_ms": 450,
            "opening_speed": "果断开口/犹豫拖延/大量填充词起手",
            "panic_signals": false,
            "thinking_pauses": [
                {"before_word": "我", "after_word": "认为", "pause_duration_ms": 1200, "position_time_ms": 3500}
            ],
            "assessment": "起步反应速度与情绪表现评价"
        },
        "structure_formation": {
            "formed_in_15s": true,
            "structure_signal": "我从两个方面来说",
            "structure_pattern": "总分总",
            "has_opening": true, "has_body": true, "has_closing": true,
            "assessment": "结构形成评价"
        },
        "content_relevance": {
            "topic_relevance": "紧扣主题/略微偏题/完全跑题",
            "on_topic": true, "topic_drift": false,
            "off_topic_parts": [],
            "relevance_description": "相关性分析",
            "assessment": "内容相关性评价"
        },
        "logic_coherence": {
            "coherence_level": "流畅连贯/基本连贯/偶有跳跃/逻辑混乱",
            "logic_jumps": [{"from_point": "...", "to_point": "...", "description": "..."}],
            "transition_quality": "过渡质量评价",
            "assessment": "逻辑连贯度评价"
        },
        "expression_redundancy": {
            "filler_words": [{"word": "嗯", "count": 3}, {"word": "然后", "count": 5}],
            "total_filler_count": 8,
            "filler_ratio": "每分钟8次",
            "redundancy_level": "极低/正常/偏高/极高",
            "effective_content_ratio": "80%",
            "assessment": "表达冗余度评价"
        },
        "overall_scores": {
            "overall_score": 75,
            "pronunciation_accuracy": 88.5, "pronunciation_fluency": 82.3,
            "pronunciation_completion": 95.0, "suggested_score": 85.0,
            "speech_rate_value": 195,
            "speech_rate_level": "良好",
            "level": "良好",
            "one_sentence_comment": "反应迅速但主线略显松散"
        },
        "structure_visualization": {
            "key_points": ["要点1", "要点2"],
            "conclusion": "结论"
        },
        "strengths": ["优点1", "优点2"],
        "improvements": ["改进建议1", "改进建议2"],
        "next_action": "在开头先用一句话说清你的核心观点"
    }
    ```
    """
    # Verify signature from header
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # 1. Download audio
        audio_data = await asr_service.download_audio(audio_url)

        # 2. Run ASR (with timestamps) and SOE in parallel
        # Impromptu reaction uses eval_mode=3 (free speech mode)
        engine_type = "16k_zh" if request.language == "zh" else "16k_en"

        asr_result, soe_result = await asyncio.gather(
            asr_service.recognize_audio(
                audio_data,
                engine_type=engine_type,
                word_info=1
            ),
            soe_service.evaluate_audio(
                audio_data,
                ref_text="",
                eval_mode=3,
                score_coeff=request.score_coeff,
                server_type=0
            )
        )

        # 3. Extract ASR results
        speech_text = asr_result.get("text", "")
        word_info_list = asr_result.get("word_info_list", [])

        if not speech_text or not speech_text.strip():
            return ImpromptuReactionResponse(
                success=False,
                message="音频内容为空，未识别到有效语音",
                message_id=msg_id,
                audio_url=audio_url,
                error="ASR returned empty text"
            )

        # 4. Extract SOE results
        scores_data = soe_result.get("scores", {})
        low_score_words_data = soe_result.get("low_score_words", [])
        statistics_data = soe_result.get("statistics", {})

        # Build typed score objects for response
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

        # 5. Calculate audio duration and speech rate
        real_audio_duration = await get_audio_duration(audio_data)
        audio_duration = real_audio_duration
        if audio_duration is None and word_info_list:
            audio_duration = max(w.get("end_time", 0) for w in word_info_list) / 1000
        speech_rate = None

        # Only calculate speech rate with real audio duration (not word-timestamp fallback)
        if real_audio_duration and real_audio_duration > 0 and speech_text:
            if request.language == "zh":
                punctuation = string.punctuation + '。，！？、；：""''（）【】《》…—'
                char_count = len([c for c in speech_text if c not in punctuation and not c.isspace()])
            else:
                char_count = len(speech_text.split())
            speech_rate = round(char_count / (real_audio_duration / 60), 1)

        # 6. Generate impromptu reaction report via Hunyuan
        evaluation_report = await hunyuan_service.generate_impromptu_reaction_report(
            speech_text=speech_text,
            speech_scores=scores_data,
            low_score_words=low_score_words_data,
            statistics=statistics_data,
            scenario=request.scenario,
            speech_rate=speech_rate,
            audio_duration=audio_duration,
            word_info_list=word_info_list,
            language=request.language
        )

        return ImpromptuReactionResponse(
            success=True,
            message="Impromptu reaction report generated successfully",
            message_id=msg_id,
            audio_url=audio_url,
            speech_text=speech_text,
            speech_rate=speech_rate,
            speech_scores=speech_scores,
            statistics=statistics,
            low_score_words=[
                WordScore(word=w.get("word", ""), accuracy=w.get("accuracy", 0), fluency=w.get("fluency", 0))
                for w in low_score_words_data
            ] if low_score_words_data else None,
            evaluation_report=evaluation_report
        )

    except HTTPException:
        raise
    except Exception as e:
        return ImpromptuReactionResponse(
            success=False,
            message="Impromptu reaction report generation failed",
            message_id=msg_id,
            audio_url=audio_url,
            error=str(e)
        )
