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
    VoiceChatResponse
)
from app.services.tencent import asr_service, soe_service, hunyuan_service, tts_service

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

        # Calculate audio duration from word timestamps
        audio_duration = None
        if word_info_list:
            last_word = word_info_list[-1]
            audio_duration = last_word.get("end_time", 0) / 1000  # Convert ms to seconds

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
    绕口令语音评测接口 - 分析用户朗读绕口令的语音表现。

    本接口通过ASR识别音频内容（带时间戳），结合SOE发音评分，
    再由混元AI综合分析优势和待提升之处：
    - 优势：发音准确性、流畅度、节奏感等方面的亮点
    - 待提升-多读：朗读中多出原文没有的字词
    - 待提升-漏读：原文中有但朗读中遗漏的字词
    - 待提升-发音问题：结合SOE低分字词分析具体发音问题
    - 流畅度分析：基于时间戳分析长停顿、节奏和语速
    - 练习建议：针对具体问题给出可操作的练习方法

    **功能特性**:
    - ASR带时间戳识别（WordInfo=1）
    - SOE发音评测（句子模式，对齐原文评分）
    - ASR与SOE并行执行提升性能
    - 混元AI综合分析优势和待提升

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/tongue-twister.mp3",
        "tongue_twister_text": "八百标兵奔北坡，炮兵并排北边跑",
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | audio_url | string | 是 | 绕口令音频文件URL |
    | tongue_twister_text | string | 是 | 绕口令原文文本 |
    | message_id | string | 否 | 消息ID，不传则自动生成UUID |
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # Download audio
        audio_data = await asr_service.download_audio(audio_url)

        # Run ASR (with timestamps) and SOE (with ref_text) in parallel
        asr_result, soe_result = await asyncio.gather(
            asr_service.recognize_audio(
                audio_data,
                engine_type="16k_zh",
                word_info=1
            ),
            soe_service.evaluate_audio(
                audio_data,
                ref_text=request.tongue_twister_text,
                eval_mode=1,
                score_coeff=request.score_coeff,
                server_type=0
            )
        )

        # Extract ASR results
        speech_text = asr_result.get("text", "")
        word_info_list = asr_result.get("word_info_list", [])

        # Extract SOE results
        scores_data = soe_result.get("scores", {})
        low_score_words_data = soe_result.get("low_score_words", [])
        statistics_data = soe_result.get("statistics", {})
        soe_words_data = soe_result.get("words", [])
        soe_sentences_data = soe_result.get("sentences", [])

        # Calculate audio duration from timestamps
        audio_duration = None
        if word_info_list:
            last_word = word_info_list[-1]
            audio_duration = last_word.get("end_time", 0) / 1000

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


# 预设场景系统提示词
VOICE_CHAT_SCENE_PROMPTS = {
    "interview": "你是一位专业的面试官，正在对候选人进行面试。请根据候选人的回答进行追问、评价或提出新的面试问题。语气专业但友好，回答简洁有针对性。",
    "daily": "你是一个友好的对话伙伴，正在进行日常中文对话练习。请用自然、口语化的方式回应，适当引导话题，保持对话轻松有趣。回答简洁，适合口语交流。",
    "customer_service": "你是一位专业的客服人员，正在处理客户的咨询和问题。请耐心倾听，准确回答问题，提供有效的解决方案。语气礼貌专业。",
}

DEFAULT_VOICE_CHAT_PROMPT = "你是一个智能对话助手，正在与用户进行语音对话。请用自然、简洁的方式回应，回答适合语音播报的长度，避免过长的文字。"


@router.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    request: VoiceChatRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> VoiceChatResponse:
    """
    语音对话接口 - 支持情景对话（面试、日常对话等）。

    接收用户语音URL，ASR转文字（带时间戳），发送给AI对话，
    再将AI回复通过TTS转为语音，返回AI文本 + 音频Base64数据。

    **功能特性**:
    - ASR带时间戳识别（WordInfo=1）
    - 支持多轮对话（客户端传递历史messages）
    - 支持预设场景（interview/daily/customer_service）和自定义system_prompt
    - AI回复通过TTS转语音，返回Base64编码的mp3音频

    **Headers**:
    - X-Signature: AES加密签名（必填）

    **Request body**:
    ```json
    {
        "audio_url": "https://example.com/user-speech.mp3",
        "messages": [
            {"role": "user", "content": "你好，我来面试的"},
            {"role": "assistant", "content": "你好，请先做一下自我介绍吧"}
        ],
        "system_prompt": null,
        "scene": "interview",
        "voice_type": 101001,
        "message_id": "可选的消息ID"
    }
    ```

    **参数说明**:
    | 参数名 | 类型 | 必填 | 说明 |
    |--------|------|------|------|
    | audio_url | string | 是 | 用户语音文件URL |
    | messages | array | 否 | 对话历史（不含本次语音） |
    | system_prompt | string | 否 | 自定义系统提示词，优先级高于scene |
    | scene | string | 否 | 预设场景：interview/daily/customer_service |
    | voice_type | int | 否 | TTS音色ID，默认101001(智瑜-女) |
    | message_id | string | 否 | 消息ID，不传则自动生成UUID |
    """
    verify_signature(x_signature)

    msg_id = request.message_id or str(uuid.uuid4())
    audio_url = str(request.audio_url)

    try:
        # 1. Download and recognize audio with timestamps
        audio_data = await asr_service.download_audio(audio_url)
        asr_result = await asr_service.recognize_audio(
            audio_data,
            engine_type="16k_zh",
            word_info=1
        )

        speech_text = asr_result.get("text", "")
        word_info_list = asr_result.get("word_info_list", [])

        # 2. Build conversation messages for Hunyuan
        # Determine system prompt: custom > scene preset > default
        if request.system_prompt:
            system_content = request.system_prompt
        elif request.scene and request.scene in VOICE_CHAT_SCENE_PROMPTS:
            system_content = VOICE_CHAT_SCENE_PROMPTS[request.scene]
        else:
            system_content = DEFAULT_VOICE_CHAT_PROMPT

        hunyuan_messages = [{"Role": "system", "Content": system_content}]

        # Append conversation history
        if request.messages:
            for msg in request.messages:
                hunyuan_messages.append({
                    "Role": msg.role,
                    "Content": msg.content
                })

        # Append current user message from ASR
        hunyuan_messages.append({"Role": "user", "Content": speech_text})

        # 3. Get AI response
        chat_result = await hunyuan_service.chat(hunyuan_messages, temperature=0.7)
        assistant_text = chat_result.get("content", "")

        # 4. TTS: convert AI response to audio and encode as base64
        audio_bytes = await tts_service.synthesize(
            text=assistant_text,
            voice_type=request.voice_type,
            codec="mp3"
        )
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return VoiceChatResponse(
            success=True,
            message="Chat completed successfully",
            message_id=msg_id,
            user_text=speech_text,
            assistant_text=assistant_text,
            audio_base64=audio_base64,
            asr_data={
                "text": speech_text,
                "word_info_list": word_info_list
            }
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
