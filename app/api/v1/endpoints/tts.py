# -*- coding: utf-8 -*-
"""
TTS (Text-to-Speech) endpoint - returns complete audio file.
"""
from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import Response
from typing import Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import aes_service
from app.services.tencent import tts_service

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


class TTSRequest(BaseModel):
    """TTS request body."""
    text: str = Field(..., description="Text to synthesize", max_length=5000)
    voice_type: int = Field(
        default=101001,
        description="Voice type ID. Common options: 101001(智瑜-女), 101005(智华-男), 101050(英文女), 101051(英文男)"
    )
    codec: str = Field(
        default="mp3",
        description="Audio format: 'mp3' or 'pcm'"
    )
    sample_rate: int = Field(
        default=16000,
        description="Sample rate: 8000 or 16000"
    )
    speed: float = Field(
        default=1.0,
        ge=-2.0,
        le=6.0,
        description="Speech speed, range -2.0 to 6.0"
    )
    volume: float = Field(
        default=0.0,
        ge=-10.0,
        le=10.0,
        description="Volume adjustment in dB, range -10.0 to 10.0"
    )


@router.post("/synthesize")
async def synthesize_speech(
    request: TTSRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
):
    """
    Synthesize text to speech and return complete audio file.

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Request body**:
    ```json
    {
        "text": "要合成的文本",
        "voice_type": 101001,
        "codec": "mp3",
        "sample_rate": 16000,
        "speed": 1.0,
        "volume": 0.0
    }
    ```

    **Voice Types**:
    - 101001: 智瑜 (通用女声)
    - 101002: 智聆 (通用女声)
    - 101003: 智美 (客服女声)
    - 101004: 智云 (通用女声)
    - 101005: 智华 (通用男声)
    - 101006: 智龙 (新闻男声)
    - 101007: 智明 (新闻男声)
    - 101050: WeJack (英文女声)
    - 101051: WeRose (英文男声)

    **Response**: Complete audio file (Content-Type based on codec)
    """
    verify_signature(x_signature)

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.codec not in ("mp3", "pcm"):
        raise HTTPException(status_code=400, detail="Codec must be 'mp3' or 'pcm'")

    if request.sample_rate not in (8000, 16000):
        raise HTTPException(status_code=400, detail="Sample rate must be 8000 or 16000")

    # Synthesize and get complete audio data
    audio_data = await tts_service.synthesize(
        text=request.text,
        voice_type=request.voice_type,
        codec=request.codec,
        sample_rate=request.sample_rate,
        speed=request.speed,
        volume=request.volume
    )

    content_type = "audio/mpeg" if request.codec == "mp3" else "audio/wav"

    return Response(
        content=audio_data,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{request.codec}",
            "Content-Length": str(len(audio_data))
        }
    )


@router.get("/synthesize")
async def synthesize_speech_get(
    text: str = Query(..., description="Text to synthesize", max_length=5000),
    voice_type: int = Query(101001, description="Voice type ID"),
    codec: str = Query("mp3", description="Audio format: 'mp3' or 'pcm'"),
    sample_rate: int = Query(16000, description="Sample rate: 8000 or 16000"),
    speed: float = Query(1.0, ge=-2.0, le=6.0, description="Speech speed"),
    volume: float = Query(0.0, ge=-10.0, le=10.0, description="Volume adjustment in dB"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
):
    """
    Synthesize text to speech and return complete audio file (GET method).

    Same as POST /synthesize but with query parameters.
    """
    verify_signature(x_signature)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if codec not in ("mp3", "pcm"):
        raise HTTPException(status_code=400, detail="Codec must be 'mp3' or 'pcm'")

    if sample_rate not in (8000, 16000):
        raise HTTPException(status_code=400, detail="Sample rate must be 8000 or 16000")

    audio_data = await tts_service.synthesize(
        text=text,
        voice_type=voice_type,
        codec=codec,
        sample_rate=sample_rate,
        speed=speed,
        volume=volume
    )

    content_type = "audio/mpeg" if codec == "mp3" else "audio/wav"

    return Response(
        content=audio_data,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{codec}",
            "Content-Length": str(len(audio_data))
        }
    )


@router.get("/voices")
async def list_voices(
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> dict:
    """
    List available voice types.

    **Headers**:
    - X-Signature: AES encrypted signature (required)

    **Response**:
    ```json
    {
        "voices": [
            {"id": 101001, "name": "智瑜", "gender": "female", "language": "zh"},
            ...
        ]
    }
    ```
    """
    verify_signature(x_signature)

    voices = [
        {"id": 101001, "name": "智瑜", "gender": "female", "language": "zh", "description": "通用女声"},
        {"id": 101002, "name": "智聆", "gender": "female", "language": "zh", "description": "通用女声"},
        {"id": 101003, "name": "智美", "gender": "female", "language": "zh", "description": "客服女声"},
        {"id": 101004, "name": "智云", "gender": "female", "language": "zh", "description": "通用女声"},
        {"id": 101005, "name": "智华", "gender": "male", "language": "zh", "description": "通用男声"},
        {"id": 101006, "name": "智龙", "gender": "male", "language": "zh", "description": "新闻男声"},
        {"id": 101007, "name": "智明", "gender": "male", "language": "zh", "description": "新闻男声"},
        {"id": 101050, "name": "WeJack", "gender": "female", "language": "en", "description": "英文女声"},
        {"id": 101051, "name": "WeRose", "gender": "male", "language": "en", "description": "英文男声"},
    ]

    return {"voices": voices}
