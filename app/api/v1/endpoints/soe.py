# -*- coding: utf-8 -*-
"""
Simple SOE (Speech Oral Evaluation) API endpoints.

These endpoints only perform speech evaluation without ASR and AI report generation.
Ported from tl_speech project.
"""
import uuid
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import aes_service
from app.schemas.soe import SOEUrlRequest, SOEResponse, AudioMeta
from app.services.tencent.soe import soe_service
from app.services.tencent.audio import convert_audio_to_wav, get_audio_duration

router = APIRouter()

# Thread pool for sync SDK calls
executor = ThreadPoolExecutor(max_workers=10)

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Maximum download file size (50MB)
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024


def verify_signature(signature: Optional[str]) -> None:
    """Verify AES signature from header, raise HTTPException if invalid."""
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Signature header"
        )

    success, message = aes_service.verify_signature(
        signature,
        max_age_seconds=settings.request_expire_seconds
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )


async def process_audio(audio_data: bytes, filename: str) -> tuple[bytes, dict]:
    """
    Process audio: convert to 16kHz mono WAV format.

    Returns:
        Tuple of (processed_audio_data, audio_metadata)
    """
    original_size = len(audio_data)

    # Get audio duration (non-blocking, failure is OK)
    try:
        duration = await get_audio_duration(audio_data)
        if duration is None:
            duration = -1
    except Exception:
        duration = -1

    # Convert to standard format
    processed_audio = await convert_audio_to_wav(
        audio_data,
        sample_rate=16000,
        channels=1,
        bit_depth=16
    )

    meta = {
        "duration": duration,
        "original_sample_rate": 0,  # Not easily available without ffprobe details
        "target_sample_rate": 16000,
        "sample_width": 2,
        "converted": True,
        "original_size": original_size,
        "processed_size": len(processed_audio)
    }

    return processed_audio, meta


def sync_evaluate(
    audio_data: bytes,
    ref_text: str,
    engine_model_type: str,
    text_mode: int,
    eval_mode: int,
    score_coeff: float,
    keyword: str,
    sentence_info_enabled: int
) -> dict:
    """
    Synchronous speech evaluation using SOE service.
    This wraps the existing SOE service for use with the thread pool.
    """
    import threading
    import time
    import sys
    import os

    # Add SDK path
    SDK_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "core", "util", "tencentcloud-speech-sdk-python"
    )
    if SDK_PATH not in sys.path:
        sys.path.insert(0, SDK_PATH)

    from common.credential import Credential
    from soe.speaking_assessment import SpeakingAssessment, SpeakingAssessmentListener

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
    credential = Credential(settings.tencent_secret_id, settings.tencent_secret_key)

    recognizer = SpeakingAssessment(
        settings.tencent_appid, credential, engine_model_type, listener
    )

    # Configure evaluation parameters
    recognizer.set_text_mode(text_mode)
    recognizer.set_ref_text(ref_text)
    recognizer.set_eval_mode(eval_mode)
    recognizer.set_keyword(keyword)
    recognizer.set_sentence_info_enabled(sentence_info_enabled)
    recognizer.set_voice_format(1)  # WAV format
    recognizer.set_rec_mode(1)  # Recording mode (send all audio at once)
    recognizer.score_coeff = score_coeff

    try:
        recognizer.start()

        # Wait for connection
        wait_time = 0
        while recognizer.status == 1:  # STARTED
            time.sleep(0.1)
            wait_time += 0.1
            if wait_time > 10:
                try:
                    recognizer.ws.close()
                except:
                    pass
                return {"error": "Connection timeout"}

        if recognizer.status != 2:  # OPENED
            try:
                recognizer.ws.close()
            except:
                pass
            return {"error": f"Connection failed, status: {recognizer.status}"}

        # Send audio data
        recognizer.write(audio_data)
        recognizer.stop()

        # Wait for result
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


@router.post("/upload", response_model=SOEResponse)
async def upload_and_assess(
    file: UploadFile = File(..., description="Audio file"),
    ref_text: str = Form(default="", description="被评估语音对应的文本。句子模式≤30字/词，段落模式≤120字/词，自由说模式可不填"),
    engine_model_type: str = Form(default="16k_zh", description="语言引擎：16k_zh(中文)，16k_en(英文)"),
    text_mode: int = Form(default=0, description="文本模式：0=普通文本(默认)，1=音素结构文本"),
    eval_mode: int = Form(default=3, description="评测模式：0=单词/单字，1=句子，2=段落，3=自由说，4=单词音素纠错，5=情景评测，6=句子多分支，7=单词实时，8=拼音评测"),
    score_coeff: float = Form(default=2.0, ge=1.0, le=4.0, description="评价苛刻指数：1.0=儿童(宽松)，2.0=标准，4.0=成人(严格)"),
    keyword: str = Form(default="", description="主题词和关键词"),
    sentence_info_enabled: int = Form(default=0, description="输出断句中间结果：0=不输出(默认)，1=输出"),
    message_id: Optional[str] = Form(default=None, description="消息ID，不传则自动生成UUID"),
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> SOEResponse:
    """
    上传音频文件进行语音评测。

    **认证方式**: X-Signature Header 携带 AES 加密签名

    **支持格式**: ffmpeg 支持的所有格式 (wav, mp3, m4a, ogg, flac 等)

    **音频处理**: 自动转换为 16kHz, 16bit, 单声道 WAV

    ---

    ## 参数说明

    **text_mode 文本模式**:
    | 值 | 说明 |
    |---|---|
    | 0 | 普通文本（默认） |
    | 1 | 音素结构文本（用于自定义注音） |

    **eval_mode 评测模式**:
    | 值 | 说明 |
    |---|---|
    | 0 | 单词/单字模式（中文为单字模式） |
    | 1 | 句子模式 |
    | 2 | 段落模式 |
    | 3 | 自由说模式（默认） |
    | 4 | 单词音素纠错模式 |
    | 5 | 情景评测模式 |
    | 6 | 句子多分支评测模式 |
    | 7 | 单词实时评测模式 |
    | 8 | 拼音评测模式 |

    **score_coeff 评价苛刻指数**:
    | 值 | 说明 |
    |---|---|
    | 1.0 | 儿童应用场景（宽松） |
    | 2.0 | 标准（默认） |
    | 4.0 | 成人严格打分场景 |

    **ref_text 参考文本**:
    - 句子模式：不超过 30 个单词或中文文字
    - 段落模式：不超过 120 个单词或中文文字
    - 自由说模式：可不填
    - 中文使用 UTF-8 编码

    **sentence_info_enabled 断句中间结果**:
    | 值 | 说明 |
    |---|---|
    | 0 | 不输出（默认） |
    | 1 | 输出，用于客户端 UI 实时更新 |
    """
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = message_id or str(uuid.uuid4())

    # Read audio data
    audio_data = await file.read()
    if len(audio_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty"
        )

    # Process audio
    try:
        processed_audio, audio_meta = await process_audio(audio_data, file.filename or "audio.wav")
        print(f"[SOE UPLOAD] Audio processed: duration={audio_meta['duration']:.1f}s, "
              f"converted={audio_meta['converted']}, "
              f"original_size={audio_meta['original_size']}bytes, "
              f"processed_size={audio_meta['processed_size']}bytes")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing failed: {str(e)}"
        )

    # Save processed audio
    audio_filename = f"{uuid.uuid4()}.wav"
    audio_path = UPLOAD_DIR / audio_filename
    with open(audio_path, 'wb') as f:
        f.write(processed_audio)

    # Run evaluation in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: sync_evaluate(
            audio_data=processed_audio,
            ref_text=ref_text,
            engine_model_type=engine_model_type,
            text_mode=text_mode,
            eval_mode=eval_mode,
            score_coeff=score_coeff,
            keyword=keyword,
            sentence_info_enabled=sentence_info_enabled
        )
    )

    # Check for errors
    if "error" in result:
        return SOEResponse(
            message_id=msg_id,
            eval_mode=eval_mode,
            ref_text=ref_text,
            error=result["error"]
        )

    # Extract score
    score = None
    if result and 'result' in result and result['result']:
        score = result['result'].get('SuggestedScore', None)

    return SOEResponse(
        message_id=msg_id,
        voice_id=result.get('voice_id', ''),
        ref_text=ref_text,
        eval_mode=eval_mode,
        score=score,
        result=result,
        audio_filename=audio_filename,
        audio_meta=AudioMeta(
            duration=audio_meta['duration'],
            original_sample_rate=audio_meta['original_sample_rate'],
            target_sample_rate=audio_meta['target_sample_rate'],
            sample_width=audio_meta['sample_width'],
            converted=audio_meta['converted'],
            original_size=audio_meta['original_size'],
            processed_size=audio_meta['processed_size']
        )
    )


@router.post("/url", response_model=SOEResponse)
async def assess_from_url(
    request: SOEUrlRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
) -> SOEResponse:
    """
    通过音频 URL 进行语音评测。

    **认证方式**: X-Signature Header 携带 AES 加密签名

    **支持格式**: ffmpeg 支持的所有格式 (wav, mp3, m4a, ogg, flac 等)

    **限制**:
    - 文件大小：最大 50MB
    - 音频时长：最大 300 秒
    - 自动转换为 16kHz, 16bit, 单声道 WAV

    ---

    ## 参数说明

    **text_mode 文本模式**:
    | 值 | 说明 |
    |---|---|
    | 0 | 普通文本（默认） |
    | 1 | 音素结构文本（用于自定义注音） |

    **eval_mode 评测模式**:
    | 值 | 说明 |
    |---|---|
    | 0 | 单词/单字模式（中文为单字模式） |
    | 1 | 句子模式 |
    | 2 | 段落模式 |
    | 3 | 自由说模式（默认） |
    | 4 | 单词音素纠错模式 |
    | 5 | 情景评测模式 |
    | 6 | 句子多分支评测模式 |
    | 7 | 单词实时评测模式 |
    | 8 | 拼音评测模式 |

    **score_coeff 评价苛刻指数**:
    | 值 | 说明 |
    |---|---|
    | 1.0 | 儿童应用场景（宽松） |
    | 2.0 | 标准（默认） |
    | 4.0 | 成人严格打分场景 |

    **ref_text 参考文本**:
    - 句子模式：不超过 30 个单词或中文文字
    - 段落模式：不超过 120 个单词或中文文字
    - 自由说模式：可不填
    - 中文使用 UTF-8 编码

    **sentence_info_enabled 断句中间结果**:
    | 值 | 说明 |
    |---|---|
    | 0 | 不输出（默认） |
    | 1 | 输出，用于客户端 UI 实时更新 |

    ---

    ## 请求示例

    ```json
    {
        "audio_url": "https://example.com/audio.mp3",
        "ref_text": "",
        "engine_model_type": "16k_zh",
        "text_mode": 0,
        "eval_mode": 3,
        "score_coeff": 2.0,
        "keyword": "",
        "sentence_info_enabled": 0,
        "message_id": "optional-custom-id"
    }
    ```
    """
    verify_signature(x_signature)

    # Generate message_id if not provided
    msg_id = request.message_id or str(uuid.uuid4())

    audio_url = str(request.audio_url)

    # Get filename from URL
    parsed_url = urlparse(audio_url)
    url_path = parsed_url.path
    filename = Path(url_path).name if url_path else "audio.wav"
    if not filename or '.' not in filename:
        filename = "audio.wav"

    # Download audio file
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(audio_url, follow_redirects=True)
            response.raise_for_status()

            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large, maximum {MAX_DOWNLOAD_SIZE // 1024 // 1024}MB"
                )

            audio_data = response.content

            if len(audio_data) > MAX_DOWNLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large, maximum {MAX_DOWNLOAD_SIZE // 1024 // 1024}MB"
                )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download audio: HTTP {e.response.status_code}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download audio: {str(e)}"
        )

    if len(audio_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Downloaded audio file is empty"
        )

    # Process audio
    try:
        processed_audio, audio_meta = await process_audio(audio_data, filename)
        print(f"[SOE URL] Audio processed: URL={audio_url[:50]}..., "
              f"duration={audio_meta['duration']:.1f}s, "
              f"converted={audio_meta['converted']}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing failed: {str(e)}"
        )

    # Save processed audio
    audio_filename = f"{uuid.uuid4()}.wav"
    audio_path = UPLOAD_DIR / audio_filename
    with open(audio_path, 'wb') as f:
        f.write(processed_audio)

    # Run evaluation in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: sync_evaluate(
            audio_data=processed_audio,
            ref_text=request.ref_text,
            engine_model_type=request.engine_model_type,
            text_mode=request.text_mode,
            eval_mode=request.eval_mode,
            score_coeff=request.score_coeff,
            keyword=request.keyword,
            sentence_info_enabled=request.sentence_info_enabled
        )
    )

    # Check for errors
    if "error" in result:
        return SOEResponse(
            message_id=msg_id,
            eval_mode=request.eval_mode,
            ref_text=request.ref_text,
            source_url=audio_url,
            error=result["error"]
        )

    # Extract score
    score = None
    if result and 'result' in result and result['result']:
        score = result['result'].get('SuggestedScore', None)

    return SOEResponse(
        message_id=msg_id,
        voice_id=result.get('voice_id', ''),
        ref_text=request.ref_text,
        eval_mode=request.eval_mode,
        score=score,
        result=result,
        audio_filename=audio_filename,
        source_url=audio_url,
        audio_meta=AudioMeta(
            duration=audio_meta['duration'],
            original_sample_rate=audio_meta['original_sample_rate'],
            target_sample_rate=audio_meta['target_sample_rate'],
            sample_width=audio_meta['sample_width'],
            converted=audio_meta['converted'],
            original_size=audio_meta['original_size'],
            processed_size=audio_meta['processed_size']
        )
    )


@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Get uploaded audio file (no authentication required).

    Can be used to play or download processed audio files.
    """
    # Security check: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )

    audio_path = UPLOAD_DIR / filename

    if not audio_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=filename
    )
