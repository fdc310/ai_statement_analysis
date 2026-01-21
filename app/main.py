"""
AI Statement Analysis API

A FastAPI application that provides speech evaluation services:
- Audio transcription (ASR)
- Speech scoring (SOE)
- AI-powered evaluation report generation (Hunyuan)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import api_router

app = FastAPI(
    title=settings.app_name,
    description="""
## 语音演讲评测 API

本 API 提供语音评测服务，包括语音转文字、语音评分和 AI 评测报告生成。

---

## 认证方式

所有接口使用 AES 加密签名认证，在 Header 中携带 `X-Signature`。

**获取签名：**
```
POST /api/v1/evaluation/signature
Body: {"aes_key": "your_aes_key"}
```

---

## 接口分类

### 1. SOE 简单语音评测 (`/api/v1/soe`)

仅进行语音评分，**同步返回**结果，适合快速评测场景。

| 接口 | 方法 | 说明 |
|-----|------|-----|
| `/api/v1/soe/upload` | POST | 上传音频文件评测 |
| `/api/v1/soe/url` | POST | 通过URL评测音频 |
| `/api/v1/soe/audio/{filename}` | GET | 获取已上传的音频文件 |

**请求示例 (upload)：**
```
POST /api/v1/soe/upload
Content-Type: multipart/form-data
X-Signature: <签名>

file: 音频文件
message_id: 可选，不传自动生成
eval_mode: 3 (自由说模式)
score_coeff: 2.0 (标准评分)
```

**请求示例 (url)：**
```json
POST /api/v1/soe/url
{
    "audio_url": "https://example.com/audio.mp3",
    "message_id": "optional-id",
    "eval_mode": 3,
    "score_coeff": 2.0
}
```

---

### 2. 完整语音评测 (`/api/v1/evaluation`)

包含 ASR 转写 + SOE 评分 + AI 报告生成，**异步回调**返回结果。

| 接口 | 方法 | 说明 |
|-----|------|-----|
| `/api/v1/evaluation/signature` | POST | 生成认证签名 |
| `/api/v1/evaluation/analyze` | POST | 通过URL提交评测任务 |
| `/api/v1/evaluation/analyze/upload` | POST | 上传文件提交评测任务 |

**工作流程：**
1. 调用接口 → 立即返回 `message_id`
2. 后台异步处理（ASR + SOE + AI报告）
3. 处理完成后 POST 回调到 `callback_url`

**请求示例 (analyze)：**
```json
POST /api/v1/evaluation/analyze
{
    "audio_url": "https://example.com/audio.mp3",
    "language": "zh",
    "callback_url": "https://your-server.com/callback",
    "message_id": "optional-id"
}
```

**立即返回：**
```json
{
    "success": true,
    "message": "Task accepted",
    "message_id": "uuid"
}
```

**回调数据 (POST 到 callback_url)：**
```json
{
    "message_id": "uuid",
    "success": true,
    "speech_text": "转写文字",
    "speech_scores": {...},
    "evaluation_report": "AI生成的Markdown报告"
}
```

---

## 评测参数说明

**eval_mode 评测模式：**
| 值 | 说明 |
|---|---|
| 0 | 单词/单字模式 |
| 1 | 句子模式 |
| 2 | 段落模式 |
| 3 | 自由说模式（默认） |

**score_coeff 评分系数：**
| 值 | 说明 |
|---|---|
| 1.0 | 儿童模式（宽松） |
| 2.0 | 标准模式（默认） |
| 4.0 | 成人严格模式 |

**engine_model_type 语言：**
| 值 | 说明 |
|---|---|
| 16k_zh | 中文 |
| 16k_en | 英文 |

---

## 音频要求

- **支持格式：** WAV, MP3, M4A, OGG, FLAC 等（ffmpeg 支持的格式）
- **自动转换：** 音频会自动转换为 16kHz, 16bit, 单声道
- **文件大小：** URL 模式最大 50MB
- **时长限制：** 最大 300 秒
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/auth/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
