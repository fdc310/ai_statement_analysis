"""
AI Statement Analysis API

A FastAPI application that provides speech evaluation services:
- Audio transcription (ASR)
- Speech scoring (SOE)
- AI-powered evaluation report generation (Hunyuan)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

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

## 接口分组

### Evaluation - 综合评测
| 接口 | 说明 |
|------|------|
| `POST /evaluation/signature` | 生成 AES 签名 |
| `POST /evaluation/analyze` | 语音评测（URL，异步回调） |
| `POST /evaluation/analyze/upload` | 语音评测（上传文件，异步回调） |
| `POST /evaluation/report` | AI 评测报告（URL，同步） |
| `POST /evaluation/report/upload` | AI 评测报告（上传文件，同步） |
| `POST /evaluation/text-analysis` | 文本结构分析 |
| `POST /evaluation/tongue-twister` | 绕口令发音分析（纯文本） |
| `POST /evaluation/sentence-interpretation` | 句子解读分析 |
| `POST /evaluation/story-reading` | 故事阅读评测（ASR + AI 分析） |
| `POST /evaluation/tongue-twister-reading` | 绕口令/文章朗读评测（ASR + SOE + AI 分析） |
| `POST /evaluation/voice-chat` | 语音对话（ASR + AI 对话 + TTS） |
| `POST /evaluation/opinion-statement` | 一分钟观点陈述评测（SOE + AI 分析） |
| `POST /evaluation/impromptu-reaction` | 即兴反应评测（ASR + SOE + AI 分析） |

### SOE - 语音评分
| 接口 | 说明 |
|------|------|
| `POST /soe/upload` | 语音评分（上传文件） |
| `POST /soe/url` | 语音评分（URL） |

### TTS - 语音合成
| 接口 | 说明 |
|------|------|
| `POST /tts/synthesize` | 文字转语音 |
| `GET /tts/synthesize` | 文字转语音（GET） |
| `GET /tts/voices` | 获取可用音色列表 |

---

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

# Mount admin frontend
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "admin-frontend")
if os.path.exists(frontend_dir):
    app.mount("/admin-ui", StaticFiles(directory=frontend_dir, html=True), name="admin-ui")



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
