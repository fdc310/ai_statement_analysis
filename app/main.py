"""
AI Statement Analysis API

A FastAPI application that provides speech evaluation services:
- Audio transcription (ASR)
- Speech scoring (SOE)
- AI-powered evaluation report generation (Hunyuan)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    logger.info("Application starting up...")
    yield
    # Shutdown
    logger.info("Application shutting down...")
    # Clean up singletons
    try:
        from app.core.thread_pool import ThreadPool
        ThreadPool.shutdown()
    except Exception as e:
        logger.warning(f"ThreadPool shutdown error: {e}")
    try:
        from app.services.tasks.callback import callback_dispatcher
        await callback_dispatcher.close()
    except Exception as e:
        logger.warning(f"CallbackDispatcher close error: {e}")
    try:
        from app.services.chat.session_manager import chat_session_manager
        await chat_session_manager.cleanup_expired()
    except Exception as e:
        logger.warning(f"ChatSessionManager cleanup error: {e}")

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
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
| `POST /evaluation/voice-chat` | 语音对话（传统/多模态 + 服务端会话管理） |
| `POST /evaluation/voice-chat/scene` | 会话级场景切换 |
| `POST /evaluation/opinion-statement` | 一分钟观点陈述评测（ASR + SOE + AI 分析），评测观点明确性、结构完整度、逻辑清晰度、时间节奏、表达精炼度 |
| `POST /evaluation/impromptu-reaction` | 即兴反应评测（ASR + SOE + AI 分析），评测反应速度、结构形成、内容切题、逻辑连贯、表达冗余 |

### Agents - 独立评测代理
| 接口 | 说明 |
|------|------|
| `POST /agents/asr` | 独立 ASR 语音转文字 |
| `POST /agents/soe` | 独立 SOE 语音评分 |
| `POST /agents/content` | 独立内容分析 |
| `POST /agents/fluency` | 独立流畅度分析 |
| `POST /agents/report` | 独立报告生成 |

### Tasks - 异步任务管理
| 接口 | 说明 |
|------|------|
| `GET /tasks/` | 查询任务列表 |
| `GET /tasks/{task_id}` | 查询任务状态 |
| `GET /tasks/stats` | 任务统计 |

### Monitoring - 使用监控
| 接口 | 说明 |
|------|------|
| `GET /monitoring/usage` | 使用量汇总 |
| `GET /monitoring/usage/daily` | 每日使用量 |
| `GET /monitoring/usage/provider` | 按供应商统计 |
| `GET /monitoring/usage/endpoint` | 按接口统计 |
| `GET /monitoring/usage/agent` | 按代理统计 |
| `GET /monitoring/cost` | 费用汇总 |
| `POST /monitoring/cost/estimate` | 费用估算 |

### Streaming - 实时流式评测
| 接口 | 说明 |
|------|------|
| `WS /streaming/ws/stream` | WebSocket 实时音频流评测 |

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
# 兼容小程序端多了一层 /api 前缀的情况
app.include_router(api_router, prefix="/api/api/v1")


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
