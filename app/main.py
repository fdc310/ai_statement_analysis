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
## 语音演讲评测API

本API提供以下功能：
1. **语音转文字** - 使用腾讯云ASR将音频转换为文字
2. **语音评分** - 使用腾讯云SOE对语音进行评分（发音准确度、流利度等）
3. **AI评测报告** - 使用腾讯云混元大模型生成详细的Markdown格式评测报告

### 认证方式
使用AES加密的token进行接口验证。调用 `/api/v1/auth/token` 接口生成token。

### API版本
- v1: 当前版本

### 评测报告格式
AI生成的评测报告包含以下部分：
- 逻辑完整性评分
- 结构可视化（论点提取）
- 结论
- 优点
- 改进意见
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
