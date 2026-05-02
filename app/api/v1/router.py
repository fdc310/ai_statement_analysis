"""
API v1 router - aggregates all endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import evaluation, auth, soe, tts, tasks, monitoring, agents, ws_streaming, ws_chat

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    auth.router,
    prefix="/health",
    tags=["Health"]
)
api_router.include_router(
    evaluation.router,
    prefix="/evaluation",
    tags=["Evaluation"]
)
api_router.include_router(
    soe.router,
    prefix="/soe",
    tags=["SOE - Speech Evaluation"]
)
api_router.include_router(
    tts.router,
    prefix="/tts",
    tags=["TTS - Text to Speech"]
)
api_router.include_router(
    tasks.router,
    prefix="/tasks",
    tags=["Tasks - Async Task Management"]
)
api_router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["Monitoring - Usage Dashboard"]
)
api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["Agents - Standalone Agent Endpoints"]
)
api_router.include_router(
    ws_streaming.router,
    prefix="/streaming",
    tags=["Streaming - WebSocket Audio Stream"]
)
api_router.include_router(
    ws_chat.router,
    prefix="/streaming",
    tags=["Streaming - WebSocket Voice Chat"]
)
