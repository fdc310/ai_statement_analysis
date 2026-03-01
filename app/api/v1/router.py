"""
API v1 router - aggregates all endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import evaluation, auth, soe, tts, admin

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
    admin.router,
    prefix="/admin",
    tags=["Admin & Configuration"]
)
