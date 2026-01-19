"""
API v1 router - aggregates all endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import evaluation, auth

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
