# Schemas module
from app.schemas.base import BaseResponse
from app.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    SpeechScores,
    EvaluationStatistics,
    WordScore,
    SignatureRequest,
    SignatureResponse
)

__all__ = [
    "BaseResponse",
    "EvaluationRequest",
    "EvaluationResponse",
    "SpeechScores",
    "EvaluationStatistics",
    "WordScore",
    "SignatureRequest",
    "SignatureResponse"
]
