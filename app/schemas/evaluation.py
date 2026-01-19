"""
Schemas for speech evaluation API.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class SpeechScores(BaseModel):
    """Speech evaluation scores."""

    pronunciation_accuracy: float = Field(
        0.0, description="Pronunciation accuracy score (0-100)"
    )
    pronunciation_fluency: float = Field(
        0.0, description="Pronunciation fluency score (0-100)"
    )
    pronunciation_completion: float = Field(
        0.0, description="Pronunciation completion score (0-100)"
    )
    suggested_score: float = Field(
        0.0, description="Suggested overall score (0-100)"
    )
    overall_score: float = Field(
        0.0, description="Calculated overall score (0-100)"
    )


class WordScore(BaseModel):
    """Individual word score."""

    word: str = Field(..., description="The word")
    accuracy: float = Field(..., description="Accuracy score")
    fluency: float = Field(..., description="Fluency score")


class EvaluationStatistics(BaseModel):
    """Evaluation statistics."""

    total_words: int = Field(0, description="Total number of words")
    average_accuracy: float = Field(0.0, description="Average accuracy score")
    low_score_count: int = Field(0, description="Number of low-scoring words")


class EvaluationRequest(BaseModel):
    """Request model for speech evaluation API."""

    # Audio source (one of the following)
    audio_url: Optional[str] = Field(
        None, description="URL of the audio file to evaluate"
    )
    audio_path: Optional[str] = Field(
        None, description="Local path of the audio file to evaluate"
    )

    # Evaluation options
    ref_text: Optional[str] = Field(
        "", description="Reference text for speech evaluation (optional)"
    )
    custom_prompt: Optional[str] = Field(
        None, description="Custom prompt for AI evaluation generation"
    )
    language: str = Field(
        "zh", description="Language: 'zh' for Chinese, 'en' for English"
    )


class EvaluationResponse(BaseModel):
    """Response model for speech evaluation API."""

    success: bool = Field(..., description="Whether the evaluation succeeded")
    message: str = Field(..., description="Status message")

    # Results
    speech_text: Optional[str] = Field(
        None, description="Transcribed speech text"
    )
    speech_scores: Optional[SpeechScores] = Field(
        None, description="Speech evaluation scores"
    )
    statistics: Optional[EvaluationStatistics] = Field(
        None, description="Evaluation statistics"
    )
    low_score_words: Optional[List[WordScore]] = Field(
        None, description="Words with low pronunciation scores"
    )
    evaluation_report: Optional[str] = Field(
        None, description="AI-generated evaluation report in Markdown format"
    )

    # Error info
    error: Optional[str] = Field(
        None, description="Error message if evaluation failed"
    )


class SignatureRequest(BaseModel):
    """Request model for signature generation."""

    aes_key: str = Field(..., description="AES key for signature generation")


class SignatureResponse(BaseModel):
    """Response model for signature generation."""

    success: bool = Field(..., description="Whether the generation succeeded")
    signature: Optional[str] = Field(None, description="Generated signature")
    timestamp: Optional[int] = Field(None, description="Timestamp used in signature")
    expires_in: Optional[int] = Field(None, description="Seconds until signature expires")
    error: Optional[str] = Field(None, description="Error message if failed")
