"""
Schemas for speech evaluation API.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl


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

    # Async callback
    message_id: Optional[str] = Field(
        None, description="Message ID for tracking (auto-generated if not provided)"
    )
    callback_url: HttpUrl = Field(
        ..., description="Callback URL to receive evaluation results"
    )


class EvaluationAcceptedResponse(BaseModel):
    """Response model when evaluation task is accepted (includes SOE scores)."""

    success: bool = Field(True, description="Whether the task was accepted")
    message: str = Field("Task accepted", description="Status message")
    message_id: str = Field(..., description="Message ID for tracking")

    # SOE results (returned immediately)
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

    # Error info
    error: Optional[str] = Field(
        None, description="Error message if SOE evaluation failed"
    )


class EvaluationCallbackData(BaseModel):
    """Data sent to callback URL when AI report generation completes."""

    message_id: str = Field(..., description="Message ID for tracking")
    success: bool = Field(..., description="Whether the report generation succeeded")
    message: str = Field(..., description="Status message")

    # AI Report (sent via callback)
    evaluation_report: Optional[str] = Field(
        None, description="AI-generated evaluation report in Markdown format"
    )

    # Error info
    error: Optional[str] = Field(
        None, description="Error message if report generation failed"
    )


# Keep old response model for backward compatibility
class EvaluationResponse(BaseModel):
    """Response model for speech evaluation API (deprecated, use callback instead)."""

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


class ReportRequest(BaseModel):
    """Request model for AI report generation with pre-existing SOE scores."""

    audio_url: HttpUrl = Field(..., description="音频文件URL")
    speech_text: Optional[str] = Field(None, description="语音转写文本，不传则自动调用ASR识别")
    soe_result: dict = Field(..., description="SOE评测返回的result数据，包含SuggestedScore、PronAccuracy、Words等字段")
    audio_duration: Optional[float] = Field(None, description="音频时长（秒），用于计算语速")
    topic: Optional[str] = Field(None, description="演讲主题，用于分析内容贴题性。不传则为自由说模式")
    custom_prompt: Optional[str] = Field(None, description="自定义AI评测提示词")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")
    language: str = Field(default="zh", description="语言：'zh'中文，'en'英文，用于ASR识别")


class ReportResponse(BaseModel):
    """Response model for AI report generation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    audio_url: str = Field(..., description="音频URL")
    speech_text: Optional[str] = Field(None, description="语音转写文本（ASR识别结果）")
    speech_rate: Optional[float] = Field(None, description="语速（字/分钟或词/分钟）")
    evaluation_report: Optional[str] = Field(None, description="AI生成的Markdown格式评测报告")
    error: Optional[str] = Field(None, description="错误信息")


class TextAnalysisRequest(BaseModel):
    """Request model for text structure analysis."""

    text: str = Field(..., description="待分析的文本内容", min_length=10, max_length=50000)
    custom_prompt: Optional[str] = Field(None, description="自定义分析要求")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class TextAnalysisResponse(BaseModel):
    """Response model for text structure analysis."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    analysis_result: Optional[str] = Field(None, description="分析结果（JSON格式字符串）")
    error: Optional[str] = Field(None, description="错误信息")


class TongueTwisterRequest(BaseModel):
    """Request model for tongue twister pronunciation analysis."""

    text: str = Field(..., description="绕口令文本", min_length=2, max_length=5000)
    language: str = Field(default="zh", description="语言：'zh'中文，'en'英文")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class TongueTwisterResponse(BaseModel):
    """Response model for tongue twister pronunciation analysis."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    tongue_twister: str = Field(..., description="绕口令原文")
    analysis_result: Optional[str] = Field(None, description="发音分析结果（JSON格式字符串）")
    error: Optional[str] = Field(None, description="错误信息")
