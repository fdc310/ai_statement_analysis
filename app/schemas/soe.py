# -*- coding: utf-8 -*-
"""
Schemas for simple SOE (Speech Oral Evaluation) API.
These endpoints only perform speech evaluation without ASR and AI report generation.
"""
from typing import Optional, List, Any
from pydantic import BaseModel, Field, HttpUrl


class SOEUploadRequest(BaseModel):
    """Form parameters for upload endpoint."""
    ref_text: str = Field(default="", description="Reference text (optional for free speech mode)")
    engine_model_type: str = Field(default="16k_zh", description="Engine model: 16k_en (English), 16k_zh (Chinese)")
    text_mode: int = Field(default=0, description="Text mode")
    eval_mode: int = Field(default=3, description="Evaluation mode: 0=word, 1=sentence, 2=paragraph, 3=free speech")
    score_coeff: float = Field(default=2.0, ge=1.0, le=4.0, description="Score coefficient: 1.0=children, 2.0=standard, 4.0=strict")
    keyword: str = Field(default="", description="Keywords")
    sentence_info_enabled: int = Field(default=0, description="Sentence info: 0=off, 1=on")


class SOEUrlRequest(BaseModel):
    """Request model for URL-based evaluation."""
    audio_url: HttpUrl = Field(..., description="URL of the audio file")
    ref_text: str = Field(default="", description="Reference text (optional for free speech mode)")
    engine_model_type: str = Field(default="16k_zh", description="Engine model: 16k_en (English), 16k_zh (Chinese)")
    text_mode: int = Field(default=0, description="Text mode")
    eval_mode: int = Field(default=3, description="Evaluation mode: 0=word, 1=sentence, 2=paragraph, 3=free speech")
    score_coeff: float = Field(default=2.0, ge=1.0, le=4.0, description="Score coefficient: 1.0=children, 2.0=standard, 4.0=strict")
    keyword: str = Field(default="", description="Keywords")
    sentence_info_enabled: int = Field(default=0, description="Sentence info: 0=off, 1=on")
    message_id: Optional[str] = Field(default=None, description="Message ID for tracking (auto-generated if not provided)")


class AudioMeta(BaseModel):
    """Audio file metadata."""
    duration: float = Field(..., description="Audio duration in seconds")
    original_sample_rate: int = Field(..., description="Original sample rate in Hz")
    target_sample_rate: int = Field(default=16000, description="Target sample rate in Hz")
    sample_width: int = Field(default=2, description="Sample width in bytes (2=16bit)")
    converted: bool = Field(..., description="Whether the audio was converted")
    original_size: int = Field(..., description="Original file size in bytes")
    processed_size: int = Field(..., description="Processed file size in bytes")


class SOEResponse(BaseModel):
    """Response model for SOE evaluation."""
    message_id: str = Field(..., description="Message ID for tracking")
    id: Optional[int] = Field(None, description="Assessment record ID")
    voice_id: str = Field(default="", description="Voice session ID")
    ref_text: str = Field(default="", description="Reference text used")
    eval_mode: int = Field(..., description="Evaluation mode used")
    score: Optional[float] = Field(None, description="Suggested score (0-100)")
    result: Optional[dict] = Field(None, description="Full evaluation result")
    audio_filename: Optional[str] = Field(None, description="Saved audio filename")
    source_url: Optional[str] = Field(None, description="Source URL (for URL endpoint)")
    audio_meta: Optional[AudioMeta] = Field(None, description="Audio metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
