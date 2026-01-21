# -*- coding: utf-8 -*-
"""
Schemas for simple SOE (Speech Oral Evaluation) API.
These endpoints only perform speech evaluation without ASR and AI report generation.
"""
from typing import Optional, List, Any
from pydantic import BaseModel, Field, HttpUrl


class SOEUploadRequest(BaseModel):
    """Form parameters for upload endpoint."""
    ref_text: str = Field(default="", description="被评估语音对应的文本。句子模式≤30字/词，段落模式≤120字/词，自由说模式可不填")
    engine_model_type: str = Field(default="16k_zh", description="语言引擎：16k_zh(中文)，16k_en(英文)")
    text_mode: int = Field(default=0, description="文本模式：0=普通文本(默认)，1=音素结构文本")
    eval_mode: int = Field(default=3, description="评测模式：0=单词/单字，1=句子，2=段落，3=自由说，4=单词音素纠错，5=情景评测，6=句子多分支，7=单词实时，8=拼音评测")
    score_coeff: float = Field(default=2.0, ge=1.0, le=4.0, description="评价苛刻指数：1.0=儿童(宽松)，2.0=标准，4.0=成人(严格)")
    keyword: str = Field(default="", description="主题词和关键词")
    sentence_info_enabled: int = Field(default=0, description="输出断句中间结果：0=不输出(默认)，1=输出")


class SOEUrlRequest(BaseModel):
    """Request model for URL-based evaluation."""
    audio_url: HttpUrl = Field(..., description="音频文件URL")
    ref_text: str = Field(default="", description="被评估语音对应的文本。句子模式≤30字/词，段落模式≤120字/词，自由说模式可不填")
    engine_model_type: str = Field(default="16k_zh", description="语言引擎：16k_zh(中文)，16k_en(英文)")
    text_mode: int = Field(default=0, description="文本模式：0=普通文本(默认)，1=音素结构文本")
    eval_mode: int = Field(default=3, description="评测模式：0=单词/单字，1=句子，2=段落，3=自由说，4=单词音素纠错，5=情景评测，6=句子多分支，7=单词实时，8=拼音评测")
    score_coeff: float = Field(default=2.0, ge=1.0, le=4.0, description="评价苛刻指数：1.0=儿童(宽松)，2.0=标准，4.0=成人(严格)")
    keyword: str = Field(default="", description="主题词和关键词")
    sentence_info_enabled: int = Field(default=0, description="输出断句中间结果：0=不输出(默认)，1=输出")
    message_id: Optional[str] = Field(default=None, description="消息ID，不传则自动生成UUID")


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
