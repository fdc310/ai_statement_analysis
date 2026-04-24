"""
WebSocket streaming schemas.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


class StreamConfigMessage(BaseModel):
    """Configuration message sent by client."""
    type: str = "config"
    data: dict = Field(default_factory=dict)


class StreamAudioMessage(BaseModel):
    """Audio data message (binary frame, not JSON)."""
    pass


class StreamEndMessage(BaseModel):
    """End signal message."""
    type: str = "end"


class StreamResultMessage(BaseModel):
    """Result message sent by server."""
    type: str  # "asr_partial", "soe_intermediate", "complete", "error"
    data: Optional[dict] = None
    message: Optional[str] = None


class StreamConfig(BaseModel):
    """Streaming session configuration."""
    language: str = "zh"
    ref_text: str = ""
    eval_mode: int = 3
    score_coeff: float = 1.0
    server_type: int = 0
    word_info: int = 1
    enable_asr: bool = True
    enable_soe: bool = True
