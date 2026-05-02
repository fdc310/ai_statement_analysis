"""
WebSocket streaming chat schemas.
"""
from typing import Optional
from pydantic import BaseModel, Field


class StreamChatConfig(BaseModel):
    """Streaming chat session configuration."""
    language: str = "zh"
    scene: str = ""  # interview / daily / customer_service
    system_prompt: str = ""
    voice_type: int = 101001
    ref_text: str = ""
    eval_mode: int = 3
    score_coeff: float = 1.0
    server_type: int = 0
    word_info: int = 1
    enable_asr: bool = True
    enable_soe: bool = False
    enable_timestamps: bool = False
    eval_type: str = ""  # optional: impromptu_reaction etc.
    scenario: str = ""  # for eval_type
