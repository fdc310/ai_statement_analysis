"""
Base schemas for API responses.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""

    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")
    data: Optional[Any] = Field(None, description="Response data")
