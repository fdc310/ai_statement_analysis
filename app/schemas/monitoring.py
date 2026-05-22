"""
Monitoring and usage schemas for API responses.
"""
from typing import Optional
from pydantic import BaseModel, Field


class UsageSummary(BaseModel):
    """Overall usage summary."""
    success: bool = True
    period_days: int = 7
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_tokens_per_request: int = 0
    avg_cost_per_request: float = 0.0


class DailyUsage(BaseModel):
    """Daily usage entry."""
    date: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0


class DailyUsageResponse(BaseModel):
    """Response for daily usage query."""
    success: bool = True
    days: int = 7
    daily_usage: list[DailyUsage] = Field(default_factory=list)


class ProviderUsage(BaseModel):
    """Provider usage entry."""
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0
    models: dict = Field(default_factory=dict)


class ProviderUsageResponse(BaseModel):
    """Response for provider usage query."""
    success: bool = True
    providers: list[ProviderUsage] = Field(default_factory=list)


class EndpointUsage(BaseModel):
    """Endpoint usage entry."""
    endpoint: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0


class EndpointUsageResponse(BaseModel):
    """Response for endpoint usage query."""
    success: bool = True
    endpoints: list[EndpointUsage] = Field(default_factory=list)


class AgentUsage(BaseModel):
    """Agent usage entry."""
    agent_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0


class AgentUsageResponse(BaseModel):
    """Response for agent usage query."""
    success: bool = True
    agents: list[AgentUsage] = Field(default_factory=list)


class CostEstimateRequest(BaseModel):
    """Request to estimate cost for a hypothetical API call."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class CostEstimateResponse(BaseModel):
    """Response with cost estimate."""
    success: bool = True
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
