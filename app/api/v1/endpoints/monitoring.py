"""
Monitoring dashboard endpoints for LLM usage tracking.
All endpoints require the same AES signature authentication.
"""
from fastapi import APIRouter, Query

from app.schemas.monitoring import (
    UsageSummary, DailyUsage, DailyUsageResponse,
    ProviderUsage, ProviderUsageResponse,
    EndpointUsage, EndpointUsageResponse,
    AgentUsage, AgentUsageResponse,
    CostEstimateRequest, CostEstimateResponse,
)
from app.services.monitoring.token_tracker import token_tracker
from app.services.monitoring.cost_calculator import cost_calculator

router = APIRouter()


@router.get("/usage", response_model=UsageSummary)
async def get_usage_summary(days: int = Query(7, ge=1, le=365)):
    """Get overall usage summary for the given period."""
    summary = await token_tracker.get_summary(days=days)
    return UsageSummary(**summary)


@router.get("/usage/daily", response_model=DailyUsageResponse)
async def get_daily_usage(days: int = Query(7, ge=1, le=365)):
    """Get daily usage breakdown."""
    daily_stats = await token_tracker.get_daily_stats(days=days)
    daily_usage = [
        DailyUsage(date=date, **stats)
        for date, stats in sorted(daily_stats.items())
    ]
    return DailyUsageResponse(days=days, daily_usage=daily_usage)


@router.get("/usage/provider", response_model=ProviderUsageResponse)
async def get_provider_usage():
    """Get usage breakdown by LLM provider."""
    provider_stats = await token_tracker.get_provider_stats()
    providers = [
        ProviderUsage(provider=provider, **stats)
        for provider, stats in provider_stats.items()
    ]
    return ProviderUsageResponse(providers=providers)


@router.get("/usage/endpoint", response_model=EndpointUsageResponse)
async def get_endpoint_usage():
    """Get usage breakdown by API endpoint."""
    endpoint_stats = await token_tracker.get_usage_by_endpoint()
    endpoints = [
        EndpointUsage(endpoint=endpoint, **stats)
        for endpoint, stats in endpoint_stats.items()
    ]
    return EndpointUsageResponse(endpoints=endpoints)


@router.get("/usage/agent", response_model=AgentUsageResponse)
async def get_agent_usage():
    """Get usage breakdown by agent name."""
    agent_stats = await token_tracker.get_usage_by_agent()
    agents = [
        AgentUsage(agent_name=agent, **stats)
        for agent, stats in agent_stats.items()
    ]
    return AgentUsageResponse(agents=agents)


@router.get("/cost", response_model=UsageSummary)
async def get_cost_summary(days: int = Query(7, ge=1, le=365)):
    """Get cost summary for the given period."""
    summary = await token_tracker.get_summary(days=days)
    return UsageSummary(**summary)


@router.post("/cost/estimate", response_model=CostEstimateResponse)
async def estimate_cost(request: CostEstimateRequest):
    """Estimate cost for a hypothetical API call."""
    cost = await cost_calculator.estimate_cost(
        provider=request.provider,
        model=request.model,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
    )
    return CostEstimateResponse(
        provider=request.provider,
        model=request.model,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        estimated_cost_usd=cost,
    )
