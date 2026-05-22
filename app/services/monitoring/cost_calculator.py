"""
Cost calculation utilities for LLM API usage.
"""
from typing import Optional

from app.services.monitoring.token_tracker import token_tracker


class CostCalculator:
    """Calculate costs from token usage data."""

    async def get_total_cost(self, days: int = 7) -> float:
        """Get total cost for the given period."""
        summary = await token_tracker.get_summary(days=days)
        return summary["total_cost_usd"]

    async def estimate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost for a hypothetical API call."""
        return token_tracker._calculate_cost(provider, model, prompt_tokens, completion_tokens)

    async def get_cost_breakdown(self, days: int = 7) -> dict:
        """Get detailed cost breakdown by provider and model."""
        provider_stats = await token_tracker.get_provider_stats()
        daily_stats = await token_tracker.get_daily_stats(days=days)

        return {
            "by_provider": provider_stats,
            "by_day": daily_stats,
        }


# Singleton
cost_calculator = CostCalculator()
