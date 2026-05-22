"""
Token usage tracking for LLM API calls.
Stores usage data in memory with configurable retention.
"""
import uuid
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Single LLM API call usage record."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: str  # "hunyuan" or "openai"
    model: str  # "hunyuan-turbo", "gpt-4o", etc.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float = 0
    cost_usd: float = 0
    endpoint: str = ""  # Which evaluation endpoint triggered this
    agent_name: str = ""  # Which agent used the LLM


class TokenTracker:
    """In-memory token usage tracker with aggregation capabilities."""

    # Pricing per 1M tokens (USD)
    DEFAULT_PRICING = {
        "hunyuan-turbo": {"input": 0.5, "output": 1.0},
        "hunyuan": {"input": 0.5, "output": 1.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    }

    def __init__(self, retention_days: int = 30):
        self._records: list[TokenUsage] = []
        self._retention_days = retention_days

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost in USD based on token counts."""
        from app.core.config import settings

        # Check env-based pricing first
        if provider in ("hunyuan", "tencent"):
            input_price = getattr(settings, 'tencent_input_price', 0.5)
            output_price = getattr(settings, 'tencent_output_price', 1.0)
        elif provider == "openai":
            input_price = getattr(settings, 'openai_input_price', 2.5)
            output_price = getattr(settings, 'openai_output_price', 10.0)
        else:
            pricing = self.DEFAULT_PRICING.get(model, {"input": 1.0, "output": 1.0})
            input_price = pricing["input"]
            output_price = pricing["output"]

        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        return round(cost, 6)

    async def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float = 0,
        endpoint: str = "",
        agent_name: str = "",
        request_id: Optional[str] = None,
    ) -> TokenUsage:
        """Record a single LLM API call usage."""
        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(provider, model, prompt_tokens, completion_tokens)

        record = TokenUsage(
            request_id=request_id or str(uuid.uuid4()),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            cost_usd=cost,
            endpoint=endpoint,
            agent_name=agent_name,
        )

        self._records.append(record)
        logger.debug(
            f"Token usage recorded: {provider}/{model} "
            f"prompt={prompt_tokens} completion={completion_tokens} "
            f"cost=${cost:.4f}"
        )

        # Cleanup old records periodically
        if len(self._records) % 100 == 0:
            await self._cleanup()

        return record

    async def get_recent_usage(self, limit: int = 100) -> list[TokenUsage]:
        """Get recent usage records."""
        return sorted(self._records, key=lambda r: r.timestamp, reverse=True)[:limit]

    async def get_daily_stats(self, days: int = 7) -> dict:
        """Get daily usage aggregation."""
        cutoff = datetime.now() - timedelta(days=days)
        daily = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "request_count": 0,
        })

        for record in self._records:
            if record.timestamp >= cutoff:
                day_key = record.timestamp.strftime("%Y-%m-%d")
                daily[day_key]["prompt_tokens"] += record.prompt_tokens
                daily[day_key]["completion_tokens"] += record.completion_tokens
                daily[day_key]["total_tokens"] += record.total_tokens
                daily[day_key]["cost_usd"] += record.cost_usd
                daily[day_key]["request_count"] += 1

        return dict(daily)

    async def get_provider_stats(self) -> dict:
        """Get usage aggregation by provider."""
        stats = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "request_count": 0,
            "models": defaultdict(lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0,
                "request_count": 0,
            })
        })

        for record in self._records:
            provider_stats = stats[record.provider]
            provider_stats["prompt_tokens"] += record.prompt_tokens
            provider_stats["completion_tokens"] += record.completion_tokens
            provider_stats["total_tokens"] += record.total_tokens
            provider_stats["cost_usd"] += record.cost_usd
            provider_stats["request_count"] += 1

            model_stats = provider_stats["models"][record.model]
            model_stats["prompt_tokens"] += record.prompt_tokens
            model_stats["completion_tokens"] += record.completion_tokens
            model_stats["total_tokens"] += record.total_tokens
            model_stats["cost_usd"] += record.cost_usd
            model_stats["request_count"] += 1

        # Convert nested defaultdicts to dicts
        result = {}
        for provider, data in stats.items():
            result[provider] = {
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "total_tokens": data["total_tokens"],
                "cost_usd": round(data["cost_usd"], 4),
                "request_count": data["request_count"],
                "models": dict(data["models"]),
            }
        return result

    async def get_usage_by_endpoint(self) -> dict:
        """Get usage aggregation by API endpoint."""
        stats = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "request_count": 0,
        })

        for record in self._records:
            endpoint = record.endpoint or "unknown"
            stats[endpoint]["prompt_tokens"] += record.prompt_tokens
            stats[endpoint]["completion_tokens"] += record.completion_tokens
            stats[endpoint]["total_tokens"] += record.total_tokens
            stats[endpoint]["cost_usd"] += record.cost_usd
            stats[endpoint]["request_count"] += 1

        return {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in stats.items()}

    async def get_usage_by_agent(self) -> dict:
        """Get usage aggregation by agent name."""
        stats = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "request_count": 0,
        })

        for record in self._records:
            agent = record.agent_name or "unknown"
            stats[agent]["prompt_tokens"] += record.prompt_tokens
            stats[agent]["completion_tokens"] += record.completion_tokens
            stats[agent]["total_tokens"] += record.total_tokens
            stats[agent]["cost_usd"] += record.cost_usd
            stats[agent]["request_count"] += 1

        return {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in stats.items()}

    async def get_summary(self, days: int = 7) -> dict:
        """Get overall usage summary."""
        cutoff = datetime.now() - timedelta(days=days)
        total_prompt = 0
        total_completion = 0
        total_cost = 0
        count = 0

        for record in self._records:
            if record.timestamp >= cutoff:
                total_prompt += record.prompt_tokens
                total_completion += record.completion_tokens
                total_cost += record.cost_usd
                count += 1

        return {
            "period_days": days,
            "total_requests": count,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": round(total_cost, 4),
            "avg_tokens_per_request": (total_prompt + total_completion) // count if count > 0 else 0,
            "avg_cost_per_request": round(total_cost / count, 6) if count > 0 else 0,
        }

    async def _cleanup(self):
        """Remove records older than retention period."""
        cutoff = datetime.now() - timedelta(days=self._retention_days)
        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        removed = before - len(self._records)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old token usage records")


# Singleton
token_tracker = TokenTracker()
