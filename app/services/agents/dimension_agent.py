"""
Generic dimension agent for parallel evaluation.
Each instance evaluates a single dimension of speech quality.
"""
import logging
from typing import Callable

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services import get_llm_service
from app.services.agents.prompts.common import extract_json
from app.services.monitoring.token_tracker import token_tracker

logger = logging.getLogger(__name__)


class DimensionAgent(BaseAgent):
    """
    A configurable agent that evaluates one dimension of speech.

    Each dimension has:
    - name: unique identifier (e.g. "speech_rate", "content")
    - system_prompt_fn: function(context) -> str
    - user_prompt_fn: function(context) -> str
    """

    def __init__(
        self,
        dim_name: str,
        system_prompt_fn: Callable[[EvaluationContext], str],
        user_prompt_fn: Callable[[EvaluationContext], str],
    ):
        self._dim_name = dim_name
        self._system_prompt_fn = system_prompt_fn
        self._user_prompt_fn = user_prompt_fn

    @property
    def name(self) -> str:
        return self._dim_name

    async def execute(self, context: EvaluationContext) -> AgentResult:
        speech_text = context.speech_text or ""
        if not speech_text.strip():
            return AgentResult(
                agent_name=self._dim_name,
                success=True,
                data={},
            )

        llm = get_llm_service()

        system_prompt = self._system_prompt_fn(context)
        user_prompt = self._user_prompt_fn(context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = await llm.chat(
            messages,
            temperature=0.3,
            status_callback=context.request.get("_llm_status_callback"),
        )
        content = result.get("content", "")

        # Record token usage
        usage = result.get("usage", {})
        if usage:
            try:
                await token_tracker.record_usage(
                    provider=getattr(llm, 'provider_name', 'unknown'),
                    model=getattr(llm, '_model', 'unknown'),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    agent_name=self._dim_name,
                )
            except Exception as e:
                logger.warning(f"Failed to record token usage for {self._dim_name}: {e}")

        data = extract_json(content)
        if not data:
            data = {"raw_analysis": content}

        return AgentResult(
            agent_name=self._dim_name,
            success=True,
            data=data,
            token_usage=usage,
        )
