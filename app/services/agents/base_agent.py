"""
Base agent interface and shared context for the multi-agent evaluation system.
Each agent is a standalone component that can be called independently or
as part of a pipeline orchestrated by the EvaluationOrchestrator.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """Result from an agent execution."""
    agent_name: str
    success: bool
    data: dict = Field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0
    token_usage: Optional[dict] = None


class EvaluationContext:
    """
    Shared mutable context passed between agents.
    Agents read from and write to this context during pipeline execution.
    """

    def __init__(self, request_data: dict):
        self.request = request_data

        # Audio data
        self.audio_data: Optional[bytes] = None
        self.audio_url: Optional[str] = None
        self.audio_duration: Optional[float] = None

        # ASR results
        self.speech_text: str = ""
        self.word_info_list: list[dict] = []

        # SOE results
        self.soe_result: Optional[dict] = None
        self.scores_data: dict = {}
        self.low_score_words: list[dict] = []
        self.statistics_data: dict = {}

        # Derived data
        self.speech_rate: Optional[float] = None

        # Agent results storage
        self.agent_results: dict[str, AgentResult] = {}

        # Request parameters
        self.language: str = request_data.get("language", "zh")
        self.ref_text: str = request_data.get("ref_text", "")
        self.eval_mode: int = request_data.get("eval_mode", 3)
        self.score_coeff: float = request_data.get("score_coeff", 1.0)
        self.server_type: int = request_data.get("server_type", 0)
        self.custom_prompt: str = request_data.get("custom_prompt", "")

    def set_agent_result(self, agent_name: str, result: AgentResult):
        """Store an agent's result."""
        self.agent_results[agent_name] = result

    def get_agent_result(self, agent_name: str) -> Optional[AgentResult]:
        """Get a specific agent's result."""
        return self.agent_results.get(agent_name)

    def has_agent_result(self, agent_name: str) -> bool:
        """Check if an agent has been executed."""
        return agent_name in self.agent_results

    def to_dict(self) -> dict:
        """Export context as a dictionary for serialization."""
        return {
            "request": self.request,
            "speech_text": self.speech_text,
            "word_info_list": self.word_info_list,
            "scores_data": self.scores_data,
            "low_score_words": self.low_score_words,
            "statistics_data": self.statistics_data,
            "audio_duration": self.audio_duration,
            "speech_rate": self.speech_rate,
            "agent_results": {
                name: result.model_dump()
                for name, result in self.agent_results.items()
            },
        }


class BaseAgent(ABC):
    """
    Abstract base class for all evaluation agents.

    Each agent is responsible for a specific aspect of the evaluation:
    - ASR: Speech-to-text conversion
    - SOE: Pronunciation scoring
    - Content: Content quality analysis (LLM-based)
    - Fluency: Speech fluency analysis (LLM-based)
    - Report: Final report generation (LLM-based)

    Agents can be called standalone or as part of a pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name, used for logging and context storage."""
        ...

    @abstractmethod
    async def execute(self, context: EvaluationContext) -> AgentResult:
        """
        Execute the agent's logic.

        Args:
            context: Shared evaluation context. Agent reads inputs from
                     context and writes results back to it.

        Returns:
            AgentResult with success status, data, and optional error.
        """
        ...

    async def _run(self, context: EvaluationContext) -> AgentResult:
        """Internal wrapper with timing and error handling."""
        start_time = time.time()
        try:
            logger.info(f"[{self.name}] Starting execution")
            result = await self.execute(context)
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            logger.info(f"[{self.name}] Completed in {duration_ms:.0f}ms (success={result.success})")
            context.set_agent_result(self.name, result)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{self.name}] Failed in {duration_ms:.0f}ms: {e}")
            result = AgentResult(
                agent_name=self.name,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration_ms=duration_ms,
            )
            context.set_agent_result(self.name, result)
            return result
