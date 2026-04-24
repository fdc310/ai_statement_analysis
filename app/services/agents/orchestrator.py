"""
Evaluation Orchestrator - Manages agent pipelines for different evaluation types.
Agents at the same dependency level run in parallel for maximum throughput.
"""
import asyncio
import logging
from typing import Optional, Callable, Awaitable

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services.agents.asr_agent import ASRAgent
from app.services.agents.soe_agent import SOEAgent
from app.services.agents.content_agent import ContentAnalysisAgent
from app.services.agents.fluency_agent import FluencyAnalysisAgent
from app.services.agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[str, float, str], Awaitable[None]]


class EvaluationOrchestrator:
    """
    Orchestrates multi-agent evaluation pipelines.

    Pipeline definitions map evaluation types to ordered agent lists.
    Agents at the same dependency level run in parallel.
    """

    # Pipeline definitions: evaluation_type -> list of agent names
    PIPELINES = {
        "basic_evaluation": ["asr", "soe", "report"],
        "extended_evaluation": ["asr", "soe", "content", "fluency", "report"],
        "opinion_statement": ["asr", "soe", "content", "fluency", "report"],
        "impromptu_reaction": ["asr", "soe", "content", "fluency", "report"],
        "story_reading": ["asr", "content", "fluency", "report"],
        "tongue_twister_reading": ["asr", "soe", "report"],
        "text_analysis": ["content"],
        "text_only": ["content"],
    }

    # Dependency levels: agents at the same level can run in parallel
    DEPENDENCY_LEVELS = {
        "asr": 0,
        "soe": 0,        # Level 0: Both need audio_data
        "content": 1,    # Level 1: Needs text + scores
        "fluency": 1,    # Level 1: Needs text + timestamps
        "report": 2,     # Level 2: Needs everything
    }

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {
            "asr": ASRAgent(),
            "soe": SOEAgent(),
            "content": ContentAnalysisAgent(),
            "fluency": FluencyAnalysisAgent(),
            "report": ReportAgent(),
        }

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def register_agent(self, agent: BaseAgent):
        """Register a custom agent."""
        self._agents[agent.name] = agent

    async def run_pipeline(
        self,
        pipeline_name: str,
        request_data: dict,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """
        Run an evaluation pipeline.

        Args:
            pipeline_name: Name of the pipeline to run
            request_data: Request data dict (will be used to create EvaluationContext)
            progress_callback: Optional async callback for progress updates

        Returns:
            Dictionary with all agent results and the final report
        """
        # Get pipeline definition
        agent_names = self.PIPELINES.get(pipeline_name)
        if not agent_names:
            raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(self.PIPELINES.keys())}")

        logger.info(f"Starting pipeline '{pipeline_name}' with agents: {agent_names}")

        # Create context
        context = EvaluationContext(request_data)

        # Group agents by dependency level
        levels: dict[int, list[str]] = {}
        for name in agent_names:
            level = self.DEPENDENCY_LEVELS.get(name, 99)
            if level not in levels:
                levels[level] = []
            levels[level].append(name)

        # Execute levels in order, agents within each level in parallel
        total_levels = len(levels)
        for level_idx, (level, names) in enumerate(sorted(levels.items())):
            progress = level_idx / total_levels
            if progress_callback:
                await progress_callback(
                    f"level_{level}",
                    progress,
                    f"Running agents: {', '.join(names)}"
                )

            logger.info(f"Level {level}: Running {names}")

            # Run agents at this level in parallel
            tasks = []
            for name in names:
                agent = self._agents.get(name)
                if agent:
                    tasks.append(agent._run(context))
                else:
                    logger.warning(f"Agent '{name}' not found, skipping")

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent '{names[i]}' raised exception: {result}")
                        context.set_agent_result(
                            names[i],
                            AgentResult(
                                agent_name=names[i],
                                success=False,
                                error=str(result)
                            )
                        )

        # Collect final results
        final_result = {
            "pipeline": pipeline_name,
            "agents_executed": agent_names,
            "agent_results": {},
        }

        for name in agent_names:
            agent_result = context.get_agent_result(name)
            if agent_result:
                final_result["agent_results"][name] = agent_result.model_dump()

        # Get the report if available
        report_result = context.get_agent_result("report")
        if report_result and report_result.success:
            final_result["report"] = report_result.data.get("report", "")

        # Get content analysis if available
        content_result = context.get_agent_result("content")
        if content_result and content_result.success:
            final_result["content_analysis"] = content_result.data

        # Get fluency analysis if available
        fluency_result = context.get_agent_result("fluency")
        if fluency_result and fluency_result.success:
            final_result["fluency_analysis"] = fluency_result.data

        # Add raw data for backward compatibility
        final_result["speech_text"] = context.speech_text
        final_result["scores_data"] = context.scores_data
        final_result["statistics_data"] = context.statistics_data
        final_result["low_score_words"] = context.low_score_words
        final_result["speech_rate"] = context.speech_rate

        if progress_callback:
            await progress_callback("done", 1.0, "Pipeline completed")

        logger.info(f"Pipeline '{pipeline_name}' completed successfully")
        return final_result


# Singleton
orchestrator = EvaluationOrchestrator()
