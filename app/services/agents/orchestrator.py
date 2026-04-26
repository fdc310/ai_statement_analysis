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
from app.services.agents.dimension_agent import DimensionAgent
from app.services.agents.dimensions import DIMENSION_REGISTRY, PIPELINE_DIMENSIONS

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
    # Dimension agents (dim_*) are registered dynamically from DIMENSION_REGISTRY
    PIPELINES = {
        "basic_evaluation": ["asr", "soe", "report"],
        "extended_evaluation": ["asr", "soe", "content", "fluency", "report"],
        "opinion_statement": ["asr", "soe", "content", "fluency", "report"],
        "impromptu_reaction": ["asr", "soe", "content", "fluency", "report"],
        "story_reading": ["asr", "content", "fluency", "report"],
        "tongue_twister_reading": ["asr", "soe", "report"],
        "article_reading": ["asr", "soe", "report"],
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

        # Register dimension agents
        for dim_name, (sys_fn, usr_fn) in DIMENSION_REGISTRY.items():
            agent_name = f"dim_{dim_name}"
            self._agents[agent_name] = DimensionAgent(
                dim_name=agent_name,
                system_prompt_fn=sys_fn,
                user_prompt_fn=usr_fn,
            )
            self.DEPENDENCY_LEVELS[agent_name] = 1  # All at Level 1 (parallel)

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

    async def run_remaining_agents(
        self,
        pipeline_name: str,
        context: EvaluationContext,
        progress_callback: Optional[ProgressCallback] = None,
        on_agent_result: Optional[Callable[[str, AgentResult], Awaitable[None]]] = None,
    ) -> dict:
        """
        Run only the post-ASR/SOE agents (Level 1+) for a streaming session.

        Designed for streaming where ASR and SOE are already completed
        by StreamingASR/StreamingSOE. Skips Level 0 agents and runs
        Level 1 (content, fluency, dimension agents) then Level 2 (report).

        If the pipeline has dimension agents defined in PIPELINE_DIMENSIONS,
        those are used instead of the old monolithic agents.

        Args:
            pipeline_name: Pipeline name from PIPELINES dict
            context: Pre-built EvaluationContext with ASR/SOE data already populated
            progress_callback: Async callback for level-level progress
            on_agent_result: Async callback after each agent completes,
                             receives (agent_name, AgentResult). Used for
                             progressive WebSocket streaming.

        Returns:
            Dictionary with agent results and final report
        """
        # Check if this pipeline has dimension agents
        dim_names = PIPELINE_DIMENSIONS.get(pipeline_name)

        if dim_names:
            # Use dimension agents (all at Level 1, run in parallel)
            remaining_agents = [f"dim_{d}" for d in dim_names]
            logger.info(f"Running dimension agents for '{pipeline_name}': {remaining_agents}")
        else:
            # Fall back to traditional pipeline
            agent_names = self.PIPELINES.get(pipeline_name)
            if not agent_names:
                raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(self.PIPELINES.keys())}")

            remaining_agents = [
                name for name in agent_names
                if self.DEPENDENCY_LEVELS.get(name, 99) > 0
            ]

        if not remaining_agents:
            logger.info(f"No remaining agents for pipeline '{pipeline_name}'")
            return {}

        logger.info(f"Running remaining agents for '{pipeline_name}': {remaining_agents}")

        # Group remaining agents by dependency level
        levels: dict[int, list[str]] = {}
        for name in remaining_agents:
            level = self.DEPENDENCY_LEVELS.get(name, 1)
            levels.setdefault(level, []).append(name)

        total_levels = len(levels)
        for level_idx, (level, names) in enumerate(sorted(levels.items())):
            progress = level_idx / total_levels
            if progress_callback:
                await progress_callback(
                    f"level_{level}", progress,
                    f"Running agents: {', '.join(names)}"
                )

            logger.info(f"Level {level}: Running {names}")

            # Run agents at this level in parallel
            tasks = []
            for name in names:
                agent = self._agents.get(name)
                if agent:
                    tasks.append(self._run_and_notify(agent, context, on_agent_result))
                else:
                    logger.warning(f"Agent '{name}' not found, skipping")

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

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

        # Collect results
        result = {
            "pipeline": pipeline_name,
            "agents_executed": remaining_agents,
            "agent_results": {},
        }

        for name in remaining_agents:
            agent_result = context.get_agent_result(name)
            if agent_result:
                result["agent_results"][name] = agent_result.model_dump()

        # Convenience keys from traditional agents (if present)
        report_result = context.get_agent_result("report")
        if report_result and report_result.success:
            result["report"] = report_result.data.get("report", "")

        content_result = context.get_agent_result("content")
        if content_result and content_result.success:
            result["content_analysis"] = content_result.data

        fluency_result = context.get_agent_result("fluency")
        if fluency_result and fluency_result.success:
            result["fluency_analysis"] = fluency_result.data

        result["speech_rate"] = context.speech_rate
        result["statistics_data"] = context.statistics_data
        result["low_score_words"] = context.low_score_words

        if progress_callback:
            await progress_callback("done", 1.0, "Post-stream agents completed")

        logger.info(f"Remaining agents for '{pipeline_name}' completed")
        return result

    async def _run_and_notify(
        self,
        agent: BaseAgent,
        context: EvaluationContext,
        on_agent_result: Optional[Callable[[str, AgentResult], Awaitable[None]]] = None,
    ) -> AgentResult:
        """Run an agent and notify callback on completion."""
        result = await agent._run(context)
        if on_agent_result:
            try:
                await on_agent_result(agent.name, result)
            except Exception as e:
                logger.error(f"Agent result callback error for '{agent.name}': {e}")
        return result


# Singleton
orchestrator = EvaluationOrchestrator()
