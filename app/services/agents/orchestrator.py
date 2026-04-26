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

        # Calculate overall scores for pipelines with dimension agents
        if dim_names:
            overall = self._calculate_overall_score(pipeline_name, context)
            if overall:
                context.set_agent_result(
                    "overall_score",
                    AgentResult(
                        agent_name="overall_score",
                        success=True,
                        data=overall,
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

        # Add overall score if calculated
        overall_result = context.get_agent_result("overall_score")
        if overall_result and overall_result.success:
            result["overall_score"] = overall_result.data

        if progress_callback:
            await progress_callback("done", 1.0, "Post-stream agents completed")

        logger.info(f"Remaining agents for '{pipeline_name}' completed")
        return result

    def _calculate_overall_score(self, pipeline_name: str, context: EvaluationContext) -> Optional[dict]:
        """
        Calculate overall score from dimension results using weighted formula.

        Returns overall score dict or None if calculation not applicable.
        """
        # Weight definitions for each pipeline
        WEIGHTS = {
            "opinion_statement": {
                "op_viewpoint": 0.20,
                "op_structure": 0.10,
                "op_logic": 0.20,
                "op_expression": 0.15,
                "op_time_rhythm": 0.10,
                "speech_rate": 0.10,
                # fluency from SOE: 15%
            },
            "impromptu_reaction": {
                "ir_reaction_speed": 0.25,
                "ir_content_relevance": 0.25,
                "ir_logic": 0.20,
                "ir_expression": 0.10,
                "ir_structure": 0.05,
                # fluency from SOE: 15%
            },
            "story_reading": {
                "sr_structure": 0.30,
                "sr_logic": 0.25,
                "sr_fluency": 0.25,
                "sr_event_distribution": 0.20,
            },
            "tongue_twister_reading": {
                "tw_completeness": 0.30,
                "tw_pronunciation": 0.35,
                "tw_fluency": 0.25,
                "tw_strengths": 0.10,
            },
            "article_reading": {
                "ar_completeness": 0.25,
                "ar_pronunciation": 0.30,
                "ar_fluency": 0.25,
                "ar_pause": 0.10,
                "ar_strengths": 0.10,
            },
        }

        weights = WEIGHTS.get(pipeline_name)
        if not weights:
            return None

        total_score = 0.0
        total_weight = 0.0
        breakdown = {}

        for dim_name, weight in weights.items():
            if dim_name == "speech_rate":
                # Use speech_rate from context (0-100 scale)
                score = context.speech_rate or 0
                # Convert from 字/分钟 to 0-100 score if needed
                # speech_rate in context is actual rate, not a score
                # Skip speech_rate in weighted calculation, use SOE fluency instead
                continue
            else:
                agent_result = context.get_agent_result(f"dim_{dim_name}")
                if agent_result and agent_result.success:
                    score = agent_result.data.get("score", 0)
                else:
                    score = 0

            breakdown[f"{dim_name}_score"] = score
            total_score += score * weight
            total_weight += weight

        # Add SOE fluency score (15% for opinion_statement and impromptu_reaction)
        if pipeline_name in ("opinion_statement", "impromptu_reaction"):
            fluency_weight = 0.15
            soe_scores = context.scores_data or {}
            fluency_score = soe_scores.get("pronunciation_fluency", 0)
            breakdown["fluency_score"] = fluency_score
            total_score += fluency_score * fluency_weight
            total_weight += fluency_weight

        # Add speech_rate score (10% for opinion_statement)
        if pipeline_name == "opinion_statement":
            sr_weight = 0.10
            # Convert speech_rate to 0-100 score
            rate = context.speech_rate or 0
            if 120 <= rate <= 180:
                sr_score = 95
            elif 100 <= rate < 120 or 180 < rate <= 200:
                sr_score = 80
            elif 80 <= rate < 100 or 200 < rate <= 220:
                sr_score = 60
            else:
                sr_score = 40
            breakdown["speech_rate_score"] = sr_score
            total_score += sr_score * sr_weight
            total_weight += sr_weight

        # Calculate final score
        overall_score = round(total_score / total_weight) if total_weight > 0 else 0

        # Determine level
        if overall_score >= 85:
            level = "优秀"
        elif overall_score >= 70:
            level = "良好"
        elif overall_score >= 55:
            level = "一般"
        else:
            level = "需改进"

        # Add SOE raw scores to breakdown
        soe_scores = context.scores_data or {}
        breakdown["pronunciation_accuracy"] = soe_scores.get("pronunciation_accuracy", 0)
        breakdown["pronunciation_fluency"] = soe_scores.get("pronunciation_fluency", 0)
        breakdown["pronunciation_completion"] = soe_scores.get("pronunciation_completion", 0)
        breakdown["suggested_score"] = soe_scores.get("suggested_score", 0)
        breakdown["speech_rate_value"] = context.speech_rate or 0

        return {
            "score": overall_score,
            "level": level,
            "breakdown": breakdown,
        }

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
