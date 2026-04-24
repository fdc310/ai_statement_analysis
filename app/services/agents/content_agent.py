"""
Content Analysis Agent - Analyzes content quality, logic, and structure.
Uses LLM for analysis.
"""
import logging
from typing import Optional

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services.agents.prompts.common import extract_json
from app.services.agents.prompts.text_analysis import (
    text_structure_system_prompt,
    text_structure_user_prompt,
)
from app.services.agents.prompts.opinion_statement import (
    opinion_statement_system_prompt,
    opinion_statement_user_prompt,
)
from app.services.agents.prompts.impromptu_reaction import (
    impromptu_reaction_system_prompt,
    impromptu_reaction_user_prompt,
)
from app.services.monitoring.token_tracker import token_tracker

logger = logging.getLogger(__name__)


class ContentAnalysisAgent(BaseAgent):
    """Agent for content quality, logic, and structure analysis."""

    @property
    def name(self) -> str:
        return "content"

    async def execute(self, context: EvaluationContext) -> AgentResult:
        """
        Analyze content quality using LLM.

        Reads from context:
            - speech_text: Transcribed text
            - scores_data: SOE scores
            - request: Original request (for eval_type, topic, etc.)

        Writes to context:
            - content_analysis: Analysis results
        """
        from app.services import get_llm_service

        speech_text = context.speech_text
        if not speech_text:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="No speech text to analyze"
            )

        llm = get_llm_service()
        eval_type = context.request.get("eval_type", "basic")
        topic = context.request.get("topic", "")
        custom_prompt = context.custom_prompt

        # Build prompts based on evaluation type
        system_prompt = self._get_system_prompt(eval_type, context.language)
        user_prompt = self._build_user_prompt(
            speech_text, context.scores_data, eval_type,
            topic, custom_prompt, context
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await llm.chat(messages, temperature=0.3)
        content = result.get("content", "")

        # Record token usage
        usage = result.get("usage", {})
        if usage:
            await token_tracker.record_usage(
                provider=llm.provider_name,
                model=getattr(llm, '_provider._model', 'unknown'),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                agent_name=self.name,
            )

        # Try to parse as JSON
        analysis = extract_json(content)
        if not analysis:
            # If JSON parsing fails, store raw content
            analysis = {"raw_analysis": content}

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=analysis,
            token_usage=result.get("usage"),
        )

    def _get_system_prompt(self, eval_type: str, language: str) -> str:
        """Get system prompt based on evaluation type."""
        if eval_type == "opinion_statement":
            return opinion_statement_system_prompt(language=language)
        elif eval_type == "impromptu_reaction":
            return impromptu_reaction_system_prompt(language=language)
        elif eval_type == "text_structure":
            return text_structure_system_prompt()
        else:
            return self._basic_content_system_prompt()

    def _basic_content_system_prompt(self) -> str:
        return """你是一个专业的文本内容分析专家。分析文本的内容质量、逻辑结构和表达方式。

输出纯JSON格式：
{
    "core_idea": "核心思想",
    "key_points": [{"title": "要点", "content": "内容", "importance": "高/中/低"}],
    "logic_score": 0,
    "content_score": 0,
    "strengths": ["优点1", "优点2"],
    "improvements": ["改进1", "改进2"]
}

只输出纯JSON，不要添加markdown代码块标记。"""

    def _build_user_prompt(
        self,
        speech_text: str,
        scores_data: dict,
        eval_type: str,
        topic: str,
        custom_prompt: str,
        context: EvaluationContext
    ) -> str:
        """Build user prompt based on evaluation type."""
        if eval_type == "opinion_statement":
            return opinion_statement_user_prompt(
                speech_text=speech_text,
                speech_scores=scores_data or {},
                topic=topic,
                language=context.language,
            )
        elif eval_type == "impromptu_reaction":
            return impromptu_reaction_user_prompt(
                speech_text=speech_text,
                speech_scores=scores_data or {},
                scenario=topic,
                language=context.language,
            )
        elif eval_type == "text_structure":
            return text_structure_user_prompt(
                text=speech_text,
                custom_prompt=custom_prompt,
            )
        else:
            return self._basic_content_user_prompt(speech_text, scores_data, topic, custom_prompt)

    def _basic_content_user_prompt(
        self,
        speech_text: str,
        scores_data: dict,
        topic: str,
        custom_prompt: str
    ) -> str:
        prompt = f"## 待分析文本\n\n{speech_text}\n"

        if scores_data:
            prompt += f"""
## 语音评分数据
- 发音准确度: {scores_data.get('pronunciation_accuracy', 0)}分
- 发音流利度: {scores_data.get('pronunciation_fluency', 0)}分
- 综合建议分: {scores_data.get('suggested_score', 0)}分
"""

        if topic:
            prompt += f"\n## 话题/场景\n{topic}\n"

        if custom_prompt:
            prompt += f"\n## 额外要求\n{custom_prompt}\n"

        prompt += "\n请严格按照JSON格式输出分析结果。"
        return prompt
