"""
Fluency Analysis Agent - Analyzes speech fluency, rate, and pauses.
Uses LLM for analysis.
"""
import logging

from app.services.agents.base_agent import BaseAgent, AgentResult, EvaluationContext
from app.services.agents.prompts.common import extract_json, build_word_info_table
from app.services.monitoring.token_tracker import token_tracker

logger = logging.getLogger(__name__)


class FluencyAnalysisAgent(BaseAgent):
    """Agent for speech fluency, rate, and pause analysis."""

    @property
    def name(self) -> str:
        return "fluency"

    async def execute(self, context: EvaluationContext) -> AgentResult:
        """
        Analyze speech fluency using LLM.

        Reads from context:
            - speech_text: Transcribed text
            - word_info_list: Word-level timestamps
            - audio_duration: Audio duration in seconds
            - scores_data: SOE scores (for fluency data)

        Writes to context:
            - fluency_analysis: Analysis results
            - speech_rate: Calculated speech rate
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

        # Calculate speech rate
        char_count = len(speech_text.replace(" ", "").replace("\n", ""))
        duration = context.audio_duration
        if duration and duration > 0:
            speech_rate = char_count / (duration / 60)  # chars per minute
            context.speech_rate = round(speech_rate, 1)
        else:
            speech_rate = 0

        # Build prompts
        system_prompt = self._get_system_prompt(context.language)
        user_prompt = self._build_user_prompt(
            speech_text, context.word_info_list, speech_rate,
            duration, context.scores_data, context.low_score_words
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
            await token_tracker.record_usage(
                provider=llm.provider_name,
                model=llm.model_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                agent_name=self.name,
            )

        analysis = extract_json(content)
        if not analysis:
            analysis = {"raw_analysis": content}

        # Add calculated speech rate to analysis
        analysis["calculated_speech_rate"] = speech_rate
        analysis["audio_duration"] = duration

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=analysis,
            token_usage=result.get("usage"),
        )

    def _get_system_prompt(self, language: str) -> str:
        return """你是一个专业的语音流畅度分析专家。分析语音的流畅度、语速和停顿模式。

评估维度：
1. 语速评估: 中文正常语速120-180字/分钟
2. 流畅度: 是否有不自然的停顿、重复、修正
3. 停顿分析: 停顿是否合理（句间停顿、思考停顿）
4. 连贯性: 语句之间是否连贯

输出纯JSON格式：
{
    "speech_rate_score": 0,
    "speech_rate_analysis": "语速分析",
    "fluency_score": 0,
    "fluency_analysis": "流畅度分析",
    "pause_analysis": {
        "total_pauses": 0,
        "avg_pause_duration": 0,
        "proper_pauses": 0,
        "improper_pauses": 0,
        "analysis": "停顿分析"
    },
    "overall_fluency_score": 0,
    "strengths": [],
    "improvements": []
}

只输出纯JSON，不要添加markdown代码块标记。"""

    def _build_user_prompt(
        self,
        speech_text: str,
        word_info_list: list,
        speech_rate: float,
        duration: float,
        scores_data: dict,
        low_score_words: list
    ) -> str:
        prompt = f"## 语音转文字内容\n\n{speech_text}\n"

        prompt += f"""
## 基本数据
- 语速: {speech_rate:.1f} 字/分钟
- 音频时长: {duration:.1f} 秒
- 字数: {len(speech_text.replace(' ', '').replace(chr(10), ''))} 字
"""

        if scores_data:
            prompt += f"- 发音流利度: {scores_data.get('pronunciation_fluency', 0)}分\n"

        if word_info_list:
            prompt += f"\n## 字词时间戳\n\n{build_word_info_table(word_info_list)}\n"

        if low_score_words:
            prompt += f"\n## 低分字词 ({len(low_score_words)}个)\n"
            for w in low_score_words[:10]:
                prompt += f"- {w.get('word', '')}: 准确度 {w.get('accuracy', 0)}\n"

        prompt += "\n请严格按照JSON格式输出分析结果。"
        return prompt
