"""
Schemas for speech evaluation API.
"""
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class ReportType(str, Enum):
    """报告类型枚举"""
    simple = "simple"  # 简洁报告
    full = "full"      # 完整报告


class SpeechScores(BaseModel):
    """Speech evaluation scores."""

    pronunciation_accuracy: float = Field(
        0.0, description="Pronunciation accuracy score (0-100)"
    )
    pronunciation_fluency: float = Field(
        0.0, description="Pronunciation fluency score (0-100)"
    )
    pronunciation_completion: float = Field(
        0.0, description="Pronunciation completion score (0-100)"
    )
    suggested_score: float = Field(
        0.0, description="Suggested overall score (0-100)"
    )
    overall_score: float = Field(
        0.0, description="Calculated overall score (0-100)"
    )


class WordScore(BaseModel):
    """Individual word score."""

    word: str = Field(..., description="The word")
    accuracy: float = Field(..., description="Accuracy score")
    fluency: float = Field(..., description="Fluency score")


class EvaluationStatistics(BaseModel):
    """Evaluation statistics."""

    total_words: int = Field(0, description="Total number of words")
    average_accuracy: float = Field(0.0, description="Average accuracy score")
    low_score_count: int = Field(0, description="Number of low-scoring words")


class EvaluationRequest(BaseModel):
    """Request model for speech evaluation API."""

    # Audio source (one of the following)
    audio_url: Optional[str] = Field(
        None, description="URL of the audio file to evaluate"
    )
    audio_path: Optional[str] = Field(
        None, description="Local path of the audio file to evaluate"
    )

    # Evaluation options
    ref_text: Optional[str] = Field(
        "", description="Reference text for speech evaluation (optional)"
    )
    custom_prompt: Optional[str] = Field(
        None, description="Custom prompt for AI evaluation generation"
    )
    language: str = Field(
        "zh", description="Language: 'zh' for Chinese, 'en' for English"
    )

    # Async callback
    message_id: Optional[str] = Field(
        None, description="Message ID for tracking (auto-generated if not provided)"
    )
    callback_url: HttpUrl = Field(
        ..., description="Callback URL to receive evaluation results"
    )


class EvaluationAcceptedResponse(BaseModel):
    """Response model when evaluation task is accepted (includes SOE scores)."""

    success: bool = Field(True, description="Whether the task was accepted")
    message: str = Field("Task accepted", description="Status message")
    message_id: str = Field(..., description="Message ID for tracking")

    # SOE results (returned immediately)
    speech_text: Optional[str] = Field(
        None, description="Transcribed speech text"
    )
    speech_scores: Optional[SpeechScores] = Field(
        None, description="Speech evaluation scores"
    )
    statistics: Optional[EvaluationStatistics] = Field(
        None, description="Evaluation statistics"
    )
    low_score_words: Optional[List[WordScore]] = Field(
        None, description="Words with low pronunciation scores"
    )

    # Error info
    error: Optional[str] = Field(
        None, description="Error message if SOE evaluation failed"
    )


class EvaluationCallbackData(BaseModel):
    """Data sent to callback URL when AI report generation completes."""

    message_id: str = Field(..., description="Message ID for tracking")
    success: bool = Field(..., description="Whether the report generation succeeded")
    message: str = Field(..., description="Status message")

    # AI Report (sent via callback)
    evaluation_report: Optional[str] = Field(
        None, description="AI-generated evaluation report in Markdown format"
    )

    # Error info
    error: Optional[str] = Field(
        None, description="Error message if report generation failed"
    )


# Keep old response model for backward compatibility
class EvaluationResponse(BaseModel):
    """Response model for speech evaluation API (deprecated, use callback instead)."""

    success: bool = Field(..., description="Whether the evaluation succeeded")
    message: str = Field(..., description="Status message")

    # Results
    speech_text: Optional[str] = Field(
        None, description="Transcribed speech text"
    )
    speech_scores: Optional[SpeechScores] = Field(
        None, description="Speech evaluation scores"
    )
    statistics: Optional[EvaluationStatistics] = Field(
        None, description="Evaluation statistics"
    )
    low_score_words: Optional[List[WordScore]] = Field(
        None, description="Words with low pronunciation scores"
    )
    evaluation_report: Optional[str] = Field(
        None, description="AI-generated evaluation report in Markdown format"
    )

    # Error info
    error: Optional[str] = Field(
        None, description="Error message if evaluation failed"
    )


class SignatureRequest(BaseModel):
    """Request model for signature generation."""

    aes_key: str = Field(..., description="AES key for signature generation")


class SignatureResponse(BaseModel):
    """Response model for signature generation."""

    success: bool = Field(..., description="Whether the generation succeeded")
    signature: Optional[str] = Field(None, description="Generated signature")
    timestamp: Optional[int] = Field(None, description="Timestamp used in signature")
    expires_in: Optional[int] = Field(None, description="Seconds until signature expires")
    error: Optional[str] = Field(None, description="Error message if failed")


class ReportRequest(BaseModel):
    """Request model for AI report generation with pre-existing SOE scores."""

    audio_url: HttpUrl = Field(..., description="音频文件URL")
    speech_text: Optional[str] = Field(None, description="语音转写文本，不传则自动调用ASR识别")
    soe_result: dict = Field(..., description="SOE评测返回的result数据，包含SuggestedScore、PronAccuracy、Words等字段")
    audio_duration: Optional[float] = Field(None, description="音频时长（秒），用于计算语速")
    topic: Optional[str] = Field(None, description="演讲主题，用于分析内容贴题性。不传则为自由说模式")
    custom_prompt: Optional[str] = Field(None, description="自定义AI评测提示词")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")
    language: str = Field(default="zh", description="语言：'zh'中文，'en'英文，用于ASR识别")
    report_type: ReportType = Field(default=ReportType.full, description="报告类型：'simple'简洁报告，'full'完整报告")


class ReportResponse(BaseModel):
    """Response model for AI report generation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    audio_url: str = Field(..., description="音频URL")
    speech_text: Optional[str] = Field(None, description="语音转写文本（ASR识别结果）")
    speech_rate: Optional[float] = Field(None, description="语速（字/分钟或词/分钟）")
    evaluation_report: Optional[dict] = Field(None, description="AI生成的JSON格式评测报告")
    error: Optional[str] = Field(None, description="错误信息")


# JSON格式的报告模型
class ParagraphAnalysis(BaseModel):
    """段落分析"""
    paragraph_index: int = Field(..., description="段落索引（从1开始）")
    content: str = Field(..., description="段落内容")
    low_score_words: List[dict] = Field(default_factory=list, description="该段落中的低分字词")
    suggestion: str = Field(..., description="改进建议")


class SpeechRateAnalysis(BaseModel):
    """语速分析"""
    rate: float = Field(..., description="语速（字/分钟或词/分钟）")
    score: int = Field(..., description="语速评分（0-100）")
    level: str = Field(..., description="语速等级：优秀/良好/一般/较差")
    suggestion: str = Field(..., description="语速建议")


class SimpleReport(BaseModel):
    """简洁报告结构"""
    speech_rate: SpeechRateAnalysis = Field(..., description="语速分析")
    weak_paragraphs: List[ParagraphAnalysis] = Field(default_factory=list, description="读的不太好的段落")
    overall_suggestion: str = Field(..., description="整体建议")


class LogicCompletenessScore(BaseModel):
    """逻辑完整性评分"""
    overall_score: int = Field(..., description="综合评分（0-100）")
    logic_score: int = Field(..., description="逻辑性评分（0-100）")
    fluency_score: int = Field(..., description="流畅度评分（0-100）")
    speech_rate_score: int = Field(..., description="语速评分（0-100）")
    topic_relevance_score: Optional[int] = Field(None, description="贴题性评分（0-100），如有主题")
    speech_rate_value: float = Field(..., description="语速数值")
    speech_rate_level: str = Field(..., description="语速等级：优秀/良好/一般/较差")
    speech_rate_suggestion: str = Field(..., description="语速建议")


class StructureVisualization(BaseModel):
    """结构可视化"""
    arguments: List[str] = Field(default_factory=list, description="论点列表")
    conclusion: str = Field(..., description="结论要点")


class SpeechRateEvaluation(BaseModel):
    """语速评价"""
    score: int = Field(..., description="语速评分（0-100）")
    rate_value: float = Field(..., description="语速数值")
    level: str = Field(..., description="语速等级：优秀/良好/一般/较差")
    analysis: str = Field(..., description="语速分析")
    suggestion: str = Field(..., description="语速改进建议")


class ContentPerspective(BaseModel):
    """内容角度"""
    score: int = Field(..., description="内容角度评分（0-100）")
    topic_relevance: str = Field(..., description="贴题性分析")
    depth: str = Field(..., description="内容深度分析")
    coverage: str = Field(..., description="内容覆盖面分析")
    suggestion: str = Field(..., description="内容改进建议")


class LogicStructure(BaseModel):
    """逻辑与结构"""
    score: int = Field(..., description="逻辑结构评分（0-100）")
    organization: str = Field(..., description="整体结构分析")
    coherence: str = Field(..., description="连贯性分析")
    reasoning: str = Field(..., description="论证逻辑分析")
    suggestion: str = Field(..., description="逻辑结构改进建议")


class ExpressionWording(BaseModel):
    """表达与用词"""
    score: int = Field(..., description="表达用词评分（0-100）")
    vocabulary_level: str = Field(..., description="用词水平分析")
    expression_style: str = Field(..., description="表达风格分析")
    highlights: List[str] = Field(default_factory=list, description="表达亮点")
    suggestion: str = Field(..., description="表达用词改进建议")


class FullReport(BaseModel):
    """完整报告结构"""
    logic_completeness: LogicCompletenessScore = Field(..., description="逻辑完整性评分")
    structure_visualization: StructureVisualization = Field(..., description="结构可视化")
    speech_rate_evaluation: SpeechRateEvaluation = Field(..., description="语速评价")
    content_perspective: ContentPerspective = Field(..., description="内容角度")
    logic_structure: LogicStructure = Field(..., description="逻辑与结构")
    expression_wording: ExpressionWording = Field(..., description="表达与用词")
    strengths: List[str] = Field(default_factory=list, description="优点")
    improvements: List[str] = Field(default_factory=list, description="改进意见")
    weak_paragraphs: List[ParagraphAnalysis] = Field(default_factory=list, description="读的不太好的段落")


class TextAnalysisRequest(BaseModel):
    """Request model for text structure analysis."""

    text: str = Field(..., description="待分析的文本内容", min_length=10, max_length=50000)
    custom_prompt: Optional[str] = Field(None, description="自定义分析要求")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class TextAnalysisResponse(BaseModel):
    """Response model for text structure analysis."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    analysis_result: Optional[str] = Field(None, description="分析结果（JSON格式字符串）")
    error: Optional[str] = Field(None, description="错误信息")


class TongueTwisterRequest(BaseModel):
    """Request model for tongue twister pronunciation analysis."""

    text: str = Field(..., description="绕口令文本", min_length=2, max_length=5000)
    language: str = Field(default="zh", description="语言：'zh'中文，'en'英文")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class TongueTwisterResponse(BaseModel):
    """Response model for tongue twister pronunciation analysis."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    tongue_twister: str = Field(..., description="绕口令原文")
    analysis_result: Optional[str] = Field(None, description="发音分析结果（JSON格式字符串）")
    error: Optional[str] = Field(None, description="错误信息")


class SentenceInterpretationRequest(BaseModel):
    """Request model for sentence interpretation."""

    text: str = Field(..., description="待解读的句子内容", min_length=1, max_length=5000)
    custom_prompt: Optional[str] = Field(None, description="自定义解读要求")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class SentenceInterpretationResponse(BaseModel):
    """Response model for sentence interpretation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    interpretation: Optional[dict] = Field(None, description="AI解读结果JSON对象，包含中心内容、朗读重点、注意事项")
    error: Optional[str] = Field(None, description="错误信息")


class StoryReadingRequest(BaseModel):
    """Request model for story reading evaluation."""

    audio_url: HttpUrl = Field(..., description="音频文件URL")
    story_text: str = Field(..., description="短故事文本，用户要围绕此故事发挥", min_length=10, max_length=2000)
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class StoryReadingResponse(BaseModel):
    """Response model for story reading evaluation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")

    # 结构完整性分析
    structure_analysis: Optional[dict] = Field(
        None, description="结构完整性分析（开头、发展、高潮、结尾）"
    )

    # 逻辑连贯性分析
    logic_analysis: Optional[dict] = Field(
        None, description="逻辑连贯性分析（时间跳跃、因果错误、事件遗漏、逻辑矛盾）"
    )

    # 语言流畅度分析
    fluency_analysis: Optional[dict] = Field(
        None, description="语言流畅度分析（长停顿、重复修正、填空词、完整度）"
    )

    # 事件分布分析
    event_distribution: Optional[dict] = Field(
        None, description="事件分布（各事件的时间位置和时长）"
    )

    # 待改进
    improvements: List[str] = Field(
        default_factory=list, description="待改进建议列表"
    )

    # ASR时间戳数据
    asr_data: Optional[dict] = Field(
        None, description="ASR识别结果（带时间戳）"
    )

    error: Optional[str] = Field(None, description="错误信息")


class TongueTwisterReadingRequest(BaseModel):
    """Request model for tongue twister / article speech evaluation."""

    audio_url: HttpUrl = Field(..., description="音频文件URL")
    tongue_twister_text: str = Field(
        ..., description="原文文本（绕口令或文章）", min_length=2, max_length=5000
    )
    eval_type: str = Field(
        default="tongue_twister",
        description="评测类型：tongue_twister(绕口令，默认)、article(文章朗读)"
    )
    score_coeff: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description="SOE评分苛刻指数：1.0(儿童/宽松) 2.0(标准) 4.0(成人/严格)"
    )
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class TongueTwisterReadingResponse(BaseModel):
    """Response model for tongue twister speech evaluation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")

    # SOE完整评分数据
    speech_scores: Optional[SpeechScores] = Field(
        None, description="语音评测评分（准确度、流利度、完整度等）"
    )
    statistics: Optional[EvaluationStatistics] = Field(
        None, description="评测统计数据"
    )
    soe_words: Optional[List[dict]] = Field(
        None, description="SOE逐字评分详情（每个字的准确度、流利度、音素信息）"
    )
    low_score_words: Optional[List[dict]] = Field(
        None, description="低分字词列表（准确度<90分）"
    )
    soe_sentences: Optional[List[dict]] = Field(
        None, description="SOE句子级评分详情"
    )
    soe_data: Optional[dict] = Field(
        None, description="SOE完整原始评测数据"
    )

    # AI分析结果
    strengths: List[str] = Field(
        default_factory=list, description="优势列表"
    )
    improvements: Optional[dict] = Field(
        None,
        description="待提升分析，包含extra_words(多读)、missed_words(漏读)、pronunciation_issues(发音问题)"
    )
    fluency_analysis: Optional[dict] = Field(
        None, description="流畅度分析（基于时间戳数据）"
    )
    overall_assessment: Optional[str] = Field(
        None, description="总体评价"
    )
    practice_suggestions: List[str] = Field(
        default_factory=list, description="练习建议列表"
    )
    speech_rate_analysis: Optional[dict] = Field(
        None, description="语速分析（仅article模式，含整体语速、分段语速、快慢段落）"
    )
    pause_analysis: Optional[dict] = Field(
        None, description="断句停顿分析（仅article模式，含正确停顿、不当停顿、遗漏停顿）"
    )

    # ASR完整数据
    asr_data: Optional[dict] = Field(
        None, description="ASR识别结果（含时间戳），包含text和word_info_list"
    )

    error: Optional[str] = Field(None, description="错误信息")


class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: str = Field(..., description="消息角色：system / user / assistant")
    content: str = Field(..., description="消息内容")


class VoiceChatRequest(BaseModel):
    """Request model for voice chat conversation."""

    audio_url: HttpUrl = Field(..., description="用户语音文件URL")
    messages: Optional[List[ChatMessage]] = Field(
        None, description="对话历史（不含本次语音），按时间顺序排列"
    )
    system_prompt: Optional[str] = Field(
        None, description="自定义系统提示词，优先级高于scene预设场景"
    )
    scene: Optional[str] = Field(
        None,
        description="预设场景类型：interview(面试)、daily(日常对话)、customer_service(客服)"
    )
    voice_type: int = Field(
        default=101001,
        description="TTS音色ID：101001(智瑜-女)、101005(智华-男)、101050(英文女)、101051(英文男)"
    )
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class VoiceChatResponse(BaseModel):
    """Response model for voice chat conversation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")

    user_text: Optional[str] = Field(None, description="ASR识别的用户语音文本")
    assistant_text: Optional[str] = Field(None, description="AI回复文本")
    audio_base64: Optional[str] = Field(None, description="AI回复的TTS音频Base64编码(mp3格式)")

    # ASR原始数据
    asr_data: Optional[dict] = Field(
        None, description="ASR识别结果（含时间戳），包含text和word_info_list"
    )

    error: Optional[str] = Field(None, description="错误信息")


class OpinionStatementRequest(BaseModel):
    """Request model for one-minute opinion statement evaluation."""

    audio_url: HttpUrl = Field(..., description="音频文件URL")
    ref_text: Optional[str] = Field(None, description="参考文本，用于SOE评测对照（不传则SOE使用自由说模式）")
    topic: Optional[str] = Field(None, description="观点陈述的题目/话题，用于分析贴题性")
    score_coeff: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description="SOE评分苛刻指数：1.0(宽松) 2.0(标准) 4.0(严格)"
    )
    language: str = Field(default="zh", description="语言：'zh'中文，'en'英文")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class OpinionStatementResponse(BaseModel):
    """Response model for one-minute opinion statement evaluation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    audio_url: str = Field(..., description="音频URL")
    speech_text: Optional[str] = Field(None, description="语音转写文本（ASR识别结果）")
    speech_rate: Optional[float] = Field(None, description="语速（字/分钟或词/分钟）")

    # SOE评分数据
    speech_scores: Optional[SpeechScores] = Field(None, description="语音评测评分")
    statistics: Optional[EvaluationStatistics] = Field(None, description="评测统计数据")
    low_score_words: Optional[List[WordScore]] = Field(None, description="低分字词列表")

    # AI评测报告
    evaluation_report: Optional[dict] = Field(None, description="AI生成的观点陈述评测报告（JSON格式）")
    error: Optional[str] = Field(None, description="错误信息")
class ImpromptuReactionRequest(BaseModel):
    """Request model for impromptu reaction (即兴反应) evaluation."""

    audio_url: HttpUrl = Field(..., description="音频文件URL")
    scenario: str = Field(..., description="触发情境/题目")
    score_coeff: float = Field(
        default=3.5,
        ge=1.0,
        le=4.0,
        description="SOE评分苛刻指数：默认3.5（偏严格）"
    )
    language: str = Field(default="zh", description="语言：'zh'中文，'en'英文")
    message_id: Optional[str] = Field(None, description="消息ID，不传则自动生成UUID")


class ImpromptuReactionResponse(BaseModel):
    """Response model for impromptu reaction evaluation."""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态消息")
    message_id: str = Field(..., description="消息ID")
    audio_url: str = Field(..., description="音频URL")
    speech_text: Optional[str] = Field(None, description="语音转写文本（ASR识别结果）")
    speech_rate: Optional[float] = Field(None, description="语速（字/分钟或词/分钟）")

    # SOE评分数据
    speech_scores: Optional[SpeechScores] = Field(None, description="语音评测评分")
    statistics: Optional[EvaluationStatistics] = Field(None, description="评测统计数据")
    low_score_words: Optional[List[WordScore]] = Field(None, description="低分字词列表")

    # AI评测报告
    evaluation_report: Optional[dict] = Field(None, description="AI生成的即兴反应评测报告（JSON格式）")
    error: Optional[str] = Field(None, description="错误信息")
