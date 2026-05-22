# LLM Service - 多供应商支持

项目支持多个大模型提供商，采用 Provider 模式架构，可灵活扩展。

## 支持的供应商

1. **OpenAI** (默认) - 标准 OpenAI API 及兼容 API（如 Azure、Ollama、vLLM 等）
2. **Tencent** (腾讯混元) - 腾讯云原生 SDK
3. **Anthropic** - Claude 系列模型

## 安装依赖

```bash
pip install openai>=1.0.0 anthropic>=0.18.0
```

## 配置

在 `.env` 文件中添加以下配置：

```bash
# 通用 LLM 配置
LLM_PROVIDER=openai  # openai / tencent / anthropic
LLM_TIMEOUT=120

# OpenAI 配置（默认）
OPENAI_MODEL=gpt-4o
OPENAI_MULTIMODAL_MODEL=gpt-4o-audio-preview
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 腾讯混元配置
TENCENT_MODEL=hunyuan-turbo
TENCENT_MULTIMODAL_MODEL=hunyuan-multimodal

# Anthropic 配置
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_MULTIMODAL_MODEL=claude-sonnet-4-20250514
```

## 使用方法

### 1. 使用 OpenAI（默认）

```python
from app.services.llm_service import LLMService

llm = LLMService(provider="openai")

result = await llm.chat([
    {"role": "user", "content": "你好"}
])
print(result["content"])
```

### 2. 使用腾讯混元

```python
from app.services.llm_service import LLMService

llm = LLMService(provider="tencent")

result = await llm.chat([
    {"role": "user", "content": "你好"}
])
print(result["content"])
```

### 3. 使用 Anthropic Claude

```python
from app.services.llm_service import LLMService

llm = LLMService(provider="anthropic")

result = await llm.chat([
    {"role": "user", "content": "你好"}
])
print(result["content"])
```

### 4. 使用本地模型（如 Ollama）

```python
from app.services.llm_service import LLMService

llm = LLMService(
    provider="openai",
    model="llama3-70b",
    api_key="dummy-key",  # 本地 API 可能不需要真实 key
    base_url="http://localhost:11434/v1"
)

result = await llm.chat([
    {"role": "user", "content": "Hello"}
])
print(result["content"])
```

### 5. 流式输出

```python
from app.services.llm_service import LLMService

llm = LLMService(provider="openai")

async for chunk in llm.chat_stream([
    {"role": "user", "content": "请写一首诗"}
]):
    print(chunk, end="", flush=True)
```

### 6. 多模态对话（音频输入）

```python
from app.services.llm_service import LLMService

llm = LLMService(provider="openai")

# 直接发送音频给多模态模型
result = await llm.chat_multimodal(
    audio_url="https://example.com/user-speech.mp3",
    messages=[],
    system_prompt="你是一位友善的对话伙伴。",
    temperature=0.7,
)
print(result["content"])  # AI回复文本
```

## Provider 架构

```
LLMService (统一入口)
    │
    └── ProviderRegistry.get_provider() → BaseLLMProvider
            │
            ├── OpenAIProvider (OpenAI / Azure / Ollama / vLLM)
            ├── TencentProvider (腾讯原生 SDK)
            └── AnthropicProvider (Claude)
```

### 扩展新供应商

1. 创建新文件 `app/services/llm/your_provider.py`
2. 实现 `BaseLLMProvider` 抽象基类
3. 在 `app/services/llm/registry.py` 中注册

```python
from app.services.llm.base import BaseLLMProvider, ChatResponse

class YourProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "your_provider"

    async def chat(self, messages, **kwargs) -> ChatResponse:
        # 实现对话逻辑
        ...

    async def chat_stream(self, messages, **kwargs):
        # 实现流式对话
        ...

    async def chat_multimodal(self, audio_url, messages, system_prompt, **kwargs) -> ChatResponse:
        # 实现多模态对话
        ...
```

## 兼容性

所有现有的 LLM 方法都已迁移到新的 Provider 架构：

- `chat()` - 基础对话
- `chat_stream()` - 流式对话
- `chat_multimodal()` - 多模态对话（音频输入）
- `generate_evaluation()` - 生成评测报告
- `generate_evaluation_extended()` - 生成扩展评测报告
- `generate_simple_report_json()` - 生成简洁 JSON 报告
- `generate_full_report_json()` - 生成完整 JSON 报告
- `analyze_text_structure()` - 分析文本结构
- `analyze_tongue_twister()` - 分析绕口令
- `analyze_sentence_interpretation()` - 分析句子朗读
- `analyze_story_reading()` - 分析故事阅读
- `analyze_tongue_twister_reading()` - 分析绕口令朗读

## 注意事项

1. 所有供应商使用统一的消息格式（`role`/`content`），Provider 内部负责转换
2. 超时时间通过 `LLM_TIMEOUT` 统一配置
3. 多模态对话需要配置对应的多模态模型
4. 切换供应商只需修改 `LLM_PROVIDER` 环境变量
