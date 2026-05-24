# AI Statement Analysis API

基于 FastAPI 的 AI 语音评测分析服务，集成腾讯云语音服务（ASR、SOE、TTS）和大语言模型（OpenAI/腾讯混元/Anthropic），提供语音识别、口语评测、语音合成及 AI 智能评价报告生成。

## 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **大语言模型**: OpenAI (默认) / 腾讯混元 / Anthropic Claude
- **语音服务**: 腾讯云 ASR (语音识别) / SOE (口语评测) / TTS (语音合成)
- **音频处理**: FFmpeg
- **对象存储**: 阿里云 OSS / MinIO (S3 兼容) / POST 接口上传
- **请求鉴权**: AES-256-CBC 加密签名
- **容器化**: Docker / Docker Compose

## 项目结构

```
ai_statement_analysis/
├── app/
│   ├── main.py                          # FastAPI 应用入口
│   ├── api/v1/
│   │   ├── router.py                    # 路由聚合
│   │   └── endpoints/
│   │       ├── auth.py                  # 健康检查
│   │       ├── evaluation.py            # 评测接口（核心）
│   │       ├── soe.py                   # 口语评测接口
│   │       └── tts.py                   # 语音合成接口
│   ├── core/
│   │   ├── config.py                    # 配置管理
│   │   ├── security.py                  # AES 加密鉴权
│   │   └── util/                        # 腾讯云语音 SDK
│   ├── schemas/
│   │   ├── base.py                      # 基础响应模型
│   │   ├── evaluation.py                # 评测请求/响应模型
│   │   └── soe.py                       # SOE 请求/响应模型
│   └── services/
│       ├── llm_service.py               # 统一 LLM 服务（Provider 模式）
│       ├── llm/                         # LLM 供应商实现
│       │   ├── base.py                  # 抽象基类
│       │   ├── openai_provider.py       # OpenAI 兼容供应商
│       │   ├── tencent_provider.py      # 腾讯原生供应商
│       │   ├── anthropic_provider.py    # Anthropic Claude 供应商
│       │   └── registry.py              # 供应商注册表
│       ├── s3_storage.py                # 对象存储服务（OSS/API 双模式）
│       ├── chat/
│       │   └── session_manager.py       # 语音对话会话管理
│       ├── agents/                      # 评测代理（多步骤流水线）
│       ├── tasks/                       # 异步任务管理
│       ├── streaming/                   # 实时流式评测
│       └── tencent/
│           ├── asr.py                   # 语音识别（极速识别）
│           ├── audio.py                 # 音频格式转换（FFmpeg）
│           ├── soe.py                   # 口语评测服务
│           └── tts.py                   # 语音合成服务
├── requirements.txt                     # Python 依赖
├── .env.example                         # 环境变量模板
├── run.py                               # 启动脚本
└── docker-compose.yml                   # Docker 配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要系统安装 FFmpeg：
```bash
# Ubuntu/Debian
apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 下载 https://ffmpeg.org/download.html 并添加到 PATH
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入实际的密钥和配置
```

### 3. 启动服务

```bash
python run.py
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 接口

所有接口（除健康检查外）需要在请求头中携带 `X-Signature` 进行鉴权。

### 鉴权

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/evaluation/signature` | POST | 生成 AES 签名 |

签名生成方式：将 `{"timestamp": 当前时间戳}` 用 AES-256-CBC 加密后 Base64 编码，放入 `X-Signature` 请求头。

### 评测接口 `/api/v1/evaluation/`

| 接口 | 方法 | 说明 |
|------|------|------|
| `/analyze` | POST | 音频 URL 评测（异步回调） |
| `/analyze/upload` | POST | 音频文件上传评测（异步回调） |
| `/report` | POST | AI 报告生成（URL，同步） |
| `/report/upload` | POST | AI 报告生成（上传，同步） |
| `/text-analysis` | POST | 文本结构分析 |
| `/tongue-twister` | POST | 绕口令发音分析 |
| `/sentence-interpretation` | POST | 句子朗读解读分析 |
| `/story-reading` | POST | 故事朗读评价（ASR + AI） |
| `/tongue-twister-reading` | POST | 文章/绕口令朗读评价（ASR + SOE + AI） |
| `/voice-chat` | POST | 语音对话（传统/多模态 + 服务端会话管理） |
| `/voice-chat/text` | POST | 文本情景对话（无需语音 + 服务端会话管理） |
| `/voice-chat/scene` | POST | 会话级场景切换 |
| `/opinion-statement` | POST | 一分钟陈述评价（5 维度评分） |
| `/impromptu-reaction` | POST | 即兴反应评价（5 维度评分） |

### 口语评测接口 `/api/v1/soe/`

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传音频进行口语评测 |
| `/url` | POST | 通过音频 URL 进行口语评测 |
| `/audio/{filename}` | GET | 获取已处理的音频文件 |

### 语音合成接口 `/api/v1/tts/`

| 接口 | 方法 | 说明 |
|------|------|------|
| `/synthesize` | POST | 文本转语音并上传存储 |
| `/synthesize` | GET | 文本转语音（Query 参数） |
| `/voices` | GET | 获取可用音色列表 |

## 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| **腾讯云** | | |
| `TENCENT_SECRET_ID` | 腾讯云 SecretId | |
| `TENCENT_SECRET_KEY` | 腾讯云 SecretKey | |
| `TENCENT_APPID` | 腾讯云 AppId | |
| **AES 鉴权** | | |
| `AES_KEY` | AES 加密密钥（16/24/32 字节） | |
| `AES_IV` | AES 初始向量（16 字节） | |
| `REQUEST_EXPIRE_SECONDS` | 签名有效期（秒） | `300` |
| **服务配置** | | |
| `API_HOST` | 监听地址 | `0.0.0.0` |
| `API_PORT` | 监听端口 | `8000` |
| `DEBUG` | 调试模式 | `false` |
| **LLM 配置** | | |
| `LLM_PROVIDER` | LLM 提供商 (`openai` / `tencent` / `anthropic`) | `openai` |
| `LLM_TIMEOUT` | LLM 请求超时（秒） | `120` |
| `TENCENT_MODEL` | 腾讯混元模型 | `hunyuan-turbo` |
| `TENCENT_MULTIMODAL_MODEL` | 腾讯混元多模态模型 | `hunyuan-multimodal` |
| `OPENAI_MODEL` | OpenAI 模型 | `gpt-4o` |
| `OPENAI_MULTIMODAL_MODEL` | OpenAI 多模态模型 | `gpt-4o-audio-preview` |
| `OPENAI_API_KEY` | OpenAI API Key | |
| `OPENAI_BASE_URL` | OpenAI 兼容接口地址 | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | |
| `ANTHROPIC_MODEL` | Anthropic 模型 | `claude-sonnet-4-20250514` |
| `ANTHROPIC_MULTIMODAL_MODEL` | Anthropic 多模态模型 | `claude-sonnet-4-20250514` |
| **对象存储** | | |
| `S3_ENDPOINT` | OSS 操作域名 | |
| `S3_ACCESS_KEY` | AccessKey | |
| `S3_SECRET_KEY` | SecretKey | |
| `S3_BUCKET_NAME` | Bucket 名称 | |
| `S3_PREFIX` | 上传路径前缀 | |
| `S3_SECURE` | 使用 HTTPS | `false` |
| `S3_PUBLIC_URL` | OSS 公共访问域名 | |
| **上传模式** | | |
| `UPLOAD_MODE` | 上传模式 (`oss` / `api`) | `api` |
| `UPLOAD_API_URL` | POST 上传接口地址 | |
| **语音对话** | | |
| `CHAT_SESSION_TTL` | 对话会话过期时间（秒） | `3600` |

## Docker 部署

```bash
docker-compose up -d
```

## 评测维度

### 一分钟陈述（opinion-statement）
- 观点明确性、结构完整性、逻辑清晰度、语速节奏、表达简洁性

### 即兴反应（impromptu-reaction）
- 反应速度、结构形成、内容相关性、逻辑连贯性、表达简洁性

### 口语评测（SOE）
- 发音准确度、流利度、完整度，支持单词/句子/段落/自由说等多种模式

## 语音对话（Voice Chat）

支持两种对话模式，服务端管理会话历史，支持会话级场景切换。

### 对话模式

| 模式 | 说明 | 流程 |
|------|------|------|
| `traditional` | 传统模式（默认） | ASR转文字 → LLM对话 → TTS语音 |
| `multimodal` | 多模态模式 | 音频直接发给多模态模型 → TTS语音 |

### 预设场景

| scene | 说明 |
|-------|------|
| `interview` | 面试官角色，追问评价 |
| `daily` | 日常对话伙伴，口语化交流 |
| `customer_service` | 客服人员，处理咨询 |

### 多轮对话流程

1. **首轮**：不传 `session_id`，服务端自动创建会话并返回 `session_id`
2. **后续轮**：传入 `session_id`，服务端自动管理对话历史
3. **场景切换**：调用 `POST /voice-chat/scene` 接口，后续对话使用新场景

### 示例

```bash
# 首轮对话（面试场景）
curl -X POST /api/v1/evaluation/voice-chat \
  -H "X-Signature: {encrypted_signature}" \
  -F "audio_url=https://example.com/user-speech.mp3" \
  -F "scene=interview"

# 响应中获取 session_id，后续轮传入
curl -X POST /api/v1/evaluation/voice-chat \
  -H "X-Signature: {encrypted_signature}" \
  -F "audio_url=https://example.com/user-speech-2.mp3" \
  -F "session_id={session_id_from_response}"

# 文本情景对话（无需语音）
curl -X POST /api/v1/evaluation/voice-chat/text \
  -H "X-Signature: {encrypted_signature}" \
  -H "Content-Type: application/json" \
  -d '{"text":"我想练习一下面试自我介绍。","scene":"interview","enable_tts":false}'

# 切换场景
curl -X POST /api/v1/evaluation/voice-chat/scene \
  -H "X-Signature: {encrypted_signature}" \
  -F "session_id={session_id}" \
  -F "scene=daily"
```
