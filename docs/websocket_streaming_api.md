# WebSocket 流式评测 API 文档

## 概述

基于 WebSocket 的实时语音评测接口，支持边录音边识别，流式返回 ASR/SOE 中间结果，录音结束后异步执行多维度 AI 评测并逐步推送结果。

**端点地址：** `ws[s]://<host>/api/v1/streaming/ws/stream`

**音频格式：** 原始 16-bit 有符号 PCM，单声道，16kHz 采样率（32000 字节/秒）

---

## 通信流程

```
客户端                                            服务端
  │                                                 │
  │──── WebSocket 连接 ────────────────────────────>│
  │                                                 │
  │<─── { type: "session_started", session_id } ────│  ① 会话建立
  │                                                 │
  │──── { type: "config", data: {...} } ───────────>│  ② 发送配置
  │                                                 │
  │<─── { type: "asr_partial", data: {...} } ───────│  ③ 实时识别（重复）
  │<─── { type: "soe_intermediate", data: {...} } ──│  ③ 实时评分（重复）
  │                                                 │
  │════ [binary: PCM 音频帧] ══════════════════════>│  ④ 持续发送音频
  │════ [binary: PCM 音频帧] ══════════════════════>│
  │  ...                                            │
  │                                                 │
  │──── { type: "end" } ──────────────────────────>│  ⑤ 录音结束
  │                                                 │
  │<─── { type: "streaming_complete", data } ───────│  ⑥ 流式结果
  │                                                 │
  │<─── { type: "agent_result", data } ─────────────│  ⑦ 评测结果（逐个）
  │<─── { type: "agent_result", data } ─────────────│
  │  ...                                            │
  │                                                 │
  │<─── { type: "complete", data } ─────────────────│  ⑧ 最终结果
  │                                                 │
  │    或                                           │
  │                                                 │
  │<─── { type: "error", message } ─────────────────│  ⑧ 错误
```

---

## ① 客户端发送消息

### 1.1 Config — 配置消息

WebSocket 连接成功后**立即发送**，用于初始化评测会话。

```json
{
  "type": "config",
  "data": {
    "language": "zh",
    "ref_text": "床前明月光，疑是地上霜",
    "eval_mode": 3,
    "score_coeff": 2.0,
    "server_type": 0,
    "word_info": 1,
    "enable_asr": true,
    "enable_soe": true,
    "enable_timestamps": true,
    "eval_type": "tongue_twister_reading",
    "topic": "",
    "scenario": "",
    "reference_text": "",
    "progressive": true,
    "report_format": "markdown",
    "custom_prompt": ""
  }
}
```

#### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `language` | string | `"zh"` | 语言：`"zh"` 中文 / `"en"` 英文 |
| `ref_text` | string | `""` | SOE 评测参考文本（绕口令/朗读等需对比的场景） |
| `eval_mode` | int | `3` | SOE 评测模式：`0`=单词, `1`=句子, `2`=段落, `3`=自由说, `5`=情景评测 |
| `score_coeff` | float | `1.0` | SOE 评分苛刻系数：`1.0`=儿童(宽松) ~ `4.0`=成人(严格) |
| `server_type` | int | `0` | 引擎类型：`0`=中文, `1`=英文 |
| `word_info` | int | `1` | 词级信息：`0`=无, `1`=词, `2`=词+标点 |
| `enable_asr` | bool | `true` | 是否启用实时 ASR 语音识别 |
| `enable_soe` | bool | `true` | 是否启用实时 SOE 发音评测 |
| `enable_timestamps` | bool | `true` | 是否返回词级时间戳 |
| `eval_type` | string | `"none"` | 评测管线名称（见下方评测类型表） |
| `topic` | string | `""` | 陈述题目（用于 opinion_statement） |
| `scenario` | string | `""` | 场景/题目（用于 impromptu_reaction） |
| `reference_text` | string | `""` | 备用参考文本字段 |
| `progressive` | bool | `true` | `true`=渐进式逐个推送结果, `false`=等所有完成后一次性返回 |
| `report_format` | string | `"markdown"` | 报告输出格式 |
| `custom_prompt` | string | `""` | 自定义 LLM 提示词覆盖 |

#### 评测类型 (eval_type)

| eval_type | 说明 | 维度数 | 推荐 eval_mode |
|-----------|------|--------|----------------|
| `"none"` | 仅 ASR+SOE，不做 AI 评测 | 0 | 任意 |
| `"basic_evaluation"` | 基础评测 | 5 | 3 |
| `"extended_evaluation"` | 扩展评测 | 8 | 3 |
| `"opinion_statement"` | 一分钟观点陈述 | 6 | 3 |
| `"impromptu_reaction"` | 即兴反应 | 6 | 3 |
| `"story_reading"` | 小故事朗读 | 5 | 2/3 |
| `"tongue_twister_reading"` | 绕口令朗读 | 5 | 2/3 |
| `"article_reading"` | 文章朗读 | 6 | 2/3 |

### 1.2 Audio — 音频数据

录音期间以二进制帧持续发送。

- **格式：** 原始 16-bit 有符号 PCM (`Int16Array` → `ArrayBuffer`)
- **采样率：** 16,000 Hz
- **声道：** 单声道
- **帧大小：** 建议 200ms/帧（6400 字节），前端实际使用 4096 samples/帧

```javascript
// 前端发送示例
const pcm = floatTo16BitPCM(float32Array);
ws.send(pcm.buffer);
```

### 1.3 End — 结束录音

录音停止后发送，触发服务端执行 ASR/SOE 收尾和后续 AI 评测。

```json
{ "type": "end" }
```

---

## ② 服务端返回消息

### 2.1 session_started — 会话已建立

```json
{
  "type": "session_started",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 2.2 asr_partial — 实时识别结果

ASR 产生中间识别结果时推送（录音期间重复发送）。

```json
{
  "type": "asr_partial",
  "data": {
    "text": "床前明月光疑是"
  }
}
```

### 2.3 soe_intermediate — 实时评分结果

SOE 产生中间评分时推送（录音期间重复发送）。

```json
{
  "type": "soe_intermediate",
  "data": {
    "scores": {
      "pronunciation_accuracy": 85.2,
      "pronunciation_fluency": 78.5,
      "pronunciation_completion": 92.0,
      "suggested_score": 82.0
    }
  }
}
```

### 2.4 streaming_complete — 流式阶段完成

ASR 和 SOE 处理完毕后发送，包含所有流式阶段结果。此时 AI 评测尚未开始。

```json
{
  "type": "streaming_complete",
  "data": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "speech_text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "scores_data": {
      "pronunciation_accuracy": 88.5,
      "pronunciation_fluency": 82.3,
      "pronunciation_completion": 95.0,
      "suggested_score": 85.0,
      "overall_score": 85.0
    },
    "word_info_list": [
      {
        "word": "床",
        "reference_word": "床",
        "pron_accuracy": 92.5,
        "pron_fluency": 88.0,
        "begin_time": 0,
        "end_time": 320,
        "duration": 320
      }
    ],
    "low_score_words": [
      { "word": "疑", "accuracy": 65.2, "fluency": 58.0 }
    ],
    "statistics_data": {
      "total_words": 20,
      "average_accuracy": 88.5,
      "low_score_count": 2
    },
    "speech_rate": 156.0,
    "audio_duration": 8.2,
    "audio_url": "https://s3.example.com/audio/xxx.wav",
    "asr_result": { "...": "原始 ASR SDK 响应" },
    "soe_result": { "...": "原始 SOE SDK 响应" }
  }
}
```

#### scores_data 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `pronunciation_accuracy` | float | 发音准确度 (0-100) |
| `pronunciation_fluency` | float | 发音流利度 (0-100) |
| `pronunciation_completion` | float | 发音完整度 (0-100) |
| `suggested_score` | float | SOE 综合建议分 |
| `overall_score` | float | 与 suggested_score 相同 |

#### word_info_list 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `word` | string | 识别出的字词 |
| `reference_word` | string | 对应的参考文本字词 |
| `pron_accuracy` | float | 该字词发音准确度 |
| `pron_fluency` | float | 该字词发音流利度 |
| `begin_time` | int | 开始时间 (ms) |
| `end_time` | int | 结束时间 (ms) |
| `duration` | int | 持续时长 (ms) |

#### low_score_words 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `word` | string | 低分字词 |
| `accuracy` | float | 准确度 (0-100) |
| `fluency` | float | 流利度 (0-100) |

### 2.5 agent_result — 评测维度结果（渐进式）

当 `progressive=true` 时，每个维度 agent 完成后立即推送。

```json
{
  "type": "agent_result",
  "data": {
    "agent": "dim_op_viewpoint",
    "success": true,
    "duration_ms": 2350,
    "result": {
      "score": 82,
      "level": "良好",
      "analysis": "观点表达较为明确，开头直接亮明观点...",
      "suggestion": "建议在开头用更简洁的语言概括核心观点...",
      "details": {
        "has_clear_viewpoint": true,
        "viewpoint_summary": "认为环境保护需要每个人参与",
        "opening_type": "直接亮明观点",
        "opening_quote": "我认为环境保护是每个人的责任",
        "evasion_signals": [],
        "assessment": "观点鲜明，开门见山"
      }
    },
    "error": null
  }
}
```

#### data 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `agent` | string | agent 名称（如 `dim_op_viewpoint`, `dim_tw_pronunciation`） |
| `success` | bool | 是否成功 |
| `duration_ms` | float | 执行耗时（毫秒） |
| `result` | object | agent 输出数据（成功时） |
| `error` | string | 错误信息（失败时） |

#### 各维度 result 统一结构

所有维度 agent 输出统一的顶层结构：

```json
{
  "score": 82,
  "level": "良好",
  "analysis": "...",
  "suggestion": "...",
  "details": { "...": "各维度特有字段" }
}
```

### 2.6 complete — 最终结果

所有 agent 完成后发送（或 `progressive=false` 时一次性发送）。

```json
{
  "type": "complete",
  "data": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "speech_text": "床前明月光，疑是地上霜...",
    "scores_data": { "...": "同 streaming_complete" },
    "statistics_data": { "...": "同 streaming_complete" },
    "low_score_words": [ "...": "同 streaming_complete" ],
    "speech_rate": 156.0,
    "audio_url": "https://s3.example.com/audio/xxx.wav",
    "report": "# 评测报告\n\n## 综合评分\n...",
    "content_analysis": null,
    "fluency_analysis": null,
    "overall_score": {
      "score": 78,
      "level": "良好",
      "breakdown": {
        "op_viewpoint_score": 82,
        "op_structure_score": 70,
        "op_logic_score": 75,
        "op_expression_score": 80,
        "op_time_rhythm_score": 72,
        "fluency_score": 82.3,
        "speech_rate_score": 80,
        "pronunciation_accuracy": 88.5,
        "pronunciation_fluency": 82.3,
        "pronunciation_completion": 95.0,
        "suggested_score": 85.0,
        "speech_rate_value": 156.0
      }
    },
    "agent_results": {
      "dim_op_viewpoint": {
        "agent_name": "dim_op_viewpoint",
        "success": true,
        "data": { "score": 82, "level": "良好", "analysis": "...", "suggestion": "...", "details": {...} },
        "duration_ms": 2350,
        "error": null
      },
      "dim_op_structure": { "...": "同上结构" },
      "dim_speech_rate": { "...": "同上结构" }
    }
  }
}
```

#### agent_results 结构

`agent_results` 包含所有已执行维度 agent 的结果，key 为 agent 名称。每个值的结构与 `agent_result` 消息中的 `data` 字段一致。

| Pipeline | agent_results 中的 key |
|----------|----------------------|
| opinion_statement | `dim_op_viewpoint`, `dim_op_structure`, `dim_op_logic`, `dim_op_time_rhythm`, `dim_op_expression`, `dim_speech_rate` |
| impromptu_reaction | `dim_ir_reaction_speed`, `dim_ir_structure`, `dim_ir_content_relevance`, `dim_ir_logic`, `dim_ir_expression`, `dim_speech_rate` |
| story_reading | `dim_sr_structure`, `dim_sr_logic`, `dim_sr_fluency`, `dim_sr_event_distribution`, `dim_speech_rate` |
| tongue_twister_reading | `dim_tw_completeness`, `dim_tw_pronunciation`, `dim_tw_fluency`, `dim_tw_strengths`, `dim_speech_rate` |
| article_reading | `dim_ar_completeness`, `dim_ar_pronunciation`, `dim_ar_fluency`, `dim_ar_pause`, `dim_ar_strengths`, `dim_speech_rate` |

#### overall_score 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | int | 综合评分 (0-100) |
| `level` | string | 等级：优秀(85-100) / 良好(70-84) / 一般(55-69) / 需改进(0-54) |
| `breakdown` | object | 各维度单项分数 + SOE 原始分 |

#### 各类型综合评分权重

**一分钟观点陈述 (opinion_statement)**

| 维度 | 权重 |
|------|------|
| op_viewpoint (观点明确性) | 20% |
| op_logic (逻辑清晰度) | 20% |
| op_expression (表达精炼度) | 15% |
| fluency (SOE 发音流利度) | 15% |
| op_structure (结构完整度) | 10% |
| op_time_rhythm (时间节奏) | 10% |
| speech_rate (语速) | 10% |

**即兴反应 (impromptu_reaction)**

| 维度 | 权重 |
|------|------|
| ir_reaction_speed (反应速度) | 25% |
| ir_content_relevance (内容相关性) | 25% |
| ir_logic (逻辑连贯度) | 20% |
| fluency (SOE 发音流利度) | 15% |
| ir_expression (表达精炼度) | 10% |
| ir_structure (结构形成) | 5% |

**小故事 (story_reading)**

| 维度 | 权重 |
|------|------|
| sr_structure (结构分析) | 30% |
| sr_logic (逻辑分析) | 25% |
| sr_fluency (流畅度分析) | 25% |
| sr_event_distribution (事件分布) | 20% |

**绕口令 (tongue_twister_reading)**

| 维度 | 权重 |
|------|------|
| tw_pronunciation (发音) | 35% |
| tw_completeness (完整度) | 30% |
| tw_fluency (流畅度) | 25% |
| tw_strengths (优势) | 10% |

**文章朗读 (article_reading)**

| 维度 | 权重 |
|------|------|
| ar_pronunciation (发音) | 30% |
| ar_completeness (完整度) | 25% |
| ar_fluency (流畅度) | 25% |
| ar_pause (停顿) | 10% |
| ar_strengths (优势) | 10% |

### 2.7 error — 错误

```json
{
  "type": "error",
  "message": "No active session. Send config first."
}
```

| 错误信息 | 说明 |
|----------|------|
| `"No active session"` | end 消息在 config 之前发送 |
| `"No active session. Send config first."` | 音频帧在 config 之前发送 |
| `"Unknown message type: xxx"` | 未知的 JSON 消息类型 |
| `"Invalid JSON: xxx"` | JSON 解析失败 |

---

## 各评测类型维度详情

### opinion_statement — 一分钟观点陈述

| 维度 Agent | 说明 | details 关键字段 |
|-----------|------|-----------------|
| `dim_op_viewpoint` | 观点明确性 | has_clear_viewpoint, viewpoint_summary, opening_type, evasion_signals |
| `dim_op_structure` | 结构完整度 | has_viewpoint, has_reason, has_example, has_summary, missing_parts |
| `dim_op_logic` | 逻辑清晰度 | logic_jumps, contradictions, argument_piling, reasoning_chain |
| `dim_op_time_rhythm` | 时间节奏 | first_half_rate, second_half_rate, panic_acceleration, time_allocation |
| `dim_op_expression` | 表达精炼度 | filler_words, redundant_expressions, effective_content_ratio |

### impromptu_reaction — 即兴反应

| 维度 Agent | 说明 | details 关键字段 |
|-----------|------|-----------------|
| `dim_ir_reaction_speed` | 反应速度 | first_word_time_ms, opening_speed, panic_signals, thinking_pauses |
| `dim_ir_structure` | 结构形成 | formed_in_15s, structure_signal, structure_pattern |
| `dim_ir_content_relevance` | 内容相关性 | topic_relevance, is_mere_repetition, has_original_response |
| `dim_ir_logic` | 逻辑连贯度 | coherence_level, logic_jumps, transition_quality |
| `dim_ir_expression` | 表达精炼度 | filler_words, redundancy_level, effective_content_ratio |

### story_reading — 小故事

| 维度 Agent | 说明 | details 关键字段 |
|-----------|------|-----------------|
| `dim_sr_structure` | 结构分析 | opening, development, climax, ending |
| `dim_sr_logic` | 逻辑分析 | time_jumps, causal_errors, missing_events |
| `dim_sr_fluency` | 流畅度分析 | long_pauses, repetition_count, filler_words_count |
| `dim_sr_event_distribution` | 事件分布 | events[], transition_time |

### tongue_twister_reading — 绕口令

| 维度 Agent | 说明 | details 关键字段 |
|-----------|------|-----------------|
| `dim_tw_completeness` | 完整度 | extra_words, missed_words, accuracy_rate |
| `dim_tw_pronunciation` | 发音问题 | pronunciation_issues[], confusion_pairs[] |
| `dim_tw_fluency` | 流畅度 | rhythm_pattern, stress_pattern, long_pauses |
| `dim_tw_strengths` | 优势 | strengths[] |

### article_reading — 文章朗读

| 维度 Agent | 说明 | details 关键字段 |
|-----------|------|-----------------|
| `dim_ar_completeness` | 完整度 | extra_words, missed_words, wrong_words, accuracy_rate |
| `dim_ar_pronunciation` | 发音 | pronunciation_issues[] |
| `dim_ar_fluency` | 流畅度 | interruptions[], repeated_reads[], stutters |
| `dim_ar_pause` | 停顿 | proper_pauses, improper_pauses, missed_pauses |
| `dim_ar_strengths` | 优势 | strengths[] |

---

## 架构说明

### 执行流程

```
Level 0 (流式阶段)          Level 1 (并行)                    Level 2
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────┐
│ StreamingASR │     │ dim_speech_rate          │     │              │
│ StreamingSOE │────>│ dim_<维度1>              │────>│ overall_score│
│              │     │ dim_<维度2>              │     │ (纯计算)     │
│              │     │ dim_<维度3>              │     │              │
│              │     │ ...                      │     └──────────────┘
└──────────────┘     └─────────────────────────┘
```

- **Level 0：** 流式 ASR + SOE，录音期间实时运行，通过后台异步任务将中间结果推送给前端（`asr_partial` / `soe_intermediate`）
- **Level 1：** 所有维度 agent 并行运行，互不依赖，每个 agent 完成后通过 `on_agent_result` 回调即时推送
- **综合评分：** Level 1 完成后由 orchestrator 按权重公式纯计算，无额外 LLM 调用

### 渐进式 vs 一次性

| 模式 | progressive=true | progressive=false |
|------|-----------------|-------------------|
| 中间结果 | 每个 agent 完成后立即推送 `agent_result` | 不推送 |
| 最终结果 | 所有 agent 完成后推送 `complete` | 所有 agent 完成后推送 `complete` |
| 适用场景 | 前端需要实时展示进度 | 前端只需要最终结果 |

### 音频处理

- 音频以原始 PCM 二进制帧发送，无编码头、无帧分隔
- 服务端内部以 6400 字节（200ms）为单位分片
- SOE 发送有 200ms 最小间隔限流
- 录音结束后音频上传至 S3，返回 `audio_url`

---

## 当前版本补充说明

本节记录当前实现中新增或需要前端特别处理的 WebSocket 行为。

### 端点

| 端点 | 用途 |
|------|------|
| `ws[s]://<host>/api/v1/streaming/ws/stream?token=<signature>` | 实时录音评测：ASR/SOE + 可选 AI 多维评测 |
| `ws[s]://<host>/api/v1/streaming/ws/chat?token=<signature>` | 实时情景对话：ASR + LLM 流式回复 + 可选 TTS |

`token` 使用与 HTTP 接口相同的 AES 签名，放在 query string 中。缺失或过期会关闭连接。

### AI 排队状态 `ai_status`

当录音结束后进入 AI 阶段，服务端可能会发送 `ai_status`。前端建议展示为“排队中 / 分析中 / 重试中”，避免用户误以为卡住。

```json
{
  "type": "ai_status",
  "data": {
    "stage": "queued",
    "provider": "openai",
    "operation": "chat",
    "attempt": 1,
    "queue_position": 2,
    "queue_size": 3
  }
}
```

常见 `stage`：

| stage | 说明 |
|------|------|
| `preparing` | AI 后处理准备开始 |
| `queued` | 等待 LLM 并发槽位 |
| `running` | LLM 请求已开始 |
| `retrying` | 遇到限流、超时或 5xx，正在重试 |

相关环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MAX_CONCURRENT` | `0` | 全局并发覆盖值。`0` 表示使用 provider 默认值 |
| `LLM_TENCENT_MAX_CONCURRENT` | `5` | 腾讯/混元 provider 的并发上限 |
| `LLM_DEFAULT_MAX_CONCURRENT` | `50` | 非腾讯 provider 的并发上限 |
| `LLM_MAX_TOKENS` | `4000` | 单次 AI 回复的最大生成 token 数。`0` 表示使用 provider 默认值 |
| `LLM_QUEUE_MAX_SIZE` | `100` | 最多等待中的 LLM 请求数 |
| `LLM_QUEUE_TIMEOUT` | `60` | 等待槽位的最长秒数 |
| `LLM_MIN_INTERVAL_MS` | `500` | 两次 LLM 请求启动之间的最小间隔 |
| `LLM_MAX_RETRIES` | `3` | 429、超时、5xx 等可重试错误的重试次数 |

### 音频时长限制

WebSocket 录音音频按 16kHz / 16bit / 单声道 PCM 计算，当前默认最长 `600` 秒，也就是 10 分钟。

超限时服务端会发送：

```json
{
  "type": "error",
  "message": "Streaming audio exceeds max duration: 600s"
}
```

随后连接会以 code `4009` 关闭，reason 为 `Audio duration limit exceeded`。

可通过环境变量调整：

```env
STREAM_MAX_SESSION_DURATION=600
```

### `/ws/chat` 情景对话上下文

`/ws/chat` 支持服务端会话历史。首轮可以不传 `session_id`，服务端会在 `chat_done.data.chat_session_id` 中返回情景对话会话 ID。后续轮次需要把这个 ID 放回 config 的 `session_id` 字段，服务端会自动带上历史消息。

首轮 config：

```json
{
  "type": "config",
  "data": {
    "scene": "interview",
    "language": "zh",
    "enable_tts": true
  }
}
```

首轮完成：

```json
{
  "type": "chat_done",
  "data": {
    "session_id": "streaming-session-id",
    "chat_session_id": "chat-session-id",
    "user_text": "我的回答...",
    "assistant_text": "AI 回复...",
    "tts_url": "https://..."
  }
}
```

后续轮 config：

```json
{
  "type": "config",
  "data": {
    "session_id": "chat-session-id",
    "scene": "interview",
    "language": "zh",
    "enable_tts": true
  }
}
```

`/ws/chat` 会话由服务端内存管理，默认空闲 `CHAT_SESSION_TTL=3600` 秒后过期，并由后台任务定期清理。

### `/ws/chat` 服务端消息

录音期间：

```json
{
  "type": "asr_partial",
  "data": {
    "text": "实时识别文本"
  }
}
```

AI 回复期间：

```json
{
  "type": "llm_delta",
  "data": {
    "text": "增量文本"
  }
}
```

TTS 开启时：

```json
{
  "type": "tts_chunk",
  "data": {
    "audio": "base64-encoded-mp3-chunk"
  }
}
```

完成时：

```json
{
  "type": "chat_done",
  "data": {
    "session_id": "streaming-session-id",
    "chat_session_id": "chat-session-id",
    "user_text": "用户识别文本",
    "assistant_text": "AI 完整回复",
    "tts_url": "可选，合成音频上传后的 URL",
    "blood_bar": {
      "hp": 85,
      "delta": -15,
      "reason": "回答偏离主题",
      "game_over": false
    }
  }
}
```

`blood_bar` 仅在 config 中 `enable_blood_bar=true` 时可能返回。
