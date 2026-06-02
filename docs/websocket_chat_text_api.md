# WebSocket Text Scene Chat

The existing `/api/v1/streaming/ws/chat` WebSocket endpoint supports text-only input as well as voice input.

## URL

```text
ws[s]://<host>/api/v1/streaming/ws/chat?token=<signature>
```

`token` is the same AES signature used by the other WebSocket endpoints.

## Flow

1. Connect to `/ws/chat`.
2. Send `config` with `enable_asr=false` and `enable_soe=false`.
3. Send a `text` message.
4. Receive `llm_delta`, optional `tts_chunk`, and `chat_done`.
5. Repeat step 3-4 for multi-turn dialogue.
6. Dialogue ends either naturally (HP reaches 0) or manually (client sends `end_dialogue`).

## Config

```json
{
  "type": "config",
  "data": {
    "scene": "interview",
    "language": "zh",
    "system_prompt": "",
    "voice_type": 101001,
    "enable_asr": false,
    "enable_soe": false,
    "enable_tts": true,
    "enable_blood_bar": false,
    "initial_hp": 100
  }
}
```

### Scene Types

| Scene | Name | Description |
|-------|------|-------------|
| `interview` | 求职面试 | 应届求职、社会招聘、考公考编 |
| `office_work` | 职场办公 | 工作汇报、升职加薪、离职跳槽 |
| `business_social` | 商务社交 | 销售沟通、商务洽谈、商务社交 |
| `custom` | 自定义 | 通用评估（说服力/共情力/应变力） |
| `daily` | 日常对话 | 轻松自然的日常聊天 |
| `customer_service` | 客服场景 | 客户咨询与问题处理 |

For multi-turn context, pass the previous `chat_done.data.chat_session_id` back as `data.session_id` in the next config. When multiple `text` messages are sent on the same WebSocket connection, the server also keeps using the latest chat session automatically.

## Text Message

```json
{
  "type": "text",
  "data": {
    "text": "我想练习一下面试自我介绍。"
  }
}
```

## Response Messages

### LLM streaming delta

```json
{
  "type": "llm_delta",
  "data": {
    "text": "增量回复文本"
  }
}
```

### TTS audio chunk

```json
{
  "type": "tts_chunk",
  "data": {
    "audio": "base64-encoded-mp3-chunk"
  }
}
```

### Chat done (per turn)

```json
{
  "type": "chat_done",
  "data": {
    "session_id": "ws-input-session-id",
    "chat_session_id": "chat-context-session-id",
    "user_text": "用户输入文本",
    "assistant_text": "AI 完整回复",
    "tts_url": "https://...",
    "blood_bar": {
      "hp": 85,
      "delta": -15,
      "reason": "回答偏离主题",
      "game_over": false
    },
    "report": {
      "summary": "回答逻辑混乱，多次偏离主题",
      "detail": { "...see Report Format below..." }
    }
  }
}
```

`blood_bar` is present only when `enable_blood_bar=true`.
`report` is present only when `game_over=true` (HP reached 0).

## End Dialogue (Manual)

The client can manually end the dialogue at any time. The server generates a full evaluation report.

### Client sends

```json
{
  "type": "end_dialogue"
}
```

### Server responds

```json
{
  "type": "dialogue_ended",
  "data": {
    "chat_session_id": "chat-context-session-id",
    "summary": "整体表达较流畅，岗位匹配度高，建议加强反问环节",
    "report": { "...see Report Format below..." }
  }
}
```

## Natural End (Game Over)

When `enable_blood_bar=true` and HP reaches 0, the server automatically generates a report. The report is included in the `chat_done` message of the turn that caused HP to reach 0.

The `blood_bar.game_over` field will be `true`, and the `report` field will contain the evaluation.

## Report Format

The `report.detail` object contains the full evaluation:

```json
{
  "scene": "求职面试",
  "overall_score": 7,
  "summary": "整体表现中等，表达有条理但内容深度不足",
  "dimensions": [
    {
      "name": "对话亮点",
      "score": 7,
      "comment": "发言有条理，能分点陈述..."
    },
    {
      "name": "岗位匹配度",
      "score": 6,
      "comment": "自我介绍偏泛化，未突出核心优势..."
    }
  ],
  "highlights": [
    "开场问候得体，展现了良好的礼仪素养",
    "回答优缺点时思路清晰"
  ],
  "improvements": [
    "自我介绍应突出与岗位相关的核心经历",
    "回答薪资期望时应给出合理区间而非模糊表述"
  ],
  "better_examples": [
    "面试官问期望薪资时，可以说：基于我的经验和市场调研，期望薪资在X-Y区间..."
  ]
}
```

The evaluation dimensions vary by scene type. See `app/services/agents/prompts/scenario_report.py` for the full dimension definitions per scene.
```
