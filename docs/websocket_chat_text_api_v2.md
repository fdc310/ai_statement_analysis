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
    "sub_type": "campus",
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

### Scene & Sub-types

Each main scene has multiple sub-types (扮演角色). The sub-type determines the AI's role and evaluation criteria.

| Scene | Sub-type | AI Role |
|-------|----------|---------|
| `interview` | `campus` | 校园招聘面试官 |
| `interview` | `social` | 社会招聘面试官 |
| `interview` | `civil` | 公考结构化面试考官 |
| `office_work` | `report` | 部门领导（工作汇报） |
| `office_work` | `promotion` | 部门领导（升职加薪） |
| `office_work` | `resignation` | 部门领导（离职跳槽） |
| `business_social` | `sales` | 潜在客户 |
| `business_social` | `deal` | 合作方负责人 |
| `business_social` | `networking` | 行业同行 |
| `custom` | — | 专业对话伙伴 |

If `sub_type` is empty, the main scene's default prompt is used.

### System Prompt Priority

1. `config.system_prompt` (frontend-provided, highest priority)
2. `scene:sub_type` preset (e.g., `interview:civil`)
3. `scene` preset (e.g., `interview`)
4. Default generic prompt

The frontend typically fills `system_prompt` automatically when the user selects a scene/sub-type, and the user can freely modify it.

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
      "delta": -5,
      "reason": "说话啰嗦，重点不明确",
      "game_over": false,
      "fatal": false
    },
    "report": {
      "summary": "回答逻辑混乱，多次偏离主题",
      "detail": { "...see Report Format below..." },
      "duration": { "total": 8.5, "summary": 2.3, "report": 6.2 }
    }
  }
}
```

- `blood_bar` is present only when `enable_blood_bar=true`.
- `report` is present only when `game_over=true` (HP reached 0).
- `duration` shows the time spent generating the report (in seconds).

## Blood Bar Rules

When `enable_blood_bar=true`, the AI evaluates each user turn and adjusts HP.

### Damage Tiers

| Tier | Delta | Category | Examples |
|------|-------|----------|----------|
| 轻微 | **-5** | 表达能力问题 | 说话啰嗦、重点不明、口头禅多、回答过短 |
| 中度 | **-10** | 沟通效果变差 | 答非所问、逻辑矛盾、情绪化、频繁冷场 |
| 重度 | **-20** | 现实翻车 | 没结论、暴露致命问题、顶撞对方、回避核心 |
| 致命 | **-999** | 直接结束 | 侮辱对方、关键数据没看、"不买就算了" |

### Bonus Tiers

| Tier | Delta | Examples |
|------|-------|----------|
| 优秀 | **+5~+15** | 切题有条理、主动挖掘需求、话术得体 |
| 极佳 | **+20** | 超出预期的精彩回答、化解刁难、高情商 |

### Fatal

When `blood_bar.fatal=true`, the dialogue ends immediately regardless of remaining HP. The server:
1. Sets HP to 0
2. Generates the evaluation report
3. Includes `report` in the `chat_done` message

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
    "report": { "...see Report Format below..." },
    "duration": { "total": 8.5, "summary": 2.3, "report": 6.2 }
  }
}
```

## Natural End (Game Over)

When `enable_blood_bar=true` and HP reaches 0 (either through accumulated damage or a fatal hit), the server automatically generates a report. The report is included in the `chat_done` message of the turn that caused HP to reach 0.

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

## Report Duration

The `report.duration` object shows time spent generating the report:

```json
{
  "total": 8.5,
  "summary": 2.3,
  "report": 6.2
}
```

- `total`: Total time in seconds
- `summary`: Time to generate the 20-30 char summary
- `report`: Time to generate the full evaluation report
