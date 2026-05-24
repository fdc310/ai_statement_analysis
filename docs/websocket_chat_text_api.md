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

```json
{
  "type": "llm_delta",
  "data": {
    "text": "增量回复文本"
  }
}
```

```json
{
  "type": "tts_chunk",
  "data": {
    "audio": "base64-encoded-mp3-chunk"
  }
}
```

```json
{
  "type": "chat_done",
  "data": {
    "session_id": "ws-input-session-id",
    "chat_session_id": "chat-context-session-id",
    "user_text": "用户输入文本",
    "assistant_text": "AI 完整回复",
    "tts_url": "https://..."
  }
}
```
