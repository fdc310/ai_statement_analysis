"""
WebSocket streaming chat endpoint — LLM stream + persistent TTS connection.

Flow:
  1. Client connects, sends config (scene/system_prompt/voice_type)
  2. Client streams PCM audio frames
  3. Client sends {"type":"end"}
  4. Server returns asr_partial in real time during recording
  5. After end: server runs LLM stream + TTS in parallel
     - LLM chunks → client (llm_delta)
     - Text chunks → single TTS WebSocket → audio → client (tts_chunk)
  6. Server sends chat_done when both finish
"""
import asyncio
import json
import logging
import re
import base64
import queue
import threading
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.services.streaming.session_manager import StreamingSession
from app.schemas.streaming_chat import StreamChatConfig
from app.core.config import settings
from app.core.security import aes_service
from app.core.sdk_path import SDK_PATH  # noqa: F401 — ensures SDK is on sys.path

logger = logging.getLogger(__name__)
router = APIRouter()

_SENTENCE_RE = re.compile(r'(?<=[。！？\n])')
_PAUSE_RE = re.compile(r'(?<=[，、；])')

from common.credential import Credential
from tts.flowing_speech_synthesizer import FlowingSpeechSynthesizer, FlowingSpeechSynthesisListener


class _ChatTTSListener(FlowingSpeechSynthesisListener):
    """Listener that queues audio chunks and signals completion."""

    def __init__(self):
        self._audio_queue: queue.Queue = queue.Queue()
        self._done = False
        self._error = None
        self._lock = threading.Lock()

    def on_synthesis_start(self, session_id):
        pass

    def on_synthesis_end(self):
        with self._lock:
            self._done = True
        self._audio_queue.put(None)  # sentinel

    def on_audio_result(self, audio_bytes):
        self._audio_queue.put(audio_bytes)

    def on_text_result(self, response):
        pass

    def on_synthesis_fail(self, response):
        with self._lock:
            self._error = f"TTS failed: code={response.get('code')}, msg={response.get('message')}"
            self._done = True
        self._audio_queue.put(None)


@router.websocket("/ws/chat")
async def websocket_streaming_chat(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for streaming voice chat.

    Protocol:
    1. Client connects with ?token=<AES signature>
    2. Client sends config: {"type":"config","data":{...}}
    3. Client sends audio: binary PCM frames
    4. Client sends end: {"type":"end"}
    5. Server streams results:
       - {"type":"asr_partial","data":{"text":"..."}}  (during recording)
       - {"type":"llm_delta","data":{"text":"..."}}     (LLM stream)
       - {"type":"tts_chunk","data":{"audio":"base64"}} (TTS stream)
       - {"type":"chat_done","data":{...}}              (finished)
       - {"type":"error","message":"..."}
    """
    await websocket.accept()

    # Authenticate via AES signature in query param
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    success, message = aes_service.verify_signature(token, max_age_seconds=settings.request_expire_seconds)
    if not success:
        await websocket.close(code=4003, reason="Unauthorized")
        logger.warning(f"WS chat auth failed: {message}")
        return

    logger.info("WS chat client connected")

    session: Optional[StreamingSession] = None
    config: Optional[StreamChatConfig] = None

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.receive":
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")

                        if msg_type == "config":
                            config_data = data.get("data", {})
                            config = StreamChatConfig(**config_data)

                            async def _make_on_asr_partial(ws: WebSocket):
                                async def on_asr_partial(event):
                                    try:
                                        await ws.send_json({
                                            "type": "asr_partial",
                                            "data": event.get("data", {}),
                                        })
                                    except Exception:
                                        pass
                                return on_asr_partial

                            on_asr_partial = await _make_on_asr_partial(websocket)
                            session = StreamingSession(
                                config=config,
                                on_asr_partial=on_asr_partial,
                                on_soe_intermediate=None,
                            )
                            await session.start()

                            await websocket.send_json({
                                "type": "session_started",
                                "session_id": session.session_id,
                            })
                            logger.info(f"WS chat session started: {session.session_id}")

                        elif msg_type == "end":
                            if not session:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "No active session",
                                })
                                continue

                            await _handle_end(websocket, session, config)

                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown message type: {msg_type}",
                            })

                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format",
                        })

                elif "bytes" in message:
                    if session:
                        await session.feed_audio(message["bytes"])
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No active session. Send config first.",
                        })

    except WebSocketDisconnect:
        logger.info("WS chat client disconnected")
    except Exception as e:
        logger.error(f"WS chat error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        if session and not session._finished:
            try:
                await session.finish()
            except Exception as e:
                logger.error(f"WS chat session cleanup error: {e}")


_BLOOD_BAR_SCENE_CRITERIA = {
    "interview": "面试场景：回答是否专业、有条理、切题，是否展现了良好的表达能力和逻辑思维",
    "daily": "日常对话场景：回答是否自然流畅、有互动感，是否像正常的日常聊天",
    "customer_service": "客服场景：回答是否礼貌、有耐心，是否有效回应了客服的需求和问题",
}

_BLOOD_BAR_SYSTEM_PROMPT = """你是一位情景对话裁判。你的任务是判断用户在当前情景对话中的回答质量。

评判标准：{criteria}

请根据用户的回答内容，判断其在当前情景下的表现，给出一个血量变化值（delta）。
- 回答切题、质量好：加血 +5 到 +20
- 回答一般、勉强合格：不加不减 0
- 回答偏离主题、质量差：扣血 -5 到 -15
- 回答完全跑题或无意义：扣血 -15 到 -30

你必须严格返回以下JSON格式，不要返回任何其他内容：
{{"delta": <整数>, "reason": "<简短中文说明，15字以内>"}}"""


async def _evaluate_blood_bar(
    llm_service,
    scene: str,
    user_text: str,
    assistant_text: str,
    current_hp: int,
    status_callback=None,
) -> Optional[dict]:
    """Use LLM to evaluate user's answer for blood bar mechanism."""
    criteria = _BLOOD_BAR_SCENE_CRITERIA.get(
        scene, "通用对话场景：回答是否合理、有逻辑、切题"
    )
    system_prompt = _BLOOD_BAR_SYSTEM_PROMPT.format(criteria=criteria)

    user_prompt = (
        f"当前情景：{scene or '通用对话'}\n"
        f"当前血量：{current_hp}/100\n"
        f"用户的回答：{user_text}\n"
        f"AI的回复：{assistant_text}"
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = await llm_service.chat(
            messages,
            temperature=0.3,
            status_callback=status_callback,
        )
        content = result.get("content", "").strip()

        # Extract JSON from response (handle possible markdown code blocks)
        if "```" in content:
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                content = json_match.group()

        parsed = json.loads(content)
        delta = int(parsed.get("delta", 0))
        # Clamp delta to reasonable range
        delta = max(-30, min(20, delta))
        reason = str(parsed.get("reason", ""))[:50]

        new_hp = max(0, current_hp + delta)
        return {
            "hp": new_hp,
            "delta": delta,
            "reason": reason,
            "game_over": new_hp <= 0,
        }
    except Exception as e:
        logger.error(f"Blood bar evaluation error: {e}")
        return None


async def _handle_end(websocket: WebSocket, session: StreamingSession, config: StreamChatConfig):
    """Handle 'end' message: ASR -> parallel LLM + persistent TTS -> chat_done."""
    ws_send_lock = asyncio.Lock()

    async def send_ws_json(payload: dict):
        async with ws_send_lock:
            await websocket.send_json(payload)

    async def send_ai_status(data: dict):
        await send_ws_json({
            "type": "ai_status",
            "data": data,
        })

    """Handle 'end' message: ASR → parallel LLM + persistent TTS → chat_done."""

    # ── Phase 1: finish ASR/SOE ──────────────────────────────────────────
    result = await session.finish()

    if not result.speech_text or not result.speech_text.strip():
        await send_ws_json({
            "type": "chat_done",
            "data": {
                "session_id": result.session_id,
                "user_text": "",
                "assistant_text": "",
                "tts_url": None,
            },
        })
        return

    user_text = result.speech_text

    # ── Phase 2: build chat context ──────────────────────────────────────
    from app.services.chat.session_manager import (
        chat_session_manager,
        VOICE_CHAT_SCENE_PROMPTS,
        DEFAULT_VOICE_CHAT_PROMPT,
    )
    from app.services import get_llm_service

    scene = config.scene or ""
    system_prompt = config.system_prompt or ""
    if not system_prompt:
        system_prompt = VOICE_CHAT_SCENE_PROMPTS.get(scene, DEFAULT_VOICE_CHAT_PROMPT)

    chat_sess = await chat_session_manager.get_or_create_session(
        session_id=None,
        scene=scene or None,
        system_prompt=system_prompt or None,
        enable_blood_bar=config.enable_blood_bar,
        initial_hp=config.initial_hp,
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in chat_sess.messages:
        llm_messages.append({"role": m.get("role", ""), "content": m.get("content", "")})
    llm_messages.append({"role": "user", "content": user_text})

    voice_type = config.voice_type or 101001
    enable_tts = config.enable_tts

    # ── Phase 3: LLM stream + optional TTS ──────────────────────────────
    llm = get_llm_service()
    await send_ai_status({
        "stage": "preparing",
        "message": "AI chat response is starting",
    })
    tts_text_queue: queue.Queue = queue.Queue()
    tts_audio_queue: asyncio.Queue = asyncio.Queue()
    tts_collected_chunks: list[bytes] = []
    tts_error_holder: list[str] = []
    llm_full_text: list[str] = []
    tts_stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def put_tts_audio(item):
        loop.call_soon_threadsafe(tts_audio_queue.put_nowait, item)

    def _tts_persistent_worker():
        """Daemon thread: ONE TTS WebSocket connection, reads text → produces audio."""
        import time as _time

        listener = _ChatTTSListener()
        synthesizer = FlowingSpeechSynthesizer(
            int(settings.tencent_appid),
            Credential(settings.tencent_secret_id, settings.tencent_secret_key),
            listener,
        )
        synthesizer.set_voice_type(voice_type)
        synthesizer.set_codec("mp3")
        synthesizer.set_sample_rate(16000)
        synthesizer.set_speed(1.0)
        synthesizer.set_volume(0.0)

        synthesizer.start()
        if not synthesizer.wait_ready(10000):
            logger.error("TTS persistent connection not ready within 10s")
            put_tts_audio(None)
            return

        logger.info("TTS persistent connection ready")

        last_feed_time = _time.time()

        while not tts_stop_event.is_set():
            try:
                text = tts_text_queue.get(timeout=0.05)
            except queue.Empty:
                now = _time.time()
                if now - last_feed_time > 20:
                    try:
                        synthesizer.process("")
                    except Exception:
                        pass
                    last_feed_time = now
                continue

            if text is None:
                break

            synthesizer.process(text)
            last_feed_time = _time.time()
            _time.sleep(0.05)

        try:
            synthesizer.complete()
        except Exception as e:
            logger.error(f"TTS complete() failed: {e}")

        synthesizer.wait()

        while True:
            try:
                chunk = listener._audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                tts_collected_chunks.append(chunk)
                put_tts_audio({"audio": base64.b64encode(chunk).decode("utf-8")})
            except queue.Empty:
                break

        if listener._error:
            logger.error(f"TTS error: {listener._error}")
            tts_error_holder.append(listener._error)

        logger.info("TTS persistent connection closed")
        put_tts_audio(None)

    async def run_llm_stream():
        """Stream LLM response, send llm_delta to client, feed TTS on sentence boundaries."""
        sentence_buffer = ""
        try:
            async for delta in llm.chat_stream(
                llm_messages,
                temperature=0.7,
                status_callback=send_ai_status,
            ):
                llm_full_text.append(delta)
                sentence_buffer += delta

                await send_ws_json({
                    "type": "llm_delta",
                    "data": {"text": delta},
                })

                if enable_tts:
                    parts = _SENTENCE_RE.split(sentence_buffer)
                    if len(parts) > 1:
                        to_send = "".join(parts[:-1])
                        sentence_buffer = parts[-1] or ""
                        if to_send.strip():
                            tts_text_queue.put_nowait(to_send)

                    elif len(sentence_buffer) > 80:
                        mid_parts = _PAUSE_RE.split(sentence_buffer)
                        if len(mid_parts) > 1:
                            to_send = "".join(mid_parts[:-1])
                            sentence_buffer = mid_parts[-1] or ""
                            if to_send.strip():
                                tts_text_queue.put_nowait(to_send)

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
        finally:
            if enable_tts:
                if sentence_buffer.strip():
                    tts_text_queue.put_nowait(sentence_buffer)
                tts_text_queue.put_nowait(None)

    async def send_tts_chunks():
        """Read from tts_audio_queue and send to client until sentinel."""
        while True:
            item = await tts_audio_queue.get()
            if item is None:
                break
            try:
                await send_ws_json({"type": "tts_chunk", "data": item})
            except Exception as e:
                logger.error(f"send_tts_chunks error: {e}")

    # Start TTS thread only if enabled
    tts_thread = None
    tts_sender = None
    if enable_tts:
        tts_thread = threading.Thread(target=_tts_persistent_worker, daemon=True)
        tts_thread.start()
        tts_sender = asyncio.create_task(send_tts_chunks())

    llm_task = asyncio.create_task(run_llm_stream())

    try:
        await llm_task
        if tts_sender is not None:
            await tts_sender
    except asyncio.CancelledError:
        if enable_tts:
            tts_stop_event.set()
            while not tts_text_queue.empty():
                try:
                    tts_text_queue.get_nowait()
                except queue.Empty:
                    break
            tts_text_queue.put_nowait(None)
        raise

    assistant_text = "".join(llm_full_text)

    # ── Phase 4: blood bar evaluation ────────────────────────────────────
    blood_bar_data = None
    if config.enable_blood_bar and chat_sess.enable_blood_bar:
        blood_bar_data = await _evaluate_blood_bar(
            llm, scene, user_text, assistant_text, chat_sess.hp,
            status_callback=send_ai_status,
        )
        if blood_bar_data:
            await chat_session_manager.update_hp(
                chat_sess.session_id,
                blood_bar_data["delta"],
                blood_bar_data.get("reason", ""),
            )

    # ── Phase 5: update session, upload audio ────────────────────────────
    await chat_session_manager.append_message(chat_sess.session_id, "user", user_text)
    await chat_session_manager.append_message(chat_sess.session_id, "assistant", assistant_text)

    tts_url = None
    if enable_tts and tts_collected_chunks and not tts_error_holder:
        try:
            from app.services.s3_storage import s3_storage
            full_audio = b"".join(tts_collected_chunks)
            upload_result = s3_storage.upload_tts_audio(
                audio_data=full_audio,
                codec="mp3",
                text=assistant_text,
                subfolder="tts",
            )
            if upload_result.get("success"):
                tts_url = upload_result.get("url")
        except Exception as e:
            logger.error(f"TTS upload error: {e}")

    done_data = {
        "session_id": result.session_id,
        "chat_session_id": chat_sess.session_id,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "tts_url": tts_url,
    }
    if blood_bar_data:
        done_data["blood_bar"] = blood_bar_data

    await send_ws_json({"type": "chat_done", "data": done_data})
    logger.info(f"WS chat completed: {session.session_id}")
