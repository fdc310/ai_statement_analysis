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
import uuid
from typing import Optional, AsyncGenerator

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


async def _handle_end(websocket: WebSocket, session: StreamingSession, config: StreamChatConfig):
    """Handle 'end' message: ASR → parallel LLM + persistent TTS → chat_done."""

    # ── Phase 1: finish ASR/SOE ──────────────────────────────────────────
    result = await session.finish()

    if not result.speech_text or not result.speech_text.strip():
        await websocket.send_json({
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
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in chat_sess.messages:
        llm_messages.append({"role": m.get("role", ""), "content": m.get("content", "")})
    llm_messages.append({"role": "user", "content": user_text})

    voice_type = config.voice_type or 101001

    # ── Phase 3: LLM stream + persistent TTS connection ─────────────────
    llm = get_llm_service()
    tts_text_queue: queue.Queue = queue.Queue()
    tts_audio_queue: asyncio.Queue = asyncio.Queue()
    tts_collected_chunks: list[bytes] = []  # collect raw audio for final upload
    tts_error_holder: list[str] = []  # capture TTS error for caller
    llm_full_text: list[str] = []
    tts_stop_event = threading.Event()

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
            tts_audio_queue.put_nowait(None)
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

        # Drain remaining audio from listener queue
        while True:
            try:
                chunk = listener._audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                tts_collected_chunks.append(chunk)  # save raw bytes for upload
                tts_audio_queue.put_nowait({"audio": base64.b64encode(chunk).decode("utf-8")})
            except queue.Empty:
                break

        if listener._error:
            logger.error(f"TTS error: {listener._error}")
            tts_error_holder.append(listener._error)

        logger.info("TTS persistent connection closed")
        tts_audio_queue.put_nowait(None)

    async def run_llm_stream():
        """Stream LLM response, send llm_delta to client, feed TTS on sentence boundaries."""
        sentence_buffer = ""
        try:
            async for delta in llm.chat_stream(llm_messages, temperature=0.7):
                llm_full_text.append(delta)
                sentence_buffer += delta

                await websocket.send_json({
                    "type": "llm_delta",
                    "data": {"text": delta},
                })

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
                await websocket.send_json({"type": "tts_chunk", "data": item})
            except Exception as e:
                logger.error(f"send_tts_chunks error: {e}")

    tts_thread = threading.Thread(target=_tts_persistent_worker, daemon=True)
    tts_thread.start()

    tts_sender = asyncio.create_task(send_tts_chunks())
    llm_task = asyncio.create_task(run_llm_stream())

    try:
        await llm_task
        await tts_sender
    except asyncio.CancelledError:
        # Client disconnected — signal TTS thread to stop immediately
        tts_stop_event.set()
        # Drain the text queue so TTS thread doesn't block on get()
        while not tts_text_queue.empty():
            try:
                tts_text_queue.get_nowait()
            except queue.Empty:
                break
        tts_text_queue.put_nowait(None)
        raise

    assistant_text = "".join(llm_full_text)

    # ── Phase 4: update session, upload streamed audio ───────────────────
    await chat_session_manager.append_message(chat_sess.session_id, "user", user_text)
    await chat_session_manager.append_message(chat_sess.session_id, "assistant", assistant_text)

    tts_url = None
    # Upload the audio collected during streaming (no second synthesis)
    if tts_collected_chunks and not tts_error_holder:
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

    await websocket.send_json({
        "type": "chat_done",
        "data": {
            "session_id": result.session_id,
            "chat_session_id": chat_sess.session_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "tts_url": tts_url,
        },
    })
    logger.info(f"WS chat completed: {session.session_id}")
