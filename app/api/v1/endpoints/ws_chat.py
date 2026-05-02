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
import sys
import os
import queue
import threading
import uuid
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.services.streaming.session_manager import StreamingSession
from app.schemas.streaming_chat import StreamChatConfig
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

_SENTENCE_RE = re.compile(r'(?<=[。！？\n])')
_PAUSE_RE = re.compile(r'(?<=[，、；])')

# Import TTS SDK
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "core", "util", "tencentcloud-speech-sdk-python")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from common.credential import Credential
from tts.flowing_speech_synthesizer import FlowingSpeechSynthesizer, FlowingSpeechSynthesisListener


class _ChatTTSListener(FlowingSpeechSynthesisListener):
    """Listener that queues audio chunks and signals completion."""

    def __init__(self):
        self._audio_queue: queue.Queue = queue.Queue()
        self._done = False
        self._error = None

    def on_synthesis_start(self, session_id):
        pass

    def on_synthesis_end(self):
        self._done = True
        self._audio_queue.put(None)  # sentinel

    def on_audio_result(self, audio_bytes):
        self._audio_queue.put(audio_bytes)

    def on_text_result(self, response):
        pass

    def on_synthesis_fail(self, response):
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
    1. Client connects
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
    logger.info("WS chat client connected")

    session: Optional[StreamingSession] = None

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

                            async def on_asr_partial(event):
                                try:
                                    await websocket.send_json({
                                        "type": "asr_partial",
                                        "data": event.get("data", {}),
                                    })
                                except Exception:
                                    pass

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

                    except json.JSONDecodeError as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Invalid JSON: {e}",
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
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if session and not session._finished:
            try:
                await session.finish()
            except Exception:
                pass


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
    from app.services.tencent import tts_service

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
    # ONE TTS WebSocket connection. LLM feeds text in, audio comes out.
    # We keep the connection alive by NOT calling complete() until the LLM
    # is done. Periodic empty process() calls serve as heartbeats.
    # NOTE: tts_text_queue must be queue.Queue (not asyncio.Queue) because
    #       the TTS worker runs in a daemon thread.
    llm = get_llm_service()
    tts_text_queue: queue.Queue = queue.Queue()
    tts_audio_queue: asyncio.Queue = asyncio.Queue()
    llm_full_text: list[str] = []

    def _tts_persistent_worker():
        """
        Runs in a daemon thread. Creates ONE TTS WebSocket connection,
        reads text from tts_text_queue, feeds to TTS via process(),
        collects audio from listener, puts into tts_audio_queue.
        Only calls complete() when LLM is done (None signal received).
        """
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

        # Feed text chunks into the single connection
        while True:
            try:
                text = tts_text_queue.get(timeout=0.05)
            except queue.Empty:
                # Queue empty — send heartbeat to keep connection alive
                now = _time.time()
                if now - last_feed_time > 20:
                    try:
                        synthesizer.process("")
                    except Exception:
                        pass
                    last_feed_time = now
                continue

            if text is None:
                # End signal from LLM — flush and close
                break

            # Split text at sentence boundaries for natural pauses
            parts = _SENTENCE_RE.split(text)
            if len(parts) > 1:
                for part in parts:
                    if part.strip():
                        synthesizer.process(part)
            else:
                synthesizer.process(text)
            last_feed_time = _time.time()

        # Tell TTS server we're done — it will send FINAL frame
        try:
            synthesizer.complete()
        except Exception as e:
            logger.error(f"TTS complete() failed: {e}")

        # Wait for synthesis to finish
        synthesizer.wait()

        # Drain remaining audio from listener queue
        while True:
            try:
                chunk = listener._audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                tts_audio_queue.put_nowait({"audio": base64.b64encode(chunk).decode("utf-8")})
            except queue.Empty:
                break

        if listener._error:
            logger.error(f"TTS error: {listener._error}")

        logger.info("TTS persistent connection closed")
        tts_audio_queue.put_nowait(None)  # sentinel for sender

    async def run_llm_stream():
        """Stream LLM response, send llm_delta to client, feed TTS on sentence boundaries."""
        sentence_buffer = ""
        try:
            async for delta in llm.chat_stream(llm_messages, temperature=0.7):
                llm_full_text.append(delta)
                sentence_buffer += delta

                # Send llm_delta to client immediately
                await websocket.send_json({
                    "type": "llm_delta",
                    "data": {"text": delta},
                })

                # Split at sentence endings
                parts = _SENTENCE_RE.split(sentence_buffer)
                if len(parts) > 1:
                    to_send = "".join(parts[:-1])
                    sentence_buffer = parts[-1] or ""
                    if to_send.strip():
                        tts_text_queue.put_nowait(to_send)

                # Split at mid-sentence pauses if buffer is large
                elif len(sentence_buffer) > 80:
                    mid_parts = _PAUSE_RE.split(sentence_buffer)
                    if len(mid_parts) > 1:
                        to_send = "".join(mid_parts[:-1])
                        sentence_buffer = mid_parts[-1] or ""
                        if to_send.strip():
                            tts_text_queue.put_nowait(to_send)

        except Exception as e:
            logger.error(f"LLM stream error: {e}")

        # Flush remaining text
        if sentence_buffer.strip():
            tts_text_queue.put_nowait(sentence_buffer)

        # Signal TTS worker that LLM is done
        tts_text_queue.put_nowait(None)

    async def send_tts_chunks():
        """Read from tts_audio_queue and send to client until sentinel."""
        while True:
            item = await tts_audio_queue.get()
            if item is None:
                break
            try:
                await websocket.send_json({"type": "tts_chunk", "data": item})
            except Exception:
                pass

    # Start all three concurrently:
    # - tts_thread: persistent TTS connection, reads text → produces audio
    # - llm_task: streams LLM → sends llm_delta + feeds tts_text_queue
    # - tts_sender: reads tts_audio_queue → sends tts_chunk to client
    tts_thread = threading.Thread(target=_tts_persistent_worker, daemon=True)
    tts_thread.start()

    llm_task = asyncio.create_task(run_llm_stream())
    tts_sender = asyncio.create_task(send_tts_chunks())

    # Wait for LLM to finish (this signals TTS thread to complete)
    await llm_task

    # Wait for TTS thread to finish and sender to drain
    await tts_sender

    assistant_text = "".join(llm_full_text)

    # ── Phase 4: update session, upload full audio ───────────────────────
    await chat_session_manager.append_message(chat_sess.session_id, "user", user_text)
    await chat_session_manager.append_message(chat_sess.session_id, "assistant", assistant_text)

    tts_url = None
    try:
        tts_result = await tts_service.synthesize_and_upload(
            text=assistant_text,
            voice_type=voice_type,
            codec="mp3",
        )
        if tts_result.get("success"):
            tts_url = tts_result.get("url")
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
