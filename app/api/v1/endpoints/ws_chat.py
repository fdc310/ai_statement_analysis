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
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.services.streaming.session_manager import StreamResult, StreamingSession
from app.schemas.streaming_chat import StreamChatConfig
from app.services.agents.prompts.scenario_report import (
    scenario_summary_system_prompt,
    scenario_summary_user_prompt,
    scenario_report_system_prompt,
    scenario_report_user_prompt,
)
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


class _TextChatSession:
    """Small adapter that lets text input reuse the existing chat pipeline."""

    def __init__(self, text: str, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._text = text
        self._finished = False

    async def finish(self) -> StreamResult:
        if self._finished:
            raise RuntimeError("Session already finished")
        self._finished = True
        return StreamResult(
            session_id=self.session_id,
            speech_text=self._text,
        )


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
    text_stream_session_id: Optional[str] = None
    chat_session_id: Optional[str] = None  # Track current chat session for end_dialogue

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                logger.info("WS chat client disconnected")
                break

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

                            if config.enable_asr or config.enable_soe:
                                on_asr_partial = await _make_on_asr_partial(websocket)
                                session = StreamingSession(
                                    config=config,
                                    on_asr_partial=on_asr_partial,
                                    on_soe_intermediate=None,
                                )
                                await session.start()
                                started_session_id = session.session_id
                                logger.info(f"WS chat session started: {session.session_id}")
                            else:
                                text_stream_session_id = str(uuid.uuid4())
                                started_session_id = text_stream_session_id
                                logger.info(f"WS text chat session prepared: {text_stream_session_id}")

                            await websocket.send_json({
                                "type": "session_started",
                                "session_id": started_session_id,
                            })

                        elif msg_type == "end":
                            if not session:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "No active session",
                                })
                                continue

                            done_data = await _handle_end(websocket, session, config)
                            if done_data and done_data.get("chat_session_id"):
                                chat_session_id = done_data["chat_session_id"]

                        elif msg_type == "text":
                            if config is None:
                                config_data = data.get("config", data.get("data", {}).get("config", {}))
                                config = StreamChatConfig(**config_data)
                                config.enable_asr = False
                                config.enable_soe = False
                                text_stream_session_id = str(uuid.uuid4())

                            payload = data.get("data", {})
                            if isinstance(payload, dict):
                                user_text = payload.get("text", "")
                            else:
                                user_text = data.get("text", "")
                            user_text = str(user_text).strip()
                            if not user_text:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "text cannot be empty",
                                })
                                continue

                            text_session = _TextChatSession(
                                user_text,
                                session_id=text_stream_session_id,
                            )
                            done_data = await _handle_end(websocket, text_session, config)
                            if done_data and done_data.get("chat_session_id"):
                                config.session_id = done_data["chat_session_id"]
                                chat_session_id = done_data["chat_session_id"]

                        elif msg_type == "end_dialogue":
                            # Manual end: generate report and close dialogue
                            if not chat_session_id:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "No active chat session to end",
                                })
                                continue

                            from app.services.chat.session_manager import (
                                chat_session_manager as _csm,
                            )
                            from app.services import get_llm_service as _get_llm

                            _chat_sess = await _csm.get_session(chat_session_id)
                            if not _chat_sess:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "Chat session not found or expired",
                                })
                                continue

                            _llm = _get_llm()
                            _scene = config.scene if config else ""
                            _sub_type = config.sub_type if config else ""
                            _report_scene = f"{_scene}:{_sub_type}" if _sub_type else _scene
                            _report = await _generate_dialogue_report(
                                _llm, _chat_sess, _report_scene,
                            )
                            if _report:
                                await _csm.end_session(chat_session_id, _report)

                            await websocket.send_json({
                                "type": "dialogue_ended",
                                "data": {
                                    "chat_session_id": chat_session_id,
                                    "summary": _report.get("summary", "") if _report else "",
                                    "report": _report.get("detail", {}) if _report else {},
                                    "duration": _report.get("duration") if _report else None,
                                },
                            })
                            logger.info(f"Dialogue manually ended: {chat_session_id}")

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
                        try:
                            await session.feed_audio(message["bytes"])
                        except ValueError as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e),
                            })
                            await websocket.close(code=4009, reason="Audio duration limit exceeded")
                            return
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
    "interview": "求职面试场景（应届/社招/考公考编）",
    "interview:campus": "应届求职面试场景",
    "interview:social": "社会招聘面试场景",
    "interview:civil": "考公考编结构化面试场景",
    "office_work": "职场办公场景（工作汇报/升职加薪/离职跳槽）",
    "office_work:report": "工作汇报场景",
    "office_work:promotion": "升职加薪沟通场景",
    "office_work:resignation": "离职跳槽沟通场景",
    "business_social": "商务社交场景（销售/洽谈/社交）",
    "business_social:sales": "销售沟通场景",
    "business_social:deal": "商务洽谈场景",
    "business_social:networking": "商务社交破冰场景",
    "custom": "自定义对话场景",
    "daily": "日常对话场景",
    "customer_service": "客服场景",
}

_BLOOD_BAR_SYSTEM_PROMPT = """你是一位情景对话裁判。你的任务是判断用户在当前情景对话中的回答质量，决定血量变化。

当前场景：{criteria}

═══════════════════════════════════════
扣血规则（总血量100）
═══════════════════════════════════════

【第一层：轻微失误 -5】表达能力问题
- 说话啰嗦、绕圈子、废话多
- 重点不明确，说了一大段没有结论
- 口头禅过多（然后、那个、就是、嗯...连续出现）
- 回答过短，信息不足（如领导问原因，只回"有点问题"）

【第二层：中度失误 -10】沟通效果变差
- 答非所问，未正面回应问题
- 逻辑混乱，前后矛盾
- 情绪化表达（如"这又不是我的问题"）
- 频繁冷场，5秒以上不知道怎么接，或连续多次说"不知道"

【第三层：重度失误 -20】现实里已经翻车
- 汇报没有结论，问建议说"我也不太清楚"
- 面试暴露致命问题（如离职原因说"领导太傻了"）
- 顶撞沟通对象（如领导问为什么没完成，反问"你怎么不早点说"）
- 关键问题回避，领导连续追问一直绕圈

【第四层：致命失误 -999】直接结束对话
- 面试场景：暴露严重职业道德问题、侮辱面试官
- 工作汇报：承认关键数据没看、完全不准备
- 客户谈判：威胁客户、态度恶劣（"不买就算了"）
- 商务社交：严重失礼、触碰底线

═══════════════════════════════════════
加分规则
═══════════════════════════════════════

【表现优秀 +5~+15】
- 回答切题、有条理、有深度
- 主动挖掘需求、体现思考
- 话术得体、展现专业素养

【表现极佳 +20】
- 超出预期的精彩回答
- 化解刁难问题、展现高情商

═══════════════════════════════════════
返回格式
═══════════════════════════════════════

你必须严格返回以下JSON格式，不要返回任何其他内容：
{{"delta": <整数>, "reason": "<简短中文说明，15字以内>", "fatal": <true或false>}}

- delta：血量变化值，范围 -999 到 +20
- reason：简短说明扣血/加血原因
- fatal：是否为致命失误（true=直接结束对话，false=正常继续）"""


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
        reason = str(parsed.get("reason", ""))[:50]
        is_fatal = bool(parsed.get("fatal", False))

        # Handle fatal: directly end or halve HP
        if is_fatal:
            new_hp = 0
            delta = -current_hp  # drain all remaining HP
        else:
            # Clamp normal delta to [-20, +20]
            delta = max(-20, min(20, delta))
            new_hp = max(0, current_hp + delta)
        return {
            "hp": new_hp,
            "delta": delta,
            "reason": reason,
            "game_over": new_hp <= 0,
            "fatal": is_fatal,
        }
    except Exception as e:
        logger.error(f"Blood bar evaluation error: {e}")
        return None


async def _generate_dialogue_report(
    llm_service,
    chat_sess,
    scene: str,
    status_callback=None,
) -> Optional[dict]:
    """Generate short summary + full report for a completed dialogue session."""
    import time
    try:
        from app.services.agents.prompts.common import extract_json

        messages = chat_sess.messages or []
        blood_history = chat_sess.blood_history or []
        initial_hp = 100
        final_hp = chat_sess.hp

        total_start = time.time()

        # ── Step 1: Generate short summary ──
        step1_start = time.time()
        summary_sys = scenario_summary_system_prompt()
        summary_user = scenario_summary_user_prompt(
            scene=scene,
            messages=messages,
            blood_history=blood_history,
            final_hp=final_hp,
            initial_hp=initial_hp,
        )
        summary_result = await llm_service.chat(
            [
                {"role": "system", "content": summary_sys},
                {"role": "user", "content": summary_user},
            ],
            temperature=0.3,
            status_callback=status_callback,
        )
        step1_elapsed = round(time.time() - step1_start, 2)
        summary_content = summary_result.get("content", "").strip()
        summary_parsed = extract_json(summary_content)
        short_summary = summary_parsed.get("summary", "") if summary_parsed else ""
        if not short_summary:
            # Fallback: extract from raw content
            short_summary = summary_content[:50]
        logger.info(f"Report step1 (summary) completed in {step1_elapsed}s")

        # ── Step 2: Generate full report ──
        step2_start = time.time()
        report_sys = scenario_report_system_prompt(scene)
        report_user = scenario_report_user_prompt(
            scene=scene,
            messages=messages,
            blood_history=blood_history,
            final_hp=final_hp,
            initial_hp=initial_hp,
        )
        report_result = await llm_service.chat(
            [
                {"role": "system", "content": report_sys},
                {"role": "user", "content": report_user},
            ],
            temperature=0.3,
            status_callback=status_callback,
        )
        step2_elapsed = round(time.time() - step2_start, 2)
        report_content = report_result.get("content", "").strip()
        report_data = extract_json(report_content)
        if not report_data:
            report_data = {"raw_report": report_content}

        total_elapsed = round(time.time() - total_start, 2)
        logger.info(
            f"Report generation completed in {total_elapsed}s "
            f"(summary={step1_elapsed}s, report={step2_elapsed}s)"
        )

        return {
            "summary": short_summary,
            "detail": report_data,
            "duration": {
                "total": total_elapsed,
                "summary": step1_elapsed,
                "report": step2_elapsed,
            },
        }

    except Exception as e:
        logger.error(f"Dialogue report generation error: {e}", exc_info=True)
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
        done_data = {
            "session_id": result.session_id,
            "user_text": "",
            "assistant_text": "",
            "tts_url": None,
        }
        await send_ws_json({
            "type": "chat_done",
            "data": done_data,
        })
        return done_data

    user_text = result.speech_text

    # ── Phase 2: build chat context ──────────────────────────────────────
    from app.services.chat.session_manager import (
        chat_session_manager,
        VOICE_CHAT_SCENE_PROMPTS,
        DEFAULT_VOICE_CHAT_PROMPT,
    )
    from app.services import get_llm_service

    scene = config.scene or ""
    sub_type = config.sub_type or ""
    system_prompt = config.system_prompt or ""
    if not system_prompt:
        # Try scene:sub_type first, then scene, then default
        composite_key = f"{scene}:{sub_type}" if sub_type else scene
        system_prompt = VOICE_CHAT_SCENE_PROMPTS.get(
            composite_key,
            VOICE_CHAT_SCENE_PROMPTS.get(scene, DEFAULT_VOICE_CHAT_PROMPT),
        )

    chat_sess = await chat_session_manager.get_or_create_session(
        session_id=config.session_id,
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
        blood_scene_key = f"{scene}:{sub_type}" if sub_type else scene
        blood_bar_data = await _evaluate_blood_bar(
            llm, blood_scene_key, user_text, assistant_text, chat_sess.hp,
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

    # ── Phase 5b: generate report on game_over ───────────────────────────
    report_data = None
    if blood_bar_data and blood_bar_data.get("game_over"):
        await send_ai_status({
            "stage": "report",
            "message": "Generating dialogue report",
        })
        # Refresh session to get latest messages
        chat_sess = await chat_session_manager.get_session(chat_sess.session_id)
        if chat_sess:
            report_scene_key = f"{scene}:{sub_type}" if sub_type else scene
            report_data = await _generate_dialogue_report(
                llm, chat_sess, report_scene_key,
                status_callback=send_ai_status,
            )
            if report_data:
                await chat_session_manager.end_session(
                    chat_sess.session_id, report_data
                )

    done_data = {
        "session_id": result.session_id,
        "chat_session_id": chat_sess.session_id,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "tts_url": tts_url,
    }
    if blood_bar_data:
        done_data["blood_bar"] = blood_bar_data
    if report_data:
        done_data["report"] = report_data

    await send_ws_json({"type": "chat_done", "data": done_data})
    logger.info(f"WS chat completed: {session.session_id}")
    return done_data
