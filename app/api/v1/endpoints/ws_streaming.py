"""
WebSocket streaming endpoint for real-time audio evaluation.
Clients send PCM audio chunks and receive real-time ASR/SOE results.
"""
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.services.streaming.session_manager import StreamingSession
from app.schemas.streaming import StreamConfig, StreamResultMessage
from app.core.config import settings
from app.core.security import aes_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/stream")
async def websocket_streaming_eval(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time audio streaming evaluation.

    Protocol:
    1. Client connects with ?token=<AES signature>
    2. Client sends config: {"type": "config", "data": {"language": "zh", ...}}
    3. Client sends audio: binary frames (raw PCM 16kHz 16bit mono bytes)
    4. Client sends end: {"type": "end"}
    5. Server streams results:
       - {"type": "asr_partial", "data": {"text": "..."}}
       - {"type": "soe_intermediate", "data": {"scores": {...}}}
       - {"type": "complete", "data": {"final_result": {...}}}
       - {"type": "error", "message": "..."}
    """
    await websocket.accept()

    # Authenticate via AES signature in query param
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    success, message = aes_service.verify_signature(token, max_age_seconds=settings.request_expire_seconds)
    if not success:
        await websocket.close(code=4003, reason="Unauthorized")
        logger.warning(f"WS stream auth failed: {message}")
        return

    logger.info("WebSocket client connected")

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
                            config = StreamConfig(**config_data)

                            async def _make_on_asr_partial(ws: WebSocket):
                                async def on_asr_partial(event):
                                    try:
                                        await ws.send_json({
                                            "type": "asr_partial",
                                            "data": event.get("data", {})
                                        })
                                    except Exception:
                                        pass
                                return on_asr_partial

                            async def _make_on_soe_intermediate(ws: WebSocket):
                                async def on_soe_intermediate(event):
                                    try:
                                        await ws.send_json({
                                            "type": "soe_intermediate",
                                            "data": event.get("data", {})
                                        })
                                    except Exception:
                                        pass
                                return on_soe_intermediate

                            on_asr_partial = await _make_on_asr_partial(websocket)
                            on_soe_intermediate = await _make_on_soe_intermediate(websocket)

                            session = StreamingSession(
                                config=config,
                                on_asr_partial=on_asr_partial,
                                on_soe_intermediate=on_soe_intermediate,
                            )
                            await session.start()

                            await websocket.send_json({
                                "type": "session_started",
                                "session_id": session.session_id
                            })
                            logger.info(f"Session started: {session.session_id}")

                        elif msg_type == "end":
                            if session:
                                result = await session.finish()

                                await websocket.send_json({
                                    "type": "streaming_complete",
                                    "data": {
                                        "session_id": result.session_id,
                                        "speech_text": result.speech_text,
                                        "scores_data": result.scores_data,
                                        "word_info_list": result.word_info_list,
                                        "low_score_words": result.low_score_words,
                                        "statistics_data": result.statistics_data,
                                        "speech_rate": result.speech_rate,
                                        "audio_duration": result.audio_duration,
                                        "audio_url": result.audio_url,
                                        "asr_result": result.asr_result,
                                        "soe_result": result.soe_result,
                                    }
                                })

                                eval_type = session.config.eval_type
                                if eval_type and eval_type != "none":
                                    from app.services.agents.base_agent import EvaluationContext
                                    from app.services.agents.orchestrator import orchestrator

                                    ws_send_lock = asyncio.Lock()

                                    async def send_ai_status(data: dict):
                                        async with ws_send_lock:
                                            await websocket.send_json({
                                                "type": "ai_status",
                                                "data": data,
                                            })

                                    await send_ai_status({
                                        "stage": "preparing",
                                        "message": "AI post-processing is starting",
                                        "eval_type": eval_type,
                                    })

                                    request_data = {
                                        "language": session.config.language,
                                        "ref_text": session.config.ref_text,
                                        "eval_mode": session.config.eval_mode,
                                        "score_coeff": session.config.score_coeff,
                                        "server_type": session.config.server_type,
                                        "eval_type": eval_type,
                                        "topic": session.config.topic or session.config.scenario,
                                        "scenario": session.config.scenario,
                                        "reference_text": session.config.reference_text,
                                        "report_format": session.config.report_format,
                                        "custom_prompt": session.config.custom_prompt,
                                        "word_info": session.config.word_info,
                                        "_llm_status_callback": send_ai_status,
                                    }

                                    context = EvaluationContext(request_data)
                                    context.speech_text = result.speech_text
                                    context.word_info_list = result.word_info_list
                                    context.soe_result = result.soe_result
                                    context.scores_data = result.scores_data
                                    context.low_score_words = result.low_score_words
                                    context.statistics_data = result.statistics_data
                                    context.speech_rate = result.speech_rate
                                    context.audio_duration = result.audio_duration
                                    context.audio_url = result.audio_url

                                    if session.config.progressive:
                                        async def _make_on_agent_result(ws: WebSocket):
                                            async def on_agent_result(agent_name, agent_result):
                                                try:
                                                    async with ws_send_lock:
                                                        await ws.send_json({
                                                            "type": "agent_result",
                                                            "data": {
                                                                "agent": agent_name,
                                                                "success": agent_result.success,
                                                                "result": agent_result.data,
                                                                "duration_ms": agent_result.duration_ms,
                                                                "error": agent_result.error,
                                                            }
                                                        })
                                                except Exception as e:
                                                    logger.error(f"send agent_result error: {e}")
                                            return on_agent_result

                                        on_agent_result = await _make_on_agent_result(websocket)
                                        agent_results = await orchestrator.run_remaining_agents(
                                            pipeline_name=eval_type,
                                            context=context,
                                            on_agent_result=on_agent_result,
                                        )
                                    else:
                                        agent_results = await orchestrator.run_remaining_agents(
                                            pipeline_name=eval_type,
                                            context=context,
                                        )

                                    async with ws_send_lock:
                                        await websocket.send_json({
                                            "type": "complete",
                                            "data": {
                                                "session_id": result.session_id,
                                                "speech_text": result.speech_text,
                                                "scores_data": result.scores_data,
                                                "statistics_data": result.statistics_data,
                                                "low_score_words": result.low_score_words,
                                                "speech_rate": result.speech_rate,
                                                "audio_url": result.audio_url,
                                                "report": agent_results.get("report"),
                                                "content_analysis": agent_results.get("content_analysis"),
                                                "fluency_analysis": agent_results.get("fluency_analysis"),
                                                "overall_score": agent_results.get("overall_score"),
                                                "agent_results": agent_results.get("agent_results", {}),
                                            }
                                        })
                                else:
                                    await websocket.send_json({
                                        "type": "complete",
                                        "data": {
                                            "session_id": result.session_id,
                                            "speech_text": result.speech_text,
                                            "scores_data": result.scores_data,
                                            "statistics_data": result.statistics_data,
                                            "low_score_words": result.low_score_words,
                                            "speech_rate": result.speech_rate,
                                            "audio_url": result.audio_url,
                                        }
                                    })

                                logger.info(f"Session completed: {session.session_id}")
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "No active session"
                                })

                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown message type: {msg_type}"
                            })

                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format"
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
                            "message": "No active session. Send config first."
                        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Internal server error"
            })
        except Exception:
            pass
    finally:
        if session and not session._finished:
            try:
                await session.finish()
            except Exception as e:
                logger.error(f"WS stream session cleanup error: {e}")
