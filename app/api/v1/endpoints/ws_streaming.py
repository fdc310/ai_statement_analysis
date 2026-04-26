"""
WebSocket streaming endpoint for real-time audio evaluation.
Clients send PCM audio chunks and receive real-time ASR/SOE results.
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.services.streaming.session_manager import StreamingSession
from app.schemas.streaming import StreamConfig, StreamResultMessage

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
    1. Client connects (optionally with auth token)
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
    logger.info("WebSocket client connected")

    session: Optional[StreamingSession] = None

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.receive":
                if "text" in message:
                    # Text message (JSON)
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")

                        if msg_type == "config":
                            # Initialize session with config
                            config_data = data.get("data", {})
                            config = StreamConfig(**config_data)

                            # Create callbacks for real-time updates
                            async def on_asr_partial(event):
                                await websocket.send_json({
                                    "type": "asr_partial",
                                    "data": event.get("data", {})
                                })

                            async def on_soe_intermediate(event):
                                await websocket.send_json({
                                    "type": "soe_intermediate",
                                    "data": event.get("data", {})
                                })

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
                            # Finish session and send results
                            if session:
                                result = await session.finish()

                                # Step 1: Send streaming results immediately
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

                                # Step 2: Run post-stream agents if eval_type is set
                                eval_type = session.config.eval_type
                                if eval_type and eval_type != "none":
                                    from app.services.agents.base_agent import EvaluationContext
                                    from app.services.agents.orchestrator import orchestrator

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
                                        # Progressive: send each agent result as it completes
                                        async def on_agent_result(agent_name, agent_result):
                                            try:
                                                await websocket.send_json({
                                                    "type": "agent_result",
                                                    "data": {
                                                        "agent": agent_name,
                                                        "success": agent_result.success,
                                                        "result": agent_result.data,
                                                        "duration_ms": agent_result.duration_ms,
                                                        "error": agent_result.error,
                                                    }
                                                })
                                            except Exception:
                                                pass

                                        agent_results = await orchestrator.run_remaining_agents(
                                            pipeline_name=eval_type,
                                            context=context,
                                            on_agent_result=on_agent_result,
                                        )
                                    else:
                                        # One-shot: wait for all agents
                                        agent_results = await orchestrator.run_remaining_agents(
                                            pipeline_name=eval_type,
                                            context=context,
                                        )

                                    # Final complete with all data
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
                                    # No agents — send basic complete
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

                    except json.JSONDecodeError as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Invalid JSON: {e}"
                        })

                elif "bytes" in message:
                    # Binary message (audio data)
                    if session:
                        await session.feed_audio(message["bytes"])
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No active session. Send config first."
                        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        if session and not session._finished:
            try:
                await session.finish()
            except:
                pass
