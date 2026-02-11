#
# Multi-Pipeline Voice Agent Server
#
# Single-port FastAPI server supporting multiple concurrent pipeline configs.
# Each WebRTC session can use a different STT/TTS combination.
# Serves a custom dashboard at / with 4 pipeline comparison panels.
#

import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Validate required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ICE / TURN configuration
# ---------------------------------------------------------------------------

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)


def _parse_turn_urls(raw: str) -> List[str]:
    """Parse TURN_URL which may contain multiple URLs separated by commas or whitespace."""
    urls = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if part.split(":")[0] in ("stun", "stuns", "turn", "turns"):
            urls.append(part)
        else:
            logger.warning(f"Skipping invalid TURN URL (missing scheme): {part}")
    return urls


def build_ice_servers():
    server_ice = []
    client_ice = []

    server_ice.append(IceServer(urls="stun:stun.l.google.com:19302"))
    client_ice.append({"urls": ["stun:stun.l.google.com:19302"]})

    turn_url_raw = (os.getenv("TURN_URL") or "").strip()
    turn_username = (os.getenv("TURN_USERNAME") or "").strip()
    turn_credential = (os.getenv("TURN_CREDENTIAL") or "").strip()

    if turn_url_raw and turn_username and turn_credential:
        turn_urls = _parse_turn_urls(turn_url_raw)
        if not turn_urls:
            logger.error(f"TURN_URL set but no valid URLs found: {turn_url_raw!r}")
        else:
            logger.info(f"TURN configured with {len(turn_urls)} URL(s): {turn_urls}")
            for url in turn_urls:
                server_ice.append(
                    IceServer(urls=url, username=turn_username, credential=turn_credential)
                )
            client_ice.append({
                "urls": turn_urls,
                "username": turn_username,
                "credential": turn_credential,
            })
    else:
        logger.warning("No TURN server configured. Set TURN_URL, TURN_USERNAME, TURN_CREDENTIAL")

    return server_ice, client_ice


ICE_SERVERS, _CLIENT_ICE_LIST = build_ice_servers()
ICE_CONFIG_FOR_CLIENT = {"iceServers": _CLIENT_ICE_LIST}

logger.info(f"ICE servers: {len(ICE_SERVERS)} (server-side)")
logger.info(f"Client ICE config: {ICE_CONFIG_FOR_CLIENT}")

# ---------------------------------------------------------------------------
# Import bot module
# ---------------------------------------------------------------------------

import bot as bot_module

# ---------------------------------------------------------------------------
# SmallWebRTC handler
# ---------------------------------------------------------------------------

small_webrtc_handler = SmallWebRTCRequestHandler(ice_servers=ICE_SERVERS)

# In-memory session store: session_id -> pipeline config dict
active_sessions: Dict[str, Dict[str, Any]] = {}

# Pipeline selected from dashboard (consumed by the next /start call)
_next_pipeline: Dict[str, str] = {"pipeline_stt": "deepgram", "pipeline_tts": "openai"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


app = FastAPI(title="Multi-Pipeline Voice Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "turn_configured": any(
            s.username for s in ICE_SERVERS if hasattr(s, "username") and s.username
        ),
    }


@app.get("/debug/ice")
async def debug_ice():
    return {
        "server_ice_count": len(ICE_SERVERS),
        "client_ice_config": ICE_CONFIG_FOR_CLIENT,
    }

##added
@app.get("/debug/audio-config")
async def debug_audio_config():
    """Debug endpoint to check audio configuration."""
    return {
        "edge_tts": {
            "sample_rate": 24000,
            "voice": os.getenv("EDGE_TTS_VOICE", "hi-IN-SwaraNeural"),
            "rate": os.getenv("EDGE_TTS_RATE", "+0%"),
        },
        "deepgram_stt": {
            "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
            "language": os.getenv("DEEPGRAM_LANGUAGE", "hi"),
        },
        "openai": {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "tts_voice": os.getenv("OPENAI_TTS_VOICE", "shimmer"),
        },
    }


@app.post("/set-pipeline")
async def set_pipeline(request: Request):
    """Called by the dashboard before opening the prebuilt client."""
    global _next_pipeline
    data = await request.json()
    _next_pipeline = {
        "pipeline_stt": data.get("stt", "deepgram"),
        "pipeline_tts": data.get("tts", "openai"),
    }
    logger.info(f"Pipeline selected: {_next_pipeline}")
    return {"status": "ok", "pipeline": _next_pipeline}


# ---------------------------------------------------------------------------
# WebRTC signaling routes
# ---------------------------------------------------------------------------

# @app.post("/api/offer")
# async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
#     from pipecat.runner.types import SmallWebRTCRunnerArguments

#     logger.info(f"POST /api/offer (direct)")

#     async def webrtc_connection_callback(connection: SmallWebRTCConnection):
#         logger.info("WebRTC connection established, starting bot pipeline")
#         runner_args = SmallWebRTCRunnerArguments(
#             webrtc_connection=connection,
#             body=request.request_data,
#         )
#         background_tasks.add_task(bot_module.bot, runner_args)

#     try:
#         answer = await small_webrtc_handler.handle_web_request(
#             request=request,
#             webrtc_connection_callback=webrtc_connection_callback,
#         )
#         logger.info("SDP answer generated successfully")
#         return answer
#     except Exception as e:
#         logger.error(f"Error handling WebRTC offer: {e}", exc_info=True)
#         raise
@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    from pipecat.runner.types import SmallWebRTCRunnerArguments

    # Add this logging
    pipeline_config = request.request_data or {}
    stt = pipeline_config.get("pipeline_stt", "deepgram")
    tts = pipeline_config.get("pipeline_tts", "openai")
    logger.info(f"POST /api/offer - Pipeline: STT={stt}, TTS={tts}")

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        logger.info(f"WebRTC connection established (STT={stt}, TTS={tts})")
        runner_args = SmallWebRTCRunnerArguments(
            webrtc_connection=connection,
            body=request.request_data,
        )
        background_tasks.add_task(bot_module.bot, runner_args)

    try:
        answer = await small_webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )
        logger.info(f"SDP answer generated successfully for {stt}+{tts}")
        return answer
    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}", exc_info=True)
        raise

@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.post("/start")
async def rtvi_start(request: Request):
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())

    # Pipeline comes from: request body (custom dashboard) > dashboard selection > defaults
    body = request_data.get("body") or {}
    if not body.get("pipeline_stt"):
        body = dict(_next_pipeline)
    active_sessions[session_id] = body

    result = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers") or ICE_CONFIG_FOR_CLIENT["iceServers"]:
        result["iceConfig"] = ICE_CONFIG_FOR_CLIENT

    logger.info(
        f"POST /start -> session={session_id}, "
        f"pipeline={body.get('pipeline_stt', 'deepgram')}+{body.get('pipeline_tts', 'openai')}"
    )
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_request(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    active_session = active_sessions.get(session_id)
    if active_session is None:
        return Response(content="Invalid session", status_code=404)

    if path.endswith("api/offer"):
        try:
            request_data = await request.json()
            if request.method == "POST":
                webrtc_request = SmallWebRTCRequest(
                    sdp=request_data["sdp"],
                    type=request_data["type"],
                    pc_id=request_data.get("pc_id"),
                    restart_pc=request_data.get("restart_pc"),
                    request_data=request_data.get("request_data")
                    or request_data.get("requestData")
                    or active_session,
                )
                return await offer(webrtc_request, background_tasks)
            elif request.method == "PATCH":
                patch_request = SmallWebRTCPatchRequest(
                    pc_id=request_data["pc_id"],
                    candidates=[
                        IceCandidate(
                            candidate=c["candidate"],
                            sdp_mid=c.get("sdpMid", c.get("sdp_mid", "")),
                            sdp_mline_index=c.get("sdpMLineIndex", c.get("sdp_mline_index", 0)),
                        )
                        for c in request_data.get("candidates", [])
                    ],
                )
                return await ice_candidate(patch_request)
        except Exception as e:
            logger.error(f"WebRTC proxy error: {e}")
            return Response(content="Invalid request", status_code=400)

    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Static files (must be mounted AFTER API routes)
# ---------------------------------------------------------------------------

# Mount the prebuilt WebRTC client (agent metrics, etc.) at /client
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

app.mount("/client", SmallWebRTCPrebuiltUI)

# Mount custom dashboard at /
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("SERVER_PORT", "7860")))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Multi-Pipeline server on {host}:{port}")
    logger.info(f"ICE servers: {len(ICE_SERVERS)} configured")
    uvicorn.run(app, host=host, port=port)
