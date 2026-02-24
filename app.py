#
# Multi-Pipeline Voice Agent Server
#
# Single-port FastAPI server supporting multiple concurrent pipeline configs.
# Each WebRTC session can use a different STT/TTS/LLM combination.
# Serves a custom dashboard at / with 5 pipeline comparison panels.
#

import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Validate required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# Optional service keys (warn if missing)
if not os.getenv("SARVAM_API_KEY"):
    logger.warning("SARVAM_API_KEY not set - Sarvam AI pipeline will not work")
if not os.getenv("OLLAMA_BASE_URL") and not os.getenv("GROQ_API_KEY"):
    logger.warning("Neither OLLAMA_BASE_URL nor GROQ_API_KEY set - Ollama/Groq pipeline will fall back to OpenAI")

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
_next_pipeline: Dict[str, Any] = {
    "pipeline_stt": "deepgram",
    "pipeline_tts": "openai",
    "pipeline_llm": "openai",
    "agent_config": {},
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from database import Conversation, User, get_db, init_db
from auth import create_access_token, decode_token, verify_password

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.username == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/login")
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "user": {"username": user.username, "role": user.role}}


@app.get("/auth/me")
async def me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "role": current_user.role, "email": current_user.email}


# ---------------------------------------------------------------------------
# Data API endpoints (all require auth)
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def get_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dashboard KPIs aggregated from the conversations table."""
    from sqlalchemy import func

    total = db.query(func.count(Conversation.id)).scalar() or 0
    completed = db.query(func.count(Conversation.id)).filter(Conversation.outcome.isnot(None)).scalar() or 0
    ptp_count = db.query(func.count(Conversation.id)).filter(Conversation.outcome == "ptp").scalar() or 0
    avg_dur = db.query(func.avg(Conversation.duration_seconds)).filter(
        Conversation.duration_seconds.isnot(None)
    ).scalar() or 0
    total_cost = db.query(func.sum(Conversation.estimated_cost_usd)).scalar() or 0

    # Outcome breakdown
    outcome_rows = db.query(
        Conversation.outcome, func.count(Conversation.id)
    ).group_by(Conversation.outcome).all()
    outcome_breakdown = {row[0] or "unknown": row[1] for row in outcome_rows}

    # Pipeline usage breakdown (stt+tts+llm)
    pipeline_rows = db.query(
        Conversation.stt_type, Conversation.tts_type, Conversation.llm_type,
        func.count(Conversation.id)
    ).group_by(Conversation.stt_type, Conversation.tts_type, Conversation.llm_type).all()
    pipeline_breakdown = {
        f"{r[0]}+{r[1]}+{r[2]}": r[3] for r in pipeline_rows
    }

    # Daily counts — last 7 days
    daily_counts = []
    for days_ago in range(6, -1, -1):
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_ago)
        day_end = day_start + timedelta(days=1)
        count = db.query(func.count(Conversation.id)).filter(
            Conversation.started_at >= day_start,
            Conversation.started_at < day_end,
        ).scalar() or 0
        daily_counts.append({"date": day_start.strftime("%b %d"), "count": count})

    return {
        "total": total,
        "completed": completed,
        "ptp_count": ptp_count,
        "ptp_rate": round(ptp_count / completed, 3) if completed else 0,
        "avg_duration_seconds": round(avg_dur, 1),
        "total_cost_usd": round(total_cost, 4),
        "active_now": len(bot_module.session_data),
        "outcome_breakdown": outcome_breakdown,
        "pipeline_breakdown": pipeline_breakdown,
        "daily_counts": daily_counts,
    }


@app.get("/api/conversations")
async def list_conversations(
    page: int = 1,
    limit: int = 20,
    outcome: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Conversation).order_by(Conversation.started_at.desc())
    if outcome:
        q = q.filter(Conversation.outcome == outcome)
    if search:
        q = q.filter(Conversation.borrower_name.ilike(f"%{search}%"))
    total = q.count()
    rows = q.offset((page - 1) * limit).limit(limit).all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "conversations": [
            {
                "id": c.id,
                "borrower_name": c.borrower_name,
                "account_number": c.account_number,
                "pipeline": f"{c.stt_type}+{c.tts_type}+{c.llm_type}",
                "stt_type": c.stt_type,
                "tts_type": c.tts_type,
                "llm_type": c.llm_type,
                "outcome": c.outcome,
                "payment_plan": c.payment_plan,
                "payment_date": c.payment_date,
                "duration_seconds": c.duration_seconds,
                "estimated_cost_usd": c.estimated_cost_usd,
                "started_at": c.started_at.isoformat() if c.started_at else None,
            }
            for c in rows
        ],
    }


@app.get("/api/conversations/{conv_id}")
async def get_conversation(
    conv_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    transcript = json.loads(c.transcript_json) if c.transcript_json else []
    return {
        "id": c.id,
        "pc_id": c.pc_id,
        "borrower_name": c.borrower_name,
        "account_number": c.account_number,
        "agent_name": c.agent_name,
        "company_name": c.company_name,
        "language": c.language,
        "stt_type": c.stt_type,
        "tts_type": c.tts_type,
        "llm_type": c.llm_type,
        "outcome": c.outcome,
        "payment_plan": c.payment_plan,
        "payment_date": c.payment_date,
        "duration_seconds": c.duration_seconds,
        "estimated_cost_usd": c.estimated_cost_usd,
        "started_at": c.started_at.isoformat() if c.started_at else None,
        "ended_at": c.ended_at.isoformat() if c.ended_at else None,
        "transcript": transcript,
    }


@app.post("/set-pipeline")
async def set_pipeline(request: Request):
    """Called by the dashboard before opening the prebuilt client."""
    global _next_pipeline
    data = await request.json()
    _next_pipeline = {
        "pipeline_stt": data.get("stt", "deepgram"),
        "pipeline_tts": data.get("tts", "openai"),
        "pipeline_llm": data.get("llm", "openai"),
        "agent_config": data.get("config", {}),
    }
    logger.info(f"Pipeline selected: STT={_next_pipeline['pipeline_stt']}, "
                f"TTS={_next_pipeline['pipeline_tts']}, LLM={_next_pipeline['pipeline_llm']}")
    return {"status": "ok", "pipeline": _next_pipeline}


@app.get("/api/config-status")
async def config_status():
    """Return which services have their required env vars configured."""
    return {
        "deepgram": bool(os.getenv("DEEPGRAM_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "sarvam": bool(os.getenv("SARVAM_API_KEY")),
        "ollama_or_groq": bool(os.getenv("OLLAMA_BASE_URL") or os.getenv("GROQ_API_KEY")),
    }


# ---------------------------------------------------------------------------
# WebRTC signaling routes
# ---------------------------------------------------------------------------

@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    pipeline_config = request.request_data or {}
    stt = pipeline_config.get("pipeline_stt", _next_pipeline.get("pipeline_stt", "deepgram"))
    tts = pipeline_config.get("pipeline_tts", _next_pipeline.get("pipeline_tts", "openai"))
    llm = pipeline_config.get("pipeline_llm", _next_pipeline.get("pipeline_llm", "openai"))
    agent_config = _next_pipeline.get("agent_config", {})
    logger.info(f"POST /api/offer - Pipeline: STT={stt}, TTS={tts}, LLM={llm}")

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        logger.info(f"WebRTC connection established (STT={stt}, TTS={tts}, LLM={llm})")
        background_tasks.add_task(bot_module.run_bot, connection, stt, tts, llm, agent_config)

    try:
        answer = await small_webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )
        logger.info(f"SDP answer generated for {stt}+{tts}+{llm}")
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
# Flow state API (polled by the dashboard for real-time visualization)
# ---------------------------------------------------------------------------

@app.get("/api/active-flow")
async def get_active_flow():
    """Return flow state and transcript for the current active session."""
    for pc_id, data in bot_module.session_data.items():
        return {
            "current_node": data.get("current_node"),
            "flow_nodes": bot_module.FLOW_NODES,
            "transcript": data.get("transcript", []),
            "stt_type": data.get("stt_type"),
            "tts_type": data.get("tts_type"),
            "llm_type": data.get("llm_type"),
            "duration": time.time() - data.get("start_time", time.time()),
        }
    return {"current_node": None, "flow_nodes": bot_module.FLOW_NODES, "transcript": []}


@app.get("/api/flow-nodes")
async def get_flow_nodes():
    """Return the list of flow node definitions."""
    return bot_module.FLOW_NODES


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
