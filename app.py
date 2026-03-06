#
# Multi-Pipeline Voice Agent Server
#
# Single-port FastAPI server supporting multiple concurrent pipeline configs.
# Each WebRTC session can use a different STT/TTS/LLM combination.
# Serves a custom dashboard at / with 5 pipeline comparison panels.
#

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, status
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

from database import (
    Campaign, CampaignCall, Conversation, Customer, SessionLocal, User,
    get_db, init_db, parse_customer_csv, normalize_phone,
)
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
        "inya": bool(os.getenv("INYA_WEBHOOK_SECRET")),
        "twilio": bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN")),
        "exotel": bool(os.getenv("EXOTEL_API_KEY") and os.getenv("EXOTEL_API_TOKEN")),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Campaign runner state (module-level)
# ---------------------------------------------------------------------------

_campaign_events: Dict[int, asyncio.Event] = {}   # call_id → done event
_running_campaigns: Dict[int, asyncio.Task] = {}

_PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")


async def _campaign_runner(campaign_id: int):
    """Sequential campaign dialer — runs in a background asyncio Task."""
    from dialer import initiate_call
    from datetime import datetime, timezone

    logger.info(f"Campaign runner started: campaign_id={campaign_id}")
    db = SessionLocal()
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if not campaign:
            logger.error(f"Campaign {campaign_id} not found")
            return

        provider = campaign.provider
        if provider == "twilio":
            from_number = os.getenv("TWILIO_PHONE_NUMBER", "")
        else:
            from_number = os.getenv("EXOTEL_PHONE_NUMBER", "")

        pending_calls = (
            db.query(CampaignCall)
            .filter(CampaignCall.campaign_id == campaign_id, CampaignCall.call_status == "pending")
            .all()
        )

        for cc in pending_calls:
            # Refresh campaign status
            db.refresh(campaign)
            if campaign.status == "paused":
                logger.info(f"Campaign {campaign_id} paused — stopping runner")
                break

            customer = db.query(Customer).filter(Customer.id == cc.customer_id).first()
            if not customer:
                continue

            webhook_url = f"{_PUBLIC_URL}/telephony/twiml/{cc.id}" if provider == "twilio" else f"{_PUBLIC_URL}/telephony/exotel/{cc.id}"
            status_url = f"{_PUBLIC_URL}/telephony/status/{cc.id}"

            try:
                campaign.calls_attempted += 1
                cc.call_status = "dialing"
                cc.started_at = datetime.now(timezone.utc)
                db.commit()

                sid = await initiate_call(
                    provider=provider,
                    to=customer.phone,
                    from_=from_number,
                    webhook_url=webhook_url,
                    status_url=status_url,
                )
                cc.provider_call_sid = sid
                db.commit()
                logger.info(f"Call initiated: call_id={cc.id}, SID={sid}, to={customer.phone}")

                # Wait for call to complete (signaled by on_client_disconnected or status callback)
                event = asyncio.Event()
                _campaign_events[cc.id] = event
                try:
                    await asyncio.wait_for(event.wait(), timeout=180)
                except asyncio.TimeoutError:
                    logger.warning(f"Call {cc.id} timed out after 180s")
                    cc.call_status = "failed"
                    db.commit()
                finally:
                    _campaign_events.pop(cc.id, None)

            except Exception as e:
                logger.error(f"Failed to initiate call for customer {customer.id}: {e}")
                cc.call_status = "failed"
                db.commit()

            await asyncio.sleep(3)  # brief pause between calls

        # Mark campaign completed (if not paused)
        db.refresh(campaign)
        if campaign.status == "running":
            campaign.status = "completed"
            campaign.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(f"Campaign {campaign_id} completed")

    finally:
        db.close()
        _running_campaigns.pop(campaign_id, None)


# ---------------------------------------------------------------------------
# Customer endpoints
# ---------------------------------------------------------------------------

class CustomerCreate(BaseModel):
    name: str
    phone: str
    account_number: Optional[str] = None
    loan_type: Optional[str] = "Personal Loan"
    emi_amount: Optional[str] = None
    total_due: Optional[str] = None
    overdue_months: Optional[str] = None
    overdue_period: Optional[str] = None
    late_fee: Optional[str] = None
    notes: Optional[str] = None


@app.post("/api/customers/upload")
async def upload_customers(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Import customers from CSV file."""
    content = await file.read()
    try:
        text = content.decode("utf-8-sig")  # handle BOM
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    try:
        rows = parse_customer_csv(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    imported = 0
    errors = []
    for i, row in enumerate(rows):
        try:
            cust = Customer(
                name=row.get("name", ""),
                phone=row.get("phone", ""),
                account_number=row.get("account_number"),
                loan_type=row.get("loan_type", "Personal Loan"),
                emi_amount=row.get("emi_amount"),
                total_due=row.get("total_due"),
                overdue_months=row.get("overdue_months"),
                overdue_period=row.get("overdue_period"),
                late_fee=row.get("late_fee"),
                notes=row.get("notes"),
            )
            db.add(cust)
            db.flush()
            imported += 1
        except Exception as e:
            errors.append({"row": i + 2, "error": str(e)})
    db.commit()
    return {"imported": imported, "errors": errors}


@app.post("/api/customers")
async def create_customer(
    body: CustomerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    phone = normalize_phone(body.phone)
    cust = Customer(
        name=body.name,
        phone=phone,
        account_number=body.account_number,
        loan_type=body.loan_type or "Personal Loan",
        emi_amount=body.emi_amount,
        total_due=body.total_due,
        overdue_months=body.overdue_months,
        overdue_period=body.overdue_period,
        late_fee=body.late_fee,
        notes=body.notes,
    )
    db.add(cust)
    db.commit()
    db.refresh(cust)
    return {"id": cust.id, "name": cust.name, "phone": cust.phone}


@app.get("/api/customers")
async def list_customers(
    page: int = 1,
    limit: int = 50,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Customer).order_by(Customer.created_at.desc())
    if search:
        q = q.filter(
            Customer.name.ilike(f"%{search}%") | Customer.phone.ilike(f"%{search}%")
        )
    total = q.count()
    rows = q.offset((page - 1) * limit).limit(limit).all()
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "customers": [
            {
                "id": c.id,
                "name": c.name,
                "phone": c.phone,
                "account_number": c.account_number,
                "loan_type": c.loan_type,
                "emi_amount": c.emi_amount,
                "total_due": c.total_due,
                "overdue_months": c.overdue_months,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in rows
        ],
    }


@app.delete("/api/customers/{customer_id}")
async def delete_customer(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cust = db.query(Customer).filter(Customer.id == customer_id).first()
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    db.delete(cust)
    db.commit()
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Campaign endpoints
# ---------------------------------------------------------------------------

class CampaignCreate(BaseModel):
    name: str
    description: Optional[str] = None
    provider: str  # twilio | exotel
    customer_ids: List[int]
    stt: str = "deepgram"
    tts: str = "openai"
    llm: str = "openai"


@app.post("/api/campaigns")
async def create_campaign(
    body: CampaignCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.provider not in ("twilio", "exotel"):
        raise HTTPException(status_code=400, detail="provider must be 'twilio' or 'exotel'")
    if not body.customer_ids:
        raise HTTPException(status_code=400, detail="Select at least one customer")

    camp = Campaign(
        name=body.name,
        description=body.description,
        status="draft",
        provider=body.provider,
        pipeline_stt=body.stt,
        pipeline_tts=body.tts,
        pipeline_llm=body.llm,
        total_customers=len(body.customer_ids),
    )
    db.add(camp)
    db.flush()

    for cid in body.customer_ids:
        cc = CampaignCall(campaign_id=camp.id, customer_id=cid)
        db.add(cc)

    db.commit()
    db.refresh(camp)
    return {"id": camp.id, "name": camp.name, "status": camp.status}


@app.get("/api/campaigns")
async def list_campaigns(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    camps = db.query(Campaign).order_by(Campaign.created_at.desc()).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "status": c.status,
            "provider": c.provider,
            "total_customers": c.total_customers,
            "calls_attempted": c.calls_attempted,
            "calls_completed": c.calls_completed,
            "ptp_count": c.ptp_count,
            "payment_confirmed_count": c.payment_confirmed_count,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "started_at": c.started_at.isoformat() if c.started_at else None,
        }
        for c in camps
    ]


@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(
    campaign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "status": c.status,
        "provider": c.provider,
        "pipeline_stt": c.pipeline_stt,
        "pipeline_tts": c.pipeline_tts,
        "pipeline_llm": c.pipeline_llm,
        "total_customers": c.total_customers,
        "calls_attempted": c.calls_attempted,
        "calls_connected": c.calls_connected,
        "calls_completed": c.calls_completed,
        "ptp_count": c.ptp_count,
        "payment_confirmed_count": c.payment_confirmed_count,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "started_at": c.started_at.isoformat() if c.started_at else None,
        "completed_at": c.completed_at.isoformat() if c.completed_at else None,
    }


@app.get("/api/campaigns/{campaign_id}/calls")
async def list_campaign_calls(
    campaign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    calls = (
        db.query(CampaignCall)
        .filter(CampaignCall.campaign_id == campaign_id)
        .all()
    )
    result = []
    for cc in calls:
        cust = db.query(Customer).filter(Customer.id == cc.customer_id).first()
        result.append({
            "id": cc.id,
            "customer_name": cust.name if cust else "",
            "customer_phone": cust.phone if cust else "",
            "call_status": cc.call_status,
            "outcome": cc.outcome,
            "payment_made": cc.payment_made,
            "payment_amount": cc.payment_amount,
            "has_receipt": cc.has_receipt,
            "duration_seconds": cc.duration_seconds,
            "started_at": cc.started_at.isoformat() if cc.started_at else None,
            "ended_at": cc.ended_at.isoformat() if cc.ended_at else None,
        })
    return result


@app.post("/api/campaigns/{campaign_id}/start")
async def start_campaign(
    campaign_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from datetime import datetime, timezone

    camp = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not camp:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if camp.status == "running":
        raise HTTPException(status_code=400, detail="Campaign is already running")
    if camp.status == "completed":
        raise HTTPException(status_code=400, detail="Campaign already completed")

    if not _PUBLIC_URL:
        raise HTTPException(
            status_code=400,
            detail="PUBLIC_URL env var not set — cannot generate telephony webhook URLs",
        )

    camp.status = "running"
    if not camp.started_at:
        camp.started_at = datetime.now(timezone.utc)
    db.commit()

    task = asyncio.create_task(_campaign_runner(campaign_id))
    _running_campaigns[campaign_id] = task
    return {"status": "running", "campaign_id": campaign_id}


@app.post("/api/campaigns/{campaign_id}/pause")
async def pause_campaign(
    campaign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    camp = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not camp:
        raise HTTPException(status_code=404, detail="Campaign not found")
    camp.status = "paused"
    db.commit()
    return {"status": "paused"}


@app.post("/api/campaigns/{campaign_id}/resume")
async def resume_campaign(
    campaign_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    camp = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not camp:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if camp.status != "paused":
        raise HTTPException(status_code=400, detail="Campaign is not paused")

    camp.status = "running"
    db.commit()

    task = asyncio.create_task(_campaign_runner(campaign_id))
    _running_campaigns[campaign_id] = task
    return {"status": "running"}


# ---------------------------------------------------------------------------
# Telephony webhooks (NO JWT — providers call these)
# ---------------------------------------------------------------------------

@app.post("/telephony/twiml/{call_id}")
async def twiml_webhook(call_id: int, request: Request):
    """Return TwiML that connects the call to our WebSocket stream."""
    ws_url = f"{_PUBLIC_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/telephony/ws/{call_id}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}"/>
  </Connect>
</Response>"""
    from fastapi.responses import Response as FastAPIResponse
    return FastAPIResponse(content=xml, media_type="application/xml")


@app.post("/telephony/exotel/{call_id}")
async def exotel_webhook(call_id: int, request: Request):
    """Return Exotel applet JSON that streams audio to our WebSocket."""
    ws_url = f"{_PUBLIC_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/telephony/ws/{call_id}"
    return {
        "applet": "stream",
        "parameters": {
            "url": ws_url,
            "track": "both_tracks",
        }
    }


@app.post("/telephony/status/{call_id}")
async def telephony_status(call_id: int, request: Request):
    """
    Status callback from Twilio/Exotel.
    Sets the asyncio.Event to unblock the campaign runner if still waiting.
    """
    try:
        body = await request.form()
        call_status = body.get("CallStatus") or body.get("Status") or ""
    except Exception:
        try:
            body = await request.json()
            call_status = body.get("CallStatus") or body.get("Status") or ""
        except Exception:
            call_status = ""

    logger.info(f"Telephony status callback: call_id={call_id}, status={call_status}")

    # Update DB call status
    db = SessionLocal()
    try:
        cc = db.query(CampaignCall).filter(CampaignCall.id == call_id).first()
        if cc and call_status:
            status_map = {
                "no-answer": "no_answer",
                "failed": "failed",
                "busy": "busy",
                "completed": "completed",
            }
            mapped = status_map.get(call_status.lower(), cc.call_status)
            if cc.call_status not in ("completed",):
                cc.call_status = mapped
            db.commit()
    finally:
        db.close()

    # Signal campaign runner
    event = _campaign_events.get(call_id)
    if event:
        event.set()

    return {"status": "ok"}


@app.websocket("/telephony/ws/{call_id}")
async def telephony_ws(websocket: WebSocket, call_id: int):
    """WebSocket endpoint for Twilio/Exotel media streams."""
    await websocket.accept()

    db = SessionLocal()
    try:
        cc = db.query(CampaignCall).filter(CampaignCall.id == call_id).first()
        if not cc:
            await websocket.close(code=1008)
            return

        camp = db.query(Campaign).filter(Campaign.id == cc.campaign_id).first()
        cust = db.query(Customer).filter(Customer.id == cc.customer_id).first()
        if not camp or not cust:
            await websocket.close(code=1008)
            return

        cc.call_status = "in_progress"
        camp.calls_connected += 1
        db.commit()

        customer_dict = {
            "borrower_name": cust.name,
            "account_number": cust.account_number or "",
            "loan_type": cust.loan_type or "Personal Loan",
            "emi_amount": cust.emi_amount or "0",
            "total_due": cust.total_due or "0",
            "overdue_months": cust.overdue_months or "1",
            "overdue_period": cust.overdue_period or "",
            "late_fee": cust.late_fee or "0",
        }
        provider = camp.provider
        stt_type = camp.pipeline_stt
        tts_type = camp.pipeline_tts
        llm_type = camp.pipeline_llm
    finally:
        db.close()

    from campaign_bot import run_campaign_bot
    await run_campaign_bot(
        websocket=websocket,
        call_id=call_id,
        customer=customer_dict,
        stt_type=stt_type,
        tts_type=tts_type,
        llm_type=llm_type,
        provider=provider,
    )


# inya.ai webhook endpoints
# ---------------------------------------------------------------------------
# inya agents call GET /api/inya/borrower at call start to fetch borrower data,
# and POST /api/inya/outcome after the call to report results.
# Authentication: inya sends "Authorization: Bearer <INYA_WEBHOOK_SECRET>"
# ---------------------------------------------------------------------------

_INYA_SECRET = os.getenv("INYA_WEBHOOK_SECRET", "")


def _verify_inya_auth(request: Request):
    """Validate the Bearer token inya sends with every webhook call."""
    if not _INYA_SECRET:
        return  # no secret configured — open (dev mode)
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {_INYA_SECRET}":
        raise HTTPException(status_code=401, detail="Invalid inya webhook token")


@app.get("/api/inya/borrower")
async def inya_get_borrower(
    request: Request,
    account_number: Optional[str] = None,
    phone: Optional[str] = None,
    caller_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    inya calls this (GET, On-Call action) at the start of a conversation
    to retrieve borrower details. inya injects the returned values into
    its agent prompt via Jinja2 variables.

    Configure in inya dashboard → Custom Integration → GET
      URL: https://<your-app>.onrender.com/api/inya/borrower
      Headers: Authorization: Bearer <INYA_WEBHOOK_SECRET>
      Params: account_number={{account_number}}  (or phone={{caller_phone}})
    """
    _verify_inya_auth(request)

    # If we have a matching past conversation in the DB, use that config
    cfg = {}
    if account_number:
        last = (
            db.query(Conversation)
            .filter(Conversation.account_number == account_number)
            .order_by(Conversation.started_at.desc())
            .first()
        )
        if last:
            cfg = {
                "borrower_name": last.borrower_name,
                "account_number": last.account_number,
                "company_name": last.company_name,
                "agent_name": last.agent_name,
                "language": last.language,
            }

    # Fall back to defaults from env / DEFAULT_CONFIG
    from bot import DEFAULT_CONFIG as _DC
    return {
        "agent_name":      cfg.get("agent_name",     os.getenv("INYA_AGENT_NAME",    _DC["agent_name"])),
        "company_name":    cfg.get("company_name",   os.getenv("INYA_COMPANY_NAME",  _DC["company_name"])),
        "borrower_name":   cfg.get("borrower_name",  os.getenv("INYA_BORROWER_NAME", _DC["borrower_name"])),
        "account_number":  cfg.get("account_number", account_number or _DC["account_number"]),
        "emi_amount":      os.getenv("INYA_EMI_AMOUNT",      _DC["emi_amount"]),
        "total_due":       os.getenv("INYA_TOTAL_DUE",       _DC["total_due"]),
        "overdue_months":  os.getenv("INYA_OVERDUE_MONTHS",  _DC["overdue_months"]),
        "overdue_period":  os.getenv("INYA_OVERDUE_PERIOD",  _DC["overdue_period"]),
        "late_fee":        os.getenv("INYA_LATE_FEE",        _DC["late_fee"]),
        "language":        cfg.get("language",       os.getenv("INYA_LANGUAGE",      _DC["language"])),
        "phone":           phone or caller_id or "",
    }


@app.api_route("/api/inya/outcome", methods=["POST", "PUT"])
async def inya_post_outcome(request: Request, db: Session = Depends(get_db)):
    """
    inya calls this (POST or PUT, Post-Call action) after a conversation ends
    to report the outcome. We save it as a Conversation record in our DB so it
    appears alongside PipeCat calls in the dashboard.

    Configure in inya dashboard → Custom Integration → POST
      URL: https://<your-app>.onrender.com/api/inya/outcome
      Headers: Authorization: Bearer <INYA_WEBHOOK_SECRET>
              Content-Type: application/json
      Body: {
        "call_id": "{{call_id}}",
        "borrower_name": "{{borrower_name}}",
        "account_number": "{{account_number}}",
        "outcome": "{{outcome}}",
        "payment_plan": "{{payment_plan}}",
        "payment_date": "{{payment_date}}",
        "duration_seconds": {{duration}},
        "transcript": "{{transcript}}"
      }
    """
    _verify_inya_auth(request)

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    logger.info(f"inya outcome received: {data}")

    # Parse transcript — inya may send a string or a list
    transcript_raw = data.get("transcript", "")
    if isinstance(transcript_raw, str) and transcript_raw:
        transcript = [{"role": "assistant", "text": transcript_raw}]
    elif isinstance(transcript_raw, list):
        transcript = transcript_raw
    else:
        transcript = []

    duration = data.get("duration_seconds") or data.get("duration") or 0
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0

    conv = Conversation(
        pc_id=data.get("call_id", f"inya-{int(datetime.now(timezone.utc).timestamp())}"),
        stt_type="inya",
        tts_type="inya",
        llm_type="inya",
        agent_name=data.get("agent_name", os.getenv("INYA_AGENT_NAME", "inya Agent")),
        company_name=data.get("company_name", os.getenv("INYA_COMPANY_NAME", "")),
        borrower_name=data.get("borrower_name", ""),
        account_number=data.get("account_number", ""),
        language=data.get("language", "hinglish"),
        status="completed",
        outcome=data.get("outcome"),
        payment_plan=data.get("payment_plan"),
        payment_date=data.get("payment_date"),
        ended_at=datetime.now(timezone.utc),
        duration_seconds=duration if duration else None,
        transcript_json=json.dumps(transcript, ensure_ascii=False),
        estimated_cost_usd=0.0,  # inya manages its own billing
    )
    db.add(conv)
    db.commit()
    logger.info(f"inya conversation saved (id={conv.id}, outcome={conv.outcome})")
    return {"status": "ok", "conversation_id": conv.id}


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
