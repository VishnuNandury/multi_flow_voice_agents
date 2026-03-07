"""
Database models and helpers for the Multi-Pipeline Voice Agent.

Tables: users, conversations
Engine: SQLite by default (swappable via DATABASE_URL env var).
"""

import json
import os
from datetime import datetime, timezone

from loguru import logger
import csv
import io
import re

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./data/agent.db")

# Neon (and some other Postgres hosts) give a URL starting with "postgres://"
# SQLAlchemy 2.x requires "postgresql://"
if _DB_URL.startswith("postgres://"):
    _DB_URL = _DB_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if _DB_URL.startswith("sqlite") else {}
engine = create_engine(_DB_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversations = relationship("Conversation", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    pc_id = Column(String(100), nullable=True)

    # Pipeline
    stt_type = Column(String(20))
    tts_type = Column(String(20))
    llm_type = Column(String(20))

    # Agent / borrower config (denormalised for quick display)
    agent_name = Column(String(100), default="Priya")
    company_name = Column(String(100), default="QuickFinance Ltd.")
    borrower_name = Column(String(100), default="")
    account_number = Column(String(50), default="")
    language = Column(String(20), default="hinglish")

    # Call lifecycle
    status = Column(String(20), default="active")    # active | completed | dropped
    outcome = Column(String(30), nullable=True)       # ptp | callback | wrong_person | incomplete | payment_confirmed
    payment_plan = Column(String(200), nullable=True)
    payment_date = Column(String(50), nullable=True)
    payment_amount = Column(Float, nullable=True)     # actual amount paid (if borrower already paid)
    has_receipt = Column(Boolean, nullable=True)      # whether borrower has payment receipt

    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Full transcript stored as JSON text
    transcript_json = Column(Text, nullable=True)

    # Pipecat metrics summary: TTFB, token usage, turn latency, etc.
    metrics_json = Column(Text, nullable=True)

    # Estimated cost in USD
    estimated_cost_usd = Column(Float, default=0.0)

    user = relationship("User", back_populates="conversations")


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=False)
    account_number = Column(String(50), nullable=True)
    loan_type = Column(String(50), nullable=True, default="Personal Loan")
    emi_amount = Column(String(20), nullable=True)
    total_due = Column(String(20), nullable=True)
    overdue_months = Column(String(10), nullable=True)
    overdue_period = Column(String(100), nullable=True)
    late_fee = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    campaign_calls = relationship("CampaignCall", back_populates="customer")


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="draft")  # draft/running/paused/completed
    provider = Column(String(20), nullable=False)  # twilio/exotel
    pipeline_stt = Column(String(20), default="deepgram")
    pipeline_tts = Column(String(20), default="openai")
    pipeline_llm = Column(String(20), default="openai")
    total_customers = Column(Integer, default=0)
    calls_attempted = Column(Integer, default=0)
    calls_connected = Column(Integer, default=0)
    calls_completed = Column(Integer, default=0)
    ptp_count = Column(Integer, default=0)
    payment_confirmed_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    calls = relationship("CampaignCall", back_populates="campaign")


class CampaignCall(Base):
    __tablename__ = "campaign_calls"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    provider_call_sid = Column(String(100), nullable=True)
    call_status = Column(String(20), default="pending")  # pending/dialing/in_progress/completed/failed/no_answer/busy
    outcome = Column(String(30), nullable=True)
    payment_made = Column(Boolean, default=False)
    payment_amount = Column(Float, nullable=True)
    has_receipt = Column(Boolean, nullable=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    campaign = relationship("Campaign", back_populates="calls")
    customer = relationship("Customer", back_populates="campaign_calls")


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Initialise DB + seed default admin user
# ---------------------------------------------------------------------------

def init_db():
    """Create tables and ensure at least one admin user exists."""
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "changeme")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")

        if admin_password == "changeme":
            logger.warning("ADMIN_PASSWORD is set to default 'changeme' — change it in production!")

        existing = db.query(User).filter(User.username == admin_username).first()
        if not existing:
            from auth import get_password_hash
            admin = User(
                username=admin_username,
                email=admin_email,
                password_hash=get_password_hash(admin_password),
                role="admin",
            )
            db.add(admin)
            db.commit()
            logger.info(f"Created default admin user: {admin_username}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Costs per unit (USD)
_STT_COST_PER_SEC = {
    "deepgram": 0.0043 / 60,
    "whisper":  0.006  / 60,
    "sarvam":   0.001  / 60,   # estimated
}
_TTS_COST_PER_CHAR = {
    "openai": 15.0    / 1_000_000,
    "edge":   0.0,
    "sarvam": 1.0     / 1_000_000,  # estimated
}
_LLM_INPUT_COST_PER_TOKEN = {
    "openai": 2.5  / 1_000_000,   # gpt-4o
    "ollama": 0.0,
}
_LLM_OUTPUT_COST_PER_TOKEN = {
    "openai": 10.0 / 1_000_000,
    "ollama": 0.0,
}


def _estimate_cost(
    stt_type: str,
    tts_type: str,
    llm_type: str,
    duration_seconds: float,
    transcript: list,
) -> float:
    cost = 0.0

    # STT: charged per second
    cost += duration_seconds * _STT_COST_PER_SEC.get(stt_type, 0.0)

    # TTS: charged per character (assistant turns)
    assistant_chars = sum(len(m.get("text", "")) for m in transcript if m.get("role") == "assistant")
    cost += assistant_chars * _TTS_COST_PER_CHAR.get(tts_type, 0.0)

    # LLM: rough estimate — treat every 4 chars as 1 token
    total_chars = sum(len(m.get("text", "")) for m in transcript)
    est_tokens = total_chars / 4.0
    input_cost = _LLM_INPUT_COST_PER_TOKEN.get(llm_type, _LLM_INPUT_COST_PER_TOKEN["openai"])
    output_cost = _LLM_OUTPUT_COST_PER_TOKEN.get(llm_type, _LLM_OUTPUT_COST_PER_TOKEN["openai"])
    # Assume input:output ratio roughly 3:1
    cost += est_tokens * 0.75 * input_cost
    cost += est_tokens * 0.25 * output_cost

    return round(cost, 6)


# ---------------------------------------------------------------------------
# Save conversation (called from bot.py on disconnect)
# ---------------------------------------------------------------------------

def save_conversation_sync(pc_id: str, data: dict):
    """Synchronous DB write — wrap in asyncio.to_thread from async code."""
    try:
        db: Session = SessionLocal()
        cfg = data.get("agent_config", {})
        transcript = data.get("transcript", [])
        start_time = data.get("start_time")
        end_time = data.get("end_time", datetime.now(timezone.utc).timestamp())
        duration = end_time - start_time if start_time else None

        cost = _estimate_cost(
            stt_type=data.get("stt_type", "deepgram"),
            tts_type=data.get("tts_type", "openai"),
            llm_type=data.get("llm_type", "openai"),
            duration_seconds=duration or 0.0,
            transcript=transcript,
        )

        conv = Conversation(
            pc_id=pc_id,
            stt_type=data.get("stt_type"),
            tts_type=data.get("tts_type"),
            llm_type=data.get("llm_type"),
            agent_name=cfg.get("agent_name", "Priya"),
            company_name=cfg.get("company_name", ""),
            borrower_name=cfg.get("borrower_name", ""),
            account_number=cfg.get("account_number", ""),
            language=cfg.get("language", "hinglish"),
            status="completed",
            outcome=data.get("outcome"),
            payment_plan=data.get("payment_plan"),
            payment_date=data.get("payment_date"),
            payment_amount=data.get("payment_amount"),
            has_receipt=data.get("has_receipt"),
            started_at=datetime.fromtimestamp(start_time, tz=timezone.utc) if start_time else None,
            ended_at=datetime.fromtimestamp(end_time, tz=timezone.utc),
            duration_seconds=duration,
            transcript_json=json.dumps(transcript, ensure_ascii=False),
            metrics_json=json.dumps(data.get("metrics_summary") or {}, ensure_ascii=False),
            estimated_cost_usd=cost,
        )
        db.add(conv)
        db.commit()
        logger.info(f"Conversation saved to DB (pc_id={pc_id}, outcome={conv.outcome}, cost=${cost:.4f})")
    except Exception as e:
        logger.error(f"Failed to save conversation to DB: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phone normalization + CSV import
# ---------------------------------------------------------------------------

def normalize_phone(phone: str) -> str:
    """Strip non-digits and prepend +91 for 10-digit Indian numbers."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 10:
        return f"+91{digits}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    if len(digits) == 13 and digits.startswith("091"):
        return f"+91{digits[3:]}"
    # Already in E.164-ish form
    if phone.startswith("+"):
        return f"+{digits}"
    return digits


_CSV_FIELD_ALIASES = {
    "name": ["name", "borrower_name", "borrower", "customer_name"],
    "phone": ["phone", "mobile", "phone_number", "mobile_number", "contact"],
    "account_number": ["account_number", "account", "acc_no", "account_no"],
    "loan_type": ["loan_type", "loan", "type"],
    "emi_amount": ["emi_amount", "emi", "monthly_emi"],
    "total_due": ["total_due", "total", "due_amount", "outstanding"],
    "overdue_months": ["overdue_months", "overdue", "months_overdue"],
    "overdue_period": ["overdue_period", "period"],
    "late_fee": ["late_fee", "fee", "penalty"],
    "notes": ["notes", "note", "remarks", "comment"],
}


def _map_csv_header(header: str) -> str | None:
    """Map a CSV column header to our internal field name."""
    h = header.strip().lower().replace(" ", "_").replace("-", "_")
    for field, aliases in _CSV_FIELD_ALIASES.items():
        if h in aliases:
            return field
    return None


def parse_customer_csv(text: str) -> list[dict]:
    """
    Parse CSV text into a list of customer dicts.
    Returns list of dicts with keys matching Customer model fields.
    Raises ValueError if 'name' or 'phone' columns are missing.
    """
    reader = csv.DictReader(io.StringIO(text.strip()))
    headers = reader.fieldnames or []
    col_map = {}
    for h in headers:
        mapped = _map_csv_header(h)
        if mapped:
            col_map[h] = mapped

    if not any(v == "name" for v in col_map.values()):
        raise ValueError("CSV must have a 'name' column")
    if not any(v == "phone" for v in col_map.values()):
        raise ValueError("CSV must have a 'phone' column")

    customers = []
    for row in reader:
        record: dict = {}
        for csv_col, field in col_map.items():
            val = (row.get(csv_col) or "").strip()
            if val:
                record[field] = val
        if not record.get("name") or not record.get("phone"):
            continue  # skip rows with missing required fields
        record["phone"] = normalize_phone(record["phone"])
        customers.append(record)

    return customers
