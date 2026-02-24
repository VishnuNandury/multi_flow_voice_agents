"""
Database models and helpers for the Multi-Pipeline Voice Agent.

Tables: users, conversations
Engine: SQLite by default (swappable via DATABASE_URL env var).
"""

import json
import os
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./data/agent.db")

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
    outcome = Column(String(30), nullable=True)       # ptp | callback | wrong_person | incomplete
    payment_plan = Column(String(200), nullable=True)
    payment_date = Column(String(50), nullable=True)

    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Full transcript stored as JSON text
    transcript_json = Column(Text, nullable=True)

    # Estimated cost in USD
    estimated_cost_usd = Column(Float, default=0.0)

    user = relationship("User", back_populates="conversations")


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
            started_at=datetime.fromtimestamp(start_time, tz=timezone.utc) if start_time else None,
            ended_at=datetime.fromtimestamp(end_time, tz=timezone.utc),
            duration_seconds=duration,
            transcript_json=json.dumps(transcript, ensure_ascii=False),
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
