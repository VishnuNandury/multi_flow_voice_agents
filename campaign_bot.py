"""
campaign_bot.py — Outbound loan collection bot via Twilio/Exotel WebSocket.

Uses FastAPIWebsocketTransport with TwilioFrameSerializer or ExotelFrameSerializer.
DTMFAggregator converts DTMF keypresses into TranscriptionFrame("DTMF: 1") for LLM.

Conversation flow:
  greeting → confirm_identity → overdue_info → payment_check
    → payment_details (if already paid) → payment_confirmed [outcome=payment_confirmed]
    → payment_intent → commitment → ptp_end  [outcome=ptp]
    → callback_end                           [outcome=callback]
  → wrong_person_end                         [outcome=wrong_person]
"""

import asyncio
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

from deepgram import LiveOptions

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    BotStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import (
    TTFBMetricsData, ProcessingMetricsData, LLMUsageMetricsData, TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.tts import SarvamHttpTTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.exotel import ExotelFrameSerializer

from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector

from bot import _build_config, _make_role_message, create_stt, create_tts, create_llm

# Optional DB save
try:
    from database import save_conversation_sync as _db_save, CampaignCall, Campaign, SessionLocal
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Campaign call session state
# call_id (CampaignCall.id) → session data + asyncio.Event
# ---------------------------------------------------------------------------

campaign_sessions: dict = {}   # call_id -> session dict
campaign_events: dict = {}     # call_id -> asyncio.Event (set on disconnect)


async def _save_campaign_conversation(call_id: int, data: dict):
    """Save conversation to DB and update CampaignCall record."""
    if not _DB_AVAILABLE:
        return
    try:
        await asyncio.to_thread(_db_save_campaign_sync, call_id, data)
    except Exception as e:
        logger.warning(f"DB save failed for campaign call {call_id}: {e}")


def _db_save_campaign_sync(call_id: int, data: dict):
    import json
    from datetime import datetime, timezone

    db = SessionLocal()
    try:
        cc = db.query(CampaignCall).filter(CampaignCall.id == call_id).first()
        if not cc:
            return

        outcome = data.get("outcome")
        start_time = data.get("start_time")
        end_time = data.get("end_time", datetime.now(timezone.utc).timestamp())
        duration = end_time - start_time if start_time else None

        cc.call_status = "completed"
        cc.outcome = outcome
        cc.payment_made = outcome == "payment_confirmed"
        cc.payment_amount = data.get("payment_amount")
        cc.has_receipt = data.get("has_receipt")
        cc.ended_at = datetime.fromtimestamp(end_time, tz=timezone.utc)
        cc.duration_seconds = duration

        # Update campaign counters
        camp = db.query(Campaign).filter(Campaign.id == cc.campaign_id).first()
        if camp:
            camp.calls_completed += 1
            if outcome == "ptp":
                camp.ptp_count += 1
            elif outcome == "payment_confirmed":
                camp.payment_confirmed_count += 1

        db.commit()
        logger.info(f"CampaignCall {call_id} saved: outcome={outcome}")
    except Exception as e:
        logger.error(f"Failed to save campaign call {call_id}: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Metrics observer (reuses same logic as bot.py SessionMetricsObserver)
# ---------------------------------------------------------------------------

class CampaignMetricsObserver(BaseObserver):
    """Captures pipecat MetricsFrame data + turn latency into campaign_sessions."""

    _MAX = 200

    def __init__(self, call_id: int):
        super().__init__()
        self._call_id = call_id
        self._seen_ids: set = set()
        self._user_stopped_at: float = 0.0

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, MetricsFrame):
            if frame.id in self._seen_ids:
                return
            self._seen_ids.add(frame.id)
            self._handle_metrics(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_stopped_at = time.monotonic()
        elif isinstance(frame, BotStartedSpeakingFrame) and self._user_stopped_at:
            latency = round(time.monotonic() - self._user_stopped_at, 3)
            self._user_stopped_at = 0.0
            self._append("turn_latency", latency)

    def _m(self) -> dict:
        sd = campaign_sessions.get(self._call_id)
        if sd is None:
            return {}
        return sd.setdefault("metrics", {
            "ttfb": [], "processing": [], "llm_tokens": [],
            "tts_chars": [], "turn_latency": [],
        })

    def _append(self, key: str, value):
        m = self._m()
        if m and len(m.get(key, [])) < self._MAX:
            m[key].append(value)

    def _handle_metrics(self, frame: MetricsFrame):
        m = self._m()
        if not m:
            return
        for item in frame.data:
            if isinstance(item, TTFBMetricsData):
                if len(m["ttfb"]) < self._MAX:
                    m["ttfb"].append({"p": item.processor, "model": item.model, "v": round(item.value, 4)})
            elif isinstance(item, ProcessingMetricsData):
                if len(m["processing"]) < self._MAX:
                    m["processing"].append({"p": item.processor, "v": round(item.value, 4)})
            elif isinstance(item, LLMUsageMetricsData):
                if len(m["llm_tokens"]) < self._MAX:
                    tok = item.value
                    m["llm_tokens"].append({"p": item.processor, "model": item.model, "in": tok.prompt_tokens, "out": tok.completion_tokens})
            elif isinstance(item, TTSUsageMetricsData):
                if len(m["tts_chars"]) < self._MAX:
                    m["tts_chars"].append({"p": item.processor, "chars": item.value})


def _summarize_metrics(raw: dict) -> dict:
    def avg(lst, key):
        vals = [x[key] for x in lst if key in x]
        return round(sum(vals) / len(vals), 4) if vals else None

    def by_type(lst, keyword):
        return [x for x in lst if keyword.lower() in x.get("p", "").lower()]

    ttfb = raw.get("ttfb", [])
    llm_tokens = raw.get("llm_tokens", [])
    tts = raw.get("tts_chars", [])
    return {
        "stt_ttfb_avg": avg(by_type(ttfb, "STT") or by_type(ttfb, "Deepgram") or by_type(ttfb, "Sarvam"), "v"),
        "llm_ttfb_avg": avg(by_type(ttfb, "LLM") or by_type(ttfb, "OpenAI") or by_type(ttfb, "Ollama"), "v"),
        "tts_ttfb_avg": avg(by_type(ttfb, "TTS"), "v"),
        "tokens_in":  sum(x.get("in", 0) for x in llm_tokens),
        "tokens_out": sum(x.get("out", 0) for x in llm_tokens),
        "llm_calls":  len(llm_tokens),
        "tts_chars":  sum(x.get("chars", 0) for x in tts),
        "turn_latency": raw.get("turn_latency", []),
    }


# ---------------------------------------------------------------------------
# Transcript capture processors
# ---------------------------------------------------------------------------

class UserTranscriptCapture(FrameProcessor):
    def __init__(self, call_id: int):
        super().__init__()
        self._call_id = call_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            if self._call_id in campaign_sessions:
                campaign_sessions[self._call_id].setdefault("transcript", []).append(
                    {"role": "user", "text": frame.text.strip()}
                )
        await self.push_frame(frame, direction)


class AssistantTranscriptCapture(FrameProcessor):
    """Captures each LLM response turn individually using LLMFullResponse frames."""

    def __init__(self, call_id: int):
        super().__init__()
        self._call_id = call_id
        self._buf = ""
        self._capturing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._buf = ""
            self._capturing = True
        elif isinstance(frame, TextFrame) and self._capturing and frame.text:
            self._buf += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buf.strip() and self._call_id in campaign_sessions:
                campaign_sessions[self._call_id].setdefault("transcript", []).append(
                    {"role": "assistant", "text": self._buf.strip()}
                )
            self._buf = ""
            self._capturing = False
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Flow nodes
# ---------------------------------------------------------------------------

def _make_campaign_system_prompt(cfg: dict) -> dict:
    """System prompt that includes DTMF instructions."""
    base = _make_role_message(cfg)
    dtmf_note = (
        "\n\nDTMF INPUT:\n"
        "When you see 'DTMF: 1' in the conversation, treat it as the borrower pressing 1 (yes/haan).\n"
        "'DTMF: 2' means no/nahi. Respond naturally as if they spoke the answer."
    )
    base["content"] += dtmf_note
    return base


def create_campaign_greeting_node(cfg: dict) -> NodeConfig:
    bname = cfg["borrower_name"]
    agent = cfg["agent_name"]
    company = cfg["company_name"]

    async def confirm_identity(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "overdue_info", campaign_sessions)
        return "Identity confirmed", create_campaign_overdue_info_node(cfg)

    async def wrong_person(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        call_id = flow_manager.state.get("call_id")
        if call_id and call_id in campaign_sessions:
            campaign_sessions[call_id]["outcome"] = "wrong_person"
        _track_node(flow_manager, "wrong_person_end", campaign_sessions)
        return "Wrong person", create_wrong_person_end_node(cfg)

    return NodeConfig(
        name="greeting",
        messages=[_make_campaign_system_prompt(cfg)],
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"You just called {bname}. Greet them professionally. "
                    f'Say: "Namaste, main {agent} bol rahi hoon {company} se. '
                    f'Kya main {bname} ji se baat kar sakti hoon?" '
                    f"If it is them, use confirm_identity. "
                    f"If someone else answers, use wrong_person."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_identity",
                handler=confirm_identity,
                description="The person confirms they are the borrower",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="wrong_person",
                handler=wrong_person,
                description="Someone else answered (not the borrower)",
                properties={},
                required=[],
            ),
        ],
    )


def create_campaign_overdue_info_node(cfg: dict) -> NodeConfig:
    bname = cfg["borrower_name"]
    emi = cfg["emi_amount"]
    due = cfg["total_due"]
    months = cfg["overdue_months"]
    period = cfg["overdue_period"]
    fee = cfg["late_fee"]

    async def borrower_responds(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "payment_check", campaign_sessions)
        return "Borrower acknowledged", create_payment_check_node(cfg)

    return NodeConfig(
        name="overdue_info",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Inform {bname} about their overdue. Say something like: "
                    f"Aapka {months} mahine ka EMI ({period}) pending hai. "
                    f"EMI Rs. {emi} + late fee Rs. {fee} = total Rs. {due} baaki hai. "
                    f"Once they respond, use borrower_responds."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="borrower_responds",
                handler=borrower_responds,
                description="Borrower has heard and responded to the overdue information",
                properties={},
                required=[],
            ),
        ],
    )


def create_payment_check_node(cfg: dict) -> NodeConfig:
    """Ask if the borrower has already made payment."""
    bname = cfg["borrower_name"]

    async def payment_already_made(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "payment_details", campaign_sessions)
        return "Borrower says payment made", create_payment_details_node(cfg)

    async def not_paid_yet(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "payment_intent", campaign_sessions)
        return "Borrower has not paid", create_payment_intent_node(cfg)

    return NodeConfig(
        name="payment_check",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Ask {bname}: 'Kya aapne recently koi payment ki hai?' "
                    f"If yes → payment_already_made. If no → not_paid_yet."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="payment_already_made",
                handler=payment_already_made,
                description="Borrower claims they have already made payment",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="not_paid_yet",
                handler=not_paid_yet,
                description="Borrower has not made any payment",
                properties={},
                required=[],
            ),
        ],
    )


def create_payment_details_node(cfg: dict) -> NodeConfig:
    """Collect payment amount and receipt confirmation via speech or DTMF."""
    bname = cfg["borrower_name"]

    async def record_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        amount = args.get("amount", "")
        has_receipt = args.get("has_receipt", False)
        call_id = flow_manager.state.get("call_id")
        if call_id and call_id in campaign_sessions:
            campaign_sessions[call_id]["outcome"] = "payment_confirmed"
            campaign_sessions[call_id]["payment_amount"] = amount
            campaign_sessions[call_id]["has_receipt"] = has_receipt
        _track_node(flow_manager, "payment_confirmed_end", campaign_sessions)
        return f"Payment recorded: {amount}, receipt={has_receipt}", create_payment_confirmed_end_node(cfg)

    return NodeConfig(
        name="payment_details",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Ask {bname}: 'Aapne kitna payment kiya?' Wait for their answer. "
                    f"Then ask: 'Kya aapke paas receipt hai? Press 1 for haan, 2 for nahi.' "
                    f"If they say yes/haan OR you see 'DTMF: 1', has_receipt=True. "
                    f"If they say no/nahi OR you see 'DTMF: 2', has_receipt=False. "
                    f"Then call record_payment with the amount and has_receipt."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="record_payment",
                handler=record_payment,
                description="Record the payment amount and whether borrower has a receipt",
                properties={
                    "amount": {
                        "type": "string",
                        "description": "Payment amount mentioned by borrower e.g. '5000' or 'Rs. 5,000'",
                    },
                    "has_receipt": {
                        "type": "boolean",
                        "description": "True if borrower has receipt (DTMF 1 or spoken yes), False otherwise",
                    },
                },
                required=["amount", "has_receipt"],
            ),
        ],
    )


def create_payment_intent_node(cfg: dict) -> NodeConfig:
    """For borrowers who haven't paid — offer options."""
    bname = cfg["borrower_name"]
    emi = cfg["emi_amount"]
    due = cfg["total_due"]

    async def select_full_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = f"Full payment of Rs. {due}"
        _track_node(flow_manager, "commitment", campaign_sessions)
        return f"Full payment: {due}", create_commitment_node(cfg)

    async def select_split_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = f"Split: Rs. {emi} now + rest in 15 days"
        _track_node(flow_manager, "commitment", campaign_sessions)
        return f"Split payment", create_commitment_node(cfg)

    async def select_partial_plan(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Partial payment + installment plan"
        _track_node(flow_manager, "commitment", campaign_sessions)
        return "Partial plan", create_commitment_node(cfg)

    async def request_callback(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        call_id = flow_manager.state.get("call_id")
        if call_id and call_id in campaign_sessions:
            campaign_sessions[call_id]["outcome"] = "callback"
        _track_node(flow_manager, "callback_end", campaign_sessions)
        return "Callback requested", create_callback_end_node(cfg)

    return NodeConfig(
        name="payment_intent",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Offer {bname} payment options:\n"
                    f"1. Full payment Rs. {due} → select_full_payment\n"
                    f"2. Split: Rs. {emi} now + rest in 15 days → select_split_payment\n"
                    f"3. Partial + installments → select_partial_plan\n"
                    f"4. Request senior callback → request_callback\n"
                    f"Listen carefully and call the right function."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="select_full_payment",
                handler=select_full_payment,
                description="Borrower agrees to pay full amount",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="select_split_payment",
                handler=select_split_payment,
                description=f"Borrower wants to pay Rs. {emi} now and rest in 15 days",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="select_partial_plan",
                handler=select_partial_plan,
                description="Borrower wants partial payment with installments",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="request_callback",
                handler=request_callback,
                description="Borrower requests a callback from senior representative",
                properties={},
                required=[],
            ),
        ],
    )


def create_commitment_node(cfg: dict) -> NodeConfig:
    bname = cfg["borrower_name"]

    async def confirm_commitment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        date = args.get("payment_date", "not specified")
        flow_manager.state["payment_date"] = date
        plan = flow_manager.state.get("plan", "")
        call_id = flow_manager.state.get("call_id")
        if call_id and call_id in campaign_sessions:
            campaign_sessions[call_id]["outcome"] = "ptp"
            campaign_sessions[call_id]["payment_plan"] = plan
            campaign_sessions[call_id]["payment_date"] = date
        _track_node(flow_manager, "ptp_end", campaign_sessions)
        return f"PTP: {plan} by {date}", create_ptp_end_node(cfg)

    return NodeConfig(
        name="commitment",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Ask {bname} for a specific date to complete payment. "
                    f"Once they give a date, call confirm_commitment."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_commitment",
                handler=confirm_commitment,
                description="Borrower commits to a specific payment date",
                properties={
                    "payment_date": {
                        "type": "string",
                        "description": "Date the borrower commits to pay",
                    }
                },
                required=["payment_date"],
            ),
        ],
    )


def create_ptp_end_node(cfg: dict) -> NodeConfig:
    return NodeConfig(
        name="ptp_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the Promise to Pay warmly and close the call. "
                    "Thank the borrower and end on a positive note."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_payment_confirmed_end_node(cfg: dict) -> NodeConfig:
    return NodeConfig(
        name="payment_confirmed_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Thank the borrower for the payment. Confirm it will reflect in 2-3 working days. "
                    "Close the call warmly."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_wrong_person_end_node(cfg: dict) -> NodeConfig:
    return NodeConfig(
        name="wrong_person_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Apologize politely for the wrong call. "
                    'Say: "Maafi chahti hoon aapko disturb karne ke liye. Aapka din accha ho!"'
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_callback_end_node(cfg: dict) -> NodeConfig:
    return NodeConfig(
        name="callback_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the callback request. Say a senior will call within 24 hours. "
                    "Close the call warmly."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


# ---------------------------------------------------------------------------
# Flow tracking helper
# ---------------------------------------------------------------------------

def _track_node(flow_manager: FlowManager, node_name: str, sessions: dict):
    call_id = flow_manager.state.get("call_id")
    if call_id and call_id in sessions:
        sessions[call_id]["current_node"] = node_name
        logger.info(f"Campaign flow -> {node_name} (call_id={call_id})")


# ---------------------------------------------------------------------------
# Main bot entry point
# ---------------------------------------------------------------------------

async def run_campaign_bot(
    websocket,
    call_id: int,
    customer: dict,
    stt_type: str = "deepgram",
    tts_type: str = "openai",
    llm_type: str = "openai",
    provider: str = "twilio",
    aiohttp_session=None,
):
    """
    Run the outbound call bot over a WebSocket connection.

    websocket: FastAPI WebSocket object
    call_id:   CampaignCall.id from DB (used to update records on completion)
    customer:  dict with borrower details (name, phone, emi_amount, etc.)
    provider:  "twilio" or "exotel" — selects the frame serializer
    """
    cfg = _build_config(customer)
    logger.info(
        f"=== CampaignBot: call_id={call_id}, provider={provider}, "
        f"STT={stt_type}, TTS={tts_type}, LLM={llm_type} ==="
    )
    logger.info(f"Borrower: {cfg['borrower_name']} | {cfg['account_number']}")

    # --- Read initial WebSocket messages to extract stream_sid / call_sid ---
    # Twilio sends: {"event":"connected"} then {"event":"start","streamSid":"...","start":{"callSid":"..."}}
    # Exotel sends: {"event":"start","streamSid":"...","start":{"callSid":"..."}}
    import json as _json
    stream_sid = ""
    call_sid_ws = ""
    for _ in range(5):  # read up to 5 messages looking for "start"
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
            msg = _json.loads(raw)
            if msg.get("event") == "start":
                stream_sid = msg.get("streamSid", "") or msg.get("stream_sid", "")
                call_sid_ws = (msg.get("start") or {}).get("callSid", "")
                logger.info(f"Got stream_sid={stream_sid}, call_sid={call_sid_ws}")
                break
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Error reading initial WS message: {e}")
            break

    if not stream_sid:
        logger.warning("stream_sid not found in initial messages — using placeholder")
        stream_sid = f"stream-{call_id}"

    # --- Serializer selection ---
    if provider == "exotel":
        serializer = ExotelFrameSerializer(stream_sid=stream_sid, call_sid=call_sid_ws or None)
    else:
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid_ws or None,
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        )

    # --- Transport ---
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # --- Services ---
    stt = create_stt(stt_type)
    tts = create_tts(tts_type, aiohttp_session=aiohttp_session)
    llm = create_llm(llm_type)

    # --- Context ---
    context = OpenAILLMContext([])
    context_aggregator = llm.create_context_aggregator(context)

    # --- Capture processors ---
    user_capture = UserTranscriptCapture(call_id)
    assistant_capture = AssistantTranscriptCapture(call_id)

    # --- Voicemail detection (outbound calls only) ---
    cfg = _build_config(customer)
    vmd_llm = create_llm(llm_type)   # separate LLM instance for classifier
    voicemail_detector = VoicemailDetector(
        llm=vmd_llm,
        voicemail_response_delay=float(os.getenv("VMD_DELAY_SECS", "2.5")),
    )

    @voicemail_detector.event_handler("on_voicemail_detected")
    async def on_voicemail(processor):
        logger.info(f"[call_id={call_id}] Voicemail detected — leaving automated message")
        from pipecat.frames.frames import TTSSpeakFrame
        msg = (
            f"Namaste, yeh {cfg['company_name']} ki taraf se {cfg['borrower_name']} ji ke liye "
            f"ek zaroori message hai. Aapke {cfg['overdue_months']} EMIs pending hain. "
            f"Kripya jaldi se humse sampark karein. Dhanyavaad."
        )
        await processor.push_frame(TTSSpeakFrame(msg))

    @voicemail_detector.event_handler("on_conversation_detected")
    async def on_human(processor):
        logger.info(f"[call_id={call_id}] Human answered — proceeding with collection flow")

    # --- Pipeline with DTMFAggregator + voicemail detection ---
    pipeline = Pipeline([
        transport.input(),
        stt,
        voicemail_detector.detector(),
        DTMFAggregator(),
        user_capture,
        context_aggregator.user(),
        llm,
        assistant_capture,
        tts,
        voicemail_detector.gate(),
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            MetricsLogObserver(),
            CampaignMetricsObserver(call_id),
        ],
        idle_timeout_secs=180,
    )

    # --- Flow Manager ---
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # --- Session state ---
    campaign_sessions[call_id] = {
        "current_node": "greeting",
        "start_time": time.time(),
        "stt_type": stt_type,
        "tts_type": tts_type,
        "llm_type": llm_type,
        "transcript": [],
        "outcome": None,
        "payment_amount": None,
        "has_receipt": None,
        "payment_plan": None,
        "payment_date": None,
        "agent_config": cfg,
        "metrics": {
            "ttfb": [], "processing": [], "llm_tokens": [],
            "tts_chars": [], "turn_latency": [],
        },
    }

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Campaign call connected (call_id={call_id})")
        flow_manager.state["call_id"] = call_id
        await flow_manager.initialize(create_campaign_greeting_node(cfg))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Campaign call disconnected (call_id={call_id})")
        data = dict(campaign_sessions.pop(call_id, {}))
        if data:
            if not data.get("outcome"):
                data["outcome"] = "incomplete"
            data["end_time"] = time.time()
            raw_metrics = data.pop("metrics", {})
            data["metrics_summary"] = _summarize_metrics(raw_metrics)
            asyncio.create_task(_save_campaign_conversation(call_id, data))

        # Signal campaign runner that call is done
        event = campaign_events.get(call_id)
        if event:
            event.set()

        await task.queue_frames([EndFrame()])

    # --- Run ---
    runner = PipelineRunner()
    await runner.run(task)
