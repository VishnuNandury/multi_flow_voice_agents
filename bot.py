#
# Loan Collection Voice Agent - Multi-Pipeline with PipeCat Flows
# ================================================================
# Structured conversation flow for loan collection.
# Stages: greeting -> overdue_info -> understand_situation
#        -> payment_options -> commitment -> promise_to_pay -> end
#
# Pipeline configs (STT/TTS/LLM) are passed per-session from the server.
# Agent persona and borrower details are configurable from the dashboard.
#

import os
import sys
import time

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

from deepgram import LiveOptions

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from edge_tts_service import EdgeTTSService
from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig


# ---------------------------------------------------------------------------
# Configurable agent persona + borrower details
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "agent_name": "Priya",
    "company_name": "QuickFinance Ltd.",
    "borrower_name": "Rajesh Kumar",
    "account_number": "QF-2024-78432",
    "loan_type": "Personal Loan",
    "emi_amount": "5,280",
    "overdue_months": "2",
    "overdue_period": "Dec 2025, Jan 2026",
    "late_fee": "1,200",
    "total_due": "11,760",
    "last_payment": "Nov 28, 2025",
    "language": "hinglish",
}


def _build_config(agent_config: dict = None) -> dict:
    """Merge user-provided config with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if agent_config:
        for k, v in agent_config.items():
            if v:  # only override with non-empty values
                cfg[k] = v
    return cfg


def _make_role_message(cfg: dict) -> dict:
    """Build the system role message dynamically from config."""
    lang_instructions = {
        "hindi": (
            "- Always speak in Hindi (Devanagari transliteration).\n"
            "- Do not use English unless the borrower does."
        ),
        "english": (
            "- Always speak in English.\n"
            "- Use simple, clear English suitable for Indian borrowers."
        ),
        "hinglish": (
            "- Speak in the SAME language the borrower uses.\n"
            "- If they speak Hindi, respond in Hindi.\n"
            "- If they speak English, respond in English.\n"
            "- If they mix Hindi and English (Hinglish), you also mix naturally.\n"
            "- Default to Hinglish as it feels most natural for Indian borrowers."
        ),
    }
    lang_rule = lang_instructions.get(cfg["language"], lang_instructions["hinglish"])

    return {
        "role": "system",
        "content": (
            f'You are "{cfg["agent_name"]}", a professional and empathetic loan collection agent '
            f'working for "{cfg["company_name"]}"\n\n'
            f"LANGUAGE RULES:\n{lang_rule}\n\n"
            "STYLE:\n"
            "- Warm, professional, empathetic — NEVER threatening or rude.\n"
            "- Short, natural sentences (this is a voice call, not text).\n"
            "- No bullet points, markdown, special characters, or emojis.\n"
            "- Max 2-3 sentences at a time.\n"
            "- You must ALWAYS use one of the available functions to progress the conversation.\n"
            "- Your responses will be converted to audio so avoid any formatting."
        ),
    }


def _make_borrower_info(cfg: dict) -> str:
    """Build borrower info string from config."""
    return (
        f"Borrower: {cfg['borrower_name']} | Account: {cfg['account_number']} | {cfg['loan_type']}\n"
        f"EMI: Rs. {cfg['emi_amount']} | Overdue: {cfg['overdue_months']} months ({cfg['overdue_period']}) | "
        f"Late Fee: Rs. {cfg['late_fee']} | Total Due: Rs. {cfg['total_due']}\n"
        f"Last Payment: {cfg['last_payment']}"
    )


# ---------------------------------------------------------------------------
# Flow state + session tracking — exposed to app.py for dashboard API
# ---------------------------------------------------------------------------

FLOW_NODES = [
    {"id": "greeting", "label": "Greeting"},
    {"id": "overdue_info", "label": "Overdue Info"},
    {"id": "understand_situation", "label": "Situation"},
    {"id": "payment_options", "label": "Options"},
    {"id": "commitment", "label": "Commitment"},
    {"id": "promise_to_pay", "label": "PTP"},
    {"id": "end", "label": "Complete"},
]

# pc_id -> {current_node, start_time, stt_type, tts_type, llm_type, transcript}
session_data: dict = {}


def _track_node(flow_manager: FlowManager, node_name: str):
    """Update the current flow node for the visualization dashboard."""
    pc_id = flow_manager.state.get("pc_id", "")
    if pc_id and pc_id in session_data:
        session_data[pc_id]["current_node"] = node_name
        logger.info(f"Flow -> {node_name}")


def _add_transcript(pc_id: str, role: str, text: str):
    """Add a transcript entry to the session data."""
    if pc_id in session_data and text.strip():
        session_data[pc_id].setdefault("transcript", []).append(
            {"role": role, "text": text.strip()}
        )


# ---------------------------------------------------------------------------
# Transcript Capture Processors
# ---------------------------------------------------------------------------

class UserTranscriptCapture(FrameProcessor):
    """Capture user speech transcriptions (placed after STT in pipeline)."""

    def __init__(self, pc_id: str):
        super().__init__()
        self._pc_id = pc_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            _add_transcript(self._pc_id, "user", frame.text)
        await self.push_frame(frame, direction)


class AssistantTranscriptCapture(FrameProcessor):
    """Capture assistant LLM response text (placed after LLM in pipeline)."""

    def __init__(self, pc_id: str):
        super().__init__()
        self._pc_id = pc_id
        self._buffer = ""
        self._capturing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            self._capturing = True
        elif isinstance(frame, TextFrame) and self._capturing:
            self._buffer += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer.strip():
                _add_transcript(self._pc_id, "assistant", self._buffer)
            self._buffer = ""
            self._capturing = False
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Flow Node Definitions (all accept cfg dict for dynamic personalization)
# ---------------------------------------------------------------------------

def create_greeting_node(cfg: dict) -> NodeConfig:
    """Node 1: Greet and confirm identity."""
    role_msg = _make_role_message(cfg)
    info = _make_borrower_info(cfg)
    bname = cfg["borrower_name"]
    aname = cfg["agent_name"]
    cname = cfg["company_name"]

    async def confirm_identity(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["identity_confirmed"] = True
        _track_node(flow_manager, "overdue_info")
        return f"Identity confirmed as {bname}", create_overdue_info_node(cfg)

    async def wrong_person(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "end")
        return "Wrong person on the line", create_wrong_person_end_node(cfg)

    return NodeConfig(
        name="greeting",
        role_messages=[role_msg],
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Greet warmly in Hinglish. Say something like: "
                    f'"Namaste, kya main {bname} ji se baat kar rahi hoon? '
                    f'Main {aname} bol rahi hoon, {cname} ki taraf se."\n\n'
                    f"Wait for their response. Use confirm_identity if they confirm "
                    f"(even partially), or wrong_person if they deny.\n\n"
                    f"{info}"
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_identity",
                handler=confirm_identity,
                description=f"Person confirms they are {bname} or acknowledges their identity",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="wrong_person",
                handler=wrong_person,
                description=f"Person says they are NOT {bname} or denies their identity",
                properties={},
                required=[],
            ),
        ],
    )


def create_overdue_info_node(cfg: dict) -> NodeConfig:
    """Node 2: Inform about overdue EMIs."""
    info = _make_borrower_info(cfg)
    bname = cfg["borrower_name"]

    async def borrower_responds(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "understand_situation")
        return "Borrower acknowledged overdue information", create_situation_node(cfg)

    return NodeConfig(
        name="overdue_info",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Politely inform about overdue EMIs. Say something like: "
                    f'"{bname} ji, main aapko ek zaroori baat batana chahti thi. '
                    f"Aapke {cfg['overdue_months']} EMIs pending hain, {cfg['overdue_period']} ke. "
                    f'Total Rs. {cfg["total_due"]} outstanding hai including late fee."\n\n'
                    f"Be gentle and empathetic. After they respond in any way, "
                    f"use borrower_responds to move forward.\n\n"
                    f"{info}"
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="borrower_responds",
                handler=borrower_responds,
                description="Borrower responds to the overdue information",
                properties={},
                required=[],
            ),
        ],
    )


def create_situation_node(cfg: dict) -> NodeConfig:
    """Node 3: Understand borrower's situation."""
    bname = cfg["borrower_name"]

    async def record_situation(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        reason = args.get("reason", "not specified")
        flow_manager.state["reason"] = reason
        _track_node(flow_manager, "payment_options")
        return f"Borrower's reason: {reason}", create_payment_options_node(cfg)

    return NodeConfig(
        name="understand_situation",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Ask empathetically about their situation. Say something like: "
                    f'"Main samajh sakti hoon {bname} ji. Kya aap bata sakte hain ki '
                    f"koi specific wajah thi EMI miss hone ki? "
                    f'Main aapki help karna chahti hoon."\n\n'
                    f"Listen with empathy, then use record_situation to move to payment options."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="record_situation",
                handler=record_situation,
                description="Record the borrower's reason for delayed payment",
                properties={
                    "reason": {
                        "type": "string",
                        "description": "Brief summary of why the borrower missed payments",
                    }
                },
                required=["reason"],
            ),
        ],
    )


def create_payment_options_node(cfg: dict) -> NodeConfig:
    """Node 4: Present payment options."""
    bname = cfg["borrower_name"]
    total = cfg["total_due"]
    emi = cfg["emi_amount"]

    async def select_full_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = f"Full payment of Rs. {total}"
        _track_node(flow_manager, "commitment")
        return "Full payment selected", create_commitment_node(cfg)

    async def select_split_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = f"Rs. {emi} now + remaining in 15 days"
        _track_node(flow_manager, "commitment")
        return "Split payment plan selected", create_commitment_node(cfg)

    async def select_partial_plan(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Rs. 4,000 now + remaining in 2 installments"
        _track_node(flow_manager, "commitment")
        return "Partial payment plan selected", create_commitment_node(cfg)

    async def request_callback(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Callback requested"
        _track_node(flow_manager, "end")
        return "Senior representative callback requested", create_callback_end_node(cfg)

    return NodeConfig(
        name="payment_options",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Present payment options naturally and sympathetically. "
                    f"Do NOT read them as a numbered list. Weave them into conversation.\n\n"
                    f"Options available:\n"
                    f"- Full Rs. {total} payment right away (late fee discount possible)\n"
                    f"- Pay one EMI Rs. {emi} now, remaining within 15 days\n"
                    f"- Rs. 4,000 now, remaining in 2 easy installments\n"
                    f"- Request a callback from senior representative for restructuring\n\n"
                    f"Recommend based on what the borrower has shared. "
                    f"Use the matching function when they choose."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="select_full_payment",
                handler=select_full_payment,
                description=f"Borrower agrees to pay full Rs. {total} immediately",
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
                description="Borrower wants to pay Rs. 4,000 now and rest in installments",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="request_callback",
                handler=request_callback,
                description="Borrower requests a callback from a senior representative",
                properties={},
                required=[],
            ),
        ],
    )


def create_commitment_node(cfg: dict) -> NodeConfig:
    """Node 5: Get payment commitment with a specific date."""
    bname = cfg["borrower_name"]

    async def confirm_commitment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        date = args.get("payment_date", "not specified")
        flow_manager.state["payment_date"] = date
        plan = flow_manager.state.get("plan", "")
        _track_node(flow_manager, "promise_to_pay")
        return f"Payment commitment: {plan} by {date}", create_promise_to_pay_node(cfg)

    return NodeConfig(
        name="commitment",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Confirm the chosen payment plan and ask for a specific date. "
                    f"Say something like: "
                    f'"Bahut accha {bname} ji! Kya aap mujhe ek specific date bata sakte hain '
                    f'jab tak aap payment kar denge?"\n\n'
                    f"Once they give a date, use confirm_commitment."
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
                        "description": "The date the borrower commits to make payment",
                    }
                },
                required=["payment_date"],
            ),
        ],
    )


def create_promise_to_pay_node(cfg: dict) -> NodeConfig:
    """Node 6: Formal Promise to Pay (PTP) confirmation."""
    bname = cfg["borrower_name"]

    async def confirm_ptp(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        plan = flow_manager.state.get("plan", "")
        date = flow_manager.state.get("payment_date", "")
        logger.info(f"PTP confirmed: {plan} by {date}")
        _track_node(flow_manager, "end")
        return f"PTP confirmed: {plan} by {date}", create_end_node(cfg)

    async def revise_plan(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "payment_options")
        return "Borrower wants to revise the plan", create_payment_options_node(cfg)

    return NodeConfig(
        name="promise_to_pay",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Formally confirm the Promise to Pay. Summarize the commitment clearly. "
                    f"Say something like: "
                    f'"{bname} ji, toh main confirm kar rahi hoon — aap [plan details] '
                    f"[date] tak kar denge. Kya aap is commitment ko confirm karte hain? "
                    f'Yeh aapka Promise to Pay hoga."\n\n'
                    f"If they confirm, use confirm_ptp. "
                    f"If they want to change, use revise_plan."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_ptp",
                handler=confirm_ptp,
                description="Borrower formally confirms their Promise to Pay",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="revise_plan",
                handler=revise_plan,
                description="Borrower wants to choose a different payment plan",
                properties={},
                required=[],
            ),
        ],
    )


def create_end_node(cfg: dict) -> NodeConfig:
    """Final node: Thank the borrower and close."""
    bname = cfg["borrower_name"]
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Thank the borrower warmly and close the call. Summarize their commitment. "
                    f"Say something like: "
                    f'"Bahut bahut dhanyavaad {bname} ji! Main aapka Promise to Pay note kar rahi hoon. '
                    f"Aap UPI ya net banking se payment kar sakte hain. "
                    f'Aapka din shubh ho!"\n\n'
                    f"Be warm, professional, and end on a positive note."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_wrong_person_end_node(cfg: dict) -> NodeConfig:
    """End node when the person is not the borrower."""
    return NodeConfig(
        name="wrong_person_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Apologize politely. Say: "
                    '"Maafi chahti hoon aapko disturb karne ke liye. '
                    'Galti se call lag gayi. Aapka din accha ho!"'
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_callback_end_node(cfg: dict) -> NodeConfig:
    """End node when borrower requests a senior callback."""
    bname = cfg["borrower_name"]
    return NodeConfig(
        name="callback_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Confirm the callback request warmly. Say something like: "
                    f'"Bilkul {bname} ji, main aapki request note kar rahi hoon. '
                    f"Humare senior representative aapko 24 ghante mein call karenge. "
                    f'Dhanyavaad aapke time ke liye!"'
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


# ---------------------------------------------------------------------------
# Service factories (multi-pipeline)
# ---------------------------------------------------------------------------

def create_stt(stt_type: str):
    """Create STT service based on type."""
    if stt_type == "sarvam":
        logger.info("STT: Sarvam saarika:v2.5 (Hindi)")
        return SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY", ""),
            model="saarika:v2.5",
            params=SarvamSTTService.InputParams(language=Language.HI_IN),
        )
    elif stt_type == "whisper":
        logger.info("STT: OpenAI Whisper (gpt-4o-transcribe)")
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-transcribe",
            language="hi",
            prompt="Hindi and English (Hinglish) phone conversation about loan collection.",
        )
    else:
        logger.info("STT: Deepgram Nova-3 (Hindi)")
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                language="hi",
                model="nova-3",
                smart_format=True,
                encoding="linear16",
                sample_rate=16000,
                channels=1,
            ),
        )


def create_tts(tts_type: str):
    """Create TTS service based on type."""
    if tts_type == "sarvam":
        logger.info("TTS: Sarvam bulbul:v2 (anushka)")
        return SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY", ""),
            model="bulbul:v2",
            voice_id=os.getenv("SARVAM_TTS_VOICE", "anushka"),
            params=SarvamTTSService.InputParams(language=Language.HI),
        )
    elif tts_type == "edge":
        logger.info("TTS: Edge TTS (hi-IN-SwaraNeural)")
        return EdgeTTSService(
            voice=os.getenv("EDGE_TTS_VOICE", "hi-IN-SwaraNeural"),
            rate=os.getenv("EDGE_TTS_RATE", "+0%"),
        )
    else:
        logger.info("TTS: OpenAI TTS (shimmer)")
        return OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=os.getenv("OPENAI_TTS_VOICE", "shimmer"),
            instructions=(
                "You are Priya, a warm and professional Indian woman speaking on a phone call. "
                "Speak naturally in Hindi and Hinglish with clear pronunciation. "
                "Use a conversational, friendly tone. Do not rush."
            ),
        )


def create_llm(llm_type: str):
    """Create LLM service based on type.

    For 'ollama' type: tries Ollama first (OLLAMA_BASE_URL), falls back to
    Groq (GROQ_API_KEY) if Ollama is not configured.
    """
    if llm_type == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()

        if ollama_url:
            model = os.getenv("OLLAMA_MODEL", "llama3.1")
            logger.info(f"LLM: Ollama ({model}) at {ollama_url}")
            return OLLamaLLMService(model=model, base_url=ollama_url)
        elif groq_key:
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info(f"LLM: Groq fallback ({model})")
            return OpenAILLMService(
                api_key=groq_key,
                model=model,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            logger.warning("LLM: Neither OLLAMA_BASE_URL nor GROQ_API_KEY set, falling back to OpenAI")
            return OpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            )
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"LLM: OpenAI ({model})")
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
        )


# ---------------------------------------------------------------------------
# Bot Pipeline
# ---------------------------------------------------------------------------

async def run_bot(
    webrtc_connection: SmallWebRTCConnection,
    stt_type: str = "deepgram",
    tts_type: str = "openai",
    llm_type: str = "openai",
    agent_config: dict = None,
):
    """Create and run the voice agent pipeline with PipeCat Flows."""

    pc_id = webrtc_connection.pc_id
    cfg = _build_config(agent_config)
    logger.info(
        f"=== Pipeline: STT={stt_type} | TTS={tts_type} | LLM={llm_type} | pc_id={pc_id} ==="
    )
    logger.info(f"Agent: {cfg['agent_name']} @ {cfg['company_name']} | Borrower: {cfg['borrower_name']}")

    # --- Transport (WebRTC — no explicit sample rates, matches working Agent_pipecat) ---
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    # --- Services ---
    stt = create_stt(stt_type)
    tts = create_tts(tts_type)
    llm = create_llm(llm_type)

    # --- Context (empty — FlowManager populates it per node) ---
    context = OpenAILLMContext([])
    context_aggregator = llm.create_context_aggregator(context)

    # --- Transcript capture processors ---
    user_capture = UserTranscriptCapture(pc_id)
    assistant_capture = AssistantTranscriptCapture(pc_id)

    # --- Pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_capture,
            context_aggregator.user(),
            llm,
            assistant_capture,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # --- Flow Manager ---
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # --- Session tracking for the dashboard API ---
    session_data[pc_id] = {
        "current_node": "greeting",
        "start_time": time.time(),
        "stt_type": stt_type,
        "tts_type": tts_type,
        "llm_type": llm_type,
        "transcript": [],
    }

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected ({stt_type}+{tts_type}+{llm_type})")
        flow_manager.state["pc_id"] = pc_id
        await flow_manager.initialize(create_greeting_node(cfg))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected ({stt_type}+{tts_type}+{llm_type})")
        session_data.pop(pc_id, None)
        await task.queue_frames([EndFrame()])

    # --- Run ---
    runner = PipelineRunner()
    await runner.run(task)
