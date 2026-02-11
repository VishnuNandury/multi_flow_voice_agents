#
# Loan Collection Voice Agent - Multi-Pipeline with PipeCat Flows
# ================================================================
# Structured conversation flow for loan collection.
# Stages: greeting -> overdue_info -> understand_situation
#        -> payment_options -> commitment -> promise_to_pay -> end
#
# Pipeline configs (STT/TTS) are passed per-session from the server.
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
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from edge_tts_service import EdgeTTSService
from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig


# ---------------------------------------------------------------------------
# Priya persona — consistent across all flow nodes
# ---------------------------------------------------------------------------

ROLE_MESSAGE = {
    "role": "system",
    "content": (
        'You are "Priya", a professional and empathetic loan collection agent '
        'working for "QuickFinance Ltd."\n\n'
        "LANGUAGE RULES:\n"
        "- Speak in the SAME language the borrower uses.\n"
        "- If they speak Hindi, respond in Hindi.\n"
        "- If they speak English, respond in English.\n"
        "- If they mix Hindi and English (Hinglish), you also mix naturally.\n"
        "- Default to Hinglish as it feels most natural for Indian borrowers.\n\n"
        "STYLE:\n"
        "- Warm, professional, empathetic — NEVER threatening or rude.\n"
        "- Short, natural sentences (this is a voice call, not text).\n"
        "- No bullet points, markdown, special characters, or emojis.\n"
        "- Max 2-3 sentences at a time.\n"
        "- You must ALWAYS use one of the available functions to progress the conversation.\n"
        "- Your responses will be converted to audio so avoid any formatting."
    ),
}

BORROWER_INFO = (
    "Borrower: Rajesh Kumar | Account: QF-2024-78432 | Personal Loan\n"
    "EMI: Rs. 5,280 | Overdue: 2 months (Dec 2025, Jan 2026) | "
    "Late Fee: Rs. 1,200 | Total Due: Rs. 11,760\n"
    "Last Payment: Nov 28, 2025"
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

# pc_id -> {current_node, start_time, stt_type, tts_type, transcript}
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
# Flow Node Definitions
# ---------------------------------------------------------------------------

def create_greeting_node() -> NodeConfig:
    """Node 1: Greet and confirm identity."""

    async def confirm_identity(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["identity_confirmed"] = True
        _track_node(flow_manager, "overdue_info")
        return "Identity confirmed as Rajesh Kumar", create_overdue_info_node()

    async def wrong_person(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "end")
        return "Wrong person on the line", create_wrong_person_end_node()

    return NodeConfig(
        name="greeting",
        role_messages=[ROLE_MESSAGE],
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Greet warmly in Hinglish. Say something like: "
                    '"Namaste, kya main Rajesh Kumar ji se baat kar rahi hoon? '
                    'Main Priya bol rahi hoon, QuickFinance ki taraf se."\n\n'
                    "Wait for their response. Use confirm_identity if they confirm "
                    "(even partially), or wrong_person if they deny.\n\n"
                    f"{BORROWER_INFO}"
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_identity",
                handler=confirm_identity,
                description="Person confirms they are Rajesh Kumar or acknowledges their identity",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="wrong_person",
                handler=wrong_person,
                description="Person says they are NOT Rajesh Kumar or denies their identity",
                properties={},
                required=[],
            ),
        ],
    )


def create_overdue_info_node() -> NodeConfig:
    """Node 2: Inform about overdue EMIs."""

    async def borrower_responds(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "understand_situation")
        return "Borrower acknowledged overdue information", create_situation_node()

    return NodeConfig(
        name="overdue_info",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Politely inform about overdue EMIs. Say something like: "
                    '"Rajesh ji, main aapko ek zaroori baat batana chahti thi. '
                    "Aapke do EMIs pending hain, December aur January ke. "
                    'Total Rs. 11,760 outstanding hai including late fee."\n\n'
                    "Be gentle and empathetic. After they respond in any way, "
                    "use borrower_responds to move forward.\n\n"
                    f"{BORROWER_INFO}"
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


def create_situation_node() -> NodeConfig:
    """Node 3: Understand borrower's situation."""

    async def record_situation(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        reason = args.get("reason", "not specified")
        flow_manager.state["reason"] = reason
        _track_node(flow_manager, "payment_options")
        return f"Borrower's reason: {reason}", create_payment_options_node()

    return NodeConfig(
        name="understand_situation",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask empathetically about their situation. Say something like: "
                    '"Main samajh sakti hoon Rajesh ji. Kya aap bata sakte hain ki '
                    "koi specific wajah thi EMI miss hone ki? "
                    'Main aapki help karna chahti hoon."\n\n'
                    "Listen with empathy, then use record_situation to move to payment options."
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


def create_payment_options_node() -> NodeConfig:
    """Node 4: Present payment options."""

    async def select_full_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Full payment of Rs. 11,760"
        _track_node(flow_manager, "commitment")
        return "Full payment selected", create_commitment_node()

    async def select_split_payment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Rs. 5,280 now + Rs. 6,480 in 15 days"
        _track_node(flow_manager, "commitment")
        return "Split payment plan selected", create_commitment_node()

    async def select_partial_plan(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Rs. 4,000 now + remaining in 2 installments"
        _track_node(flow_manager, "commitment")
        return "Partial payment plan selected", create_commitment_node()

    async def request_callback(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        flow_manager.state["plan"] = "Callback requested"
        _track_node(flow_manager, "end")
        return "Senior representative callback requested", create_callback_end_node()

    return NodeConfig(
        name="payment_options",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Present payment options naturally and sympathetically. "
                    "Do NOT read them as a numbered list. Weave them into conversation.\n\n"
                    "Options available:\n"
                    "- Full Rs. 11,760 payment right away (late fee discount possible)\n"
                    "- Pay one EMI Rs. 5,280 now, remaining Rs. 6,480 within 15 days\n"
                    "- Rs. 4,000 now, remaining in 2 easy installments\n"
                    "- Request a callback from senior representative for restructuring\n\n"
                    "Recommend based on what the borrower has shared. "
                    "Use the matching function when they choose."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="select_full_payment",
                handler=select_full_payment,
                description="Borrower agrees to pay full Rs. 11,760 immediately",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="select_split_payment",
                handler=select_split_payment,
                description="Borrower wants to pay Rs. 5,280 now and rest in 15 days",
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


def create_commitment_node() -> NodeConfig:
    """Node 5: Get payment commitment with a specific date."""

    async def confirm_commitment(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        date = args.get("payment_date", "not specified")
        flow_manager.state["payment_date"] = date
        plan = flow_manager.state.get("plan", "")
        _track_node(flow_manager, "promise_to_pay")
        return f"Payment commitment: {plan} by {date}", create_promise_to_pay_node()

    return NodeConfig(
        name="commitment",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the chosen payment plan and ask for a specific date. "
                    "Say something like: "
                    '"Bahut accha Rajesh ji! Kya aap mujhe ek specific date bata sakte hain '
                    'jab tak aap payment kar denge?"\n\n'
                    "Once they give a date, use confirm_commitment."
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


def create_promise_to_pay_node() -> NodeConfig:
    """Node 6: Formal Promise to Pay (PTP) confirmation."""

    async def confirm_ptp(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        plan = flow_manager.state.get("plan", "")
        date = flow_manager.state.get("payment_date", "")
        logger.info(f"PTP confirmed: {plan} by {date}")
        _track_node(flow_manager, "end")
        return f"PTP confirmed: {plan} by {date}", create_end_node()

    async def revise_plan(args: FlowArgs, flow_manager: FlowManager) -> tuple:
        _track_node(flow_manager, "payment_options")
        return "Borrower wants to revise the plan", create_payment_options_node()

    return NodeConfig(
        name="promise_to_pay",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Formally confirm the Promise to Pay. Summarize the commitment clearly. "
                    "Say something like: "
                    '"Rajesh ji, toh main confirm kar rahi hoon — aap [plan details] '
                    "[date] tak kar denge. Kya aap is commitment ko confirm karte hain? "
                    'Yeh aapka Promise to Pay hoga."\n\n'
                    "If they confirm, use confirm_ptp. "
                    "If they want to change, use revise_plan."
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


def create_end_node() -> NodeConfig:
    """Final node: Thank the borrower and close."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Thank the borrower warmly and close the call. Summarize their commitment. "
                    "Say something like: "
                    '"Bahut bahut dhanyavaad Rajesh ji! Main aapka Promise to Pay note kar rahi hoon. '
                    "Aap UPI ya net banking se payment kar sakte hain. "
                    'Aapka din shubh ho!"\n\n'
                    "Be warm, professional, and end on a positive note."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_wrong_person_end_node() -> NodeConfig:
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


def create_callback_end_node() -> NodeConfig:
    """End node when borrower requests a senior callback."""
    return NodeConfig(
        name="callback_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the callback request warmly. Say something like: "
                    '"Bilkul Rajesh ji, main aapki request note kar rahi hoon. '
                    "Humare senior representative aapko 24 ghante mein call karenge. "
                    'Dhanyavaad aapke time ke liye!"'
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
    if stt_type == "whisper":
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
    if tts_type == "edge":
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


# ---------------------------------------------------------------------------
# Bot Pipeline
# ---------------------------------------------------------------------------

async def run_bot(
    webrtc_connection: SmallWebRTCConnection,
    stt_type: str = "deepgram",
    tts_type: str = "openai",
):
    """Create and run the voice agent pipeline with PipeCat Flows."""

    pc_id = webrtc_connection.pc_id
    logger.info(f"=== Pipeline: STT={stt_type} | TTS={tts_type} | pc_id={pc_id} ===")

    # --- Transport (WebRTC — no explicit sample rates, matches working Agent_pipecat) ---
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    # --- Services ---
    stt = create_stt(stt_type)
    tts = create_tts(tts_type)

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )

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
        "transcript": [],
    }

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected ({stt_type}+{tts_type})")
        flow_manager.state["pc_id"] = pc_id
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected ({stt_type}+{tts_type})")
        session_data.pop(pc_id, None)
        await task.queue_frames([EndFrame()])

    # --- Run ---
    runner = PipelineRunner()
    await runner.run(task)
