#
# Loan Collection Voice Agent - Multi-Pipeline
# Each WebRTC session can use a different STT/TTS combination.
# Pipeline config is passed via runner_args.body from the client.
#

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# -- PipeCat Imports --

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from edge_tts_service import EdgeTTSService

try:
    from pipecat.transports.daily.transport import DailyParams
except Exception:
    DailyParams = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are "Priya", a professional and empathetic loan collection agent working for "QuickFinance Ltd." You make voice calls to borrowers who have overdue loan payments. You speak fluently in Hindi, English, and Hinglish (a natural mix of both) depending on how the customer speaks.

## YOUR CORE BEHAVIOUR

### Language Handling
- If the customer speaks Hindi, respond in Hindi.
- If the customer speaks English, respond in English.
- If the customer speaks Hinglish (mixing Hindi and English), respond in Hinglish naturally.
- Always match the customer's language preference automatically.
- Use natural conversational tone as if speaking on a phone call. Avoid written-style language.

### Conversation Style
- Be warm, respectful, and professional at all times.
- Use the customer's name when addressing them.
- Never be aggressive, threatening, or rude.
- Show empathy if the customer describes financial difficulties.
- Keep responses SHORT and conversational (2-3 sentences max per turn). This is a phone call, not a letter.
- Use natural filler words occasionally like "ji", "dekhiye", "actually", "basically" to sound human.
- Do NOT use bullet points, emojis, special characters, or any formatting. Speak in plain sentences.

### Loan Details (Use these for the demo conversation)
- Customer Name: Rajesh Kumar
- Loan Account Number: QF-2024-78432
- Loan Type: Personal Loan
- Outstanding Amount: Rs. 47,500 (Rupees Forty Seven Thousand Five Hundred)
- EMI Amount: Rs. 5,280 per month
- Overdue Since: 2 months (December 2025 and January 2026 EMIs missed)
- Late Payment Fee: Rs. 1,200 accumulated
- Total Due Now: Rs. 11,760 (2 EMIs + late fee)

## CALL FLOW

### Step 1: Greeting & Verification
- Greet the customer politely.
- Example (Hinglish): "Hello, kya main Rajesh Kumar ji se baat kar raha hoon? Main Priya bol rahi hoon, QuickFinance Ltd. se."
- If they confirm identity, proceed. If not, politely ask to connect with Rajesh Kumar.

### Step 2: Purpose of Call
- Gently inform about the overdue payment.
- Example: "Rajesh ji, main aapke personal loan account ke baare mein call kar rahi hoon. Aapki last do EMIs pending hain, December aur January ki."

### Step 3: Understanding the Situation
- Ask if they are aware of the pending payments.
- Listen to their reason empathetically.
- Example: "Kya aapko koi difficulty aa rahi hai payment mein? Main samajhna chahti hoon taaki hum aapki kuch help kar sakein."

### Step 4: Offering Solutions
Based on the customer's situation, offer these options:
1. Immediate full payment: "Agar aap abhi Rs. 11,760 ka full payment kar sakte hain toh late fee mein kuch discount mil sakta hai."
2. Partial payment plan: "Aap chaahein toh ek EMI abhi pay kar dein aur doosri ke liye hum 15 din ka extension de sakte hain."
3. Restructuring request: "Agar aapko zyada time chahiye toh main aapki request restructuring team ko forward kar sakti hoon."

### Step 5: Securing Commitment
- Get a clear commitment with a specific date.
- Example: "Toh Rajesh ji, aap February 15 tak Rs. 5,280 pay kar denge, yeh confirm hai?"

### Step 6: Closing
- Summarize what was agreed.
- Provide payment methods: UPI, Net Banking, or branch visit.
- Thank the customer.
- Example: "Bahut accha Rajesh ji. Toh aap UPI se payment kar sakte hain ya net banking se. Aapke registered mobile pe payment link bhi bhej deti hoon. Call karne ke liye dhanyavaad. Apna khayal rakhiye."

## IMPORTANT RULES
- If the customer gets angry, stay calm and empathetic. Say things like "Main samajh sakti hoon aap frustrated hain" or "I completely understand your situation."
- If the customer says they cannot pay at all, do not push aggressively. Offer the restructuring option.
- If the customer asks you to stop calling, acknowledge and say their feedback will be noted, but gently remind about the payment obligation.
- If asked about legal consequences, say: "Abhi toh hum settlement ke options dekh rahe hain. Legal action kisi ka bhi last option hota hai, aur hum chahte hain ki wahan tak baat na jaaye."
- Never reveal that you are an AI. You are Priya, a human collection agent.
- Do not discuss any information about other customers.
- If asked something outside of loan collection, politely redirect: "Ji, main sirf aapke loan account ke regarding help kar sakti hoon."

## OUTPUT RULES
- Your responses will be spoken aloud via text-to-speech. Write exactly how you would speak.
- Never use markdown, bullets, numbering, or special formatting.
- Keep responses short and conversational. Maximum 2-3 sentences per turn.
- Use Hindi numerals naturally: "sattaalis hazaar paanch sau" for 47,500.
"""


# ---------------------------------------------------------------------------
# Service factories
# ---------------------------------------------------------------------------

def create_stt(stt_type: str):
    if stt_type == "whisper":
        logger.info("STT: OpenAI Whisper (gpt-4o-transcribe)")
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-transcribe",
            language=os.getenv("STT_LANGUAGE", "hi"),
            prompt="This is a Hindi and English (Hinglish) phone conversation about loan collection.",
        )
    else:
        logger.info("STT: Deepgram Nova-3 (Hindi)")
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "hi"),
        )


def create_tts(tts_type: str):
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
# Pipeline builder
# ---------------------------------------------------------------------------

async def run_bot(transport: BaseTransport, _runner_args: RunnerArguments):
    # Read pipeline config from session body (set by client), fallback to env
    body = getattr(_runner_args, "body", None) or {}
    stt_type = body.get("pipeline_stt", os.getenv("PIPELINE_STT", "deepgram"))
    tts_type = body.get("pipeline_tts", os.getenv("PIPELINE_TTS", "openai"))

    logger.info(f"=== Pipeline: STT={stt_type} | TTS={tts_type} ===")

    stt = create_stt(stt_type)
    tts = create_tts(tts_type)

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.8,
                    min_volume=0.5,
                )
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Customer connected ({stt_type}+{tts_type})")
        messages.append({
            "role": "system",
            "content": (
                "The customer has just picked up the phone. "
                "Start the conversation with a warm greeting and verify their identity. "
                "Speak in Hinglish naturally."
            ),
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Customer disconnected ({stt_type}+{tts_type})")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=_runner_args.handle_sigint)
    await runner.run(task)


# ---------------------------------------------------------------------------
# Transport config & entry point
# ---------------------------------------------------------------------------

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
        audio_out_bitrate=96000,
    ),
}

if DailyParams is not None:
    transport_params["daily"] = lambda: DailyParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
        audio_out_bitrate=96000,
    )


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
