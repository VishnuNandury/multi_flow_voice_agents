# #
# # Edge TTS Service for PipeCat
# # Free Microsoft Edge TTS with excellent Hindi/English pronunciation
# #

# import io
# import struct
# from typing import AsyncGenerator, Optional

# import edge_tts
# from loguru import logger

# from pipecat.frames.frames import (
#     ErrorFrame,
#     Frame,
#     TTSAudioRawFrame,
#     TTSStartedFrame,
#     TTSStoppedFrame,
# )
# from pipecat.services.tts_service import TTSService

# # Edge TTS outputs MP3. We decode to raw PCM using the av library
# # (already installed as a dependency of aiortc/pipecat[webrtc]).
# import av


# class EdgeTTSService(TTSService):
#     """PipeCat TTS service using Microsoft Edge TTS (free, no API key needed).

#     Excellent Hindi, English, and Hinglish pronunciation.
#     Voices: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
#     """

#     EDGE_SAMPLE_RATE = 24000

#     def __init__(
#         self,
#         *,
#         voice: str = "hi-IN-SwaraNeural",
#         rate: str = "+0%",
#         pitch: str = "+0Hz",
#         sample_rate: Optional[int] = None,
#         **kwargs,
#     ):
#         super().__init__(sample_rate=sample_rate or self.EDGE_SAMPLE_RATE, **kwargs)
#         self.set_model_name("edge-tts")
#         self.set_voice(voice)
#         self._rate = rate
#         self._pitch = pitch

#     def can_generate_metrics(self) -> bool:
#         return True

#     def _decode_mp3_to_pcm(self, mp3_bytes: bytes) -> bytes:
#         """Decode MP3 bytes to raw PCM 16-bit mono at target sample rate."""
#         container = av.open(io.BytesIO(mp3_bytes), format="mp3")
#         resampler = av.AudioResampler(
#             format="s16",
#             layout="mono",
#             rate=self.sample_rate,
#         )
#         pcm_data = bytearray()
#         for frame in container.decode(audio=0):
#             resampled = resampler.resample(frame)
#             for r in resampled:
#                 pcm_data.extend(bytes(r.planes[0]))
#         container.close()
#         return bytes(pcm_data)

#     async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
#         await self.start_ttfb_metrics()

#         # Ensure sample rate is set (fallback for pre-pipeline calls)
#         if self._sample_rate == 0:
#             self._sample_rate = self._init_sample_rate or self.EDGE_SAMPLE_RATE

#         try:
#             communicate = edge_tts.Communicate(
#                 text=text,
#                 voice=self._voice_id,
#                 rate=self._rate,
#                 pitch=self._pitch,
#             )

#             # Collect MP3 chunks from edge-tts
#             mp3_buffer = bytearray()
#             async for chunk in communicate.stream():
#                 if chunk["type"] == "audio":
#                     mp3_buffer.extend(chunk["data"])

#             if not mp3_buffer:
#                 logger.warning("Edge TTS returned no audio")
#                 yield ErrorFrame("Edge TTS returned no audio")
#                 return

#             # Decode MP3 to PCM
#             pcm_data = self._decode_mp3_to_pcm(bytes(mp3_buffer))

#             await self.start_tts_usage_metrics(text)
#             await self.stop_ttfb_metrics()

#             yield TTSStartedFrame()

#             # Stream PCM in chunks
#             chunk_size = self.chunk_size
#             offset = 0
#             while offset < len(pcm_data):
#                 end = min(offset + chunk_size, len(pcm_data))
#                 yield TTSAudioRawFrame(
#                     audio=pcm_data[offset:end],
#                     sample_rate=self.sample_rate,
#                     num_channels=1,
#                 )
#                 offset = end

#             yield TTSStoppedFrame()

#         except Exception as e:
#             logger.error(f"Edge TTS error: {e}")
#             yield ErrorFrame(f"Edge TTS error: {e}")

import io
from typing import AsyncGenerator, Optional
import av
import edge_tts
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

class EdgeTTSService(TTSService):
    """Microsoft Edge TTS service for PipeCat."""
    
    EDGE_SAMPLE_RATE = 24000
    
    def __init__(
        self,
        *,
        voice: str = "hi-IN-SwaraNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate or self.EDGE_SAMPLE_RATE, **kwargs)
        self.set_model_name("edge-tts")
        self.set_voice(voice)
        self._rate = rate
        self._pitch = pitch
        self._pcm_cache: dict = {}  # text -> PCM bytes (pre-warmed or previously generated)
    
    def can_generate_metrics(self) -> bool:
        return True
    
    def _decode_mp3_to_pcm(self, mp3_bytes: bytes) -> bytes:
        """Decode MP3 bytes to 16-bit mono PCM at target sample rate."""
        buf = io.BytesIO(mp3_bytes)
        container = av.open(buf, format="mp3")
        
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sample_rate,
        )
        
        pcm_chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for rf in resampled:
                pcm_chunks.append(rf.to_ndarray().tobytes())
        
        container.close()
        return b"".join(pcm_chunks)
    
    async def _fetch_pcm(self, text: str) -> bytes | None:
        """Download full MP3 from Edge TTS and decode to PCM bytes. Used by pre_warm."""
        try:
            communicate = edge_tts.Communicate(
                text=text, voice=self._voice_id, rate=self._rate, pitch=self._pitch,
            )
            mp3_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])
            if mp3_buffer:
                return self._decode_mp3_to_pcm(bytes(mp3_buffer))
        except Exception as e:
            logger.debug(f"EdgeTTS _fetch_pcm error: {e}")
        return None

    async def pre_warm(self, texts: list) -> None:
        """Pre-generate PCM for each text and store in cache.

        Call this as a background task while the bot is greeting the user so
        that all subsequent node opening texts are served from cache instantly.
        """
        for text in texts:
            if not text or text in self._pcm_cache:
                continue
            pcm = await self._fetch_pcm(text)
            if pcm:
                self._pcm_cache[text] = pcm
                logger.debug(f"EdgeTTS pre-warmed [{text[:50]}]")

    async def run_tts(self, text: str, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        # Strip function-call tokens and XML wrappers that LLMs (Groq/llama) emit
        import re
        text = re.sub(r'<function=[^>]*>.*?</function>', '', text, flags=re.DOTALL)
        text = re.sub(r'<function=[^>]*>\s*\{[^}]*\}', '', text)
        text = re.sub(r'</?function[^>]*>', '', text)
        text = re.sub(r'\(function=[^)>]*>[^(]*', '', text)
        text = re.sub(r'</?[a-zA-Z][a-zA-Z0-9_-]{0,15}(?:\s[^>]{0,80})?/?>', '', text)
        text = re.sub(r'^<[^/][^>]*>', '', text.strip())
        text = re.sub(r'</[^>]*>$', '', text.strip())
        text = text.strip()
        if not text:
            return

        # --- Cache hit: serve pre-warmed PCM without any API call ---
        cached = self._pcm_cache.get(text)
        if cached:
            logger.debug(f"EdgeTTS cache HIT [{text[:50]}]")
            await self.start_ttfb_metrics()
            await self.stop_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()
            chunk_size = 8192
            offset = 0
            while offset < len(cached):
                end = min(offset + chunk_size, len(cached))
                yield TTSAudioRawFrame(audio=cached[offset:end], sample_rate=self.sample_rate, num_channels=1)
                offset = end
            yield TTSStoppedFrame()
            return

        logger.debug(f"EdgeTTS: Generating [{text[:50]}...]")

        try:
            await self.start_ttfb_metrics()

            communicate = edge_tts.Communicate(
                text=text,
                voice=self._voice_id,
                rate=self._rate,
                pitch=self._pitch,
            )

            # Stream MP3 chunks and decode progressively.
            # edge-tts sends audio over WebSocket in bursts — we accumulate in
            # a bytearray but attempt a partial decode every DECODE_INTERVAL bytes
            # so the first PCM frames can be yielded before the full download.
            DECODE_INTERVAL = 12_000   # ~0.4 s of 128 kbps MP3
            mp3_buffer = bytearray()
            pcm_queue: list[bytes] = []
            ttfb_done = False
            decoded_up_to = 0

            async for chunk in communicate.stream():
                if chunk["type"] != "audio":
                    continue
                mp3_buffer.extend(chunk["data"])

                # Try a partial decode once we have a new DECODE_INTERVAL chunk
                if len(mp3_buffer) - decoded_up_to >= DECODE_INTERVAL:
                    try:
                        partial_pcm = self._decode_mp3_to_pcm(bytes(mp3_buffer))
                        # Only yield the newly decoded portion
                        if len(partial_pcm) > sum(len(p) for p in pcm_queue):
                            new_pcm = partial_pcm[sum(len(p) for p in pcm_queue):]
                            pcm_queue.append(new_pcm)
                            decoded_up_to = len(mp3_buffer)
                            if not ttfb_done:
                                await self.stop_ttfb_metrics()
                                await self.start_tts_usage_metrics(text)
                                yield TTSStartedFrame()
                                ttfb_done = True
                            chunk_size = 8192
                            offset = 0
                            while offset < len(new_pcm):
                                end = min(offset + chunk_size, len(new_pcm))
                                yield TTSAudioRawFrame(
                                    audio=new_pcm[offset:end],
                                    sample_rate=self.sample_rate,
                                    num_channels=1,
                                )
                                offset = end
                    except Exception:
                        pass  # partial MP3 decode failed — wait for more data

            if not mp3_buffer:
                logger.warning("Edge TTS returned no audio")
                yield ErrorFrame("Edge TTS returned no audio")
                return

            # Final decode: pick up any remaining PCM not yet yielded
            try:
                full_pcm = self._decode_mp3_to_pcm(bytes(mp3_buffer))
            except Exception as e:
                logger.error(f"Edge TTS final decode error: {e}")
                yield ErrorFrame(f"Edge TTS decode error: {e}")
                return

            already_yielded = sum(len(p) for p in pcm_queue)
            remaining = full_pcm[already_yielded:]
            # Store full PCM so repeated calls (e.g. same greeting) hit cache
            self._pcm_cache[text] = full_pcm

            if not ttfb_done:
                await self.stop_ttfb_metrics()
                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

            if remaining:
                chunk_size = 8192
                offset = 0
                while offset < len(remaining):
                    end = min(offset + chunk_size, len(remaining))
                    yield TTSAudioRawFrame(
                        audio=remaining[offset:end],
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    offset = end

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"Edge TTS error: {e}", exc_info=True)
            yield ErrorFrame(f"Edge TTS error: {e}")