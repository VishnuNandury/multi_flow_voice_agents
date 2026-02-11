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
    
    async def run_tts(self, text: str, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        logger.debug(f"EdgeTTS: Generating [{text[:50]}...]")
        
        try:
            await self.start_ttfb_metrics()
            
            communicate = edge_tts.Communicate(
                text=text,
                voice=self._voice_id,
                rate=self._rate,
                pitch=self._pitch,
            )
            
            # Collect ALL MP3 data first (critical for proper decoding)
            mp3_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])
            
            if not mp3_buffer:
                logger.warning("Edge TTS returned no audio")
                yield ErrorFrame("Edge TTS returned no audio")
                return
            
            # Decode MP3 to PCM in one shot
            pcm_data = self._decode_mp3_to_pcm(bytes(mp3_buffer))
            
            await self.stop_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            
            yield TTSStartedFrame()
            
            # Stream PCM in chunks (use larger chunks for smoother playback)
            chunk_size = 8192  # Increased from default
            offset = 0
            while offset < len(pcm_data):
                end = min(offset + chunk_size, len(pcm_data))
                yield TTSAudioRawFrame(
                    audio=pcm_data[offset:end],
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                offset = end
            
            yield TTSStoppedFrame()
            
        except Exception as e:
            logger.error(f"Edge TTS error: {e}", exc_info=True)
            yield ErrorFrame(f"Edge TTS error: {e}")