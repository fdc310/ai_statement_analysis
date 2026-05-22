"""
Audio buffer for chunked PCM audio processing.
Handles buffering and chunking of raw PCM 16kHz 16bit mono audio.
"""
import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# 16kHz 16bit mono: 1 second = 32000 bytes
BYTES_PER_SEC = 32000
CHUNK_SIZE = 6400  # 200ms of audio


class AudioBuffer:
    """
    Buffers raw PCM audio data and provides chunked access.

    Used for streaming audio to ASR/SOE services that require
    specific chunk sizes (e.g., 6400 bytes for SOE).
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self._buffer = bytearray()
        self._chunk_size = chunk_size
        self._lock = asyncio.Lock()
        self._is_final = False

    async def append(self, pcm_data: bytes) -> None:
        """Append PCM data to the buffer."""
        async with self._lock:
            self._buffer.extend(pcm_data)

    async def get_chunks(self) -> AsyncGenerator[bytes, None]:
        """
        Yield complete chunks from the buffer.
        Blocks until a complete chunk is available.
        """
        while True:
            async with self._lock:
                if len(self._buffer) >= self._chunk_size:
                    chunk = bytes(self._buffer[:self._chunk_size])
                    self._buffer = self._buffer[self._chunk_size:]
                    yield chunk
                elif len(self._buffer) > 0 and self._is_final:
                    # Final partial chunk
                    chunk = bytes(self._buffer)
                    self._buffer = bytearray()
                    yield chunk
                    return
                else:
                    # Wait for more data
                    await asyncio.sleep(0.05)

    async def get_full_buffer(self) -> bytes:
        """Get the complete buffer contents."""
        async with self._lock:
            return bytes(self._buffer)

    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self._buffer = bytearray()

    @property
    def size(self) -> int:
        """Current buffer size in bytes."""
        return len(self._buffer)

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        return self.size / BYTES_PER_SEC

    def set_final(self) -> None:
        """Mark the buffer as final (no more data coming)."""
        self._is_final = True
