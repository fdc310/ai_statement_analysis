"""
Abstract base class for persistent TTS stream sessions.

Used by ws_chat.py for the pattern: one long-lived TTS connection
that receives text chunks incrementally and produces audio in real time.
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseTTSStreamSession(ABC):
    """Persistent TTS stream session for real-time audio generation."""

    @abstractmethod
    def start(self) -> None:
        """Start the TTS connection."""
        ...

    @abstractmethod
    def wait_ready(self, timeout_ms: int) -> bool:
        """Wait until the connection is ready. Returns True if ready."""
        ...

    @abstractmethod
    def process(self, text: str) -> None:
        """Feed text to the synthesizer."""
        ...

    @abstractmethod
    def complete(self) -> None:
        """Signal that all text has been fed. Triggers final synthesis."""
        ...

    @abstractmethod
    def wait(self) -> None:
        """Wait until synthesis is fully complete."""
        ...

    @abstractmethod
    def get_audio_chunks(self) -> list[bytes]:
        """Return all audio chunks collected during the session."""
        ...

    @property
    @abstractmethod
    def error(self) -> Optional[str]:
        """Return error message if synthesis failed, else None."""
        ...
