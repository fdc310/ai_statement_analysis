"""
Global LLM request limiter with queueing and retry.
"""
import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
StatusCallback = Callable[[dict], Awaitable[None]]


class LLMQueueFullError(Exception):
    """Raised when the LLM queue is full."""


class LLMQueueTimeoutError(Exception):
    """Raised when a request waits too long for an LLM slot."""


class LLMRequestLimiter:
    """A process-local limiter for all provider LLM calls."""

    def __init__(self):
        self._semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrent))
        self._queue_lock = asyncio.Lock()
        self._rate_lock = asyncio.Lock()
        self._waiting_count = 0
        self._last_start_time = 0.0

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        provider: str,
        operation_name: str,
        status_callback: Optional[StatusCallback] = None,
    ) -> T:
        """Run a non-streaming LLM operation with queueing and retry."""
        max_retries = max(0, settings.llm_max_retries)
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._notify(status_callback, {
                    "stage": "queued",
                    "provider": provider,
                    "operation": operation_name,
                    "attempt": attempt,
                    "queue_size": self.queue_size,
                })
                async with await self._acquire_slot(status_callback, provider, operation_name):
                    await self._notify(status_callback, {
                        "stage": "running",
                        "provider": provider,
                        "operation": operation_name,
                        "attempt": attempt,
                    })
                    return await operation()
            except Exception as exc:
                if attempt > max_retries + 1 or not self._is_retryable(exc):
                    raise
                delay = self._retry_delay(attempt)
                logger.warning(
                    "Retryable LLM error provider=%s operation=%s attempt=%s delay=%.2fs error=%s",
                    provider, operation_name, attempt, delay, exc,
                )
                await self._notify(status_callback, {
                    "stage": "retrying",
                    "provider": provider,
                    "operation": operation_name,
                    "attempt": attempt + 1,
                    "delay_seconds": round(delay, 2),
                    "message": str(exc),
                })
                await asyncio.sleep(delay)

    async def stream(
        self,
        operation: Callable[[], object],
        *,
        provider: str,
        operation_name: str,
        status_callback: Optional[StatusCallback] = None,
    ):
        """
        Run a streaming LLM operation.

        Retries are only safe before the first chunk has been yielded.
        Once data is emitted, later failures are surfaced to the caller.
        """
        max_retries = max(0, settings.llm_max_retries)
        attempt = 0
        while True:
            attempt += 1
            yielded_any = False
            try:
                await self._notify(status_callback, {
                    "stage": "queued",
                    "provider": provider,
                    "operation": operation_name,
                    "attempt": attempt,
                    "queue_size": self.queue_size,
                })
                async with await self._acquire_slot(status_callback, provider, operation_name):
                    await self._notify(status_callback, {
                        "stage": "running",
                        "provider": provider,
                        "operation": operation_name,
                        "attempt": attempt,
                    })
                    async for chunk in operation():
                        yielded_any = True
                        yield chunk
                return
            except Exception as exc:
                if yielded_any or attempt > max_retries + 1 or not self._is_retryable(exc):
                    raise
                delay = self._retry_delay(attempt)
                logger.warning(
                    "Retryable LLM stream error provider=%s operation=%s attempt=%s delay=%.2fs error=%s",
                    provider, operation_name, attempt, delay, exc,
                )
                await self._notify(status_callback, {
                    "stage": "retrying",
                    "provider": provider,
                    "operation": operation_name,
                    "attempt": attempt + 1,
                    "delay_seconds": round(delay, 2),
                    "message": str(exc),
                })
                await asyncio.sleep(delay)

    @property
    def queue_size(self) -> int:
        return self._waiting_count

    async def _acquire_slot(
        self,
        status_callback: Optional[StatusCallback],
        provider: str,
        operation_name: str,
    ):
        async with self._queue_lock:
            if self._waiting_count >= settings.llm_queue_max_size:
                raise LLMQueueFullError("LLM queue is full")
            self._waiting_count += 1
            queue_position = self._waiting_count

        await self._notify(status_callback, {
            "stage": "queued",
            "provider": provider,
            "operation": operation_name,
            "queue_position": queue_position,
            "queue_size": self.queue_size,
        })

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=max(0.1, settings.llm_queue_timeout),
            )
        except asyncio.TimeoutError as exc:
            raise LLMQueueTimeoutError("Timed out waiting for LLM queue slot") from exc
        finally:
            async with self._queue_lock:
                self._waiting_count = max(0, self._waiting_count - 1)

        await self._wait_min_interval()
        return _LLMSlot(self._semaphore)

    async def _wait_min_interval(self) -> None:
        min_interval = max(0, settings.llm_min_interval_ms) / 1000
        if min_interval <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            now = loop.time()
            wait_for = self._last_start_time + min_interval - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_start_time = loop.time()

    def _retry_delay(self, attempt: int) -> float:
        base = max(0.1, settings.llm_retry_base_delay)
        max_delay = max(base, settings.llm_retry_max_delay)
        delay = min(max_delay, base * (2 ** max(0, attempt - 1)))
        return delay + random.uniform(0, min(0.5, delay * 0.2))

    def _is_retryable(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        retry_markers = (
            "429",
            "too many requests",
            "rate limit",
            "ratelimit",
            "requestlimitexceeded",
            "limitexceeded",
            "throttl",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection error",
            "server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "internal server error",
        )
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        return any(marker in text for marker in retry_markers)

    async def _notify(self, callback: Optional[StatusCallback], data: dict) -> None:
        if callback is None:
            return
        try:
            await callback(data)
        except Exception as exc:
            logger.debug("LLM status callback failed: %s", exc)


class _LLMSlot:
    def __init__(self, semaphore: asyncio.Semaphore):
        self._semaphore = semaphore

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()


llm_limiter = LLMRequestLimiter()
