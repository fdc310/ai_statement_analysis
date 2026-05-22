"""
Centralized thread pool for wrapping synchronous SDK calls in async context.
All Tencent Cloud SDK calls (ASR, SOE, TTS) use synchronous websocket-client
and must be run in a thread pool to avoid blocking the event loop.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ThreadPool:
    """Singleton thread pool for SDK operations."""

    _pool: Optional[ThreadPoolExecutor] = None

    @classmethod
    def get_pool(cls) -> ThreadPoolExecutor:
        """Get or create the shared thread pool."""
        if cls._pool is None:
            from app.core.config import settings
            pool_size = getattr(settings, 'sdk_thread_pool_size', 20)
            cls._pool = ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="sdk-worker"
            )
            logger.info(f"ThreadPool initialized with {pool_size} workers")
        return cls._pool

    @classmethod
    async def run(cls, func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in the thread pool.

        Usage:
            result = await ThreadPool.run(some_sync_function, arg1, arg2)
        """
        pool = cls.get_pool()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(pool, lambda: func(*args, **kwargs))

    @classmethod
    def shutdown(cls, wait: bool = True):
        """Shutdown the thread pool."""
        if cls._pool is not None:
            cls._pool.shutdown(wait=wait)
            cls._pool = None
            logger.info("ThreadPool shut down")
