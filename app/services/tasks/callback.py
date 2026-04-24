"""
Callback dispatcher for async task completion notifications.
Sends HTTP POST to client's callback_url with task results.
Includes exponential backoff retry (max 3 attempts).
"""
import logging
import asyncio
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 3
BASE_RETRY_DELAY = 1.0  # seconds


class CallbackDispatcher:
    """Dispatches callback notifications with retry logic."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the shared httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def send_success(
        self,
        callback_url: str,
        task_id: str,
        result: dict,
        message: str = "Task completed successfully"
    ) -> bool:
        """Send success callback."""
        payload = {
            "task_id": task_id,
            "success": True,
            "message": message,
            "result": result,
        }
        return await self._send_with_retry(callback_url, payload)

    async def send_failure(
        self,
        callback_url: str,
        task_id: str,
        error: str,
        message: str = "Task failed"
    ) -> bool:
        """Send failure callback."""
        payload = {
            "task_id": task_id,
            "success": False,
            "message": message,
            "error": error,
        }
        return await self._send_with_retry(callback_url, payload)

    async def send_progress(
        self,
        callback_url: str,
        task_id: str,
        progress: float,
        stage: str,
        message: str = ""
    ) -> bool:
        """Send progress update callback."""
        payload = {
            "task_id": task_id,
            "success": True,
            "message": message or f"Progress: {stage}",
            "progress": progress,
            "current_stage": stage,
        }
        return await self._send_with_retry(callback_url, payload, max_attempts=1)

    async def _send_with_retry(
        self,
        callback_url: str,
        payload: dict,
        max_attempts: int = MAX_RETRY_ATTEMPTS
    ) -> bool:
        """Send HTTP POST with exponential backoff retry."""
        last_error = None

        for attempt in range(max_attempts):
            try:
                client = self._get_client()
                response = await client.post(
                    callback_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code < 300:
                    logger.info(
                        f"Callback sent successfully to {callback_url} "
                        f"(attempt {attempt + 1})"
                    )
                    return True
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(
                        f"Callback failed (attempt {attempt + 1}/{max_attempts}): "
                        f"{last_error}"
                    )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Callback error (attempt {attempt + 1}/{max_attempts}): {last_error}"
                )

            # Exponential backoff (skip delay on last attempt)
            if attempt < max_attempts - 1:
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

        logger.error(
            f"Callback delivery failed after {max_attempts} attempts to {callback_url}: "
            f"{last_error}"
        )
        return False


# Singleton
callback_dispatcher = CallbackDispatcher()
