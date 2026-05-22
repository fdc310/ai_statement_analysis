"""
In-memory task manager for async evaluation tracking.
Tasks are stored in memory and lost on server restart (acceptable for v1).
"""
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    current_stage: str = ""
    stages_completed: list[str] = Field(default_factory=list)
    pipeline_name: str = ""


class TaskManager:
    """In-memory task store with lifecycle management."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        callback_url: Optional[str] = None,
        pipeline_name: str = "",
        message: str = ""
    ) -> str:
        """Create a new task and return its ID."""
        async with self._lock:
            task = TaskInfo(
                callback_url=callback_url,
                pipeline_name=pipeline_name,
                message=message or "Task created"
            )
            self._tasks[task.task_id] = task
            logger.info(f"Task created: {task.task_id} (pipeline={pipeline_name})")
            return task.task_id

    async def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        stage_completed: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update task state."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task not found: {task_id}")
                return

            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = min(max(progress, 0.0), 1.0)
            if current_stage is not None:
                task.current_stage = current_stage
            if stage_completed is not None:
                if stage_completed not in task.stages_completed:
                    task.stages_completed.append(stage_completed)
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if message is not None:
                task.message = message

            task.updated_at = datetime.now()

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task by ID."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[TaskInfo]:
        """List tasks with optional filtering."""
        async with self._lock:
            tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        # Sort by creation time, newest first
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[offset:offset + limit]

    async def cleanup_old_tasks(self, max_age_seconds: int = 86400) -> int:
        """Remove tasks older than max_age_seconds. Returns count removed."""
        async with self._lock:
            cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
            old_ids = [
                tid for tid, task in self._tasks.items()
                if task.created_at < cutoff
            ]
            for tid in old_ids:
                del self._tasks[tid]
            if old_ids:
                logger.info(f"Cleaned up {len(old_ids)} old tasks")
            return len(old_ids)

    async def get_stats(self) -> dict:
        """Get task statistics."""
        async with self._lock:
            tasks = list(self._tasks.values())
        return {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "processing": sum(1 for t in tasks if t.status == TaskStatus.PROCESSING),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
        }


# Singleton
task_manager = TaskManager()
