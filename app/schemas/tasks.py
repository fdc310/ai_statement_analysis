"""
Task status schemas for API responses.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.services.tasks.manager import TaskStatus


class TaskStatusResponse(BaseModel):
    """Response for task status query."""
    success: bool = True
    message: str = "Task found"
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    current_stage: str = ""
    stages_completed: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    result: Optional[dict] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response for task list query."""
    success: bool = True
    message: str = "Tasks retrieved"
    total: int = 0
    tasks: list[TaskStatusResponse] = Field(default_factory=list)


class TaskStatsResponse(BaseModel):
    """Response for task statistics."""
    success: bool = True
    message: str = "Stats retrieved"
    total: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
