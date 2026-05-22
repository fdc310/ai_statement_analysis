"""
Task status query endpoints.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.tasks import TaskStatusResponse, TaskListResponse, TaskStatsResponse
from app.services.tasks.manager import TaskStatus, task_manager

router = APIRouter()


@router.get("/stats", response_model=TaskStatsResponse)
async def get_task_stats():
    """Get task queue statistics."""
    stats = await task_manager.get_stats()
    return TaskStatsResponse(**stats)


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        current_stage=task.current_stage,
        stages_completed=task.stages_completed,
        created_at=task.created_at,
        updated_at=task.updated_at,
        result=task.result,
        error=task.error,
    )


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List tasks with optional filtering."""
    task_status = None
    if status:
        try:
            task_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: {[s.value for s in TaskStatus]}"
            )

    tasks = await task_manager.list_tasks(status=task_status, limit=limit, offset=offset)
    stats = await task_manager.get_stats()

    return TaskListResponse(
        total=stats["total"],
        tasks=[
            TaskStatusResponse(
                task_id=t.task_id,
                status=t.status,
                progress=t.progress,
                current_stage=t.current_stage,
                stages_completed=t.stages_completed,
                created_at=t.created_at,
                updated_at=t.updated_at,
                result=t.result,
                error=t.error,
            )
            for t in tasks
        ]
    )
