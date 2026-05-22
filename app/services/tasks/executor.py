"""
Task execution engine. Wraps the orchestrator with progress tracking
and callback dispatch.
"""
import logging
import traceback
from typing import Optional, Callable

from app.services.tasks.manager import TaskManager, TaskStatus, task_manager
from app.services.tasks.callback import CallbackDispatcher, callback_dispatcher

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes evaluation tasks with progress tracking."""

    def __init__(
        self,
        manager: TaskManager = None,
        dispatcher: CallbackDispatcher = None
    ):
        self.manager = manager or task_manager
        self.dispatcher = dispatcher or callback_dispatcher

    async def execute_evaluation(
        self,
        task_id: str,
        pipeline_name: str,
        request_data: dict,
        orchestrator,  # EvaluationOrchestrator instance
        callback_url: Optional[str] = None,
    ) -> None:
        """
        Execute an evaluation task with progress tracking.

        This method is designed to be run as a BackgroundTask.
        """
        try:
            # Update status to processing
            await self.manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                message="Starting evaluation",
                progress=0.0
            )

            # Progress callback for orchestrator
            async def progress_callback(stage: str, progress: float, message: str = ""):
                await self.manager.update_task(
                    task_id,
                    current_stage=stage,
                    progress=progress,
                    message=message or f"Processing: {stage}"
                )
                # Send progress callback if URL provided
                if callback_url:
                    await self.dispatcher.send_progress(
                        callback_url, task_id, progress, stage, message
                    )

            # Run the pipeline
            result = await orchestrator.run_pipeline(
                pipeline_name=pipeline_name,
                request_data=request_data,
                progress_callback=progress_callback
            )

            # Update task as completed
            await self.manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                result=result,
                message="Evaluation completed",
                stage_completed="done"
            )

            # Send success callback
            if callback_url:
                await self.dispatcher.send_success(callback_url, task_id, result)

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Task {task_id} failed: {error_msg}")
            logger.debug(f"Task {task_id} traceback: {traceback.format_exc()}")

            # Update task as failed
            await self.manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                message=f"Evaluation failed: {error_msg}"
            )

            # Send failure callback
            if callback_url:
                await self.dispatcher.send_failure(callback_url, task_id, error_msg)


# Singleton
task_executor = TaskExecutor()
