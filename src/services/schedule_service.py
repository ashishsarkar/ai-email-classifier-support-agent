"""Scheduling service for follow-up tasks using APScheduler."""

from datetime import datetime, timedelta
from typing import Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from src.services.base import BaseService


class ScheduleService(BaseService):
    """Service for scheduling follow-up tasks."""

    def __init__(self):
        """Initialize schedule service."""
        self.scheduler = AsyncIOScheduler()
        self.scheduled_jobs = {}

    async def initialize(self) -> None:
        """Initialize and start scheduler."""
        await super().initialize()
        try:
            self.scheduler.start()
            logger.info("Scheduler started successfully")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    async def cleanup(self) -> None:
        """Shutdown scheduler."""
        try:
            self.scheduler.shutdown()
            logger.info("Scheduler shutdown")
        except Exception:
            pass
        await super().cleanup()

    async def schedule_followup(
        self,
        email_id: int,
        delay_minutes: int,
        callback: Callable,
        followup_type: str = "reminder",
    ) -> bool:
        """Schedule a follow-up task.

        Args:
            email_id: Email ID
            delay_minutes: Minutes until execution
            callback: Async callback function to execute
            followup_type: Type of follow-up

        Returns:
            True if scheduled successfully
        """
        try:
            run_time = datetime.utcnow() + timedelta(minutes=delay_minutes)
            job_id = f"followup_email_{email_id}_{followup_type}"

            # Remove existing job if exists
            if job_id in self.scheduled_jobs:
                self.scheduler.remove_job(job_id)

            # Schedule new job
            job = self.scheduler.add_job(
                callback,
                "date",
                run_date=run_time,
                id=job_id,
                args=[email_id],
                replace_existing=True,
            )

            self.scheduled_jobs[job_id] = job
            logger.info(f"Scheduled {followup_type} for email {email_id} in {delay_minutes} minutes")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule follow-up: {e}")
            return False

    async def cancel_followup(self, email_id: int, followup_type: str = "reminder") -> bool:
        """Cancel scheduled follow-up.

        Args:
            email_id: Email ID
            followup_type: Type of follow-up

        Returns:
            True if cancelled successfully
        """
        try:
            job_id = f"followup_email_{email_id}_{followup_type}"

            if job_id in self.scheduled_jobs:
                self.scheduler.remove_job(job_id)
                del self.scheduled_jobs[job_id]
                logger.info(f"Cancelled {followup_type} for email {email_id}")
                return True

            logger.warning(f"Job {job_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel follow-up: {e}")
            return False

    async def health_check(self) -> dict:
        """Check scheduler health."""
        try:
            running = self.scheduler.running
            job_count = len(self.scheduler.get_jobs())
            return {
                "status": "healthy" if running else "unhealthy",
                "service": "scheduler",
                "running": running,
                "scheduled_jobs": job_count,
            }
        except Exception as e:
            logger.error(f"Scheduler health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "scheduler",
                "error": str(e),
            }
