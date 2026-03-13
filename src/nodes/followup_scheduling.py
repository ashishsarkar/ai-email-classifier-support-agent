"""Follow-up scheduling node - schedules follow-up actions if needed."""

from datetime import datetime, timedelta
from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.db.models import FollowUpTypeEnum, EmailStatusEnum
from src.services.db_service import DatabaseService


@traceable(run_type="chain", name="followup_scheduling")
async def followup_scheduling_node(state: EmailAgentState) -> dict:
    """Schedule follow-up actions based on email category and priority.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with follow-up scheduling result
    """
    try:
        email_id = state.get("email_id")
        category = state.get("category", "other")
        priority = state.get("priority", "medium")
        status = state.get("status", EmailStatusEnum.RESPONDED.value)

        if status != EmailStatusEnum.RESPONDED.value:
            logger.debug(f"Email {email_id} not responded, skipping follow-up")
            return {"followup_scheduled": False}

        logger.info(f"Scheduling follow-ups for email {email_id}")

        db_service = DatabaseService()
        followup_scheduled = False

        # Rule 1: Urgent/High priority emails - 24hr follow-up
        if priority in ["urgent", "high"]:
            scheduled_for = datetime.utcnow() + timedelta(hours=24)
            await db_service.create_followup(
                email_id=email_id,
                followup_type=FollowUpTypeEnum.REMINDER,
                scheduled_for=scheduled_for,
            )
            logger.info(f"Scheduled 24-hour reminder for email {email_id}")
            followup_scheduled = True

        # Rule 2: Billing category - 48hr verification
        if category == "billing":
            scheduled_for = datetime.utcnow() + timedelta(hours=48)
            await db_service.create_followup(
                email_id=email_id,
                followup_type=FollowUpTypeEnum.VERIFICATION,
                scheduled_for=scheduled_for,
            )
            logger.info(f"Scheduled 48-hour verification for email {email_id}")
            followup_scheduled = True

        # Rule 3: Technical support - 12hr follow-up for quality check
        if category == "technical_support":
            scheduled_for = datetime.utcnow() + timedelta(hours=12)
            await db_service.create_followup(
                email_id=email_id,
                followup_type=FollowUpTypeEnum.REMINDER,
                scheduled_for=scheduled_for,
            )
            logger.info(f"Scheduled 12-hour follow-up for email {email_id}")
            followup_scheduled = True

        # Rule 4: Complaints - escalation check after 24hr
        if category == "complaint":
            scheduled_for = datetime.utcnow() + timedelta(hours=24)
            await db_service.create_followup(
                email_id=email_id,
                followup_type=FollowUpTypeEnum.ESCALATION,
                scheduled_for=scheduled_for,
            )
            logger.info(f"Scheduled escalation check for email {email_id}")
            followup_scheduled = True

        # Rule 5: Churn risk - urgent 12hr follow-up for retention
        churn_risk = state.get("churn_risk", False)
        if churn_risk:
            scheduled_for = datetime.utcnow() + timedelta(hours=12)
            await db_service.create_followup(
                email_id=email_id,
                followup_type=FollowUpTypeEnum.ESCALATION,
                scheduled_for=scheduled_for,
            )
            logger.warning(f"Scheduled 12-hour churn risk follow-up for email {email_id}")
            followup_scheduled = True

        # Rule 6: Negative sentiment - faster follow-up (6hr)
        sentiment = state.get("sentiment", "neutral")
        sentiment_score = state.get("sentiment_score", 0.0)
        if sentiment in ["negative", "angry"] and sentiment_score <= -0.5:
            scheduled_for = datetime.utcnow() + timedelta(hours=6)
            await db_service.create_followup(
                email_id=email_id,
                followup_type=FollowUpTypeEnum.VERIFICATION,
                scheduled_for=scheduled_for,
            )
            logger.info(f"Scheduled 6-hour sentiment follow-up for email {email_id} (score={sentiment_score})")
            followup_scheduled = True

        logger.info(f"Follow-up scheduling complete for email {email_id}")

        return {"followup_scheduled": followup_scheduled}

    except Exception as e:
        logger.error(f"Error in followup_scheduling: {str(e)}", exc_info=True)
        return {
            "error_message": f"Follow-up scheduling error: {str(e)}",
            "followup_scheduled": False,
        }
