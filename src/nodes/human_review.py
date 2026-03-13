"""Human review node - pauses workflow for real human approval.

This node implements a true Human-in-the-Loop (HITL) pattern:
1. Marks the email status as AWAITING_REVIEW
2. Returns state that signals the workflow to END (paused)
3. A reviewer uses the /api/reviews webhook endpoints to approve/reject
4. The approve/reject endpoint resumes the remaining pipeline steps
"""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.db.models import EmailStatusEnum
from src.services.db_service import DatabaseService


@traceable(run_type="chain", name="human_review")
async def human_review_node(state: EmailAgentState) -> dict:
    """Pause workflow and wait for human review via webhook.

    Instead of auto-approving, this node:
    - Sets email status to AWAITING_REVIEW
    - Returns a paused state so the workflow ends here
    - The reviewer will use POST /api/reviews/{id}/approve or /reject
      to complete the remaining steps (response_sending, followup_scheduling)

    Args:
        state: Current workflow state

    Returns:
        Updated state dict signaling workflow pause
    """
    try:
        review_id = state.get("review_id")
        email_id = state.get("email_id")

        if not review_id:
            logger.warning(f"No review ID for email {email_id} - skipping review")
            return {"generated_response": state.get("generated_response")}

        logger.info(f"Pausing workflow for human review {review_id} (email {email_id})")

        db_service = DatabaseService()

        # Mark email as awaiting review - workflow pauses here
        await db_service.update_email_status(email_id, EmailStatusEnum.AWAITING_REVIEW)

        logger.info(
            f"Email {email_id} is now AWAITING_REVIEW. "
            f"Use POST /api/reviews/{review_id}/approve or /reject to continue."
        )

        return {
            "status": EmailStatusEnum.AWAITING_REVIEW.value,
            "awaiting_review": True,
        }

    except Exception as e:
        logger.error(f"Error in human_review: {str(e)}", exc_info=True)
        return {
            "error_message": f"Human review processing error: {str(e)}",
        }
