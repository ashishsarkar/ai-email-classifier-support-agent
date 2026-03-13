"""Review routing node - creates human review task if needed."""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.db.models import ReviewReasonEnum
from src.services.db_service import DatabaseService


@traceable(run_type="chain", name="review_routing")
async def review_routing_node(state: EmailAgentState) -> dict:
    """Create human review task for emails needing approval.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with review task ID
    """
    try:
        email_id = state.get("email_id")
        needs_review = state.get("needs_human_review", False)
        review_reason = state.get("review_reason", "Requires review")

        if not needs_review:
            logger.debug(f"Email {email_id} does not need review")
            return {"review_id": None}

        logger.info(f"Creating review task for email {email_id}")

        db_service = DatabaseService()

        # Map review reason to enum
        reason_mapping = {
            "Low classification confidence": ReviewReasonEnum.LOW_CONFIDENCE,
            "Escalated complaint": ReviewReasonEnum.ESCALATED_COMPLAINT,
            "Critical keywords detected": ReviewReasonEnum.CRITICAL_KEYWORDS,
            "Uncertain category": ReviewReasonEnum.UNCERTAIN_CATEGORY,
            "Threatening tone detected": ReviewReasonEnum.THREATENING_TONE,
        }

        # Match partial reasons for sentiment-based rules
        if review_reason and "escalation risk" in review_reason.lower():
            reason_mapping[review_reason] = ReviewReasonEnum.HIGH_ESCALATION_RISK
        elif review_reason and "churn risk" in review_reason.lower():
            reason_mapping[review_reason] = ReviewReasonEnum.CHURN_RISK
        elif review_reason and "angry" in review_reason.lower():
            reason_mapping[review_reason] = ReviewReasonEnum.ANGRY_CUSTOMER

        reason_enum = reason_mapping.get(review_reason, ReviewReasonEnum.CUSTOM)

        # Create review task
        review = await db_service.create_review(
            email_id=email_id,
            reason=reason_enum,
            notes=f"Generated response ready for approval: {state.get('generated_response', '')[:200]}...",
        )

        logger.info(f"Created review task {review.id} for email {email_id}")

        return {
            "review_id": review.id,
            "review_reason": review_reason,
        }

    except Exception as e:
        logger.error(f"Error in review_routing: {str(e)}", exc_info=True)
        return {
            "error_message": f"Failed to create review task: {str(e)}",
            "review_id": None,
        }
