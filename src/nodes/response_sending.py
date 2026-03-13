"""Response sending node - sends final response email."""

from datetime import datetime
from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.db.models import EmailStatusEnum
from src.services.email_service import EmailService
from src.services.db_service import DatabaseService


@traceable(run_type="chain", name="response_sending")
async def response_sending_node(state: EmailAgentState) -> dict:
    """Send response email to customer.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with sending result
    """
    try:
        email_id = state.get("email_id")
        sender = state.get("sender")
        response_text = state.get("generated_response") or state.get("approved_response")
        response_subject = state.get("response_subject", "Re: Customer Support")

        if not response_text:
            logger.error(f"No response text to send for email {email_id}")
            return {
                "error_message": "No response text available",
                "status": EmailStatusEnum.FAILED.value,
            }

        logger.info(f"Sending response for email {email_id} to {sender}")

        email_service = EmailService()
        db_service = DatabaseService()

        # Send email
        send_success = await email_service.send_email(
            to_address=sender,
            subject=response_subject,
            body=response_text,
        )

        if not send_success:
            logger.error(f"Failed to send email to {sender}")
            return {
                "error_message": f"Failed to send email to {sender}",
                "status": EmailStatusEnum.FAILED.value,
                "response_sent": False,
            }

        # Create response record in database
        response = await db_service.create_response(
            email_id=email_id,
            response_text=response_text,
            model_used=state.get("model_used", "gpt-4-turbo-preview"),
            tokens_used=state.get("tokens_used", 0),
            confidence_score=state.get("confidence_score", 0.85),
            requires_review=state.get("needs_human_review", False),
        )

        # Update email status
        await db_service.update_email_status(email_id, EmailStatusEnum.RESPONDED)

        logger.info(f"Response sent successfully for email {email_id}")

        return {
            "final_response": response_text,
            "status": EmailStatusEnum.RESPONDED.value,
            "response_sent": True,
        }

    except Exception as e:
        logger.error(f"Error in response_sending: {str(e)}", exc_info=True)
        return {
            "error_message": f"Response sending error: {str(e)}",
            "status": EmailStatusEnum.FAILED.value,
            "response_sent": False,
        }
