"""Error handler node - handles and logs processing errors."""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.db.models import EmailStatusEnum
from src.services.db_service import DatabaseService


@traceable(run_type="chain", name="error_handler")
async def error_handler_node(state: EmailAgentState) -> dict:
    """Handle errors and update email status.

    Args:
        state: Current workflow state

    Returns:
        Final state dict
    """
    try:
        email_id = state.get("email_id")
        error_message = state.get("error_message", "Unknown error")

        logger.error(f"Error handling email {email_id}: {error_message}")

        # Update email status to failed
        db_service = DatabaseService()
        await db_service.update_email_status(
            email_id,
            EmailStatusEnum.FAILED,
            error_msg=error_message,
        )

        # Log error details
        logger.error(f"Email {email_id} processing failed")
        logger.error(f"Error details: {error_message}")

        # Optional: Send notification to admin
        # await notify_admin(f"Email {email_id} failed: {error_message}")

        return {
            "status": EmailStatusEnum.FAILED.value,
            "error_message": error_message,
        }

    except Exception as e:
        logger.critical(f"Error in error_handler: {str(e)}", exc_info=True)
        return {
            "status": EmailStatusEnum.FAILED.value,
            "error_message": f"Critical error in error handler: {str(e)}",
        }
