"""Email classification node - categorizes and assesses priority."""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.services.llm_service import LLMService
from src.services.db_service import DatabaseService


@traceable(run_type="chain", name="classification")
async def classification_node(state: EmailAgentState) -> dict:
    """Classify email and assess priority.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with classification results
    """
    try:
        subject = state.get("subject", "")
        body = state.get("body", "")
        email_id = state.get("email_id")

        logger.info(f"Classifying email {email_id}")

        llm_service = LLMService()

        # Classify email
        classification_result = await llm_service.classify_email(subject, body)
        category = classification_result.get("category", "other")
        confidence = classification_result.get("confidence_score", 0.0)

        # Assess priority
        priority_result = await llm_service.assess_priority(body)
        priority = priority_result.get("priority", "medium")

        logger.info(f"Email classified as {category} (confidence: {confidence}) - Priority: {priority}")

        # Store classification in database
        db_service = DatabaseService()
        await db_service.update_email_classification(email_id, category, confidence, priority)

        return {
            "category": category,
            "priority": priority,
            "confidence_score": confidence,
        }

    except Exception as e:
        logger.error(f"Error in classification: {str(e)}", exc_info=True)
        return {
            "error_message": f"Classification failed: {str(e)}",
            "category": "other",
            "priority": "medium",
            "confidence_score": 0.0,
        }
