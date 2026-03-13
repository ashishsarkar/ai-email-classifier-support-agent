"""Context analysis node - gathers customer history and KB documents."""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.services.db_service import DatabaseService
from src.services.vector_kb_service import VectorKBService


@traceable(run_type="chain", name="context_analysis")
async def context_analysis_node(state: EmailAgentState) -> dict:
    """Gather customer history and search knowledge base.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with context
    """
    try:
        email_id = state.get("email_id")
        customer_id = state.get("customer_id")
        subject = state.get("subject", "")
        body = state.get("body", "")
        category = state.get("category", "other")

        logger.info(f"Analyzing context for email {email_id}")

        db_service = DatabaseService()
        vector_kb = VectorKBService()
        await vector_kb.initialize()

        # Get customer history
        customer_history = []
        if customer_id:
            history_emails = await db_service.get_customer_emails(customer_id, limit=5)
            customer_history = [
                {
                    "date": email.received_at,
                    "subject": email.subject,
                    "category": email.category,
                    "status": email.status,
                }
                for email in history_emails
            ]
            logger.debug(f"Found {len(customer_history)} previous emails from customer")

        # Search FAISS knowledge base using vector similarity
        search_query = f"{subject} {body[:200]}"
        kb_results = await vector_kb.search(search_query, category=category, limit=5, threshold=0.3)
        logger.debug(f"Found {len(kb_results)} KB documents via FAISS vector search")

        # Format context for LLM
        context_summary = await vector_kb.format_context(kb_results)

        return {
            "customer_history": customer_history,
            "kb_results": kb_results,
            "context_summary": context_summary,
        }

    except Exception as e:
        logger.error(f"Error in context_analysis: {str(e)}", exc_info=True)
        return {
            "customer_history": [],
            "kb_results": [],
            "context_summary": "No additional context available.",
        }
