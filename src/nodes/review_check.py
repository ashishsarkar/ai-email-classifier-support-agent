"""Review check node - determines if human review is needed."""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState


@traceable(run_type="chain", name="review_check")
async def review_check_node(state: EmailAgentState) -> dict:
    """Determine if email needs human review.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with review decision
    """
    try:
        category = state.get("category", "other")
        priority = state.get("priority", "medium")
        confidence = state.get("confidence_score", 1.0)
        body = state.get("body", "").lower()

        email_id = state.get("email_id")
        logger.info(f"Checking if review needed for email {email_id}")

        needs_review = False
        review_reason = None

        # Rule 1: Low confidence classification
        if confidence < 0.6:
            needs_review = True
            review_reason = "Low classification confidence"
            logger.debug(f"Low confidence ({confidence}) - flagging for review")

        # Rule 2: Escalated complaints
        if category == "complaint" and priority in ["high", "urgent"]:
            needs_review = True
            review_reason = "Escalated complaint"
            logger.debug("Complaint with high/urgent priority - flagging for review")

        # Rule 3: Critical keywords
        critical_keywords = ["urgent", "fire", "down", "broken", "help", "emergency", "asap", "critical"]
        if any(keyword in body for keyword in critical_keywords):
            needs_review = True
            review_reason = "Critical keywords detected"
            logger.debug(f"Critical keywords found - flagging for review")

        # Rule 4: Uncertain categories
        if category == "other" and confidence < 0.8:
            needs_review = True
            review_reason = "Uncertain category"
            logger.debug("Uncertain category - flagging for review")

        # Rule 5: High/Critical escalation risk from sentiment analysis
        escalation_risk = state.get("escalation_risk", "low")
        if escalation_risk in ["high", "critical"]:
            needs_review = True
            review_reason = f"High escalation risk ({escalation_risk})"
            logger.debug(f"Escalation risk {escalation_risk} - flagging for review")

        # Rule 6: Churn risk detected
        churn_risk = state.get("churn_risk", False)
        if churn_risk:
            needs_review = True
            churn_indicators = state.get("churn_indicators", [])
            review_reason = f"Churn risk detected: {', '.join(churn_indicators[:3])}"
            logger.warning(f"Churn risk for email {email_id} - flagging for review")

        # Rule 7: Very negative sentiment (angry customer, any category)
        sentiment = state.get("sentiment", "neutral")
        sentiment_score = state.get("sentiment_score", 0.0)
        if sentiment == "angry" or sentiment_score <= -0.7:
            needs_review = True
            review_reason = f"Angry customer (sentiment_score={sentiment_score})"
            logger.debug(f"Angry sentiment detected - flagging for review")

        # Rule 8: Threatening tone
        tone = state.get("tone", "neutral")
        if tone == "threatening":
            needs_review = True
            review_reason = "Threatening tone detected"
            logger.warning(f"Threatening tone for email {email_id} - flagging for review")

        logger.info(f"Review check complete: needs_review={needs_review}, reason={review_reason}")

        return {
            "needs_human_review": needs_review,
            "review_reason": review_reason,
        }

    except Exception as e:
        logger.error(f"Error in review_check: {str(e)}", exc_info=True)
        return {
            "needs_human_review": True,
            "review_reason": f"Review check error: {str(e)}",
        }
