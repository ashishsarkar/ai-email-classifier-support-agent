"""Sentiment analysis node - analyzes email emotion, tone, and churn risk."""

from loguru import logger

from src.core.tracing import traceable
from src.graph.state import EmailAgentState
from src.services.llm_service import LLMService


@traceable(run_type="chain", name="sentiment_analysis")
async def sentiment_analysis_node(state: EmailAgentState) -> dict:
    """Analyze email sentiment, tone, escalation risk, and churn signals.

    Args:
        state: Current workflow state

    Returns:
        Updated state dict with sentiment analysis results
    """
    try:
        email_id = state.get("email_id")
        subject = state.get("subject", "")
        body = state.get("body", "")

        logger.info(f"Analyzing sentiment for email {email_id}")

        llm_service = LLMService()
        result = await llm_service.analyze_sentiment(subject=subject, body=body)

        error = result.get("error")
        if error:
            logger.warning(f"Sentiment analysis had error (using defaults): {error}")

        sentiment = result.get("sentiment", "neutral")
        sentiment_score = result.get("sentiment_score", 0.0)
        tone = result.get("tone", "neutral")
        escalation_risk = result.get("escalation_risk", "low")
        churn_risk = result.get("churn_risk", False)
        churn_indicators = result.get("churn_indicators", [])

        logger.info(
            f"Sentiment analysis complete for email {email_id}: "
            f"sentiment={sentiment}, score={sentiment_score:.2f}, "
            f"tone={tone}, escalation_risk={escalation_risk}, "
            f"churn_risk={churn_risk}"
        )

        if churn_indicators:
            logger.warning(f"Churn indicators detected for email {email_id}: {churn_indicators}")

        if escalation_risk in ["high", "critical"]:
            logger.warning(f"High escalation risk for email {email_id}: {escalation_risk}")

        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "tone": tone,
            "escalation_risk": escalation_risk,
            "churn_risk": churn_risk,
            "churn_indicators": churn_indicators,
        }

    except Exception as e:
        logger.error(f"Error in sentiment_analysis: {str(e)}", exc_info=True)
        # Return safe defaults - don't block the pipeline
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "tone": "neutral",
            "escalation_risk": "low",
            "churn_risk": False,
            "churn_indicators": [],
        }
