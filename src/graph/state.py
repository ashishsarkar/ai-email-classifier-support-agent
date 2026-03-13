"""Graph state definition for email processing workflow."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from typing_extensions import TypedDict


class EmailAgentState(TypedDict, total=False):
    """State object passed through LangGraph nodes.

    This state object carries all information needed throughout the email processing pipeline.
    """

    # Input Information
    email_id: int
    sender: str
    subject: str
    body: str
    html_body: Optional[str]
    received_at: datetime

    # Classification Results
    category: Optional[str]
    priority: Optional[str]
    confidence_score: Optional[float]

    # Sentiment Analysis Results
    sentiment: Optional[str]  # positive, neutral, negative, angry
    sentiment_score: Optional[float]  # -1.0 to 1.0
    tone: Optional[str]  # frustrated, confused, angry, satisfied, neutral, anxious
    escalation_risk: Optional[str]  # low, medium, high, critical
    churn_risk: Optional[bool]  # True if churn signals detected
    churn_indicators: Optional[List[str]]  # List of detected churn signals

    # Context Information
    customer_id: Optional[int]
    customer_history: Optional[List[Dict[str, Any]]]
    kb_results: Optional[List[Dict[str, Any]]]
    context_summary: Optional[str]

    # Response Generation
    generated_response: Optional[str]
    response_subject: Optional[str]
    response_attempt: int
    model_used: Optional[str]
    tokens_used: int

    # Review Workflow (HITL)
    needs_human_review: bool
    review_reason: Optional[str]
    review_id: Optional[int]
    approved_response: Optional[str]
    reviewer_notes: Optional[str]
    awaiting_review: bool  # True when workflow is paused for human review

    # Status Tracking
    status: str
    final_response: Optional[str]
    response_sent: bool
    followup_scheduled: bool

    # Metadata
    error_message: Optional[str]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    execution_time_ms: Optional[float]
