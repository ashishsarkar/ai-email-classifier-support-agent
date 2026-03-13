"""Main LangGraph workflow definition for email processing pipeline."""

from typing import Literal

from langgraph.graph import StateGraph, END
from loguru import logger

from src.db.models import EmailStatusEnum
from src.graph.state import EmailAgentState


def create_workflow(nodes: dict):
    """Create the email processing workflow.

    Args:
        nodes: Dictionary of node functions

    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(EmailAgentState)

    # Add all nodes
    workflow.add_node("email_retrieval", nodes["email_retrieval"])
    workflow.add_node("classification", nodes["classification"])
    workflow.add_node("sentiment_analysis", nodes["sentiment_analysis"])
    workflow.add_node("context_analysis", nodes["context_analysis"])
    workflow.add_node("review_check", nodes["review_check"])
    workflow.add_node("response_generation", nodes["response_generation"])
    workflow.add_node("review_routing", nodes["review_routing"])
    workflow.add_node("human_review", nodes["human_review"])
    workflow.add_node("response_sending", nodes["response_sending"])
    workflow.add_node("followup_scheduling", nodes["followup_scheduling"])
    workflow.add_node("error_handler", nodes["error_handler"])

    # Define edges with conditional routing
    workflow.set_entry_point("email_retrieval")

    # Email retrieval → Classification (or Error)
    workflow.add_edge("email_retrieval", "classification")

    # Classification → Sentiment Analysis (or Error)
    workflow.add_conditional_edges(
        "classification",
        lambda state: "sentiment_analysis" if not state.get("error_message") else "error_handler",
    )

    # Sentiment Analysis → Context Analysis
    workflow.add_edge("sentiment_analysis", "context_analysis")

    # Context Analysis → Review Check
    workflow.add_edge("context_analysis", "review_check")

    # Review Check → always go to response generation (decide on review after)
    workflow.add_edge("review_check", "response_generation")

    # Response Generation → Review Routing or Response Sending
    def route_after_generation(state: EmailAgentState) -> Literal["review_routing", "response_sending", "error_handler"]:
        if state.get("error_message"):
            return "error_handler"
        if state.get("needs_human_review"):
            return "review_routing"
        return "response_sending"

    workflow.add_conditional_edges("response_generation", route_after_generation)

    # Review Routing → Human Review
    workflow.add_edge("review_routing", "human_review")

    # Human Review → END (workflow pauses, webhook resumes remaining steps)
    # When awaiting_review=True, workflow stops here. The reviewer uses
    # POST /api/reviews/{id}/approve to trigger response_sending + followup_scheduling.
    workflow.add_conditional_edges(
        "human_review",
        lambda state: END if state.get("awaiting_review") else (
            "response_sending" if not state.get("error_message") else "error_handler"
        ),
    )

    # Response Sending → Follow-up or End
    workflow.add_conditional_edges(
        "response_sending",
        lambda state: "followup_scheduling" if state.get("status") == EmailStatusEnum.RESPONDED else "error_handler",
    )

    # Follow-up Scheduling → End
    workflow.add_edge("followup_scheduling", END)

    # Error Handler → End
    workflow.add_edge("error_handler", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("Email processing workflow created successfully")

    return app
