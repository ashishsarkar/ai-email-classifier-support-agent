"""Human review webhook endpoints for HITL (Human-in-the-Loop).

These endpoints allow human reviewers to:
1. List all pending reviews
2. View a specific review with email + AI-generated response
3. Approve a review (with optional edits to the response)
4. Reject a review (with reason)
5. Escalate a review to a senior agent

When a review is approved/rejected, the remaining pipeline steps
(response_sending, followup_scheduling) are executed automatically.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from src.db.database import SessionLocal
from src.db.models import (
    Email,
    EmailResponse,
    HumanReview,
    EmailStatusEnum,
    ReviewStatusEnum,
)
from src.nodes.response_sending import response_sending_node
from src.nodes.followup_scheduling import followup_scheduling_node
from src.services.db_service import DatabaseService

router = APIRouter(prefix="/api/reviews", tags=["Reviews"])


# --- Request/Response Models ---


class ReviewApproveRequest(BaseModel):
    """Request body for approving a review."""

    approved_response: Optional[str] = None  # If None, use AI-generated response as-is
    reviewer_notes: Optional[str] = None


class ReviewRejectRequest(BaseModel):
    """Request body for rejecting a review."""

    rejection_reason: str
    reviewer_notes: Optional[str] = None


class ReviewEscalateRequest(BaseModel):
    """Request body for escalating a review."""

    escalate_to: Optional[str] = None  # e.g., "senior-agent", "manager"
    reviewer_notes: Optional[str] = None


class ReviewListItem(BaseModel):
    """Review item for listing."""

    id: int
    email_id: int
    sender: str
    subject: str
    category: Optional[str]
    priority: Optional[str]
    reason: str
    status: str
    generated_response: Optional[str]
    created_at: Optional[str]

    class Config:
        from_attributes = True


class ReviewDetail(BaseModel):
    """Full review detail for the reviewer."""

    id: int
    email_id: int
    sender: str
    subject: str
    body: str
    category: Optional[str]
    priority: Optional[str]
    confidence_score: Optional[float]
    reason: str
    notes: Optional[str]
    status: str
    generated_response: Optional[str]
    sentiment: Optional[str]
    tone: Optional[str]
    escalation_risk: Optional[str]
    churn_risk: Optional[bool]
    created_at: Optional[str]

    class Config:
        from_attributes = True


class ReviewActionResponse(BaseModel):
    """Response after a review action (approve/reject/escalate)."""

    review_id: int
    email_id: int
    action: str
    status: str
    response_sent: bool
    message: str


# --- Endpoints ---


@router.get("/pending", response_model=list)
async def list_pending_reviews():
    """List all reviews awaiting human decision.

    Returns reviews with status PENDING or IN_PROGRESS,
    along with associated email data.
    """
    try:
        session = SessionLocal()

        reviews = (
            session.query(HumanReview)
            .filter(
                HumanReview.status.in_([
                    ReviewStatusEnum.PENDING,
                    ReviewStatusEnum.IN_PROGRESS,
                ])
            )
            .order_by(HumanReview.created_at.desc())
            .all()
        )

        result = []
        for review in reviews:
            email = session.query(Email).filter(Email.id == review.email_id).first()
            # Get the AI-generated response
            response = (
                session.query(EmailResponse)
                .filter(EmailResponse.email_id == review.email_id)
                .first()
            )

            result.append(
                ReviewListItem(
                    id=review.id,
                    email_id=review.email_id,
                    sender=email.sender if email else "unknown",
                    subject=email.subject if email else "unknown",
                    category=email.category if email else None,
                    priority=email.priority if email else None,
                    reason=review.reason,
                    status=review.status,
                    generated_response=response.response_text if response else None,
                    created_at=review.created_at.isoformat() if review.created_at else None,
                ).model_dump()
            )

        session.close()
        return result

    except Exception as e:
        logger.error(f"List pending reviews failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{review_id}", response_model=ReviewDetail)
async def get_review_detail(review_id: int):
    """Get full review detail for a reviewer to make a decision.

    Includes the original email, AI classification, sentiment analysis,
    and the AI-generated response that needs approval.
    """
    try:
        session = SessionLocal()

        review = session.query(HumanReview).filter(HumanReview.id == review_id).first()
        if not review:
            session.close()
            raise HTTPException(status_code=404, detail="Review not found")

        email = session.query(Email).filter(Email.id == review.email_id).first()
        if not email:
            session.close()
            raise HTTPException(status_code=404, detail="Associated email not found")

        # Get AI-generated response
        response = (
            session.query(EmailResponse)
            .filter(EmailResponse.email_id == review.email_id)
            .first()
        )

        session.close()

        return ReviewDetail(
            id=review.id,
            email_id=review.email_id,
            sender=email.sender,
            subject=email.subject,
            body=email.body,
            category=email.category,
            priority=email.priority,
            confidence_score=email.confidence_score,
            reason=review.reason,
            notes=review.notes,
            status=review.status,
            generated_response=response.response_text if response else None,
            sentiment=None,  # Could be stored in email or a separate table
            tone=None,
            escalation_risk=None,
            churn_risk=None,
            created_at=review.created_at.isoformat() if review.created_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get review detail failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{review_id}/approve", response_model=ReviewActionResponse)
async def approve_review(review_id: int, request_body: ReviewApproveRequest):
    """Approve a review and resume the pipeline.

    This webhook:
    1. Updates review status to APPROVED
    2. Runs response_sending_node (sends the email)
    3. Runs followup_scheduling_node (schedules follow-ups)

    The reviewer can optionally edit the response before approving.
    """
    try:
        db_service = DatabaseService()
        session = SessionLocal()

        # Validate review exists and is pending
        review = session.query(HumanReview).filter(HumanReview.id == review_id).first()
        if not review:
            session.close()
            raise HTTPException(status_code=404, detail="Review not found")

        if review.status not in [ReviewStatusEnum.PENDING, ReviewStatusEnum.IN_PROGRESS]:
            session.close()
            raise HTTPException(
                status_code=400,
                detail=f"Review already {review.status} - cannot approve",
            )

        email = session.query(Email).filter(Email.id == review.email_id).first()
        if not email:
            session.close()
            raise HTTPException(status_code=404, detail="Associated email not found")

        # Get the AI-generated response
        response_record = (
            session.query(EmailResponse)
            .filter(EmailResponse.email_id == review.email_id)
            .first()
        )

        # Determine final response text
        final_response = request_body.approved_response or (
            response_record.response_text if response_record else None
        )

        if not final_response:
            session.close()
            raise HTTPException(status_code=400, detail="No response text available to approve")

        session.close()

        # 1. Update review status to APPROVED
        await db_service.update_review_with_decision(
            review_id=review_id,
            status=ReviewStatusEnum.APPROVED,
            approved_response=final_response,
            reviewer_notes=request_body.reviewer_notes,
        )

        logger.info(f"Review {review_id} APPROVED - resuming pipeline for email {email.id}")

        # 2. Build state to resume pipeline from response_sending
        resume_state = {
            "email_id": email.id,
            "sender": email.sender,
            "subject": email.subject,
            "body": email.body,
            "category": email.category,
            "priority": email.priority,
            "confidence_score": email.confidence_score,
            "generated_response": final_response,
            "approved_response": final_response,
            "response_subject": f"Re: {email.subject}",
            "needs_human_review": True,
            "review_id": review_id,
            "model_used": response_record.model_used if response_record else "gpt-4-turbo-preview",
            "tokens_used": response_record.tokens_used if response_record else 0,
        }

        # 3. Run response_sending_node
        sending_result = await response_sending_node(resume_state)
        resume_state.update(sending_result)

        # 4. Run followup_scheduling_node if response was sent successfully
        response_sent = sending_result.get("response_sent", False)
        if resume_state.get("status") == EmailStatusEnum.RESPONDED.value:
            followup_result = await followup_scheduling_node(resume_state)
            resume_state.update(followup_result)

        return ReviewActionResponse(
            review_id=review_id,
            email_id=email.id,
            action="approved",
            status=resume_state.get("status", "unknown"),
            response_sent=response_sent,
            message=f"Review approved. Response {'sent successfully' if response_sent else 'sending failed'}.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{review_id}/reject", response_model=ReviewActionResponse)
async def reject_review(review_id: int, request_body: ReviewRejectRequest):
    """Reject a review - response will NOT be sent to customer.

    The email status is updated to FAILED with the rejection reason.
    """
    try:
        db_service = DatabaseService()
        session = SessionLocal()

        review = session.query(HumanReview).filter(HumanReview.id == review_id).first()
        if not review:
            session.close()
            raise HTTPException(status_code=404, detail="Review not found")

        if review.status not in [ReviewStatusEnum.PENDING, ReviewStatusEnum.IN_PROGRESS]:
            session.close()
            raise HTTPException(
                status_code=400,
                detail=f"Review already {review.status} - cannot reject",
            )

        email_id = review.email_id
        session.close()

        # Update review status to REJECTED
        await db_service.update_review_with_decision(
            review_id=review_id,
            status=ReviewStatusEnum.REJECTED,
            reviewer_notes=request_body.reviewer_notes or request_body.rejection_reason,
        )

        # Update email status to FAILED with rejection reason
        await db_service.update_email_status(
            email_id,
            EmailStatusEnum.FAILED,
            error_msg=f"Review rejected: {request_body.rejection_reason}",
        )

        logger.info(f"Review {review_id} REJECTED - email {email_id} marked as failed")

        return ReviewActionResponse(
            review_id=review_id,
            email_id=email_id,
            action="rejected",
            status=EmailStatusEnum.FAILED.value,
            response_sent=False,
            message=f"Review rejected: {request_body.rejection_reason}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reject review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{review_id}/escalate", response_model=ReviewActionResponse)
async def escalate_review(review_id: int, request_body: ReviewEscalateRequest):
    """Escalate a review to a senior agent or manager.

    The review remains open but status changes to ESCALATED.
    """
    try:
        db_service = DatabaseService()
        session = SessionLocal()

        review = session.query(HumanReview).filter(HumanReview.id == review_id).first()
        if not review:
            session.close()
            raise HTTPException(status_code=404, detail="Review not found")

        if review.status not in [ReviewStatusEnum.PENDING, ReviewStatusEnum.IN_PROGRESS]:
            session.close()
            raise HTTPException(
                status_code=400,
                detail=f"Review already {review.status} - cannot escalate",
            )

        email_id = review.email_id

        # Update assigned_to if escalation target provided
        if request_body.escalate_to:
            review.assigned_to = request_body.escalate_to
            session.commit()

        session.close()

        # Update review status to ESCALATED
        await db_service.update_review_with_decision(
            review_id=review_id,
            status=ReviewStatusEnum.ESCALATED,
            reviewer_notes=request_body.reviewer_notes or f"Escalated to {request_body.escalate_to or 'senior agent'}",
        )

        logger.warning(f"Review {review_id} ESCALATED for email {email_id}")

        return ReviewActionResponse(
            review_id=review_id,
            email_id=email_id,
            action="escalated",
            status=EmailStatusEnum.AWAITING_REVIEW.value,
            response_sent=False,
            message=f"Review escalated to {request_body.escalate_to or 'senior agent'}.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Escalate review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
