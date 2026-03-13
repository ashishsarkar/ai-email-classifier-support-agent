"""Email processing and management endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, EmailStr
from sqlalchemy import desc, select

from src.db.database import SessionLocal
from src.db.models import Email, EmailResponse, FollowUp, HumanReview
from src.schemas.email import EmailIn
from src.services.db_service import DatabaseService

router = APIRouter(prefix="/api/emails", tags=["Emails"])


class TestEmailRequest(BaseModel):
    """Test email submission."""

    sender: EmailStr
    subject: str
    body: str


class TestEmailResponse(BaseModel):
    """Test email response."""

    email_id: int
    category: Optional[str]
    priority: Optional[str]
    confidence_score: Optional[float]
    generated_response: Optional[str]
    context_summary: Optional[str]
    needs_human_review: bool
    awaiting_review: bool = False
    review_id: Optional[int] = None
    status: str
    execution_time_ms: Optional[int]
    error_message: Optional[str]


class EmailListItem(BaseModel):
    """Email list item for inbox."""

    id: int
    sender: str
    subject: str
    category: Optional[str]
    priority: Optional[str]
    confidence_score: Optional[float]
    status: str
    received_at: datetime

    class Config:
        from_attributes = True


class EmailDetail(BaseModel):
    """Full email detail with all related data."""

    id: int
    sender: str
    subject: str
    body: str
    category: Optional[str]
    priority: Optional[str]
    confidence_score: Optional[float]
    status: str
    received_at: datetime
    processed_at: Optional[datetime]
    error_message: Optional[str]
    response: Optional[dict]
    review: Optional[dict]
    followups: list = []

    class Config:
        from_attributes = True


@router.post("/test", response_model=TestEmailResponse)
async def test_email(request_body: TestEmailRequest, request: Request):
    """Test email submission: creates email and runs workflow.

    Args:
        request_body: sender, subject, body
        request: FastAPI request object

    Returns:
        TestEmailResponse with classification, response, and status
    """
    try:
        logger.info(f"Test email from {request_body.sender}")

        # Create customer and email in database
        db_service = DatabaseService()
        customer = await db_service.get_or_create_customer(
            email=request_body.sender, name=request_body.sender.split("@")[0]
        )

        email_schema = EmailIn(
            sender=request_body.sender,
            subject=request_body.subject,
            body=request_body.body,
            html_body=None,
            received_at=datetime.utcnow(),
            message_id=f"test-{datetime.utcnow().timestamp()}",
        )
        email_db = await db_service.create_email(email_schema, customer.id)
        logger.info(f"Created email record: {email_db.id}")

        # Run workflow
        workflow = request.app.state.workflow
        logger.info(f"Running workflow for email {email_db.id}")
        final_state = await workflow.ainvoke({"email_id": email_db.id})

        # Extract response data
        return TestEmailResponse(
            email_id=email_db.id,
            category=final_state.get("category"),
            priority=final_state.get("priority"),
            confidence_score=final_state.get("confidence_score"),
            generated_response=final_state.get("generated_response"),
            context_summary=final_state.get("context_summary"),
            needs_human_review=final_state.get("needs_human_review", False),
            awaiting_review=final_state.get("awaiting_review", False),
            review_id=final_state.get("review_id"),
            status=final_state.get("status", "unknown"),
            execution_time_ms=final_state.get("execution_time_ms"),
            error_message=final_state.get("error_message"),
        )

    except Exception as e:
        logger.error(f"Test email failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=dict)
async def list_emails(
    page: int = 1,
    per_page: int = 20,
    status: Optional[str] = None,
):
    """List all emails with pagination.

    Args:
        page: page number (1-indexed)
        per_page: emails per page
        status: filter by status (optional)

    Returns:
        {emails: [...], total: N, page: N, pages: N}
    """
    try:
        session = SessionLocal()
        query = select(Email).order_by(desc(Email.received_at))

        # Filter by status if provided
        if status:
            query = query.where(Email.status == status)

        # Count total
        total_query = select(Email)
        if status:
            total_query = total_query.where(Email.status == status)
        result = session.execute(select(lambda: 1).select_from(Email))
        total = session.query(Email).count()
        if status:
            total = session.query(Email).filter(Email.status == status).count()

        # Pagination
        offset = (page - 1) * per_page
        result = session.execute(query.offset(offset).limit(per_page))
        emails = result.scalars().all()

        # Convert to list items
        email_items = [
            EmailListItem(
                id=e.id,
                sender=e.sender,
                subject=e.subject,
                category=e.category,
                priority=e.priority,
                confidence_score=e.confidence_score,
                status=e.status,
                received_at=e.received_at,
            )
            for e in emails
        ]

        session.close()

        pages = (total + per_page - 1) // per_page
        return {
            "emails": [item.model_dump() for item in email_items],
            "total": total,
            "page": page,
            "pages": pages,
            "per_page": per_page,
        }

    except Exception as e:
        logger.error(f"List emails failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{email_id}", response_model=EmailDetail)
async def get_email_detail(email_id: int):
    """Get full email detail with all related data.

    Args:
        email_id: email ID

    Returns:
        EmailDetail with email, response, review, and followups
    """
    try:
        session = SessionLocal()

        # Get email
        email = session.query(Email).filter(Email.id == email_id).first()
        if not email:
            session.close()
            raise HTTPException(status_code=404, detail="Email not found")

        # Get response if exists
        response = None
        response_record = (
            session.query(EmailResponse).filter(EmailResponse.email_id == email_id).first()
        )
        if response_record:
            response = {
                "id": response_record.id,
                "response_text": response_record.response_text,
                "response_subject": response_record.response_subject,
                "model_used": response_record.model_used,
                "tokens_used": response_record.tokens_used,
                "confidence_score": response_record.confidence_score,
                "generated_at": response_record.generated_at.isoformat()
                if response_record.generated_at
                else None,
                "sent_at": response_record.sent_at.isoformat()
                if response_record.sent_at
                else None,
            }

        # Get review if exists
        review = None
        review_record = (
            session.query(HumanReview).filter(HumanReview.email_id == email_id).first()
        )
        if review_record:
            review = {
                "id": review_record.id,
                "reason": review_record.reason,
                "status": review_record.status,
                "notes": review_record.notes,
                "approved_response": review_record.approved_response,
                "reviewer_notes": review_record.reviewer_notes,
                "assigned_to": review_record.assigned_to,
                "created_at": review_record.created_at.isoformat()
                if review_record.created_at
                else None,
            }

        # Get followups
        followups = session.query(FollowUp).filter(FollowUp.email_id == email_id).all()
        followups_list = [
            {
                "id": f.id,
                "followup_type": f.followup_type,
                "scheduled_for": f.scheduled_for.isoformat()
                if f.scheduled_for
                else None,
                "executed_at": f.executed_at.isoformat() if f.executed_at else None,
                "status": f.status,
                "result": f.result,
            }
            for f in followups
        ]

        session.close()

        return EmailDetail(
            id=email.id,
            sender=email.sender,
            subject=email.subject,
            body=email.body,
            category=email.category,
            priority=email.priority,
            confidence_score=email.confidence_score,
            status=email.status,
            received_at=email.received_at,
            processed_at=email.processed_at,
            error_message=email.error_message,
            response=response,
            review=review,
            followups=followups_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get email detail failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
