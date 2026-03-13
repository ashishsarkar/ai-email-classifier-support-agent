"""Database service for repository pattern."""

from datetime import datetime
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.db.models import (
    Customer,
    Email,
    EmailResponse,
    HumanReview,
    FollowUp,
    EmailStatusEnum,
    ReviewStatusEnum,
    ReviewReasonEnum,
    FollowUpTypeEnum,
)
from src.schemas.email import EmailIn
from src.services.base import BaseService


class DatabaseService(BaseService):
    """Service for database operations."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize database service.

        Args:
            db: Optional SQLAlchemy session. If None, will create new ones.
        """
        self.db = db

    def _get_session(self) -> Session:
        """Get database session."""
        return self.db or SessionLocal()

    async def get_or_create_customer(self, email: str, name: Optional[str] = None) -> Customer:
        """Get existing customer or create new one.

        Args:
            email: Customer email
            name: Optional customer name

        Returns:
            Customer object
        """
        try:
            db = self._get_session()
            customer = db.query(Customer).filter(Customer.email == email).first()

            if not customer:
                customer = Customer(email=email, name=name)
                db.add(customer)
                db.commit()
                db.refresh(customer)
                logger.info(f"Created new customer: {email}")
            else:
                logger.debug(f"Found existing customer: {email}")

            return customer

        except Exception as e:
            logger.error(f"Failed to get/create customer: {e}")
            raise

    async def create_email(self, email_in: EmailIn, customer_id: Optional[int] = None) -> Email:
        """Create new email record.

        Args:
            email_in: Email input schema
            customer_id: Optional customer ID

        Returns:
            Created Email object
        """
        try:
            db = self._get_session()
            email = Email(
                customer_id=customer_id,
                sender=email_in.sender,
                subject=email_in.subject,
                body=email_in.body,
                html_body=email_in.html_body,
                message_id=email_in.message_id,
                received_at=email_in.received_at,
            )
            db.add(email)
            db.commit()
            db.refresh(email)
            logger.info(f"Created email record: {email.id} from {email.sender}")
            return email

        except Exception as e:
            logger.error(f"Failed to create email: {e}")
            raise

    async def update_email_status(self, email_id: int, status: EmailStatusEnum, error_msg: Optional[str] = None) -> Email:
        """Update email status.

        Args:
            email_id: Email ID
            status: New status
            error_msg: Optional error message

        Returns:
            Updated Email object
        """
        try:
            db = self._get_session()
            email = db.query(Email).filter(Email.id == email_id).first()

            if not email:
                logger.warning(f"Email {email_id} not found")
                return None

            email.status = status
            if error_msg:
                email.error_message = error_msg
            if status == EmailStatusEnum.PROCESSING:
                email.processed_at = datetime.utcnow()
            email.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(email)
            logger.debug(f"Updated email {email_id} status to {status}")
            return email

        except Exception as e:
            logger.error(f"Failed to update email status: {e}")
            raise

    async def update_email_classification(
        self, email_id: int, category: str, confidence_score: float, priority: str
    ) -> Email:
        """Update email classification.

        Args:
            email_id: Email ID
            category: Email category
            confidence_score: Classification confidence
            priority: Priority level

        Returns:
            Updated Email object
        """
        try:
            db = self._get_session()
            email = db.query(Email).filter(Email.id == email_id).first()

            if not email:
                logger.warning(f"Email {email_id} not found")
                return None

            email.category = category
            email.confidence_score = confidence_score
            email.priority = priority
            email.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(email)
            logger.debug(f"Updated email {email_id} classification: {category}")
            return email

        except Exception as e:
            logger.error(f"Failed to update email classification: {e}")
            raise

    async def create_response(
        self,
        email_id: int,
        response_text: str,
        model_used: str,
        tokens_used: int,
        confidence_score: float,
        requires_review: bool = False,
    ) -> EmailResponse:
        """Create email response record.

        Args:
            email_id: Email ID
            response_text: Response text
            model_used: Model used for generation
            tokens_used: Tokens consumed
            confidence_score: Confidence score
            requires_review: Whether response needs review

        Returns:
            Created EmailResponse object
        """
        try:
            db = self._get_session()
            response = EmailResponse(
                email_id=email_id,
                response_text=response_text,
                model_used=model_used,
                tokens_used=tokens_used,
                confidence_score=confidence_score,
                requires_review=requires_review,
            )
            db.add(response)
            db.commit()
            db.refresh(response)
            logger.info(f"Created response for email {email_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to create response: {e}")
            raise

    async def create_review(
        self, email_id: int, reason: ReviewReasonEnum, notes: Optional[str] = None
    ) -> HumanReview:
        """Create human review task.

        Args:
            email_id: Email ID
            reason: Reason for review
            notes: Optional notes

        Returns:
            Created HumanReview object
        """
        try:
            db = self._get_session()
            review = HumanReview(email_id=email_id, reason=reason, notes=notes)
            db.add(review)
            db.commit()
            db.refresh(review)
            logger.info(f"Created review task for email {email_id}")
            return review

        except Exception as e:
            logger.error(f"Failed to create review task: {e}")
            raise

    async def update_review_status(
        self, review_id: int, status: ReviewStatusEnum, approved_response: Optional[str] = None
    ) -> HumanReview:
        """Update review task status.

        Args:
            review_id: Review ID
            status: New status
            approved_response: Optional approved response text

        Returns:
            Updated HumanReview object
        """
        try:
            db = self._get_session()
            review = db.query(HumanReview).filter(HumanReview.id == review_id).first()

            if not review:
                logger.warning(f"Review {review_id} not found")
                return None

            review.status = status
            if approved_response:
                review.approved_response = approved_response
            if status == ReviewStatusEnum.IN_PROGRESS:
                review.started_at = datetime.utcnow()
            elif status in [ReviewStatusEnum.APPROVED, ReviewStatusEnum.REJECTED]:
                review.completed_at = datetime.utcnow()

            db.commit()
            db.refresh(review)
            logger.debug(f"Updated review {review_id} status to {status}")
            return review

        except Exception as e:
            logger.error(f"Failed to update review status: {e}")
            raise

    async def create_followup(
        self, email_id: int, followup_type: FollowUpTypeEnum, scheduled_for: datetime
    ) -> FollowUp:
        """Create follow-up task.

        Args:
            email_id: Email ID
            followup_type: Type of follow-up
            scheduled_for: When to execute

        Returns:
            Created FollowUp object
        """
        try:
            db = self._get_session()
            followup = FollowUp(
                email_id=email_id, followup_type=followup_type, scheduled_for=scheduled_for
            )
            db.add(followup)
            db.commit()
            db.refresh(followup)
            logger.info(f"Created follow-up for email {email_id}: {followup_type}")
            return followup

        except Exception as e:
            logger.error(f"Failed to create follow-up: {e}")
            raise

    async def get_pending_reviews(self) -> List[HumanReview]:
        """Get all pending review tasks with associated email data.

        Returns:
            List of HumanReview objects with status PENDING or IN_PROGRESS
        """
        try:
            db = self._get_session()
            reviews = (
                db.query(HumanReview)
                .filter(
                    HumanReview.status.in_([
                        ReviewStatusEnum.PENDING,
                        ReviewStatusEnum.IN_PROGRESS,
                    ])
                )
                .order_by(HumanReview.created_at.desc())
                .all()
            )
            return reviews
        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []

    async def get_review_by_id(self, review_id: int) -> Optional[HumanReview]:
        """Get a single review by ID.

        Args:
            review_id: Review ID

        Returns:
            HumanReview object or None
        """
        try:
            db = self._get_session()
            return db.query(HumanReview).filter(HumanReview.id == review_id).first()
        except Exception as e:
            logger.error(f"Failed to get review {review_id}: {e}")
            return None

    async def update_review_with_decision(
        self,
        review_id: int,
        status: ReviewStatusEnum,
        approved_response: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
    ) -> Optional[HumanReview]:
        """Update review with human decision (approve/reject/escalate).

        Args:
            review_id: Review ID
            status: New status (APPROVED, REJECTED, ESCALATED)
            approved_response: Optional edited response text (for APPROVED)
            reviewer_notes: Optional reviewer notes

        Returns:
            Updated HumanReview object
        """
        try:
            db = self._get_session()
            review = db.query(HumanReview).filter(HumanReview.id == review_id).first()

            if not review:
                logger.warning(f"Review {review_id} not found")
                return None

            review.status = status
            review.completed_at = datetime.utcnow()

            if approved_response:
                review.approved_response = approved_response
            if reviewer_notes:
                review.reviewer_notes = reviewer_notes

            db.commit()
            db.refresh(review)
            logger.info(f"Review {review_id} updated to {status}")
            return review

        except Exception as e:
            logger.error(f"Failed to update review decision: {e}")
            raise

    async def get_email_with_history(self, email_id: int) -> Optional[Email]:
        """Get email with full history/context.

        Args:
            email_id: Email ID

        Returns:
            Email object with related data loaded
        """
        try:
            db = self._get_session()
            email = db.query(Email).filter(Email.id == email_id).first()
            return email

        except Exception as e:
            logger.error(f"Failed to get email with history: {e}")
            return None

    async def get_customer_emails(self, customer_id: int, limit: int = 10) -> List[Email]:
        """Get customer's previous emails.

        Args:
            customer_id: Customer ID
            limit: Max emails to return

        Returns:
            List of Email objects
        """
        try:
            db = self._get_session()
            emails = (
                db.query(Email)
                .filter(Email.customer_id == customer_id)
                .order_by(Email.received_at.desc())
                .limit(limit)
                .all()
            )
            return emails

        except Exception as e:
            logger.error(f"Failed to get customer emails: {e}")
            return []

    async def health_check(self) -> dict:
        """Check database health."""
        try:
            db = self._get_session()
            db.execute("SELECT 1")
            return {"status": "healthy", "service": "database"}
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "service": "database", "error": str(e)}
