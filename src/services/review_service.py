"""Human review service for managing escalation workflows."""

from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.db.models import HumanReview, ReviewStatusEnum
from src.services.base import BaseService


class ReviewService(BaseService):
    """Service for human review task management."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize review service.

        Args:
            db: Optional SQLAlchemy session
        """
        self.db = db

    def _get_session(self) -> Session:
        """Get database session."""
        return self.db or SessionLocal()

    async def get_pending_reviews(self, agent_id: Optional[str] = None, limit: int = 10) -> List[HumanReview]:
        """Get pending review tasks.

        Args:
            agent_id: Optional agent ID to filter by
            limit: Max reviews to return

        Returns:
            List of HumanReview objects
        """
        try:
            db = self._get_session()
            query = db.query(HumanReview).filter(HumanReview.status == ReviewStatusEnum.PENDING)

            if agent_id:
                query = query.filter(HumanReview.assigned_to == agent_id)

            reviews = query.limit(limit).all()
            logger.info(f"Retrieved {len(reviews)} pending reviews")
            return reviews

        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []

    async def assign_to_agent(self, review_id: int, agent_id: str) -> Optional[HumanReview]:
        """Assign review task to agent.

        Args:
            review_id: Review task ID
            agent_id: Agent ID

        Returns:
            Updated HumanReview or None
        """
        try:
            db = self._get_session()
            review = db.query(HumanReview).filter(HumanReview.id == review_id).first()

            if not review:
                logger.warning(f"Review {review_id} not found")
                return None

            review.assigned_to = agent_id
            db.commit()
            db.refresh(review)
            logger.info(f"Assigned review {review_id} to agent {agent_id}")
            return review

        except Exception as e:
            logger.error(f"Failed to assign review: {e}")
            return None

    async def health_check(self) -> dict:
        """Check review service health."""
        try:
            db = self._get_session()
            pending_count = db.query(HumanReview).filter(
                HumanReview.status == ReviewStatusEnum.PENDING
            ).count()
            return {
                "status": "healthy",
                "service": "review",
                "pending_reviews": pending_count,
            }
        except Exception as e:
            logger.error(f"Review service health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "review",
                "error": str(e),
            }
