"""Metrics and analytics API endpoints.

Provides aggregated metrics for the monitoring dashboard:
- Classification accuracy & confidence score analysis
- Response times (pipeline, per-node, review wait time)
- Review rates (auto-send vs HITL, approval/rejection rates)
- Category & priority distribution
- Confidence improvement insights
- Token usage & cost tracking
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from sqlalchemy import func, case, and_, or_

from src.db.database import SessionLocal
from src.db.models import (
    Email,
    EmailResponse,
    HumanReview,
    FollowUp,
    EmailStatusEnum,
    ReviewStatusEnum,
    ReviewReasonEnum,
)

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])


@router.get("/overview")
async def get_metrics_overview(days: int = Query(default=30, ge=1, le=365)):
    """Get high-level metrics overview.

    Returns key KPIs: total emails, auto-send rate, review rate,
    avg confidence, avg response time, approval rate.
    """
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Total emails in period
        total_emails = (
            session.query(func.count(Email.id))
            .filter(Email.created_at >= cutoff)
            .scalar()
        ) or 0

        # Status distribution
        status_counts = dict(
            session.query(Email.status, func.count(Email.id))
            .filter(Email.created_at >= cutoff)
            .group_by(Email.status)
            .all()
        )

        responded = status_counts.get(EmailStatusEnum.RESPONDED, 0)
        awaiting = status_counts.get(EmailStatusEnum.AWAITING_REVIEW, 0)
        failed = status_counts.get(EmailStatusEnum.FAILED, 0)

        # Review stats
        total_reviews = (
            session.query(func.count(HumanReview.id))
            .filter(HumanReview.created_at >= cutoff)
            .scalar()
        ) or 0

        review_status_counts = dict(
            session.query(HumanReview.status, func.count(HumanReview.id))
            .filter(HumanReview.created_at >= cutoff)
            .group_by(HumanReview.status)
            .all()
        )

        approved = review_status_counts.get(ReviewStatusEnum.APPROVED, 0)
        rejected = review_status_counts.get(ReviewStatusEnum.REJECTED, 0)
        escalated = review_status_counts.get(ReviewStatusEnum.ESCALATED, 0)

        # Auto-send = responded emails that had no review
        auto_sent = (
            session.query(func.count(Email.id))
            .outerjoin(HumanReview, Email.id == HumanReview.email_id)
            .filter(
                Email.created_at >= cutoff,
                Email.status == EmailStatusEnum.RESPONDED,
                HumanReview.id.is_(None),
            )
            .scalar()
        ) or 0

        # Confidence score stats
        confidence_stats = (
            session.query(
                func.avg(Email.confidence_score),
                func.min(Email.confidence_score),
                func.max(Email.confidence_score),
                func.count(case((Email.confidence_score < 0.6, 1))),
                func.count(case((Email.confidence_score >= 0.8, 1))),
            )
            .filter(
                Email.created_at >= cutoff,
                Email.confidence_score.isnot(None),
            )
            .first()
        )

        avg_confidence = round(float(confidence_stats[0] or 0), 3)
        min_confidence = round(float(confidence_stats[1] or 0), 3)
        max_confidence = round(float(confidence_stats[2] or 0), 3)
        low_confidence_count = int(confidence_stats[3] or 0)
        high_confidence_count = int(confidence_stats[4] or 0)

        # Token usage
        token_stats = (
            session.query(
                func.sum(EmailResponse.tokens_used),
                func.avg(EmailResponse.tokens_used),
            )
            .filter(EmailResponse.created_at >= cutoff)
            .first()
        )

        total_tokens = int(token_stats[0] or 0)
        avg_tokens = round(float(token_stats[1] or 0), 1)

        # Average review wait time (created_at to completed_at)
        avg_review_time = (
            session.query(
                func.avg(
                    func.julianday(HumanReview.completed_at) - func.julianday(HumanReview.created_at)
                )
            )
            .filter(
                HumanReview.created_at >= cutoff,
                HumanReview.completed_at.isnot(None),
            )
            .scalar()
        )
        # Convert from days to minutes
        avg_review_wait_minutes = round(float(avg_review_time or 0) * 24 * 60, 1)

        session.close()

        # Calculate rates
        auto_send_rate = round((auto_sent / total_emails * 100) if total_emails > 0 else 0, 1)
        review_rate = round((total_reviews / total_emails * 100) if total_emails > 0 else 0, 1)
        approval_rate = round((approved / total_reviews * 100) if total_reviews > 0 else 0, 1)
        resolution_rate = round((responded / total_emails * 100) if total_emails > 0 else 0, 1)

        return {
            "period_days": days,
            "total_emails": total_emails,
            "status_distribution": {
                "responded": responded,
                "awaiting_review": awaiting,
                "failed": failed,
                "pending": status_counts.get(EmailStatusEnum.PENDING, 0),
                "processing": status_counts.get(EmailStatusEnum.PROCESSING, 0),
            },
            "rates": {
                "auto_send_rate": auto_send_rate,
                "review_rate": review_rate,
                "approval_rate": approval_rate,
                "resolution_rate": resolution_rate,
                "rejection_rate": round((rejected / total_reviews * 100) if total_reviews > 0 else 0, 1),
            },
            "confidence": {
                "average": avg_confidence,
                "min": min_confidence,
                "max": max_confidence,
                "low_confidence_count": low_confidence_count,
                "high_confidence_count": high_confidence_count,
                "low_confidence_rate": round((low_confidence_count / total_emails * 100) if total_emails > 0 else 0, 1),
            },
            "reviews": {
                "total": total_reviews,
                "approved": approved,
                "rejected": rejected,
                "escalated": escalated,
                "pending": review_status_counts.get(ReviewStatusEnum.PENDING, 0),
                "avg_wait_time_minutes": avg_review_wait_minutes,
            },
            "tokens": {
                "total_used": total_tokens,
                "avg_per_email": avg_tokens,
            },
        }

    except Exception as e:
        logger.error(f"Metrics overview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/category-distribution")
async def get_category_distribution(days: int = Query(default=30, ge=1, le=365)):
    """Get email category distribution with confidence scores per category."""
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        results = (
            session.query(
                Email.category,
                func.count(Email.id).label("count"),
                func.avg(Email.confidence_score).label("avg_confidence"),
                func.min(Email.confidence_score).label("min_confidence"),
                func.count(case((Email.confidence_score < 0.6, 1))).label("low_confidence_count"),
            )
            .filter(
                Email.created_at >= cutoff,
                Email.category.isnot(None),
            )
            .group_by(Email.category)
            .order_by(func.count(Email.id).desc())
            .all()
        )

        total = sum(r.count for r in results) or 1

        categories = [
            {
                "category": r.category or "uncategorized",
                "count": r.count,
                "percentage": round(r.count / total * 100, 1),
                "avg_confidence": round(float(r.avg_confidence or 0), 3),
                "min_confidence": round(float(r.min_confidence or 0), 3),
                "low_confidence_count": r.low_confidence_count,
            }
            for r in results
        ]

        session.close()
        return {"period_days": days, "categories": categories, "total": total}

    except Exception as e:
        logger.error(f"Category distribution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/priority-distribution")
async def get_priority_distribution(days: int = Query(default=30, ge=1, le=365)):
    """Get email priority distribution."""
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        results = (
            session.query(
                Email.priority,
                func.count(Email.id).label("count"),
                func.avg(Email.confidence_score).label("avg_confidence"),
            )
            .filter(Email.created_at >= cutoff)
            .group_by(Email.priority)
            .order_by(func.count(Email.id).desc())
            .all()
        )

        total = sum(r.count for r in results) or 1

        priorities = [
            {
                "priority": str(r.priority) if r.priority else "unknown",
                "count": r.count,
                "percentage": round(r.count / total * 100, 1),
                "avg_confidence": round(float(r.avg_confidence or 0), 3),
            }
            for r in results
        ]

        session.close()
        return {"period_days": days, "priorities": priorities, "total": total}

    except Exception as e:
        logger.error(f"Priority distribution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/review-reasons")
async def get_review_reasons_breakdown(days: int = Query(default=30, ge=1, le=365)):
    """Get breakdown of why emails needed human review."""
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        results = (
            session.query(
                HumanReview.reason,
                func.count(HumanReview.id).label("count"),
                func.count(case((HumanReview.status == ReviewStatusEnum.APPROVED, 1))).label("approved"),
                func.count(case((HumanReview.status == ReviewStatusEnum.REJECTED, 1))).label("rejected"),
            )
            .filter(HumanReview.created_at >= cutoff)
            .group_by(HumanReview.reason)
            .order_by(func.count(HumanReview.id).desc())
            .all()
        )

        total = sum(r.count for r in results) or 1

        reasons = [
            {
                "reason": str(r.reason) if r.reason else "unknown",
                "count": r.count,
                "percentage": round(r.count / total * 100, 1),
                "approved": r.approved,
                "rejected": r.rejected,
                "approval_rate": round((r.approved / r.count * 100) if r.count > 0 else 0, 1),
            }
            for r in results
        ]

        session.close()
        return {"period_days": days, "reasons": reasons, "total": total}

    except Exception as e:
        logger.error(f"Review reasons failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confidence-analysis")
async def get_confidence_analysis(days: int = Query(default=30, ge=1, le=365)):
    """Analyze confidence scores to find improvement opportunities.

    Returns:
    - Confidence score distribution (buckets)
    - Low-confidence emails by category (where to improve prompts)
    - Confidence trend over time
    - Actionable improvement recommendations
    """
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Confidence score buckets
        buckets = (
            session.query(
                case(
                    (Email.confidence_score < 0.3, "very_low_0_30"),
                    (Email.confidence_score < 0.5, "low_30_50"),
                    (Email.confidence_score < 0.6, "below_threshold_50_60"),
                    (Email.confidence_score < 0.7, "moderate_60_70"),
                    (Email.confidence_score < 0.8, "good_70_80"),
                    (Email.confidence_score < 0.9, "high_80_90"),
                    else_="excellent_90_100",
                ).label("bucket"),
                func.count(Email.id).label("count"),
            )
            .filter(
                Email.created_at >= cutoff,
                Email.confidence_score.isnot(None),
            )
            .group_by("bucket")
            .all()
        )

        score_distribution = {b.bucket: b.count for b in buckets}

        # Low confidence by category (where prompts need improvement)
        low_confidence_categories = (
            session.query(
                Email.category,
                func.count(Email.id).label("count"),
                func.avg(Email.confidence_score).label("avg_confidence"),
            )
            .filter(
                Email.created_at >= cutoff,
                Email.confidence_score < 0.6,
                Email.category.isnot(None),
            )
            .group_by(Email.category)
            .order_by(func.count(Email.id).desc())
            .all()
        )

        problem_categories = [
            {
                "category": r.category,
                "low_confidence_count": r.count,
                "avg_confidence": round(float(r.avg_confidence or 0), 3),
            }
            for r in low_confidence_categories
        ]

        # Confidence trend (daily average over the period)
        daily_confidence = (
            session.query(
                func.date(Email.created_at).label("date"),
                func.avg(Email.confidence_score).label("avg_confidence"),
                func.count(Email.id).label("count"),
                func.count(case((Email.confidence_score < 0.6, 1))).label("low_count"),
            )
            .filter(
                Email.created_at >= cutoff,
                Email.confidence_score.isnot(None),
            )
            .group_by(func.date(Email.created_at))
            .order_by(func.date(Email.created_at))
            .all()
        )

        trend = [
            {
                "date": str(r.date),
                "avg_confidence": round(float(r.avg_confidence or 0), 3),
                "total_emails": r.count,
                "low_confidence_count": r.low_count,
            }
            for r in daily_confidence
        ]

        # Overall stats
        total_with_scores = sum(b.count for b in buckets) or 1
        low_count = sum(
            b.count for b in buckets
            if b.bucket in ["very_low_0_30", "low_30_50", "below_threshold_50_60"]
        )

        # Generate improvement recommendations
        recommendations = _generate_confidence_recommendations(
            score_distribution, problem_categories, low_count, total_with_scores
        )

        session.close()

        return {
            "period_days": days,
            "score_distribution": score_distribution,
            "problem_categories": problem_categories,
            "confidence_trend": trend,
            "summary": {
                "total_classified": total_with_scores,
                "low_confidence_total": low_count,
                "low_confidence_rate": round(low_count / total_with_scores * 100, 1),
            },
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Confidence analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-trend")
async def get_daily_trend(days: int = Query(default=14, ge=1, le=90)):
    """Get daily email volume and status trend."""
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        results = (
            session.query(
                func.date(Email.created_at).label("date"),
                func.count(Email.id).label("total"),
                func.count(case((Email.status == EmailStatusEnum.RESPONDED, 1))).label("responded"),
                func.count(case((Email.status == EmailStatusEnum.AWAITING_REVIEW, 1))).label("awaiting_review"),
                func.count(case((Email.status == EmailStatusEnum.FAILED, 1))).label("failed"),
                func.avg(Email.confidence_score).label("avg_confidence"),
            )
            .filter(Email.created_at >= cutoff)
            .group_by(func.date(Email.created_at))
            .order_by(func.date(Email.created_at))
            .all()
        )

        trend = [
            {
                "date": str(r.date),
                "total": r.total,
                "responded": r.responded,
                "awaiting_review": r.awaiting_review,
                "failed": r.failed,
                "avg_confidence": round(float(r.avg_confidence or 0), 3),
            }
            for r in results
        ]

        session.close()
        return {"period_days": days, "trend": trend}

    except Exception as e:
        logger.error(f"Daily trend failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/token-usage")
async def get_token_usage(days: int = Query(default=30, ge=1, le=365)):
    """Get token usage breakdown by model."""
    try:
        session = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=days)

        results = (
            session.query(
                EmailResponse.model_used,
                func.count(EmailResponse.id).label("count"),
                func.sum(EmailResponse.tokens_used).label("total_tokens"),
                func.avg(EmailResponse.tokens_used).label("avg_tokens"),
            )
            .filter(EmailResponse.created_at >= cutoff)
            .group_by(EmailResponse.model_used)
            .order_by(func.sum(EmailResponse.tokens_used).desc())
            .all()
        )

        models = [
            {
                "model": r.model_used or "unknown",
                "request_count": r.count,
                "total_tokens": int(r.total_tokens or 0),
                "avg_tokens_per_request": round(float(r.avg_tokens or 0), 1),
            }
            for r in results
        ]

        # Daily token trend
        daily_tokens = (
            session.query(
                func.date(EmailResponse.created_at).label("date"),
                func.sum(EmailResponse.tokens_used).label("total_tokens"),
                func.count(EmailResponse.id).label("count"),
            )
            .filter(EmailResponse.created_at >= cutoff)
            .group_by(func.date(EmailResponse.created_at))
            .order_by(func.date(EmailResponse.created_at))
            .all()
        )

        token_trend = [
            {
                "date": str(r.date),
                "total_tokens": int(r.total_tokens or 0),
                "request_count": r.count,
            }
            for r in daily_tokens
        ]

        session.close()
        return {"period_days": days, "by_model": models, "daily_trend": token_trend}

    except Exception as e:
        logger.error(f"Token usage failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _generate_confidence_recommendations(
    score_distribution: dict,
    problem_categories: list,
    low_count: int,
    total: int,
) -> list:
    """Generate actionable recommendations to improve confidence scores."""
    recommendations = []

    low_rate = (low_count / total * 100) if total > 0 else 0

    # Recommendation 1: High low-confidence rate
    if low_rate > 20:
        recommendations.append({
            "severity": "high",
            "title": "High low-confidence rate",
            "description": f"{low_rate:.1f}% of emails have confidence < 0.6. This means the AI is uncertain about classification for 1 in {int(100/max(low_rate,1))} emails.",
            "action": "Review and improve classification prompt templates with more few-shot examples. Consider adding structured JSON output mode.",
        })
    elif low_rate > 10:
        recommendations.append({
            "severity": "medium",
            "title": "Moderate low-confidence rate",
            "description": f"{low_rate:.1f}% of emails have confidence < 0.6.",
            "action": "Add few-shot examples for the most problematic categories to the classification prompt.",
        })

    # Recommendation 2: Problem categories
    if problem_categories:
        top_problem = problem_categories[0]
        recommendations.append({
            "severity": "high",
            "title": f"Category '{top_problem['category']}' has consistently low confidence",
            "description": f"{top_problem['low_confidence_count']} emails in '{top_problem['category']}' had low confidence (avg: {top_problem['avg_confidence']}).",
            "action": f"Add more training examples and clearer classification criteria for '{top_problem['category']}' in the prompt template. Consider if this category needs to be split into sub-categories.",
        })

    # Recommendation 3: Very low scores present
    very_low = score_distribution.get("very_low_0_30", 0)
    if very_low > 0:
        recommendations.append({
            "severity": "high",
            "title": f"{very_low} emails with very low confidence (< 0.3)",
            "description": "These emails are essentially unclassifiable by the current model. They likely contain new topics or formats the model hasn't seen.",
            "action": "Review these emails manually to identify new categories or edge cases. Consider adding a 'needs_triage' category for truly ambiguous emails.",
        })

    # Recommendation 4: Model upgrade suggestion
    excellent = score_distribution.get("excellent_90_100", 0)
    if total > 0 and (excellent / total) < 0.3:
        recommendations.append({
            "severity": "medium",
            "title": "Less than 30% of emails achieve excellent confidence (> 0.9)",
            "description": "The model may benefit from a stronger LLM or better prompting strategy.",
            "action": "Consider using the Claude fallback model for classification, or switch to structured output (JSON mode) for more consistent results.",
        })

    # Recommendation 5: Knowledge base gaps
    if problem_categories and len(problem_categories) > 1:
        cats = ", ".join(c["category"] for c in problem_categories[:3])
        recommendations.append({
            "severity": "medium",
            "title": "Multiple categories with low confidence",
            "description": f"Categories with issues: {cats}",
            "action": "Expand the knowledge base with more articles for these categories. The context_analysis node provides KB context that helps the model classify better.",
        })

    if not recommendations:
        recommendations.append({
            "severity": "low",
            "title": "Confidence scores look healthy",
            "description": "No significant issues detected with classification confidence.",
            "action": "Continue monitoring. Consider A/B testing different prompt templates to further optimize.",
        })

    return recommendations
