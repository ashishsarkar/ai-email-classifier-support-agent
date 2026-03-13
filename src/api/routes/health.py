"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter
from loguru import logger

from src.services.db_service import DatabaseService
from src.services.llm_service import LLMService

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """Check application and database health."""
    try:
        db_service = DatabaseService()
        db_health = await db_service.health_check()

        llm_service = LLMService()
        llm_health = await llm_service.health_check()

        # Overall status: unhealthy if any critical service is down
        overall = "healthy"
        if llm_health.get("status") == "unhealthy":
            overall = "unhealthy"
        elif llm_health.get("status") == "degraded":
            overall = "degraded"

        return {
            "status": overall,
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_health,
            "llm": llm_health,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@router.get("/health/circuit-breaker", tags=["Health"])
async def circuit_breaker_status():
    """Get circuit breaker status and failover metrics."""
    llm_service = LLMService()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "circuit_breaker": llm_service.circuit_breaker.get_status(),
        "primary_model": llm_service.primary_model,
        "fallback_model": llm_service.fallback_model,
        "fallback_configured": llm_service.fallback_client is not None,
    }
