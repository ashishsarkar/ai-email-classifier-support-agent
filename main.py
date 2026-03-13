"""
Customer Support Email Agent - Main Entry Point

This module initializes and runs the FastAPI application with LangGraph integration.
"""

import uvicorn
from src.core.config import settings


def main() -> None:
    """Start the FastAPI application."""
    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()
