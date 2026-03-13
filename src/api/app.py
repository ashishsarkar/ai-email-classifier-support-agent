"""
FastAPI Application Factory

This module creates and configures the FastAPI application with all routes and middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.api.routes import health, emails, reviews, metrics, ui
from src.core.config import settings
from src.core.logging import setup_logging
from src.core.tracing import setup_langsmith
from src.db.database import init_db
from src.graph.workflow import create_workflow
from src.nodes.factory import get_all_nodes

# Setup logging
setup_logging()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Customer Support Email Agent",
        description="A LangGraph-based intelligent customer support email agent",
        version="0.1.0",
        debug=settings.DEBUG,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup Jinja2 templates
    env = Environment(loader=FileSystemLoader("src/templates"))
    app.state.templates = env

    # Register routers
    app.include_router(health.router)
    app.include_router(emails.router)
    app.include_router(reviews.router)
    app.include_router(metrics.router)
    app.include_router(ui.router)

    @app.on_event("startup")
    async def startup_event():
        """Initialize database, LangSmith tracing, and workflow on startup."""
        try:
            # Initialize LangSmith tracing (must be before LangGraph workflow)
            langsmith_active = setup_langsmith()
            app.state.langsmith_enabled = langsmith_active

            logger.info("Initializing database...")
            init_db()
            logger.info("Database initialized")

            logger.info("Building LangGraph workflow...")
            nodes = get_all_nodes()
            workflow_app = create_workflow(nodes)
            app.state.workflow = workflow_app
            logger.info("Workflow initialized")
        except Exception as e:
            logger.error(f"Startup error: {e}", exc_info=True)
            raise

    @app.get("/", response_class=RedirectResponse)
    async def root():
        """Redirect to test page."""
        return RedirectResponse(url="/ui/test")

    return app


# Create the app instance
app = create_app()
