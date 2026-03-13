"""UI page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from loguru import logger

router = APIRouter(tags=["UI"])


@router.get("/", response_class=HTMLResponse)
async def test_email_page(request: Request):
    """Render test email submission page."""
    return request.app.state.templates.get_template("test_email.html").render()


@router.get("/ui/inbox", response_class=HTMLResponse)
async def inbox_page(request: Request):
    """Render inbox dashboard page."""
    return request.app.state.templates.get_template("inbox.html").render()


@router.get("/ui/reviews", response_class=HTMLResponse)
async def reviews_page(request: Request):
    """Render human review dashboard page."""
    return request.app.state.templates.get_template("reviews.html").render()


@router.get("/ui/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    """Render metrics monitoring dashboard page."""
    return request.app.state.templates.get_template("metrics.html").render()


@router.get("/ui/test", response_class=HTMLResponse)
async def test_email_page_alt(request: Request):
    """Alias for test email page."""
    return request.app.state.templates.get_template("test_email.html").render()
