"""
Pytest configuration and fixtures.

This module provides shared fixtures for all tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="session")
def app():
    """Create and return a FastAPI app for testing."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Create and return a test client."""
    return TestClient(app)
