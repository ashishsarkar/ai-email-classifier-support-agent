"""
Health check tests.

Tests for the API health endpoints.
"""

import pytest


@pytest.mark.unit
def test_root_endpoint(client):
    """Test root endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
