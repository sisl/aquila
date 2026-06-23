"""Tests for the host backend health endpoints."""

import sys
from pathlib import Path
from unittest import mock

import pytest

# Add the host backend to sys.path.
_HOST_BACKEND = Path(__file__).resolve().parent.parent / "aquila" / "assets" / "host" / "backend"
if str(_HOST_BACKEND) not in sys.path:
    sys.path.insert(0, str(_HOST_BACKEND))

from app.api.health import router

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

# Create a minimal app with just the health router for testing.
_app = FastAPI()
_app.include_router(router)


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_vllm_version_returns_cached():
    """When a cached version exists and is fresh, it should be returned without a network call."""
    import app.api.health as health_mod

    # Seed the cache.
    health_mod._latest_vllm_version = "0.8.5"
    health_mod._latest_vllm_version_fetched_at = 1e12  # far future

    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/vllm-version")
        assert resp.status_code == 200
        assert resp.json()["version"] == "0.8.5"

    # Clean up.
    health_mod._latest_vllm_version = ""
    health_mod._latest_vllm_version_fetched_at = 0.0


@pytest.mark.anyio
async def test_vllm_version_fetches_when_stale():
    """When cache is stale, the endpoint should attempt to fetch from GitHub."""
    import app.api.health as health_mod

    health_mod._latest_vllm_version = ""
    health_mod._latest_vllm_version_fetched_at = 0.0

    fake_response = b'{"tag_name": "v0.9.0"}'
    mock_urlopen = mock.MagicMock()
    mock_urlopen.__enter__ = mock.MagicMock(return_value=mock.MagicMock(read=lambda: fake_response))
    mock_urlopen.__exit__ = mock.MagicMock(return_value=False)
    mock_urlopen.read = lambda: fake_response

    with mock.patch("app.api.health.urlopen", return_value=mock_urlopen):
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/vllm-version")
            assert resp.status_code == 200
            assert resp.json()["version"] == "0.9.0"

    # Clean up.
    health_mod._latest_vllm_version = ""
    health_mod._latest_vllm_version_fetched_at = 0.0
