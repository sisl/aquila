"""Tests for the host backend client_api service (HTTP calls to satellites)."""

import sys
from pathlib import Path
from unittest import mock

import pytest
import httpx

# Add the host backend to sys.path.
_HOST_BACKEND = Path(__file__).resolve().parent.parent / "vllm_cluster_manager" / "assets" / "host" / "backend"
if str(_HOST_BACKEND) not in sys.path:
    sys.path.insert(0, str(_HOST_BACKEND))

import app.services.client_api as client_api
from app.services.client_api import _satellite_url, upload_local_model_file
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# _satellite_url
# ---------------------------------------------------------------------------


class TestSatelliteUrl:
    def test_with_port(self):
        url = _satellite_url("10.0.0.1", 9000, "/health")
        assert url == "http://10.0.0.1:9000/health"

    def test_with_none_port_uses_default(self):
        url = _satellite_url("10.0.0.1", None, "/health")
        # Falls back to settings.satellite_port (9000 by default)
        assert url.startswith("http://10.0.0.1:")
        assert url.endswith("/health")

    def test_path_preserved(self):
        url = _satellite_url("10.0.0.1", 9000, "/deployments/start")
        assert url.endswith("/deployments/start")


# ---------------------------------------------------------------------------
# Streaming relay for local-model uploads
# ---------------------------------------------------------------------------


def _mock_transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_upload_local_model_file_streams_body():
    """The relay must forward the body iterator, path param, and content-length."""
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query_path"] = request.url.params.get("path")
        seen["content_length"] = request.headers.get("content-length")
        seen["body"] = request.content
        return httpx.Response(200, json={"status": "ok", "received_bytes": 6})

    async def stream():
        yield b"abc"
        yield b"def"

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        result = await upload_local_model_file(
            "10.0.0.1", 9000, "sid123", "sub/w.bin", stream(), content_length="6"
        )

    assert result == {"status": "ok", "received_bytes": 6}
    assert seen["method"] == "PUT"
    assert seen["path"] == "/local-models/upload/sid123/file"
    assert seen["query_path"] == "sub/w.bin"
    # httpx must honor the caller-supplied content-length with an iterator body
    # (the design relies on it passing through end-to-end).
    assert seen["content_length"] == "6"
    assert seen["body"] == b"abcdef"


@pytest.mark.anyio
async def test_stream_log_download_yields_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/deployments/logs/download"
        assert request.url.params.get("key") == "org/model:8001"
        return httpx.Response(200, content=b"[2026-06-11 08:00:32] line1\nline2\n")

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        chunks = [
            chunk
            async for chunk in client_api.stream_log_download(
                "10.0.0.1", 9000, "org/model:8001"
            )
        ]
    assert b"".join(chunks) == b"[2026-06-11 08:00:32] line1\nline2\n"


@pytest.mark.anyio
async def test_stream_log_download_propagates_404():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No log file for this deployment."})

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        with pytest.raises(HTTPException) as excinfo:
            async for _ in client_api.stream_log_download("10.0.0.1", 9000, "k:1"):
                pass
    assert excinfo.value.status_code == 404
    assert "No log file" in excinfo.value.detail


@pytest.mark.anyio
async def test_upload_local_model_file_propagates_client_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "already exists"})

    async def stream():
        yield b"x"

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        with pytest.raises(HTTPException) as excinfo:
            await upload_local_model_file("10.0.0.1", 9000, "sid", "a", stream())
    assert excinfo.value.status_code == 409
    assert "already exists" in excinfo.value.detail
