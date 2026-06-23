"""Tests for the host backend client_api service (HTTP calls to satellites)."""

import sys
from pathlib import Path
from unittest import mock

import pytest
import httpx

# Add the host backend to sys.path.
_HOST_BACKEND = Path(__file__).resolve().parent.parent / "aquila" / "assets" / "host" / "backend"
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


# ---------------------------------------------------------------------------
# Warm cache RPCs
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_start_model_sends_warm_flags():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        seen.update(_json.loads(request.content))
        return httpx.Response(200, json={"status": "started"})

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        await client_api.start_model(
            "10.0.0.1", 9000, "org/model", 8000, 0.5,
            warm_offload=True, pinned=True,
        )
    assert seen["warm_offload"] is True
    assert seen["pinned"] is True


@pytest.mark.anyio
async def test_pause_model_posts_tier():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        seen["path"] = request.url.path
        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"status": "paused", "tier": "ram"})

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        result = await client_api.pause_model("10.0.0.1", 9000, "org/model:8000", "ram")
    assert seen["path"] == "/deployments/pause"
    assert seen["body"] == {"key": "org/model:8000", "tier": "ram"}
    assert result["tier"] == "ram"


@pytest.mark.anyio
async def test_plan_deployment_posts_params():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        seen["path"] = request.url.path
        seen["body"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"fits": True, "warm_enabled": True, "plan": [], "reason": None},
        )

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        result = await client_api.plan_deployment(
            "10.0.0.1", 9000, "org/model", 8000, 0.5, [0, 1]
        )
    assert seen["path"] == "/deployments/plan"
    assert seen["body"] == {
        "model_name": "org/model",
        "port": 8000,
        "gpu_memory_fraction": 0.5,
        "gpu_ids": [0, 1],
    }
    assert result["fits"] is True


@pytest.mark.anyio
async def test_resume_model_posts_key():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/deployments/resume"
        return httpx.Response(200, json={"status": "running", "key": "org/model:8000"})

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        result = await client_api.resume_model("10.0.0.1", 9000, "org/model:8000")
    assert result["status"] == "running"


@pytest.mark.anyio
async def test_push_node_config_sends_policy():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        seen["path"] = request.url.path
        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"status": "ok"})

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        await client_api.push_node_config("10.0.0.1", 9000, True, 20480, ["a:8000"])
    assert seen["path"] == "/config"
    assert seen["body"] == {
        "warm_offload_enabled": True,
        "ram_cache_limit_mb": 20480,
        "pins": ["a:8000"],
    }


@pytest.mark.anyio
async def test_list_warm_artifacts_returns_payload():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/warm-artifacts"
        return httpx.Response(200, json={"ram_sleepers": [{"pid": 1}], "disk_caches": []})

    with mock.patch.object(client_api, "get_client", return_value=_mock_transport(handler)):
        result = await client_api.list_warm_artifacts("10.0.0.1", 9000)
    assert result["ram_sleepers"] == [{"pid": 1}]
