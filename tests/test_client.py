"""Tests for the client (satellite) application."""

import hashlib
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add client app to sys.path.
_CLIENT_DIR = Path(__file__).resolve().parent.parent / "vllm_cluster_manager" / "assets" / "client"
if str(_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIENT_DIR))

# Stub out consul registration before importing the app (it runs on import via lifespan).
with mock.patch.dict("sys.modules", {"app.consul": mock.MagicMock()}):
    from app.main import (
        _venv_hash,
        StartRequest,
        StopRequest,
        app,
    )

from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# _venv_hash
# ---------------------------------------------------------------------------


class TestVenvHash:
    def test_deterministic(self):
        h1 = _venv_hash("0.8.5", "model:8000")
        h2 = _venv_hash("0.8.5", "model:8000")
        assert h1 == h2

    def test_different_versions(self):
        h1 = _venv_hash("0.8.5", "model:8000")
        h2 = _venv_hash("0.9.0", "model:8000")
        assert h1 != h2

    def test_different_keys(self):
        h1 = _venv_hash("0.8.5", "model:8000")
        h2 = _venv_hash("0.8.5", "model:9000")
        assert h1 != h2

    def test_length(self):
        h = _venv_hash("0.8.5")
        assert len(h) == 16

    def test_matches_sha256(self):
        version = "0.8.5"
        key = "model:8000"
        canonical = version + "\n" + key
        expected = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        assert _venv_hash(version, key) == expected


# ---------------------------------------------------------------------------
# StartRequest / StopRequest Pydantic models
# ---------------------------------------------------------------------------


class TestStartRequest:
    def test_minimal(self):
        r = StartRequest(model_name="llama", port=8000, gpu_memory_fraction=0.9)
        assert r.gpu_ids is None
        assert r.vllm_version is None
        assert r.extra_packages is None

    def test_full(self):
        r = StartRequest(
            model_name="llama",
            port=8000,
            gpu_memory_fraction=0.9,
            gpu_ids=[0, 1],
            tensor_parallel_size=2,
            extra_args=["--max-model-len", "4096"],
            env_vars=[{"key": "HF_TOKEN", "value": "tok"}],
            vllm_version="0.8.5",
            extra_packages=["transformers"],
        )
        assert r.gpu_ids == [0, 1]
        assert r.vllm_version == "0.8.5"


class TestStopRequest:
    def test_basic(self):
        r = StopRequest(key="llama:8000")
        assert r.key == "llama:8000"


# ---------------------------------------------------------------------------
# Client HTTP endpoints (no external dependencies)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_deployments_list_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data


@pytest.mark.anyio
async def test_deployment_status_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "deployments" in data


@pytest.mark.anyio
async def test_deployment_logs_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments/logs", params={"key": "nonexistent:1234"})
        assert resp.status_code == 404


@pytest.mark.anyio
async def test_stop_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/deployments/stop", json={"key": "no-such:1234"})
        assert resp.status_code == 404


@pytest.mark.anyio
async def test_venvs_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/venvs")
        assert resp.status_code == 200
        data = resp.json()
        assert "venvs" in data


@pytest.mark.anyio
async def test_delete_venv_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/venvs/nonexistent")
        assert resp.status_code == 404


@pytest.mark.anyio
async def test_packages_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/packages")
        assert resp.status_code == 200
        data = resp.json()
        assert "packages" in data


@pytest.mark.anyio
async def test_delete_package_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/packages/nonexistent")
        assert resp.status_code == 404
