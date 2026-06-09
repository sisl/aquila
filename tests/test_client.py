"""Tests for the client (satellite) application."""

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
    import app.main as client_main
    from app.main import (
        _resolve_image_tag,
        _derived_image_tag,
        StartRequest,
        StopRequest,
        app,
    )

from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# _resolve_image_tag
# ---------------------------------------------------------------------------


class TestResolveImageTag:
    def test_release_version(self):
        image, resolved = _resolve_image_tag("0.8.5")
        assert image == "vllm/vllm-openai:v0.8.5"
        assert resolved == "0.8.5"

    def test_nightly(self):
        image, resolved = _resolve_image_tag("nightly")
        assert image == "vllm/vllm-openai:nightly"
        assert resolved == "nightly"

    def test_commit_hash(self):
        commit = "a" * 40
        image, resolved = _resolve_image_tag(commit)
        assert image == f"vllm/vllm-openai:nightly-{commit}"
        assert resolved == commit

    def test_blank_uses_latest(self):
        with mock.patch.object(client_main, "_get_latest_vllm_version", return_value="1.2.3"):
            image, resolved = _resolve_image_tag("")
        assert image == "vllm/vllm-openai:v1.2.3"
        assert resolved == "1.2.3"

    def test_none_uses_latest(self):
        with mock.patch.object(client_main, "_get_latest_vllm_version", return_value="1.2.3"):
            image, _ = _resolve_image_tag(None)
        assert image == "vllm/vllm-openai:v1.2.3"


class TestDerivedImageTag:
    def test_order_independent(self):
        a = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["transformers", "numpy"])
        b = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["numpy", "transformers"])
        assert a == b

    def test_varies_by_packages(self):
        a = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["transformers"])
        b = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["numpy"])
        assert a != b

    def test_repo_prefix(self):
        tag = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["transformers"])
        assert tag.startswith("vllm-cluster-manager/local:")


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
async def test_images_list():
    fake = mock.MagicMock()
    fake.images.list.return_value = []
    with mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/images")
            assert resp.status_code == 200
            assert "images" in resp.json()


@pytest.mark.anyio
async def test_delete_image_not_found():
    # Use the ImageNotFound class the module itself imported so the endpoint's
    # `except` matches it regardless of how docker was imported under test.
    fake = mock.MagicMock()
    fake.images.remove.side_effect = client_main.ImageNotFound("nope")
    with mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/images/nonexistent")
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
