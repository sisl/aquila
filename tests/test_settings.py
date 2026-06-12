"""Tests for the DB-backed runtime settings (registry, API, gateway gate)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# The client and the host backend both ship a regular package named `app`;
# import the backend's in a clean slate and restore afterwards (see
# tests/test_admin.py for the long-form rationale).
_HOST_BACKEND = (
    Path(__file__).resolve().parent.parent
    / "vllm_cluster_manager"
    / "assets"
    / "host"
    / "backend"
)
_backend_path = str(_HOST_BACKEND)
if _backend_path in sys.path:
    sys.path.remove(_backend_path)
sys.path.insert(0, _backend_path)

_saved_app_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == "app" or name.startswith("app.")
}
for _name in _saved_app_modules:
    del sys.modules[_name]

import app.api.gateway as gateway  # noqa: E402
import app.api.settings as settings_api  # noqa: E402
import app.services.runtime_settings as runtime_settings  # noqa: E402
from app.core.config import settings as env_settings  # noqa: E402
from app.db.session import get_session  # noqa: E402

for _name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
    del sys.modules[_name]
sys.modules.update(_saved_app_modules)
sys.path.remove(_backend_path)

from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
def _clean_overrides():
    runtime_settings._overrides.clear()
    yield
    runtime_settings._overrides.clear()


class TestRegistry:
    def test_env_seeded_defaults(self):
        with mock.patch.object(env_settings, "start_timeout_seconds", 999):
            assert runtime_settings.get_int("start_timeout_seconds") == 999

    def test_literal_defaults(self):
        assert runtime_settings.get_bool("gateway_enabled") is True
        assert runtime_settings.get_int("default_port") == 8001
        assert runtime_settings.get_float("default_gpu_fraction") == 0.5
        assert runtime_settings.get_str("default_vllm_version") == ""
        assert runtime_settings.get("default_max_failed_restarts") is None

    def test_override_wins(self):
        runtime_settings._overrides["default_port"] = 9123
        assert runtime_settings.get_int("default_port") == 9123
        assert runtime_settings.effective()["default_port"] == 9123

    def test_effective_contains_all_keys(self):
        effective = runtime_settings.effective()
        assert set(effective) == set(runtime_settings._REGISTRY)


class _FakeSettingsSession:
    """Records upserts for runtime_settings.apply."""

    def __init__(self):
        self.rows: dict[str, object] = {}
        self.committed = False

    async def get(self, model, key):
        if key in self.rows:
            return SimpleNamespace(key=key, value=self.rows[key])
        return None

    def add(self, obj):
        self.rows[obj.key] = obj.value

    async def commit(self):
        self.committed = True


def _settings_test_app(session):
    test_app = FastAPI()
    test_app.include_router(settings_api.router, prefix="/settings")

    async def _override():
        yield session

    test_app.dependency_overrides[get_session] = _override
    return test_app


class TestSettingsApi:
    @pytest.mark.anyio
    async def test_get_returns_effective(self):
        transport = ASGITransport(app=_settings_test_app(_FakeSettingsSession()))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["gateway_enabled"] is True
        assert body["default_port"] == 8001

    @pytest.mark.anyio
    async def test_put_applies_and_broadcasts(self):
        session = _FakeSettingsSession()
        with mock.patch.object(
            settings_api.manager, "broadcast", new=mock.AsyncMock()
        ) as broadcast:
            transport = ASGITransport(app=_settings_test_app(session))
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/settings",
                    json={"default_port": 9123, "gateway_enabled": False},
                )
        assert resp.status_code == 200
        assert resp.json()["default_port"] == 9123
        assert resp.json()["gateway_enabled"] is False
        assert session.committed
        assert session.rows == {"default_port": 9123, "gateway_enabled": False}
        assert runtime_settings.get_bool("gateway_enabled") is False
        broadcast.assert_awaited_once_with({"type": "settings_changed"})

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "payload",
        [
            {"default_port": 80},  # below 1024
            {"default_gpu_fraction": 1.5},
            {"default_duration_choice": "custom"},
            {"webhook_url": "ftp://nope"},
            {"unknown_key": 1},
            {"nodes_sync_interval_seconds": 0},
        ],
    )
    async def test_put_rejects_invalid(self, payload):
        transport = ASGITransport(app=_settings_test_app(_FakeSettingsSession()))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put("/settings", json=payload)
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_put_empty_rejected(self):
        transport = ASGITransport(app=_settings_test_app(_FakeSettingsSession()))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put("/settings", json={})
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_put_null_clears_max_restarts(self):
        session = _FakeSettingsSession()
        with mock.patch.object(settings_api.manager, "broadcast", new=mock.AsyncMock()):
            transport = ASGITransport(app=_settings_test_app(session))
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/settings", json={"default_max_failed_restarts": None}
                )
        assert resp.status_code == 200
        assert session.rows == {"default_max_failed_restarts": None}


class TestGatewayGate:
    def test_disabled_raises_503(self):
        runtime_settings._overrides["gateway_enabled"] = False
        with pytest.raises(Exception) as excinfo:
            gateway._require_gateway()
        assert getattr(excinfo.value, "status_code", None) == 503
        assert "disabled" in str(excinfo.value.detail)

    def test_enabled_passes(self):
        assert gateway._require_gateway() is None
