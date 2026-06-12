"""Tests for the admin purge endpoint (backend factory reset)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# The client and the host backend both ship a regular package named `app`.
# This file runs first alphabetically, so import the backend's package in a
# clean slate, bind what we need, then drop the modules again so later test
# modules (e.g. test_client.py) can import the client's `app` package fresh.
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

import app.api.admin as admin  # noqa: E402
import app.api.nodes as nodes_api  # noqa: E402
import app.services.sync as sync  # noqa: E402
from app.db.session import get_session  # noqa: E402
from app.services.node_state import rogue_container_counts  # noqa: E402
from app.services.notify import _warned_expiring  # noqa: E402

for _name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
    del sys.modules[_name]
sys.modules.update(_saved_app_modules)
# Also restore sys.path: later test modules rely on insertion order to pick
# the right `app` package (client vs backend).
sys.path.remove(_backend_path)

from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


class _FakeSession:
    """Records delete statements; reports a fixed rowcount per table."""

    def __init__(self):
        self.deleted_tables: list[str] = []
        self.committed = False

    async def execute(self, stmt):
        self.deleted_tables.append(stmt.table.name)
        return SimpleNamespace(rowcount=2)

    async def commit(self):
        self.committed = True


def _seed_caches():
    sync._usage_last_seen[1] = {"prompt": 1}
    sync.live_usage[1] = {"tokens_per_second": 1.0}
    sync._deployment_fail_counts[1] = 3
    rogue_container_counts[1] = 1
    _warned_expiring.add((1, "2026-06-11"))


@pytest.mark.anyio
async def test_purge_database_deletes_all_tables_and_clears_caches():
    session = _FakeSession()
    _seed_caches()

    counts = await admin.purge_database(session)

    # FK-safe order: children (deployments, node_metrics) before nodes.
    assert session.deleted_tables == [
        "deployments",
        "node_metrics",
        "nodes",
        "deployment_configs",
    ]
    assert session.committed
    assert counts == {
        "deployments": 2,
        "node_metrics": 2,
        "nodes": 2,
        "deployment_configs": 2,
    }
    assert not sync._usage_last_seen
    assert not sync.live_usage
    assert not sync._deployment_fail_counts
    assert not rogue_container_counts
    assert not _warned_expiring


@pytest.mark.anyio
async def test_purge_endpoint_returns_counts_and_broadcasts():
    session = _FakeSession()

    test_app = FastAPI()
    test_app.include_router(admin.router, prefix="/admin")

    async def _override():
        yield session

    test_app.dependency_overrides[get_session] = _override

    with mock.patch.object(
        admin.manager, "broadcast", new=mock.AsyncMock()
    ) as broadcast:
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/purge")

    assert resp.status_code == 200
    assert resp.json()["purged"]["deployments"] == 2
    sent_types = {call.args[0]["type"] for call in broadcast.call_args_list}
    assert sent_types == {"deployments_changed", "nodes_changed"}


# ---------------------------------------------------------------------------
# DELETE /nodes/{id} (remove a single — possibly stale — node)
# ---------------------------------------------------------------------------


class _FakeNodeSession:
    """Fake session for delete_node: select -> deployment ids, deletes recorded."""

    def __init__(self, node):
        self.node = node
        self.deleted_tables: list[str] = []
        self.deleted_objects: list[object] = []
        self.committed = False

    async def get(self, model, node_id):
        return self.node if self.node and self.node.id == node_id else None

    async def execute(self, stmt):
        if stmt.__class__.__name__ == "Select":
            return SimpleNamespace(all=lambda: [(11,), (12,)])
        self.deleted_tables.append(stmt.table.name)
        return SimpleNamespace(rowcount=1)

    async def delete(self, obj):
        self.deleted_objects.append(obj)

    async def commit(self):
        self.committed = True


def _nodes_test_app(session):
    test_app = FastAPI()
    test_app.include_router(nodes_api.router, prefix="/nodes")

    async def _override():
        yield session

    test_app.dependency_overrides[get_session] = _override
    return test_app


@pytest.mark.anyio
async def test_delete_node_removes_rows_caches_and_consul():
    node = SimpleNamespace(id=7, hostname="stale-host")
    session = _FakeNodeSession(node)
    rogue_container_counts[7] = 2
    sync._node_fail_counts["stale-host"] = 5
    sync.live_usage[11] = {"tokens_per_second": 1.0}
    sync.pull_progress[12] = {"percent": 50}
    sync._usage_last_seen[11] = {"prompt": 1}
    _warned_expiring.add((11, "2026-06-12"))
    _warned_expiring.add((99, "2026-06-12"))  # unrelated; must survive

    with mock.patch.object(
        nodes_api.consul_service, "deregister_service"
    ) as deregister, mock.patch.object(
        nodes_api.manager, "broadcast", new=mock.AsyncMock()
    ) as broadcast:
        transport = ASGITransport(app=_nodes_test_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/nodes/7")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "deleted", "hostname": "stale-host", "deployments_deleted": 2}
    # Children deleted before the node row (deployments FK has no cascade).
    assert session.deleted_tables == ["deployments", "node_metrics"]
    assert session.deleted_objects == [node]
    assert session.committed
    deregister.assert_called_once_with("stale-host")
    assert 7 not in rogue_container_counts
    assert "stale-host" not in sync._node_fail_counts
    assert 11 not in sync.live_usage
    assert 12 not in sync.pull_progress
    assert 11 not in sync._usage_last_seen
    assert (11, "2026-06-12") not in _warned_expiring
    assert (99, "2026-06-12") in _warned_expiring
    _warned_expiring.discard((99, "2026-06-12"))
    sent_types = {call.args[0]["type"] for call in broadcast.call_args_list}
    assert sent_types == {"deployments_changed", "nodes_changed"}


@pytest.mark.anyio
async def test_delete_node_404():
    session = _FakeNodeSession(node=None)
    transport = ASGITransport(app=_nodes_test_app(session))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/nodes/123")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_delete_node_survives_consul_failure():
    node = SimpleNamespace(id=7, hostname="stale-host")
    session = _FakeNodeSession(node)
    with mock.patch.object(
        nodes_api.consul_service,
        "deregister_service",
        side_effect=RuntimeError("consul down"),
    ), mock.patch.object(nodes_api.manager, "broadcast", new=mock.AsyncMock()):
        transport = ASGITransport(app=_nodes_test_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/nodes/7")
    assert resp.status_code == 200
    assert session.committed
