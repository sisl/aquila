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
    / "aquila"
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
import app.api.deployments as deployments_api  # noqa: E402
import app.api.nodes as nodes_api  # noqa: E402
import app.services.model_names as model_names  # noqa: E402
import app.services.sync as sync  # noqa: E402
from app.db.session import get_session  # noqa: E402
from app.services.node_state import (  # noqa: E402
    rogue_container_counts,
    rogue_process_counts,
)
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
    rogue_process_counts[1] = 2
    _warned_expiring.add((1, "2026-06-11"))


def _purge_app(session):
    test_app = FastAPI()
    test_app.include_router(admin.router, prefix="/admin")

    async def _override():
        yield session

    test_app.dependency_overrides[get_session] = _override
    return test_app


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
    assert not rogue_process_counts
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
    rogue_process_counts[7] = 3
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
    assert 7 not in rogue_process_counts
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


# ---------------------------------------------------------------------------
# Granular purge targets
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_purge_configs_only():
    session = _FakeSession()
    with mock.patch.object(admin.manager, "broadcast", new=mock.AsyncMock()) as broadcast:
        transport = ASGITransport(app=_purge_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/purge", json={"targets": ["configs"]})
    assert resp.status_code == 200
    assert session.deleted_tables == ["deployment_configs"]
    sent_types = {call.args[0]["type"] for call in broadcast.call_args_list}
    assert sent_types == {"deployments_changed"}


@pytest.mark.anyio
async def test_purge_nodes_force_includes_children():
    session = _FakeSession()
    _seed_caches()
    with mock.patch.object(admin.manager, "broadcast", new=mock.AsyncMock()):
        transport = ASGITransport(app=_purge_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/purge", json={"targets": ["nodes"]})
    assert resp.status_code == 200
    # Children deleted before nodes; configs untouched.
    assert session.deleted_tables == ["deployments", "node_metrics", "nodes"]
    assert not sync.live_usage
    assert not rogue_container_counts


@pytest.mark.anyio
async def test_purge_deployments_keeps_node_caches():
    session = _FakeSession()
    _seed_caches()
    with mock.patch.object(admin.manager, "broadcast", new=mock.AsyncMock()):
        transport = ASGITransport(app=_purge_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/purge", json={"targets": ["deployments"]})
    assert resp.status_code == 200
    assert session.deleted_tables == ["deployments"]
    assert not sync.live_usage  # deployment caches cleared
    assert rogue_container_counts  # node caches kept
    assert rogue_process_counts  # node caches kept
    rogue_container_counts.clear()
    rogue_process_counts.clear()
    sync._deployment_fail_counts.clear()
    _warned_expiring.clear()


@pytest.mark.anyio
async def test_purge_invalid_target_rejected():
    session = _FakeSession()
    transport = ASGITransport(app=_purge_app(session))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/admin/purge", json={"targets": ["everything"]})
    assert resp.status_code == 400
    assert session.deleted_tables == []


@pytest.mark.anyio
async def test_purge_empty_body_purges_everything():
    session = _FakeSession()
    with mock.patch.object(admin.manager, "broadcast", new=mock.AsyncMock()):
        transport = ASGITransport(app=_purge_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/purge")
    assert resp.status_code == 200
    assert session.deleted_tables == [
        "deployments",
        "node_metrics",
        "nodes",
        "deployment_configs",
    ]


# ---------------------------------------------------------------------------
# Container runtime resolution + per-node runtime endpoint
# ---------------------------------------------------------------------------


def _node_ns(**kw):
    base = dict(
        id=1, hostname="gpu-01", available_runtimes=["docker"], container_runtime=None
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestResolveRuntime:
    def test_explicit_override_when_available(self):
        node = _node_ns(
            available_runtimes=["docker", "podman"], container_runtime="podman"
        )
        assert deployments_api._resolve_runtime(node) == "podman"

    def test_explicit_override_ignored_when_unavailable(self):
        node = _node_ns(available_runtimes=["docker"], container_runtime="podman")
        assert deployments_api._resolve_runtime(node) == "docker"

    def test_preferred_breaks_tie(self):
        node = _node_ns(available_runtimes=["docker", "podman"])
        with mock.patch.object(
            deployments_api.runtime_settings, "get_str", return_value="podman"
        ):
            assert deployments_api._resolve_runtime(node) == "podman"

    def test_single_available_wins_over_preference(self):
        node = _node_ns(available_runtimes=["podman"])
        with mock.patch.object(
            deployments_api.runtime_settings, "get_str", return_value="docker"
        ):
            assert deployments_api._resolve_runtime(node) == "podman"

    def test_none_available_raises_409(self):
        node = _node_ns(available_runtimes=[])
        with pytest.raises(Exception) as excinfo:
            deployments_api._resolve_runtime(node)
        assert getattr(excinfo.value, "status_code", None) == 409
        assert "no container runtime" in str(excinfo.value.detail)


class _FakeRuntimeNodeSession:
    def __init__(self, node):
        self.node = node
        self.committed = False

    async def get(self, model, node_id):
        return self.node if self.node and self.node.id == node_id else None

    async def commit(self):
        self.committed = True

    async def refresh(self, obj):
        pass


@pytest.mark.anyio
async def test_set_node_runtime_persists_and_broadcasts():
    node = SimpleNamespace(
        id=7,
        hostname="gpu-01",
        ip_address="10.0.0.5",
        port=9000,
        status="healthy",
        maintenance=False,
        gpu_usage=[],
        disk_usage=None,
        default_pip_packages=[],
        installed_packages=[],
        available_runtimes=["docker", "podman"],
        container_runtime=None,
        rogue_container_count=None,
        last_heartbeat_at=None,
        created_at=None,
    )
    session = _FakeRuntimeNodeSession(node)
    with mock.patch.object(nodes_api.manager, "broadcast", new=mock.AsyncMock()) as broadcast:
        transport = ASGITransport(app=_nodes_test_app(session))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/nodes/7/runtime", json={"runtime": "podman"})
    assert resp.status_code == 200
    assert node.container_runtime == "podman"
    assert session.committed
    broadcast.assert_awaited_once_with({"type": "nodes_changed"})


@pytest.mark.anyio
async def test_set_node_runtime_rejects_unknown_value():
    session = _FakeRuntimeNodeSession(None)
    transport = ASGITransport(app=_nodes_test_app(session))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/nodes/7/runtime", json={"runtime": "lxc"})
    assert resp.status_code == 400


class _FakeManifestSession:
    """Resolves session.get() for the manifest endpoint by model name."""

    def __init__(self, deployment, node):
        self.deployment = deployment
        self.node = node

    async def get(self, model, obj_id):
        if model.__name__ == "Deployment":
            return self.deployment if self.deployment.id == obj_id else None
        return self.node


@pytest.mark.anyio
async def test_manifest_includes_container_runtime():
    deployment = SimpleNamespace(
        id=21,
        node_id=14,
        model_name="org/model",
        engine_args={},
        gpu_ids=[0],
        vllm_version="0.9.1",
        image_digest=None,
        container_runtime="podman",
        extra_args=[],
        extra_packages=[],
        lora_modules=[],
        env_vars=[],
        gpu_memory_fraction=0.5,
        tensor_parallel_size=None,
        owner="alice",
        created_at=None,
        duration_seconds=None,
        status="running",
    )
    node = SimpleNamespace(hostname="gpu-01", gpu_usage=[])
    test_app = FastAPI()
    test_app.include_router(deployments_api.router, prefix="/deployments")
    session = _FakeManifestSession(deployment, node)

    async def _override():
        yield session

    test_app.dependency_overrides[get_session] = _override
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments/21/manifest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["container_runtime"] == "podman"
    assert body["model"] == "org/model"


# ---------------------------------------------------------------------------
# Served-name uniqueness
# ---------------------------------------------------------------------------


def _dep(id, model_name, served=None, lora=None):
    return SimpleNamespace(
        id=id,
        model_name=model_name,
        engine_args={"served_model_name": served} if served else {},
        lora_modules=[{"name": n, "path": "p"} for n in (lora or [])],
        status="running",
    )


class _AliasSession:
    """Returns a fixed set of active deployments from execute().scalars().all()."""

    def __init__(self, deployments):
        self._deployments = deployments

    async def execute(self, stmt):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self._deployments))


class TestModelNameHelpers:
    def test_effective_served_name_prefers_override(self):
        assert model_names.effective_served_name(_dep(1, "org/m", served="alias")) == "alias"

    def test_effective_served_name_defaults_to_model(self):
        assert model_names.effective_served_name(_dep(1, "org/m")) == "org/m"

    def test_primary_aliases_includes_served_and_lora(self):
        d = _dep(1, "org/m", served="alias", lora=["a1", "a2"])
        assert model_names.primary_aliases(d) == {"alias", "a1", "a2"}

    def test_suggest_served_name(self):
        assert deployments_api._suggest_served_name("m", set()) == "m"
        assert deployments_api._suggest_served_name("m", {"m"}) == "m-2"
        assert deployments_api._suggest_served_name("m", {"m", "m-2"}) == "m-3"


@pytest.mark.anyio
async def test_served_name_conflict_raises_409():
    session = _AliasSession([_dep(1, "org/m")])  # effective served name "org/m"
    with pytest.raises(deployments_api.HTTPException) as exc:
        await deployments_api._check_served_name_conflict(
            session, model_name="org/m", engine_args=None, lora_modules=None
        )
    assert exc.value.status_code == 409
    assert "org/m" in exc.value.detail
    assert "different served model name" in exc.value.detail


@pytest.mark.anyio
async def test_same_model_distinct_served_name_allowed():
    session = _AliasSession([_dep(1, "org/m")])  # already serving "org/m"
    # Same base model, explicit distinct served name -> no conflict.
    await deployments_api._check_served_name_conflict(
        session,
        model_name="org/m",
        engine_args={"served_model_name": "org/m-2"},
        lora_modules=None,
    )


@pytest.mark.anyio
async def test_lora_name_conflict_raises_409():
    session = _AliasSession([_dep(1, "base", served="base", lora=["adapter"])])
    with pytest.raises(deployments_api.HTTPException) as exc:
        await deployments_api._check_served_name_conflict(
            session,
            model_name="other",
            engine_args={"served_model_name": "adapter"},
            lora_modules=None,
        )
    assert exc.value.status_code == 409
    assert "adapter" in exc.value.detail


@pytest.mark.anyio
async def test_check_served_name_endpoint_reports_conflict_and_suggestion():
    session = _AliasSession([_dep(1, "org/m", served="taken")])
    taken = await deployments_api.check_served_name("taken", None, session)
    assert taken["available"] is False
    assert taken["conflict_id"] == 1
    assert taken["suggestion"] == "taken-2"
    free = await deployments_api.check_served_name("fresh", None, session)
    assert free == {"available": True}


# ---------------------------------------------------------------------------
# POST /deployments/plan (warm-offload deploy preview)
# ---------------------------------------------------------------------------


class _FakePlanSession:
    """get() returns the node; execute() returns the seeded deployment rows."""

    def __init__(self, node, deployments):
        self.node = node
        self._deployments = deployments

    async def get(self, model, obj_id):
        return self.node if self.node and self.node.id == obj_id else None

    async def execute(self, stmt):
        rows = list(self._deployments)
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))


def _plan_app(session):
    test_app = FastAPI()
    test_app.include_router(deployments_api.router, prefix="/deployments")

    async def _override():
        yield session

    test_app.dependency_overrides[get_session] = _override
    return test_app


async def _post_plan(session):
    transport = ASGITransport(app=_plan_app(session))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/deployments/plan",
            json={
                "node_id": 1,
                "model_name": "new",
                "port": 9001,
                "gpu_memory_fraction": 0.5,
                "gpu_ids": [0],
            },
        )


@pytest.mark.anyio
async def test_plan_endpoint_maps_keys_to_models():
    node = SimpleNamespace(
        id=1, ip_address="10.0.0.1", port=9000, warm_offload_enabled=True
    )
    dep = SimpleNamespace(
        id=10, model_name="old", port=8000, node_id=1, status="running"
    )
    agent_plan = {
        "fits": True,
        "warm_enabled": True,
        "plan": [{"key": "old:8000", "model_name": "old", "tier": "ram"}],
        "reason": None,
    }
    with mock.patch.object(
        deployments_api, "plan_deployment", new=mock.AsyncMock(return_value=agent_plan)
    ):
        resp = await _post_plan(_FakePlanSession(node, [dep]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["fits"] is True
    assert body["warm_enabled"] is True
    assert body["would_offload"] == [
        {"deployment_id": 10, "model_name": "old", "tier": "ram"}
    ]
    assert body["blocked_reason"] is None


@pytest.mark.anyio
async def test_plan_endpoint_blocked_sets_reason():
    node = SimpleNamespace(
        id=1, ip_address="10.0.0.1", port=9000, warm_offload_enabled=True
    )
    agent_plan = {
        "fits": False,
        "warm_enabled": True,
        "plan": [],
        "reason": "GPU 0 full and no warm model is eligible.",
    }
    with mock.patch.object(
        deployments_api, "plan_deployment", new=mock.AsyncMock(return_value=agent_plan)
    ):
        resp = await _post_plan(_FakePlanSession(node, []))
    assert resp.status_code == 200
    body = resp.json()
    assert body["fits"] is False
    assert body["blocked_reason"]
    assert body["would_offload"] == []


@pytest.mark.anyio
async def test_plan_endpoint_warm_disabled_short_circuits():
    node = SimpleNamespace(
        id=1, ip_address="10.0.0.1", port=9000, warm_offload_enabled=False
    )
    # The agent must NOT be consulted when warm-offload is off for the node.
    with mock.patch.object(
        deployments_api, "plan_deployment", new=mock.AsyncMock()
    ) as agent:
        resp = await _post_plan(_FakePlanSession(node, []))
    assert resp.status_code == 200
    body = resp.json()
    assert body["warm_enabled"] is False
    assert body["fits"] is True
    agent.assert_not_awaited()
