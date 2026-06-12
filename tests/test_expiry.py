"""Tests for deployment auto-expiry (backend expire_due_deployments)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# Add the host backend app to sys.path.
_BACKEND_DIR = (
    Path(__file__).resolve().parent.parent
    / "vllm_cluster_manager"
    / "assets"
    / "host"
    / "backend"
)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import app.services.deployment_stop as deployment_stop
import app.services.sync as sync


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return self._items


class _FakeSession:
    """Minimal async session stub: execute -> deployments, get -> node."""

    def __init__(self, deployments, node):
        self._deployments = deployments
        self._node = node
        self.committed = False

    async def execute(self, *_args, **_kwargs):
        return _FakeResult(self._deployments)

    async def get(self, _model, _id):
        return self._node

    async def commit(self):
        self.committed = True


def _dep(dep_id, status, expires_at, *, port=8000):
    return SimpleNamespace(
        id=dep_id,
        status=status,
        expires_at=expires_at,
        model_name="m",
        port=port,
        node_id=1,
        duration_seconds=3600,
    )


@pytest.mark.anyio
async def test_expire_due_deployments():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    due = _dep(1, "running", now - timedelta(minutes=1), port=8000)
    not_due = _dep(2, "running", now + timedelta(hours=2), port=8001)
    infinite = _dep(3, "running", None, port=8002)
    loading_due = _dep(4, "loading", now - timedelta(seconds=5), port=8003)
    node = SimpleNamespace(ip_address="10.0.0.1", port=9000)
    session = _FakeSession([due, not_due, infinite, loading_due], node)

    with mock.patch.object(deployment_stop, "stop_model", new=mock.AsyncMock()) as stop:
        expired_ids = await sync.expire_due_deployments(session, now=now)

    # Only the past-expiry live deployments are expired.
    assert set(expired_ids) == {1, 4}
    assert due.status == "expired"
    assert loading_due.status == "expired"
    assert not_due.status == "running"
    assert infinite.status == "running"
    assert session.committed

    # Each expired deployment was stopped like a manual stop (key = model:port).
    assert stop.await_count == 2
    stopped_keys = {call.args[2] for call in stop.call_args_list}
    assert stopped_keys == {"m:8000", "m:8003"}


@pytest.mark.anyio
async def test_expire_naive_timestamp_treated_as_utc():
    # Some DB backends return naive datetimes; they must be treated as UTC.
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    naive_past = datetime(2026, 1, 1, 11, 0, 0)  # no tzinfo, before `now`
    dep = _dep(1, "running", naive_past)
    node = SimpleNamespace(ip_address="10.0.0.1", port=9000)
    session = _FakeSession([dep], node)

    with mock.patch.object(deployment_stop, "stop_model", new=mock.AsyncMock()):
        expired_ids = await sync.expire_due_deployments(session, now=now)

    assert expired_ids == [1]
    assert dep.status == "expired"
