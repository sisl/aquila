"""Tests for webhook notifications and the expiring-soon warning logic."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_BACKEND_DIR = (
    Path(__file__).resolve().parent.parent
    / "athanor"
    / "assets"
    / "host"
    / "backend"
)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import app.services.notify as notify_mod
import app.services.sync as sync
from app.core.config import settings


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return self._items


class _FakeSession:
    def __init__(self, deployments):
        self._deployments = deployments

    async def execute(self, *_args, **_kwargs):
        return _FakeResult(self._deployments)


def _dep(dep_id, expires_at, status="running"):
    return SimpleNamespace(
        id=dep_id,
        status=status,
        expires_at=expires_at,
        model_name="m",
        port=8000,
    )


@pytest.fixture(autouse=True)
def _clean_warned():
    notify_mod._warned_expiring.clear()
    yield
    notify_mod._warned_expiring.clear()


class TestPayloadFormatting:
    def test_message_includes_error_reason(self):
        msg = notify_mod._format_message(
            "deployment_error", "m is error", {"error": "GPU ran out of memory."}
        )
        assert "m is error" in msg
        assert "GPU ran out of memory." in msg

    def test_slack_payload(self):
        with mock.patch.object(
            settings, "webhook_url", "https://hooks.slack.com/services/T/B/x"
        ):
            payload = notify_mod._payload("e", "hello", {})
        assert payload == {"text": "[athanor] hello"}

    def test_generic_payload(self):
        with mock.patch.object(settings, "webhook_url", "https://example.com/hook"):
            payload = notify_mod._payload("e", "hello", {"deployment_id": 3})
        assert payload["event"] == "e"
        assert payload["deployment_id"] == 3

    def test_notify_noop_without_url(self):
        with mock.patch.object(settings, "webhook_url", None):
            notify_mod.notify("e", "hello")  # must not raise


class TestTransitionEvents:
    def test_running_transition(self):
        node = SimpleNamespace(hostname="gpu-01")
        dep = SimpleNamespace(
            id=1, status="running", model_name="m", port=8000, last_error=None
        )
        event = sync._transition_event(dep, "loading", node)
        assert event[0] == "deployment_running"
        assert "gpu-01:8000" in event[1]

    def test_error_transition_includes_reason(self):
        node = SimpleNamespace(hostname="gpu-01")
        dep = SimpleNamespace(
            id=1, status="error", model_name="m", port=8000, last_error="boom"
        )
        event = sync._transition_event(dep, "running", node)
        assert event[0] == "deployment_error"
        assert event[2]["error"] == "boom"

    def test_error_to_unreachable_not_renotified(self):
        node = SimpleNamespace(hostname="gpu-01")
        dep = SimpleNamespace(
            id=1, status="unreachable", model_name="m", port=8000, last_error="x"
        )
        assert sync._transition_event(dep, "error", node) is None

    def test_no_event_for_stop(self):
        node = SimpleNamespace(hostname="gpu-01")
        dep = SimpleNamespace(
            id=1, status="stopped", model_name="m", port=8000, last_error=None
        )
        assert sync._transition_event(dep, "running", node) is None


@pytest.mark.anyio
class TestWarnExpiring:
    async def test_warns_once_inside_window(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dep = _dep(1, now + timedelta(minutes=10))
        session = _FakeSession([dep])
        with mock.patch.object(notify_mod, "notify") as fake_notify:
            sync_notify = mock.patch.object(sync, "notify", fake_notify)
            with sync_notify:
                warned = await sync.warn_expiring_deployments(session, now=now)
                warned_again = await sync.warn_expiring_deployments(session, now=now)
        assert warned == [1]
        assert warned_again == []

    async def test_extension_rearms_warning(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dep = _dep(1, now + timedelta(minutes=10))
        session = _FakeSession([dep])
        with mock.patch.object(sync, "notify"):
            assert await sync.warn_expiring_deployments(session, now=now) == [1]
            # Extend: expiry moves out past the window — warning key changes.
            dep.expires_at = now + timedelta(hours=5)
            assert await sync.warn_expiring_deployments(session, now=now) == []
            # Time passes; back inside the window with the NEW expiry → re-warn.
            later = now + timedelta(hours=4, minutes=40)
            assert await sync.warn_expiring_deployments(session, now=later) == [1]

    async def test_outside_window_no_warning(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dep = _dep(1, now + timedelta(hours=5))
        session = _FakeSession([dep])
        with mock.patch.object(sync, "notify"):
            assert await sync.warn_expiring_deployments(session, now=now) == []

    async def test_infinite_deployment_ignored(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        session = _FakeSession([_dep(1, None)])
        with mock.patch.object(sync, "notify"):
            assert await sync.warn_expiring_deployments(session, now=now) == []
