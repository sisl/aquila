"""Tests for the deployment lifecycle helpers (set_status + start watchdog)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add the host backend app to sys.path.
_BACKEND_DIR = (
    Path(__file__).resolve().parent.parent
    / "athanor"
    / "assets"
    / "host"
    / "backend"
)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import app.services.sync as sync
from app.core.config import settings
from app.services.deployment_state import set_status


def _dep(status="stopped", **overrides):
    base = dict(
        id=1,
        status=status,
        last_error=None,
        detail=None,
        status_changed_at=None,
        created_at=None,
        expires_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestSetStatus:
    def test_transition_stamps_changed_at(self):
        dep = _dep("stopped")
        set_status(dep, "starting")
        assert dep.status == "starting"
        assert dep.status_changed_at is not None

    def test_same_status_keeps_changed_at(self):
        dep = _dep("running")
        set_status(dep, "running")
        assert dep.status_changed_at is None

    def test_error_records_reason(self):
        dep = _dep("loading")
        set_status(dep, "error", error="boom")
        assert dep.status == "error"
        assert dep.last_error == "boom"

    def test_healthy_status_clears_stale_error(self):
        dep = _dep("error", last_error="boom")
        set_status(dep, "running")
        assert dep.last_error is None

    def test_unreachable_keeps_last_error(self):
        # Not a healthy status: the old failure reason stays visible.
        dep = _dep("error", last_error="boom")
        set_status(dep, "unreachable")
        assert dep.last_error == "boom"


class TestStartTimedOut:
    def test_not_timed_out_within_window(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dep = _dep("loading", status_changed_at=now - timedelta(seconds=60))
        assert sync._start_timed_out(dep, now) is False

    def test_timed_out_past_window(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dep = _dep(
            "loading",
            status_changed_at=now - timedelta(seconds=settings.start_timeout_seconds + 1),
        )
        assert sync._start_timed_out(dep, now) is True

    def test_falls_back_to_created_at(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dep = _dep(
            "loading",
            status_changed_at=None,
            created_at=now - timedelta(seconds=settings.start_timeout_seconds + 1),
        )
        assert sync._start_timed_out(dep, now) is True

    def test_no_timestamps_never_times_out(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert sync._start_timed_out(_dep("loading"), now) is False

    def test_naive_timestamp_treated_as_utc(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        naive_old = datetime(2026, 1, 1, 10, 0, 0)  # 2h ago, no tzinfo
        dep = _dep("starting", status_changed_at=naive_old)
        assert sync._start_timed_out(dep, now) is True
