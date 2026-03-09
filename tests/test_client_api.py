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

from app.services.client_api import _satellite_url


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
