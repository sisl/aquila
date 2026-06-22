"""Tests for the sync service constants (parsed from source to avoid DB deps)."""

import ast
from pathlib import Path

import pytest

_SYNC_PATH = (
    Path(__file__).resolve().parent.parent
    / "athanor"
    / "assets"
    / "host"
    / "backend"
    / "app"
    / "services"
    / "sync.py"
)


def _parse_constants() -> dict[str, object]:
    """Extract top-level constant assignments from sync.py without importing it."""
    source = _SYNC_PATH.read_text()
    tree = ast.parse(source)
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    constants[target.id] = node.value.value
    return constants


class TestSyncConstants:
    def test_node_failure_threshold(self):
        constants = _parse_constants()
        assert constants["_NODE_FAILURE_THRESHOLD"] >= 1

    def test_deployment_failure_threshold(self):
        constants = _parse_constants()
        assert constants["_DEPLOYMENT_FAILURE_THRESHOLD"] >= 1

    def test_sync_file_exists(self):
        assert _SYNC_PATH.exists()


# ---------------------------------------------------------------------------
# _adopted_deployment (re-adoption of client containers after DB loss)
# ---------------------------------------------------------------------------

import sys
from datetime import datetime, timezone

_BACKEND_DIR = _SYNC_PATH.parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import app.services.sync as sync  # noqa: E402

_NOW = datetime(2026, 6, 11, 10, 0, 0, tzinfo=timezone.utc)


def _client_dep(**extra) -> dict:
    return {
        "key": "org/model:8001",
        "model_name": "org/model",
        "port": 8001,
        "status": "running",
        "gpu_memory_fraction": 0.5,
        "gpu_ids": [0, 1],
        "tensor_parallel_size": 2,
        "vllm_version": "0.9.1",
        **extra,
    }


_MANIFEST = {
    "version": 1,
    "owner": "alice",
    "duration_seconds": 3600,
    "expires_at": "2026-06-11T12:00:00+00:00",
    "extra_args": ["--seed", "7"],
    "env_vars": [{"key": "HF_TOKEN", "value": "tok"}],
    "engine_args": {"max_model_len": 4096},
    "lora_modules": [{"name": "ad", "path": "p"}],
    "extra_packages": ["transformers"],
    "gpu_memory_fraction": 0.5,
    "gpu_ids": [0, 1],
    "tensor_parallel_size": 2,
    "max_failed_restarts": 5,
    "container_runtime": "podman",
}


class TestAdoptedDeployment:
    def test_fills_metadata_from_manifest(self):
        dep = sync._adopted_deployment(3, _client_dep(launch_manifest=dict(_MANIFEST)), _NOW)
        assert dep.node_id == 3
        assert dep.model_name == "org/model"
        assert dep.port == 8001
        assert dep.status == "running"
        assert dep.owner == "alice"
        assert dep.duration_seconds == 3600
        assert dep.expires_at == datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)
        assert dep.extra_args == ["--seed", "7"]
        assert dep.env_vars == [{"key": "HF_TOKEN", "value": "tok"}]
        assert dep.engine_args == {"max_model_len": 4096}
        assert dep.lora_modules == [{"name": "ad", "path": "p"}]
        assert dep.extra_packages == ["transformers"]
        assert dep.max_failed_restarts == 5
        assert dep.container_runtime == "podman"

    def test_without_manifest_keeps_legacy_minimal_row(self):
        dep = sync._adopted_deployment(3, _client_dep(), _NOW)
        assert dep.model_name == "org/model"
        assert dep.owner is None
        assert dep.duration_seconds is None
        assert dep.expires_at is None
        assert not dep.extra_args

    def test_bad_expires_at_restores_rest(self):
        manifest = dict(_MANIFEST, expires_at="not-a-timestamp")
        dep = sync._adopted_deployment(3, _client_dep(launch_manifest=manifest), _NOW)
        assert dep.expires_at is None
        assert dep.owner == "alice"
        assert dep.duration_seconds == 3600

    def test_non_dict_manifest_ignored(self):
        dep = sync._adopted_deployment(3, _client_dep(launch_manifest="garbage"), _NOW)
        assert dep.owner is None
        assert dep.expires_at is None
