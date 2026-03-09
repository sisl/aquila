"""Tests for the sync service constants (parsed from source to avoid DB deps)."""

import ast
from pathlib import Path

import pytest

_SYNC_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm_cluster_manager"
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
