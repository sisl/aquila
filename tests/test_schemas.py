"""Tests for Pydantic schemas (host backend)."""

import sys
from pathlib import Path

import pytest

# Add the host backend to sys.path so schemas can be imported directly.
_HOST_BACKEND = Path(__file__).resolve().parent.parent / "vllm_cluster_manager" / "assets" / "host" / "backend"
if str(_HOST_BACKEND) not in sys.path:
    sys.path.insert(0, str(_HOST_BACKEND))

from app.schemas.deployment import DeploymentBase, DeploymentCreate, DeploymentRead, DeploymentStart
from app.schemas.node import NodeBase, NodeRead, DiscoveredNode
from app.schemas.deployment_config import DeploymentConfigBase, DeploymentConfigRead


# ---------------------------------------------------------------------------
# Deployment schemas
# ---------------------------------------------------------------------------


class TestDeploymentSchemas:
    def test_deployment_base_defaults(self):
        d = DeploymentBase(node_id=1, model_name="meta/llama", port=8080, gpu_memory_fraction=0.9)
        assert d.status == "stopped"
        assert d.gpu_ids is None
        assert d.tensor_parallel_size is None
        assert d.extra_args is None
        assert d.env_vars is None
        assert d.vllm_version is None
        assert d.extra_packages is None

    def test_deployment_base_with_all_fields(self):
        d = DeploymentBase(
            node_id=1,
            model_name="meta/llama",
            port=8080,
            gpu_memory_fraction=0.9,
            gpu_ids=[0, 1],
            tensor_parallel_size=2,
            extra_args=["--max-model-len", "4096"],
            env_vars=[{"key": "HF_TOKEN", "value": "abc"}],
            vllm_version="0.8.5",
            extra_packages=["transformers"],
            status="running",
        )
        assert d.gpu_ids == [0, 1]
        assert d.tensor_parallel_size == 2
        assert d.vllm_version == "0.8.5"
        assert d.extra_packages == ["transformers"]

    def test_deployment_create_inherits(self):
        d = DeploymentCreate(node_id=1, model_name="m", port=80, gpu_memory_fraction=0.5)
        assert d.node_id == 1

    def test_deployment_read_from_attributes(self):
        d = DeploymentRead(
            id=42,
            node_id=1,
            model_name="m",
            port=80,
            gpu_memory_fraction=0.5,
            created_at=None,
        )
        assert d.id == 42

    def test_deployment_start_inherits(self):
        d = DeploymentStart(node_id=1, model_name="m", port=80, gpu_memory_fraction=0.5)
        assert d.status == "stopped"


# ---------------------------------------------------------------------------
# Node schemas
# ---------------------------------------------------------------------------


class TestNodeSchemas:
    def test_node_base_defaults(self):
        n = NodeBase(hostname="gpu-01", ip_address="10.0.0.1")
        assert n.status == "unknown"
        assert n.port is None
        assert n.gpu_usage is None

    def test_node_read_model_config(self):
        n = NodeRead(
            id=1,
            hostname="gpu-01",
            ip_address="10.0.0.1",
            last_heartbeat_at=None,
            created_at=None,
        )
        assert n.id == 1

    def test_discovered_node_defaults(self):
        dn = DiscoveredNode()
        assert dn.node is None
        assert dn.address is None
        assert dn.port is None
        assert dn.service_id is None


# ---------------------------------------------------------------------------
# DeploymentConfig schemas
# ---------------------------------------------------------------------------


class TestDeploymentConfigSchemas:
    def test_config_base(self):
        c = DeploymentConfigBase(name="my-config", payload={"model": "llama"})
        assert c.name == "my-config"
        assert c.payload == {"model": "llama"}

    def test_config_read(self):
        c = DeploymentConfigRead(id=1, name="cfg", payload={}, created_at=None)
        assert c.id == 1
