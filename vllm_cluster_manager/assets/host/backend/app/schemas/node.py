from datetime import datetime
from pydantic import BaseModel


class NodeBase(BaseModel):
    hostname: str
    ip_address: str
    port: int | None = None
    status: str = "unknown"
    maintenance: bool = False
    gpu_usage: list[dict[str, object]] | None = None
    # {total_gb, free_gb, hf_cache_gb} as reported by the client agent.
    disk_usage: dict[str, object] | None = None
    default_pip_packages: list[str] | None = None
    installed_packages: list[str] | None = None
    # Detected container runtimes ("docker"/"podman"), synced from metrics.
    available_runtimes: list[str] | None = None
    # Per-node runtime override; null = auto.
    container_runtime: str | None = None


class NodeCreate(NodeBase):
    pass


class NodeRead(NodeBase):
    id: int
    # Derived (not persisted): rogue/untracked vLLM containers seen on the node.
    rogue_container_count: int | None = None
    last_heartbeat_at: datetime | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class NodeMaintenanceRequest(BaseModel):
    enabled: bool
    # Also stop all active deployments on the node when cordoning.
    drain: bool = False


class NodeSetRuntimeRequest(BaseModel):
    # "docker" / "podman", or null to return to auto (preferred runtime).
    runtime: str | None = None


class DiscoveredNode(BaseModel):
    node: str | None = None
    address: str | None = None
    port: int | None = None
    service_id: str | None = None
