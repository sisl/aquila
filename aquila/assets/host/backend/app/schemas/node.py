from datetime import datetime
from pydantic import BaseModel, Field


class NodeBase(BaseModel):
    hostname: str = Field(description="Machine hostname.")
    ip_address: str = Field(description="Reachable IP address of the node agent.")
    port: int | None = Field(None, description="Agent HTTP port on the node.")
    status: str = Field("unknown", description="Health status (online, degraded, critical, unknown).")
    maintenance_gpus: list[int] = Field(default_factory=list, description="GPU indices currently under maintenance.")
    gpu_usage: list[dict[str, object]] | None = Field(None, description="Per-GPU utilization and memory stats from the latest scrape.")
    # {total_gb, free_gb, hf_cache_gb} as reported by the client agent.
    disk_usage: dict[str, object] | None = Field(None, description="Disk usage breakdown (total_gb, free_gb, hf_cache_gb).")
    default_pip_packages: list[str] | None = Field(None, description="Pip packages auto-installed in every container on this node.")
    installed_packages: list[str] | None = Field(None, description="System packages detected on the node.")
    # Detected container runtimes ("docker"/"podman"), synced from metrics.
    available_runtimes: list[str] | None = Field(None, description="Container runtimes detected on the node (docker/podman).")
    # Per-node runtime override; null = auto.
    container_runtime: str | None = Field(None, description="Per-node container runtime override; null = auto.")
    # Warm cache: opt-in auto-offload toggle + CPU-RAM cache budget (MB).
    warm_offload_enabled: bool = Field(False, description="Whether warm-cache auto-offload is enabled on this node.")
    ram_cache_limit_mb: int | None = Field(None, description="CPU RAM budget in MB for warm-cached models; null = unlimited.")


class NodeCreate(NodeBase):
    pass


class NodeRead(NodeBase):
    id: int
    maintenance: bool = Field(False, description="True when all GPUs are under maintenance.")
    partial_maintenance: bool = Field(False, description="True when some (but not all) GPUs are under maintenance.")
    # Derived (not persisted): rogue/untracked vLLM containers seen on the node.
    rogue_container_count: int | None = Field(None, description="Rogue/untracked vLLM containers detected on the node.")
    # Derived (not persisted): orphaned vLLM GPU processes (no live container).
    rogue_process_count: int | None = Field(None, description="Orphaned vLLM GPU processes with no live container.")
    # Derived (not persisted): orphaned warm-cache artifacts (RAM + disk).
    rogue_artifact_count: int | None = Field(None, description="Orphaned warm-cache artifacts (RAM + disk).")
    # Derived (not persisted): CPU RAM (MB) held by RAM-paused models.
    ram_cache_used_mb: float | None = Field(None, description="CPU RAM in MB currently held by RAM-paused models.")
    last_heartbeat_at: datetime | None = Field(None, description="Timestamp of the last successful heartbeat from this node.")
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class NodeWarmCacheRequest(BaseModel):
    enabled: bool
    # MB; null = unlimited.
    ram_cache_limit_mb: int | None = None


class NodeMaintenanceRequest(BaseModel):
    gpu_ids: list[int] = []
    enabled: bool
    drain: bool = False


class NodeSetRuntimeRequest(BaseModel):
    # "docker" / "podman", or null to return to auto (preferred runtime).
    runtime: str | None = None


class DiscoveredNode(BaseModel):
    node: str | None = None
    address: str | None = None
    port: int | None = None
    service_id: str | None = None
