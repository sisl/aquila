from sqlalchemy import Boolean, DateTime, Integer, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    hostname: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    ip_address: Mapped[str] = mapped_column(String(64))
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="unknown")
    maintenance_gpus: Mapped[list[int]] = mapped_column(JSON, default=list)
    gpu_usage: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)

    @property
    def maintenance(self) -> bool:
        """True when every GPU on this node is cordoned."""
        if not self.maintenance_gpus:
            return False
        if not self.gpu_usage:
            return bool(self.maintenance_gpus)
        all_indices = {g.get("index", i) for i, g in enumerate(self.gpu_usage)}
        return all_indices <= set(self.maintenance_gpus)

    @property
    def has_partial_maintenance(self) -> bool:
        return bool(self.maintenance_gpus) and not self.maintenance
    # {total_gb, free_gb, hf_cache_gb} as reported by the client agent.
    disk_usage: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    default_pip_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    installed_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Detected container runtimes ("docker"/"podman"), synced from metrics.
    available_runtimes: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Per-node runtime override; null = auto (preferred, else whichever exists).
    container_runtime: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Warm cache: when enabled, new deployments launch pausable and the agent
    # auto-offloads the LRU model (GPU -> RAM -> disk) to fit new/woken models.
    warm_offload_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )
    # Cap (MB) on CPU RAM held by RAM-paused models; null = unlimited.
    ram_cache_limit_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    default_vllm_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_heartbeat_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
