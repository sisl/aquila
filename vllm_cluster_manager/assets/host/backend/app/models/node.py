from sqlalchemy import Boolean, DateTime, Integer, JSON, String
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
    # Cordoned for maintenance: no new deployments, health flaps ignored.
    maintenance: Mapped[bool] = mapped_column(Boolean, default=False)
    gpu_usage: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    # {total_gb, free_gb, hf_cache_gb} as reported by the client agent.
    disk_usage: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    default_pip_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    installed_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Detected container runtimes ("docker"/"podman"), synced from metrics.
    available_runtimes: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Per-node runtime override; null = auto (preferred, else whichever exists).
    container_runtime: Mapped[str | None] = mapped_column(String(16), nullable=True)
    default_vllm_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_heartbeat_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
