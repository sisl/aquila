from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base

# Statuses that occupy a node:port slot. A partial unique index over these
# makes port reservation atomic: two concurrent starts can both pass the
# friendly pre-check SELECT, but only one insert wins. Paused deployments keep
# their slot: the warm-mode agent proxy holds the public port across pauses.
ACTIVE_STATUSES = (
    "starting", "loading", "running", "stopping", "paused_ram", "offloading"
)

_ACTIVE_PREDICATE = text(
    "status IN ('starting', 'loading', 'running', 'stopping', "
    "'paused_ram', 'offloading')"
)


class Deployment(Base):
    """A vLLM model deployment running on a node.

    Tracks the full lifecycle from creation through running to stopped/error,
    including cumulative usage counters scraped from the vLLM container.
    """

    __tablename__ = "deployments"
    __table_args__ = (
        Index(
            "uq_active_node_port",
            "node_id",
            "port",
            unique=True,
            postgresql_where=_ACTIVE_PREDICATE,
            sqlite_where=_ACTIVE_PREDICATE,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id"))
    model_name: Mapped[str] = mapped_column(String(255), index=True)
    port: Mapped[int] = mapped_column(Integer)
    gpu_memory_fraction: Mapped[float]
    gpu_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    tensor_parallel_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_args: Mapped[list[str]] = mapped_column(JSON, default=list)
    env_vars: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    pip_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    vllm_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Which container runtime (docker/podman) runs this deployment.
    container_runtime: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Exact image identity (repo@sha256 or content id) reported by the client;
    # provenance for reproducibility manifests.
    image_digest: Mapped[str | None] = mapped_column(String(512), nullable=True)
    extra_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Structured vLLM engine flags (max_model_len, dtype, quantization, ...);
    # translated to CLI args by the client. extra_args stays the escape hatch.
    engine_args: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    # LoRA adapters served alongside the base model: [{name, path}].
    lora_modules: Mapped[list[dict[str, str]] | None] = mapped_column(JSON, nullable=True)
    # Per-deployment crash-loop threshold; NULL uses the client's default.
    max_failed_restarts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Warm cache: protect this deployment from automatic eviction.
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    status: Mapped[str] = mapped_column(String(32), default="stopped")
    # Last failure reason (client error detail or watchdog timeout); cleared
    # whenever the deployment returns to a healthy status.
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Load phase reported by the client while status is "loading"
    # (downloading / loading_weights / compiling).
    detail: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # When status last changed; drives the stuck-start watchdog.
    status_changed_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Cumulative token/request counters accumulated from the vLLM instance's
    # own Prometheus metrics (covers gateway AND direct traffic).
    total_prompt_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    total_completion_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    total_requests: Mapped[int] = mapped_column(BigInteger, default=0)
    # Last-known per-request token speeds (read/generation averages from the
    # client scrape), persisted so stopped deployments keep their stats.
    prompt_tps: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_tps: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Who launched the deployment (free text id/name; required at the API layer).
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Requested serve duration in seconds; NULL means serve indefinitely.
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Absolute expiry time, set when serving starts; NULL while loading or infinite.
    expires_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
