from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class DeploymentBase(BaseModel):
    node_id: int = Field(description="ID of the node to deploy on.")
    model_name: str = Field(description="HuggingFace model identifier (e.g. meta-llama/Meta-Llama-3-8B).")
    port: int = Field(description="Port the vLLM server listens on.")
    gpu_memory_fraction: float = Field(description="Fraction of each GPU's memory to allocate (0.0-1.0).")
    gpu_ids: list[int] | None = Field(None, description="Specific GPU indices to use; omit for automatic selection.")
    tensor_parallel_size: int | None = Field(None, description="Number of GPUs for tensor parallelism.")
    extra_args: list[str] | None = Field(None, description="Additional CLI arguments passed to the vLLM server.")
    env_vars: list[dict[str, str]] | None = Field(None, description="Environment variables injected into the container.")
    pip_packages: list[str] | None = Field(None, description="Python packages to pip-install before starting vLLM.")
    vllm_version: str | None = Field(None, description="vLLM Docker image tag to use.")
    # Which container runtime (docker/podman) runs this deployment; resolved
    # by the host at launch (per-node override > preferred > available).
    container_runtime: str | None = Field(None, description="Container runtime for this deployment (docker/podman); auto-resolved if omitted.")
    extra_packages: list[str] | None = Field(None, description="Additional OS packages to install in the container.")
    # Structured vLLM engine flags (max_model_len, dtype, quantization, ...).
    engine_args: dict[str, object] | None = Field(None, description="Structured vLLM engine flags (max_model_len, dtype, quantization, etc.).")
    # LoRA adapters served alongside the base model: [{name, path}].
    lora_modules: list[dict[str, str]] | None = Field(None, description="LoRA adapters served alongside the base model; each entry has name and path.")
    # Per-deployment crash-loop threshold; None uses the client default.
    max_failed_restarts: int | None = Field(None, description="Crash-loop restart threshold; omit to use the cluster default.")
    owner: str | None = Field(None, description="User or team who owns this deployment.")
    # Requested serve duration in seconds; None means serve indefinitely.
    duration_seconds: int | None = Field(None, description="Serve duration in seconds; omit for indefinite serving.")
    # Warm cache: protect this deployment from automatic eviction.
    pinned: bool = Field(False, description="Pin this deployment to prevent automatic warm-cache eviction.")
    status: str = Field("stopped", description="Current lifecycle status (stopped, loading, running, error, paused).")


class DeploymentCreate(DeploymentBase):
    pass


class DeploymentRead(DeploymentBase):
    id: int
    # Set when serving starts; None while loading or for an infinite duration.
    expires_at: datetime | None = Field(None, description="When this deployment will auto-stop; None for indefinite.")
    created_at: datetime | None = None
    # Exact image identity reported by the client (provenance).
    image_digest: str | None = Field(None, description="Docker image digest reported by the client.")
    # Last failure reason (client error or watchdog timeout).
    last_error: str | None = Field(None, description="Last failure reason (client error or watchdog timeout).")
    # Load phase while status is "loading" (downloading/loading_weights/compiling).
    detail: str | None = Field(None, description="Load phase while status is loading (downloading/loading_weights/compiling).")
    status_changed_at: datetime | None = None
    # Cumulative usage from the vLLM instance's Prometheus counters.
    total_prompt_tokens: int = Field(0, description="Cumulative prompt tokens processed.")
    total_completion_tokens: int = Field(0, description="Cumulative completion tokens generated.")
    total_requests: int = Field(0, description="Cumulative requests served.")
    # Live metrics from the latest scrape (not persisted; attached to running
    # deployments by the list endpoint). Speeds are split read (prefill) vs
    # generation (decode): *_tps are per-request, idle-free (token deltas over
    # processing-time deltas); *_throughput are engine-wide over the window.
    prompt_tps: float | None = None
    generation_tps: float | None = None
    prompt_throughput: float | None = None
    generation_throughput: float | None = None
    requests_running: int | None = None
    requests_waiting: int | None = None
    # Image-pull progress while starting (not persisted; attached by the list
    # endpoint from the client's transfer report).
    pull_percent: float | None = None
    pull_downloaded_mb: int | None = None
    pull_total_mb: int | None = None

    model_config = {"from_attributes": True}


class DeploymentStart(DeploymentBase):
    owner: str = Field(description="User or team launching this deployment (required).")
    # Bypass the client's GPU memory pre-check (not persisted).
    skip_resource_check: bool = Field(False, description="Bypass the client's GPU memory pre-check.")


class DeploymentRestart(BaseModel):
    """Parameters for restarting a stopped, expired, or errored deployment."""

    owner: str = Field(description="User or team taking ownership of the restarted deployment.")
    duration_seconds: int | None = Field(None, description="Serve duration in seconds for the new run; omit for indefinite.")


class DeploymentPin(BaseModel):
    """Toggle the pin flag on a deployment."""

    pinned: bool = Field(description="True to pin (protect from warm-cache eviction), False to unpin.")


class DeploymentPause(BaseModel):
    """Pause a running deployment by offloading its model weights from GPU."""

    # "ram" | "disk"; omit for auto (RAM if it fits the budget, else disk).
    tier: str | None = Field(None, description="Offload target: 'ram' or 'disk'; omit for automatic selection.")


class DeploymentPlanRequest(BaseModel):
    """Dry-run a deploy to preview which warm models would be offloaded."""

    node_id: int
    model_name: str
    port: int
    gpu_memory_fraction: float
    gpu_ids: list[int] | None = None


class OffloadItem(BaseModel):
    # deployment_id is None if the agent reports a model the host has no row for.
    deployment_id: int | None = None
    model_name: str
    tier: str  # "ram" | "disk"


class DeploymentPlanRead(BaseModel):
    # Whether the deploy fits (possibly after the listed offloads).
    fits: bool
    # False when the node has warm-offload turned off (no auto-offload at all).
    warm_enabled: bool
    would_offload: list[OffloadItem] = []
    # Set when fits is False: why no plan can make room.
    blocked_reason: str | None = None


class DeploymentFromManifest(BaseModel):
    """Redeploy from an exported manifest; runtime placement is chosen anew."""

    manifest: dict[str, object]
    node_id: int
    port: int
    owner: str
    duration_seconds: int | None = None
    # Env var VALUES never travel in manifests; re-supply them here.
    env_vars: list[dict[str, str]] | None = None
    skip_resource_check: bool = False


class DeploymentExtend(BaseModel):
    # Up to two weeks at a time; omit hours and set infinite=True to drop
    # the expiry entirely (serve until stopped).
    hours: float | None = Field(default=None, gt=0, le=24 * 14)
    infinite: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> "DeploymentExtend":
        if self.infinite == (self.hours is not None):
            raise ValueError("Provide either 'hours' or 'infinite', not both.")
        return self
