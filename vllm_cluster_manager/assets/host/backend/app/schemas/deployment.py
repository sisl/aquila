from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class DeploymentBase(BaseModel):
    node_id: int
    model_name: str
    port: int
    gpu_memory_fraction: float
    gpu_ids: list[int] | None = None
    tensor_parallel_size: int | None = None
    extra_args: list[str] | None = None
    env_vars: list[dict[str, str]] | None = None
    pip_packages: list[str] | None = None
    vllm_version: str | None = None
    # Which container runtime (docker/podman) runs this deployment; resolved
    # by the host at launch (per-node override > preferred > available).
    container_runtime: str | None = None
    extra_packages: list[str] | None = None
    # Structured vLLM engine flags (max_model_len, dtype, quantization, ...).
    engine_args: dict[str, object] | None = None
    # LoRA adapters served alongside the base model: [{name, path}].
    lora_modules: list[dict[str, str]] | None = None
    # Per-deployment crash-loop threshold; None uses the client default.
    max_failed_restarts: int | None = None
    owner: str | None = None
    # Requested serve duration in seconds; None means serve indefinitely.
    duration_seconds: int | None = None
    status: str = "stopped"


class DeploymentCreate(DeploymentBase):
    pass


class DeploymentRead(DeploymentBase):
    id: int
    # Set when serving starts; None while loading or for an infinite duration.
    expires_at: datetime | None = None
    created_at: datetime | None = None
    # Exact image identity reported by the client (provenance).
    image_digest: str | None = None
    # Last failure reason (client error or watchdog timeout).
    last_error: str | None = None
    # Load phase while status is "loading" (downloading/loading_weights/compiling).
    detail: str | None = None
    status_changed_at: datetime | None = None
    # Cumulative usage from the vLLM instance's Prometheus counters.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
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
    owner: str  # required when launching a deployment
    # Bypass the client's GPU memory pre-check (not persisted).
    skip_resource_check: bool = False


class DeploymentRestart(BaseModel):
    owner: str
    duration_seconds: int | None = None


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
