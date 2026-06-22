from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.services import runtime_settings
from app.ws.manager import manager

router = APIRouter()


class RuntimeSettingsUpdate(BaseModel):
    """Partial update; only provided fields are applied."""

    model_config = ConfigDict(extra="forbid")

    gateway_enabled: bool | None = Field(None, description="Enable or disable the OpenAI-compatible /v1 gateway.")
    gateway_timeout_seconds: int | None = Field(None, ge=10, le=86400, description="Non-streaming request timeout in seconds.")
    start_timeout_seconds: int | None = Field(None, ge=60, le=86400, description="Mark deployment errored if stuck starting this long.")
    preferred_container_runtime: Literal["docker", "podman"] | None = Field(None, description="Preferred container runtime when a node has both Docker and Podman.")
    default_port: int | None = Field(None, ge=1024, le=65535, description="Default port pre-filled in the deploy form.")
    default_gpu_fraction: float | None = Field(None, ge=0.05, le=1.0, description="Default GPU memory fraction pre-filled in the deploy form.")
    default_duration_choice: str | None = Field(None, description="Default serve-duration choice for the deploy form.")
    default_vllm_version: str | None = Field(None, description="Default vLLM version pre-filled in the deploy form.")
    # Nullable on purpose: explicit null clears the override back to "use the
    # client's default".
    default_max_failed_restarts: int | None = Field(None, ge=1, le=20, description="Default crash-loop threshold for new deployments.")
    webhook_url: str | None = Field(None, description="Webhook URL for deployment lifecycle notifications.")
    expiry_warning_minutes: int | None = Field(None, ge=1, le=1440, description="Warn this many minutes before a deployment expires.")
    node_metrics_retention_hours: int | None = Field(None, ge=1, le=8760, description="How many hours of node metric history to retain.")
    temp_api_key_ttl_seconds: int | None = Field(None, ge=0, le=3600, description="Temporary API key lifespan in seconds for endpoint snippets.")
    default_warm_offload_enabled: bool | None = Field(None, description="Enable warm cache by default on newly discovered nodes.")
    busy_guard_seconds: int | None = Field(None, ge=0, le=300, description="Seconds after last request before warm cache can auto-evict a model.")
    nodes_sync_interval_seconds: int | None = Field(None, ge=1, le=300, description="Node sync loop interval in seconds.")
    deployments_sync_interval_seconds: int | None = Field(None, ge=1, le=300, description="Deployment sync loop interval in seconds.")
    expiry_check_interval_seconds: int | None = Field(None, ge=5, le=600, description="Expiry check loop interval in seconds.")
    node_failure_threshold: int | None = Field(None, ge=1, le=20, description="Consecutive failures before a node turns critical.")
    deployment_failure_threshold: int | None = Field(None, ge=1, le=20, description="Unreachable polls before a deployment degrades.")

    @field_validator("default_duration_choice")
    @classmethod
    def _duration_choice(cls, value: str | None) -> str | None:
        if value is not None and value not in runtime_settings.DURATION_CHOICES:
            raise ValueError(
                f"Must be one of {', '.join(runtime_settings.DURATION_CHOICES)}."
            )
        return value

    @field_validator("webhook_url")
    @classmethod
    def _webhook_url(cls, value: str | None) -> str | None:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("Must be an http(s) URL (or empty to disable).")
        return value


# Fields where an explicit null is a meaningful value rather than "unset".
_NULLABLE = {"default_max_failed_restarts"}


@router.get("")
async def read_settings() -> dict[str, object]:
    """Return all current runtime settings with their effective values."""
    return runtime_settings.effective()


@router.put("")
async def update_settings(
    payload: RuntimeSettingsUpdate, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    """Apply a partial update to runtime settings; only provided fields are changed."""
    provided = payload.model_dump(exclude_unset=True)
    values = {
        key: value
        for key, value in provided.items()
        if value is not None or key in _NULLABLE
    }
    if not values:
        raise HTTPException(status_code=400, detail="No settings provided.")
    await runtime_settings.apply(session, values)
    await manager.broadcast({"type": "settings_changed"})
    return runtime_settings.effective()
