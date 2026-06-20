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

    gateway_enabled: bool | None = None
    gateway_timeout_seconds: int | None = Field(None, ge=10, le=86400)
    start_timeout_seconds: int | None = Field(None, ge=60, le=86400)
    preferred_container_runtime: Literal["docker", "podman"] | None = None
    default_port: int | None = Field(None, ge=1024, le=65535)
    default_gpu_fraction: float | None = Field(None, ge=0.05, le=1.0)
    default_duration_choice: str | None = None
    default_vllm_version: str | None = None
    # Nullable on purpose: explicit null clears the override back to "use the
    # client's default".
    default_max_failed_restarts: int | None = Field(None, ge=1, le=20)
    webhook_url: str | None = None
    expiry_warning_minutes: int | None = Field(None, ge=1, le=1440)
    node_metrics_retention_hours: int | None = Field(None, ge=1, le=8760)
    busy_guard_seconds: int | None = Field(None, ge=0, le=300)
    nodes_sync_interval_seconds: int | None = Field(None, ge=2, le=300)
    deployments_sync_interval_seconds: int | None = Field(None, ge=2, le=300)
    expiry_check_interval_seconds: int | None = Field(None, ge=5, le=600)
    node_failure_threshold: int | None = Field(None, ge=1, le=20)
    deployment_failure_threshold: int | None = Field(None, ge=1, le=20)

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
    return runtime_settings.effective()


@router.put("")
async def update_settings(
    payload: RuntimeSettingsUpdate, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
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
