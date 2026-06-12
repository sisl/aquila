"""DB-backed global settings with env-seeded defaults.

Resolution order per key: app_settings row (saved from the dashboard) >
environment default (app/core/config.py) > literal default. Consumers read
through the typed accessors on every use, so saved changes apply live without
a backend restart. `load()` runs once at startup; `apply()` upserts overrides
and refreshes the in-process cache.
"""

import logging
from typing import Callable

from sqlalchemy import select

from app.core.config import settings
from app.models.app_setting import AppSetting

logger = logging.getLogger(__name__)

# Serve-duration choices the deploy form offers (seconds as strings + "inf").
DURATION_CHOICES = (
    "3600",
    "7200",
    "14400",
    "28800",
    "43200",
    "86400",
    "172800",
    "inf",
)

# key -> default provider. Callables so env-sourced defaults are read lazily
# (and remain patchable in tests).
_REGISTRY: dict[str, Callable[[], object]] = {
    # Gateway
    "gateway_enabled": lambda: True,
    "gateway_timeout_seconds": lambda: settings.gateway_timeout_seconds,
    # Deployments
    "start_timeout_seconds": lambda: settings.start_timeout_seconds,
    "default_port": lambda: 8001,
    "default_gpu_fraction": lambda: 0.5,
    "default_duration_choice": lambda: "43200",
    "default_vllm_version": lambda: "",
    "default_max_failed_restarts": lambda: None,
    # Notifications
    "webhook_url": lambda: settings.webhook_url or "",
    "expiry_warning_minutes": lambda: settings.expiry_warning_minutes,
    # Data
    "node_metrics_retention_hours": lambda: settings.node_metrics_retention_hours,
    # Advanced sync tuning
    "nodes_sync_interval_seconds": lambda: 10,
    "deployments_sync_interval_seconds": lambda: 5,
    "expiry_check_interval_seconds": lambda: 30,
    "node_failure_threshold": lambda: 3,
    "deployment_failure_threshold": lambda: 3,
}

# DB-sourced overrides; absent key = use the registry default.
_overrides: dict[str, object] = {}


def effective() -> dict[str, object]:
    """The full resolved settings dict (registry defaults + overrides)."""
    values = {key: provider() for key, provider in _REGISTRY.items()}
    values.update({k: v for k, v in _overrides.items() if k in _REGISTRY})
    return values


def get(key: str) -> object:
    if key in _overrides:
        return _overrides[key]
    return _REGISTRY[key]()


def get_bool(key: str) -> bool:
    return bool(get(key))


def get_int(key: str) -> int:
    return int(get(key))  # type: ignore[arg-type]


def get_float(key: str) -> float:
    return float(get(key))  # type: ignore[arg-type]


def get_str(key: str) -> str:
    value = get(key)
    return "" if value is None else str(value)


async def load(session) -> None:
    """Populate the override cache from the app_settings table (startup)."""
    result = await session.execute(select(AppSetting))
    _overrides.clear()
    for row in result.scalars().all():
        if row.key in _REGISTRY:
            _overrides[row.key] = row.value
        else:
            logger.warning("Ignoring unknown persisted setting %r", row.key)
    if _overrides:
        logger.info("Loaded %d setting override(s)", len(_overrides))


async def apply(session, values: dict[str, object]) -> None:
    """Upsert validated overrides and refresh the cache (values pre-validated
    by the API schema)."""
    for key, value in values.items():
        if key not in _REGISTRY:
            raise ValueError(f"Unknown setting: {key}")
        existing = await session.get(AppSetting, key)
        if existing is None:
            session.add(AppSetting(key=key, value=value))
        else:
            existing.value = value
    await session.commit()
    _overrides.update(values)
