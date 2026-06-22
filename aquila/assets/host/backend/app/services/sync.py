import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.db.session import SessionLocal
from app.models.node import Node
from app.models.node_metric import NodeMetric
from app.models.deployment import ACTIVE_STATUSES, Deployment
from app.core.config import settings
from app.services import runtime_settings
from app.services.consul import consul_service
from app.services.client_api import (
    get_metrics,
    get_statuses,
    list_containers,
    list_gpu_processes,
    list_warm_artifacts,
    push_node_config,
)
from app.services.deployment_state import set_status
from app.services.deployment_stop import stop_deployment_internal
from app.services.node_state import (
    rogue_container_counts,
    rogue_process_counts,
    rogue_artifact_counts,
    ram_cache_used_mb,
)
from app.services.notify import _warned_expiring, notify
from app.ws.manager import manager

logger = logging.getLogger(__name__)

# Consecutive failure counts before degrading status.
# Defaults only — the live values come from runtime_settings
# (node_failure_threshold / deployment_failure_threshold).
_NODE_FAILURE_THRESHOLD = 3
_DEPLOYMENT_FAILURE_THRESHOLD = 3

# Track consecutive non-healthy results per service_id / node_id.
_node_fail_counts: dict[str, int] = {}
_deployment_fail_counts: dict[int, int] = {}

# Consecutive whole-loop failures, so a permanently broken loop logs about
# once a minute instead of every tick.
_loop_fail_counts: dict[str, int] = {}


def _log_loop_error(loop_name: str, exc: Exception, every: int = 12) -> None:
    count = _loop_fail_counts.get(loop_name, 0)
    if count % every == 0:
        logger.error(
            "%s failed (%d consecutive): %s", loop_name, count + 1, exc, exc_info=True
        )
    _loop_fail_counts[loop_name] = count + 1


def _clear_loop_error(loop_name: str) -> None:
    if _loop_fail_counts.pop(loop_name, None):
        logger.info("%s recovered", loop_name)


async def _broadcast(message: dict) -> None:
    # Push notifications must never take down a sync tick.
    try:
        await manager.broadcast(message)
    except Exception as exc:
        logger.debug("WebSocket broadcast failed: %s", exc)


async def sync_nodes_from_consul(interval_seconds: int = 10) -> None:
    while True:
        try:
            services = consul_service.list_service("vllm-satellite")
            health = consul_service.service_health("vllm-satellite")
            nodes_changed = False
            async with SessionLocal() as session:
                for service in services:
                    service_id = service.get("ServiceID") or service.get("Node")
                    address = service.get("ServiceAddress") or service.get("Address")
                    service_port = service.get("ServicePort") or settings.satellite_port
                    if not service_id or not address:
                        continue
                    checks = health.get(service_id, [])
                    if checks:
                        if all(check == "passing" for check in checks):
                            consul_status = "healthy"
                        elif any(check == "critical" for check in checks):
                            consul_status = "critical"
                        else:
                            consul_status = "warning"
                    else:
                        consul_status = "unknown"

                    # Only degrade to critical after consecutive failures.
                    if consul_status in ("critical", "warning"):
                        _node_fail_counts[service_id] = _node_fail_counts.get(service_id, 0) + 1
                        if _node_fail_counts[service_id] < runtime_settings.get_int("node_failure_threshold"):
                            consul_status = "healthy"
                    else:
                        _node_fail_counts.pop(service_id, None)

                    result = await session.execute(
                        select(Node).where(Node.hostname == service_id)
                    )
                    node = result.scalar_one_or_none()

                    # A fully-cordoned node reports "maintenance" regardless
                    # of Consul health so expected downtime doesn't read as
                    # an outage.  Partial cordon suppresses flap counts but
                    # keeps the real health status.
                    if node is not None and node.maintenance:
                        consul_status = "maintenance"
                        _node_fail_counts.pop(service_id, None)
                    elif node is not None and node.maintenance_gpus:
                        _node_fail_counts.pop(service_id, None)

                    metrics: dict[str, object] | None = None
                    try:
                        metrics = await get_metrics(address, service_port)
                    except Exception as exc:
                        logger.debug("Metrics fetch from %s failed: %s", service_id, exc)
                    gpu_usage = metrics.get("gpus") if metrics else None
                    default_pip_packages = (
                        metrics.get("default_pip_packages") if metrics else None
                    )
                    installed_packages = (
                        metrics.get("installed_packages") if metrics else None
                    )
                    disk_usage = metrics.get("disk") if metrics else None
                    available_runtimes = (
                        metrics.get("available_runtimes") if metrics else None
                    )
                    # A reachable agent with no container runtime cannot run
                    # anything — surface that as the node status (hard fact,
                    # so the flap threshold doesn't apply).
                    if (
                        isinstance(available_runtimes, list)
                        and not available_runtimes
                        and consul_status != "maintenance"
                    ):
                        consul_status = "no-runtime"

                    if node is None:
                        node = Node(
                            hostname=service_id,
                            ip_address=address,
                            port=service_port,
                            status=consul_status,
                            last_heartbeat_at=datetime.now(timezone.utc),
                            gpu_usage=gpu_usage or [],
                            disk_usage=disk_usage,
                            default_pip_packages=default_pip_packages or [],
                            installed_packages=installed_packages or [],
                            available_runtimes=available_runtimes
                            if isinstance(available_runtimes, list)
                            else [],
                            warm_offload_enabled=runtime_settings.get_bool(
                                "default_warm_offload_enabled"
                            ),
                        )
                        session.add(node)
                        nodes_changed = True
                    else:
                        if node.status != consul_status:
                            nodes_changed = True
                        node.ip_address = address
                        node.port = service_port
                        node.status = consul_status
                        node.last_heartbeat_at = datetime.now(timezone.utc)
                        if gpu_usage is not None:
                            node.gpu_usage = gpu_usage
                        if disk_usage is not None:
                            node.disk_usage = disk_usage
                        if default_pip_packages is not None:
                            node.default_pip_packages = default_pip_packages
                        if installed_packages is not None:
                            node.installed_packages = installed_packages
                        if isinstance(available_runtimes, list):
                            node.available_runtimes = available_runtimes

                    # Keep a history sample for the metrics charts.
                    if metrics is not None:
                        await session.flush()  # ensure node.id for new nodes
                        session.add(
                            NodeMetric(
                                node_id=node.id,
                                recorded_at=datetime.now(timezone.utc),
                                gpus=gpu_usage or [],
                                cpu_percent=metrics.get("cpu_percent"),
                                memory_percent=metrics.get("memory_percent"),
                            )
                        )
                await session.commit()
            if nodes_changed:
                await _broadcast({"type": "nodes_changed"})
            _clear_loop_error("sync_nodes_from_consul")
        except Exception as exc:
            # Avoid crashing the API if Consul/DB is temporarily unavailable.
            _log_loop_error("sync_nodes_from_consul", exc)

        await asyncio.sleep(runtime_settings.get_int("nodes_sync_interval_seconds"))


# Last cumulative counters seen per deployment, to compute deltas. vLLM's
# counters reset when its container restarts; a value below the last seen one
# means "reset", in which case the whole new value is the delta.
_usage_last_seen: dict[int, dict[str, int]] = {}

# Live (non-persisted) per-deployment metrics from the latest vLLM scrape:
# tokens_per_second, requests_running, requests_waiting. Read by the
# deployments API and attached to running deployments only.
live_usage: dict[int, dict[str, float | int]] = {}

# Transient image-pull progress reported by clients while a deployment is
# starting: {deployment_id: {downloaded_mb, total_mb, percent}}. Attached to
# DeploymentRead by the deployments API; never persisted.
pull_progress: dict[int, dict[str, float | int]] = {}

_LIVE_USAGE_KEYS = (
    "prompt_tps",
    "generation_tps",
    "prompt_throughput",
    "generation_throughput",
    "requests_running",
    "requests_waiting",
)


def _update_live_usage(deployment_id: int, usage: dict[str, object]) -> None:
    values = {
        key: usage[key]
        for key in _LIVE_USAGE_KEYS
        if isinstance(usage.get(key), (int, float))
    }
    if values:
        live_usage[deployment_id] = values
    else:
        live_usage.pop(deployment_id, None)

_USAGE_FIELDS = (
    ("prompt_tokens", "total_prompt_tokens"),
    ("generation_tokens", "total_completion_tokens"),
    ("requests", "total_requests"),
)


def _persist_token_speeds(deployment, usage: dict[str, object]) -> None:
    """Store the latest read/generation averages on the row.

    Live values vanish with the agent's status report once a deployment
    stops; the persisted copy keeps the usage stats visible afterwards.
    """
    for field in ("prompt_tps", "generation_tps"):
        value = usage.get(field)
        if isinstance(value, (int, float)):
            setattr(deployment, field, float(value))


def _accumulate_usage(deployment, counters: dict[str, object]) -> None:
    """Fold a cumulative counter snapshot into the deployment's totals."""
    last = _usage_last_seen.get(deployment.id, {})
    seen: dict[str, int] = {}
    for metric, attr in _USAGE_FIELDS:
        value = counters.get(metric)
        if not isinstance(value, (int, float)):
            continue
        current = int(value)
        seen[metric] = current
        previous = last.get(metric)
        if previous is not None and current >= previous:
            delta = current - previous
        elif previous is not None:
            delta = current  # counter reset (vLLM container restarted)
        else:
            existing = int(getattr(deployment, attr) or 0)
            delta = current if existing == 0 else 0
        if delta > 0:
            setattr(deployment, attr, int(getattr(deployment, attr) or 0) + delta)
    if seen:
        _usage_last_seen[deployment.id] = seen


def _transition_event(deployment, before_status: str, node) -> tuple | None:
    """Map a status transition to a notification event, or None."""
    status = deployment.status
    if status == before_status:
        return None
    if status == "running":
        return (
            "deployment_running",
            f"{deployment.model_name} is ready on {node.hostname}:{deployment.port}",
            {"deployment_id": deployment.id, "model": deployment.model_name},
        )
    if status in ("error", "unreachable") and before_status not in ("error", "unreachable"):
        return (
            "deployment_error",
            f"{deployment.model_name} on {node.hostname}:{deployment.port} is {status}",
            {
                "deployment_id": deployment.id,
                "model": deployment.model_name,
                "error": deployment.last_error,
            },
        )
    return None


def _start_timed_out(deployment, now: datetime) -> bool:
    changed_at = deployment.status_changed_at or deployment.created_at
    if changed_at is None:
        return False
    return now - _as_aware(changed_at) > timedelta(
        seconds=runtime_settings.get_int("start_timeout_seconds")
    )


def _adopted_deployment(node_id: int, client_dep: dict, now: datetime) -> Deployment:
    """Build a Deployment row for a container the client runs but the DB lacks.

    The client's launch-manifest (stored as a container label at start) lets us
    restore owner, lease, and launch config after the host database is lost.
    Older containers without the manifest fall back to the bare client report.
    """
    deployment = Deployment(
        node_id=node_id,
        model_name=str(client_dep.get("model_name", "")),
        port=int(client_dep.get("port", 0)),
        gpu_memory_fraction=float(client_dep.get("gpu_memory_fraction") or 0.0),
        gpu_ids=client_dep.get("gpu_ids") or [],
        tensor_parallel_size=client_dep.get("tensor_parallel_size"),
        vllm_version=client_dep.get("vllm_version") or None,
        status=str(client_dep.get("status")),
        status_changed_at=now,
    )
    manifest = client_dep.get("launch_manifest")
    if not isinstance(manifest, dict):
        return deployment
    deployment.owner = manifest.get("owner")
    deployment.duration_seconds = manifest.get("duration_seconds")
    deployment.extra_args = manifest.get("extra_args") or []
    deployment.env_vars = manifest.get("env_vars") or []
    deployment.engine_args = manifest.get("engine_args") or {}
    deployment.lora_modules = manifest.get("lora_modules") or None
    deployment.extra_packages = manifest.get("extra_packages") or []
    deployment.max_failed_restarts = manifest.get("max_failed_restarts")
    runtime = manifest.get("container_runtime")
    if isinstance(runtime, str):
        deployment.container_runtime = runtime
    # Restore the launch-anchored lease verbatim; an already-elapsed lease is
    # then enforced by the expiry loop (the deployment outlived its grant).
    # Without it (but with a duration), the running-transition block grants a
    # fresh lease on the next tick.
    expires_at = manifest.get("expires_at")
    if isinstance(expires_at, str):
        try:
            deployment.expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            pass
    return deployment


async def sync_deployments_from_clients(interval_seconds: int = 5) -> None:
    while True:
        try:
            changed_ids: list[int] = []
            # Collected as primitives and fired only after a successful
            # commit, so a failed commit can't produce phantom notifications.
            events: list[tuple] = []
            async with SessionLocal() as session:
                result = await session.execute(select(Deployment))
                deployments = list(result.scalars().all())
                node_result = await session.execute(select(Node))
                nodes = {node.id: node for node in node_result.scalars().all()}
                now = datetime.now(timezone.utc)

                for node_id, node in nodes.items():
                    try:
                        statuses = await get_statuses(node.ip_address, node.port)
                        reachable = True
                        _deployment_fail_counts.pop(node_id, None)
                    except Exception as exc:
                        _deployment_fail_counts[node_id] = (
                            _deployment_fail_counts.get(node_id, 0) + 1
                        )
                        if _deployment_fail_counts[node_id] < runtime_settings.get_int("deployment_failure_threshold"):
                            # Keep previous deployment statuses on transient failure.
                            continue
                        if _deployment_fail_counts[node_id] == runtime_settings.get_int("deployment_failure_threshold"):
                            logger.warning(
                                "Client %s unreachable for %d checks: %s",
                                node.hostname,
                                runtime_settings.get_int("deployment_failure_threshold"),
                                exc,
                            )
                        reachable = False
                        statuses = []

                    # Surface rogue (untracked) vLLM containers in the node list.
                    if reachable:
                        try:
                            containers = await list_containers(node.ip_address, node.port)
                            rogue_container_counts[node_id] = sum(
                                1 for c in containers if not c.get("tracked")
                            )
                        except Exception as exc:
                            # Docker unreachable / transient error: keep last known.
                            logger.debug(
                                "Container list from %s failed: %s", node.hostname, exc
                            )

                    # Surface orphaned vLLM GPU processes (no live container).
                    if reachable:
                        try:
                            processes = await list_gpu_processes(node.ip_address, node.port)
                            rogue_process_counts[node_id] = sum(
                                1 for p in processes if not p.get("tracked")
                            )
                        except Exception as exc:
                            # GPU/process query unavailable: keep last known.
                            logger.debug(
                                "GPU process list from %s failed: %s", node.hostname, exc
                            )

                    client_dep_map: dict[str, dict] = {
                        s.get("key"): s for s in statuses if s.get("key")
                    }

                    # Build a set of known deployment keys for this node
                    node_deployments = [
                        d for d in deployments if d.node_id == node_id
                    ]
                    known_keys = {
                        f"{d.model_name}:{d.port}" for d in node_deployments
                    }
                    active_ports = {
                        d.port for d in node_deployments if d.status in ACTIVE_STATUSES
                    }

                    # Push the node's warm-cache policy + live pin set so the
                    # agent's autonomous (direct-call) wakes use current settings,
                    # and surface RAM-cache usage + orphaned warm artifacts.
                    if reachable:
                        pins = [
                            f"{d.model_name}:{d.port}"
                            for d in node_deployments
                            if getattr(d, "pinned", False)
                        ]
                        try:
                            await push_node_config(
                                node.ip_address,
                                node.port,
                                bool(node.warm_offload_enabled),
                                node.ram_cache_limit_mb,
                                pins,
                                busy_guard_seconds=runtime_settings.get_int("busy_guard_seconds"),
                            )
                        except Exception as exc:
                            logger.debug("Config push to %s failed: %s", node.hostname, exc)
                        ram_cache_used_mb[node_id] = sum(
                            float(s.get("paused_ram_mb") or 0.0)
                            for s in statuses
                            if s.get("pause_tier") == "ram"
                        )
                        if node.warm_offload_enabled:
                            try:
                                artifacts = await list_warm_artifacts(
                                    node.ip_address, node.port
                                )
                                rogue_artifact_counts[node_id] = len(
                                    artifacts.get("ram_sleepers") or []
                                ) + len(artifacts.get("disk_caches") or [])
                            except Exception as exc:
                                logger.debug(
                                    "Warm-artifact list from %s failed: %s",
                                    node.hostname,
                                    exc,
                                )

                    for deployment in node_deployments:
                        # 'expired' is a terminal state owned by the expiry task;
                        # never let the sync loop revert it to stopped/running.
                        if deployment.status == "expired":
                            continue
                        before = (deployment.status, deployment.detail)
                        key = f"{deployment.model_name}:{deployment.port}"
                        if not reachable:
                            # Expected downtime on a cordoned node: keep the
                            # last known status instead of flapping.  For
                            # partial maintenance, suppress only deployments
                            # whose GPUs are all in maintenance.
                            if not node.maintenance:
                                dep_gpus = set(deployment.gpu_ids) if deployment.gpu_ids else None
                                maint = set(node.maintenance_gpus or [])
                                if dep_gpus is None or not (dep_gpus <= maint):
                                    set_status(deployment, "unreachable")
                            if (deployment.status, deployment.detail) != before:
                                changed_ids.append(deployment.id)
                                event = _transition_event(deployment, before[0], node)
                                if event:
                                    events.append(event)
                            continue
                        client_dep = client_dep_map.get(key)
                        if client_dep:
                            client_status = str(client_dep.get("status", "unknown"))
                            client_error = client_dep.get("error")
                            set_status(
                                deployment,
                                client_status,
                                error=str(client_error) if client_error else None,
                            )
                            phase = client_dep.get("phase")
                            deployment.detail = (
                                str(phase)
                                if phase and client_status in ("starting", "loading", "offloading")
                                else None
                            )
                            # Transient image-pull progress (starting only).
                            progress = client_dep.get("pull_progress")
                            if isinstance(progress, dict) and client_status in (
                                "starting",
                                "loading",
                            ):
                                pull_progress[deployment.id] = progress
                            else:
                                pull_progress.pop(deployment.id, None)
                            # Backfill vllm_version if missing
                            if not deployment.vllm_version and client_dep.get("vllm_version"):
                                deployment.vllm_version = str(client_dep["vllm_version"])
                            # Keep the exact image identity for provenance.
                            client_digest = client_dep.get("image_digest")
                            if client_digest and deployment.image_digest != str(client_digest):
                                deployment.image_digest = str(client_digest)
                            # Token accounting from the vLLM Prometheus scrape.
                            # Deliberately not added to changed_ids: totals
                            # tick every cycle and would spam the websocket.
                            usage = client_dep.get("usage")
                            if isinstance(usage, dict):
                                _accumulate_usage(deployment, usage)
                                _update_live_usage(deployment.id, usage)
                                _persist_token_speeds(deployment, usage)
                            # Start the serve countdown the first time the model is
                            # actually serving (status -> running).
                            if (
                                deployment.status == "running"
                                and deployment.expires_at is None
                                and deployment.duration_seconds is not None
                            ):
                                deployment.expires_at = now + timedelta(
                                    seconds=deployment.duration_seconds
                                )
                        elif deployment.status in ("starting", "loading", "offloading"):
                            # The start request may still be in flight on the
                            # client (image pull etc.); only give up after the
                            # configured timeout.
                            if _start_timed_out(deployment, now):
                                set_status(
                                    deployment,
                                    "error",
                                    error=(
                                        f"Start timed out: {node.hostname} never "
                                        f"reported deployment {key} within "
                                        f"{runtime_settings.get_int('start_timeout_seconds')}s."
                                    ),
                                )
                        elif deployment.status == "running":
                            set_status(
                                deployment,
                                "error",
                                error=(
                                    f"Container for {key} disappeared from "
                                    f"{node.hostname} without a stop request."
                                ),
                            )
                        else:
                            set_status(deployment, "stopped")
                        if not client_dep:
                            pull_progress.pop(deployment.id, None)
                        if (deployment.status, deployment.detail) != before:
                            changed_ids.append(deployment.id)
                            event = _transition_event(deployment, before[0], node)
                            if event:
                                events.append(event)

                    # Create DB rows for deployments the client knows about
                    # but the backend doesn't (e.g. after a backend restart).
                    if reachable:
                        for client_dep in statuses:
                            dep_key = client_dep.get("key")
                            if not dep_key or dep_key in known_keys:
                                continue
                            client_status = client_dep.get("status")
                            if client_status in ("stopped", "error"):
                                continue
                            port = int(client_dep.get("port", 0))
                            # An active row already holds this port (e.g. a
                            # start in flight) — adopting would violate the
                            # unique port reservation.
                            if port in active_ports:
                                continue
                            session.add(
                                _adopted_deployment(node_id, client_dep, now)
                            )
                            logger.info(
                                "Adopted untracked deployment %s on %s",
                                dep_key,
                                node.hostname,
                            )

                # The expiry loop may have committed "expired" for a
                # deployment while this sync tick was in flight.
                # Re-read from DB without autoflush (so we see the
                # committed state, not our pending ORM writes) and
                # revert any accidental overwrites before committing.
                with session.no_autoflush:
                    _expired_in_db = {
                        row[0]
                        for row in (
                            await session.execute(
                                select(Deployment.id).where(
                                    Deployment.status == "expired"
                                )
                            )
                        ).all()
                    }
                for d in deployments:
                    if d.id in _expired_in_db and d.status != "expired":
                        set_status(d, "expired")
                        d.expires_at = None

                await session.commit()
            if changed_ids:
                await _broadcast({"type": "deployments_changed", "ids": changed_ids})
            for event, message, fields in events:
                notify(event, message, fields)
            _clear_loop_error("sync_deployments_from_clients")
        except Exception as exc:
            _log_loop_error("sync_deployments_from_clients", exc)

        await asyncio.sleep(runtime_settings.get_int("deployments_sync_interval_seconds"))


# Statuses considered "live" — eligible for expiry once past expires_at.
# paused_ram counts: the model is still served (first request wakes it).
_LIVE_STATUSES = ("running", "loading", "paused_ram")


def _as_aware(dt: datetime) -> datetime:
    """Treat naive timestamps (some DB backends drop tzinfo) as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def expire_due_deployments(session, now: datetime | None = None) -> list[int]:
    """Stop and mark 'expired' any live deployment past its expires_at.

    Stops the container exactly like a manual stop (stop_model -> client
    container.stop + remove). Returns the ids that were expired.
    """
    now = now or datetime.now(timezone.utc)
    result = await session.execute(
        select(Deployment).where(Deployment.status.in_(_LIVE_STATUSES))
    )
    due = [
        d
        for d in result.scalars().all()
        if d.expires_at is not None and _as_aware(d.expires_at) <= now
    ]
    for dep in due:
        node = await session.get(Node, dep.node_id)
        # Best effort: still mark expired so it isn't retried forever.
        await stop_deployment_internal(
            dep, node, final_status="expired", best_effort=True
        )
    await session.commit()
    if due:
        logger.info("Expired deployments: %s", [d.id for d in due])
        await _broadcast(
            {"type": "deployments_changed", "ids": [d.id for d in due]}
        )
        for dep in due:
            notify(
                "deployment_expired",
                f"{dep.model_name} (deployment {dep.id}) reached its serve "
                "duration and was stopped",
                {"deployment_id": dep.id, "model": dep.model_name},
            )
    return [d.id for d in due]


async def warn_expiring_deployments(session, now: datetime | None = None) -> list[int]:
    """Notify once per (deployment, expiry time) shortly before auto-expiry.

    Keying the de-dup on the expiry timestamp means an extension re-arms the
    warning automatically. Returns the ids warned about (for tests).
    """
    now = now or datetime.now(timezone.utc)
    window = timedelta(minutes=runtime_settings.get_int("expiry_warning_minutes"))
    result = await session.execute(
        select(Deployment).where(Deployment.status.in_(_LIVE_STATUSES))
    )
    live = list(result.scalars().all())
    warned: list[int] = []
    live_keys: set[tuple[int, str]] = set()
    for dep in live:
        if dep.expires_at is None:
            continue
        expires_at = _as_aware(dep.expires_at)
        key = (dep.id, expires_at.isoformat())
        live_keys.add(key)
        remaining = expires_at - now
        if timedelta(0) < remaining <= window and key not in _warned_expiring:
            _warned_expiring.add(key)
            warned.append(dep.id)
            minutes = max(1, int(remaining.total_seconds() // 60))
            notify(
                "deployment_expiring",
                f"{dep.model_name} (deployment {dep.id}) expires in ~{minutes}m — "
                "extend it from the dashboard to keep it serving",
                {"deployment_id": dep.id, "model": dep.model_name},
            )
    # Bound the set: drop keys whose deployment/expiry is no longer live.
    _warned_expiring.intersection_update(live_keys)
    return warned


async def prune_node_metrics(session, now: datetime | None = None) -> None:
    """Drop metric samples older than the retention window."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=runtime_settings.get_int("node_metrics_retention_hours"))
    await session.execute(delete(NodeMetric).where(NodeMetric.recorded_at < cutoff))
    await session.commit()


async def enforce_deployment_expiry(interval_seconds: int = 30) -> None:
    while True:
        try:
            async with SessionLocal() as session:
                await expire_due_deployments(session)
                await warn_expiring_deployments(session)
                await prune_node_metrics(session)
            _clear_loop_error("enforce_deployment_expiry")
        except Exception as exc:
            _log_loop_error("enforce_deployment_expiry", exc)
        await asyncio.sleep(runtime_settings.get_int("expiry_check_interval_seconds"))
