import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment import ACTIVE_STATUSES, Deployment
from app.models.node import Node
from app.models.node_metric import NodeMetric
from app.schemas.node import (
    DiscoveredNode,
    NodeCreate,
    NodeMaintenanceRequest,
    NodeRead,
    NodeSetRuntimeRequest,
    NodeWarmCacheRequest,
)
from app.services import sync as sync_service
from app.services.consul import consul_service
from app.services.deployment_stop import stop_deployment_internal
from app.services.node_state import (
    rogue_container_counts,
    rogue_process_counts,
    rogue_artifact_counts,
    ram_cache_used_mb,
)
from app.services.notify import _warned_expiring
from app.services.client_api import (
    check_port,
    upload_package,
    get_packages,
    list_containers,
    stop_container,
    list_gpu_processes,
    kill_gpu_process,
    push_node_config,
    list_warm_artifacts,
    kill_ram_sleeper,
    delete_warm_cache,
    list_images,
    delete_image,
    prune_images,
    list_model_cache,
    delete_model_cache,
    begin_local_model_upload,
    upload_local_model_file,
    finish_local_model_upload,
    abort_local_model_upload,
    upload_local_model_archive,
    pull_local_model,
    get_local_model_transfers,
    list_local_models,
    delete_local_model,
)
from app.services import runtime_settings
from app.ws.manager import manager

router = APIRouter()
logger = logging.getLogger(__name__)


def _attach_derived(node: Node) -> None:
    """Merge transient, sync-loop-derived values onto a node (not DB columns)."""
    node.rogue_container_count = rogue_container_counts.get(node.id)
    node.rogue_process_count = rogue_process_counts.get(node.id)
    node.rogue_artifact_count = rogue_artifact_counts.get(node.id)
    node.ram_cache_used_mb = ram_cache_used_mb.get(node.id)
    node.partial_maintenance = node.has_partial_maintenance


@router.get("/", response_model=list[NodeRead])
async def list_nodes(session: AsyncSession = Depends(get_session)) -> list[NodeRead]:
    result = await session.execute(select(Node).order_by(Node.hostname))
    nodes = list(result.scalars().all())
    for node in nodes:
        _attach_derived(node)
    return nodes


@router.post("/", response_model=NodeRead)
async def create_node(payload: NodeCreate, session: AsyncSession = Depends(get_session)) -> NodeRead:
    node = Node(**payload.model_dump())
    session.add(node)
    await session.commit()
    await session.refresh(node)
    return node


@router.delete("/{node_id}")
async def delete_node(
    node_id: int, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    """Remove a node and its deployment records (the stale-node escape hatch).

    Containers on the node are untouched: a live node re-registers via Consul
    within seconds and its deployments are re-adopted from container manifests;
    a stale node disappears for good.
    """
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    hostname = node.hostname

    result = await session.execute(
        select(Deployment.id).where(Deployment.node_id == node_id)
    )
    deployment_ids = [row[0] for row in result.all()]

    # deployments.node_id has no ON DELETE cascade — clear children first.
    await session.execute(delete(Deployment).where(Deployment.node_id == node_id))
    await session.execute(delete(NodeMetric).where(NodeMetric.node_id == node_id))
    await session.delete(node)
    await session.commit()

    # Best-effort: drop the Consul registration, or the sync loop re-creates
    # a stale node within seconds. A live agent re-registers itself anyway.
    try:
        consul_service.deregister_service(hostname)
    except Exception as exc:
        logger.warning("Consul deregistration of %s failed: %s", hostname, exc)

    # Drop in-memory state keyed by the deleted ids.
    rogue_container_counts.pop(node_id, None)
    rogue_process_counts.pop(node_id, None)
    rogue_artifact_counts.pop(node_id, None)
    ram_cache_used_mb.pop(node_id, None)
    sync_service._node_fail_counts.pop(hostname, None)
    for dep_id in deployment_ids:
        sync_service.live_usage.pop(dep_id, None)
        sync_service.pull_progress.pop(dep_id, None)
        sync_service._usage_last_seen.pop(dep_id, None)
    for entry in [e for e in _warned_expiring if e[0] in deployment_ids]:
        _warned_expiring.discard(entry)

    await manager.broadcast({"type": "nodes_changed"})
    await manager.broadcast({"type": "deployments_changed"})
    return {
        "status": "deleted",
        "hostname": hostname,
        "deployments_deleted": len(deployment_ids),
    }


@router.post("/{node_id}/runtime", response_model=NodeRead)
async def set_node_runtime(
    node_id: int,
    payload: NodeSetRuntimeRequest,
    session: AsyncSession = Depends(get_session),
) -> NodeRead:
    """Set the node's container runtime override (null = auto).

    Applies to new deployments only; running containers keep the runtime
    they started with.
    """
    if payload.runtime is not None and payload.runtime not in ("docker", "podman"):
        raise HTTPException(
            status_code=400, detail="Runtime must be 'docker', 'podman', or null."
        )
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    node.container_runtime = payload.runtime
    await session.commit()
    await session.refresh(node)
    _attach_derived(node)
    await manager.broadcast({"type": "nodes_changed"})
    return node


@router.post("/{node_id}/warm-cache", response_model=NodeRead)
async def set_node_warm_cache(
    node_id: int,
    payload: NodeWarmCacheRequest,
    session: AsyncSession = Depends(get_session),
) -> NodeRead:
    """Enable/disable warm-cache auto-offload and set the RAM-cache budget.

    The toggle gates new deployments into warm mode; already-running ones keep
    their mode until redeployed. The policy is also pushed to the live agent so
    request-triggered wakes use the current limit immediately.
    """
    if payload.ram_cache_limit_mb is not None and payload.ram_cache_limit_mb <= 0:
        raise HTTPException(
            status_code=400, detail="ram_cache_limit_mb must be positive or null (unlimited)."
        )
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    node.warm_offload_enabled = payload.enabled
    node.ram_cache_limit_mb = payload.ram_cache_limit_mb
    await session.commit()
    await session.refresh(node)
    # Best-effort push; the sync loop re-pushes each tick, so an unreachable
    # node still converges.
    try:
        await push_node_config(
            node.ip_address, node.port, node.warm_offload_enabled, node.ram_cache_limit_mb,
            busy_guard_seconds=runtime_settings.get_int("busy_guard_seconds"),
        )
    except Exception as exc:
        logger.warning("Pushing warm-cache config to %s failed: %s", node.hostname, exc)
    _attach_derived(node)
    await manager.broadcast({"type": "nodes_changed"})
    return node


@router.post("/{node_id}/maintenance", response_model=NodeRead)
async def set_node_maintenance(
    node_id: int,
    payload: NodeMaintenanceRequest,
    session: AsyncSession = Depends(get_session),
) -> NodeRead:
    """Cordon/uncordon specific GPUs (or all); optionally drain affected deployments."""
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    all_gpu_indices = [g.get("index", i) for i, g in enumerate(node.gpu_usage or [])]
    target_gpus = payload.gpu_ids if payload.gpu_ids else all_gpu_indices

    if payload.gpu_ids:
        invalid = set(payload.gpu_ids) - set(all_gpu_indices)
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"GPU indices {sorted(invalid)} not found on {node.hostname}. "
                       f"Available: {sorted(all_gpu_indices)}",
            )

    current = set(node.maintenance_gpus or [])
    if payload.enabled:
        current |= set(target_gpus)
    else:
        current -= set(target_gpus)
    node.maintenance_gpus = sorted(current)

    if node.maintenance:
        node.status = "maintenance"

    drained_ids: list[int] = []
    if payload.enabled and payload.drain:
        newly_cordoned = set(target_gpus)
        result = await session.execute(
            select(Deployment).where(
                Deployment.node_id == node_id,
                Deployment.status.in_(ACTIVE_STATUSES),
            )
        )
        for deployment in result.scalars().all():
            dep_gpus = set(deployment.gpu_ids) if deployment.gpu_ids else set(all_gpu_indices)
            if dep_gpus & newly_cordoned:
                await stop_deployment_internal(deployment, node, best_effort=True)
                drained_ids.append(deployment.id)

    await session.commit()
    await session.refresh(node)
    _attach_derived(node)
    await manager.broadcast({"type": "nodes_changed"})
    if drained_ids:
        await manager.broadcast({"type": "deployments_changed", "ids": drained_ids})
    return node


@router.get("/{node_id}/metrics/history")
async def node_metrics_history(
    node_id: int,
    minutes: int = 60,
    step: int = 1,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Metric samples for the last *minutes*, downsampled to every *step*-th row."""
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    minutes = max(1, min(minutes, 60 * 48))
    step = max(1, step)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    result = await session.execute(
        select(NodeMetric)
        .where(NodeMetric.node_id == node_id, NodeMetric.recorded_at >= cutoff)
        .order_by(NodeMetric.recorded_at)
    )
    samples = list(result.scalars().all())
    points = [
        {
            "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
            "gpus": m.gpus or [],
            "cpu_percent": m.cpu_percent,
            "memory_percent": m.memory_percent,
        }
        for m in samples[::step]
    ]
    return {"node_id": node_id, "points": points}


@router.get("/{node_id}/models/cache")
async def list_node_model_cache(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await list_model_cache(node.ip_address, node.port)


@router.delete("/{node_id}/models/cache/{name:path}")
async def delete_node_model_cache(
    node_id: int,
    name: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await delete_model_cache(node.ip_address, node.port, name)


# --- Managed local models (streamed uploads / URL pulls) -------------------


async def _node_or_404(session: AsyncSession, node_id: int) -> Node:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.post("/{node_id}/local-models/upload/begin")
async def begin_node_local_model_upload(
    node_id: int,
    payload: dict,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await begin_local_model_upload(node.ip_address, node.port, payload)


@router.put("/{node_id}/local-models/upload/{session_id}/file")
async def upload_node_local_model_file(
    node_id: int,
    session_id: str,
    path: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await upload_local_model_file(
        node.ip_address,
        node.port,
        session_id,
        path,
        request.stream(),
        request.headers.get("content-length"),
    )


@router.post("/{node_id}/local-models/upload/{session_id}/finish")
async def finish_node_local_model_upload(
    node_id: int,
    session_id: str,
    payload: dict | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await finish_local_model_upload(node.ip_address, node.port, session_id, payload)


@router.post("/{node_id}/local-models/upload/{session_id}/abort")
async def abort_node_local_model_upload(
    node_id: int,
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await abort_local_model_upload(node.ip_address, node.port, session_id)


@router.post("/{node_id}/local-models/archive")
async def upload_node_local_model_archive(
    node_id: int,
    name: str,
    filename: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await upload_local_model_archive(
        node.ip_address,
        node.port,
        name,
        filename,
        request.stream(),
        request.headers.get("content-length"),
    )


@router.post("/{node_id}/local-models/pull")
async def pull_node_local_model(
    node_id: int,
    payload: dict,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await pull_local_model(node.ip_address, node.port, payload)


@router.get("/{node_id}/local-models/transfers")
async def list_node_local_model_transfers(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await _node_or_404(session, node_id)
    return await get_local_model_transfers(node.ip_address, node.port)


@router.get("/{node_id}/local-models")
async def list_node_local_models(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await _node_or_404(session, node_id)
    return await list_local_models(node.ip_address, node.port)


@router.delete("/{node_id}/local-models/{name}")
async def delete_node_local_model(
    node_id: int,
    name: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await _node_or_404(session, node_id)
    return await delete_local_model(node.ip_address, node.port, name)


@router.get("/discovered", response_model=dict[str, list[DiscoveredNode]])
async def discovered_nodes() -> dict[str, list[DiscoveredNode]]:
    services = consul_service.list_service("vllm-satellite")
    cleaned = [
        {
            "node": s.get("Node"),
            "address": s.get("ServiceAddress") or s.get("Address"),
            "port": s.get("ServicePort"),
            "service_id": s.get("ServiceID"),
        }
        for s in services
    ]
    return {"nodes": cleaned}


@router.get("/{node_id}/ports/check")
async def check_node_port(
    node_id: int, port: int, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await check_port(node.ip_address, node.port, port)


@router.post("/{node_id}/packages/upload")
async def upload_node_package(
    node_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    content = await file.read()
    return await upload_package(node.ip_address, node.port, file.filename or "package", content)


@router.get("/{node_id}/packages")
async def list_node_packages(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await get_packages(node.ip_address, node.port)


@router.get("/{node_id}/containers")
async def list_node_containers(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await list_containers(node.ip_address, node.port)


@router.post("/{node_id}/containers/{container_id}/stop")
async def stop_node_container(
    node_id: int,
    container_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await stop_container(node.ip_address, node.port, container_id)


@router.get("/{node_id}/gpu-processes")
async def list_node_gpu_processes(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await list_gpu_processes(node.ip_address, node.port)


@router.post("/{node_id}/gpu-processes/{pid}/kill")
async def kill_node_gpu_process(
    node_id: int,
    pid: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await kill_gpu_process(node.ip_address, node.port, pid)


@router.get("/{node_id}/warm-artifacts")
async def list_node_warm_artifacts(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await list_warm_artifacts(node.ip_address, node.port)


@router.post("/{node_id}/warm-artifacts/sleepers/{pid}/kill")
async def kill_node_ram_sleeper(
    node_id: int,
    pid: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await kill_ram_sleeper(node.ip_address, node.port, pid)


@router.delete("/{node_id}/warm-artifacts/caches/{name}")
async def delete_node_warm_cache(
    node_id: int,
    name: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await delete_warm_cache(node.ip_address, node.port, name)


@router.get("/{node_id}/images")
async def list_node_images(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await list_images(node.ip_address, node.port)


@router.delete("/{node_id}/images/{image_id}")
async def delete_node_image(
    node_id: int,
    image_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await delete_image(node.ip_address, node.port, image_id)


@router.post("/{node_id}/images/prune")
async def prune_node_images(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return await prune_images(node.ip_address, node.port)
