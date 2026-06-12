from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy import select
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
)
from app.services.consul import consul_service
from app.services.deployment_stop import stop_deployment_internal
from app.services.node_state import rogue_container_counts
from app.services.client_api import (
    check_port,
    upload_package,
    get_packages,
    list_containers,
    stop_container,
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
from app.ws.manager import manager

router = APIRouter()


@router.get("/", response_model=list[NodeRead])
async def list_nodes(session: AsyncSession = Depends(get_session)) -> list[NodeRead]:
    result = await session.execute(select(Node).order_by(Node.hostname))
    nodes = list(result.scalars().all())
    # Attach the transient, sync-loop-derived rogue container count (not a DB column).
    for node in nodes:
        node.rogue_container_count = rogue_container_counts.get(node.id)
    return nodes


@router.post("/", response_model=NodeRead)
async def create_node(payload: NodeCreate, session: AsyncSession = Depends(get_session)) -> NodeRead:
    node = Node(**payload.model_dump())
    session.add(node)
    await session.commit()
    await session.refresh(node)
    return node


@router.post("/{node_id}/maintenance", response_model=NodeRead)
async def set_node_maintenance(
    node_id: int,
    payload: NodeMaintenanceRequest,
    session: AsyncSession = Depends(get_session),
) -> NodeRead:
    """Cordon/uncordon a node; optionally drain its active deployments."""
    node = await session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    node.maintenance = payload.enabled
    node.status = "maintenance" if payload.enabled else node.status

    drained_ids: list[int] = []
    if payload.enabled and payload.drain:
        result = await session.execute(
            select(Deployment).where(
                Deployment.node_id == node_id,
                Deployment.status.in_(ACTIVE_STATUSES),
            )
        )
        for deployment in result.scalars().all():
            await stop_deployment_internal(deployment, node, best_effort=True)
            drained_ids.append(deployment.id)

    await session.commit()
    await session.refresh(node)
    node.rogue_container_count = rogue_container_counts.get(node.id)
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
