from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment import Deployment
from app.models.node import Node
from app.schemas.deployment import DeploymentCreate, DeploymentRead, DeploymentStart
from app.services.client_api import get_logs, start_model, stop_model

router = APIRouter()


def _normalize_env_value(value: object) -> str:
    text = "" if value is None else str(value)
    trimmed = text.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {"'", '"'}:
        return trimmed[1:-1]
    return text


def _normalize_env_vars(
    env_vars: list[dict[str, str]] | None,
) -> list[dict[str, str]] | None:
    if not env_vars:
        return env_vars
    normalized: list[dict[str, str]] = []
    for pair in env_vars:
        if not isinstance(pair, dict):
            continue
        key = str(pair.get("key", "")).strip()
        if not key:
            continue
        normalized.append({"key": key, "value": _normalize_env_value(pair.get("value", ""))})
    return normalized


@router.get("/", response_model=list[DeploymentRead])
async def list_deployments(session: AsyncSession = Depends(get_session)) -> list[DeploymentRead]:
    result = await session.execute(select(Deployment).order_by(Deployment.id.desc()))
    return list(result.scalars().all())


@router.post("/", response_model=DeploymentRead)
async def create_deployment(
    payload: DeploymentCreate, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    payload_data = payload.model_dump()
    payload_data["env_vars"] = _normalize_env_vars(payload_data.get("env_vars"))
    deployment = Deployment(**payload_data)
    session.add(deployment)
    await session.commit()
    await session.refresh(deployment)
    return deployment


@router.post("/start", response_model=DeploymentRead)
async def start_deployment(
    payload: DeploymentStart, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    node = await session.get(Node, payload.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    active_statuses = {"running", "loading", "stopping"}
    existing = await session.execute(
        select(Deployment).where(
            Deployment.node_id == payload.node_id,
            Deployment.port == payload.port,
            Deployment.status.in_(active_statuses),
        )
    )
    conflict = existing.scalar_one_or_none()
    if conflict:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Port {payload.port} is already in use on {node.hostname} "
                f"by deployment {conflict.id} ({conflict.model_name})."
            ),
        )

    normalized_env = _normalize_env_vars(payload.env_vars)
    await start_model(
        node.ip_address,
        node.port,
        payload.model_name,
        payload.port,
        payload.gpu_memory_fraction,
        payload.gpu_ids,
        payload.tensor_parallel_size,
        payload.extra_args,
        normalized_env,
    )

    payload_data = payload.model_dump(exclude={"status"})
    payload_data["env_vars"] = normalized_env
    deployment = Deployment(**payload_data, status="loading")
    session.add(deployment)
    await session.commit()
    await session.refresh(deployment)
    return deployment


@router.post("/stop/{deployment_id}", response_model=DeploymentRead)
async def stop_deployment(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    key = f"{deployment.model_name}:{deployment.port}"
    await stop_model(node.ip_address, node.port, key)

    deployment.status = "stopped"
    await session.commit()
    await session.refresh(deployment)
    return deployment


@router.delete("/{deployment_id}")
async def delete_deployment(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> dict[str, str]:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    await session.delete(deployment)
    await session.commit()
    return {"status": "deleted"}


@router.get("/{deployment_id}/logs")
async def deployment_logs(
    deployment_id: int, tail: int = 200, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    key = f"{deployment.model_name}:{deployment.port}"
    return await get_logs(node.ip_address, node.port, key, tail=tail)
