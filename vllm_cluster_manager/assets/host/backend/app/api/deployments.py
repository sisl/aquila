from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment import Deployment
from app.models.node import Node
from app.schemas.deployment import DeploymentCreate, DeploymentRead, DeploymentStart
from app.services.client_api import get_logs, start_model, stop_model

router = APIRouter()


@router.get("/", response_model=list[DeploymentRead])
async def list_deployments(session: AsyncSession = Depends(get_session)) -> list[DeploymentRead]:
    result = await session.execute(select(Deployment).order_by(Deployment.id.desc()))
    return list(result.scalars().all())


@router.post("/", response_model=DeploymentRead)
async def create_deployment(
    payload: DeploymentCreate, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    deployment = Deployment(**payload.model_dump())
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

    client_result = await start_model(
        node.ip_address,
        node.port,
        payload.model_name,
        payload.port,
        payload.gpu_memory_fraction,
        payload.gpu_ids,
        payload.tensor_parallel_size,
        payload.extra_args,
        payload.env_vars,
        payload.vllm_version,
        payload.extra_packages,
    )

    payload_data = payload.model_dump(exclude={"status"})
    # Use the resolved version from the client (e.g. latest stable when left blank)
    resolved_version = client_result.get("vllm_version") if client_result else None
    if resolved_version:
        payload_data["vllm_version"] = resolved_version
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
