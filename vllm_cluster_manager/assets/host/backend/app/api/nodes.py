from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.node import Node
from app.schemas.node import DiscoveredNode, NodeCreate, NodeRead
from app.services.consul import consul_service
from app.services.client_api import check_port, upload_package, get_packages

router = APIRouter()


@router.get("/", response_model=list[NodeRead])
async def list_nodes(session: AsyncSession = Depends(get_session)) -> list[NodeRead]:
    result = await session.execute(select(Node).order_by(Node.hostname))
    return list(result.scalars().all())


@router.post("/", response_model=NodeRead)
async def create_node(payload: NodeCreate, session: AsyncSession = Depends(get_session)) -> NodeRead:
    node = Node(**payload.model_dump())
    session.add(node)
    await session.commit()
    await session.refresh(node)
    return node


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
