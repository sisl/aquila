import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.node import Node
from app.models.deployment import Deployment
from app.core.config import settings
from app.services.consul import consul_service
from app.services.client_api import get_metrics, get_statuses


async def sync_nodes_from_consul(interval_seconds: int = 10) -> None:
    while True:
        try:
            services = consul_service.list_service("vllm-satellite")
            health = consul_service.service_health("vllm-satellite")
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
                            status = "healthy"
                        elif any(check == "critical" for check in checks):
                            status = "critical"
                        else:
                            status = "warning"
                    else:
                        status = "unknown"

                    result = await session.execute(
                        select(Node).where(Node.hostname == service_id)
                    )
                    node = result.scalar_one_or_none()
                    gpu_usage: list[dict[str, object]] | None = None
                    try:
                        metrics = await get_metrics(address, service_port)
                        gpu_usage = metrics.get("gpus")
                    except Exception:
                        gpu_usage = None

                    if node is None:
                        node = Node(
                            hostname=service_id,
                            ip_address=address,
                            port=service_port,
                            status=status,
                            last_heartbeat_at=datetime.now(timezone.utc),
                            gpu_usage=gpu_usage or [],
                        )
                        session.add(node)
                    else:
                        node.ip_address = address
                        node.port = service_port
                        node.status = status
                        node.last_heartbeat_at = datetime.now(timezone.utc)
                        if gpu_usage is not None:
                            node.gpu_usage = gpu_usage

                await session.commit()
        except Exception:
            # Avoid crashing the API if Consul is temporarily unavailable.
            pass

        await asyncio.sleep(interval_seconds)


async def sync_deployments_from_clients(interval_seconds: int = 5) -> None:
    while True:
        try:
            async with SessionLocal() as session:
                result = await session.execute(select(Deployment))
                deployments = list(result.scalars().all())
                node_result = await session.execute(select(Node))
                nodes = {node.id: node for node in node_result.scalars().all()}

                for node_id, node in nodes.items():
                    try:
                        statuses = await get_statuses(node.ip_address, node.port)
                        reachable = True
                    except Exception:
                        reachable = False
                        statuses = []

                    status_map = {
                        status.get("key"): status.get("status") for status in statuses
                    }

                    for deployment in deployments:
                        if deployment.node_id != node_id:
                            continue
                        key = f"{deployment.model_name}:{deployment.port}"
                        if not reachable:
                            deployment.status = "unreachable"
                            continue
                        client_status = status_map.get(key)
                        if client_status:
                            deployment.status = str(client_status)
                        else:
                            deployment.status = "stopped"

                await session.commit()
        except Exception:
            pass

        await asyncio.sleep(interval_seconds)
