import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.node import Node
from app.models.deployment import Deployment
from app.core.config import settings
from app.services.consul import consul_service
from app.services.client_api import get_metrics, get_statuses

# Consecutive failure counts before degrading status.
_NODE_FAILURE_THRESHOLD = 3
_DEPLOYMENT_FAILURE_THRESHOLD = 3

# Track consecutive non-healthy results per service_id / node_id.
_node_fail_counts: dict[str, int] = {}
_deployment_fail_counts: dict[int, int] = {}


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
                        if _node_fail_counts[service_id] < _NODE_FAILURE_THRESHOLD:
                            consul_status = "healthy"
                    else:
                        _node_fail_counts.pop(service_id, None)

                    result = await session.execute(
                        select(Node).where(Node.hostname == service_id)
                    )
                    node = result.scalar_one_or_none()
                    gpu_usage: list[dict[str, object]] | None = None
                    default_pip_packages: list[str] | None = None
                    installed_packages: list[str] | None = None
                    try:
                        metrics = await get_metrics(address, service_port)
                        gpu_usage = metrics.get("gpus")
                        default_pip_packages = metrics.get("default_pip_packages")
                        installed_packages = metrics.get("installed_packages")
                    except Exception:
                        gpu_usage = None

                    if node is None:
                        node = Node(
                            hostname=service_id,
                            ip_address=address,
                            port=service_port,
                            status=consul_status,
                            last_heartbeat_at=datetime.now(timezone.utc),
                            gpu_usage=gpu_usage or [],
                            default_pip_packages=default_pip_packages or [],
                            installed_packages=installed_packages or [],
                        )
                        session.add(node)
                    else:
                        node.ip_address = address
                        node.port = service_port
                        node.status = consul_status
                        node.last_heartbeat_at = datetime.now(timezone.utc)
                        if gpu_usage is not None:
                            node.gpu_usage = gpu_usage
                        if default_pip_packages is not None:
                            node.default_pip_packages = default_pip_packages
                        if installed_packages is not None:
                            node.installed_packages = installed_packages
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
                        _deployment_fail_counts.pop(node_id, None)
                    except Exception:
                        _deployment_fail_counts[node_id] = (
                            _deployment_fail_counts.get(node_id, 0) + 1
                        )
                        if _deployment_fail_counts[node_id] < _DEPLOYMENT_FAILURE_THRESHOLD:
                            # Keep previous deployment statuses on transient failure.
                            continue
                        reachable = False
                        statuses = []

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

                    for deployment in node_deployments:
                        key = f"{deployment.model_name}:{deployment.port}"
                        if not reachable:
                            deployment.status = "unreachable"
                            continue
                        client_dep = client_dep_map.get(key)
                        if client_dep:
                            deployment.status = str(client_dep.get("status", "unknown"))
                            # Backfill vllm_version if missing
                            if not deployment.vllm_version and client_dep.get("vllm_version"):
                                deployment.vllm_version = str(client_dep["vllm_version"])
                        else:
                            deployment.status = "stopped"

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
                            new_dep = Deployment(
                                node_id=node_id,
                                model_name=str(client_dep.get("model_name", "")),
                                port=int(client_dep.get("port", 0)),
                                gpu_memory_fraction=float(
                                    client_dep.get("gpu_memory_fraction", 0.0)
                                ),
                                gpu_ids=client_dep.get("gpu_ids") or [],
                                tensor_parallel_size=client_dep.get(
                                    "tensor_parallel_size"
                                ),
                                vllm_version=client_dep.get("vllm_version") or None,
                                status=str(client_status),
                            )
                            session.add(new_dep)

                await session.commit()
        except Exception:
            pass

        await asyncio.sleep(interval_seconds)
