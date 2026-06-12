from fastapi import APIRouter, Depends
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment import Deployment
from app.models.deployment_config import DeploymentConfig
from app.models.node import Node
from app.models.node_metric import NodeMetric
from app.services import sync
from app.services.node_state import rogue_container_counts
from app.services.notify import _warned_expiring
from app.ws.manager import manager

router = APIRouter()


async def purge_database(session: AsyncSession) -> dict[str, int]:
    """Delete all application rows (FK-safe order) and clear in-memory caches.

    Running containers on the nodes are untouched: nodes re-register via
    Consul within seconds, and their deployments are re-adopted from the
    container launch manifests with owner/lease/args intact.
    """
    counts: dict[str, int] = {}
    for model in (Deployment, NodeMetric, Node, DeploymentConfig):
        result = await session.execute(delete(model))
        counts[model.__tablename__] = result.rowcount or 0
    await session.commit()
    # Drop in-memory state keyed by the now-deleted ids.
    sync._usage_last_seen.clear()
    sync.live_usage.clear()
    sync._deployment_fail_counts.clear()
    rogue_container_counts.clear()
    _warned_expiring.clear()
    return counts


@router.post("/purge")
async def purge(session: AsyncSession = Depends(get_session)) -> dict[str, object]:
    counts = await purge_database(session)
    await manager.broadcast({"type": "deployments_changed"})
    await manager.broadcast({"type": "nodes_changed"})
    return {"purged": counts}
