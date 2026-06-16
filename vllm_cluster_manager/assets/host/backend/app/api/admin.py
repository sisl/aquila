from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment import Deployment
from app.models.deployment_config import DeploymentConfig
from app.models.node import Node
from app.models.node_metric import NodeMetric
from app.services import sync
from app.services.node_state import rogue_container_counts, rogue_process_counts
from app.services.notify import _warned_expiring
from app.ws.manager import manager

router = APIRouter()

PURGE_TARGETS = ("deployments", "nodes", "metrics", "configs")


class PurgeRequest(BaseModel):
    # Omitted/empty = everything (backward compatible with the old full purge).
    targets: list[str] | None = None


def _normalize_targets(targets: list[str] | None) -> set[str]:
    if not targets:
        return set(PURGE_TARGETS)
    invalid = set(targets) - set(PURGE_TARGETS)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown purge target(s): {', '.join(sorted(invalid))}.",
        )
    selected = set(targets)
    # Nodes cannot outlive-delete their children: deployments reference nodes
    # without a cascade, and orphaned metric rows would be unreachable.
    if "nodes" in selected:
        selected.update({"deployments", "metrics"})
    return selected


async def purge_database(
    session: AsyncSession, targets: list[str] | None = None
) -> dict[str, int]:
    """Delete the selected record categories (FK-safe order) and clear the
    matching in-memory caches.

    Running containers on the nodes are untouched: nodes re-register via
    Consul within seconds, and their deployments are re-adopted from the
    container launch manifests with owner/lease/args intact.
    """
    selected = _normalize_targets(targets)
    counts: dict[str, int] = {}
    # Children before parents: deployments and metrics reference nodes.
    plan = (
        ("deployments", Deployment),
        ("metrics", NodeMetric),
        ("nodes", Node),
        ("configs", DeploymentConfig),
    )
    for target, model in plan:
        if target not in selected:
            continue
        result = await session.execute(delete(model))
        counts[model.__tablename__] = result.rowcount or 0
    await session.commit()

    # Drop in-memory state keyed by the now-deleted ids.
    if "deployments" in selected:
        sync._usage_last_seen.clear()
        sync.live_usage.clear()
        sync.pull_progress.clear()
        sync._deployment_fail_counts.clear()
        _warned_expiring.clear()
    if "nodes" in selected:
        rogue_container_counts.clear()
        rogue_process_counts.clear()
        sync._node_fail_counts.clear()
    return counts


@router.post("/purge")
async def purge(
    payload: PurgeRequest | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    targets = payload.targets if payload else None
    counts = await purge_database(session, targets)
    selected = _normalize_targets(targets)
    if selected & {"deployments", "configs"}:
        await manager.broadcast({"type": "deployments_changed"})
    if selected & {"nodes", "metrics"}:
        await manager.broadcast({"type": "nodes_changed"})
    return {"purged": counts}
