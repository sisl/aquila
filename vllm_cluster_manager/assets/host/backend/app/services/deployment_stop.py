"""Shared stop path used by manual stop, auto-expiry, and node drain."""

import logging

from app.services.client_api import stop_model
from app.services.deployment_state import set_status

logger = logging.getLogger(__name__)


async def stop_deployment_internal(
    deployment,
    node,
    *,
    final_status: str = "stopped",
    best_effort: bool = False,
) -> None:
    """Stop the container on the client and mark the deployment.

    The client returns immediately (the container teardown runs in the
    background), so the deployment is marked "stopping" first.  The sync
    loop picks up the final "stopped" status from the client once the
    container is actually gone.

    With best_effort=True a failed client call still marks the deployment
    with *final_status* directly (used by expiry/drain so a dead node
    can't keep a deployment alive forever); otherwise the error propagates.
    The caller commits.
    """
    set_status(deployment, "stopping")
    deployment.expires_at = None

    if node is not None:
        key = f"{deployment.model_name}:{deployment.port}"
        try:
            await stop_model(node.ip_address, node.port, key)
        except Exception as exc:
            if not best_effort:
                raise
            logger.warning(
                "Best-effort stop of %s on %s failed: %s", key, node.hostname, exc
            )
            set_status(deployment, final_status)
            return

    if best_effort:
        set_status(deployment, final_status)
