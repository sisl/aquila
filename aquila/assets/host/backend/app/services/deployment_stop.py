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

    Manual stop (best_effort=False): the deployment is marked "stopping"
    and the sync loop picks up "stopped" from the client once the
    container is gone.

    Best-effort stop (expiry, drain): the deployment is stamped with
    *final_status* immediately so a concurrent sync tick cannot overwrite
    the terminal state.  The client call is still made but failures are
    logged and swallowed.

    The caller commits.
    """
    deployment.expires_at = None

    if best_effort:
        set_status(deployment, final_status)
    else:
        set_status(deployment, "stopping")

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
