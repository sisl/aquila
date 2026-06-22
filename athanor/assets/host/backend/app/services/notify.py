"""Fire-and-forget webhook notifications (Slack or generic JSON).

Configured via the webhook_url setting (dashboard Settings or WEBHOOK_URL env
default); a no-op when unset. Posting happens in a detached task so
notification latency or failures never touch the sync loops.
"""

import asyncio
import logging

from app.services import runtime_settings
from app.services.client_api import get_client

logger = logging.getLogger(__name__)

# (deployment_id, expires_at iso) pairs already warned about. Keying on the
# timestamp means an extension re-arms the warning; lost on restart (accepted).
_warned_expiring: set[tuple[int, str]] = set()


def _format_message(event: str, message: str, fields: dict[str, object]) -> str:
    parts = [f"[athanor] {message}"]
    detail = fields.get("error")
    if detail:
        parts.append(f"reason: {detail}")
    return " — ".join(parts)


def _payload(event: str, message: str, fields: dict[str, object]) -> dict[str, object]:
    text = _format_message(event, message, fields)
    if "hooks.slack.com" in runtime_settings.get_str("webhook_url"):
        return {"text": text}
    return {"event": event, "message": message, **fields}


async def _post(event: str, message: str, fields: dict[str, object]) -> None:
    try:
        response = await get_client().post(
            runtime_settings.get_str("webhook_url"),
            json=_payload(event, message, fields),
            timeout=10.0,
        )
        if response.status_code >= 400:
            logger.warning(
                "Webhook responded %s for event %s", response.status_code, event
            )
    except Exception as exc:
        logger.warning("Webhook post failed for event %s: %s", event, exc)


def notify(event: str, message: str, fields: dict[str, object] | None = None) -> None:
    """Send a notification without blocking or raising."""
    if not runtime_settings.get_str("webhook_url"):
        return
    try:
        asyncio.get_running_loop().create_task(_post(event, message, fields or {}))
    except RuntimeError:
        # No running loop (e.g. sync test context): skip rather than block.
        logger.debug("notify(%s) skipped: no running event loop", event)
