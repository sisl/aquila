"""Single place to change a deployment's status.

Keeps status_changed_at accurate (the stuck-start watchdog depends on it)
and pairs every status with its failure reason.
"""

from datetime import datetime, timezone

_HEALTHY_STATUSES = ("running", "loading", "starting", "stopped")


def set_status(deployment, status: str, error: str | None = None) -> None:
    """Assign a new status, stamping status_changed_at on transitions.

    Passing ``error`` records the failure reason; returning to a healthy
    status clears any stale reason.
    """
    if deployment.status != status:
        deployment.status_changed_at = datetime.now(timezone.utc)
    deployment.status = status
    if error is not None:
        deployment.last_error = error
    elif status in _HEALTHY_STATUSES:
        deployment.last_error = None
