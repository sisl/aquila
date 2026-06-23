import json
import os
import time
import logging
from urllib.request import Request, urlopen

from fastapi import APIRouter

_aquila_version = os.environ.get("AQUILA_VERSION", "unknown")

router = APIRouter()

logger = logging.getLogger(__name__)

_latest_vllm_version: str = ""
_latest_vllm_version_fetched_at: float = 0.0
_LATEST_VERSION_TTL: float = 3600.0


@router.get("/health")
async def health() -> dict[str, str]:
    """Backend health check."""
    return {"status": "ok", "aquila_version": _aquila_version}


@router.get("/vllm-version")
async def latest_vllm_version() -> dict[str, str]:
    """Return the latest stable vLLM release tag from GitHub."""
    global _latest_vllm_version, _latest_vllm_version_fetched_at
    now = time.monotonic()
    if _latest_vllm_version and (now - _latest_vllm_version_fetched_at) < _LATEST_VERSION_TTL:
        return {"version": _latest_vllm_version}
    try:
        req = Request(
            "https://api.github.com/repos/vllm-project/vllm/releases/latest",
            headers={"User-Agent": "aquila"},
        )
        data = json.loads(urlopen(req, timeout=10).read().decode("utf-8"))
        tag = data.get("tag_name", "")
        version = tag.lstrip("v")
        if version:
            _latest_vllm_version = version
            _latest_vllm_version_fetched_at = now
    except Exception as exc:
        logger.warning("Failed to fetch latest vLLM version: %s", exc)
        if _latest_vllm_version:
            _latest_vllm_version_fetched_at = now
    return {"version": _latest_vllm_version}
