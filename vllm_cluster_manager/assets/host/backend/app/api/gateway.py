"""OpenAI-compatible gateway.

One stable URL (`http://host:8000/v1/...`) that routes requests to the right
vLLM instance by the request's "model" field, so researchers' code never
changes when a model moves nodes. Requests and streams are passed through
unmodified; token accounting comes from the per-deployment vLLM metrics
scrape, not from the gateway.
"""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.models.deployment import Deployment
from app.models.node import Node
from app.services import api_keys, model_names, runtime_settings


def _require_gateway(request: Request) -> None:
    """Settings-controlled kill switch + API-key gate for /v1."""
    if not runtime_settings.get_bool("gateway_enabled"):
        raise HTTPException(
            status_code=503,
            detail=(
                "The OpenAI gateway is disabled by the administrator; "
                "use the node's direct URL instead."
            ),
        )

    if not api_keys.has_keys():
        request.state.allowed_deployment_ids = None
        return

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="API key required. Set Authorization: Bearer <key>.",
        )
    raw = auth.removeprefix("Bearer ").strip()
    if not api_keys.validate(raw):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    request.state.allowed_deployment_ids = api_keys.get_allowed_deployments(raw)


router = APIRouter(dependencies=[Depends(_require_gateway)])

# Separate client from services.client_api: gateway requests need different
# timeout semantics (no read timeout on streams, long reads on generations).
_client: httpx.AsyncClient | None = None


def get_gateway_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(10.0),
        )
    return _client


async def close_gateway_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


def _served_name(deployment) -> str | None:
    value = (deployment.engine_args or {}).get("served_model_name")
    return value if isinstance(value, str) and value else None


def _lora_names(deployment) -> list[str]:
    return model_names.lora_names(deployment)


def _model_aliases(deployment) -> list[str]:
    """All names this deployment answers to, primary alias first."""
    aliases = []
    served = _served_name(deployment)
    if served:
        aliases.append(served)
    if deployment.model_name not in aliases:
        aliases.append(deployment.model_name)
    aliases.extend(n for n in _lora_names(deployment) if n not in aliases)
    return aliases


def _choose(deployments, model: str):
    """Resolve *model* to one deployment.

    Precedence: explicit served_model_name, then model_name, then LoRA
    adapter names. Served names and LoRA names are kept unique at deploy
    time, so those tiers never collide; a tier with more than one match is
    only reachable through a raw model_name shared by several replicas, in
    which case the request is ambiguous and the caller must name a specific
    served model. Returns one of:
        ("ok", deployment)         exactly one match
        ("ambiguous", [names...])  several matches — distinct served names
        ("none", None)             no match
    """
    tiers = (
        [d for d in deployments if _served_name(d) == model],
        [d for d in deployments if d.model_name == model],
        [d for d in deployments if model in _lora_names(d)],
    )
    for tier in tiers:
        if not tier:
            continue
        if len(tier) == 1:
            return "ok", tier[0]
        names = sorted({model_names.effective_served_name(d) for d in tier})
        return "ambiguous", names
    return "none", None


def _openai_error(
    status_code: int, message: str, err_type: str, code: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"message": message, "type": err_type, "param": None, "code": code}
        },
    )


# Statuses the gateway will route to. Paused deployments are reachable: the
# node-side proxy transparently wakes them on the first inference request.
_ROUTABLE_STATUSES = ("running", "paused_ram")


async def _resolve(
    session: AsyncSession, model: str, allowed_ids: set[int] | None = None
):
    """Return (deployment, node) for *model*, or a JSONResponse error."""
    result = await session.execute(select(Deployment))
    deployments = list(result.scalars().all())

    routable = [d for d in deployments if d.status in _ROUTABLE_STATUSES]
    if allowed_ids is not None:
        routable = [d for d in routable if d.id in allowed_ids]

    outcome, value = _choose(routable, model)
    if outcome == "ambiguous":
        return _openai_error(
            400,
            f"Model '{model}' maps to several running deployments; "
            f"request a specific served model name: {', '.join(value)}.",
            "invalid_request_error",
            "model_ambiguous",
        )
    if outcome == "none":
        pending = [d for d in deployments if d.status in ("starting", "loading")]
        if allowed_ids is not None:
            pending = [d for d in pending if d.id in allowed_ids]
        pending_outcome, pending_match = _choose(pending, model)
        if pending_outcome == "ok":
            return _openai_error(
                503,
                f"Model '{model}' is still loading (deployment {pending_match.id}); retry shortly.",
                "upstream_error",
                "model_loading",
            )
        if pending_outcome == "ambiguous":
            return _openai_error(
                503,
                f"Model '{model}' is still loading; retry shortly.",
                "upstream_error",
                "model_loading",
            )
        available = sorted(
            {
                alias
                for d in routable
                for alias in _model_aliases(d)
            }
        )
        return _openai_error(
            404,
            f"Model '{model}' is not being served. "
            f"Available models: {', '.join(available) if available else 'none'}.",
            "invalid_request_error",
            "model_not_found",
        )

    match = value
    node = await session.get(Node, match.node_id)
    if node is None:
        return _openai_error(
            502,
            f"Deployment {match.id} for '{model}' has no node record.",
            "upstream_error",
            "node_missing",
        )
    return match, node


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


async def _proxy(request: Request, endpoint_path: str, session: AsyncSession) -> Response:
    try:
        body = await request.json()
    except Exception:
        return _openai_error(
            400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json"
        )
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return _openai_error(
            400, "Missing required 'model' field.", "invalid_request_error", "missing_model"
        )

    allowed = getattr(request.state, "allowed_deployment_ids", None)
    resolved = await _resolve(session, model, allowed_ids=allowed)
    if isinstance(resolved, JSONResponse):
        return resolved
    deployment, node = resolved

    url = f"http://{node.ip_address}:{deployment.port}/v1/{endpoint_path}"
    client = get_gateway_client()
    stream = bool(body.get("stream"))

    if not stream:
        try:
            upstream = await client.post(
                url,
                json=body,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(runtime_settings.get_int("gateway_timeout_seconds")),
                    write=30.0,
                    pool=10.0,
                ),
            )
        except httpx.RequestError as exc:
            return _openai_error(
                502,
                f"Deployment for '{model}' on {node.hostname}:{deployment.port} "
                f"is unreachable: {exc}",
                "upstream_error",
                "deployment_unreachable",
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    req = client.build_request("POST", url, json=body, timeout=_STREAM_TIMEOUT)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        return _openai_error(
            502,
            f"Deployment for '{model}' on {node.hostname}:{deployment.port} "
            f"is unreachable: {exc}",
            "upstream_error",
            "deployment_unreachable",
        )

    if upstream.status_code >= 400:
        await upstream.aread()
        content = upstream.content
        media_type = upstream.headers.get("content-type", "application/json")
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code, media_type=media_type)

    async def _relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            # Runs on normal end AND client disconnect.
            await upstream.aclose()

    return StreamingResponse(
        _relay(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


@router.post("/chat/completions")
async def chat_completions(
    request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    return await _proxy(request, "chat/completions", session)


@router.post("/completions")
async def completions(
    request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    return await _proxy(request, "completions", session)


@router.post("/embeddings")
async def embeddings(
    request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    return await _proxy(request, "embeddings", session)


@router.get("/models")
async def list_models(
    request: Request, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    result = await session.execute(
        select(Deployment).where(Deployment.status.in_(_ROUTABLE_STATUSES))
    )
    allowed = getattr(request.state, "allowed_deployment_ids", None)
    data: list[dict[str, object]] = []
    for deployment in result.scalars().all():
        if allowed is not None and deployment.id not in allowed:
            continue
        served = _served_name(deployment) or deployment.model_name
        created = (
            int(deployment.created_at.timestamp()) if deployment.created_at else 0
        )
        owned_by = deployment.owner or "unknown"
        data.append(
            {
                "id": served,
                "object": "model",
                "created": created,
                "owned_by": owned_by,
                "root": deployment.model_name,
                "parent": None,
            }
        )
        for lora_name in _lora_names(deployment):
            data.append(
                {
                    "id": lora_name,
                    "object": "model",
                    "created": created,
                    "owned_by": owned_by,
                    "root": lora_name,
                    "parent": served,
                }
            )
    return {"object": "list", "data": data}
