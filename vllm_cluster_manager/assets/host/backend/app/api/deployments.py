from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment import ACTIVE_STATUSES, Deployment
from app.models.node import Node
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.schemas.deployment import (
    DeploymentCreate,
    DeploymentExtend,
    DeploymentFromManifest,
    DeploymentPause,
    DeploymentPin,
    DeploymentPlanRead,
    DeploymentPlanRequest,
    DeploymentRead,
    DeploymentStart,
    DeploymentRestart,
    OffloadItem,
)
from app.services import model_names
from app.services import runtime_settings
from app.services import sync as sync_service
from app.services.client_api import (
    get_logs,
    pause_model,
    pin_model,
    plan_deployment,
    resume_model,
    start_model,
    stream_log_download,
)
from app.services.deployment_state import set_status
from app.services.deployment_stop import stop_deployment_internal
from app.ws.manager import manager

router = APIRouter()


async def _broadcast_change(deployment_id: int) -> None:
    await manager.broadcast({"type": "deployments_changed", "ids": [deployment_id]})


def _resolve_runtime(node: Node) -> str:
    """Which container runtime a new deployment on *node* should use.

    Precedence: the node's explicit override (when actually available) →
    the global preferred runtime → whichever single runtime exists.
    """
    available = node.available_runtimes or []
    if node.container_runtime and node.container_runtime in available:
        return node.container_runtime
    if not available:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{node.hostname} has no container runtime available — install "
                "Docker or enable the Podman socket on the node."
            ),
        )
    preferred = runtime_settings.get_str("preferred_container_runtime")
    return preferred if preferred in available else available[0]


def _launch_expires_at(duration_seconds: int | None) -> str | None:
    """Best-effort lease end stamped into the container's launch manifest.

    The authoritative expires_at is set when the deployment first reports
    running; this launch-anchored value only matters if the host database is
    lost and the deployment is re-adopted from the container label.
    """
    if duration_seconds is None:
        return None
    return (
        datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
    ).isoformat()


async def _check_port_conflict(
    session: AsyncSession, node: Node, port: int, exclude_id: int | None = None
) -> None:
    """Friendly pre-check naming the conflicting deployment.

    The partial unique index uq_active_node_port is the authoritative guard;
    this exists only to produce a better error message in the common case.
    """
    query = select(Deployment).where(
        Deployment.node_id == node.id,
        Deployment.port == port,
        Deployment.status.in_(ACTIVE_STATUSES),
    )
    if exclude_id is not None:
        query = query.where(Deployment.id != exclude_id)
    existing = await session.execute(query)
    conflict = existing.scalar_one_or_none()
    if conflict:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Port {port} is already in use on {node.hostname} "
                f"by deployment {conflict.id} ({conflict.model_name})."
            ),
        )


def _port_conflict_409(node: Node, port: int) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"Port {port} was just claimed by another deployment on {node.hostname}.",
    )


async def _active_alias_owners(
    session: AsyncSession, exclude_id: int | None = None
) -> list[Deployment]:
    """Active deployments whose names occupy the global alias namespace."""
    query = select(Deployment).where(Deployment.status.in_(ACTIVE_STATUSES))
    if exclude_id is not None:
        query = query.where(Deployment.id != exclude_id)
    result = await session.execute(query)
    return list(result.scalars().all())


def _suggest_served_name(name: str, taken: set[str]) -> str:
    """First free '<name>-2', '-3', … (or *name* itself when available)."""
    if name not in taken:
        return name
    i = 2
    while f"{name}-{i}" in taken:
        i += 1
    return f"{name}-{i}"


async def _check_served_name_conflict(
    session: AsyncSession,
    *,
    model_name: str,
    engine_args: dict | None,
    lora_modules: list | None,
    exclude_id: int | None = None,
) -> None:
    """Reject a launch whose served name / LoRA names clash with an active one.

    The served name (explicit ``served_model_name`` or the model name) and any
    LoRA adapter names are the gateway's routing keys; keeping them unique is
    what lets the gateway resolve a request to exactly one deployment instead
    of guessing between replicas.
    """
    prospective = SimpleNamespace(
        model_name=model_name,
        engine_args=engine_args or {},
        lora_modules=lora_modules or [],
    )
    wanted = model_names.primary_aliases(prospective)
    for owner in await _active_alias_owners(session, exclude_id):
        clash = wanted & model_names.primary_aliases(owner)
        if clash:
            name = sorted(clash)[0]
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Served name '{name}' is already in use by deployment "
                    f"{owner.id} ({owner.model_name}). "
                    "Choose a different served model name."
                ),
            )


@router.get("/", response_model=list[DeploymentRead])
async def list_deployments(session: AsyncSession = Depends(get_session)) -> list[DeploymentRead]:
    result = await session.execute(select(Deployment).order_by(Deployment.id.desc()))
    reads: list[DeploymentRead] = []
    for deployment in result.scalars().all():
        read = DeploymentRead.model_validate(deployment)
        # Attach live scrape metrics (tokens/s, queue depth) to running rows.
        # prompt_tps/generation_tps come from the row via model_validate (the
        # persisted last-known averages — kept after a deployment stops);
        # fresher live values override them only when actually present.
        if deployment.status == "running":
            live = sync_service.live_usage.get(deployment.id, {})
            for field in (
                "prompt_tps",
                "generation_tps",
                "prompt_throughput",
                "generation_throughput",
                "requests_running",
                "requests_waiting",
            ):
                if field in live:
                    setattr(read, field, live[field])
        # Attach image-pull progress to rows still starting up.
        elif deployment.status in ("starting", "loading"):
            progress = sync_service.pull_progress.get(deployment.id, {})
            read.pull_percent = progress.get("percent")
            read.pull_downloaded_mb = progress.get("downloaded_mb")
            read.pull_total_mb = progress.get("total_mb")
        reads.append(read)
    return reads


@router.get("/served-name/check")
async def check_served_name(
    name: str,
    exclude_id: int | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Whether a served model name is free, with a suggested alternative.

    Powers the live deploy-form prompt; the launch path re-checks under the
    real request, so this is advisory (a name can be claimed in the gap).
    """
    name = name.strip()
    if not name:
        return {"available": True}
    taken: dict[str, Deployment] = {}
    for owner in await _active_alias_owners(session, exclude_id):
        for alias in model_names.primary_aliases(owner):
            taken.setdefault(alias, owner)
    conflict = taken.get(name)
    if conflict is None:
        return {"available": True}
    return {
        "available": False,
        "conflict_id": conflict.id,
        "conflict_model": conflict.model_name,
        "suggestion": _suggest_served_name(name, set(taken)),
    }


@router.post("/", response_model=DeploymentRead)
async def create_deployment(
    payload: DeploymentCreate, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    deployment = Deployment(**payload.model_dump())
    session.add(deployment)
    await session.commit()
    await session.refresh(deployment)
    return deployment


async def _launch(payload: DeploymentStart, session: AsyncSession) -> Deployment:
    """Shared launch path for /start and /from-manifest."""
    node = await session.get(Node, payload.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    if node.maintenance:
        raise HTTPException(
            status_code=409,
            detail=f"Node {node.hostname} is in maintenance mode; no new deployments.",
        )

    await _check_port_conflict(session, node, payload.port)
    await _check_served_name_conflict(
        session,
        model_name=payload.model_name,
        engine_args=payload.engine_args,
        lora_modules=payload.lora_modules,
    )
    runtime = _resolve_runtime(node)

    # Create the row before the (potentially very long) client call so a
    # backend crash mid-start leaves a visible record instead of an
    # untracked container, and so the unique index reserves the port.
    payload_data = payload.model_dump(exclude={"status", "skip_resource_check"})
    payload_data["container_runtime"] = runtime
    deployment = Deployment(**payload_data)
    set_status(deployment, "starting")
    session.add(deployment)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise _port_conflict_409(node, payload.port)
    await session.refresh(deployment)

    try:
        client_result = await start_model(
            node.ip_address,
            node.port,
            payload.model_name,
            payload.port,
            payload.gpu_memory_fraction,
            payload.gpu_ids,
            payload.tensor_parallel_size,
            payload.extra_args,
            payload.env_vars,
            payload.vllm_version,
            payload.extra_packages,
            engine_args=payload.engine_args,
            lora_modules=payload.lora_modules,
            max_failed_restarts=payload.max_failed_restarts,
            skip_resource_check=payload.skip_resource_check,
            owner=payload.owner,
            duration_seconds=payload.duration_seconds,
            expires_at=_launch_expires_at(payload.duration_seconds),
            container_runtime=runtime,
            warm_offload=bool(node.warm_offload_enabled),
            pinned=bool(payload.pinned),
        )
    except HTTPException as exc:
        set_status(deployment, "error", error=str(exc.detail))
        await session.commit()
        await _broadcast_change(deployment.id)
        raise
    except Exception as exc:
        set_status(deployment, "error", error=str(exc))
        await session.commit()
        await _broadcast_change(deployment.id)
        raise

    # Use the resolved version from the client (e.g. latest stable when left blank)
    resolved_version = client_result.get("vllm_version") if client_result else None
    if resolved_version:
        deployment.vllm_version = str(resolved_version)
    set_status(deployment, "loading")
    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


@router.post("/start", response_model=DeploymentRead)
async def start_deployment(
    payload: DeploymentStart, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    return await _launch(payload, session)


@router.post("/plan", response_model=DeploymentPlanRead)
async def plan_deployment_preview(
    payload: DeploymentPlanRequest, session: AsyncSession = Depends(get_session)
) -> DeploymentPlanRead:
    """Preview which warm models a deploy would auto-offload, before committing.

    Read-only. Returns ``warm_enabled=False`` when the node has warm-offload off
    (the deploy path then uses the plain resource check, offloading nothing).
    """
    node = await session.get(Node, payload.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    if not node.warm_offload_enabled:
        return DeploymentPlanRead(fits=True, warm_enabled=False)
    try:
        result = await plan_deployment(
            node.ip_address,
            node.port,
            payload.model_name,
            payload.port,
            payload.gpu_memory_fraction,
            payload.gpu_ids,
        )
    except HTTPException:
        # Agent unreachable or errored — don't block the UI; deploy will still
        # try (and the deploy path itself enforces capacity authoritatively).
        return DeploymentPlanRead(fits=True, warm_enabled=True)
    result = result or {}

    # Map the agent's deployment keys ("model:port") back to host rows so the UI
    # shows friendly names and can link the affected deployments.
    rows = (
        (
            await session.execute(
                select(Deployment).where(
                    Deployment.node_id == node.id,
                    Deployment.status.in_(ACTIVE_STATUSES),
                )
            )
        )
        .scalars()
        .all()
    )
    by_key = {_deployment_key(d): d for d in rows}
    would_offload: list[OffloadItem] = []
    for step in result.get("plan") or []:
        row = by_key.get(step.get("key"))
        would_offload.append(
            OffloadItem(
                deployment_id=row.id if row else None,
                model_name=(row.model_name if row else step.get("model_name"))
                or step.get("key", ""),
                tier=step.get("tier", "disk"),
            )
        )
    return DeploymentPlanRead(
        fits=bool(result.get("fits", True)),
        warm_enabled=bool(result.get("warm_enabled", True)),
        would_offload=would_offload,
        blocked_reason=result.get("reason"),
    )


@router.post("/stop/{deployment_id}", response_model=DeploymentRead)
async def stop_deployment(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    await stop_deployment_internal(deployment, node)
    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


@router.post("/{deployment_id}/restart", response_model=DeploymentRead)
async def restart_deployment(
    deployment_id: int,
    payload: DeploymentRestart,
    session: AsyncSession = Depends(get_session),
) -> DeploymentRead:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.status not in ("stopped", "expired", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Deployment is '{deployment.status}'; only stopped/expired/error can be restarted.",
        )

    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    if node.maintenance:
        raise HTTPException(
            status_code=409,
            detail=f"Node {node.hostname} is in maintenance mode; no new deployments.",
        )

    await _check_port_conflict(session, node, deployment.port, exclude_id=deployment.id)
    await _check_served_name_conflict(
        session,
        model_name=deployment.model_name,
        engine_args=deployment.engine_args,
        lora_modules=deployment.lora_modules,
        exclude_id=deployment.id,
    )
    # Keep the original runtime when it is still available; re-resolve when
    # the node's runtimes changed underneath the stopped deployment.
    if deployment.container_runtime in (node.available_runtimes or []):
        runtime = deployment.container_runtime
    else:
        runtime = _resolve_runtime(node)
        deployment.container_runtime = runtime

    # Claim the port (and surface the attempt) before the long client call.
    deployment.owner = payload.owner
    deployment.duration_seconds = payload.duration_seconds
    deployment.expires_at = None  # countdown restarts when serving resumes
    set_status(deployment, "starting")
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise _port_conflict_409(node, deployment.port)

    try:
        client_result = await start_model(
            node.ip_address,
            node.port,
            deployment.model_name,
            deployment.port,
            deployment.gpu_memory_fraction,
            deployment.gpu_ids,
            deployment.tensor_parallel_size,
            deployment.extra_args,
            deployment.env_vars,
            deployment.vllm_version,
            deployment.extra_packages,
            engine_args=deployment.engine_args,
            lora_modules=deployment.lora_modules,
            max_failed_restarts=deployment.max_failed_restarts,
            owner=deployment.owner,
            duration_seconds=deployment.duration_seconds,
            expires_at=_launch_expires_at(deployment.duration_seconds),
            container_runtime=runtime,
            warm_offload=bool(node.warm_offload_enabled),
            pinned=bool(deployment.pinned),
        )
    except HTTPException as exc:
        set_status(deployment, "error", error=str(exc.detail))
        await session.commit()
        await _broadcast_change(deployment.id)
        raise
    except Exception as exc:
        set_status(deployment, "error", error=str(exc))
        await session.commit()
        await _broadcast_change(deployment.id)
        raise

    resolved_version = client_result.get("vllm_version") if client_result else None
    if resolved_version:
        deployment.vllm_version = str(resolved_version)
    set_status(deployment, "loading")
    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


def _deployment_key(deployment: Deployment) -> str:
    return f"{deployment.model_name}:{deployment.port}"


async def _deployment_and_node(
    deployment_id: int, session: AsyncSession
) -> tuple[Deployment, Node]:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return deployment, node


@router.post("/{deployment_id}/pin", response_model=DeploymentRead)
async def pin_deployment(
    deployment_id: int,
    payload: DeploymentPin,
    session: AsyncSession = Depends(get_session),
) -> DeploymentRead:
    deployment, node = await _deployment_and_node(deployment_id, session)
    deployment.pinned = payload.pinned
    # Best-effort push to the live agent; the sync loop re-pushes pins anyway,
    # so an unreachable node still converges once it comes back.
    try:
        await pin_model(node.ip_address, node.port, _deployment_key(deployment), payload.pinned)
    except HTTPException:
        pass
    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


@router.post("/{deployment_id}/pause", response_model=DeploymentRead)
async def pause_deployment(
    deployment_id: int,
    payload: DeploymentPause,
    session: AsyncSession = Depends(get_session),
) -> DeploymentRead:
    deployment, node = await _deployment_and_node(deployment_id, session)
    if deployment.status not in ("running", "paused_ram", "paused_disk"):
        raise HTTPException(
            status_code=409,
            detail=f"Deployment is '{deployment.status}'; only a running deployment can be paused.",
        )
    result = await pause_model(
        node.ip_address, node.port, _deployment_key(deployment), payload.tier
    )
    tier = (result or {}).get("tier")
    set_status(deployment, "paused_disk" if tier == "disk" else "paused_ram")
    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


@router.post("/{deployment_id}/resume", response_model=DeploymentRead)
async def resume_deployment(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    deployment, node = await _deployment_and_node(deployment_id, session)
    if deployment.status == "running":
        return deployment
    if deployment.status not in ("paused_ram", "paused_disk"):
        raise HTTPException(
            status_code=409,
            detail=f"Deployment is '{deployment.status}'; only a paused deployment can be resumed.",
        )
    await resume_model(node.ip_address, node.port, _deployment_key(deployment))
    set_status(deployment, "running")
    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


@router.delete("/{deployment_id}")
async def delete_deployment(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> dict[str, str]:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    await session.delete(deployment)
    await session.commit()
    await _broadcast_change(deployment_id)
    return {"status": "deleted"}


@router.post("/{deployment_id}/extend", response_model=DeploymentRead)
async def extend_deployment(
    deployment_id: int,
    payload: DeploymentExtend,
    session: AsyncSession = Depends(get_session),
) -> DeploymentRead:
    """Push the serve deadline forward without restarting the model.

    Expiry is enforced host-side only, so no client call is involved.
    """
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if deployment.status not in ("running", "loading", "starting"):
        raise HTTPException(
            status_code=409,
            detail=f"Deployment is '{deployment.status}'; only an active deployment can be extended.",
        )

    if payload.infinite:
        # Drop the deadline entirely: serve until explicitly stopped.
        deployment.expires_at = None
        deployment.duration_seconds = None
        await session.commit()
        await session.refresh(deployment)
        await _broadcast_change(deployment.id)
        return deployment

    extra_seconds = int((payload.hours or 0) * 3600)
    if deployment.expires_at is not None:
        expires_at = deployment.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        deployment.expires_at = expires_at + timedelta(seconds=extra_seconds)
        deployment.duration_seconds = (deployment.duration_seconds or 0) + extra_seconds
    elif deployment.duration_seconds is not None:
        # Countdown hasn't started yet (still loading): lengthen the duration.
        deployment.duration_seconds += extra_seconds
    else:
        raise HTTPException(
            status_code=409, detail="Deployment has no expiry to extend."
        )

    await session.commit()
    await session.refresh(deployment)
    await _broadcast_change(deployment.id)
    return deployment


@router.get("/{deployment_id}/manifest")
async def deployment_manifest(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    """Self-contained reproducibility manifest for a deployment.

    Env var VALUES are deliberately omitted (keys only). image_digest is
    informational: redeploys pin vllm_version (tag) + HF revision + seed.
    """
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    node = await session.get(Node, deployment.node_id)
    engine_args = deployment.engine_args or {}
    gpu_ids = deployment.gpu_ids or []

    gpu_names: list[str] = []
    if node and node.gpu_usage:
        by_index = {g.get("index"): g for g in node.gpu_usage if isinstance(g, dict)}
        target = gpu_ids or sorted(i for i in by_index if isinstance(i, int))
        gpu_names = [
            str(by_index[i].get("name"))
            for i in target
            if i in by_index and by_index[i].get("name")
        ]

    return {
        "manifest_version": 1,
        "deployment_id": deployment.id,
        "model": deployment.model_name,
        "served_model_name": engine_args.get("served_model_name"),
        "revision": engine_args.get("revision"),
        "seed": engine_args.get("seed"),
        "vllm_version": deployment.vllm_version,
        "image_digest": deployment.image_digest,
        "container_runtime": deployment.container_runtime,
        "engine_args": engine_args,
        "extra_args": deployment.extra_args or [],
        "extra_packages": deployment.extra_packages or [],
        "lora_modules": getattr(deployment, "lora_modules", None) or [],
        "env_var_keys": [
            pair.get("key")
            for pair in (deployment.env_vars or [])
            if isinstance(pair, dict) and pair.get("key")
        ],
        "gpu": {
            "ids": gpu_ids,
            "memory_fraction": deployment.gpu_memory_fraction,
            "tensor_parallel_size": deployment.tensor_parallel_size,
            "names": gpu_names,
        },
        "node": {"hostname": node.hostname} if node else None,
        "owner": deployment.owner,
        "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
        "duration_seconds": deployment.duration_seconds,
        "status": deployment.status,
    }


@router.post("/from-manifest", response_model=DeploymentRead)
async def deploy_from_manifest(
    payload: DeploymentFromManifest, session: AsyncSession = Depends(get_session)
) -> DeploymentRead:
    """Launch a deployment from an exported manifest.

    The manifest pins the model identity (model, revision, seed, engine args,
    vLLM version); placement (node, port, duration, env values) comes from
    this request.
    """
    manifest = payload.manifest
    if manifest.get("manifest_version") != 1:
        raise HTTPException(
            status_code=400,
            detail="Unsupported manifest_version (expected 1).",
        )
    model = manifest.get("model")
    if not isinstance(model, str) or not model:
        raise HTTPException(status_code=400, detail="Manifest is missing 'model'.")

    gpu = manifest.get("gpu") if isinstance(manifest.get("gpu"), dict) else {}
    fraction = gpu.get("memory_fraction")
    if not isinstance(fraction, (int, float)):
        raise HTTPException(
            status_code=400, detail="Manifest is missing gpu.memory_fraction."
        )

    start = DeploymentStart(
        node_id=payload.node_id,
        model_name=model,
        port=payload.port,
        gpu_memory_fraction=float(fraction),
        gpu_ids=gpu.get("ids") or None,
        tensor_parallel_size=gpu.get("tensor_parallel_size"),
        extra_args=manifest.get("extra_args") or None,
        env_vars=payload.env_vars,
        vllm_version=manifest.get("vllm_version") or None,
        extra_packages=manifest.get("extra_packages") or None,
        engine_args=manifest.get("engine_args") or None,
        lora_modules=manifest.get("lora_modules") or None,
        owner=payload.owner,
        duration_seconds=payload.duration_seconds,
        skip_resource_check=payload.skip_resource_check,
    )
    return await _launch(start, session)


@router.get("/{deployment_id}/logs")
async def deployment_logs(
    deployment_id: int, tail: int = 200, session: AsyncSession = Depends(get_session)
) -> dict[str, object]:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    key = f"{deployment.model_name}:{deployment.port}"
    return await get_logs(node.ip_address, node.port, key, tail=tail)


@router.get("/{deployment_id}/logs/download")
async def download_deployment_logs(
    deployment_id: int, session: AsyncSession = Depends(get_session)
) -> StreamingResponse:
    """Stream the full persisted log file for the deployment's current run."""
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    node = await session.get(Node, deployment.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    key = f"{deployment.model_name}:{deployment.port}"
    stream = stream_log_download(node.ip_address, node.port, key)
    # Pull the first chunk eagerly so satellite errors (e.g. no log file yet)
    # surface as proper HTTP errors instead of a broken download.
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        first = b""

    async def body():
        if first:
            yield first
        async for chunk in stream:
            yield chunk

    filename = f"{deployment.model_name.replace('/', '--')}-{deployment.port}.log"
    return StreamingResponse(
        body(),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
