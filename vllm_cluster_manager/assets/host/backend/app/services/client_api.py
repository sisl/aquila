import httpx
from fastapi import HTTPException

from app.core.config import settings

# One shared client for all satellite calls: connection pooling for the
# 5s status loop plus transport-level retries on transient connect errors.
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=2),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            timeout=httpx.Timeout(10.0),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _satellite_url(node_ip: str, node_port: int | None, path: str) -> str:
    port = node_port if node_port is not None else settings.satellite_port
    return f"http://{node_ip}:{port}{path}"


def _raise_for_client_error(response: httpx.Response) -> None:
    """Propagate the satellite's error detail with its status code."""
    detail = ""
    try:
        payload = response.json()
        detail = str(payload.get("detail") or "")
    except Exception:
        detail = response.text.strip()
    raise HTTPException(
        status_code=response.status_code,
        detail=detail or f"Client rejected request with status {response.status_code}.",
    )


def _unreachable(url: str, exc: httpx.RequestError) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail=f"Failed to reach client at {url}: {exc}",
    )


async def start_model(
    node_ip: str,
    node_port: int | None,
    model_name: str,
    port: int,
    gpu_memory_fraction: float,
    gpu_ids: list[int] | None = None,
    tensor_parallel_size: int | None = None,
    extra_args: list[str] | None = None,
    env_vars: list[dict[str, str]] | None = None,
    vllm_version: str | None = None,
    extra_packages: list[str] | None = None,
    engine_args: dict[str, object] | None = None,
    lora_modules: list[dict[str, str]] | None = None,
    max_failed_restarts: int | None = None,
    skip_resource_check: bool = False,
    owner: str | None = None,
    duration_seconds: int | None = None,
    expires_at: str | None = None,
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/deployments/start")
    try:
        # Image pulls + large model downloads can take a long time.
        response = await get_client().post(
            url,
            timeout=1800.0,
            json={
                "model_name": model_name,
                "port": port,
                "gpu_memory_fraction": gpu_memory_fraction,
                "gpu_ids": gpu_ids,
                "tensor_parallel_size": tensor_parallel_size,
                "extra_args": extra_args,
                "env_vars": env_vars,
                "vllm_version": vllm_version,
                "extra_packages": extra_packages,
                "engine_args": engine_args,
                "lora_modules": lora_modules,
                "max_failed_restarts": max_failed_restarts,
                "skip_resource_check": skip_resource_check,
                # Launch metadata the client persists on the container so the
                # host can re-adopt deployments after losing its database.
                "owner": owner,
                "duration_seconds": duration_seconds,
                "expires_at": expires_at,
            },
        )
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc

    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def stop_model(node_ip: str, node_port: int | None, key: str) -> None:
    url = _satellite_url(node_ip, node_port, "/deployments/stop")
    response = await get_client().post(url, json={"key": key}, timeout=10.0)
    response.raise_for_status()


async def get_statuses(node_ip: str, node_port: int | None) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/deployments/status")
    response = await get_client().get(url, timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    return payload.get("deployments", [])


async def get_logs(
    node_ip: str, node_port: int | None, key: str, tail: int = 200
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/deployments/logs")
    response = await get_client().get(url, params={"key": key, "tail": tail}, timeout=10.0)
    if response.status_code == 404:
        return {"key": key, "lines": []}
    response.raise_for_status()
    return response.json()


async def stream_log_download(node_ip: str, node_port: int | None, key: str):
    """Yield the satellite's full log file for a deployment, chunk by chunk.

    Returned as an async generator so the host never buffers the file; the
    request stays open for the duration of the transfer.
    """
    url = _satellite_url(node_ip, node_port, "/deployments/logs/download")
    try:
        async with get_client().stream(
            "GET", url, params={"key": key}, timeout=TRANSFER_TIMEOUT
        ) as response:
            if not response.is_success:
                await response.aread()
                _raise_for_client_error(response)
            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc


async def get_metrics(node_ip: str, node_port: int | None) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/metrics")
    response = await get_client().get(url, timeout=15.0)
    response.raise_for_status()
    return response.json()


async def check_port(node_ip: str, node_port: int | None, port: int) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/ports/check")
    try:
        response = await get_client().get(url, params={"port": port}, timeout=5.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc

    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def upload_package(
    node_ip: str, node_port: int | None, filename: str, content: bytes
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/packages/upload")
    try:
        response = await get_client().post(
            url, files={"file": (filename, content)}, timeout=120.0
        )
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc

    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def get_packages(node_ip: str, node_port: int | None) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/packages")
    response = await get_client().get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()


async def list_containers(node_ip: str, node_port: int | None) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/containers")
    try:
        response = await get_client().get(url, timeout=15.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json().get("containers", [])
    _raise_for_client_error(response)


async def stop_container(
    node_ip: str, node_port: int | None, container_id: str
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, f"/containers/{container_id}/stop")
    try:
        response = await get_client().post(url, timeout=60.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def list_images(node_ip: str, node_port: int | None) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/images")
    try:
        response = await get_client().get(url, timeout=15.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json().get("images", [])
    _raise_for_client_error(response)


async def delete_image(
    node_ip: str, node_port: int | None, image_id: str
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, f"/images/{image_id}")
    try:
        response = await get_client().delete(url, timeout=60.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def prune_images(node_ip: str, node_port: int | None) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/images/prune")
    try:
        response = await get_client().post(url, timeout=120.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def list_model_cache(node_ip: str, node_port: int | None) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/models/cache")
    try:
        response = await get_client().get(url, timeout=30.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json().get("models", [])
    _raise_for_client_error(response)


async def delete_model_cache(
    node_ip: str, node_port: int | None, name: str
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, f"/models/cache/{name}")
    try:
        response = await get_client().delete(url, timeout=120.0)
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


# ---------------------------------------------------------------------------
# Managed local models (streamed uploads / URL pulls)
# ---------------------------------------------------------------------------

# Model transfers run for as long as the network needs; only connecting is
# bounded.
TRANSFER_TIMEOUT = httpx.Timeout(None, connect=10.0)


async def _relay_json(
    method: str, url: str, json_body: dict | None = None, timeout: float = 30.0
) -> dict[str, object]:
    try:
        response = await get_client().request(
            method, url, json=json_body, timeout=timeout
        )
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def begin_local_model_upload(
    node_ip: str, node_port: int | None, payload: dict
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/local-models/upload/begin")
    return await _relay_json("POST", url, payload)


async def upload_local_model_file(
    node_ip: str,
    node_port: int | None,
    session_id: str,
    rel_path: str,
    stream,
    content_length: str | None = None,
) -> dict[str, object]:
    """Relay a raw file body to the satellite without buffering it."""
    url = _satellite_url(node_ip, node_port, f"/local-models/upload/{session_id}/file")
    headers = {"content-type": "application/octet-stream"}
    if content_length:
        headers["content-length"] = content_length
    try:
        response = await get_client().put(
            url,
            params={"path": rel_path},
            content=stream,
            headers=headers,
            timeout=TRANSFER_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def finish_local_model_upload(
    node_ip: str, node_port: int | None, session_id: str, payload: dict | None = None
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, f"/local-models/upload/{session_id}/finish")
    # Finalize can move very large trees; give it time.
    return await _relay_json("POST", url, payload or {}, timeout=600.0)


async def abort_local_model_upload(
    node_ip: str, node_port: int | None, session_id: str
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, f"/local-models/upload/{session_id}/abort")
    return await _relay_json("POST", url, timeout=120.0)


async def upload_local_model_archive(
    node_ip: str,
    node_port: int | None,
    name: str,
    filename: str,
    stream,
    content_length: str | None = None,
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/local-models/archive")
    headers = {"content-type": "application/octet-stream"}
    if content_length:
        headers["content-length"] = content_length
    try:
        response = await get_client().post(
            url,
            params={"name": name, "filename": filename},
            content=stream,
            headers=headers,
            timeout=TRANSFER_TIMEOUT,
        )
    except httpx.RequestError as exc:
        raise _unreachable(url, exc) from exc
    if response.is_success:
        return response.json()
    _raise_for_client_error(response)


async def pull_local_model(
    node_ip: str, node_port: int | None, payload: dict
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/local-models/pull")
    return await _relay_json("POST", url, payload)


async def get_local_model_transfers(
    node_ip: str, node_port: int | None
) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/local-models/transfers")
    result = await _relay_json("GET", url, timeout=15.0)
    return result.get("transfers", [])


async def list_local_models(
    node_ip: str, node_port: int | None
) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/local-models")
    # Sizing large checkpoint trees can take a moment.
    result = await _relay_json("GET", url, timeout=60.0)
    return result.get("models", [])


async def delete_local_model(
    node_ip: str, node_port: int | None, name: str
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, f"/local-models/{name}")
    # rmtree of a 100+ GB checkpoint can take a while.
    return await _relay_json("DELETE", url, timeout=120.0)
