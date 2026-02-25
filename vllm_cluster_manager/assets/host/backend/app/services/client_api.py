import httpx
from fastapi import HTTPException

from app.core.config import settings


def _satellite_url(node_ip: str, node_port: int | None, path: str) -> str:
    port = node_port if node_port is not None else settings.satellite_port
    return f"http://{node_ip}:{port}{path}"


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
    pip_packages: list[str] | None = None,
) -> None:
    url = _satellite_url(node_ip, node_port, "/deployments/start")
    # Use a longer timeout when pip_packages are specified (venv creation can take minutes)
    timeout = 600.0 if pip_packages else 10.0
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                url,
                json={
                    "model_name": model_name,
                    "port": port,
                    "gpu_memory_fraction": gpu_memory_fraction,
                    "gpu_ids": gpu_ids,
                    "tensor_parallel_size": tensor_parallel_size,
                    "extra_args": extra_args,
                    "env_vars": env_vars,
                    "pip_packages": pip_packages,
                },
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach client at {url}: {exc}",
            ) from exc

        if response.is_success:
            return

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


async def stop_model(node_ip: str, node_port: int | None, key: str) -> None:
    url = _satellite_url(node_ip, node_port, "/deployments/stop")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json={"key": key})
        response.raise_for_status()


async def get_statuses(node_ip: str, node_port: int | None) -> list[dict[str, object]]:
    url = _satellite_url(node_ip, node_port, "/deployments/status")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        return payload.get("deployments", [])


async def get_logs(
    node_ip: str, node_port: int | None, key: str, tail: int = 200
) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/deployments/logs")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params={"key": key, "tail": tail})
        if response.status_code == 404:
            return {"key": key, "lines": []}
        response.raise_for_status()
        return response.json()


async def get_metrics(node_ip: str, node_port: int | None) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/metrics")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def check_port(node_ip: str, node_port: int | None, port: int) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/ports/check")
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(url, params={"port": port})
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach client at {url}: {exc}",
            ) from exc

        if response.is_success:
            return response.json()

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


async def upload_package(
    node_ip: str,
    node_port: int | None,
    filename: str,
    file_bytes: bytes,
) -> dict[str, str]:
    url = _satellite_url(node_ip, node_port, "/packages/upload")
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                url,
                files={"file": (filename, file_bytes)},
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach client at {url}: {exc}",
            ) from exc

        if response.is_success:
            return response.json()

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


async def get_packages(node_ip: str, node_port: int | None) -> dict[str, object]:
    url = _satellite_url(node_ip, node_port, "/packages")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach client at {url}: {exc}",
            ) from exc

        if response.is_success:
            return response.json()

        raise HTTPException(
            status_code=response.status_code,
            detail=f"Client rejected request with status {response.status_code}.",
        )
