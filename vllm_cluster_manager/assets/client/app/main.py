from contextlib import asynccontextmanager
import asyncio
from collections import deque
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

import tarfile
import tempfile
import zipfile

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.types import DeviceRequest
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import psutil
from urllib.request import Request, urlopen

from app.config import settings
from app.consul import register_node, register_loop

logger = logging.getLogger("vllm-cluster-client")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_latest_vllm_version: str = ""
_latest_vllm_version_fetched_at: float = 0.0
_LATEST_VERSION_TTL: float = 3600.0  # re-fetch once per hour


def _fetch_latest_vllm_version() -> str:
    """Fetch the latest vLLM release version from GitHub."""
    url = "https://api.github.com/repos/vllm-project/vllm/releases/latest"
    request = Request(url, headers={"User-Agent": "vllm-cluster-client"})
    data = json.loads(urlopen(request, timeout=15).read().decode("utf-8"))
    tag = data.get("tag_name", "")
    version = tag.lstrip("v")
    if not version:
        raise RuntimeError("Unable to determine latest vLLM version from GitHub releases.")
    return version


def _get_latest_vllm_version() -> str:
    """Return the latest stable vLLM release from GitHub (cached with TTL)."""
    import time

    global _latest_vllm_version, _latest_vllm_version_fetched_at
    now = time.monotonic()
    if _latest_vllm_version and (now - _latest_vllm_version_fetched_at) < _LATEST_VERSION_TTL:
        return _latest_vllm_version
    try:
        _latest_vllm_version = _fetch_latest_vllm_version()
        _latest_vllm_version_fetched_at = now
        logger.info("Resolved latest vLLM version: %s", _latest_vllm_version)
    except Exception as exc:
        logger.warning("Failed to fetch latest vLLM version: %s", exc)
        if _latest_vllm_version:
            # Keep the stale value if we had one
            _latest_vllm_version_fetched_at = now
    return _latest_vllm_version


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

# key -> docker Container object for the running deployment
_containers: dict[str, "docker.models.containers.Container"] = {}
_statuses: dict[str, dict[str, object]] = {}
_logs: dict[str, deque[str]] = {}

_CLIENT_ROOT = Path(os.environ.get("VLLM_CLIENT_ROOT", Path.home() / ".vllm-client"))
_PACKAGES_DIR = _CLIENT_ROOT / ".packages"

# Container labels so the agent can rediscover and reconcile its deployments
# after a restart. Docker is the source of truth, not in-memory state.
_LABEL_MANAGED = "vllm-cluster-manager.managed"
_LABEL_KEY = "vllm-cluster-manager.key"
_LABEL_PORT = "vllm-cluster-manager.port"
_LABEL_VERSION = "vllm-cluster-manager.version"

# Where the uploaded packages dir is mounted inside every vLLM container, so
# uploaded plugin (.py) files referenced in extra_args remain accessible.
_CONTAINER_PACKAGES_MOUNT = "/packages"

_LOCAL_IMAGE_REPO = "vllm-cluster-manager/local"

_ANSI_ESCAPE = re.compile(r"(?:\x1B|␛|\x9B)\[[0-?]*[ -/]*[@-~]")
_ANSI_ESCAPE_ALT = re.compile(r"(?:\x1B|␛|\x9B)[@-Z\\-_]")


def _strip_ansi(line: str) -> str:
    """Remove ANSI escape sequences and trailing whitespace from a log line."""
    cleaned = _ANSI_ESCAPE.sub("", line)
    cleaned = _ANSI_ESCAPE_ALT.sub("", cleaned)
    return cleaned.replace("␛", "").rstrip()


_docker_client: "docker.DockerClient | None" = None


def _docker() -> "docker.DockerClient":
    """Return a cached Docker client, raising a clear error if unavailable."""
    global _docker_client
    if _docker_client is None:
        try:
            _docker_client = docker.from_env()
            _docker_client.ping()
        except Exception as exc:  # pragma: no cover - environment dependent
            _docker_client = None
            raise RuntimeError(
                "Cannot talk to the Docker daemon. Ensure Docker is installed and "
                "running, and that this user can access it (add the user to the "
                f"'docker' group or run as root). Underlying error: {exc}"
            ) from exc
    return _docker_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-fetch latest vLLM version at startup
    _get_latest_vllm_version()
    # Rediscover deployments that survived an agent restart (containers keep
    # running independently of this process).
    try:
        _reconcile_containers()
    except Exception as exc:  # best effort — never block startup
        logger.warning("Container reconciliation skipped: %s", exc)
    # Best-effort initial registration: if Consul is unreachable at boot, the
    # agent still starts and register_loop keeps retrying, instead of crashing.
    try:
        register_node()
    except Exception as exc:
        logger.warning("Initial Consul registration failed (will keep retrying): %s", exc)
    task = asyncio.create_task(register_loop())
    yield
    task.cancel()


app = FastAPI(title="vLLM Satellite", lifespan=lifespan)


class StartRequest(BaseModel):
    model_name: str
    port: int
    gpu_memory_fraction: float
    gpu_ids: list[int] | None = None
    tensor_parallel_size: int | None = None
    extra_args: list[str] | None = None
    env_vars: list[dict[str, str]] | None = None
    vllm_version: str | None = None
    extra_packages: list[str] | None = None


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------

_RELEASE_RE = re.compile(r"^\d+\.\d+(\.\d+)?.*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _resolve_image_tag(version: str | None) -> tuple[str, str]:
    """Map a requested vLLM version to an official image ref + resolved version.

    Returns ``(image_ref, resolved_version)`` where ``resolved_version`` is the
    string stored in the database for display.

    | input              | image tag           |
    | ------------------ | ------------------- |
    | blank / None       | v{latest release}   |
    | ``0.8.5``          | v0.8.5              |
    | ``nightly``        | nightly             |
    | 40-char commit     | nightly-{commit}    |
    | anything else      | used as a literal tag |
    """
    repo = settings.vllm_image_repo
    raw = (version or "").strip()

    if not raw:
        resolved = _get_latest_vllm_version()
        if not resolved:
            raise RuntimeError(
                "Could not determine the latest vLLM version. "
                "Set the vLLM version explicitly."
            )
        return f"{repo}:v{resolved}", resolved
    if raw.lower() == "nightly":
        return f"{repo}:nightly", "nightly"
    if _COMMIT_RE.match(raw):
        return f"{repo}:nightly-{raw}", raw
    if _RELEASE_RE.match(raw):
        return f"{repo}:v{raw}", raw
    # Unknown format — treat the value as a literal image tag.
    return f"{repo}:{raw}", raw


def _derived_image_tag(base_ref: str, tokens: list[str]) -> str:
    """Deterministic tag for an image derived from *base_ref* + extra packages."""
    canonical = base_ref + " " + " ".join(sorted(tokens))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    base_part = re.sub(r"[^A-Za-z0-9._-]", "-", base_ref.split("/")[-1])
    return f"{_LOCAL_IMAGE_REPO}:{base_part}-{digest}"


def _pull_image(client: "docker.DockerClient", image_ref: str, log) -> None:
    """Pull *image_ref*, streaming layer progress into the deployment log."""
    repository, _, tag = image_ref.partition(":")
    tag = tag or "latest"
    log(f"[docker] Pulling {image_ref} ...")
    seen: dict[str, str] = {}
    try:
        for chunk in client.api.pull(repository, tag=tag, stream=True, decode=True):
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            layer = chunk.get("id", "")
            status = chunk.get("status", "")
            if not status:
                continue
            line = f"{layer}: {status}" if layer else status
            # Collapse repeated per-layer progress lines to keep logs readable.
            if seen.get(layer) != status:
                seen[layer] = status
                log(f"[docker] {line}")
    except (APIError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to pull image {image_ref}: {exc}. "
            "Verify the vLLM version/tag exists on Docker Hub."
        ) from exc
    log(f"[docker] Pulled {image_ref}")


def _build_derived_image(
    client: "docker.DockerClient",
    base_ref: str,
    derived_tag: str,
    extra_packages: list[str],
    log,
) -> None:
    """Build a thin image ``FROM base_ref`` with *extra_packages* installed.

    Local package paths (uploaded wheels/sdists under the packages dir) are
    copied into the build context; everything else is treated as a pip
    specifier. Built once and cached by *derived_tag*.
    """
    ctx = Path(tempfile.mkdtemp(prefix="vllm-build-"))
    try:
        pkgs_dir = ctx / "pkgs"
        install_tokens: list[str] = []
        has_local = False
        for idx, token in enumerate(extra_packages):
            candidate = Path(token)
            if candidate.exists():
                has_local = True
                pkgs_dir.mkdir(exist_ok=True)
                dest_name = f"{idx}_{candidate.name}"
                dest = pkgs_dir / dest_name
                if candidate.is_dir():
                    shutil.copytree(candidate, dest)
                else:
                    shutil.copy2(candidate, dest)
                install_tokens.append(f"/tmp/pkgs/{dest_name}")
            else:
                install_tokens.append(token)

        dockerfile_lines = [f"FROM {base_ref}"]
        if has_local:
            dockerfile_lines.append("COPY pkgs/ /tmp/pkgs/")
        # exec-form RUN avoids shell interpretation of specifiers like
        # "transformers>=4.40".
        run_argv = ["python3", "-m", "pip", "install", "--no-cache-dir", *install_tokens]
        dockerfile_lines.append("RUN " + json.dumps(run_argv))
        (ctx / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")

        log(f"[docker] Building image with extra packages: {', '.join(extra_packages)}")
        try:
            for chunk in client.api.build(
                path=str(ctx), tag=derived_tag, rm=True, decode=True
            ):
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                stream = chunk.get("stream", "")
                if stream and stream.strip():
                    log(f"[docker] {stream.rstrip()}")
        except (APIError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to build derived image: {exc}") from exc
        log(f"[docker] Built {derived_tag}")
    finally:
        shutil.rmtree(ctx, ignore_errors=True)


def _ensure_image(
    image_ref: str, extra_packages: list[str] | None, log
) -> str:
    """Ensure the runtime image exists locally, returning the ref to run.

    When *extra_packages* are requested, returns a cached derived image tag.
    """
    client = _docker()

    # Base image: reuse if already present, otherwise pull.
    try:
        client.images.get(image_ref)
        log(f"[docker] Using cached image {image_ref}")
    except ImageNotFound:
        _pull_image(client, image_ref, log)

    if not extra_packages:
        return image_ref

    derived_tag = _derived_image_tag(image_ref, extra_packages)
    try:
        client.images.get(derived_tag)
        log(f"[docker] Reusing cached derived image {derived_tag}")
        return derived_tag
    except ImageNotFound:
        pass
    _build_derived_image(client, image_ref, derived_tag, extra_packages, log)
    return derived_tag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _container_name(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", key)
    return f"vllm-cluster-{safe}"


def _rewrite_paths_for_container(args: list[str]) -> list[str]:
    """Rewrite host package paths in CLI args to their in-container mount path."""
    prefix = str(_PACKAGES_DIR)
    rewritten: list[str] = []
    for arg in args:
        if arg.startswith(prefix):
            rewritten.append(_CONTAINER_PACKAGES_MOUNT + arg[len(prefix):])
        else:
            rewritten.append(arg)
    return rewritten


def _gpu_count() -> int | None:
    """Best-effort count of GPUs on the host; None when it can't be determined."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            return len([line for line in out.splitlines() if line.strip()])
        except Exception:
            return None
    return None


def _build_environment(payload: "StartRequest") -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in payload.env_vars or []:
        env_key = pair.get("key")
        if not env_key:
            continue
        raw_value = str(pair.get("value", ""))
        trimmed = raw_value.strip()
        if (
            len(trimmed) >= 2
            and trimmed[0] == trimmed[-1]
            and trimmed[0] in {"'", '"'}
        ):
            env[env_key] = trimmed[1:-1]
        else:
            env[env_key] = raw_value
    return env


def _mask_env_value(key_name: str, value: str) -> str:
    upper = key_name.upper()
    if any(token in upper for token in ["TOKEN", "SECRET", "KEY", "PASSWORD"]):
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-4:]}"
    return value


def _mask_env_for_log(env_map: dict[str, str]) -> dict[str, str]:
    return {k: _mask_env_value(k, str(v)) for k, v in env_map.items()}


def _device_requests(gpu_ids: list[int] | None) -> list[DeviceRequest]:
    if gpu_ids:
        return [DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])]
    return [DeviceRequest(count=-1, capabilities=[["gpu"]])]


def _volumes() -> dict[str, dict[str, str]]:
    volumes: dict[str, dict[str, str]] = {}
    hf_cache = Path(os.path.expanduser(settings.hf_cache_dir)).resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)
    volumes[str(hf_cache)] = {"bind": "/root/.cache/huggingface", "mode": "rw"}
    if _PACKAGES_DIR.exists():
        volumes[str(_PACKAGES_DIR)] = {"bind": _CONTAINER_PACKAGES_MOUNT, "mode": "ro"}
    return volumes


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "gpus": _gpu_metrics(),
    }


@app.get("/deployments")
def list_deployments() -> dict[str, list[str]]:
    return {"running": list(_containers.keys())}


@app.get("/deployments/status")
def deployment_status() -> dict[str, list[dict[str, object]]]:
    deployments = []
    for key, meta in _statuses.items():
        deployments.append({"key": key, **meta})
    return {"deployments": deployments}


@app.get("/deployments/logs")
def deployment_logs(key: str, tail: int = 200) -> dict[str, object]:
    logs = _logs.get(key)
    if logs is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    tail = max(1, min(tail, 2000))
    return {"key": key, "lines": list(logs)[-tail:]}


@app.get("/ports/check")
def check_port(port: int) -> dict[str, object]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return {"available": False}
    return {"available": True}


@app.post("/deployments/start")
async def start_deployment(payload: StartRequest) -> dict[str, str]:
    key = f"{payload.model_name}:{payload.port}"
    if key in _containers:
        raise HTTPException(status_code=400, detail="Deployment already running")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", payload.port))
        except OSError:
            raise HTTPException(
                status_code=409,
                detail=f"Port {payload.port} is already in use on this node.",
            )

    # Validate requested GPUs exist on this host.
    if payload.gpu_ids:
        count = _gpu_count()
        if count is not None:
            bad = [g for g in payload.gpu_ids if g < 0 or g >= count]
            if bad:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested GPU id(s) {bad} not available; node has {count} GPU(s).",
                )

    log_buf = _logs[key] = deque(maxlen=2000)

    def _log(msg: str) -> None:
        logger.info(msg)
        log_buf.append(_strip_ansi(msg))

    try:
        image_ref, resolved_version = _resolve_image_tag(payload.vllm_version)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        run_image = await asyncio.to_thread(
            _ensure_image, image_ref, payload.extra_packages, _log
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    command = [
        "--model", payload.model_name,
        "--port", str(payload.port),
        "--gpu-memory-utilization", str(payload.gpu_memory_fraction),
    ]
    if payload.tensor_parallel_size:
        command.extend(["--tensor-parallel-size", str(payload.tensor_parallel_size)])
    if payload.extra_args:
        command.extend(_rewrite_paths_for_container(payload.extra_args))

    environment = _build_environment(payload)
    name = _container_name(key)
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_KEY: key,
        _LABEL_PORT: str(payload.port),
        _LABEL_VERSION: resolved_version,
    }

    try:
        logger.info("Starting deployment %s", key)
        logger.info("Image: %s  Command: %s", run_image, " ".join(command))
        logger.info("Env overrides: %s", _mask_env_for_log(environment))
        container = await asyncio.to_thread(
            _run_container,
            run_image,
            command,
            name,
            environment,
            _device_requests(payload.gpu_ids),
            labels,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _containers[key] = container
    _statuses[key] = {
        "model_name": payload.model_name,
        "port": payload.port,
        "gpu_memory_fraction": payload.gpu_memory_fraction,
        "gpu_ids": payload.gpu_ids or [],
        "tensor_parallel_size": payload.tensor_parallel_size,
        "vllm_version": resolved_version,
        "image": run_image,
        "container_id": container.id,
        "status": "loading",
        "desired_state": "running",
    }
    _log(f"[docker] Started container {name}")
    asyncio.create_task(_stream_container_logs(key, container))
    asyncio.create_task(_monitor_container(key, container))
    return {"status": "started", "key": key, "vllm_version": resolved_version}


def _run_container(
    image: str,
    command: list[str],
    name: str,
    environment: dict[str, str],
    device_requests: list[DeviceRequest],
    labels: dict[str, str],
) -> "docker.models.containers.Container":
    client = _docker()
    # Remove any stale container left over from a previous run with this name.
    try:
        existing = client.containers.get(name)
        existing.remove(force=True)
    except NotFound:
        pass
    try:
        return client.containers.run(
            image,
            command=command,
            name=name,
            detach=True,
            network_mode="host",
            ipc_mode="host",
            device_requests=device_requests,
            environment=environment,
            volumes=_volumes(),
            labels=labels,
            restart_policy={"Name": "unless-stopped"},
        )
    except (APIError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to start container: {exc}") from exc


class StopRequest(BaseModel):
    key: str


@app.post("/deployments/stop")
async def stop_deployment(payload: StopRequest) -> dict[str, str]:
    key = payload.key
    container = _containers.get(key)
    if not container:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if key in _statuses:
        _statuses[key]["status"] = "stopping"
        _statuses[key]["desired_state"] = "stopped"

    await asyncio.to_thread(_remove_container, container)
    _containers.pop(key, None)
    if key in _statuses:
        _statuses[key]["status"] = "stopped"
    return {"status": "stopped", "key": key}


def _remove_container(container: "docker.models.containers.Container") -> None:
    # Stop (SIGTERM, then SIGKILL after the grace period) and remove. The image
    # is kept — it is the warm-start cache for future deployments.
    try:
        container.stop(timeout=30)
    except NotFound:
        return
    except Exception as exc:
        logger.warning("Error stopping container %s: %s", container.id[:12], exc)
    try:
        container.remove(force=True)
    except NotFound:
        pass
    except Exception as exc:
        logger.warning("Error removing container %s: %s", container.id[:12], exc)


# A deployment that keeps restarting without ever becoming ready is
# mis-configured (bad args, GPU OOM, missing token, ...). After this many failed
# restarts the agent stops it so it doesn't loop forever under the
# unless-stopped policy — which otherwise still recovers healthy deployments
# across transient crashes and host reboots.
_MAX_FAILED_RESTARTS = 3


async def _monitor_container(
    key: str, container: "docker.models.containers.Container"
) -> None:
    """Track container state; flip the deployment to running once /health is up.

    Containers use the unless-stopped restart policy, so a non-zero exit is NOT
    terminal — Docker restarts it. A deployment is therefore considered finished
    only when (a) the user stopped it, (b) the container disappeared, or (c) it
    crash-loops without ever becoming ready (the breaker below stops it).
    """
    port = _statuses.get(key, {}).get("port")
    last_restart_count = 0
    ever_ready = False
    while True:
        await asyncio.sleep(2)
        try:
            await asyncio.to_thread(container.reload)
        except NotFound:
            # Container was removed out from under us.
            desired = _statuses.get(key, {}).get("desired_state")
            if key in _statuses:
                _statuses[key]["status"] = "stopped" if desired == "stopped" else "error"
            _containers.pop(key, None)
            break
        except Exception as exc:
            logger.warning("Error inspecting container for %s: %s", key, exc)
            continue

        status = container.status
        state = container.attrs.get("State", {})
        desired = _statuses.get(key, {}).get("desired_state")

        # The user asked it to stop: terminal once it actually exits.
        if desired == "stopped":
            if status in ("exited", "dead"):
                if key in _statuses:
                    _statuses[key]["status"] = "stopped"
                _containers.pop(key, None)
                break
            continue

        # desired == "running" below.
        restart_count = int(container.attrs.get("RestartCount", 0) or 0)
        if restart_count > last_restart_count:
            last_restart_count = restart_count
            _logs.setdefault(key, deque(maxlen=2000)).append(
                f"[docker] Container restarted (exit code {state.get('ExitCode')}); Docker is retrying."
            )
            if key in _statuses:
                _statuses[key]["status"] = "error"

            # Crash-loop breaker: a deployment that never became ready and keeps
            # restarting is mis-configured (bad args, GPU OOM, ...). Stop it so
            # unless-stopped does not retry forever; healthy deployments
            # (ever_ready) are left alone so transient crashes and reboots recover.
            if not ever_ready and restart_count >= _MAX_FAILED_RESTARTS:
                logger.warning(
                    "Crash-loop breaker tripped for %s after %d failed restarts",
                    key, restart_count,
                )
                _logs.setdefault(key, deque(maxlen=2000)).append(
                    f"[docker] Deployment failed to start after {restart_count} attempts and "
                    "never became ready; stopping retries and removing the container. See the "
                    "log above for the root cause (e.g. GPU out of memory, bad args), then fix "
                    "it and redeploy. The vLLM image stays cached for a fast retry."
                )
                if key in _statuses:
                    _statuses[key]["status"] = "error"
                    _statuses[key]["error"] = (
                        f"Stopped after {restart_count} failed starts without becoming ready."
                    )
                # Stop AND remove the container: stopping halts the unless-stopped
                # restart loop, and removing keeps `docker ps -a` clean. The failure
                # logs are preserved in this deployment's log buffer (shown in the
                # UI), and the pulled image stays cached for a fast redeploy.
                await asyncio.to_thread(_remove_container, container)
                _containers.pop(key, None)
                break

        if status in ("exited", "dead"):
            # Transient under unless-stopped — Docker will restart it. Reflect
            # that it isn't healthy and keep watching; the breaker handles the
            # repeated-failure case via RestartCount above.
            if key in _statuses:
                _statuses[key]["status"] = "error"
                _statuses[key]["exit_code"] = state.get("ExitCode")
            continue

        if await _is_ready(port):
            ever_ready = True
            if key in _statuses:
                _statuses[key]["status"] = "running"
            # Keep monitoring for exits/crash loops.
            continue


async def _is_ready(port: object) -> bool:
    if not isinstance(port, int):
        return False

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Try common vLLM readiness endpoints.
        for path in ("/health", "/v1/models"):
            try:
                response = await client.get(f"http://127.0.0.1:{port}{path}")
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
    return False


async def _stream_container_logs(
    key: str, container: "docker.models.containers.Container"
) -> None:
    def _reader() -> None:
        try:
            stream = container.logs(stream=True, follow=True, tail=200)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Failed to attach to logs for %s: %s", key, exc)
            return
        buf = _logs.setdefault(key, deque(maxlen=2000))
        for raw in stream:
            try:
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            except Exception:
                continue
            cleaned = _strip_ansi(line)
            if cleaned:
                buf.append(cleaned)

    await asyncio.to_thread(_reader)


def _reconcile_containers() -> None:
    """Rebuild in-memory state from managed containers after an agent restart."""
    client = _docker()
    containers = client.containers.list(
        all=True, filters={"label": f"{_LABEL_MANAGED}=true"}
    )
    recovered = 0
    for container in containers:
        labels = container.labels or {}
        key = labels.get(_LABEL_KEY)
        if not key:
            continue
        if container.status in ("exited", "dead"):
            # A previously-stopped deployment; leave it for cleanup, don't track.
            continue
        try:
            port = int(labels.get(_LABEL_PORT, "0"))
        except ValueError:
            port = 0
        _containers[key] = container
        _logs.setdefault(key, deque(maxlen=2000))
        _statuses[key] = {
            "model_name": key.rsplit(":", 1)[0],
            "port": port,
            "vllm_version": labels.get(_LABEL_VERSION),
            "image": (container.image.tags[0] if container.image.tags else None),
            "container_id": container.id,
            "status": "loading",
            "desired_state": "running",
        }
        asyncio.create_task(_stream_container_logs(key, container))
        asyncio.create_task(_monitor_container(key, container))
        recovered += 1
    if recovered:
        logger.info("Reconciled %d running deployment(s) from Docker", recovered)


# ---------------------------------------------------------------------------
# Image cache management endpoints
# ---------------------------------------------------------------------------


@app.get("/images")
def list_images() -> dict[str, list[dict[str, object]]]:
    """List cached vLLM images (official + locally derived) on this node."""
    images: list[dict[str, object]] = []
    try:
        client = _docker()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    repo = settings.vllm_image_repo
    for image in client.images.list():
        tags = list(image.tags or [])
        relevant = [t for t in tags if t.startswith(repo) or t.startswith(_LOCAL_IMAGE_REPO)]
        if not relevant:
            continue
        images.append({
            "id": image.short_id,
            "tags": relevant,
            "size_mb": round((image.attrs.get("Size", 0) or 0) / (1024 * 1024)),
        })
    return {"images": images}


@app.delete("/images/{image_id}")
def delete_image(image_id: str) -> dict[str, str]:
    try:
        client = _docker()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        client.images.remove(image_id)
    except ImageNotFound:
        raise HTTPException(status_code=404, detail="Image not found")
    except APIError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "deleted", "id": image_id}


# ---------------------------------------------------------------------------
# Package upload endpoints
# ---------------------------------------------------------------------------


@app.post("/packages/upload")
async def upload_package(file: UploadFile = File(...)) -> dict[str, str]:
    filename = file.filename or "package"
    data = await file.read()
    content_hash = hashlib.sha256(data).hexdigest()[:16]
    pkg_dir = _PACKAGES_DIR / content_hash
    pkg_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    file_type = "package"
    try:
        if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=str(pkg_dir))
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(path=str(pkg_dir))
        elif filename.endswith(".whl"):
            shutil.copy2(tmp_path, pkg_dir / filename)
        elif filename.endswith(".py"):
            # Python plugin files are stored as-is; they get passed to vLLM
            # via CLI flags like --reasoning-parser-plugin <path>
            shutil.copy2(tmp_path, pkg_dir / filename)
            file_type = "plugin"
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use .py, .whl, .tar.gz, or .zip",
            )
    finally:
        os.unlink(tmp_path)

    # Determine the pip-installable path:
    # - .whl: point to the .whl file itself
    # - .py: point to the .py file itself (not pip-installable, used as CLI arg)
    # - archive with single top-level dir: point to that dir
    # - otherwise: point to pkg_dir
    children = list(pkg_dir.iterdir())
    if len(children) == 1 and children[0].is_file():
        install_path = str(children[0])
    elif len(children) == 1 and children[0].is_dir():
        install_path = str(children[0])
    else:
        install_path = str(pkg_dir)

    return {
        "status": "uploaded",
        "package_id": content_hash,
        "filename": filename,
        "install_path": install_path,
        "type": file_type,
    }


@app.get("/packages")
def list_packages() -> dict[str, list[dict[str, str]]]:
    packages: list[dict[str, str]] = []
    if _PACKAGES_DIR.exists():
        for entry in sorted(_PACKAGES_DIR.iterdir()):
            if entry.is_dir():
                children = list(entry.iterdir())
                install_path = str(children[0]) if len(children) == 1 and children[0].is_dir() else str(entry)
                packages.append({
                    "id": entry.name,
                    "path": str(entry),
                    "install_path": install_path,
                })
    return {"packages": packages}


@app.delete("/packages/{package_id}")
def delete_package(package_id: str) -> dict[str, str]:
    pkg_dir = _PACKAGES_DIR / package_id
    if not pkg_dir.exists() or not pkg_dir.is_dir():
        raise HTTPException(status_code=404, detail="Package not found")
    shutil.rmtree(pkg_dir)
    return {"status": "deleted", "id": package_id}


def _gpu_metrics() -> list[dict[str, object]]:
    def _unified_memory_metrics() -> list[dict[str, object]]:
        mem = psutil.virtual_memory()
        return [
            {
                "index": 0,
                "name": "Unified memory",
                "source": "unified",
                "utilization": int(round(mem.percent)),
                "memory_used_mb": round(mem.used / (1024 * 1024)),
                "memory_total_mb": round(mem.total / (1024 * 1024)),
            }
        ]

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8", errors="ignore")
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "source": "nvml",
                    "utilization": util.gpu,
                    "memory_used_mb": round(mem.used / (1024 * 1024)),
                    "memory_total_mb": round(mem.total / (1024 * 1024)),
                }
            )
        return gpus
    except Exception as exc:
        logger.warning("NVML GPU metrics failed, falling back to nvidia-smi: %s", exc)

    if not shutil.which("nvidia-smi"):
        logger.warning("nvidia-smi not found on PATH; GPU metrics unavailable.")
        logger.warning("Falling back to unified memory metrics from system RAM.")
        return _unified_memory_metrics()

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        logger.info("nvidia-smi output:\n%s", output)
    except Exception as exc:
        logger.warning("nvidia-smi query failed; GPU metrics unavailable: %s", exc)
        logger.warning("Falling back to unified memory metrics from system RAM.")
        return _unified_memory_metrics()

    gpus = []
    for line in output.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 5:
            continue
        index_str, name, util_str, used_str, total_str = parts[:5]
        try:
            gpus.append(
                {
                    "index": int(index_str),
                    "name": name,
                    "source": "nvidia-smi",
                    "utilization": int(float(util_str)),
                    "memory_used_mb": int(float(used_str)),
                    "memory_total_mb": int(float(total_str)),
                }
            )
        except ValueError:
            continue
    if not gpus:
        logger.warning("nvidia-smi returned no GPU rows; using unified memory metrics.")
        return _unified_memory_metrics()
    return gpus


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
