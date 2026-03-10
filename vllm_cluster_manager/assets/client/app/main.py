from contextlib import asynccontextmanager
import asyncio
from collections import deque
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
from urllib.request import Request, urlopen

import tarfile
import tempfile
import zipfile

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import psutil

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-fetch latest vLLM version at startup
    _get_latest_vllm_version()
    register_node()
    task = asyncio.create_task(register_loop())
    yield
    task.cancel()


app = FastAPI(title="vLLM Satellite", lifespan=lifespan)

_processes: dict[str, subprocess.Popen] = {}
_statuses: dict[str, dict[str, object]] = {}
_logs: dict[str, deque[str]] = {}


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
# Per-deployment venv management
# ---------------------------------------------------------------------------

_CLIENT_ROOT = Path(os.environ.get("VLLM_CLIENT_ROOT", Path.home() / ".vllm-client"))
_VENVS_DIR = _CLIENT_ROOT / ".venvs"
_PACKAGES_DIR = _CLIENT_ROOT / ".packages"


def _venv_hash(version: str, deployment_key: str = "") -> str:
    """Deterministic short hash for a version string and deployment key."""
    canonical = version + "\n" + deployment_key
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _detect_cuda_compact() -> int | None:
    """Return the CUDA version as a compact int (e.g. 130 for 13.0).

    Tries nvcc first, then nvidia-smi.  Returns None when detection fails.
    """
    import re as _re

    for cmd, pattern in (
        (["nvcc", "--version"], r"release\s+(\d+)\.(\d+)"),
        (["nvidia-smi"], r"CUDA Version:\s*(\d+)\.(\d+)"),
    ):
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except Exception:
            continue
        m = _re.search(pattern, out)
        if m:
            return int(m.group(1)) * 10 + int(m.group(2))
    return None


def _vllm_wheel_url(vllm_version: str, cuda_compact: int, cpu_arch: str) -> str:
    return (
        "https://github.com/vllm-project/vllm/releases/download/"
        f"v{vllm_version}/vllm-{vllm_version}+cu{cuda_compact}-"
        f"cp38-abi3-manylinux_2_35_{cpu_arch}.whl"
    )


def _url_exists(url: str) -> bool:
    request = Request(url, method="HEAD", headers={"User-Agent": "vllm-cluster-manager"})
    try:
        with urlopen(request, timeout=10):
            return True
    except Exception:
        return False


def _find_highest_available_cuda(vllm_version: str, cpu_arch: str) -> int | None:
    """Search for the highest CUDA version that has a published vLLM wheel."""
    highest: int | None = None
    consecutive_misses = 0
    for cu in range(128, 200):
        if _url_exists(_vllm_wheel_url(vllm_version, cu, cpu_arch)):
            highest = cu
            consecutive_misses = 0
        else:
            if highest is not None:
                consecutive_misses += 1
                if consecutive_misses >= 5:
                    break
    return highest


def _fetch_vllm_wheel_url_from_index(index_base_url: str, cpu_arch: str) -> str | None:
    """Fetch the PEP 503 index page and return the wheel URL matching *cpu_arch*.

    *index_base_url* is e.g. ``https://wheels.vllm.ai/nightly/cu130``.
    The function fetches ``{index_base_url}/vllm/``, parses ``<a href="...">``
    links, filters for wheels containing *cpu_arch*, resolves relative hrefs,
    and returns the absolute URL of the last (most recent) match.
    """
    from urllib.parse import urljoin

    page_url = index_base_url.rstrip("/") + "/vllm/"
    request = Request(page_url, headers={"User-Agent": "vllm-cluster-client"})
    try:
        html = urlopen(request, timeout=30).read().decode("utf-8")
    except Exception:
        return None

    hrefs = re.findall(r'href="([^"]+\.whl)"', html)
    matches = [h for h in hrefs if cpu_arch in h]
    if not matches:
        return None
    # Last entry is typically the most recent build
    return urljoin(page_url, matches[-1])


def _find_uv() -> str:
    """Return the path to the ``uv`` binary, or raise if not found."""
    uv = shutil.which("uv")
    if uv is not None:
        return uv
    # uv may not be on PATH in systemd; check common install locations.
    candidates: list[Path] = [
        Path("/usr/local/bin/uv"),
        Path("/usr/bin/uv"),
    ]
    # Check home dirs (current user, root, and all /home/* users)
    homes = [Path.home(), Path("/root")]
    try:
        homes.extend(sorted(Path("/home").iterdir()))
    except OSError:
        pass
    for home in homes:
        candidates.append(home / ".local" / "bin" / "uv")
        candidates.append(home / ".cargo" / "bin" / "uv")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        "uv is not installed or not on PATH. "
        "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


async def _get_or_create_vllm_venv(
    version: str,
    deployment_key: str = "",
    log_deque: deque | None = None,
    extra_packages: list[str] | None = None,
) -> str:
    """Create (or reuse) an isolated venv and install vLLM using uv.

    Returns the path to the venv's Python binary.
    """
    venv_id = _venv_hash(version, deployment_key)
    venv_dir = _VENVS_DIR / venv_id
    python_bin = venv_dir / "bin" / "python"
    marker = venv_dir / ".installed"

    if marker.exists() and python_bin.exists():
        logger.info("Reusing cached venv %s", venv_id)
        if log_deque is not None:
            log_deque.append(f"[uv] Reusing cached venv {venv_id}")
        return str(python_bin)

    venv_dir.mkdir(parents=True, exist_ok=True)
    uv = _find_uv()

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_deque is not None:
            log_deque.append(msg)

    # Timeouts for each step
    _VENV_CREATE_TIMEOUT = 300  # 5 minutes for venv creation
    _PIP_INSTALL_TIMEOUT = 1500  # 25 minutes for vLLM install (large wheels)
    _EXTRAS_INSTALL_TIMEOUT = 600  # 10 minutes for extra packages

    # Create venv with uv
    _log(f"[uv] Creating venv {venv_id}...")
    proc = await asyncio.create_subprocess_exec(
        uv, "venv", "--allow-existing", "--python", sys.executable, str(venv_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_VENV_CREATE_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        _log(f"[uv] venv creation timed out after {_VENV_CREATE_TIMEOUT}s")
        raise RuntimeError(f"venv creation timed out after {_VENV_CREATE_TIMEOUT}s")
    if proc.returncode != 0:
        _log(f"[uv] venv creation failed: {stdout.decode(errors='replace')}")
        raise RuntimeError(f"Failed to create venv: {stdout.decode(errors='replace')}")

    # Determine install command based on version type
    pip_cmd = [uv, "pip", "install", "--python", str(python_bin)]

    # Detect system CUDA version
    cuda_compact = _detect_cuda_compact()

    if re.match(r"^\d+\.\d+(\.\d+)?.*$", version):
        # Release version — install from the CUDA-specific GitHub release wheel
        cpu_arch = platform.machine()
        install_cuda = cuda_compact
        if install_cuda:
            wheel_url = _vllm_wheel_url(version, install_cuda, cpu_arch)
            if not _url_exists(wheel_url):
                _log(
                    f"[uv] No vLLM wheel for cu{install_cuda}, "
                    "searching for highest compatible CUDA wheel..."
                )
                fallback = _find_highest_available_cuda(version, cpu_arch)
                if fallback is None:
                    raise RuntimeError(
                        f"No vLLM wheel found for CUDA "
                        f"{install_cuda // 10}.{install_cuda % 10} "
                        f"(cu{install_cuda}) on {cpu_arch}, "
                        "and no fallback CUDA version wheel was found."
                    )
                fallback_major, fallback_minor = divmod(fallback, 10)
                _log(
                    f"[uv] Using vLLM wheel for CUDA "
                    f"{fallback_major}.{fallback_minor} (cu{fallback}) "
                    f"instead of cu{install_cuda}"
                )
                install_cuda = fallback
                wheel_url = _vllm_wheel_url(version, install_cuda, cpu_arch)
            _log(f"[uv] Installing vllm=={version}+cu{install_cuda} ...")
            pip_cmd.extend([
                wheel_url,
                "--extra-index-url", f"https://download.pytorch.org/whl/cu{install_cuda}",
                "--index-strategy", "unsafe-best-match",
            ])
        else:
            _log(f"[uv] Installing vllm=={version} (no CUDA detected) ...")
            pip_cmd.append("vllm==" + version)
    elif version.lower() == "nightly":
        # Nightly build — fetch the direct wheel URL from the vLLM index.
        # We cannot rely on uv index resolution because PEP 440 ranks the
        # stable PyPI release higher than nightly dev wheels.
        cpu_arch = platform.machine()
        if cuda_compact:
            index_base = f"https://wheels.vllm.ai/nightly/cu{cuda_compact}"
        else:
            index_base = "https://wheels.vllm.ai/nightly"
        wheel_url = _fetch_vllm_wheel_url_from_index(index_base, cpu_arch)
        if not wheel_url:
            raise RuntimeError(f"No nightly vLLM wheel found at {index_base} for {cpu_arch}")
        _log(f"[uv] Installing vllm nightly from {wheel_url} ...")
        pip_cmd.append(wheel_url)
        if cuda_compact:
            pip_cmd.extend([
                "--extra-index-url", f"https://download.pytorch.org/whl/cu{cuda_compact}",
                "--index-strategy", "unsafe-best-match",
            ])
    elif re.match(r"^[0-9a-f]{40}$", version):
        # Commit hash — fetch the direct wheel URL, same approach as nightly.
        cpu_arch = platform.machine()
        if cuda_compact:
            index_base = f"https://wheels.vllm.ai/{version}/cu{cuda_compact}"
        else:
            index_base = f"https://wheels.vllm.ai/{version}"
        wheel_url = _fetch_vllm_wheel_url_from_index(index_base, cpu_arch)
        if not wheel_url:
            raise RuntimeError(f"No vLLM wheel found at {index_base} for {cpu_arch}")
        _log(f"[uv] Installing vllm from commit {version[:12]} ...")
        pip_cmd.append(wheel_url)
        if cuda_compact:
            pip_cmd.extend([
                "--extra-index-url", f"https://download.pytorch.org/whl/cu{cuda_compact}",
                "--index-strategy", "unsafe-best-match",
            ])
    else:
        _log(f"[uv] Unrecognised version format '{version}', treating as release specifier")
        pip_cmd.append("vllm==" + version)

    proc = await asyncio.create_subprocess_exec(
        *pip_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_PIP_INSTALL_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        _log(f"[uv] vLLM installation timed out after {_PIP_INSTALL_TIMEOUT}s")
        raise RuntimeError(f"vLLM installation timed out after {_PIP_INSTALL_TIMEOUT}s")
    for line in stdout.decode(errors="replace").splitlines():
        _log(f"[uv] {line}")

    if proc.returncode != 0:
        _log(f"[uv] Installation failed (exit {proc.returncode})")
        tail = "\n".join(stdout.decode(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"uv pip install failed (exit {proc.returncode}):\n{tail}")

    # Install extra packages if provided
    if extra_packages:
        _log(f"[uv] Installing {len(extra_packages)} extra package(s)...")
        extras_cmd = [uv, "pip", "install", "--python", str(python_bin)] + extra_packages
        proc = await asyncio.create_subprocess_exec(
            *extras_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_EXTRAS_INSTALL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            _log(f"[uv] Extra packages installation timed out after {_EXTRAS_INSTALL_TIMEOUT}s")
            raise RuntimeError(f"Extra packages installation timed out after {_EXTRAS_INSTALL_TIMEOUT}s")
        for line in stdout.decode(errors="replace").splitlines():
            _log(f"[uv] {line}")

        if proc.returncode != 0:
            _log(f"[uv] Extra packages installation failed (exit {proc.returncode})")
            tail = "\n".join(stdout.decode(errors="replace").splitlines()[-20:])
            raise RuntimeError(f"uv pip install (extra packages) failed (exit {proc.returncode}):\n{tail}")

    marker.touch()
    _log(f"[uv] Venv {venv_id} ready")
    return str(python_bin)


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
    return {"running": list(_processes.keys())}


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
    if key in _processes:
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

    # Resolve Python binary: always use a venv with the requested or latest vLLM version
    log_buf = _logs[key] = deque(maxlen=2000)
    resolved_version = payload.vllm_version or _get_latest_vllm_version()
    if not resolved_version:
        raise HTTPException(
            status_code=500,
            detail="Could not determine vLLM version. Set vllm_version explicitly.",
        )
    try:
        venv_id = _venv_hash(resolved_version, key)
        python_bin = await _get_or_create_vllm_venv(
            resolved_version, key, log_buf, payload.extra_packages,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cmd = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        payload.model_name,
        "--port",
        str(payload.port),
        "--gpu-memory-utilization",
        str(payload.gpu_memory_fraction),
    ]
    if payload.tensor_parallel_size:
        cmd.extend(["--tensor-parallel-size", str(payload.tensor_parallel_size)])
    if payload.extra_args:
        cmd.extend(payload.extra_args)

    env = os.environ.copy()

    # Add nvidia and PyTorch library paths from venv so CUDA shared libs are found
    venv_site = Path(python_bin).parent.parent / "lib"
    if venv_site.exists():
        lib_dirs: list[str] = []
        # nvidia packages: site-packages/nvidia/*/lib
        for sp in venv_site.rglob("site-packages/nvidia/*/lib"):
            if sp.is_dir():
                lib_dirs.append(str(sp))
        # PyTorch bundles CUDA runtime libs in torch/lib
        for sp in venv_site.rglob("site-packages/torch/lib"):
            if sp.is_dir():
                lib_dirs.append(str(sp))
        if lib_dirs:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (
                f":{existing_ld}" if existing_ld else ""
            )
            logger.info("LD_LIBRARY_PATH set with %d venv lib dirs", len(lib_dirs))

    if payload.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in payload.gpu_ids)
    if payload.env_vars:
        for pair in payload.env_vars:
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

    def _mask_env_value(key_name: str, value: str) -> str:
        upper = key_name.upper()
        if any(token in upper for token in ["TOKEN", "SECRET", "KEY", "PASSWORD"]):
            if len(value) <= 8:
                return "*" * len(value)
            return f"{value[:4]}...{value[-4:]}"
        return value

    def _mask_env_for_log(env_map: dict[str, str]) -> dict[str, str]:
        return {k: _mask_env_value(k, str(v)) for k, v in env_map.items()}

    try:
        logger.info("Starting deployment %s", key)
        logger.info("Command: %s", " ".join(cmd))
        logger.info("Env overrides: %s", _mask_env_for_log(env))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _processes[key] = process
    _statuses[key] = {
        "model_name": payload.model_name,
        "port": payload.port,
        "gpu_memory_fraction": payload.gpu_memory_fraction,
        "gpu_ids": payload.gpu_ids or [],
        "tensor_parallel_size": payload.tensor_parallel_size,
        "vllm_version": resolved_version,
        "venv_id": venv_id,
        "status": "loading",
        "desired_state": "running",
    }
    asyncio.create_task(_stream_output(key, process))
    asyncio.create_task(_monitor_process(key, process))
    return {"status": "started", "key": key, "vllm_version": resolved_version}


class StopRequest(BaseModel):
    key: str


async def _force_kill(key: str, process: subprocess.Popen, timeout: float = 10.0) -> None:
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=timeout)
        return
    except asyncio.TimeoutError:
        logger.warning("Force killing deployment %s after %.1fs", key, timeout)
    except Exception as exc:
        logger.warning("Failed waiting for deployment %s to exit: %s", key, exc)

    try:
        process.kill()
    except Exception as exc:
        logger.warning("Failed to kill deployment %s: %s", key, exc)


@app.post("/deployments/stop")
async def stop_deployment(payload: StopRequest) -> dict[str, str]:
    key = payload.key
    process = _processes.get(key)
    if not process:
        raise HTTPException(status_code=404, detail="Deployment not found")

    venv_id = _statuses.get(key, {}).get("venv_id") if key in _statuses else None

    if key in _statuses:
        _statuses[key]["status"] = "stopping"
        _statuses[key]["desired_state"] = "stopped"

    process.terminate()
    asyncio.create_task(_force_kill(key, process))
    _processes.pop(key, None)

    # Tear down the deployment venv
    if venv_id:
        venv_dir = _VENVS_DIR / venv_id
        if venv_dir.exists():
            try:
                shutil.rmtree(venv_dir)
                logger.info("Removed venv %s for deployment %s", venv_id, key)
            except Exception as exc:
                logger.warning("Failed to remove venv %s: %s", venv_id, exc)

    return {"status": "stopped", "key": key}


async def _monitor_process(key: str, process: subprocess.Popen) -> None:
    # Mark running only after the vLLM HTTP server responds.
    port = _statuses[key].get("port")
    while True:
        await asyncio.sleep(2)
        code = process.poll()
        if code is not None:
            desired = _statuses.get(key, {}).get("desired_state")
            if desired == "stopped":
                _statuses[key]["status"] = "stopped"
            else:
                _statuses[key]["status"] = "error"
                _statuses[key]["exit_code"] = code
            _processes.pop(key, None)
            break

        if await _is_ready(port):
            if _statuses.get(key, {}).get("desired_state") == "running":
                _statuses[key]["status"] = "running"
            # Once running, continue to monitor for exits.
            continue


async def _is_ready(port: object) -> bool:
    if not isinstance(port, int):
        return False

    async with httpx.AsyncClient(timeout=2.0) as client:
        # Try common vLLM readiness endpoints.
        for path in ("/health", "/v1/models"):
            try:
                response = await client.get(f"http://127.0.0.1:{port}{path}")
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
    return False


async def _stream_output(key: str, process: subprocess.Popen) -> None:
    if process.stdout is None:
        return

    ansi_escape = re.compile(r"(?:\x1B|\u241B|\x9B)\[[0-?]*[ -/]*[@-~]")
    ansi_escape_alt = re.compile(r"(?:\x1B|\u241B|\x9B)[@-Z\\-_]")

    def _reader() -> None:
        for line in iter(process.stdout.readline, ""):
            cleaned = ansi_escape.sub("", line)
            cleaned = ansi_escape_alt.sub("", cleaned)
            cleaned = cleaned.replace("\u241b", "").rstrip()
            _logs.setdefault(key, deque(maxlen=2000)).append(cleaned)

    await asyncio.to_thread(_reader)


# ---------------------------------------------------------------------------
# Venv cache management endpoints
# ---------------------------------------------------------------------------

@app.get("/venvs")
def list_venvs() -> dict[str, list[dict[str, object]]]:
    venvs: list[dict[str, object]] = []
    if _VENVS_DIR.exists():
        for entry in sorted(_VENVS_DIR.iterdir()):
            if entry.is_dir():
                venvs.append({
                    "id": entry.name,
                    "path": str(entry),
                    "installed": (entry / ".installed").exists(),
                })
    return {"venvs": venvs}


@app.delete("/venvs/{venv_id}")
def delete_venv(venv_id: str) -> dict[str, str]:
    venv_dir = _VENVS_DIR / venv_id
    if not venv_dir.exists() or not venv_dir.is_dir():
        raise HTTPException(status_code=404, detail="Venv not found")
    shutil.rmtree(venv_dir)
    return {"status": "deleted", "id": venv_id}


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
