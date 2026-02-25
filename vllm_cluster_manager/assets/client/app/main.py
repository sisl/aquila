from contextlib import asynccontextmanager
import asyncio
from collections import deque
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
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

@asynccontextmanager
async def lifespan(_: FastAPI):
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
    pip_packages: list[str] | None = None


# ---------------------------------------------------------------------------
# Per-deployment venv management
# ---------------------------------------------------------------------------

_CLIENT_ROOT = Path(os.environ.get("VLLM_CLIENT_ROOT", Path.home() / ".vllm-client"))
_VENVS_DIR = _CLIENT_ROOT / ".venvs"
_PACKAGES_DIR = _CLIENT_ROOT / ".packages"


def _venv_hash(packages: list[str]) -> str:
    """Deterministic short hash for a sorted package list."""
    canonical = "\n".join(sorted(packages))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


async def _get_or_create_venv(packages: list[str], log_deque: deque | None = None) -> str:
    """Create (or reuse) an isolated venv and install *packages*.

    Returns the path to the venv's Python binary.
    """
    venv_id = _venv_hash(packages)
    venv_dir = _VENVS_DIR / venv_id
    python_bin = venv_dir / "bin" / "python"
    marker = venv_dir / ".installed"

    if marker.exists() and python_bin.exists():
        logger.info("Reusing cached venv %s", venv_id)
        if log_deque is not None:
            log_deque.append(f"[pip] Reusing cached venv {venv_id}")
        return str(python_bin)

    venv_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_deque is not None:
            log_deque.append(msg)

    # Create venv
    _log(f"[pip] Creating venv {venv_id}...")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "venv", str(venv_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        _log(f"[pip] venv creation failed: {stdout.decode(errors='replace')}")
        raise RuntimeError(f"Failed to create venv: {stdout.decode(errors='replace')}")

    # Build pip install command
    pip_cmd = [
        str(python_bin), "-m", "pip", "install",
        "--extra-index-url", "https://download.pytorch.org/whl/cu124",
    ]
    pip_cmd.extend(packages)

    _log(f"[pip] Installing: {' '.join(packages)}")
    proc = await asyncio.create_subprocess_exec(
        *pip_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    for line in stdout.decode(errors="replace").splitlines():
        _log(f"[pip] {line}")

    if proc.returncode != 0:
        _log(f"[pip] Installation failed (exit {proc.returncode})")
        raise RuntimeError(f"pip install failed (exit {proc.returncode})")

    marker.touch()
    _log(f"[pip] Venv {venv_id} ready")
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

    # Resolve Python binary: per-deployment venv or system default
    log_buf = _logs.setdefault(key, deque(maxlen=2000))
    if payload.pip_packages:
        try:
            python_bin = await _get_or_create_venv(payload.pip_packages, log_buf)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        python_bin = sys.executable

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
        "pip_packages": payload.pip_packages or [],
        "status": "loading",
        "desired_state": "running",
    }
    asyncio.create_task(_stream_output(key, process))
    asyncio.create_task(_monitor_process(key, process))
    return {"status": "started", "key": key}


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

    if key in _statuses:
        _statuses[key]["status"] = "stopping"
        _statuses[key]["desired_state"] = "stopped"

    process.terminate()
    asyncio.create_task(_force_kill(key, process))
    _processes.pop(key, None)
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

    try:
        if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=str(pkg_dir))
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(path=str(pkg_dir))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported archive format. Use .tar.gz or .zip",
            )
    finally:
        os.unlink(tmp_path)

    # If the archive contained a single top-level directory, point to it
    children = list(pkg_dir.iterdir())
    install_path = str(children[0]) if len(children) == 1 and children[0].is_dir() else str(pkg_dir)

    return {
        "status": "uploaded",
        "package_id": content_hash,
        "install_path": install_path,
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
