from contextlib import asynccontextmanager
import asyncio
from collections import deque
import logging
import os
import re
import shutil
import socket
import subprocess
import sys

import httpx
from fastapi import FastAPI, HTTPException
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

    cmd = [
        sys.executable,
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
            key = pair.get("key")
            if not key:
                continue
            env[key] = str(pair.get("value", ""))

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
        "status": "loading",
        "desired_state": "running",
    }
    _logs[key] = deque(maxlen=2000)
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
