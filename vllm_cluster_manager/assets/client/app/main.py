from contextlib import asynccontextmanager
import asyncio
from collections import deque
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess

import tarfile
import tempfile
import uuid
import time
import zipfile

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.types import DeviceRequest
import httpx
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psutil
from urllib.request import Request as UrllibRequest, urlopen

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
    request = UrllibRequest(url, headers={"User-Agent": "vllm-cluster-client"})
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
# Managed local models (uploaded through the host or pulled from a URL).
# Always part of the allowed model dirs and mounted into vLLM containers.
_MODELS_DIR = _CLIENT_ROOT / ".models"
# Staging area for in-flight uploads/pulls; same filesystem as _MODELS_DIR so
# the final rename into place is atomic.
_MODELS_TMP = _MODELS_DIR / ".tmp"
# Keep this much free disk after a transfer would complete.
_DISK_SAFETY_BYTES = 2 * 1024**3
_UPLOAD_SESSION_TTL = 6 * 3600.0

# Persistent per-deployment log files (full run history, not just the
# in-memory tail). One file per deployment run; the previous run is kept as
# "<file>.1". A "<file>.pos" sidecar stores the last persisted raw Docker
# timestamp so a restarted agent resumes the stream without duplicates.
_LOGS_DIR = _CLIENT_ROOT / ".logs"
# key -> {"fh": TextIO, "path": Path, "size": int, "raw_ts": str}
_log_state: dict[str, dict[str, object]] = {}

# Folder-upload sessions (id -> staging state). In-memory only: anything left
# in _MODELS_TMP after an agent restart is stale by definition and GC'd.
_upload_sessions: dict[str, dict[str, object]] = {}
# URL-pull transfers (id -> progress/status), pruned a while after finishing.
_transfers: dict[str, dict[str, object]] = {}

# Container labels so the agent can rediscover and reconcile its deployments
# after a restart. Docker is the source of truth, not in-memory state.
_LABEL_MANAGED = "vllm-cluster-manager.managed"
_LABEL_KEY = "vllm-cluster-manager.key"
_LABEL_PORT = "vllm-cluster-manager.port"
_LABEL_VERSION = "vllm-cluster-manager.version"
# Full launch manifest (JSON) so the host can restore owner/lease/args when it
# re-adopts a running container after losing its database.
_LABEL_LAUNCH = "vllm-cluster-manager.launch"
# Which runtime (docker/podman) runs this container, so cross-runtime
# enumeration can tag entries after an agent restart.
_LABEL_RUNTIME = "vllm-cluster-manager.runtime"
# Stamped on locally-built derived images so prune can reclaim the dangling
# (<none>) leftovers a moving-tag rebuild creates, without touching unrelated
# dangling images on the host.
_LABEL_DERIVED = "vllm-cluster-manager.derived"

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


# ---------------------------------------------------------------------------
# Persistent deployment logs
# ---------------------------------------------------------------------------

# Docker prepends RFC3339Nano timestamps when logs are requested with
# timestamps=True, e.g. "2026-06-11T08:00:32.123456789Z <line>".
_DOCKER_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?) ")
# Monitoring noise: uvicorn access-log lines for the endpoints the agent (and
# health checks) poll continuously. Inference requests (/v1/...) and engine
# output are deliberately kept.
_NOISE_RE = re.compile(r'"(?:GET|HEAD) /(?:metrics|health)[ ?/]')


def _log_file_for(key: str) -> Path:
    """Deterministic, filesystem-safe log path for a deployment key."""
    model, _, port = key.rpartition(":")
    safe = (model or key).replace("/", "--")
    return _LOGS_DIR / f"{safe}-{port or '0'}.log"


def _log_sidecar_for(path: Path) -> Path:
    return Path(str(path) + ".pos")


def _split_docker_ts(raw_line: str) -> tuple[str | None, str]:
    """Split a docker timestamps=True line into (raw_ts, rest)."""
    match = _DOCKER_TS_RE.match(raw_line)
    if not match:
        return None, raw_line
    return match.group(1), raw_line[match.end():]


def _format_log_line(raw_ts: str | None, text: str) -> str:
    """Prefix a log line with a readable UTC timestamp."""
    if raw_ts and len(raw_ts) >= 19:
        return f"[{raw_ts[:10]} {raw_ts[11:19]}] {text}"
    return text


def _is_noise_line(text: str) -> bool:
    return bool(_NOISE_RE.search(text))


def _open_log_file(key: str, rotate: bool = False) -> dict[str, object]:
    state = _log_state.get(key)
    if state is not None and not rotate:
        return state
    if state is not None:
        _close_log_file(key)
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _log_file_for(key)
    if rotate and path.exists():
        # One file per deployment run; keep the previous run as "<file>.1".
        path.replace(Path(str(path) + ".1"))
        _log_sidecar_for(path).unlink(missing_ok=True)
    raw_ts = ""
    sidecar = _log_sidecar_for(path)
    if not rotate and sidecar.exists():
        try:
            raw_ts = sidecar.read_text().strip()
        except OSError:
            raw_ts = ""
    fh = open(path, "a", buffering=1, encoding="utf-8")
    state = {
        "fh": fh,
        "path": path,
        "size": path.stat().st_size if path.exists() else 0,
        "raw_ts": raw_ts,
    }
    _log_state[key] = state
    return state


def _close_log_file(key: str) -> None:
    state = _log_state.pop(key, None)
    if state is not None:
        try:
            state["fh"].close()  # type: ignore[union-attr]
        except Exception:
            pass


def _append_log_line(key: str, formatted: str, raw_ts: str | None = None) -> None:
    """Append a line to the in-memory tail AND the persistent log file."""
    _logs.setdefault(key, deque(maxlen=2000)).append(formatted)
    try:
        state = _open_log_file(key)
        state["fh"].write(formatted + "\n")  # type: ignore[union-attr]
        state["size"] = int(state.get("size", 0)) + len(formatted) + 1  # type: ignore[arg-type]
        if raw_ts:
            state["raw_ts"] = raw_ts
            try:
                _log_sidecar_for(state["path"]).write_text(raw_ts)  # type: ignore[arg-type]
            except OSError:
                pass
        if int(state["size"]) > settings.log_max_mb * 1024 * 1024:  # type: ignore[arg-type]
            # Size cap reached: keep the overflow as "<file>.1", start fresh.
            _close_log_file(key)
            path = _log_file_for(key)
            path.replace(Path(str(path) + ".1"))
            _open_log_file(key)
            if raw_ts:
                _log_state[key]["raw_ts"] = raw_ts
    except OSError as exc:  # disk full / permissions — keep the in-memory tail
        logger.warning("Could not persist log line for %s: %s", key, exc)


def _append_agent_log(key: str, msg: str) -> None:
    """Agent-injected log lines ([docker]/[agent] ...), always kept."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _append_log_line(key, f"[{stamp}] {_strip_ansi(msg)}")


def _gc_old_logs() -> None:
    """Drop log files untouched for longer than the retention window."""
    if not _LOGS_DIR.exists():
        return
    cutoff = time.time() - settings.log_retention_days * 86400
    for entry in _LOGS_DIR.iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
        except OSError:
            continue


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


# Podman speaks the Docker REST API on its own socket, so the same docker-py
# client (pulls, builds, labels, log streams) works against it unchanged.
_podman_client: "docker.DockerClient | None" = None

RUNTIMES = ("docker", "podman")


def _podman_socket_candidates() -> list[str]:
    candidates = []
    env_sock = os.environ.get("PODMAN_SOCK")
    if env_sock:
        candidates.append(env_sock)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        candidates.append(f"{xdg}/podman/podman.sock")
    candidates.append(f"/run/user/{os.getuid()}/podman/podman.sock")
    candidates.append("/run/podman/podman.sock")
    return candidates


def _podman() -> "docker.DockerClient":
    """Return a cached client for Podman's Docker-compatible API socket."""
    global _podman_client
    if _podman_client is None:
        last_exc: Exception | None = None
        for sock in _podman_socket_candidates():
            if not Path(sock).exists():
                continue
            try:
                client = docker.DockerClient(base_url=f"unix://{sock}")
                client.ping()
                _podman_client = client
                break
            except Exception as exc:  # pragma: no cover - environment dependent
                last_exc = exc
        if _podman_client is None:
            raise RuntimeError(
                "Podman API socket not reachable. Enable it with "
                "`systemctl --user enable --now podman.socket` (rootless) or "
                "`systemctl enable --now podman.socket` (rootful)."
                + (f" Underlying error: {last_exc}" if last_exc else "")
            )
    return _podman_client


def _runtime_client(runtime: str) -> "docker.DockerClient":
    if runtime == "docker":
        return _docker()
    if runtime == "podman":
        return _podman()
    raise ValueError(f"Unknown container runtime: {runtime}")


# (monotonic ts, runtimes) — probing involves socket pings, so cache briefly.
_runtime_probe: tuple[float, list[str]] = (0.0, [])


def _available_runtimes() -> list[str]:
    global _runtime_probe
    ts, cached = _runtime_probe
    now = time.monotonic()
    if cached and now - ts < 60.0:
        return list(cached)
    if not cached and now - ts < 60.0 and ts > 0:
        return []
    available: list[str] = []
    for runtime in RUNTIMES:
        try:
            _runtime_client(runtime).ping()
            available.append(runtime)
        except Exception:
            continue
    _runtime_probe = (now, available)
    return list(available)


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
    # Drop staging leftovers from interrupted model uploads/pulls.
    try:
        _gc_stale_partials()
    except Exception as exc:
        logger.warning("Stale model staging cleanup skipped: %s", exc)
    # Age out persistent deployment log files.
    try:
        _gc_old_logs()
    except Exception as exc:
        logger.warning("Log retention cleanup skipped: %s", exc)
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
    # Structured engine flags (max_model_len, dtype, ...); see _ENGINE_ARG_FLAGS.
    engine_args: dict[str, object] | None = None
    # LoRA adapters to serve alongside the base model: [{name, path}]; path is
    # a host path inside an allowed model dir, or a HF hub adapter id.
    lora_modules: list[dict[str, str]] | None = None
    # Crash-loop breaker threshold for this deployment; None uses the default.
    max_failed_restarts: int | None = None
    # Bypass the GPU memory pre-check (e.g. the operator knows better).
    skip_resource_check: bool = False
    # Host-side launch metadata, persisted on the container (launch-manifest
    # label) purely so the host can restore it after losing its database. The
    # client never interprets these; expires_at is an opaque ISO-8601 string.
    owner: str | None = None
    duration_seconds: int | None = None
    expires_at: str | None = None
    # Which container runtime to use ("docker"/"podman"); None = first
    # available (also keeps old hosts working).
    container_runtime: str | None = None


# Allowlisted structured engine args -> vLLM CLI flags. Booleans emit a bare
# flag when true; everything else emits "flag value". Free-text extra_args is
# appended after these, so it wins on conflict (vLLM takes the last occurrence).
_ENGINE_ARG_FLAGS: dict[str, str] = {
    "max_model_len": "--max-model-len",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "served_model_name": "--served-model-name",
    "max_num_seqs": "--max-num-seqs",
    "enforce_eager": "--enforce-eager",
    "trust_remote_code": "--trust-remote-code",
    "kv_cache_dtype": "--kv-cache-dtype",
    "swap_space": "--swap-space",
    # Reproducibility pins: HF revision and sampling seed.
    "revision": "--revision",
    "seed": "--seed",
}


def _engine_args_to_cli(engine_args: dict[str, object] | None) -> list[str]:
    """Translate the structured engine-arg dict into vLLM CLI tokens."""
    if not engine_args:
        return []
    tokens: list[str] = []
    for key, value in engine_args.items():
        flag = _ENGINE_ARG_FLAGS.get(key)
        if flag is None or value is None or value == "":
            continue
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
        else:
            tokens.extend([flag, str(value)])
    return tokens


def _models_dir() -> Path:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _MODELS_DIR


def _allowed_model_dirs() -> list[Path]:
    # The managed dir is always allowed (and therefore always mounted), so
    # uploaded models are deployable without any MODEL_DIRS configuration.
    dirs: list[Path] = [_models_dir().resolve()]
    for raw in settings.model_dirs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(os.path.expanduser(raw)).resolve()
        if path in dirs:
            continue
        if path.is_dir():
            dirs.append(path)
        else:
            logger.warning("Configured model dir does not exist: %s", raw)
    return dirs


def _validate_host_path(path_str: str, what: str = "model") -> str:
    """Resolve a host path and require it inside an allowed model dir.

    resolve() collapses ../ and symlinks, so escapes out of the allowlist
    are rejected even via links.
    """
    allowed = _allowed_model_dirs()
    resolved = Path(path_str).resolve()
    allowed_note = (
        ", ".join(str(d) for d in allowed)
        if allowed
        else "none — set MODEL_DIRS on the agent"
    )
    if not any(d == resolved or d in resolved.parents for d in allowed):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{what} path {path_str} is not inside an allowed model "
                f"directory (allowed: {allowed_note})."
            ),
        )
    if not resolved.exists():
        raise HTTPException(
            status_code=400, detail=f"{what} path {path_str} does not exist on this node."
        )
    return str(resolved)


# ---------------------------------------------------------------------------
# Managed local models: upload / pull helpers
# ---------------------------------------------------------------------------

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _sanitize_model_name(name: str) -> str:
    """A managed model name is a single safe path segment."""
    name = (name or "").strip()
    if not _MODEL_NAME_RE.fullmatch(name) or name == ".tmp":
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model name: use letters, digits, '.', '_' or '-' "
                "(max 64 chars, must start with a letter or digit)."
            ),
        )
    return name


def _validate_rel_path(rel: str) -> PurePosixPath:
    """Validate a client-supplied relative file path inside an upload session."""
    rel = (rel or "").strip()
    if not rel or len(rel) > 1024:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    path = PurePosixPath(rel)
    parts = path.parts
    if path.is_absolute() or len(parts) > 32:
        raise HTTPException(status_code=400, detail=f"Invalid file path: {rel}")
    for part in parts:
        if part in ("..", ".") or not part or ":" in part or "\\" in part:
            raise HTTPException(status_code=400, detail=f"Invalid file path: {rel}")
    return path


def _safe_extract(archive_path: Path, dest: Path) -> None:
    """Extract a .tar.gz/.tgz/.zip while rejecting traversal and link members.

    Containment is checked per member (resolve() must stay inside dest), so
    this is safe on Python 3.10/3.11 too; on 3.12+ tar extraction additionally
    uses the stdlib "data" filter.
    """
    dest_resolved = dest.resolve()

    def _contained(member_name: str) -> bool:
        target = (dest / member_name).resolve()
        return target == dest_resolved or dest_resolved in target.parents

    name = archive_path.name
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not _contained(member.name):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Archive member escapes the target directory: {member.name}",
                    )
                if not (member.isfile() or member.isdir()):
                    logger.warning("Skipping non-regular archive member: %s", member.name)
                    continue
                if hasattr(tarfile, "data_filter"):
                    tar.extract(member, dest, filter="data")
                else:  # Python <= 3.11
                    tar.extract(member, dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if not _contained(info.filename):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Archive member escapes the target directory: {info.filename}",
                    )
                zf.extract(info, dest)
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported archive format (use .tar.gz, .tgz, or .zip)."
        )


def _flatten_single_root(dest: Path) -> None:
    """If dest holds exactly one directory, hoist its children up one level.

    Archives usually wrap the checkpoint in a single root folder; vLLM needs
    config.json at the model root.
    """
    children = [c for c in dest.iterdir()]
    if len(children) != 1 or not children[0].is_dir():
        return
    root = children[0]
    for child in list(root.iterdir()):
        shutil.move(str(child), str(dest / child.name))
    root.rmdir()


def _check_disk_for(total_bytes: int, factor: float = 1.0) -> None:
    """Refuse a transfer that would not leave _DISK_SAFETY_BYTES free."""
    if total_bytes <= 0:
        return
    free = shutil.disk_usage(_models_dir()).free
    needed = int(total_bytes * factor) + _DISK_SAFETY_BYTES
    if free < needed:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough disk space: transfer needs ~{needed / 1024**3:.1f} GB "
                f"(incl. safety margin) but only {free / 1024**3:.1f} GB are free."
            ),
        )


def _finalize_model(staging: Path, name: str, flatten: bool) -> dict[str, object]:
    """Move a staged model into place atomically and sanity-check it."""
    if flatten:
        _flatten_single_root(staging)
    target = _models_dir() / name
    if target.exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(
            status_code=409,
            detail=f"A local model named '{name}' already exists; delete it first.",
        )
    staging.rename(target)
    warnings: list[str] = []
    entries = list(target.iterdir())
    if not entries:
        warnings.append("The model directory is empty.")
    else:
        names = {e.name for e in entries}
        has_weights = any(
            e.suffix in (".safetensors", ".gguf", ".bin", ".pt") for e in entries
        )
        if "config.json" not in names and not has_weights:
            warnings.append(
                "No config.json or weight files found at the model root; "
                "vLLM may not be able to serve this directory."
            )
    return {
        "name": name,
        "path": str(target),
        "size_mb": round(_dir_size_bytes(target) / (1024 * 1024)),
        "warnings": warnings,
    }


def _local_model_in_use(path: str) -> bool:
    active = _active_model_names()
    if path in active:
        return True
    resolved = str(Path(path).resolve())
    return any(str(Path(a).resolve()) == resolved for a in active if a.startswith("/"))


def _gc_stale_partials() -> None:
    """Remove leftover staging data; sessions are in-memory, so anything in
    _MODELS_TMP after a restart is stale."""
    if _MODELS_TMP.exists():
        for entry in _MODELS_TMP.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def _sweep_expired_sessions() -> None:
    now = time.monotonic()
    for sid, session in list(_upload_sessions.items()):
        if now - float(session.get("last_activity", now)) > _UPLOAD_SESSION_TTL:
            _upload_sessions.pop(sid, None)
            shutil.rmtree(session["staging"], ignore_errors=True)  # type: ignore[arg-type]


def _lora_args_to_cli(lora_modules: list[dict[str, str]] | None) -> list[str]:
    """Translate LoRA adapter specs to vLLM CLI tokens.

    Host paths are validated against the model-dir allowlist; anything else
    is passed through as a HF hub adapter id.
    """
    if not lora_modules:
        return []
    tokens = ["--enable-lora", "--lora-modules"]
    seen_names: set[str] = set()
    for module in lora_modules:
        name = str(module.get("name") or "").strip()
        path = str(module.get("path") or "").strip()
        if not name or not path:
            raise HTTPException(
                status_code=400, detail="Each LoRA module needs both a name and a path."
            )
        if name in seen_names:
            raise HTTPException(
                status_code=400, detail=f"Duplicate LoRA module name '{name}'."
            )
        seen_names.add(name)
        if path.startswith("/") or path.startswith("~"):
            path = _validate_host_path(os.path.expanduser(path), what=f"LoRA '{name}'")
        tokens.append(f"{name}={path}")
    return tokens


# Ordered failure signatures matched against the tail of a deployment's log
# buffer. First match wins, so put the most specific patterns first.
_FAILURE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "gpu_oom",
        re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory", re.I),
        "GPU ran out of memory. Lower --gpu-memory-utilization, reduce "
        "max-model-len, pick a smaller model, or use a GPU with more memory.",
    ),
    (
        "kv_cache_too_small",
        re.compile(
            r"No available memory for the cache blocks"
            r"|max seq len .* is larger than the maximum number of tokens that can be stored in KV cache",
            re.I,
        ),
        "Not enough GPU memory left for the KV cache. Lower max-model-len or "
        "raise gpu-memory-utilization.",
    ),
    (
        "hf_auth",
        re.compile(
            r"401 Client Error|403 Client Error|GatedRepoError|gated repo"
            r"|Access to model .* is restricted|Cannot access gated repo",
            re.I,
        ),
        "Hugging Face denied access (gated or private repo). Provide a valid "
        "HF_TOKEN env var with access to this model.",
    ),
    (
        "model_not_found",
        re.compile(
            r"RepositoryNotFoundError|Repository Not Found"
            r"|does not appear to have a file named",
            re.I,
        ),
        "Model not found on Hugging Face. Check the model name for typos.",
    ),
    (
        "port_conflict",
        re.compile(r"address already in use", re.I),
        "The port is already in use on this node. Pick a different port.",
    ),
    (
        "bad_args",
        re.compile(r"unrecognized arguments|error: argument|No module named", re.I),
        "vLLM rejected the launch arguments. Check the extra args and engine "
        "options for typos or unsupported flags.",
    ),
]


def _classify_failure(lines: list[str]) -> tuple[str, str] | None:
    """Match known failure signatures against the last ~300 log lines."""
    tail = "\n".join(lines[-300:])
    for code, pattern, message in _FAILURE_PATTERNS:
        if pattern.search(tail):
            return code, message
    return None


# Cheap substring markers that track vLLM's startup phase, so the dashboard
# can show progress during the (potentially long) "loading" state.
def _phase_for_line(line: str) -> str | None:
    if "Application startup complete" in line or "Uvicorn running on" in line:
        return "ready"
    if "Capturing CUDA graph" in line or "torch.compile" in line or "Graph capturing finished" in line:
        return "compiling"
    if "Loading weights" in line or "checkpoint shards" in line or "Loading safetensors" in line:
        return "loading_weights"
    if "Downloading" in line:
        return "downloading"
    return None


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------

_RELEASE_RE = re.compile(r"^\d+\.\d+(\.\d+)?.*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Docker Hub tags that move over time. They are re-pulled on every deploy so a
# node never serves a stale build for them (immutable tags like v0.8.5 or a
# pinned nightly-{commit} stay cached).
_MUTABLE_TAGS = frozenset({"nightly", "latest"})


def _image_tag(image_ref: str) -> str:
    """Return the tag portion of an image reference (after the rightmost ':')."""
    return image_ref.rpartition(":")[2]


def _resolve_image_tag(version: str | None) -> tuple[str, str]:
    """Map a requested vLLM version to an official image ref + resolved version.

    Returns ``(image_ref, resolved_version)`` where ``resolved_version`` is the
    string stored in the database for display.

    | input              | image tag           |
    | ------------------ | ------------------- |
    | blank / None       | v{latest release}, else :latest |
    | ``0.8.5``          | v0.8.5              |
    | ``nightly``        | nightly             |
    | 40-char commit     | nightly-{commit}    |
    | anything else      | used as a literal tag |
    """
    repo = settings.vllm_image_repo
    raw = (version or "").strip()

    if not raw:
        resolved = _get_latest_vllm_version()
        if resolved:
            return f"{repo}:v{resolved}", resolved
        # GitHub was unreachable / rate-limited, so we couldn't resolve a
        # concrete version. Fall back to the moving ``latest`` tag, which Docker
        # Hub keeps pointed at the newest stable release (and which we re-pull on
        # every deploy). This avoids hard-failing when the node has no GitHub access.
        logger.warning(
            "Latest vLLM version unresolved (GitHub unreachable); using the "
            "'%s:latest' image tag.",
            repo,
        )
        return f"{repo}:latest", "latest"
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


class _PullTracker:
    """Aggregate docker pull progress chunks into overall (downloaded, total).

    Docker streams per-layer chunks like {"id": <layer>, "status": "Downloading",
    "progressDetail": {"current": n, "total": m}}. Totals grow as layers are
    enumerated, so the percentage converging upward early is expected. Layers
    that "Already exist" never report byte totals and simply stay out of the
    sums, so progress reflects actual transfer.
    """

    def __init__(self) -> None:
        self._layers: dict[str, tuple[int, int]] = {}

    def update(self, chunk: dict) -> tuple[int, int]:
        layer = chunk.get("id")
        status = chunk.get("status", "")
        if layer:
            detail = chunk.get("progressDetail") or {}
            total = int(detail.get("total") or 0)
            current = int(detail.get("current") or 0)
            if status == "Downloading" and total > 0:
                self._layers[layer] = (current, total)
            elif status in ("Download complete", "Pull complete") and layer in self._layers:
                _, known_total = self._layers[layer]
                self._layers[layer] = (known_total, known_total)
        downloaded = sum(current for current, _ in self._layers.values())
        total = sum(total for _, total in self._layers.values())
        return downloaded, total


def _pull_image(client: "docker.DockerClient", image_ref: str, log, progress_cb=None) -> None:
    """Pull *image_ref*, streaming layer progress into the deployment log.

    progress_cb, when given, receives throttled (downloaded_bytes, total_bytes)
    aggregates across all layers.
    """
    repository, _, tag = image_ref.partition(":")
    tag = tag or "latest"
    log(f"[docker] Pulling {image_ref} ...")
    seen: dict[str, str] = {}
    tracker = _PullTracker()
    last_report = (0.0, -1.0)  # (monotonic ts, percent)
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
            if progress_cb is not None:
                downloaded, total = tracker.update(chunk)
                if total > 0:
                    percent = downloaded / total * 100
                    now = time.monotonic()
                    if now - last_report[0] >= 1.0 or percent - last_report[1] >= 1.0:
                        last_report = (now, percent)
                        progress_cb(downloaded, total)
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

        dockerfile_lines = [f"FROM {base_ref}", f'LABEL "{_LABEL_DERIVED}"="true"']
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
    image_ref: str,
    extra_packages: list[str] | None,
    log,
    progress_cb=None,
    runtime: str = "docker",
) -> str:
    """Ensure the runtime image exists locally, returning the ref to run.

    When *extra_packages* are requested, returns a cached derived image tag.
    """
    client = _runtime_client(runtime)
    mutable = _image_tag(image_ref) in _MUTABLE_TAGS

    # Base image. Moving tags (nightly/latest) are always re-pulled so the node
    # never runs a stale build; Docker only downloads changed layers, so this is
    # cheap when nothing upstream changed. Immutable tags reuse the local cache.
    if mutable:
        log(f"[docker] '{image_ref}' is a moving tag — checking for a newer build ...")
        _pull_image(client, image_ref, log, progress_cb)
    else:
        try:
            client.images.get(image_ref)
            log(f"[docker] Using cached image {image_ref}")
        except ImageNotFound:
            _pull_image(client, image_ref, log, progress_cb)

    if not extra_packages:
        return image_ref

    derived_tag = _derived_image_tag(image_ref, extra_packages)
    # When the base is a moving tag, rebuild the derived image too — its cached
    # copy may sit on top of a now-outdated base. Immutable bases reuse the cache.
    if not mutable:
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


def _image_digest(container) -> str | None:
    """Exact image identity for provenance (repo@sha256 when available).

    Locally derived images (extra_packages builds) have no RepoDigests, so
    fall back to the content-addressed image id.
    """
    try:
        image = container.image
        digests = list(image.attrs.get("RepoDigests") or [])
        if digests:
            return str(digests[0])
        return str(image.id)
    except Exception:  # pragma: no cover - image removed out from under container
        return None


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


def _check_gpu_resources(payload: "StartRequest") -> str | None:
    """Return a rejection reason if the request can't fit on the target GPUs.

    Two checks per target GPU: enough free memory right now for the requested
    reservation (fraction x total), and no over-commit past 1.0 cumulative
    fraction across deployments this agent already tracks. Skipped when only
    unified-memory metrics are available (no per-GPU data).
    """
    gpus = _gpu_metrics()
    if not gpus or any(g.get("source") == "unified" for g in gpus):
        return None
    by_index = {g["index"]: g for g in gpus}
    target_ids = payload.gpu_ids or sorted(by_index)

    reserved: dict[int, float] = {i: 0.0 for i in by_index}
    for key, meta in _statuses.items():
        if key not in _containers:
            continue
        frac = float(meta.get("gpu_memory_fraction") or 0.0)
        ids = meta.get("gpu_ids") or sorted(by_index)
        for i in ids:
            if i in reserved:
                reserved[i] += frac

    problems: list[str] = []
    for gpu_id in target_ids:
        gpu = by_index.get(gpu_id)
        if gpu is None:
            continue  # invalid ids are rejected separately
        total = float(gpu.get("memory_total_mb") or 0)
        used = float(gpu.get("memory_used_mb") or 0)
        if total <= 0:
            continue
        needed = payload.gpu_memory_fraction * total
        free = total - used
        if needed > free:
            problems.append(
                f"GPU {gpu_id} has {free / 1024:.1f} GiB free but the deployment "
                f"reserves {payload.gpu_memory_fraction:.2f} x {total / 1024:.1f} GiB "
                f"= {needed / 1024:.1f} GiB"
            )
        cumulative = reserved.get(gpu_id, 0.0) + payload.gpu_memory_fraction
        if cumulative > 1.0:
            problems.append(
                f"GPU {gpu_id} would be over-committed: tracked deployments already "
                f"reserve {reserved[gpu_id]:.2f} of its memory and this one adds "
                f"{payload.gpu_memory_fraction:.2f} (total {cumulative:.2f} > 1.0)"
            )
    if problems:
        return (
            "; ".join(problems)
            + ". Lower the GPU memory fraction, choose different GPUs, stop "
            "another deployment, or pass skip_resource_check to override."
        )
    return None


def _launch_manifest(
    payload: "StartRequest", container_runtime: str | None = None
) -> dict[str, object]:
    """Launch metadata stored as a container label for host re-adoption.

    Everything needed to reconstruct the deployment row (and to re-launch with
    identical config) except fields the client already labels separately
    (key/port/version). env_vars are stored verbatim — they are already exposed
    on the container's Docker Env, so the label adds no new surface.
    """
    return {
        "version": 1,
        "owner": payload.owner,
        "duration_seconds": payload.duration_seconds,
        "expires_at": payload.expires_at,
        "extra_args": payload.extra_args or [],
        "env_vars": payload.env_vars or [],
        "engine_args": payload.engine_args or {},
        "lora_modules": payload.lora_modules or [],
        "extra_packages": payload.extra_packages or [],
        "gpu_memory_fraction": payload.gpu_memory_fraction,
        "gpu_ids": payload.gpu_ids or [],
        "tensor_parallel_size": payload.tensor_parallel_size,
        "max_failed_restarts": payload.max_failed_restarts,
        "container_runtime": container_runtime or payload.container_runtime,
    }


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
    # Allowed model dirs mount read-only at the SAME path inside the
    # container, so local model / LoRA paths need no rewriting.
    for model_dir in _allowed_model_dirs():
        volumes[str(model_dir)] = {"bind": str(model_dir), "mode": "ro"}
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
        "disk": _disk_metrics(),
        "available_runtimes": _available_runtimes(),
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


@app.get("/deployments/logs/download")
def download_deployment_logs(key: str) -> FileResponse:
    """Full persisted log of the deployment's current run."""
    path = _log_file_for(key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No log file for this deployment.")
    return FileResponse(path, media_type="text/plain", filename=path.name)


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

    # Local checkpoints must live inside an allowed model dir.
    if payload.model_name.startswith("/"):
        _validate_host_path(payload.model_name)
    # Validate LoRA specs up front (fails fast, before any image pull).
    lora_args = _lora_args_to_cli(payload.lora_modules)

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

    # Reject launches that demonstrably can't fit before burning crash-loop
    # restarts on them.
    if not payload.skip_resource_check:
        reason = _check_gpu_resources(payload)
        if reason:
            raise HTTPException(status_code=409, detail=reason)

    _logs[key] = deque(maxlen=2000)
    # Fresh run, fresh file (the previous run is kept as "<file>.1").
    _open_log_file(key, rotate=True)

    def _log(msg: str) -> None:
        logger.info(msg)
        _append_agent_log(key, msg)
        # Surface the derived-image build in the reported phase (builds have
        # no byte totals, unlike pulls).
        if msg.startswith("[docker] Building image") and key in _statuses:
            _statuses[key]["phase"] = "building image"
            _statuses[key].pop("pull_progress", None)

    try:
        image_ref, resolved_version = _resolve_image_tag(payload.vllm_version)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Resolve which container runtime runs this deployment: the host's pick
    # when valid, otherwise the first available one (old hosts send nothing).
    available_runtimes = _available_runtimes()
    runtime = payload.container_runtime or (
        available_runtimes[0] if available_runtimes else None
    )
    if runtime not in available_runtimes:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Container runtime '{payload.container_runtime or 'any'}' is not "
                f"available on this node (available: "
                f"{', '.join(available_runtimes) or 'none'})."
            ),
        )

    manifest = _launch_manifest(payload, runtime)
    # Provisional status so the host (and UI) can see image pull/build progress
    # during the potentially very long _ensure_image call. Replaced by the full
    # entry once the container starts; removed again on any failure before that.
    _statuses[key] = {
        "model_name": payload.model_name,
        "port": payload.port,
        "status": "starting",
        "phase": "preparing image",
        "desired_state": "running",
        "launch_manifest": manifest,
        "container_runtime": runtime,
    }

    def _pull_progress(downloaded: int, total: int) -> None:
        status = _statuses.get(key)
        if status is None:
            return
        status["phase"] = (
            "extracting image" if downloaded >= total > 0 else "pulling image"
        )
        status["pull_progress"] = {
            "downloaded_mb": round(downloaded / (1024 * 1024)),
            "total_mb": round(total / (1024 * 1024)),
            "percent": round(downloaded / total * 100, 1) if total else 0.0,
        }

    try:
        run_image = await asyncio.to_thread(
            _ensure_image, image_ref, payload.extra_packages, _log, _pull_progress, runtime
        )
    except RuntimeError as exc:
        _statuses.pop(key, None)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # Pull/build done; drop transfer progress but keep reporting "starting".
    _statuses[key].pop("pull_progress", None)
    _statuses[key]["phase"] = "starting container"

    try:
        command = [
            "--model", payload.model_name,
            "--port", str(payload.port),
            "--gpu-memory-utilization", str(payload.gpu_memory_fraction),
        ]
        if payload.tensor_parallel_size:
            command.extend(["--tensor-parallel-size", str(payload.tensor_parallel_size)])
        command.extend(_engine_args_to_cli(payload.engine_args))
        command.extend(lora_args)
        if payload.extra_args:
            command.extend(_rewrite_paths_for_container(payload.extra_args))

        environment = _build_environment(payload)
        name = _container_name(key)
        labels = {
            _LABEL_MANAGED: "true",
            _LABEL_KEY: key,
            _LABEL_PORT: str(payload.port),
            _LABEL_VERSION: resolved_version,
            _LABEL_RUNTIME: runtime,
            _LABEL_LAUNCH: json.dumps(manifest, separators=(",", ":")),
        }

        logger.info("Starting deployment %s via %s", key, runtime)
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
            runtime,
        )
    except RuntimeError as exc:
        _statuses.pop(key, None)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        # Validation failures after the provisional entry must not leave a
        # ghost "starting" deployment in the status report.
        _statuses.pop(key, None)
        raise

    _containers[key] = container
    _statuses[key] = {
        "model_name": payload.model_name,
        "port": payload.port,
        "gpu_memory_fraction": payload.gpu_memory_fraction,
        "gpu_ids": payload.gpu_ids or [],
        "tensor_parallel_size": payload.tensor_parallel_size,
        "vllm_version": resolved_version,
        "image": run_image,
        "image_digest": _image_digest(container),
        "container_id": container.id,
        "status": "loading",
        "phase": None,
        "engine_args": payload.engine_args or {},
        "lora_modules": payload.lora_modules or [],
        "max_failed_restarts": payload.max_failed_restarts,
        "desired_state": "running",
        "launch_manifest": manifest,
        "container_runtime": runtime,
    }
    _log(f"[docker] Started container {name} ({runtime})")
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
    runtime: str = "docker",
) -> "docker.models.containers.Container":
    client = _runtime_client(runtime)
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
    # The log file stays on disk (age-based GC); just release the handle.
    _close_log_file(key)
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


def _restart_threshold(key: str) -> int:
    """Crash-loop breaker threshold: per-deployment override or the default.

    A deployment that keeps restarting without ever becoming ready is
    mis-configured (bad args, GPU OOM, missing token, ...). After this many
    failed restarts the agent stops it so it doesn't loop forever under the
    unless-stopped policy — which otherwise still recovers healthy deployments
    across transient crashes and host reboots.
    """
    override = _statuses.get(key, {}).get("max_failed_restarts")
    if isinstance(override, int) and override > 0:
        return override
    return settings.max_failed_restarts


def _record_failure_cause(key: str) -> str | None:
    """Classify the failure from the log tail and store it on the status."""
    classified = _classify_failure(list(_logs.get(key) or []))
    if classified is None:
        return None
    code, message = classified
    if key in _statuses:
        _statuses[key]["error_code"] = code
        _statuses[key]["error"] = message
    return message


async def _monitor_container(
    key: str, container: "docker.models.containers.Container"
) -> None:
    """Track container state; flip the deployment to running once /health is up.

    Containers use the unless-stopped restart policy, so a non-zero exit is NOT
    terminal — Docker restarts it. A deployment is therefore considered finished
    only when (a) the user stopped it, (b) the container disappeared, or (c) it
    crash-loops without ever becoming ready (the breaker below stops it).
    """
    import time

    port = _statuses.get(key, {}).get("port")
    last_restart_count = 0
    last_usage_scrape = 0.0
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
            _append_agent_log(
                key,
                f"[docker] Container restarted (exit code {state.get('ExitCode')}); Docker is retrying.",
            )
            if key in _statuses:
                _statuses[key]["status"] = "error"
            # Surface the root cause (OOM, gated repo, bad args, ...) instead
            # of leaving the operator to dig through the log buffer.
            cause = _record_failure_cause(key)
            if cause:
                _append_agent_log(key, f"[agent] Probable cause: {cause}")

            # Crash-loop breaker: a deployment that never became ready and keeps
            # restarting is mis-configured (bad args, GPU OOM, ...). Stop it so
            # unless-stopped does not retry forever; healthy deployments
            # (ever_ready) are left alone so transient crashes and reboots recover.
            if not ever_ready and restart_count >= _restart_threshold(key):
                logger.warning(
                    "Crash-loop breaker tripped for %s after %d failed restarts",
                    key, restart_count,
                )
                _append_agent_log(
                    key,
                    f"[docker] Deployment failed to start after {restart_count} attempts and "
                    "never became ready; stopping retries and removing the container. See the "
                    "log above for the root cause (e.g. GPU out of memory, bad args), then fix "
                    "it and redeploy. The vLLM image stays cached for a fast retry.",
                )
                if key in _statuses:
                    _statuses[key]["status"] = "error"
                    generic = (
                        f"Stopped after {restart_count} failed starts without becoming ready."
                    )
                    _statuses[key]["error"] = (
                        f"{cause} ({generic})" if cause else generic
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
                _statuses[key]["phase"] = "ready"
            # Token accounting: pick up vLLM's cumulative counters every ~15s.
            if isinstance(port, int) and time.monotonic() - last_usage_scrape > 15:
                last_usage_scrape = time.monotonic()
                usage = await _scrape_vllm_metrics(port, key=key)
                if usage and key in _statuses:
                    _statuses[key]["usage"] = usage
            # Keep monitoring for exits/crash loops.
            continue


# vLLM's own Prometheus counters (cumulative since container start). Label
# sets vary by engine version; values are summed across them. These three
# names are stable across the V0 and V1 engines.
_VLLM_COUNTER_RE = re.compile(
    r"^vllm:(prompt_tokens_total|generation_tokens_total|request_success_total)"
    r"(?:\{[^}]*\})?\s+(\S+)\s*$"
)

# Live queue-depth gauges (best effort: present on recent vLLM versions).
_VLLM_GAUGE_RE = re.compile(
    r"^vllm:(num_requests_running|num_requests_waiting)"
    r"(?:\{[^}]*\})?\s+(\S+)\s*$"
)

# Per-request processing-time histograms. prefill/decode time (V1 engines) are
# the precise sources for idle-free read/generation speeds; TTFT/TPOT are the
# fallback on older engines (TTFT includes queue wait — a lower bound).
_VLLM_HIST_RE = re.compile(
    r"^vllm:(request_prefill_time_seconds|request_decode_time_seconds"
    r"|time_to_first_token_seconds|time_per_output_token_seconds)"
    r"_(sum|count)"
    r"(?:\{[^}]*\})?\s+(\S+)\s*$"
)

# Last usage snapshot per deployment key (counters + histogram sums + "_ts"),
# to derive rates between consecutive scrapes.
_usage_snapshots: dict[str, dict[str, float]] = {}


def _compute_usage_rates(key: str, snapshot: dict[str, float]) -> dict[str, float]:
    """Token speeds split read/generation.

    Per-request speeds are *running averages since container start*: cumulative
    token counters divided by cumulative processing-time sums from vLLM's
    per-request histograms. Idle time never enters the denominator, and once a
    request has completed the averages stay defined forever (they never drop to
    zero or disappear between bursts). Throughput keys are the engine-wide view
    over the last scrape window (wall clock) and are only present for windows
    with activity.
    """
    rates: dict[str, float] = {}

    def _rate(numerator: float, denominator: float) -> float | None:
        if numerator <= 0 or denominator <= 0:
            return None
        return round(numerator / denominator, 1)

    # Lifetime per-request averages: primary prefill/decode time sums,
    # TTFT/TPOT fallback for engines that don't expose them.
    prompt_tps = _rate(snapshot["prompt_tokens"], snapshot.get("prefill_sum", 0.0))
    if prompt_tps is None:
        prompt_tps = _rate(snapshot["prompt_tokens"], snapshot.get("ttft_sum", 0.0))
    if prompt_tps is not None:
        rates["prompt_tps"] = prompt_tps

    generation_tps = _rate(
        snapshot["generation_tokens"], snapshot.get("decode_sum", 0.0)
    )
    if generation_tps is None:
        generation_tps = _rate(
            snapshot.get("tpot_count", 0.0), snapshot.get("tpot_sum", 0.0)
        )
    if generation_tps is not None:
        rates["generation_tps"] = generation_tps

    # Engine-wide throughput over the last wall-clock window (delta-based;
    # skipped on the first sample and after a counter reset).
    snapshot = dict(snapshot)
    snapshot["_ts"] = time.monotonic()
    previous = _usage_snapshots.get(key)
    _usage_snapshots[key] = snapshot
    if previous is not None:
        deltas = {k: snapshot.get(k, 0.0) - previous.get(k, 0.0) for k in snapshot}
        if all(value >= 0 for value in deltas.values()):
            wall = deltas["_ts"]
            throughput = _rate(deltas["prompt_tokens"], wall)
            if throughput is not None:
                rates["prompt_throughput"] = throughput
            throughput = _rate(deltas["generation_tokens"], wall)
            if throughput is not None:
                rates["generation_throughput"] = throughput
    return rates


async def _scrape_vllm_metrics(port: int, key: str | None = None) -> dict[str, object] | None:
    """Read token/request counters and queue gauges from vLLM's /metrics.

    Returns None when unreachable or when no vLLM counter matched (so zeros
    are never written spuriously, e.g. against an older image).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://127.0.0.1:{port}/metrics")
            response.raise_for_status()
            text = response.text
    except Exception:
        return None

    totals = {
        "prompt_tokens_total": 0.0,
        "generation_tokens_total": 0.0,
        "request_success_total": 0.0,
    }
    gauges: dict[str, float] = {}
    hist: dict[str, float] = {}
    matched = False
    for line in text.splitlines():
        match = _VLLM_COUNTER_RE.match(line)
        if match:
            try:
                totals[match.group(1)] += float(match.group(2))
            except ValueError:
                continue
            matched = True
            continue
        match = _VLLM_GAUGE_RE.match(line)
        if match:
            try:
                gauges[match.group(1)] = gauges.get(match.group(1), 0.0) + float(
                    match.group(2)
                )
            except ValueError:
                continue
            continue
        match = _VLLM_HIST_RE.match(line)
        if match:
            hist_key = f"{match.group(1)}_{match.group(2)}"
            try:
                hist[hist_key] = hist.get(hist_key, 0.0) + float(match.group(3))
            except ValueError:
                continue
    if not matched:
        return None

    usage: dict[str, object] = {
        "prompt_tokens": int(totals["prompt_tokens_total"]),
        "generation_tokens": int(totals["generation_tokens_total"]),
        "requests": int(totals["request_success_total"]),
    }
    if "num_requests_running" in gauges:
        usage["requests_running"] = int(gauges["num_requests_running"])
    if "num_requests_waiting" in gauges:
        usage["requests_waiting"] = int(gauges["num_requests_waiting"])
    if key is not None:
        snapshot = {
            "prompt_tokens": totals["prompt_tokens_total"],
            "generation_tokens": totals["generation_tokens_total"],
            "prefill_sum": hist.get("request_prefill_time_seconds_sum", 0.0),
            "decode_sum": hist.get("request_decode_time_seconds_sum", 0.0),
            "ttft_sum": hist.get("time_to_first_token_seconds_sum", 0.0),
            "tpot_sum": hist.get("time_per_output_token_seconds_sum", 0.0),
            "tpot_count": hist.get("time_per_output_token_seconds_count", 0.0),
        }
        usage.update(_compute_usage_rates(key, snapshot))
    return usage


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
    key: str,
    container: "docker.models.containers.Container",
    resume: bool = False,
) -> None:
    def _reader() -> None:
        state = _open_log_file(key)
        last_ts = str(state.get("raw_ts") or "")
        kwargs: dict[str, object] = {"stream": True, "follow": True, "timestamps": True}
        if resume and last_ts:
            # Resume after an agent restart: ask Docker for logs since the
            # last persisted timestamp (second granularity) and drop the
            # overlap below via the exact nanosecond skip-guard.
            try:
                since_dt = datetime.fromisoformat(last_ts[:19] + "+00:00")
                kwargs["since"] = int(since_dt.timestamp())
            except ValueError:
                kwargs["tail"] = 200
        elif resume:
            # Legacy file without a sidecar: keep the old tail behavior.
            kwargs["tail"] = 200
        try:
            stream = container.logs(**kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Failed to attach to logs for %s: %s", key, exc)
            return
        for raw in stream:
            try:
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            except Exception:
                continue
            raw_ts, rest = _split_docker_ts(line)
            # RFC3339Nano timestamps are fixed-width -> lexicographic compare.
            if raw_ts and last_ts and raw_ts <= last_ts:
                continue
            if raw_ts:
                last_ts = raw_ts
            cleaned = _strip_ansi(rest)
            if not cleaned:
                continue
            phase = _phase_for_line(cleaned)
            if phase and key in _statuses:
                _statuses[key]["phase"] = phase
            if _is_noise_line(cleaned):
                continue
            _append_log_line(key, _format_log_line(raw_ts, cleaned), raw_ts)

    await asyncio.to_thread(_reader)


def _managed_containers(all_states: bool = True) -> list[tuple[str, object]]:
    """Union of managed containers across every available runtime, as
    (runtime, container) pairs. A node may hold containers in both runtimes
    (e.g. after switching its runtime), so enumeration must cover both."""
    found: list[tuple[str, object]] = []
    for runtime in _available_runtimes():
        try:
            containers = _runtime_client(runtime).containers.list(
                all=all_states, filters={"label": f"{_LABEL_MANAGED}=true"}
            )
        except Exception as exc:
            logger.warning("Listing %s containers failed: %s", runtime, exc)
            continue
        found.extend((runtime, container) for container in containers)
    return found


def _reconcile_containers() -> None:
    """Rebuild in-memory state from managed containers after an agent restart."""
    recovered = 0
    for runtime, container in _managed_containers():
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
        status: dict[str, object] = {
            "model_name": key.rsplit(":", 1)[0],
            "port": port,
            "vllm_version": labels.get(_LABEL_VERSION),
            "image": (container.image.tags[0] if container.image.tags else None),
            "image_digest": _image_digest(container),
            "container_id": container.id,
            "status": "loading",
            "desired_state": "running",
            "container_runtime": labels.get(_LABEL_RUNTIME, runtime),
        }
        # Restore the launch manifest so deployment config survives agent
        # restarts and the host can re-adopt with full metadata.
        raw_manifest = labels.get(_LABEL_LAUNCH)
        if raw_manifest:
            try:
                manifest = json.loads(raw_manifest)
            except (ValueError, TypeError):
                manifest = None
            if isinstance(manifest, dict):
                status["launch_manifest"] = manifest
                status["gpu_memory_fraction"] = manifest.get("gpu_memory_fraction")
                status["gpu_ids"] = manifest.get("gpu_ids") or []
                status["tensor_parallel_size"] = manifest.get("tensor_parallel_size")
                status["engine_args"] = manifest.get("engine_args") or {}
                status["lora_modules"] = manifest.get("lora_modules") or []
                status["max_failed_restarts"] = manifest.get("max_failed_restarts")
        _statuses[key] = status
        asyncio.create_task(_stream_container_logs(key, container, resume=True))
        asyncio.create_task(_monitor_container(key, container))
        recovered += 1
    if recovered:
        logger.info("Reconciled %d running deployment(s)", recovered)


# ---------------------------------------------------------------------------
# Container management endpoints
# ---------------------------------------------------------------------------


def _is_vllm_image(image_ref: str) -> bool:
    """True if an image reference is an official vLLM or locally-derived image."""
    return image_ref.startswith(settings.vllm_image_repo) or image_ref.startswith(
        _LOCAL_IMAGE_REPO
    )


def _container_image_ref(container: "docker.models.containers.Container") -> str:
    """Best-effort image tag/short-id for a container (image may have been pruned)."""
    try:
        image = container.image
        tags = list(image.tags or [])
        return tags[0] if tags else image.short_id
    except Exception:  # pragma: no cover - image removed out from under container
        return "<none>"


@app.get("/containers")
def list_containers() -> dict[str, list[dict[str, object]]]:
    """List vLLM-related containers on this node (managed + anything on a vLLM image).

    ``tracked`` marks a container the agent is actively monitoring as a live
    deployment; everything else is "rogue" (crash leftovers, orphans after a
    restart, or a manual ``docker run``) and safe to stop from the UI.
    """
    available = _available_runtimes()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No container runtime (Docker or Podman) is available on this node.",
        )
    containers: list[dict[str, object]] = []
    for runtime in available:
        try:
            runtime_containers = _runtime_client(runtime).containers.list(all=True)
        except Exception as exc:
            logger.warning("Listing %s containers failed: %s", runtime, exc)
            continue
        for container in runtime_containers:
            labels = container.labels or {}
            managed = labels.get(_LABEL_MANAGED) == "true"
            key = labels.get(_LABEL_KEY)
            image = _container_image_ref(container)
            if not (managed or _is_vllm_image(image)):
                continue
            containers.append({
                "id": container.short_id,
                "name": container.name,
                "image": image,
                "status": container.status,
                "managed": managed,
                "key": key,
                "tracked": bool(key) and key in _containers,
                "runtime": labels.get(_LABEL_RUNTIME, runtime),
            })
    return {"containers": containers}


@app.post("/containers/{container_id}/stop")
def stop_container(container_id: str) -> dict[str, str]:
    """Stop and remove a single (rogue) container by id."""
    container = None
    for runtime in _available_runtimes():
        try:
            container = _runtime_client(runtime).containers.get(container_id)
            break
        except NotFound:
            continue
        except Exception as exc:
            logger.warning("Container lookup via %s failed: %s", runtime, exc)
    if container is None:
        raise HTTPException(status_code=404, detail="Container not found")
    labels = container.labels or {}
    key = labels.get(_LABEL_KEY)
    if labels.get(_LABEL_MANAGED) == "true" and key and key in _containers:
        raise HTTPException(
            status_code=409,
            detail="Managed by an active deployment — stop it from the Deployments table.",
        )
    _remove_container(container)
    return {"status": "removed", "id": container_id}


# ---------------------------------------------------------------------------
# Image cache management endpoints
# ---------------------------------------------------------------------------


@app.get("/images")
def list_images() -> dict[str, list[dict[str, object]]]:
    """List cached vLLM images (official + locally derived) on this node."""
    images: list[dict[str, object]] = []
    available = _available_runtimes()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No container runtime (Docker or Podman) is available on this node.",
        )
    repo = settings.vllm_image_repo
    for runtime in available:
        try:
            runtime_images = _runtime_client(runtime).images.list()
        except Exception as exc:
            logger.warning("Listing %s images failed: %s", runtime, exc)
            continue
        for image in runtime_images:
            tags = list(image.tags or [])
            relevant = [t for t in tags if t.startswith(repo) or t.startswith(_LOCAL_IMAGE_REPO)]
            if not relevant:
                continue
            images.append({
                "id": image.short_id,
                "tags": relevant,
                "size_mb": round((image.attrs.get("Size", 0) or 0) / (1024 * 1024)),
                "runtime": runtime,
            })
    return {"images": images}


@app.delete("/images/{image_id}")
def delete_image(image_id: str, runtime: str | None = None) -> dict[str, str]:
    runtimes = [runtime] if runtime else _available_runtimes()
    for candidate in runtimes:
        try:
            _runtime_client(candidate).images.remove(image_id)
            return {"status": "deleted", "id": image_id}
        except ImageNotFound:
            continue
        except APIError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail="Image not found")


@app.post("/images/prune")
def prune_images() -> dict[str, object]:
    """Remove cached vLLM/derived images not backing any container.

    An image is considered in use if *any* container (running or stopped)
    references it, so pruning never yanks an image out from under a
    restartable deployment. Images Docker refuses to remove are reported in
    ``skipped`` rather than failing the whole request.
    """
    available = _available_runtimes()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No container runtime (Docker or Podman) is available on this node.",
        )

    removed: list[str] = []
    skipped: list[str] = []
    freed_mb = 0

    for runtime in available:
        try:
            client = _runtime_client(runtime)
            in_use: set[str] = set()
            for container in client.containers.list(all=True):
                try:
                    if container.image is not None:
                        in_use.add(container.image.id)
                except Exception:  # pragma: no cover - image already gone
                    continue

            def _try_remove(image) -> None:
                nonlocal freed_mb
                if image.id in in_use:
                    skipped.append(image.short_id)
                    return
                size_mb = round((image.attrs.get("Size", 0) or 0) / (1024 * 1024))
                try:
                    client.images.remove(image.id)
                    removed.append(image.short_id)
                    freed_mb += size_mb
                except APIError:
                    skipped.append(image.short_id)

            # Tagged vLLM / derived images.
            for image in client.images.list():
                tags = list(image.tags or [])
                if any(_is_vllm_image(t) for t in tags):
                    _try_remove(image)

            # Dangling (<none>) derived images left behind when a moving-tag
            # base (nightly/latest) was rebuilt on a newer base. The build
            # label scopes this to images we created, so unrelated dangling
            # images are never touched.
            for image in client.images.list(
                filters={"dangling": True, "label": f"{_LABEL_DERIVED}=true"}
            ):
                _try_remove(image)
        except Exception as exc:
            logger.warning("Image prune via %s failed: %s", runtime, exc)
            continue

    return {"removed": removed, "freed_mb": freed_mb, "skipped": skipped}


# ---------------------------------------------------------------------------
# Disk & HuggingFace model cache management
# ---------------------------------------------------------------------------


def _hf_cache_path() -> Path:
    return Path(os.path.expanduser(settings.hf_cache_dir)).resolve()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


# The HF cache can hold hundreds of GB; walking it is not free, so the size
# is cached for a few minutes.
_hf_cache_size: tuple[float, float] = (0.0, -1.0)  # (monotonic ts, gb)
_HF_CACHE_SIZE_TTL = 300.0


def _hf_cache_size_gb() -> float:
    import time

    global _hf_cache_size
    ts, cached = _hf_cache_size
    now = time.monotonic()
    if cached >= 0 and now - ts < _HF_CACHE_SIZE_TTL:
        return cached
    cache_dir = _hf_cache_path()
    size_gb = (
        round(_dir_size_bytes(cache_dir) / (1024**3), 1) if cache_dir.exists() else 0.0
    )
    _hf_cache_size = (now, size_gb)
    return size_gb


def _disk_metrics() -> dict[str, float] | None:
    try:
        cache_dir = _hf_cache_path()
        anchor = cache_dir if cache_dir.exists() else Path.home()
        usage = shutil.disk_usage(anchor)
        return {
            "total_gb": round(usage.total / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
            "hf_cache_gb": _hf_cache_size_gb(),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Disk metrics unavailable: %s", exc)
        return None


def _cached_model_name(dir_name: str) -> str:
    """Decode a hub cache dir name (models--org--name) to org/name."""
    return dir_name[len("models--"):].replace("--", "/")


def _active_model_names() -> set[str]:
    return {
        str(meta.get("model_name"))
        for key, meta in _statuses.items()
        if key in _containers
    }


@app.get("/models/cache")
def list_model_cache() -> dict[str, list[dict[str, object]]]:
    """List models in the shared HuggingFace hub cache with their sizes."""
    hub = _hf_cache_path() / "hub"
    models: list[dict[str, object]] = []
    active = _active_model_names()
    if hub.exists():
        for entry in sorted(hub.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            name = _cached_model_name(entry.name)
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                mtime = 0
            models.append(
                {
                    "name": name,
                    "size_mb": round(_dir_size_bytes(entry) / (1024 * 1024)),
                    "last_used_at": mtime,
                    "in_use": name in active,
                }
            )
    return {"models": models}


# Minimal image for the root-deletion fallback below; pulled on first use.
_CLEANUP_IMAGE = "alpine:3"


def _delete_cache_dirs_via_docker(hub: Path, dir_name: str) -> None:
    """Remove a hub cache dir (and its .locks twin) as root via a one-shot container.

    vLLM containers download models as root onto the bind-mounted cache, so when
    the agent runs as a regular user it cannot rmtree them natively — but a root
    container over the same mount can. Uses whichever runtime is available
    (under rootless Podman the files are agent-owned, so the native rmtree
    normally succeeds before this fallback is reached).
    """
    available = _available_runtimes()
    client = _runtime_client(available[0]) if available else _docker()
    client.containers.run(
        _CLEANUP_IMAGE,
        ["rm", "-rf", f"/hub/{dir_name}", f"/hub/.locks/{dir_name}"],
        volumes={str(hub): {"bind": "/hub", "mode": "rw"}},
        remove=True,
    )


@app.delete("/models/cache/{name:path}")
def delete_cached_model(name: str) -> dict[str, str]:
    """Delete one model from the hub cache (refused while a deployment serves it)."""
    if name in _active_model_names():
        raise HTTPException(
            status_code=409,
            detail=f"Model '{name}' is served by an active deployment; stop it first.",
        )
    hub = (_hf_cache_path() / "hub").resolve()
    dir_name = f"models--{name.replace('/', '--')}"
    target = (hub / dir_name).resolve()
    # Path traversal guard: the target must stay inside the hub cache.
    if hub not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Model not found in cache.")
    locks = hub / ".locks" / dir_name
    try:
        shutil.rmtree(target)
        if locks.exists():
            shutil.rmtree(locks)
    except OSError as exc:
        logger.warning(
            "Native delete of %s failed (%s); retrying as root via Docker.", target, exc
        )
        try:
            _delete_cache_dirs_via_docker(hub, dir_name)
        except Exception as docker_exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not delete '{name}': {exc}. "
                    f"Docker fallback also failed: {docker_exc}"
                ),
            ) from docker_exc
    if target.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Deleting '{name}' did not remove {target}; "
                "check the agent's filesystem permissions."
            ),
        )
    logger.info("Deleted model '%s' from the hub cache.", name)
    # Invalidate the cached cache-size so /metrics reflects the deletion.
    global _hf_cache_size
    _hf_cache_size = (0.0, -1.0)
    return {"status": "deleted", "name": name}


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


# ---------------------------------------------------------------------------
# Managed local models (upload from the host UI / pull from a URL)
# ---------------------------------------------------------------------------


class LocalModelUploadBegin(BaseModel):
    name: str
    total_bytes: int = 0
    file_count: int = 0


class LocalModelUploadFinish(BaseModel):
    flatten: bool = False


class LocalModelPull(BaseModel):
    url: str
    name: str | None = None


def _new_staging(transfer_id: str) -> Path:
    staging = _MODELS_TMP / f".partial-{transfer_id}"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


async def _stream_body_to_file(request: Request, target: Path, on_chunk=None) -> int:
    """Write the raw request body to target in chunks (no full-file buffering).

    Writes run in a thread so the event loop keeps serving heartbeats during
    multi-GB transfers to slow disks.
    """
    written = 0
    with open(target, "wb") as fh:
        async for chunk in request.stream():
            if not chunk:
                continue
            await asyncio.to_thread(fh.write, chunk)
            written += len(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    return written


@app.post("/local-models/upload/begin")
def begin_local_model_upload(payload: LocalModelUploadBegin) -> dict[str, str]:
    name = _sanitize_model_name(payload.name)
    _sweep_expired_sessions()
    if (_models_dir() / name).exists():
        raise HTTPException(
            status_code=409,
            detail=f"A local model named '{name}' already exists; delete it first.",
        )
    _check_disk_for(payload.total_bytes)
    session_id = uuid.uuid4().hex
    _upload_sessions[session_id] = {
        "name": name,
        "staging": _new_staging(session_id),
        "total_bytes": payload.total_bytes,
        "received_bytes": 0,
        "file_count": payload.file_count,
        "files_done": 0,
        "last_activity": time.monotonic(),
    }
    return {"session_id": session_id}


@app.put("/local-models/upload/{session_id}/file")
async def upload_local_model_file(
    session_id: str, path: str, request: Request
) -> dict[str, object]:
    session = _upload_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown upload session.")
    rel = _validate_rel_path(path)
    staging: Path = session["staging"]  # type: ignore[assignment]
    target = staging / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    written = await _stream_body_to_file(request, target)
    session["received_bytes"] = int(session.get("received_bytes", 0)) + written  # type: ignore[arg-type]
    session["files_done"] = int(session.get("files_done", 0)) + 1  # type: ignore[arg-type]
    session["last_activity"] = time.monotonic()
    return {"status": "ok", "received_bytes": session["received_bytes"]}


@app.post("/local-models/upload/{session_id}/finish")
def finish_local_model_upload(
    session_id: str, payload: LocalModelUploadFinish | None = None
) -> dict[str, object]:
    session = _upload_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown upload session.")
    flatten = bool(payload.flatten) if payload else False
    result = _finalize_model(session["staging"], str(session["name"]), flatten)  # type: ignore[arg-type]
    logger.info("Local model '%s' uploaded (%s MB)", result["name"], result["size_mb"])
    return result


@app.post("/local-models/upload/{session_id}/abort")
def abort_local_model_upload(session_id: str) -> dict[str, str]:
    session = _upload_sessions.pop(session_id, None)
    if session is not None:
        shutil.rmtree(session["staging"], ignore_errors=True)  # type: ignore[arg-type]
    return {"status": "aborted"}


@app.post("/local-models/archive")
async def upload_local_model_archive(
    name: str, filename: str, request: Request
) -> dict[str, object]:
    name = _sanitize_model_name(name)
    if not filename.endswith((".tar.gz", ".tgz", ".zip")):
        raise HTTPException(
            status_code=400, detail="Unsupported archive format (use .tar.gz, .tgz, or .zip)."
        )
    if (_models_dir() / name).exists():
        raise HTTPException(
            status_code=409,
            detail=f"A local model named '{name}' already exists; delete it first.",
        )
    declared = int(request.headers.get("content-length") or 0)
    # Compressed body + extracted copy coexist briefly.
    _check_disk_for(declared, factor=2.0)
    transfer_id = uuid.uuid4().hex
    _MODELS_TMP.mkdir(parents=True, exist_ok=True)
    archive_path = _MODELS_TMP / f"{transfer_id}-{Path(filename).name}"
    staging = _new_staging(transfer_id)
    try:
        await _stream_body_to_file(request, archive_path)
        await asyncio.to_thread(_safe_extract, archive_path, staging)
        result = _finalize_model(staging, name, flatten=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)
    logger.info("Local model '%s' uploaded from archive (%s MB)", name, result["size_mb"])
    return result


async def _run_pull(transfer_id: str) -> None:
    transfer = _transfers[transfer_id]
    name = str(transfer["name"])
    url = str(transfer["url"])
    staging = _new_staging(transfer_id)
    filename = str(transfer["filename"])
    is_archive = filename.endswith((".tar.gz", ".tgz", ".zip"))
    download_target = (
        _MODELS_TMP / f"{transfer_id}-{filename}" if is_archive else staging / filename
    )
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=15.0), follow_redirects=True
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Download failed: HTTP {response.status_code} from {url}"
                    )
                total = int(response.headers.get("content-length") or 0)
                transfer["total_bytes"] = total or None
                _check_disk_for(total, factor=2.0 if is_archive else 1.0)
                with open(download_target, "wb") as fh:
                    async for chunk in response.aiter_bytes():
                        await asyncio.to_thread(fh.write, chunk)
                        transfer["received_bytes"] = (
                            int(transfer.get("received_bytes", 0)) + len(chunk)  # type: ignore[arg-type]
                        )
        if is_archive:
            transfer["status"] = "extracting"
            await asyncio.to_thread(_safe_extract, download_target, staging)
        result = _finalize_model(staging, name, flatten=is_archive)
        transfer["status"] = "done"
        transfer["path"] = result["path"]
        transfer["warnings"] = result["warnings"]
        logger.info("Local model '%s' pulled from %s", name, url)
    except HTTPException as exc:
        transfer["status"] = "error"
        transfer["error"] = str(exc.detail)
        shutil.rmtree(staging, ignore_errors=True)
    except Exception as exc:
        transfer["status"] = "error"
        transfer["error"] = str(exc)
        shutil.rmtree(staging, ignore_errors=True)
        logger.warning("Local model pull of %s failed: %s", url, exc)
    finally:
        if is_archive:
            download_target.unlink(missing_ok=True)
        transfer["finished_at"] = time.monotonic()


def _pull_filename(url: str, name: str | None) -> str:
    """Best-effort filename from the URL path (query string stripped)."""
    tail = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return tail or (name or "model")


@app.post("/local-models/pull")
async def pull_local_model(payload: LocalModelPull) -> dict[str, str]:
    # async so create_task below binds to the running event loop.
    url = payload.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only http(s) URLs are supported.")
    filename = _pull_filename(url, payload.name)
    derived = filename
    for suffix in (".tar.gz", ".tgz", ".zip"):
        if derived.endswith(suffix):
            derived = derived[: -len(suffix)]
            break
    name = _sanitize_model_name(payload.name or derived)
    if (_models_dir() / name).exists():
        raise HTTPException(
            status_code=409,
            detail=f"A local model named '{name}' already exists; delete it first.",
        )
    transfer_id = uuid.uuid4().hex
    _transfers[transfer_id] = {
        "id": transfer_id,
        "kind": "pull",
        "name": name,
        "url": url,
        "filename": filename,
        "status": "downloading",
        "total_bytes": None,
        "received_bytes": 0,
        "error": None,
        "path": None,
        "started_at": time.monotonic(),
        "finished_at": None,
    }
    asyncio.create_task(_run_pull(transfer_id))
    return {"transfer_id": transfer_id, "name": name}


@app.get("/local-models/transfers")
def list_local_model_transfers() -> dict[str, list[dict[str, object]]]:
    # Prune transfers that finished a while ago so the dict stays small.
    now = time.monotonic()
    for tid, transfer in list(_transfers.items()):
        finished = transfer.get("finished_at")
        if finished is not None and now - float(finished) > 3600.0:  # type: ignore[arg-type]
            _transfers.pop(tid, None)
    fields = ("id", "kind", "name", "status", "total_bytes", "received_bytes", "error", "path")
    return {
        "transfers": [{k: t.get(k) for k in fields} for t in _transfers.values()]
    }


def _local_model_entry(path: Path, source: str, deletable: bool) -> dict[str, object]:
    deploy_path = path
    if path.is_dir():
        children = [c for c in path.iterdir() if c.is_file()]
        # vLLM needs the file path (not the dir) for single-file GGUF models.
        if len(children) == 1 and children[0].suffix == ".gguf":
            deploy_path = children[0]
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    return {
        "name": path.name,
        "path": str(deploy_path),
        "source": source,
        "size_mb": round(_dir_size_bytes(path) / (1024 * 1024)),
        "in_use": _local_model_in_use(str(deploy_path)),
        "deletable": deletable,
        "last_modified_at": mtime,
    }


@app.get("/local-models")
def list_local_models() -> dict[str, list[dict[str, object]]]:
    models: list[dict[str, object]] = []
    managed = _models_dir()
    for entry in sorted(managed.iterdir()):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        models.append(_local_model_entry(entry, source="managed", deletable=True))
    for base in _allowed_model_dirs():
        if base == managed.resolve():
            continue
        for entry in sorted(base.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() or entry.suffix == ".gguf":
                models.append(
                    _local_model_entry(entry, source=str(base), deletable=False)
                )
    return {"models": models}


@app.delete("/local-models/{name}")
def delete_local_model(name: str) -> dict[str, str]:
    name = _sanitize_model_name(name)
    managed = _models_dir().resolve()
    target = (managed / name).resolve()
    if managed not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Local model not found.")
    entry = _local_model_entry(target, source="managed", deletable=True)
    if entry["in_use"]:
        raise HTTPException(
            status_code=409,
            detail=f"Local model '{name}' is served by an active deployment; stop it first.",
        )
    shutil.rmtree(target)
    logger.info("Deleted local model '%s'", name)
    return {"status": "deleted", "name": name}


def _unified_memory_mb() -> tuple[int, int]:
    mem = psutil.virtual_memory()
    return round(mem.used / (1024 * 1024)), round(mem.total / (1024 * 1024))


def _smi_int(value: str) -> int | None:
    """Parse a numeric nvidia-smi CSV field; '[N/A]' (unified memory) -> None."""
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_nvidia_smi_gpus(output: str) -> list[dict[str, object]]:
    """Per-field tolerant parse: unified-memory devices (e.g. DGX Spark)
    report real utilization.gpu but '[N/A]' for dedicated VRAM fields."""
    gpus: list[dict[str, object]] = []
    for line in output.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 5:
            continue
        index = _smi_int(parts[0])
        if index is None:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "source": "nvidia-smi",
                "utilization": _smi_int(parts[2]),
                "memory_used_mb": _smi_int(parts[3]),
                "memory_total_mb": _smi_int(parts[4]),
            }
        )
    return gpus


def _substitute_unified_memory(
    gpus: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Fill missing dedicated-VRAM numbers from system RAM (unified memory),
    keeping the device's real compute utilization untouched."""
    for gpu in gpus:
        if not gpu.get("memory_total_mb"):
            used, total = _unified_memory_mb()
            gpu["memory_used_mb"] = used
            gpu["memory_total_mb"] = total
            gpu["source"] = "unified"
    return gpus


def _gpu_metrics() -> list[dict[str, object]]:
    def _unified_memory_metrics() -> list[dict[str, object]]:
        # Last resort (no NVML, no nvidia-smi): memory comes from system RAM
        # and compute is genuinely unknown — report it as such instead of
        # passing the RAM percentage off as utilization.
        used, total = _unified_memory_mb()
        return [
            {
                "index": 0,
                "name": "Unified memory",
                "source": "unified",
                "utilization": None,
                "memory_used_mb": used,
                "memory_total_mb": total,
            }
        ]

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                utilization = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = round(mem.used / (1024 * 1024))
                mem_total = round(mem.total / (1024 * 1024))
            except Exception:
                mem_used = mem_total = 0
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "source": "nvml",
                    "utilization": utilization,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                }
            )
        return _substitute_unified_memory(gpus)
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

    gpus = _parse_nvidia_smi_gpus(output)
    if not gpus:
        logger.warning("nvidia-smi returned no GPU rows; using unified memory metrics.")
        return _unified_memory_metrics()
    return _substitute_unified_memory(gpus)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
