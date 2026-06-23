import argparse
import hashlib

from aquila import __version__
import os
import shutil
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Iterable
import time

DEFAULT_CONSUL_PORT = 47528
DEFAULT_ADMIN_API_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_POSTGRES_PORT = 5757
DEFAULT_POSTGRES_DB = "aquila"
DEFAULT_POSTGRES_USER = "vllm"
DEFAULT_POSTGRES_PASSWORD = "change-me"
DEFAULT_POSTGRES_HOST = "127.0.0.1"
DEFAULT_CLIENT_HOST = "0.0.0.0"
DEFAULT_CLIENT_PORT = 9000

HOST_SERVICE_NAME = "aquila"
CLIENT_SERVICE_NAME = "aquila-client"


@dataclass
class HostConfig:
    host_ip: str
    frontend_port: int
    admin_api_port: int
    consul_port: int
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    base_path: str


@dataclass
class ClientConfig:
    host_ip: str
    consul_port: int
    client_host: str
    client_port: int
    node_name: str


class CheckStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class PreflightResult:
    label: str
    status: CheckStatus
    message: str
    hint: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aquila",
        description="Aquila — GPU inference cluster manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    host_parser = subparsers.add_parser("host", help="Manage the host services")
    host_subparsers = host_parser.add_subparsers(dest="action", required=True)

    host_up = host_subparsers.add_parser("up", help="Install and start host services")
    host_up.add_argument("--service", action="store_true", help="Run as a systemd service.")
    host_up.add_argument("--host-ip", default="127.0.0.1", help="Bind host for the backend API and UI backend target.")
    host_up.add_argument("--host-frontend-port", type=int, default=DEFAULT_FRONTEND_PORT, help="UI port.")
    host_up.add_argument("--base-path", default="/", help="Base path for the UI (for reverse proxies).")
    host_up.add_argument("--host-discover-port", type=int, default=DEFAULT_CONSUL_PORT, help="Discovery port used by clients.")
    host_up.add_argument("--host-backend-port", type=int, default=DEFAULT_ADMIN_API_PORT, help="Backend API port.")
    host_up.add_argument("--postgres-host", default=DEFAULT_POSTGRES_HOST, help="Postgres host.")
    host_up.add_argument("--postgres-port", type=int, default=DEFAULT_POSTGRES_PORT, help="Postgres port.")
    host_up.add_argument("--postgres-db", default=DEFAULT_POSTGRES_DB, help="Postgres database name.")
    host_up.add_argument("--postgres-user", default=DEFAULT_POSTGRES_USER, help="Postgres user.")
    host_up.add_argument("--postgres-password", default=DEFAULT_POSTGRES_PASSWORD, help="Postgres password.")

    host_down = host_subparsers.add_parser("down", help="Stop host services")
    host_down.add_argument(
        "--purge",
        action="store_true",
        help="Also delete the Postgres data volume (wipes all deployments, nodes, and history).",
    )

    client_parser = subparsers.add_parser("client", help="Manage a client node")
    client_subparsers = client_parser.add_subparsers(dest="action", required=True)

    client_up = client_subparsers.add_parser("up", help="Install and start the client")
    client_up.add_argument("--service", action="store_true", help="Run as a systemd service.")
    client_up.add_argument("--host-ip", default="127.0.0.1", help="Host IP for discovery.")
    client_up.add_argument("--host-discover-port", type=int, default=DEFAULT_CONSUL_PORT, help="Host discovery port.")
    client_up.add_argument("--client-host", default=DEFAULT_CLIENT_HOST, help="Client bind host.")
    client_up.add_argument("--client-port", type=int, default=DEFAULT_CLIENT_PORT, help="Client bind port.")
    client_up.add_argument("--node-name", default=socket.gethostname(), help="Node name used for registration.")

    client_down = client_subparsers.add_parser("down", help="Stop the client")

    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove the tool's runtime directories, venvs, and caches.",
    )
    clean_parser.add_argument(
        "--docker",
        action="store_true",
        help="Also remove managed vLLM containers and cached vLLM images.",
    )
    clean_parser.add_argument(
        "-y", "--yes", action="store_true", help="Do not prompt for confirmation.",
    )

    args = parser.parse_args()

    if args.command == "host":
        if args.action == "up":
            host_config = build_host_config(args)
            run_host_up(host_config, use_service=args.service)
            return
        if args.action == "down":
            run_host_down(purge=args.purge)
            return

    if args.command == "client":
        if args.action == "up":
            client_config = build_client_config(args)
            run_client_up(client_config, use_service=args.service)
            return
        if args.action == "down":
            run_client_down()
            return

    if args.command == "clean":
        run_clean(remove_docker=args.docker, assume_yes=args.yes)
        return

    parser.error("Unknown command")


def build_host_config(args: argparse.Namespace) -> HostConfig:
    consul_port = args.host_discover_port
    return HostConfig(
        host_ip=args.host_ip,
        frontend_port=args.host_frontend_port,
        admin_api_port=args.host_backend_port,
        consul_port=consul_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_db=args.postgres_db,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        base_path=args.base_path,
    )


def build_client_config(args: argparse.Namespace) -> ClientConfig:
    consul_port = args.host_discover_port
    return ClientConfig(
        host_ip=args.host_ip,
        consul_port=consul_port,
        client_host=args.client_host,
        client_port=args.client_port,
        node_name=args.node_name,
    )


def _check_python() -> PreflightResult:
    v = sys.version_info
    version_str = f"Python {v[0]}.{v[1]}.{v[2]}"
    if v >= (3, 10):
        return PreflightResult("Python >= 3.10", CheckStatus.PASS, version_str)
    return PreflightResult(
        "Python >= 3.10", CheckStatus.FAIL, version_str,
        hint="Install Python 3.10 or newer.",
    )


def _check_node() -> PreflightResult:
    node = shutil.which("node")
    if not node:
        return PreflightResult(
            "Node.js >= 23", CheckStatus.FAIL, "Not found",
            hint="Install Node.js 23+ (https://nodejs.org/).",
        )
    try:
        out = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=5,
        )
        version_str = out.stdout.strip()
        major_str = version_str.lstrip("v").split(".")[0]
        try:
            major = int(major_str)
        except ValueError:
            major = 0
        if major >= 23:
            return PreflightResult("Node.js >= 23", CheckStatus.PASS, version_str)
        return PreflightResult(
            "Node.js >= 23", CheckStatus.FAIL, version_str,
            hint="Upgrade to Node.js 23+.",
        )
    except Exception:
        return PreflightResult(
            "Node.js >= 23", CheckStatus.FAIL, "Error running node --version",
            hint="Check your Node.js installation.",
        )


def _check_npm() -> PreflightResult:
    npm = shutil.which("npm")
    if npm:
        return PreflightResult("npm", CheckStatus.PASS, npm)
    return PreflightResult(
        "npm", CheckStatus.FAIL, "Not found",
        hint="npm ships with Node.js — reinstall Node.js.",
    )


def _check_docker() -> PreflightResult:
    docker = shutil.which("docker")
    if not docker:
        return PreflightResult(
            "Docker daemon", CheckStatus.FAIL, "Not found",
            hint="Install Docker (https://docs.docker.com/engine/install/) and add your user to the docker group.",
        )
    try:
        out = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return PreflightResult("Docker daemon", CheckStatus.PASS, f"Docker {out.stdout.strip()}")
        stderr = out.stderr.strip().lower()
        if "permission denied" in stderr or "connect" in stderr:
            return PreflightResult(
                "Docker daemon", CheckStatus.FAIL, "Permission denied",
                hint="Add your user to the docker group: sudo usermod -aG docker $USER  (then log out and back in).",
            )
        return PreflightResult(
            "Docker daemon", CheckStatus.FAIL, "Daemon not running",
            hint="Start the Docker daemon: sudo systemctl start docker",
        )
    except Exception:
        return PreflightResult(
            "Docker daemon", CheckStatus.FAIL, "Error checking Docker",
            hint="Install Docker and ensure the daemon is running.",
        )


def _check_compose() -> PreflightResult:
    docker = shutil.which("docker")
    if docker:
        try:
            out = subprocess.run(
                [docker, "compose", "version", "--short"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return PreflightResult("Docker Compose", CheckStatus.PASS, f"docker compose v{out.stdout.strip()}")
        except Exception:
            pass
    dc = shutil.which("docker-compose")
    if dc:
        try:
            out = subprocess.run(
                [dc, "--version"], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                return PreflightResult("Docker Compose", CheckStatus.PASS, out.stdout.strip())
        except Exception:
            pass
    return PreflightResult(
        "Docker Compose", CheckStatus.FAIL, "Not found",
        hint="Install the Docker Compose plugin: sudo apt install docker-compose-plugin",
    )


def _check_container_runtime() -> PreflightResult:
    docker = shutil.which("docker")
    if docker:
        try:
            out = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return PreflightResult("Container runtime", CheckStatus.PASS, f"Docker {out.stdout.strip()}")
            stderr = out.stderr.strip().lower()
            if "permission denied" in stderr or "connect" in stderr:
                return PreflightResult(
                    "Container runtime", CheckStatus.FAIL, "Permission denied",
                    hint="Add your user to the docker group: sudo usermod -aG docker $USER  (then log out and back in).",
                )
        except Exception:
            pass
    podman = shutil.which("podman")
    if podman:
        try:
            out = subprocess.run(
                [podman, "version", "--format", "{{.Version}}"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return PreflightResult("Container runtime", CheckStatus.PASS, f"Podman {out.stdout.strip()}")
        except Exception:
            pass
    return PreflightResult(
        "Container runtime", CheckStatus.FAIL, "Not found",
        hint="Install Docker (https://docs.docker.com/engine/install/) or enable the Podman socket.",
    )


def _check_nvidia_smi() -> PreflightResult:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return PreflightResult(
            "GPU driver (nvidia-smi)", CheckStatus.WARN, "Not found",
            hint="Install NVIDIA drivers if this node has GPUs.",
        )
    try:
        out = subprocess.run(
            [nvsmi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            gpus = [line for line in out.stdout.strip().splitlines() if line.strip()]
            return PreflightResult("GPU driver (nvidia-smi)", CheckStatus.PASS, f"{len(gpus)} GPU(s)")
        return PreflightResult(
            "GPU driver (nvidia-smi)", CheckStatus.WARN, "nvidia-smi failed",
            hint="Check your NVIDIA driver installation.",
        )
    except Exception:
        return PreflightResult(
            "GPU driver (nvidia-smi)", CheckStatus.WARN, "Error running nvidia-smi",
            hint="Check your NVIDIA driver installation.",
        )


def _check_nvidia_ctk() -> PreflightResult:
    ctk = shutil.which("nvidia-ctk")
    if ctk:
        return PreflightResult("NVIDIA Container Toolkit", CheckStatus.PASS, ctk)
    cdi_path = Path("/etc/cdi")
    if cdi_path.exists() and any(cdi_path.glob("*.json")):
        return PreflightResult("NVIDIA Container Toolkit", CheckStatus.PASS, "CDI specs found")
    return PreflightResult(
        "NVIDIA Container Toolkit", CheckStatus.WARN, "Not found",
        hint="Install nvidia-container-toolkit so containers can access GPUs.",
    )


def _check_port(port: int, label: str, flag: str = "") -> PreflightResult:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
        return PreflightResult(f"Port {port} ({label})", CheckStatus.PASS, "Available")
    except OSError:
        hint = f"Stop the process on port {port}"
        if flag:
            hint += f", or pass {flag} <port>"
        return PreflightResult(
            f"Port {port} ({label})", CheckStatus.FAIL, "In use",
            hint=hint,
        )


def preflight_host(config: HostConfig) -> list[PreflightResult]:
    return [
        _check_python(),
        _check_node(),
        _check_npm(),
        _check_docker(),
        _check_compose(),
        _check_port(config.frontend_port, "frontend", "--host-frontend-port"),
        _check_port(config.admin_api_port, "backend API", "--host-backend-port"),
        _check_port(config.consul_port, "discovery", "--host-discover-port"),
        _check_port(config.postgres_port, "Postgres", "--postgres-port"),
    ]


def preflight_client(config: ClientConfig) -> list[PreflightResult]:
    return [
        _check_python(),
        _check_container_runtime(),
        _check_nvidia_smi(),
        _check_nvidia_ctk(),
        _check_port(config.client_port, "client API", "--client-port"),
    ]


def run_preflight(results: list[PreflightResult]) -> None:
    from aquila.banner import _rgb, _supports_color

    color = _supports_color()
    _OK = (134, 194, 132)
    _ERR = (220, 100, 100)
    _WARN_CLR = (220, 186, 100)

    label_width = max(len(r.label) for r in results)

    print("Preflight checks:")
    for r in results:
        if r.status == CheckStatus.PASS:
            icon = _rgb(*_OK, "✓") if color else "✓"
        elif r.status == CheckStatus.FAIL:
            icon = _rgb(*_ERR, "✗") if color else "✗"
        else:
            icon = _rgb(*_WARN_CLR, "!") if color else "!"
        print(f"  {icon}  {r.label:<{label_width}}   {r.message}")
        if r.hint and r.status != CheckStatus.PASS:
            print(f"     → {r.hint}")

    fails = sum(1 for r in results if r.status == CheckStatus.FAIL)
    warns = sum(1 for r in results if r.status == CheckStatus.WARN)

    if fails:
        msg = f"\n{fails} check(s) failed — cannot continue."
        print(_rgb(*_ERR, msg) if color else msg)
        sys.exit(1)
    elif warns:
        msg = f"\n{warns} warning(s) — proceeding anyway."
        print(_rgb(*_WARN_CLR, msg) if color else msg)
    print()


def run_host_up(config: HostConfig, use_service: bool) -> None:
    from aquila.banner import banner
    print(banner())
    run_preflight(preflight_host(config))
    runtime_dir = ensure_runtime_dir("host")
    ensure_host_assets(runtime_dir)
    print("Host configuration:")
    print(format_kv(
        {
            "host_ip": config.host_ip,
            "host_frontend_port": config.frontend_port,
            "host_discover_port": config.consul_port,
            "host_backend_port": config.admin_api_port,
            "postgres_host": config.postgres_host,
            "postgres_port": config.postgres_port,
            "postgres_db": config.postgres_db,
            "postgres_user": config.postgres_user,
            "postgres_password": config.postgres_password,
            "base_path": config.base_path,
            "service_mode": use_service,
        }
    ))
    write_host_env_files(runtime_dir, config)
    ensure_backend_venv(runtime_dir)
    ensure_frontend_deps(runtime_dir)
    frontend_env = load_env_file(runtime_dir / "frontend" / ".env")
    build_frontend(runtime_dir, frontend_env)
    if use_service:
        install_host_service(config)
        print(f"Host services: {HOST_SERVICE_NAME}-infra.service, {HOST_SERVICE_NAME}-backend.service, {HOST_SERVICE_NAME}-frontend.service")
        systemctl(["enable", "--now", f"{HOST_SERVICE_NAME}-infra.service"])
        systemctl(["enable", "--now", f"{HOST_SERVICE_NAME}-backend.service"])
        systemctl(["enable", "--now", f"{HOST_SERVICE_NAME}-frontend.service"])
        return
    start_infra(runtime_dir)
    backend_env = load_env_file(runtime_dir / "backend" / ".env")
    backend_cmd = [
        str(runtime_dir / "backend" / ".venv" / "bin" / "uvicorn"),
        "app.main:app",
        "--host",
        backend_env.get("ADMIN_API_HOST", config.host_ip),
        "--port",
        backend_env.get("ADMIN_API_PORT", str(config.admin_api_port)),
    ]
    npm_path = shutil.which("npm")
    if not npm_path:
        raise RuntimeError("npm is required to run the frontend.")
    frontend_cmd = [
        npm_path,
        "run",
        "preview",
        "--",
        "--host",
        frontend_env.get("FRONTEND_HOST", "0.0.0.0"),
        "--port",
        frontend_env.get("FRONTEND_PORT", str(config.frontend_port)),
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=runtime_dir / "backend",
        env=merge_env(backend_env),
    )
    backend_port = int(backend_env.get("ADMIN_API_PORT", str(config.admin_api_port)))
    _wait_for_backend(backend_proc, backend_port)
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=runtime_dir / "frontend",
        env=merge_env(frontend_env),
    )
    write_pid(runtime_dir / ".backend.pid", backend_proc.pid)
    write_pid(runtime_dir / ".frontend.pid", frontend_proc.pid)
    try:
        wait_for_processes(backend_proc, frontend_proc)
    finally:
        terminate_process(backend_proc)
        terminate_process(frontend_proc)
        remove_pid(runtime_dir / ".backend.pid")
        remove_pid(runtime_dir / ".frontend.pid")
        stop_infra(runtime_dir)
        remove_runtime_dir("host")


def run_host_down(purge: bool = False) -> None:
    runtime_dir = runtime_dir_path("host")
    if runtime_dir:
        stop_infra(runtime_dir, purge=purge)
        stop_pid(runtime_dir / ".backend.pid")
        stop_pid(runtime_dir / ".frontend.pid")
    elif purge:
        print(
            "No host runtime directory found; if a Postgres volume remains, "
            "remove it with `docker volume rm host_pgdata`."
        )
    remove_host_service()
    remove_runtime_dir("host")


def run_client_up(config: ClientConfig, use_service: bool) -> None:
    from aquila.banner import banner
    print(banner())
    run_preflight(preflight_client(config))
    runtime_dir = ensure_runtime_dir("client")
    print("Client configuration:")
    print(format_kv(
        {
            "host_ip": config.host_ip,
            "host_discover_port": config.consul_port,
            "client_host": config.client_host,
            "client_port": config.client_port,
            "node_name": config.node_name,
            "service_mode": use_service,
        }
    ))
    write_client_env_file(runtime_dir, config)
    ensure_client_venv(runtime_dir)
    if use_service:
        install_client_service(config)
        print(f"Client service: {CLIENT_SERVICE_NAME}.service")
        systemctl(["enable", "--now", f"{CLIENT_SERVICE_NAME}.service"])
        return
    client_env = load_env_file(runtime_dir / ".env")
    python_bin = runtime_dir / ".venv" / "bin" / "python"
    print(f"Client bind: {config.client_host}:{config.client_port}")
    print(f"Host port: {config.host_ip}:{config.consul_port}")
    try:
        run([str(python_bin), "-m", "app.main"], cwd=runtime_dir, env=merge_env(client_env))
    finally:
        remove_runtime_dir("client")


def run_client_down() -> None:
    runtime_dir = runtime_dir_path("client")
    if runtime_dir:
        stop_pid(runtime_dir / ".client.pid")
    remove_client_service()
    remove_runtime_dir("client")


def run_clean(remove_docker: bool = False, assume_yes: bool = False) -> None:
    """Remove the tool's runtime directories, venvs, and caches.

    Targets the host/client runtime dirs under XDG data home and the client
    working root (uploaded packages). The HuggingFace model cache is left
    intact. systemd units are not removed here — use `host down`/`client down`.
    """
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    data_root = base_dir / "aquila"
    client_root = Path(os.environ.get("VLLM_CLIENT_ROOT", Path.home() / ".vllm-client"))
    legacy_roots = [base_dir / "athanor", base_dir / "vllm_cluster_manager"]

    targets = [p for p in (data_root, client_root, *legacy_roots) if p.exists()]

    candidate_units = (
        f"{HOST_SERVICE_NAME}-infra.service",
        f"{HOST_SERVICE_NAME}-backend.service",
        f"{HOST_SERVICE_NAME}-frontend.service",
        f"{CLIENT_SERVICE_NAME}.service",
    )
    installed_units = [u for u in candidate_units if Path(f"/etc/systemd/system/{u}").exists()]

    if not targets and not remove_docker:
        print("Nothing to clean.")
        if installed_units:
            print("Installed systemd services remain; remove with `host down` / `client down`.")
        return

    print("aquila clean will remove:")
    for p in targets:
        print(f"  - {p}")
    if (data_root / "host").exists():
        print("  - Docker: Postgres data volume (all deployments, nodes, history)")
    if remove_docker:
        print("  - Docker: managed vLLM containers and cached vLLM images")
    print("  (the HuggingFace model cache is left intact)")
    if installed_units:
        print("\nNote: systemd services are installed and will NOT be removed:")
        for u in installed_units:
            print(f"    {u}")
        print("  Remove them with `aquila host down` / `client down`.")

    if not assume_yes:
        if not sys.stdin.isatty():
            print("\nRefusing to proceed without confirmation; re-run with --yes.")
            return
        try:
            answer = input("\nProceed? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    # Stop foreground processes and infra containers first so they aren't
    # orphaned when their working directories disappear.
    host_dir = data_root / "host"
    if host_dir.exists():
        try:
            stop_infra(host_dir, purge=True)
        except Exception as exc:
            print(f"  (could not stop host infra: {exc})")
        stop_pid(host_dir / ".backend.pid")
        stop_pid(host_dir / ".frontend.pid")
    client_dir = data_root / "client"
    if client_dir.exists():
        stop_pid(client_dir / ".client.pid")

    if remove_docker:
        _clean_docker()

    failures: list[Path] = []
    for p in targets:
        shutil.rmtree(p, ignore_errors=True)
        if p.exists():
            failures.append(p)

    if failures:
        print("\nSome paths could not be fully removed (likely root-owned from an")
        print("earlier sudo/root run). Re-run as root to finish:")
        print(f"  sudo rm -rf {' '.join(str(p) for p in failures)}")
    else:
        print("Clean complete.")


def _clean_docker() -> None:
    docker = shutil.which("docker")
    if not docker:
        print("  (docker not found; skipping container/image cleanup)")
        return
    try:
        out = run(
            [docker, "ps", "-aq", "--filter", "label=aquila.managed=true"],
            capture=True,
        )
        ids = out.split()
        if ids:
            run([docker, "rm", "-f", *ids], capture=True)
            print(f"  Removed {len(ids)} managed vLLM container(s).")
    except RuntimeError as exc:
        print(f"  (container cleanup failed: {exc})")
    try:
        out = run([docker, "images", "--format", "{{.Repository}}:{{.Tag}}"], capture=True)
        images = [
            line for line in out.split()
            if line.startswith("vllm/vllm-openai:")
            or line.startswith("aquila/local:")
        ]
        if images:
            run([docker, "rmi", "-f", *images], capture=True)
            print(f"  Removed {len(images)} cached vLLM image(s).")
    except RuntimeError as exc:
        print(f"  (image cleanup failed: {exc})")
    for vol in ("vllm_cluster_manager_pgdata", "athanor_pgdata"):
        try:
            run([docker, "volume", "rm", "-f", vol], capture=True)
            print(f"  Removed legacy volume {vol}.")
        except RuntimeError:
            pass


def ensure_runtime_dir(kind: str) -> Path:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "aquila" / kind
    runtime_dir.mkdir(parents=True, exist_ok=True)
    copy_assets(kind, runtime_dir)
    return runtime_dir


def runtime_dir_path(kind: str) -> Path | None:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "aquila" / kind
    if runtime_dir.exists():
        return runtime_dir
    return None


def remove_runtime_dir(kind: str) -> None:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "aquila" / kind
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


# Build artifacts that must never be copied into the runtime dir: stale
# bytecode causes permission collisions on re-runs and serves outdated code.
_ASSET_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".venv", "node_modules"
)


def _refresh_tree(src: Path, dest: Path) -> None:
    """Recursively copy *src* into *dest*, refreshing file CONTENTS only.

    Unlike ``shutil.copytree``, this never replicates directory or file
    metadata (mode/mtime/owner). That matters because containers we launch
    bind-mount and ``chown`` some runtime subdirs to their own uid — e.g. the
    Consul container takes ``infra/consul`` as uid 100. ``copytree`` would then
    fail on every re-run (``copystat`` → ``os.utime`` on a dir we no longer own
    → ``EPERM``). We only need current file contents in the runtime tree, so we
    skip metadata entirely and overwrite files by unlinking first (which needs
    only write permission on the parent dir, which we always hold).
    """
    dest.mkdir(parents=True, exist_ok=True)
    entries = sorted(p.name for p in src.iterdir())
    ignored = _ASSET_IGNORE(str(src), entries)
    for name in entries:
        if name in ignored:
            continue
        src_child = src / name
        dest_child = dest / name
        if src_child.is_dir() and not src_child.is_symlink():
            _refresh_tree(src_child, dest_child)
        else:
            if dest_child.is_symlink() or dest_child.exists():
                try:
                    dest_child.unlink()
                except OSError:
                    pass
            shutil.copyfile(src_child, dest_child, follow_symlinks=True)


def copy_assets(kind: str, dest: Path) -> None:
    src_root = resources.files("aquila.assets") / kind
    if not src_root.is_dir():
        raise RuntimeError(f"Missing packaged assets for {kind}.")
    with resources.as_file(src_root) as src_path:
        _refresh_tree(Path(src_path), dest)


def write_host_env_files(runtime_dir: Path, config: HostConfig) -> None:
    (runtime_dir / "backend").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "frontend").mkdir(parents=True, exist_ok=True)
    host_env = textwrap.dedent(
        f"""
        POSTGRES_DB={config.postgres_db}
        POSTGRES_USER={config.postgres_user}
        POSTGRES_PASSWORD={config.postgres_password}
        POSTGRES_PORT={config.postgres_port}
        CONSUL_PORT={config.consul_port}
        """
    ).strip() + "\n"
    (runtime_dir / ".env").write_text(host_env, encoding="utf-8")

    backend_env = textwrap.dedent(
        f"""
        ADMIN_API_HOST={config.host_ip}
        ADMIN_API_PORT={config.admin_api_port}
        POSTGRES_HOST={config.postgres_host}
        POSTGRES_PORT={config.postgres_port}
        POSTGRES_DB={config.postgres_db}
        POSTGRES_USER={config.postgres_user}
        POSTGRES_PASSWORD={config.postgres_password}
        CONSUL_HTTP_ADDR=http://{config.host_ip}:{config.consul_port}
        """
    ).strip() + "\n"
    (runtime_dir / "backend" / ".env").write_text(backend_env, encoding="utf-8")

    frontend_env = textwrap.dedent(
        f"""
        FRONTEND_HOST=0.0.0.0
        FRONTEND_PORT={config.frontend_port}
        VITE_BACKEND_HOST={config.host_ip}
        VITE_BACKEND_PORT={config.admin_api_port}
        VITE_BASE_PATH={config.base_path}
        """
    ).strip() + "\n"
    (runtime_dir / "frontend" / ".env").write_text(frontend_env, encoding="utf-8")


def ensure_host_assets(runtime_dir: Path) -> None:
    for subdir in ("frontend", "backend", "infra"):
        copy_assets_subdir("host", subdir, runtime_dir / subdir)


def copy_assets_subdir(kind: str, subdir: str, dest: Path) -> None:
    src_root = resources.files("aquila.assets") / kind / subdir
    if not src_root.is_dir():
        raise RuntimeError(f"Missing packaged assets for {kind}/{subdir}.")
    with resources.as_file(src_root) as src_path:
        _refresh_tree(Path(src_path), dest)


def write_client_env_file(runtime_dir: Path, config: ClientConfig) -> None:
    client_env = textwrap.dedent(
        f"""
        NODE_NAME={config.node_name}
        HOST={config.client_host}
        PORT={config.client_port}
        CONSUL_HTTP_ADDR=http://{config.host_ip}:{config.consul_port}
        """
    ).strip() + "\n"
    (runtime_dir / ".env").write_text(client_env, encoding="utf-8")


def ensure_backend_venv(runtime_dir: Path) -> None:
    backend_dir = runtime_dir / "backend"
    venv_dir = backend_dir / ".venv"
    requirements = backend_dir / "requirements.txt"
    ensure_venv(venv_dir, requirements, backend_dir / ".deps.sha256")


def ensure_client_venv(runtime_dir: Path) -> None:
    # The client agent is a lightweight FastAPI service; vLLM itself runs in
    # official Docker containers, so the venv only needs the agent's own deps.
    venv_dir = runtime_dir / ".venv"
    requirements = runtime_dir / "requirements.txt"
    ensure_venv(venv_dir, requirements, runtime_dir / ".deps.sha256")


def ensure_venv(venv_dir: Path, requirements: Path, marker: Path) -> None:
    # Validate the interpreter, not just the directory: an interrupted run (or a
    # partially-failed cleanup) can leave a .venv dir with no bin/python.
    python_bin = venv_dir / "bin" / "python"
    fresh = not python_bin.exists()
    if fresh:
        create_venv(venv_dir)
    # Force a reinstall when the venv was (re)created, even if the deps marker
    # still matches — otherwise a fresh venv would be left without dependencies.
    if fresh or needs_install(requirements, marker):
        install_requirements(venv_dir, requirements)
        write_hash_marker(requirements, marker)


def create_venv(venv_dir: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        # --allow-existing repairs/reuses a partially-created venv dir instead of
        # erroring or prompting interactively.
        run([uv, "venv", "--allow-existing", "--python=3.12", str(venv_dir)])
        return
    run([sys.executable, "-m", "venv", str(venv_dir)])


def install_requirements(venv_dir: Path, requirements: Path) -> None:
    python_bin = venv_dir / "bin" / "python"
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", str(python_bin), "-r", str(requirements)])
        return
    run([str(python_bin), "-m", "pip", "install", "-r", str(requirements)])


def ensure_frontend_deps(runtime_dir: Path) -> None:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required to install frontend dependencies.")
    frontend_dir = runtime_dir / "frontend"
    lockfile = frontend_dir / "package-lock.json"
    manifest = lockfile if lockfile.exists() else frontend_dir / "package.json"
    marker = frontend_dir / ".deps.sha256"
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists() and not needs_install(manifest, marker):
        return
    run([npm, "install", "--no-audit", "--no-fund"], cwd=frontend_dir)
    write_hash_marker(manifest, marker)


def build_frontend(runtime_dir: Path, frontend_env: dict[str, str]) -> None:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required to build the frontend.")
    frontend_dir = runtime_dir / "frontend"
    run([npm, "run", "build"], cwd=frontend_dir, env=merge_env(frontend_env))


def start_infra(runtime_dir: Path) -> None:
    compose_cmd = detect_compose_cmd()
    cmd = compose_cmd.split() + ["up", "-d"]
    run(cmd, cwd=runtime_dir)


def stop_infra(runtime_dir: Path, purge: bool = False) -> None:
    """Stop Postgres/Consul. Data volumes are kept unless purge is requested."""
    compose_cmd = detect_compose_cmd()
    cmd = compose_cmd.split() + ["down"]
    if purge:
        cmd.append("-v")
    run(cmd, cwd=runtime_dir)


def compose_service_cmd(args: str) -> str:
    return (
        '/bin/sh -c "if docker compose version >/dev/null 2>&1; then '
        f'docker compose {args}; '
        'elif command -v docker-compose >/dev/null 2>&1; then '
        f'docker-compose {args}; '
        'else echo \\"Docker Compose not found\\" >&2; exit 1; fi"'
    )


def install_host_service(config: HostConfig) -> None:
    runtime_dir = ensure_runtime_dir("host")
    compose_start = compose_service_cmd("up -d")
    compose_stop = compose_service_cmd("down")
    npm_path = shutil.which("npm")
    node_path = shutil.which("node")
    if not npm_path:
        raise RuntimeError("npm is required to run the frontend service.")
    if not node_path:
        raise RuntimeError("node is required to run the frontend service.")
    node_bin_dir = str(Path(node_path).parent)
    npm_bin_dir = str(Path(npm_path).parent)
    system_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    frontend_path = f"{node_bin_dir}:{npm_bin_dir}:{system_path}"

    infra_service = textwrap.dedent(
        f"""
        [Unit]
        Description=Aquila Infra (Postgres + Consul)
        After=docker.service
        Requires=docker.service

        [Service]
        Type=oneshot
        WorkingDirectory={runtime_dir}
        EnvironmentFile={runtime_dir}/.env
        ExecStart={compose_start}
        ExecStop={compose_stop}
        RemainAfterExit=yes

        [Install]
        WantedBy=multi-user.target
        """
    ).strip() + "\n"

    backend_service = textwrap.dedent(
        f"""
        [Unit]
        Description=Aquila Backend API
        After=network.target {HOST_SERVICE_NAME}-infra.service
        Requires={HOST_SERVICE_NAME}-infra.service

        [Service]
        Type=simple
        WorkingDirectory={runtime_dir}/backend
        EnvironmentFile={runtime_dir}/backend/.env
        Environment=AQUILA_VERSION={__version__}
        ExecStart={runtime_dir}/backend/.venv/bin/uvicorn app.main:app --host ${{ADMIN_API_HOST}} --port ${{ADMIN_API_PORT}}
        Restart=always
        RestartSec=2

        [Install]
        WantedBy=multi-user.target
        """
    ).strip() + "\n"

    frontend_service = textwrap.dedent(
        f"""
        [Unit]
        Description=Aquila Frontend
        After=network.target {HOST_SERVICE_NAME}-backend.service
        Requires={HOST_SERVICE_NAME}-backend.service

        [Service]
        Type=simple
        WorkingDirectory={runtime_dir}/frontend
        EnvironmentFile={runtime_dir}/frontend/.env
        Environment=PATH={frontend_path}
        ExecStart={npm_path} run preview -- --host ${{FRONTEND_HOST}} --port ${{FRONTEND_PORT}}
        Restart=always
        RestartSec=2

        [Install]
        WantedBy=multi-user.target
        """
    ).strip() + "\n"

    write_systemd_service(f"/etc/systemd/system/{HOST_SERVICE_NAME}-infra.service", infra_service)
    write_systemd_service(f"/etc/systemd/system/{HOST_SERVICE_NAME}-backend.service", backend_service)
    write_systemd_service(f"/etc/systemd/system/{HOST_SERVICE_NAME}-frontend.service", frontend_service)
    systemctl(["daemon-reload"])


def remove_host_service() -> None:
    if systemd_unit_exists(f"{HOST_SERVICE_NAME}-frontend.service"):
        systemctl(["disable", "--now", f"{HOST_SERVICE_NAME}-frontend.service"])
    if systemd_unit_exists(f"{HOST_SERVICE_NAME}-backend.service"):
        systemctl(["disable", "--now", f"{HOST_SERVICE_NAME}-backend.service"])
    if systemd_unit_exists(f"{HOST_SERVICE_NAME}-infra.service"):
        systemctl(["disable", "--now", f"{HOST_SERVICE_NAME}-infra.service"])
    remove_systemd_service(f"/etc/systemd/system/{HOST_SERVICE_NAME}-frontend.service")
    remove_systemd_service(f"/etc/systemd/system/{HOST_SERVICE_NAME}-backend.service")
    remove_systemd_service(f"/etc/systemd/system/{HOST_SERVICE_NAME}-infra.service")
    systemctl(["daemon-reload"])


def install_client_service(config: ClientConfig) -> None:
    runtime_dir = ensure_runtime_dir("client")
    client_service = textwrap.dedent(
        f"""
        [Unit]
        Description=Aquila Client
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory={runtime_dir}
        EnvironmentFile={runtime_dir}/.env
        Environment=AQUILA_VERSION={__version__}
        ExecStart={runtime_dir}/.venv/bin/python -m app.main
        Restart=always
        RestartSec=2

        [Install]
        WantedBy=multi-user.target
        """
    ).strip() + "\n"

    write_systemd_service(f"/etc/systemd/system/{CLIENT_SERVICE_NAME}.service", client_service)
    systemctl(["daemon-reload"])


def remove_client_service() -> None:
    if systemd_unit_exists(f"{CLIENT_SERVICE_NAME}.service"):
        systemctl(["disable", "--now", f"{CLIENT_SERVICE_NAME}.service"])
    remove_systemd_service(f"/etc/systemd/system/{CLIENT_SERVICE_NAME}.service")
    systemctl(["daemon-reload"])


def write_systemd_service(path: str, content: str) -> None:
    if os.geteuid() == 0:
        Path(path).write_text(content, encoding="utf-8")
        return
    result = subprocess.run(
        ["sudo", "tee", path],
        input=content,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to write systemd service: {path}")


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def merge_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["AQUILA_VERSION"] = __version__
    env.update(extra)
    return env


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def needs_install(requirements: Path, marker: Path) -> bool:
    if not requirements.exists():
        return False
    if not marker.exists():
        return True
    current = file_sha256(requirements)
    saved = marker.read_text(encoding="utf-8").strip()
    return current != saved


def write_hash_marker(requirements: Path, marker: Path) -> None:
    marker.write_text(file_sha256(requirements) + "\n", encoding="utf-8")


def format_kv(items: dict[str, object]) -> str:
    return "\n".join(f"  {key}={value}" for key, value in items.items())


def write_pid(path: Path, pid: int) -> None:
    path.write_text(str(pid), encoding="utf-8")


def remove_pid(path: Path) -> None:
    path.unlink(missing_ok=True)


def stop_pid(path: Path) -> None:
    if not path.exists():
        return
    pid_text = path.read_text(encoding="utf-8").strip()
    if not pid_text:
        return
    try:
        pid = int(pid_text)
    except ValueError:
        return
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
    remove_pid(path)


def _wait_for_backend(proc: subprocess.Popen, port: int, timeout: int = 90) -> None:
    """Poll the backend until it responds or dies."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/api/settings"
    deadline = time.monotonic() + timeout
    print(f"Waiting for backend on port {port} ...")
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Backend exited with code {proc.returncode} before becoming ready. "
                "Check the output above for errors."
            )
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                print("Backend ready.")
                return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise RuntimeError(
        f"Backend did not become ready within {timeout}s. "
        "Check the output above for errors."
    )


def wait_for_processes(*procs: subprocess.Popen) -> None:
    try:
        while True:
            for proc in procs:
                if proc.poll() is not None:
                    return
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        return


def remove_systemd_service(path: str) -> None:
    if os.geteuid() == 0:
        Path(path).unlink(missing_ok=True)
        return
    run(["sudo", "rm", "-f", path])


def systemctl(args: Iterable[str]) -> None:
    cmd = ["systemctl", *args]
    if os.geteuid() != 0:
        cmd.insert(0, "sudo")
    run(cmd)


def systemd_unit_exists(unit: str) -> bool:
    cmd = ["systemctl", "list-unit-files", unit]
    if os.geteuid() != 0:
        cmd.insert(0, "sudo")
    try:
        output = run(cmd, capture=True)
    except RuntimeError:
        return False
    return unit in output


def detect_compose_cmd() -> str:
    docker = shutil.which("docker")
    if docker:
        try:
            run([docker, "compose", "version"], capture=True)
            return f"{docker} compose"
        except RuntimeError:
            pass
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return docker_compose
    raise RuntimeError("Docker Compose not found. Install docker compose or docker-compose.")


def run(
    cmd: list[str],
    cwd: Path | None = None,
    capture: bool = False,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        output = result.stdout or ""
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{output}")
    return result.stdout or ""


if __name__ == "__main__":
    main()
