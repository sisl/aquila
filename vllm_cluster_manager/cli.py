import argparse
import json
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen
import time

DEFAULT_CONSUL_PORT = 47528
DEFAULT_ADMIN_API_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_POSTGRES_PORT = 5757
DEFAULT_POSTGRES_DB = "vllm_admin"
DEFAULT_POSTGRES_USER = "vllm"
DEFAULT_POSTGRES_PASSWORD = "change-me"
DEFAULT_POSTGRES_HOST = "127.0.0.1"
DEFAULT_CLIENT_HOST = "0.0.0.0"
DEFAULT_CLIENT_PORT = 9000

HOST_SERVICE_NAME = "vllm-cluster"
CLIENT_SERVICE_NAME = "vllm-cluster-client"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-cluster-manager",
        description="vLLM Cluster Manager CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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

    args = parser.parse_args()

    if args.command == "host":
        if args.action == "up":
            host_config = build_host_config(args)
            run_host_up(host_config, use_service=args.service)
            return
        if args.action == "down":
            run_host_down()
            return

    if args.command == "client":
        if args.action == "up":
            client_config = build_client_config(args)
            run_client_up(client_config, use_service=args.service)
            return
        if args.action == "down":
            run_client_down()
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


def run_host_up(config: HostConfig, use_service: bool) -> None:
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


def run_host_down() -> None:
    runtime_dir = runtime_dir_path("host")
    if runtime_dir:
        stop_infra(runtime_dir)
        stop_pid(runtime_dir / ".backend.pid")
        stop_pid(runtime_dir / ".frontend.pid")
    remove_host_service()
    remove_runtime_dir("host")


def run_client_up(config: ClientConfig, use_service: bool) -> None:
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


def ensure_runtime_dir(kind: str) -> Path:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "vllm_cluster_manager" / kind
    if not runtime_dir.exists():
        runtime_dir.mkdir(parents=True, exist_ok=True)
        copy_assets(kind, runtime_dir)
    return runtime_dir


def runtime_dir_path(kind: str) -> Path | None:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "vllm_cluster_manager" / kind
    if runtime_dir.exists():
        return runtime_dir
    return None


def remove_runtime_dir(kind: str) -> None:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "vllm_cluster_manager" / kind
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


def copy_assets(kind: str, dest: Path) -> None:
    src_root = resources.files("vllm_cluster_manager.assets") / kind
    if not src_root.is_dir():
        raise RuntimeError(f"Missing packaged assets for {kind}.")
    with resources.as_file(src_root) as src_path:
        shutil.copytree(src_path, dest, dirs_exist_ok=True)


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
    missing = []
    for subdir in ("frontend", "backend", "infra"):
        if not (runtime_dir / subdir).exists():
            missing.append(subdir)
    if not missing:
        return
    for subdir in missing:
        copy_assets_subdir("host", subdir, runtime_dir / subdir)


def copy_assets_subdir(kind: str, subdir: str, dest: Path) -> None:
    src_root = resources.files("vllm_cluster_manager.assets") / kind / subdir
    if not src_root.is_dir():
        raise RuntimeError(f"Missing packaged assets for {kind}/{subdir}.")
    with resources.as_file(src_root) as src_path:
        shutil.copytree(src_path, dest, dirs_exist_ok=True)


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
    venv_dir = runtime_dir / ".venv"
    requirements = runtime_dir / "requirements.txt"
    if not venv_dir.exists():
        create_venv(venv_dir)
    if needs_install(requirements, runtime_dir / ".deps.sha256"):
        install_requirements_without_vllm(venv_dir, requirements)
        write_hash_marker(requirements, runtime_dir / ".deps.sha256")
    try:
        install_vllm_wheel(venv_dir)
    except RuntimeError as exc:
        raise RuntimeError(build_vllm_install_error(venv_dir, exc)) from exc
    python_bin = venv_dir / "bin" / "python"
    if not vllm_installed(python_bin):
        raise RuntimeError(
            "vLLM is not installed in the client runtime venv.\n"
            f"Checked: {python_bin}\n"
            "Delete the venv and rerun `vllm-cluster-manager client up` to retry."
        )


def ensure_venv(venv_dir: Path, requirements: Path, marker: Path) -> None:
    if not venv_dir.exists():
        create_venv(venv_dir)
    if needs_install(requirements, marker):
        install_requirements(venv_dir, requirements)
        write_hash_marker(requirements, marker)


def create_venv(venv_dir: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        run([uv, "venv", "--python=3.12", str(venv_dir)])
        return
    run([sys.executable, "-m", "venv", str(venv_dir)])


def install_requirements(venv_dir: Path, requirements: Path) -> None:
    python_bin = venv_dir / "bin" / "python"
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", str(python_bin), "-r", str(requirements)])
        return
    run([str(python_bin), "-m", "pip", "install", "-r", str(requirements)])


def install_requirements_without_vllm(venv_dir: Path, requirements: Path) -> None:
    filtered = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("vllm"):
            continue
        filtered.append(line)
    tmp = venv_dir / "requirements-no-vllm.txt"
    tmp.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    try:
        install_requirements(venv_dir, tmp)
    finally:
        tmp.unlink(missing_ok=True)


def install_vllm_wheel(venv_dir: Path) -> None:
    python_bin = venv_dir / "bin" / "python"
    if vllm_installed(python_bin):
        return

    cuda_version = detect_cuda_version()
    cuda_major, cuda_minor = cuda_version.split(".")
    cuda_compact = int(cuda_major) * 10 + int(cuda_minor)
    cpu_arch = platform.machine()
    vllm_version = fetch_latest_vllm_version()

    wheel_url = (
        "https://github.com/vllm-project/vllm/releases/download/"
        f"v{vllm_version}/vllm-{vllm_version}+cu{cuda_compact}-"
        f"cp38-abi3-manylinux_2_35_{cpu_arch}.whl"
    )

    if not url_exists(wheel_url):
        raise RuntimeError(
            "No vLLM wheel found for CUDA "
            f"{cuda_version} (cu{cuda_compact}) on {cpu_arch}.\n"
            f"Checked: {wheel_url}"
        )

    uv = shutil.which("uv")
    if uv:
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python_bin),
                wheel_url,
                "--extra-index-url",
                f"https://download.pytorch.org/whl/cu{cuda_compact}",
            ]
        )
        return
    run(
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            wheel_url,
            "--extra-index-url",
            f"https://download.pytorch.org/whl/cu{cuda_compact}",
        ]
    )


def build_vllm_install_error(venv_dir: Path, exc: Exception) -> str:
    return "\n".join(
        [
            "Failed to install vLLM into the client runtime venv.",
            f"venv: {venv_dir}",
            "",
            "Reason:",
            str(exc),
            "",
            "Common fixes:",
            "- Ensure NVIDIA drivers are installed and `nvidia-smi` or `nvcc` is available.",
            "- Ensure outbound HTTPS access to GitHub releases and the PyTorch wheel index.",
            "- Ensure your CUDA version and CPU architecture match an available vLLM wheel.",
            "",
            "Retry:",
            f"  rm -rf {venv_dir}",
            "  vllm-cluster-manager client up",
        ]
    )


def detect_cuda_version() -> str:
    for cmd, parser in (
        (["nvcc", "--version"], parse_nvcc_version),
        (["nvidia-smi"], parse_smi_version),
    ):
        if shutil.which(cmd[0]) is None:
            continue
        output = run(cmd, capture=True)
        version = parser(output)
        if version:
            return version
    raise RuntimeError("Unable to detect CUDA version. Ensure nvcc or nvidia-smi is available.")


def vllm_installed(python_bin: Path) -> bool:
    try:
        run([str(python_bin), "-c", "import vllm"], capture=True)
        return True
    except RuntimeError:
        return False


def parse_nvcc_version(output: str) -> str | None:
    for line in output.splitlines():
        if "release" in line:
            parts = line.split("release", maxsplit=1)[-1].strip()
            return parts.split(",", maxsplit=1)[0].strip()
    return None


def parse_smi_version(output: str) -> str | None:
    for line in output.splitlines():
        if "CUDA Version" in line:
            return line.split("CUDA Version:", maxsplit=1)[-1].strip().split()[0]
    return None


def fetch_latest_vllm_version() -> str:
    url = "https://api.github.com/repos/vllm-project/vllm/releases/latest"
    request = Request(url, headers={"User-Agent": "vllm-cluster-manager"})
    data = json.loads(urlopen(request, timeout=15).read().decode("utf-8"))
    tag = data.get("tag_name", "")
    version = tag.lstrip("v")
    if not version:
        raise RuntimeError("Unable to determine latest vLLM version from GitHub releases.")
    return version


def url_exists(url: str) -> bool:
    request = Request(url, method="HEAD", headers={"User-Agent": "vllm-cluster-manager"})
    try:
        with urlopen(request, timeout=10):
            return True
    except Exception:
        return False


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
    run([npm, "install"], cwd=frontend_dir)
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


def stop_infra(runtime_dir: Path) -> None:
    compose_cmd = detect_compose_cmd()
    cmd = compose_cmd.split() + ["down", "-v"]
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
    compose_stop = compose_service_cmd("down -v")
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
        Description=VLLM Cluster Infra (Postgres + Consul)
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
        Description=VLLM Cluster Backend API
        After=network.target {HOST_SERVICE_NAME}-infra.service
        Requires={HOST_SERVICE_NAME}-infra.service

        [Service]
        Type=simple
        WorkingDirectory={runtime_dir}/backend
        EnvironmentFile={runtime_dir}/backend/.env
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
        Description=VLLM Cluster Frontend
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
        Description=VLLM Cluster Client
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory={runtime_dir}
        EnvironmentFile={runtime_dir}/.env
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
