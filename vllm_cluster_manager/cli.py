import argparse
import json
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


@dataclass
class ClientConfig:
    host_ip: str
    consul_port: int
    client_host: str
    client_port: int
    node_name: str


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm_cluster_manager",
        description="vLLM Cluster Manager CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    host_parser = subparsers.add_parser("host", help="Manage the host services")
    host_subparsers = host_parser.add_subparsers(dest="action", required=True)

    host_up = host_subparsers.add_parser("up", help="Install and start host services")
    host_up.add_argument("--host_ip", default="127.0.0.1")
    host_up.add_argument("--host_frontend_port", type=int, default=DEFAULT_FRONTEND_PORT)
    host_up.add_argument("--host_backend_port", type=int, default=None)
    host_up.add_argument("--admin_api_port", type=int, default=DEFAULT_ADMIN_API_PORT)
    host_up.add_argument("--consul_port", type=int, default=None)
    host_up.add_argument("--postgres_host", default=DEFAULT_POSTGRES_HOST)
    host_up.add_argument("--postgres_port", type=int, default=DEFAULT_POSTGRES_PORT)
    host_up.add_argument("--postgres_db", default=DEFAULT_POSTGRES_DB)
    host_up.add_argument("--postgres_user", default=DEFAULT_POSTGRES_USER)
    host_up.add_argument("--postgres_password", default=DEFAULT_POSTGRES_PASSWORD)

    host_down = host_subparsers.add_parser("down", help="Stop host services")

    host_service = host_subparsers.add_parser("service", help="Manage host systemd services")
    host_service_sub = host_service.add_subparsers(dest="service_action", required=True)
    host_service_install = host_service_sub.add_parser("install", help="Install systemd services")
    host_service_install.add_argument("--host_ip", default="127.0.0.1")
    host_service_install.add_argument("--host_frontend_port", type=int, default=DEFAULT_FRONTEND_PORT)
    host_service_install.add_argument("--host_backend_port", type=int, default=None)
    host_service_install.add_argument("--admin_api_port", type=int, default=DEFAULT_ADMIN_API_PORT)
    host_service_install.add_argument("--consul_port", type=int, default=None)
    host_service_install.add_argument("--postgres_host", default=DEFAULT_POSTGRES_HOST)
    host_service_install.add_argument("--postgres_port", type=int, default=DEFAULT_POSTGRES_PORT)
    host_service_install.add_argument("--postgres_db", default=DEFAULT_POSTGRES_DB)
    host_service_install.add_argument("--postgres_user", default=DEFAULT_POSTGRES_USER)
    host_service_install.add_argument("--postgres_password", default=DEFAULT_POSTGRES_PASSWORD)

    host_service_remove = host_service_sub.add_parser("remove", help="Remove systemd services")

    client_parser = subparsers.add_parser("client", help="Manage a client node")
    client_subparsers = client_parser.add_subparsers(dest="action", required=True)

    client_up = client_subparsers.add_parser("up", help="Install and start the client")
    client_up.add_argument("--host_ip", default="127.0.0.1")
    client_up.add_argument("--host_backend_port", type=int, default=None)
    client_up.add_argument("--consul_port", type=int, default=None)
    client_up.add_argument("--client_host", default=DEFAULT_CLIENT_HOST)
    client_up.add_argument("--client_port", type=int, default=DEFAULT_CLIENT_PORT)
    client_up.add_argument("--node_name", default=socket.gethostname())

    client_down = client_subparsers.add_parser("down", help="Stop the client")

    client_service = client_subparsers.add_parser("service", help="Manage client systemd service")
    client_service_sub = client_service.add_subparsers(dest="service_action", required=True)
    client_service_install = client_service_sub.add_parser("install", help="Install systemd service")
    client_service_install.add_argument("--host_ip", default="127.0.0.1")
    client_service_install.add_argument("--host_backend_port", type=int, default=None)
    client_service_install.add_argument("--consul_port", type=int, default=None)
    client_service_install.add_argument("--client_host", default=DEFAULT_CLIENT_HOST)
    client_service_install.add_argument("--client_port", type=int, default=DEFAULT_CLIENT_PORT)
    client_service_install.add_argument("--node_name", default=socket.gethostname())

    client_service_remove = client_service_sub.add_parser("remove", help="Remove systemd service")

    args = parser.parse_args()

    if args.command == "host":
        if args.action == "up":
            host_config = build_host_config(args)
            run_host_up(host_config)
            return
        if args.action == "down":
            run_host_down()
            return
        if args.action == "service":
            if args.service_action == "install":
                host_config = build_host_config(args)
                install_host_service(host_config)
                return
            if args.service_action == "remove":
                remove_host_service()
                return

    if args.command == "client":
        if args.action == "up":
            client_config = build_client_config(args)
            run_client_up(client_config)
            return
        if args.action == "down":
            run_client_down()
            return
        if args.action == "service":
            if args.service_action == "install":
                client_config = build_client_config(args)
                install_client_service(client_config)
                return
            if args.service_action == "remove":
                remove_client_service()
                return

    parser.error("Unknown command")


def build_host_config(args: argparse.Namespace) -> HostConfig:
    consul_port = args.consul_port
    if consul_port is None:
        consul_port = args.host_backend_port or DEFAULT_CONSUL_PORT
    return HostConfig(
        host_ip=args.host_ip,
        frontend_port=args.host_frontend_port,
        admin_api_port=args.admin_api_port,
        consul_port=consul_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_db=args.postgres_db,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
    )


def build_client_config(args: argparse.Namespace) -> ClientConfig:
    consul_port = args.consul_port
    if consul_port is None:
        consul_port = args.host_backend_port or DEFAULT_CONSUL_PORT
    return ClientConfig(
        host_ip=args.host_ip,
        consul_port=consul_port,
        client_host=args.client_host,
        client_port=args.client_port,
        node_name=args.node_name,
    )


def run_host_up(config: HostConfig) -> None:
    runtime_dir = ensure_runtime_dir("host")
    write_host_env_files(runtime_dir, config)
    ensure_backend_venv(runtime_dir)
    ensure_frontend_deps(runtime_dir)
    install_host_service(config)
    systemctl(["enable", "--now", f"{HOST_SERVICE_NAME}-infra.service"])
    systemctl(["enable", "--now", f"{HOST_SERVICE_NAME}-backend.service"])
    systemctl(["enable", "--now", f"{HOST_SERVICE_NAME}-frontend.service"])


def run_host_down() -> None:
    systemctl(["disable", "--now", f"{HOST_SERVICE_NAME}-frontend.service"])
    systemctl(["disable", "--now", f"{HOST_SERVICE_NAME}-backend.service"])
    systemctl(["disable", "--now", f"{HOST_SERVICE_NAME}-infra.service"])


def run_client_up(config: ClientConfig) -> None:
    runtime_dir = ensure_runtime_dir("client")
    write_client_env_file(runtime_dir, config)
    ensure_client_venv(runtime_dir)
    install_client_service(config)
    systemctl(["enable", "--now", f"{CLIENT_SERVICE_NAME}.service"])


def run_client_down() -> None:
    systemctl(["disable", "--now", f"{CLIENT_SERVICE_NAME}.service"])


def ensure_runtime_dir(kind: str) -> Path:
    base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    runtime_dir = base_dir / "vllm_cluster_manager" / kind
    if not runtime_dir.exists():
        runtime_dir.mkdir(parents=True, exist_ok=True)
        copy_assets(kind, runtime_dir)
    return runtime_dir


def copy_assets(kind: str, dest: Path) -> None:
    src_root = resources.files("vllm_cluster_manager.assets") / kind
    if not src_root.is_dir():
        raise RuntimeError(f"Missing packaged assets for {kind}.")
    with resources.as_file(src_root) as src_path:
        shutil.copytree(src_path, dest, dirs_exist_ok=True)


def write_host_env_files(runtime_dir: Path, config: HostConfig) -> None:
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
        """
    ).strip() + "\n"
    (runtime_dir / "frontend" / ".env").write_text(frontend_env, encoding="utf-8")


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
    ensure_venv(venv_dir, requirements)


def ensure_client_venv(runtime_dir: Path) -> None:
    venv_dir = runtime_dir / ".venv"
    requirements = runtime_dir / "requirements.txt"
    if not venv_dir.exists():
        create_venv(venv_dir)
    install_requirements_without_vllm(venv_dir, requirements)
    install_vllm_wheel(venv_dir)


def ensure_venv(venv_dir: Path, requirements: Path) -> None:
    if not venv_dir.exists():
        create_venv(venv_dir)
    install_requirements(venv_dir, requirements)


def create_venv(venv_dir: Path) -> None:
    run([sys.executable, "-m", "venv", str(venv_dir)])


def install_requirements(venv_dir: Path, requirements: Path) -> None:
    python_bin = venv_dir / "bin" / "python"
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

    python_bin = venv_dir / "bin" / "python"
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
    run([npm, "install"], cwd=runtime_dir / "frontend")


def install_host_service(config: HostConfig) -> None:
    runtime_dir = ensure_runtime_dir("host")
    compose_cmd = detect_compose_cmd()
    npm_path = shutil.which("npm")
    if not npm_path:
        raise RuntimeError("npm is required to run the frontend service.")

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
        ExecStart={compose_cmd} up -d
        ExecStop={compose_cmd} down
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
        ExecStart={npm_path} run dev -- --host ${{FRONTEND_HOST}} --port ${{FRONTEND_PORT}}
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
    remove_systemd_service(f"/etc/systemd/system/{CLIENT_SERVICE_NAME}.service")
    systemctl(["daemon-reload"])


def write_systemd_service(path: str, content: str) -> None:
    if os.geteuid() == 0:
        Path(path).write_text(content, encoding="utf-8")
        return
    run(["sudo", "tee", path], input_text=content)


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


def detect_compose_cmd() -> str:
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return docker_compose
    docker = shutil.which("docker")
    if docker:
        try:
            run([docker, "compose", "version"], capture=True)
            return f"{docker} compose"
        except RuntimeError:
            pass
    raise RuntimeError("Docker Compose not found. Install docker compose or docker-compose.")


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False, input_text: str | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
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
