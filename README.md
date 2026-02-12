# vLLM Cluster Manager

![VLLM Cluster Manager overview UI](img/vllm-cluster-manager-screenshot.png "VLLM Cluster Manager User Interface")

Admin dashboard + satellite clients for multi-model vLLM deployments.

Use this UI to deploy vLLM `serve` endpoints across a cluster so you can stand up multiple LLM servers (same or different models) with a few clicks. It is ideal for research labs or small business environments that need repeatable, multi-endpoint deployments without building a full MLOps stack.

Deployment is as simple as running the CLI on the host and on each client, with automatic client discovery. You can run in the foreground or with `--service` to install persistent systemd services.

## Tested hardware/software
- GPUs: NVIDIA H100, NVIDIA A100, NVIDIA L40, NVIDIA DGX Spark (GB10), NVIDIA RTX 4090.
- OS: Ubuntu 22.04 and Ubuntu 24.04.

## What it can do
- Register and manage GPU nodes that run vLLM workloads.
- Create model configurations and launch models on selected nodes.
- Monitor node health and model status.
- Stream logs from running processes for quick troubleshooting.

## Real-time logs
Stream logs from running nodes and model processes directly in the dashboard.

![Real-time logs window](img/vllm-cluster-manager-terminal.png "Real-time logs")

## Model configuration
Define and manage model settings (weights, runtime settings, resource usage) from the UI.

![Model configuration panel](img/vllm-cluster-manager-model-config.png "Model configuration")

## Architecture
- **Host**: Admin services for infrastructure, API, and UI.
  - **Infra**: Postgres + Consul (service discovery) via Docker Compose.
  - **Backend**: FastAPI service for orchestration and persistence.
  - **Frontend**: React + Vite admin dashboard.
- **Client**: Python agent running on GPU nodes; registers with the host and runs vLLM workloads.

## Repo layout
- `host/` Admin services (infra, backend, frontend)
- `client/` Satellite node agent
- `img/` Screenshots used in documentation

## Prerequisites
Host:
- Docker + Docker Compose plugin.
- Node.js + npm.
- Python 3.12.

Client:
- NVIDIA GPU with CUDA.
- `nvcc` or `nvidia-smi` on PATH (used to detect CUDA version).
- Python 3.12 + `python3.12-dev` and `build-essential` (Debian/Ubuntu).

On Debian/Ubuntu:
```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

## Install (pip)
Create and activate a Python 3.12 virtual environment:
```bash
uv venv --python=3.12
source .venv/bin/activate
```

```bash
uv pip install vllm_cluster_manager
```

## Start the host
Foreground (no sudo):
```bash
vllm_cluster_manager host up --host_ip 127.0.0.1 --host_frontend_port 5173 --host_discover_port 47528
```

Persistent service (systemd):
```bash
vllm_cluster_manager host up --service --host_ip 127.0.0.1 --host_frontend_port 5173 --host_discover_port 47528
```

`--host_discover_port` sets the discovery port used for clients. Use `--host_backend_port` to override the backend API port (default 8000).

Stop host services (foreground or systemd):
```bash
vllm_cluster_manager host down
```

## Start a client
Foreground (no sudo):
```bash
vllm_cluster_manager client up --host_ip 127.0.0.1 --host_discover_port 47528
```

Persistent service (systemd):
```bash
vllm_cluster_manager client up --service --host_ip 127.0.0.1 --host_discover_port 47528
```

Stop client services (foreground or systemd):
```bash
vllm_cluster_manager client down
```

## CLI flags
**Host (`host up`)**

| Flag | Default | Description |
| --- | --- | --- |
| `--service` | `false` | Run as a persistent systemd service. |
| `--host_ip` | `127.0.0.1` | Bind host for the backend API and UI backend target. |
| `--host_frontend_port` | `5173` | UI port. |
| `--host_discover_port` | `47528` | Discovery port used by clients. |
| `--host_backend_port` | `8000` | Backend API port. |
| `--postgres_host` | `127.0.0.1` | Postgres host. |
| `--postgres_port` | `5757` | Postgres port. |
| `--postgres_db` | `vllm_admin` | Postgres database name. |
| `--postgres_user` | `vllm` | Postgres user. |
| `--postgres_password` | `change-me` | Postgres password. |

**Client (`client up`)**

| Flag | Default | Description |
| --- | --- | --- |
| `--service` | `false` | Run as a persistent systemd service. |
| `--host_ip` | `127.0.0.1` | Host IP for discovery. |
| `--host_discover_port` | `47528` | Host discovery port. |
| `--client_host` | `0.0.0.0` | Client bind host. |
| `--client_port` | `9000` | Client bind port. |
| `--node_name` | `<hostname>` | Node name used for registration. |

**Down commands**
- `host down` and `client down` stop foreground processes and remove/stop systemd services if present.

## Configuration files
The CLI writes service-specific env files under `~/.local/share/vllm_cluster_manager`:
- `host/.env` (Docker compose: Postgres + discovery service)
- `host/backend/.env` (API service)
- `host/frontend/.env` (UI)
- `client/.env` (client agent)

If you edit any env file, restart the affected service.

## Firewall rules
Allow these network paths (adjust ports to your flags):
- User → Host UI: TCP `host_frontend_port` (default 5173).
- UI/Browser → Host API: TCP `host_backend_port` (default 8000).
- Clients → Host discovery port: TCP `host_discover_port` (default 47528).
- Host → Client agents: TCP `client_port` (default 9000).

## Data persistence
By default, shutting down the host (`host down` or stopping the systemd infra unit) runs `docker compose down -v`, which wipes the Postgres volume. Remove `-v` in code if you want to keep data.

## Quick start (dev)
1) Start infrastructure:

```bash
cd host
cp .env.example .env
# edit .env for passwords

docker compose up -d
```

2) Backend (venv recommended):

```bash
cd host/backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3) Frontend:

```bash
cd host/frontend
npm install
npm run dev
```

Open the UI at `http://localhost:5173` by default (see `host/frontend/.env`).

## Notes
- The service registry is Consul (used for client discovery).
- WebSocket log streaming is handled in `host/frontend/src/services/ws.ts`.
