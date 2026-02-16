# vLLM Cluster Manager

[![Docs](https://img.shields.io/badge/docs-online-30a2ff)](https://sisl.github.io/VLLMClusterManager/)
[![PyPI](https://img.shields.io/pypi/v/vllm-cluster-manager?color=30a2ff)](https://pypi.org/project/vllm-cluster-manager/)
[![Python](https://img.shields.io/badge/python-3.12%2B-30a2ff)](https://pypi.org/project/vllm-cluster-manager/)

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
- Docker + Docker Compose plugin (configure the `docker` group so no sudo is required).
- Node.js + npm.
- Python 3.12.
- `uv` (Python package manager).

Client:
- NVIDIA GPU with CUDA.
- `nvcc` or `nvidia-smi` on PATH (used to detect CUDA version).
- Python 3.12 + `python3.12-dev` and `build-essential` (Debian/Ubuntu).

On Debian/Ubuntu:
```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

Install `uv` if you don't already have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install (pip)
Create and activate a Python 3.12 virtual environment:
```bash
uv venv --python=3.12
source .venv/bin/activate
```

```bash
uv pip install vllm-cluster-manager
```

## Start the host
Foreground (no sudo):
```bash
vllm-cluster-manager host up --host-ip 127.0.0.1 --host-frontend-port 5173 --host-discover-port 47528
```

`host up` builds a static frontend bundle and serves it with the Vite preview server.

Persistent service (systemd):
```bash
vllm-cluster-manager host up --service --host-ip 127.0.0.1 --host-frontend-port 5173 --host-discover-port 47528
```

`--host-discover-port` sets the discovery port used for clients. Use `--host-backend-port` to override the backend API port (default 8000).

Stop host services (foreground or systemd):
```bash
vllm-cluster-manager host down
```

## Start a client
Foreground (no sudo):
```bash
vllm-cluster-manager client up --host-ip 127.0.0.1 --host-discover-port 47528
```

Persistent service (systemd):
```bash
vllm-cluster-manager client up --service --host-ip 127.0.0.1 --host-discover-port 47528
```

Stop client services (foreground or systemd):
```bash
vllm-cluster-manager client down
```

## CLI flags
**Host (`host up`)**

| Flag | Default | Description |
| --- | --- | --- |
| `--service` | `false` | Run as a persistent systemd service. |
| `--host-ip` | `127.0.0.1` | Bind host for the backend API and UI backend target. |
| `--host-frontend-port` | `5173` | UI port. |
| `--host-discover-port` | `47528` | Discovery port used by clients. |
| `--host-backend-port` | `8000` | Backend API port. |
| `--postgres-host` | `127.0.0.1` | Postgres host. |
| `--postgres-port` | `5757` | Postgres port. |
| `--postgres-db` | `vllm_admin` | Postgres database name. |
| `--postgres-user` | `vllm` | Postgres user. |
| `--postgres-password` | `change-me` | Postgres password. |

**Client (`client up`)**

| Flag | Default | Description |
| --- | --- | --- |
| `--service` | `false` | Run as a persistent systemd service. |
| `--host-ip` | `127.0.0.1` | Host IP for discovery. |
| `--host-discover-port` | `47528` | Host discovery port. |
| `--client-host` | `0.0.0.0` | Client bind host. |
| `--client-port` | `9000` | Client bind port. |
| `--node-name` | `<hostname>` | Node name used for registration. |

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
- User → Host UI: TCP `host-frontend-port` (default 5173).
- UI/Browser → Host API: TCP `host-backend-port` (default 8000).
- Clients → Host discovery port: TCP `host-discover-port` (default 47528).
- Host → Client agents: TCP `client-port` (default 9000).

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
