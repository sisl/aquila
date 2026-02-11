# vLLM Cluster Manager

![VLLM Cluster Manager overview UI](img/vllm-cluster-manager-screenshot.png "VLLM Cluster Manager User Interface")

Admin dashboard + satellite clients for multi-model vLLM deployments.

Use this UI to deploy vLLM `serve` endpoints across a cluster so you can stand up multiple LLM servers (same or different models) with a few clicks. It is ideal for research labs or small business environments that need repeatable, multi-endpoint deployments without building a full MLOps stack.

Deployment is as simple as running the install script on the host and on each client, with automatic client discovery via Consul.

Use the host UI to register GPU nodes, define model configurations, launch/stop workloads, and monitor health and logs in real time. Systemd services are enabled on install, so they automatically restart after a system reboot.

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
- **Client**: Python agent running on GPU nodes; registers with Consul and runs vLLM workloads.

## Repo layout
- `host/` Admin services (infra, backend, frontend)
- `client/` Satellite node agent
- `img/` Screenshots used in documentation

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

## Install (systemd services)
Use the host installer for a production-style setup (infra + backend + frontend services):

```bash
cd host
bash install.sh
```

The installer prompts for ports and writes env files for each service. See `host/README.md` for details.

## Client install
Clients are installed separately on each GPU node:

```bash
cd client
bash install.sh
```

See `client/README.md` for prerequisites, configuration, and troubleshooting.

## Configuration files
The installers write service-specific env files:
- `host/.env` (Docker compose: Postgres + Consul)
- `host/backend/.env` (API service)
- `host/frontend/.env` (UI)
- `client/.env` (client agent)

If you edit any env file, restart the affected service.

## Notes
- The service registry is Consul (used for client discovery).
- WebSocket log streaming is handled in `host/frontend/src/services/ws.ts`.
