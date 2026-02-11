# Host (Admin Services)

The host runs infrastructure, the admin API, and the web UI.

## Components
- **Infra**: Postgres + Consul via Docker Compose (`host/docker-compose.yml`).
- **Backend**: FastAPI API service (`host/backend`).
- **Frontend**: React + Vite admin dashboard (`host/frontend`).

## Prerequisites
- Docker + Docker Compose plugin (or `docker-compose`).
- Node.js + npm.
- Python 3.12 and `uv` (installer will install `uv` if missing).
- `systemctl` (systemd) for service management.

## Install (systemd via CLI)
Install the CLI and start the host services:

```bash
uv pip install vllm_cluster_manager
vllm_cluster_manager host up --host_ip 127.0.0.1 --host_frontend_port 5173 --host_backend_port 47528
```
The CLI uses systemd and may prompt for sudo to create and enable services.

Stop services with:
```bash
vllm_cluster_manager host down
```

If you only want to install the systemd units without starting them immediately:
```bash
vllm_cluster_manager host service install
```

Remove systemd units:
```bash
vllm_cluster_manager host service remove
```

Key options:
- `--host_ip`: backend bind host and UI backend host.
- `--host_frontend_port`: UI port (default 5173).
- `--host_backend_port`: Consul HTTP port (used for client discovery).
- `--admin_api_port`: backend API port (default 8000).
- `--postgres_*`: Postgres configuration.

The CLI writes env files, installs backend/frontend deps, and creates/enables systemd services.

Runtime files are stored under `~/.local/share/vllm_cluster_manager/host`.

### Check status
```bash
sudo systemctl status vllm-cluster-infra.service
sudo systemctl status vllm-cluster-backend.service
sudo systemctl status vllm-cluster-frontend.service
```

### View logs
```bash
journalctl -u vllm-cluster-infra.service -f
journalctl -u vllm-cluster-backend.service -f
journalctl -u vllm-cluster-frontend.service -f
```

## Local development
1) Infra:
```bash
cd host
cp .env.example .env
# edit .env for passwords

docker compose up -d
```

2) Backend:
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

The UI is available at `http://localhost:5173` by default (see `host/frontend/.env`).

## Configuration
The installer writes:
- `~/.local/share/vllm_cluster_manager/host/.env` for Docker Compose (Postgres + Consul)
- `~/.local/share/vllm_cluster_manager/host/backend/.env` for the API service
- `~/.local/share/vllm_cluster_manager/host/frontend/.env` for the UI

If you edit any env file, restart the related systemd service.

## Firewall notes
Typical ports to allow (adjust to your choices):
- Frontend: `FRONTEND_PORT`
- Backend API: `ADMIN_API_PORT`
- Consul (host API for clients): `CONSUL_PORT`
- Postgres: `POSTGRES_PORT` (only if remote access is required)

## Uninstall (systemd)
```bash
sudo systemctl disable --now vllm-cluster-frontend.service
sudo systemctl disable --now vllm-cluster-backend.service
sudo systemctl disable --now vllm-cluster-infra.service
sudo rm -f /etc/systemd/system/vllm-cluster-frontend.service
sudo rm -f /etc/systemd/system/vllm-cluster-backend.service
sudo rm -f /etc/systemd/system/vllm-cluster-infra.service
sudo systemctl daemon-reload
```

To wipe Postgres data (Docker volume):
```bash
cd host
docker compose down -v
```
