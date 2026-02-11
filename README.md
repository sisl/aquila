# vLLM Cluster Manager (Scaffold)

Admin dashboard + satellite clients for multi-model vLLM deployments.

## Structure
- `host/` Admin services (infra, backend, frontend)
- `client/` Satellite node agent

## Quick start (dev)
1) Start infrastructure:

```bash
cd /home/mschlichting/vllm_cluster_manager/host
cp .env.example .env
# edit .env for passwords

docker compose up -d
```

2) Backend (venv recommended):

```bash
cd /home/mschlichting/vllm_cluster_manager/host/backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3) Frontend:

```bash
cd /home/mschlichting/vllm_cluster_manager/host/frontend
npm install
npm run dev
```

## Install (systemd services)
Use the root installer to run infra (Postgres + Consul), backend, and frontend as
systemd services. The installer prompts for the host port and the frontend
exposed port to accommodate firewall restrictions.

```bash
cd /home/mschlichting/vllm_cluster_manager/host
bash install.sh
```

What the installer does:
- Writes `.env` for Docker (Postgres + Consul host port).
- Creates `backend/.env` and `frontend/.env`.
- Installs backend deps into `backend/.venv` using `uv`.
- Installs frontend deps with `npm install`.
- Creates and enables systemd services:
  - `vllm-cluster-infra.service`
  - `vllm-cluster-backend.service`
  - `vllm-cluster-frontend.service`

Check status:
```bash
sudo systemctl status vllm-cluster-infra.service
sudo systemctl status vllm-cluster-backend.service
sudo systemctl status vllm-cluster-frontend.service
```

View logs:
```bash
journalctl -u vllm-cluster-backend.service -f
journalctl -u vllm-cluster-frontend.service -f
```

## Configuration files
The installer writes:
- `/host/.env` for Docker compose:
  - `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`
  - `CONSUL_PORT` (host port mapped to the service registry on 8500)
- `/host/backend/.env` for the API:
  - `ADMIN_API_HOST` (defaults to `127.0.0.1`), `ADMIN_API_PORT`
  - `POSTGRES_*`
  - `CONSUL_HTTP_ADDR` (points at the host service registry port)
- `/host/frontend/.env` for the UI:
  - `FRONTEND_HOST`, `FRONTEND_PORT`
  - `VITE_BACKEND_HOST`, `VITE_BACKEND_PORT`

If you edit these files, restart the affected service:
```bash
sudo systemctl restart vllm-cluster-backend.service
sudo systemctl restart vllm-cluster-frontend.service
sudo systemctl restart vllm-cluster-infra.service
```

## Firewall notes
Typical ports to allow (adjust to your choices):
- Frontend: `FRONTEND_PORT`
- Backend API: `ADMIN_API_PORT`
- Host API (for clients): `CONSUL_PORT`
- Postgres: `POSTGRES_PORT` (only if remote access is required)

## Client install
Satellite clients are installed separately:
```bash
cd client
bash install.sh
```
The installer prompts for the client URI/port and host URI/port. Ensure the client
port is reachable via a TCP firewall rule from the host.
Prereqs for the client:
- CUDA 12.8, 12.9, or 13.0 (vLLM wheels are only published for these versions right now).
- System packages: `python3.12-dev` and `build-essential`.
On Debian/Ubuntu:
```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```


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

To wipe Postgres data (docker volume):
```bash
cd /home/mschlichting/vllm_cluster_manager/host
docker compose down -v
```

## Notes
- WebSocket is native FastAPI for now. Swap to Socket.IO if you prefer; the API surface is isolated in `host/frontend/src/services/ws.ts`.
- Service registry refers to Consul (used for client discovery).
