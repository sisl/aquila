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

## Install (systemd)
Run the installer from the `host` directory:

```bash
cd host
bash install.sh
```

The script prompts for:
- Consul HTTP port (host port)
- Frontend exposed port
- Backend bind host/port
- Postgres host/port/db/user/password

It then:
- Writes `host/.env` (infra), `host/backend/.env`, and `host/frontend/.env`.
- Creates a Python venv at `host/backend/.venv` and installs backend deps.
- Installs frontend deps.
- Creates and enables systemd services:
  - `vllm-cluster-infra.service`
  - `vllm-cluster-backend.service`
  - `vllm-cluster-frontend.service`

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
- `host/.env` for Docker Compose (Postgres + Consul)
- `host/backend/.env` for the API service
- `host/frontend/.env` for the UI

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
