# Operations

## Configuration files
The CLI writes service-specific env files under `~/.local/share/vllm_cluster_manager`:
- `host/.env` (Docker compose: Postgres + discovery service)
- `host/backend/.env` (API service)
- `host/frontend/.env` (UI)
- `client/.env` (client agent)

If you edit any env file, restart the affected service.

## Firewall rules
Allow these network paths (adjust ports to your flags):
- User → Host UI: TCP `host-frontend-port` (default 5173)
- UI/Browser → Host API: TCP `host-backend-port` (default 8000)
- Clients → Host discovery port: TCP `host-discover-port` (default 47528)
- Host → Client agents: TCP `client-port` (default 9000)

## Data persistence
By default, shutting down the host (`host down` or stopping the systemd infra unit) runs `docker compose down -v`, which wipes the Postgres volume. Remove `-v` in code if you want to keep data.

## Troubleshooting
- Missing CUDA detection: ensure `nvcc` or `nvidia-smi` is on PATH.
- Frontend fails to start: verify Node.js and npm are installed.
