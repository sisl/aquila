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

## Service management
Systemd unit names (service mode):
- `vllm-cluster-infra.service`
- `vllm-cluster-backend.service`
- `vllm-cluster-frontend.service`
- `vllm-cluster-client.service`

Frontend behavior:
- `host up` builds a static frontend bundle and serves it with the Vite preview server.
- If you change frontend config or base path, rerun `host up` to rebuild.

Restart flows:
```bash
sudo systemctl restart vllm-cluster-backend.service
sudo systemctl restart vllm-cluster-frontend.service
sudo systemctl restart vllm-cluster-client.service
```

## Host network setup
If the host should be reachable from other machines, use a non-loopback `--host-ip` (for example the host's LAN IP) and ensure firewall rules allow inbound traffic.

## Reverse proxy base path
If you proxy the frontend under a path like `/vllm/`, pass `--base-path /vllm/` when running `host up`. This ensures asset URLs and API/WebSocket paths resolve correctly.

For Nginx, make sure `/vllm/api` and `/vllm/ws` are proxied to the backend (port 8000 by default). The frontend uses the configured base path for API and WebSocket URLs, so it works both at `/` and under a subpath.

## GPU wheel selection
The client bootstrapper detects CUDA from `nvcc` or `nvidia-smi` and installs a vLLM wheel that matches the detected version. If the wheel doesn't exist for your CUDA version, the install fails with a clear error.
