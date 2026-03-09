# Operations

## Configuration files
The CLI writes service-specific env files under `~/.local/share/vllm_cluster_manager`:
- `host/.env` (Docker compose: Postgres + discovery service)
- `host/backend/.env` (API service)
- `host/frontend/.env` (UI)
- `client/.env` (client agent)

If you edit any env file, restart the affected service.

## Gated models (Huggingface)
Some models (for example Llama variants) require a Hugging Face access token. Provide the token via an env var when creating the deployment:
- `HF_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`

Set the value to your Hugging Face access token (read access) and include quotation marks, for example:
```
HUGGING_FACE_HUB_TOKEN="hf_..."
```

You can add this in the UI under env vars or by setting it in the client environment before starting a deployment.

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
sudo systemctl restart vllm-cluster-infra.service
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
The client bootstrapper detects CUDA from `nvcc` or `nvidia-smi` and installs a vLLM wheel that matches the detected version. If no wheel exists for your exact CUDA version, the installer automatically falls back to the highest compatible CUDA wheel and displays a warning.

## Per-deployment venvs
Each deployment creates an isolated virtual environment under `~/.vllm-client/.venvs/`. Venvs are managed with `uv` for fast creation and installs. When a deployment is stopped, its venv is automatically removed. Cached venvs are reused when deploying the same version + model + port combination.

You can inspect and manage venvs via the client API:

- `GET /venvs` — list all cached venvs.
- `DELETE /venvs/<id>` — remove a specific venv.

## Uploaded packages and plugins
Uploaded files (`.py`, `.whl`, `.tar.gz`, `.zip`) are stored under `~/.vllm-client/.packages/`. Each upload is content-hashed to avoid duplicates.

- `GET /packages` — list uploaded packages.
- `DELETE /packages/<id>` — remove a specific package.

## Deployment recovery
The backend sync loop runs every 5 seconds and polls each client for its current deployments. If the backend is restarted while models are running on clients, the sync loop automatically recreates the database entries so the dashboard reflects the actual cluster state. Deployments in `stopped` or `error` state on the client are not re-imported.
