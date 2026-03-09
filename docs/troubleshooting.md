# Troubleshooting

This section captures the most common pitfalls and how to resolve them quickly.

## Client does not appear in the UI
**Symptoms**: Client starts, but no node shows up in the dashboard.

Checks:
- Verify the client can reach the host on `--host-discover-port`.
- Confirm the host is reachable from the client (use a non-loopback `--host-ip`).
- Check firewall rules between client and host.

## UI loads but data is empty
**Symptoms**: UI opens, but no nodes or deployments show up.

Checks:
- Ensure the backend is running on `--host-backend-port`.
- Verify `VITE_BACKEND_HOST` in `host/frontend/.env` matches your host IP.
- If you changed ports, restart the frontend service.

## Blank page behind reverse proxy
**Symptoms**: The UI shows a blank page when accessed through Nginx or another reverse proxy.

Checks:
- If you proxy under a path (for example `/vllm/`), set `VITE_BASE_PATH=/vllm/` in `host/frontend/.env`.
- Restart the frontend service so Vite picks up the new base path.

## Consul port confusion
**Symptoms**: Clients fail to register when using the Consul default port (8500).

Explanation:
- The host maps Consul's container port `8500` to a host port (default `47528`).
- Clients must use the host port (`--host-discover-port`, default `47528`).

## CUDA detection fails
**Symptoms**: Client install fails with an error about CUDA detection.

Checks:
- Ensure `nvcc` or `nvidia-smi` is on PATH.
- Verify NVIDIA drivers are installed and the GPU is visible.

## vLLM wheel not found
**Symptoms**: Install fails after detecting CUDA.

Checks:
- The vLLM wheel must exist for your CUDA version and CPU architecture.
- If no wheel matches your exact CUDA version, the installer automatically tries the highest compatible version. If that also fails, consider installing a supported CUDA version or building vLLM from source.

## `uv` not found when running as a systemd service
**Symptoms**: Deploying a model fails with "uv is not installed or not on PATH".

Explanation:
- Systemd services run with a minimal PATH that may not include user-local directories.

Checks:
- Verify `uv` is installed: `which uv` or check `~/.local/bin/uv` and `~/.cargo/bin/uv`.
- The client automatically searches common locations (`/usr/local/bin`, `/usr/bin`, `/home/*/.local/bin`, `/home/*/.cargo/bin`), but if `uv` is installed elsewhere, add its directory to the systemd service's `Environment=PATH=...` line.

## Deployment stuck in "loading"
**Symptoms**: A deployment stays in the `loading` state and never transitions to `running`.

Checks:
- Open the deployment logs to see venv creation or vLLM startup errors.
- Common causes: network issues downloading packages, insufficient GPU memory, model not found on Hugging Face, or missing `HF_TOKEN` for gated models.
- The readiness check polls `/health` and `/v1/models` on the deployment port. Ensure no firewall blocks localhost access on the client node.

## Deployments missing after backend restart
**Symptoms**: Running models disappear from the dashboard after restarting the host backend.

Explanation:
- The sync loop automatically rediscovers running deployments from clients within ~10 seconds. If deployments still don't appear, check that the client nodes are reachable from the host.

## Data disappears after `host down`
**Symptoms**: Previously created deployments are gone after shutdown.

Explanation:
- `host down` runs `docker compose down -v`, which wipes the Postgres volume.
- Remove `-v` in code if you want persistent data.

!!! tip
    When debugging, start the host in the foreground to see backend and frontend logs in the terminal.
