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
- If you are on an unusual CUDA version, consider installing a supported version or building vLLM from source.

## Data disappears after `host down`
**Symptoms**: Previously created deployments are gone after shutdown.

Explanation:
- `host down` runs `docker compose down -v`, which wipes the Postgres volume.
- Remove `-v` in code if you want persistent data.

!!! tip
    When debugging, start the host in the foreground to see backend and frontend logs in the terminal.
