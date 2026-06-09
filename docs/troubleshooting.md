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

## Docker daemon not reachable
**Symptoms**: Client install or a deployment fails with "Cannot talk to the Docker daemon".

Checks:
- Verify Docker is running: `docker info`.
- Ensure the client user can use Docker without sudo: `sudo usermod -aG docker "$USER"` then log out/in (or run the client as root).
- The client systemd unit runs as the installing user — that user must be in the `docker` group.

## GPUs not visible to containers
**Symptoms**: A deployment fails to start, or the container cannot see the GPUs.

Checks:
- Confirm the NVIDIA Container Toolkit is installed and configured: `docker run --rm --gpus all ubuntu nvidia-smi` should list your GPUs.
- If it fails, install the toolkit and run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

## Image pull fails
**Symptoms**: The deployment log shows a "Failed to pull image" error.

Checks:
- Confirm the requested vLLM version exists as a tag on [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) (releases use the `v<version>` form; commits use `nightly-<commit>`).
- Ensure the node has outbound network access to Docker Hub. The first pull is multi-GB and may take a while; progress is streamed to the deployment log.

## Deployment stuck in "loading"
**Symptoms**: A deployment stays in the `loading` state and never transitions to `running`.

Checks:
- Open the deployment logs to see image pull/build (`[docker]`) or vLLM startup errors.
- Common causes: a large image still pulling, insufficient GPU memory, model not found on Hugging Face, or missing `HF_TOKEN` for gated models.
- The readiness check polls `/health` and `/v1/models` on the deployment port. With host networking the container binds the node port directly; ensure no firewall blocks localhost access on the client node.

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
