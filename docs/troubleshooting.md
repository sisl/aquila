# Troubleshooting

This section captures the most common pitfalls and how to resolve them quickly.

## Client does not appear in the UI
**Symptoms**: Client starts, but no node shows up in the dashboard.

Checks:
- Verify the client can reach the host on `--host-discover-port`.
- Confirm the host is reachable from the client (use a non-loopback `--host-ip`).
- Check firewall rules between client and host.

## Node stuck as "critical" after decommissioning
**Symptoms**: A node whose agent was shut down (or that no longer exists) stays in the Nodes table with status `critical`.

Nodes are created automatically from discovery but never removed automatically. Open the node's **Manage** dialog and click **Remove Node** — this deletes the node, its deployment records, and its discovery registration. If the agent is actually still running, the node simply re-registers within seconds (no harm done).

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

## Gateway returns 404 / 503 / 502
**Symptoms**: Requests to `http://<host>/v1/...` fail even though deployments exist.

Explanation:
- **404** — no *running* deployment serves the requested `model`; the error message lists the available names. Check the `model` field against the served model name (or LoRA adapter name) shown in `GET /v1/models`.
- **503** — a matching deployment exists but is still `starting`/`loading`; retry once it is `running`.
- **502** — the deployment's node did not respond; check that the host can reach the client node and that the vLLM container is healthy.
- If `/v1` itself is not found behind a reverse proxy, make sure the proxy forwards `/v1` (or `<base-path>/v1`) to the backend, like `/api` and `/ws`.

## Local model path rejected
**Symptoms**: Starting a deployment with an absolute model path fails with an "allowed model directories" error.

Checks:
- Set `MODEL_DIRS` in the client agent's `.env` to a comma-separated list of directories that may be served, then restart the agent.
- The path must exist on the **client node** and resolve to a location inside one of those directories (symlinks escaping them are rejected).

## Webhook notifications not arriving
**Symptoms**: Deployments change state but no Slack/webhook messages appear.

Checks:
- Set the webhook URL in the dashboard (Settings → Notifications; applies live) or as the `WEBHOOK_URL` default in `host/backend/.env`.
- Verify the URL with a manual `curl -X POST -d '{"text":"test"}'`.
- Delivery is fire-and-forget: failures are logged by the backend but never retried or surfaced in the UI.

## Deployments missing after backend restart
**Symptoms**: Running models disappear from the dashboard after restarting the host backend.

Explanation:
- The sync loop automatically rediscovers running deployments from clients within ~10 seconds. If deployments still don't appear, check that the client nodes are reachable from the host.

## Data disappears after `host down`
**Symptoms**: Previously created deployments are gone after shutdown.

Explanation:
- Host data persists across `host down` by default. Data is only wiped by `host down --purge`, `vllm-cluster-manager clean`, or the dashboard's Settings → Data → Purge.
- If data vanished without one of those, check whether the Postgres volume (`host_pgdata`) still exists: `docker volume ls`.

!!! tip
    When debugging, start the host in the foreground to see backend and frontend logs in the terminal.

## Podman is installed but not detected
**Symptoms**: The node's Manage dialog doesn't list `podman` even though Podman is installed.

The agent talks to Podman through its Docker-compatible API socket — **installing the podman package does not start it**. The two common causes:

- The user socket service is enabled but not running (`systemctl --user status podman.socket` shows `inactive (dead)`). Start it: `systemctl --user enable --now podman.socket`. If the agent runs as a systemd service for a user that isn't logged in, also enable lingering: `loginctl enable-linger <user>`.
- Only the **rootful** socket (`/run/podman/podman.sock`, owned `root:root`) exists, which the agent user cannot access. Use the rootless socket instead (command above); the agent logs a warning naming this case.

A non-standard socket path can be supplied via `PODMAN_SOCK` in the client `.env`. Detection refreshes within ~1 minute of the socket appearing.

Version note: Podman 3.x works for deployments, but its API streams no byte-level pull progress (the status chip shows `pulling image` without GB figures) and GPU support via CDI needs Podman ≥ 4.1 — prefer Podman 4+ where available.

## GPU deployment fails on a Podman node
**Symptoms**: Deployments on a Podman node error immediately; the log mentions devices or CDI.

Podman uses CDI for NVIDIA GPU access. Install the NVIDIA Container Toolkit and generate the CDI specs: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, then redeploy.
