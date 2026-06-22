# Getting Started

This guide takes you from a clean host to a working cluster with at least one client.

## Prerequisites
Host:
- Docker + Docker Compose plugin (configure the `docker` group so no sudo is required)
- Node.js + npm
- Python 3.10–3.14
- `uv` (Python package manager)

Client:
- NVIDIA GPU with a recent driver (`nvidia-smi` working)
- A container runtime: **Docker Engine** (add the client user to the `docker` group) **or Podman ≥ 5.4** (rootless works; enable the API socket with `systemctl --user enable --now podman.socket`; older Podman cannot pass GPUs through its Docker-compatible API). vLLM runs in the official `vllm/vllm-openai` containers either way.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access. On Podman nodes, additionally generate CDI specs: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
- Python 3.10–3.14 (for the lightweight client agent)
- `uv` (Python package manager)

Install `uv` if you don't already have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify Docker can see the GPUs before installing the client:
```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

## Install (uv)
Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

```bash
uv pip install athanor
```

## Start the host
Foreground (no sudo):
```bash
athanor host up --host-ip 0.0.0.0 --host-frontend-port 5173 --host-discover-port 11400
```

Persistent service (systemd):
```bash
athanor host up --service --host-ip 0.0.0.0 --host-frontend-port 5173 --host-discover-port 11400
```

`--host-discover-port` sets the discovery port used for clients. Use `--host-backend-port` to override the backend API port (default 8000).

## Start a client
Foreground (no sudo):
```bash
athanor client up --host-ip 1.2.3.4 --host-discover-port 11400
```

Persistent service (systemd):
```bash
athanor client up --service --host-ip 1.2.3.4 --host-discover-port 11400
```

!!! note
    If the client cannot register, verify firewall rules and that the host is reachable from the client on the discovery port.

## Stop services
```bash
athanor host down
athanor client down
```

Host data (deployments, nodes, history) survives `host down` and comes back on the next `host up`. Pass `--purge` to `host down` to also delete the Postgres volume — see [Operations → Data persistence](operations.md#data-persistence).

## Verify the UI
Open the UI at `http://<host-ip>:<host-frontend-port>`.

Common first-run checks:
- The UI loads without a network error.
- The host shows up as healthy.
- The client appears under Nodes within ~30 seconds.

!!! note
    On first run the backend creates a default **admin** API key and prints it to the startup log. Store this key — it is required for gateway requests (`Authorization: Bearer vcm-...`) and will not be shown again. You can manage keys later in Settings → Gateway & Keys.

## Next steps
Once a client node appears in the dashboard, you are ready to deploy models. See the [Deployments](deployments.md) page for details on vLLM version selection, engine options, GPU assignment, local checkpoints, and LoRA adapters. When a model is running, the **Endpoint** button gives you copy-paste connection snippets — see [Gateway & Usage](gateway.md) for the cluster-wide OpenAI-compatible API and token accounting.
