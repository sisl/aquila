# Getting Started

This guide takes you from a clean host to a working cluster with at least one client.

## Prerequisites
Host:
- Docker + Docker Compose plugin (configure the `docker` group so no sudo is required)
- Node.js + npm
- Python 3.12
- `uv` (Python package manager)

Client:
- NVIDIA GPU with CUDA
- `nvcc` or `nvidia-smi` on PATH
- Python 3.12 + `python3.12-dev` and `build-essential` (Debian/Ubuntu)
- `uv` (Python package manager)

Install `uv` if you don't already have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Debian/Ubuntu:
```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

## Install (pip)
Create and activate a Python 3.12 virtual environment:
```bash
uv venv --python=3.12
source .venv/bin/activate
```

```bash
uv pip install vllm-cluster-manager
```

## Start the host
Foreground (no sudo):
```bash
vllm-cluster-manager host up --host-ip 127.0.0.1 --host-frontend-port 5173 --host-discover-port 47528
```

Persistent service (systemd):
```bash
vllm-cluster-manager host up --service --host-ip 127.0.0.1 --host-frontend-port 5173 --host-discover-port 47528
```

`--host-discover-port` sets the discovery port used for clients. Use `--host-backend-port` to override the backend API port (default 8000).

## Start a client
Foreground (no sudo):
```bash
vllm-cluster-manager client up --host-ip 127.0.0.1 --host-discover-port 47528
```

Persistent service (systemd):
```bash
vllm-cluster-manager client up --service --host-ip 127.0.0.1 --host-discover-port 47528
```

!!! note
    If the client cannot register, verify firewall rules and that the host is reachable from the client on the discovery port.

## Stop services
```bash
vllm-cluster-manager host down
vllm-cluster-manager client down
```

## Verify the UI
Open the UI at `http://<host-ip>:<host-frontend-port>`.

Common first-run checks:
- The UI loads without a network error.
- The host shows up as healthy.
- The client appears under Nodes within ~30 seconds.
