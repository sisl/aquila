# Getting Started

## Prerequisites
Host:
- Docker + Docker Compose plugin
- Node.js + npm
- Python 3.12

Client:
- NVIDIA GPU with CUDA
- `nvcc` or `nvidia-smi` on PATH
- Python 3.12 + `python3.12-dev` and `build-essential` (Debian/Ubuntu)

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

Stop host services (foreground or systemd):
```bash
vllm-cluster-manager host down
```

## Start a client
Foreground (no sudo):
```bash
vllm-cluster-manager client up --host-ip 127.0.0.1 --host-discover-port 47528
```

Persistent service (systemd):
```bash
vllm-cluster-manager client up --service --host-ip 127.0.0.1 --host-discover-port 47528
```

Stop client services (foreground or systemd):
```bash
vllm-cluster-manager client down
```
