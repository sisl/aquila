# Client Agent

Python service that runs on each GPU node, managing vLLM deployments as Docker/Podman containers.

## What it does

- Registers with the host's Consul service for automatic discovery.
- Starts, stops, and monitors vLLM containers on the node.
- Reports GPU metrics, deployment status, and container health to the host backend.
- Manages local model uploads, image caching, and derived image builds.
- Runs the warm-cache proxy for pause/resume support.

## Running

```bash
aquila client up --host-ip <host> --host-discover-port <port>
```

Or as a systemd service:

```bash
aquila client up --service --host-ip <host> --host-discover-port <port>
```

## Configuration

Environment variables are read from `~/.local/share/aquila/client/.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_DIRS` | *(unset)* | Comma-separated directories for local models/LoRA adapters. |
| `MAX_FAILED_RESTARTS` | `3` | Crash-loop breaker per deployment. |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | Shared HuggingFace model cache. |
| `LOG_MAX_MB` | `50` | Rotate deployment logs at this size. |
| `LOG_RETENTION_DAYS` | `14` | Delete old log files after this many days. |
| `PODMAN_SOCK` | *(auto)* | Non-standard Podman API socket path. |

## Code structure

The agent is a single FastAPI application in `app/main.py` (~5,000 lines). Key sections:

- **Container management** — Docker/Podman client setup, image pull/build, container lifecycle.
- **Deployment endpoints** — `/deployments/start`, `/deployments/stop`, status reporting.
- **Warm cache** — pause/resume via vLLM sleep mode, LRU eviction planner, per-deployment proxy servers.
- **GPU metrics** — NVML/nvidia-smi scraping, unified memory detection.
- **Model management** — local model uploads, HF cache inspection, package storage.
- **Consul registration** — periodic heartbeat to the host's discovery service.
