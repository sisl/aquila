<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/aquila-logo-full-dark.svg" />
    <img src="img/aquila-logo-full.svg" alt="Aquila — GPU Inference Management" height="80" />
  </picture>
</p>

[![Docs](https://img.shields.io/badge/docs-online-111827)](https://sisl.github.io/aquila/)
[![PyPI](https://img.shields.io/pypi/v/aquila?color=111827)](https://pypi.org/project/aquila/)
[![Tests](https://github.com/sisl/aquila/actions/workflows/tests.yml/badge.svg)](https://github.com/sisl/aquila/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.14-111827)](https://pypi.org/project/aquila/)

![Aquila dashboard](img/aquila-screenshot.png "Aquila")

Admin dashboard + satellite clients for multi-model vLLM deployments. Deploy vLLM `serve` endpoints across a cluster with a few clicks — ideal for research labs or small teams that need repeatable, multi-endpoint serving without a full MLOps stack.

## Key features
- Deploy and manage models across GPU nodes via Docker or rootless Podman.
- OpenAI-compatible gateway (`/v1`) with stable URLs across node moves, API key auth, and per-deployment scoping.
- Usage metrics, reproducibility manifests, Slack/webhook notifications, and log streaming.
- Warm cache (pause/resume models between GPU and RAM), per-GPU maintenance mode, and live cluster settings.
- Upload local checkpoints and LoRA adapters from the browser, or pull them from a URL.

See the [full documentation](https://sisl.github.io/aquila/) for detailed guides.

## Supported hardware
- GPUs: NVIDIA H100, A100, L40, DGX Spark (GB10), RTX 4090
- OS: Ubuntu 22.04 and 24.04

## Prerequisites
**Host:** Docker + Compose, Node.js ≥ 23 + npm, Python 3.10–3.14, `uv`.

**Client:** NVIDIA GPU with driver, Docker or Podman ≥ 5.4, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Python 3.10–3.14, `uv`.

## Quick start

Install:
```bash
uv venv && source .venv/bin/activate
uv pip install aquila
```

Start the host:
```bash
aquila host up --host-ip 0.0.0.0 --host-frontend-port 5173 --host-discover-port 11400
```

Add a client node:
```bash
aquila client up --host-ip <host-ip> --host-discover-port 11400
```

Open `http://<host-ip>:5173` — the client node appears within seconds. Add `--service` for persistent systemd services.

## Gateway usage

Every deployment is reachable through a single gateway URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://my-host:5173/v1", api_key="vcm-...")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```
