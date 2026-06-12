# OpenAI Gateway & Usage

The host backend exposes an OpenAI-compatible gateway at `/v1`. It gives every deployment in the cluster a **single stable base URL** — clients keep working when a model moves to another node or port.

Direct access to each vLLM server (`http://<node-ip>:<deployment-port>/v1`) continues to work; the gateway is an addition, not a replacement.

## Base URL

| Access | URL |
| --- | --- |
| Gateway (recommended) | `http://<host>:<host-frontend-port>/v1` (proxied to the backend) or `http://<host>:8000/v1` directly |
| Direct | `http://<node-ip>:<deployment-port>/v1` |

The UI's frontend server proxies `/v1` to the backend, so the gateway is reachable on the same origin as the dashboard — including under a reverse-proxy base path (e.g. `https://lab.example.com/vllm/v1`).

The easiest way to get a working URL and code snippet is the **Endpoint** button on any running deployment: it shows the base URL, a ready-to-paste Python `openai` snippet, and a curl one-liner — each with a gateway/direct toggle and a copy button.

## Supported endpoints

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/chat/completions` | Proxied to the matching deployment (streaming and non-streaming). |
| `POST /v1/completions` | Same. |
| `POST /v1/embeddings` | Same. |
| `GET /v1/models` | Lists all models served by **running** deployments, including LoRA adapters (with `parent` set to the base model). |

## Routing

The `model` field of the request selects the deployment. Matching precedence:

1. `served_model_name` (if set in the deployment's engine options)
2. The deployment's model name (HF id or local path)
3. A LoRA adapter name served by the deployment

If several running deployments match (replicas of the same model), one is chosen at random — free load balancing.

Errors come back in OpenAI's error format:

| Status | Meaning |
| --- | --- |
| 404 | No deployment serves that model; the message lists the available model names. |
| 503 | A deployment matches but is still starting/loading — retry shortly. |
| 502 | The deployment's node did not respond. |

## Usage example

```python
from openai import OpenAI

client = OpenAI(base_url="http://my-host:5173/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

There is no authentication (the tool targets small trusted environments); any `api_key` value is accepted.

## Usage metrics

Usage is tracked **per deployment**, sourced from vLLM's own Prometheus counters: each client agent scrapes its containers' `/metrics` endpoint every ~15 s, and the host folds the deltas into the deployment's lifetime totals (reset-safe across container restarts). Because vLLM itself maintains the counters, every request is counted — whether it arrived via the gateway or directly at the node.

The **Usage** column of the deployments table shows, per deployment:

- lifetime prompt / completion tokens
- total completed requests
- the current generation rate in tokens/s
- (in the tooltip, when the vLLM version exposes them) requests currently running and queued

The same values are included in `GET /api/deployments/` (`total_prompt_tokens`, `total_completion_tokens`, `total_requests`, and — for running deployments — `tokens_per_second`, `requests_running`, `requests_waiting`).
