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

The served name is the routing key and is **unique across active deployments** — it defaults to the model name, and a launch that would collide is rejected (the deploy form prompts for a different name and suggests a free one). Routing is therefore deterministic: a request resolves to exactly one deployment, never a guess.

You can still serve the same base model several times — give each replica a distinct `served_model_name` and address it by that name. The bare model name is then shared by several deployments and is reported as ambiguous rather than routed (see `model_ambiguous` below); address one of the served names instead.

Errors come back in OpenAI's error format:

| Status | Meaning |
| --- | --- |
| 400 | The model name maps to several deployments (`model_ambiguous`); the message lists the served names to choose from. |
| 401 | Missing or invalid API key. |
| 403 | The API key is valid but not authorized for the requested deployment (scoped key). |
| 404 | No deployment serves that model; the message lists the available model names. |
| 503 | A deployment matches but is still starting/loading — retry shortly. Also returned when the gateway is disabled. |
| 502 | The deployment's node did not respond. |

## Usage example

```python
from openai import OpenAI

client = OpenAI(base_url="http://my-host:5173/v1", api_key="vcm-abc123...")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Authentication

The gateway is protected by API keys. When at least one **permanent** API key exists, every `/v1` request must include a valid key in the `Authorization` header:

```
Authorization: Bearer vcm-abc123...
```

Keys are managed in **Settings → Gateway & Keys**:

- **Permanent keys** authenticate long-lived clients (scripts, notebooks, CI pipelines). They can be scoped to **all deployments** (default) or restricted to **specific deployments** — a scoped key can only reach the deployments it is assigned to, and `/v1/models` only lists those models. Scopes can be changed after creation.
- **Temporary keys** are auto-created when you open a deployment's **Endpoint** dialog (the code snippets embed one). They expire after a configurable TTL (default 5 minutes, adjustable in Settings) and are always scoped to the single deployment whose dialog created them.

If **no** permanent keys exist, the gateway is open — any request is accepted without a key. The first time the host starts it creates a default `admin` key and prints it to the log (store it; it is shown only once). Deleting the last permanent key requires an explicit acknowledgment that the gateway will become unprotected.

The gateway can be turned off entirely in **Settings → Gateway**: `/v1` requests then return `503` with a clear message, while direct node URLs keep working (the Endpoint dialog switches to direct-only automatically). The non-streaming request timeout is configurable in the same section.

## Usage metrics

Usage is tracked **per deployment**, sourced from vLLM's own Prometheus counters: each client agent scrapes its containers' `/metrics` endpoint every ~15 s, and the host folds the deltas into the deployment's lifetime totals (reset-safe across container restarts). Because vLLM itself maintains the counters, every request is counted — whether it arrived via the gateway or directly at the node.

The **Usage** column of the deployments table shows, per deployment:

- lifetime prompt / completion tokens
- total completed requests
- average **read** (prefill) and **generation** (decode) speeds in tokens/s
- (in the tooltip) the full breakdown: average per-request speeds, engine-wide window throughput, and requests currently running/queued

The two speeds are **running averages over processing time**: cumulative token counters divided by the cumulative processing-time sums from vLLM's per-request time histograms (`request_prefill_time_seconds` / `request_decode_time_seconds` on V1 engines). Idle time never enters the denominator, and once a deployment has served its first request the averages stay defined permanently — they never drop to zero or disappear between bursts (they reset only when the container restarts, together with the counters). On older engines without those histograms, TTFT/TPOT histograms are used as a fallback (TTFT includes queue wait, so the read speed there is a lower bound). The tooltip additionally shows **engine throughput** — token deltas over the last wall-clock scrape window — the aggregate view across all concurrent requests, present only for windows with activity.

The same values are included in `GET /api/deployments/` (`total_prompt_tokens`, `total_completion_tokens`, `total_requests`, and — for running deployments — `prompt_tps`, `generation_tps`, `prompt_throughput`, `generation_throughput`, `requests_running`, `requests_waiting`).
