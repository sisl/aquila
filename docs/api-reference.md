# API Reference

The host backend is a FastAPI application that auto-generates interactive API documentation from its route definitions and Pydantic schemas.

## Interactive docs

When the backend is running, the following endpoints serve live, browsable documentation:

| URL | Format |
| --- | --- |
| `http://<host>:8000/docs` | Swagger UI — interactive request builder with try-it-out. |
| `http://<host>:8000/redoc` | ReDoc — clean read-only reference. |
| `http://<host>:8000/openapi.json` | Raw OpenAPI 3.x spec (JSON). |

Behind a reverse proxy, the docs are at `<base-path>/api/docs`, `<base-path>/api/redoc`, and `<base-path>/api/openapi.json`.

## Authentication

Gateway endpoints (`/v1/*`) require a valid API key when permanent keys exist — see [Gateway → Authentication](gateway.md#authentication). All other endpoints (`/api/*`) are unauthenticated (the tool targets trusted LAN environments).

---

## Health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Backend health check. Returns `{"status": "ok"}`. |
| `GET` | `/vllm-version` | Return the latest stable vLLM release tag from GitHub (cached for 1 hour). |

---

## Deployments

Prefix: `/api/deployments`

### List & inspect

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Return all deployments, newest first, with live metrics and pull progress attached. |
| `GET` | `/served-name/check` | Check whether a served model name is free, with a suggested alternative. Query params: `name`, `exclude_id`. |
| `GET` | `/{deployment_id}/manifest` | Self-contained reproducibility manifest. Env var values are omitted (keys only). |
| `GET` | `/{deployment_id}/logs` | Return the last N lines of the deployment's container log. Query param: `tail` (default 200). |
| `GET` | `/{deployment_id}/logs/download` | Stream the full persisted log file as a downloadable attachment. |

### Lifecycle

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/` | Create a deployment record without launching a container. |
| `POST` | `/start` | Create a deployment and launch its vLLM container on the target node. |
| `POST` | `/from-manifest` | Launch a deployment from an exported manifest. The manifest pins model identity; placement comes from the request. |
| `POST` | `/plan` | Preview which warm models a deploy would auto-offload, before committing. Read-only. |
| `POST` | `/stop/{deployment_id}` | Stop the deployment's container on its node and mark it as stopped. |
| `POST` | `/{deployment_id}/restart` | Re-launch a stopped, expired, or errored deployment on its original node. |
| `POST` | `/{deployment_id}/extend` | Push the serve deadline forward without restarting the model. |
| `DELETE` | `/{deployment_id}` | Delete the deployment record and remove it from scoped API keys. Does not stop a running container — call stop first. |

### Warm cache

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/{deployment_id}/pin` | Toggle the pin flag, protecting the deployment from warm-cache auto-eviction. |
| `POST` | `/{deployment_id}/pause` | Pause a running model by offloading it from GPU to RAM via vLLM sleep mode. |
| `POST` | `/{deployment_id}/resume` | Wake a paused model back to GPU and resume serving. |

### Request / response schemas

**DeploymentStart** (POST `/start`):

| Field | Type | Description |
| --- | --- | --- |
| `node_id` | `int` | ID of the node to deploy on. |
| `model_name` | `string` | HuggingFace model identifier (e.g. `meta-llama/Meta-Llama-3-8B`). |
| `port` | `int` | Port the vLLM server listens on. |
| `gpu_memory_fraction` | `float` | Fraction of each GPU's memory to allocate (0.0–1.0). |
| `gpu_ids` | `int[]` | Specific GPU indices to use; omit for automatic selection. |
| `tensor_parallel_size` | `int` | Number of GPUs for tensor parallelism. |
| `extra_args` | `string[]` | Additional CLI arguments passed to the vLLM server. |
| `env_vars` | `object[]` | Environment variables injected into the container (`{key, value}`). |
| `vllm_version` | `string` | vLLM Docker image tag to use; omit for latest stable. |
| `engine_args` | `object` | Structured vLLM engine flags (`max_model_len`, `dtype`, `quantization`, etc.). |
| `lora_modules` | `object[]` | LoRA adapters served alongside the base model (`{name, path}`). |
| `max_failed_restarts` | `int` | Crash-loop restart threshold; omit to use the cluster default. |
| `owner` | `string` | User or team launching this deployment (required). |
| `duration_seconds` | `int` | Serve duration in seconds; omit for indefinite serving. |
| `pinned` | `bool` | Pin this deployment to prevent warm-cache auto-eviction. Default `false`. |
| `skip_resource_check` | `bool` | Bypass the client's GPU memory pre-check. Default `false`. |

**DeploymentRead** (response for most deployment endpoints) extends the start fields with:

| Field | Type | Description |
| --- | --- | --- |
| `id` | `int` | Deployment ID. |
| `status` | `string` | Lifecycle status: `stopped`, `starting`, `loading`, `running`, `paused_ram`, `stopping`, `error`, `expired`. |
| `expires_at` | `datetime` | When this deployment will auto-stop; null for indefinite. |
| `image_digest` | `string` | Docker image digest reported by the client. |
| `last_error` | `string` | Last failure reason. |
| `detail` | `string` | Load phase while status is `loading` (downloading / loading_weights / compiling). |
| `total_prompt_tokens` | `int` | Cumulative prompt tokens processed. |
| `total_completion_tokens` | `int` | Cumulative completion tokens generated. |
| `total_requests` | `int` | Cumulative requests served. |
| `prompt_tps` | `float` | Live prompt tokens/s (per-request, idle-free). |
| `generation_tps` | `float` | Live generation tokens/s (per-request, idle-free). |
| `prompt_throughput` | `float` | Engine-wide prompt throughput. |
| `generation_throughput` | `float` | Engine-wide generation throughput. |
| `requests_running` | `int` | Requests currently being processed. |
| `requests_waiting` | `int` | Requests queued. |
| `pull_percent` | `float` | Image-pull progress percentage (while starting). |

**DeploymentRestart** (POST `/{id}/restart`):

| Field | Type | Description |
| --- | --- | --- |
| `owner` | `string` | User or team restarting this deployment. |
| `duration_seconds` | `int` | New serve duration in seconds; omit for indefinite. |

**DeploymentExtend** (POST `/{id}/extend`):

| Field | Type | Description |
| --- | --- | --- |
| `hours` | `float` | Hours to add to the deadline (max 336). Mutually exclusive with `infinite`. |
| `infinite` | `bool` | Drop the expiry entirely. Mutually exclusive with `hours`. |

**DeploymentPin** (POST `/{id}/pin`):

| Field | Type | Description |
| --- | --- | --- |
| `pinned` | `bool` | Whether the deployment is protected from warm-cache auto-eviction. |

**DeploymentPause** (POST `/{id}/pause`):

| Field | Type | Description |
| --- | --- | --- |
| `tier` | `string` | Target tier: `ram`. Omit for automatic selection. |

**DeploymentPlanRequest** (POST `/plan`):

| Field | Type | Description |
| --- | --- | --- |
| `node_id` | `int` | Target node. |
| `model_name` | `string` | Model to deploy. |
| `port` | `int` | Target port. |
| `gpu_memory_fraction` | `float` | GPU memory fraction. |
| `gpu_ids` | `int[]` | Specific GPU indices. |

**DeploymentPlanRead** (response):

| Field | Type | Description |
| --- | --- | --- |
| `fits` | `bool` | Whether the deploy fits (possibly after offloads). |
| `warm_enabled` | `bool` | False when the node has warm-offload off. |
| `would_offload` | `OffloadItem[]` | Models that would be evicted. |
| `blocked_reason` | `string` | Why no plan can make room (when `fits` is false). |

**DeploymentFromManifest** (POST `/from-manifest`):

| Field | Type | Description |
| --- | --- | --- |
| `manifest` | `object` | Exported manifest (from `GET /{id}/manifest`). |
| `node_id` | `int` | Target node. |
| `port` | `int` | Target port. |
| `owner` | `string` | User or team. |
| `duration_seconds` | `int` | Serve duration; omit for indefinite. |
| `env_vars` | `object[]` | Env var values (never travel in manifests; re-supply here). |
| `skip_resource_check` | `bool` | Bypass GPU memory pre-check. |

---

## Nodes

Prefix: `/api/nodes`

### List & manage

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | List all registered nodes with GPU metrics, disk usage, and status. |
| `POST` | `/` | Register a new node manually (nodes normally register via Consul). |
| `DELETE` | `/{node_id}` | Remove a node and its deployment/metric records. Live nodes re-register via Consul within seconds. |
| `GET` | `/discovered` | Return nodes discovered via Consul that haven't been registered yet. |

### Configuration

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/{node_id}/runtime` | Set the node's container runtime override (`docker` / `podman` / `null` for auto). |
| `POST` | `/{node_id}/warm-cache` | Enable/disable warm-cache auto-offload and set the RAM-cache budget. |
| `POST` | `/{node_id}/maintenance` | Cordon/uncordon specific GPUs (or all); optionally drain affected deployments. |

### Monitoring

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/{node_id}/metrics/history` | Metric samples for the last N minutes, downsampled to every step-th row. Query params: `minutes` (default 60), `step` (default 1). |
| `GET` | `/{node_id}/ports/check` | Check whether a port is available on the node. Query param: `port`. |

### Containers & processes

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/{node_id}/containers` | List vLLM containers running on the node. |
| `POST` | `/{node_id}/containers/{container_id}/stop` | Stop and remove a container on the node. |
| `GET` | `/{node_id}/gpu-processes` | List GPU processes on the node. |
| `POST` | `/{node_id}/gpu-processes/{pid}/kill` | Kill a GPU process on the node. |

### Container images

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/{node_id}/images` | List container images on the node. |
| `DELETE` | `/{node_id}/images/{image_id}` | Delete a container image from the node. |
| `POST` | `/{node_id}/images/prune` | Prune unused container images on the node. |

### Warm-cache artifacts

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/{node_id}/warm-artifacts` | List orphaned warm-cache artifacts (RAM sleepers, disk caches). |
| `POST` | `/{node_id}/warm-artifacts/sleepers/{pid}/kill` | Kill an orphaned RAM sleeper process. |
| `DELETE` | `/{node_id}/warm-artifacts/caches/{name}` | Delete an orphaned warm-cache compile artifact. |

### Model cache (HuggingFace)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/{node_id}/models/cache` | List HuggingFace model cache entries on the node. |
| `DELETE` | `/{node_id}/models/cache/{name}` | Delete a cached model from the node's HuggingFace cache. |

### Local models

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/{node_id}/local-models` | List models in the node's local model directory. |
| `DELETE` | `/{node_id}/local-models/{name}` | Delete a model from local storage. |
| `POST` | `/{node_id}/local-models/pull` | Pull a model from a URL to the node's local storage. |
| `GET` | `/{node_id}/local-models/transfers` | List active download/upload transfers on the node. |

**Streamed upload** (multi-file):

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/{node_id}/local-models/upload/begin` | Start a streamed multi-file model upload session. |
| `PUT` | `/{node_id}/local-models/upload/{session_id}/file` | Upload a single file within an active upload session. Query param: `path`. |
| `POST` | `/{node_id}/local-models/upload/{session_id}/finish` | Finalize a model upload session. |
| `POST` | `/{node_id}/local-models/upload/{session_id}/abort` | Abort and clean up an upload session. |
| `POST` | `/{node_id}/local-models/archive` | Upload a model as a single archive (tar/zip). Query params: `name`, `filename`. |

### Packages

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/{node_id}/packages/upload` | Upload a pip package wheel to the node's package cache. |
| `GET` | `/{node_id}/packages` | List pip packages in the node's package cache. |

### Request / response schemas

**NodeRead** (response):

| Field | Type | Description |
| --- | --- | --- |
| `id` | `int` | Node ID. |
| `hostname` | `string` | Machine hostname. |
| `ip_address` | `string` | Reachable IP address of the node agent. |
| `port` | `int` | Agent HTTP port. |
| `status` | `string` | Health status: `online`, `degraded`, `critical`, `maintenance`, `unknown`. |
| `gpu_usage` | `object[]` | Per-GPU utilization and memory stats from the latest scrape. |
| `disk_usage` | `object` | Disk usage breakdown (`total_gb`, `free_gb`, `hf_cache_gb`). |
| `maintenance` | `bool` | True when all GPUs are under maintenance. |
| `partial_maintenance` | `bool` | True when some (but not all) GPUs are under maintenance. |
| `maintenance_gpus` | `int[]` | GPU indices currently under maintenance. |
| `warm_offload_enabled` | `bool` | Whether warm-cache auto-offload is enabled. |
| `ram_cache_limit_mb` | `int` | CPU RAM budget in MB for warm-cached models; null = unlimited. |
| `ram_cache_used_mb` | `float` | CPU RAM currently held by paused models. |
| `available_runtimes` | `string[]` | Container runtimes detected (`docker` / `podman`). |
| `container_runtime` | `string` | Per-node runtime override; null = auto. |
| `rogue_container_count` | `int` | Untracked vLLM containers detected. |
| `rogue_process_count` | `int` | Orphaned GPU processes with no live container. |
| `rogue_artifact_count` | `int` | Orphaned warm-cache artifacts (RAM + disk). |

**NodeMaintenanceRequest** (POST `/{id}/maintenance`):

| Field | Type | Description |
| --- | --- | --- |
| `gpu_ids` | `int[]` | GPU indices to cordon/uncordon; empty = all GPUs. |
| `enabled` | `bool` | `true` to cordon, `false` to uncordon. |
| `drain` | `bool` | Stop deployments on the cordoned GPUs. Default `false`. |

**NodeWarmCacheRequest** (POST `/{id}/warm-cache`):

| Field | Type | Description |
| --- | --- | --- |
| `enabled` | `bool` | Enable or disable warm-cache auto-offload. |
| `ram_cache_limit_mb` | `int` | RAM budget in MB; null = unlimited. |

---

## Configs

Prefix: `/api/configs`

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | List all saved deployment configurations. |
| `POST` | `/` | Save a deployment configuration for later reuse. |
| `DELETE` | `/{config_id}` | Delete a saved configuration. |

**DeploymentConfigCreate** (POST `/`):

| Field | Type | Description |
| --- | --- | --- |
| `name` | `string` | Unique name for this saved configuration. |
| `payload` | `object` | Deployment parameters snapshot (model_name, port, gpu_memory_fraction, etc.). |

---

## Settings

Prefix: `/api/settings`

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Return all current runtime settings with their effective values. |
| `PUT` | `/` | Apply a partial update to runtime settings; only provided fields are changed. |

**RuntimeSettingsUpdate** (PUT `/`):

All fields are optional — only provided fields are applied.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `gateway_enabled` | `bool` | `true` | Enable or disable the OpenAI-compatible /v1 gateway. |
| `gateway_timeout_seconds` | `int` | `600` | Non-streaming request timeout in seconds (10–86400). |
| `start_timeout_seconds` | `int` | `1800` | Mark deployment errored if stuck starting this long (60–86400). |
| `preferred_container_runtime` | `string` | `docker` | Preferred runtime when a node has both Docker and Podman. |
| `default_port` | `int` | `8001` | Default port pre-filled in the deploy form (1024–65535). |
| `default_gpu_fraction` | `float` | `0.5` | Default GPU memory fraction (0.05–1.0). |
| `default_duration_choice` | `string` | `43200` | Default serve-duration choice for the deploy form (seconds). |
| `default_vllm_version` | `string` | `""` | Default vLLM version pre-filled in the deploy form. |
| `default_max_failed_restarts` | `int` | — | Crash-loop threshold for new deployments (1–20); null = client default. |
| `webhook_url` | `string` | `""` | Webhook URL for deployment lifecycle notifications. |
| `expiry_warning_minutes` | `int` | `30` | Warn this many minutes before a deployment expires (1–1440). |
| `node_metrics_retention_hours` | `int` | `48` | Hours of node metric history to retain (1–8760). |
| `temp_api_key_ttl_seconds` | `int` | `300` | Temporary API key lifespan in seconds (0–3600). |
| `default_warm_offload_enabled` | `bool` | `true` | Enable warm cache by default on newly discovered nodes. |
| `busy_guard_seconds` | `int` | `0` | Seconds after last request before a model can be auto-evicted (0–300). |
| `nodes_sync_interval_seconds` | `int` | `10` | Node sync loop interval (1–300). |
| `deployments_sync_interval_seconds` | `int` | `5` | Deployment sync loop interval (1–300). |
| `expiry_check_interval_seconds` | `int` | `30` | Expiry check loop interval (5–600). |
| `node_failure_threshold` | `int` | `3` | Consecutive failures before a node turns critical (1–20). |
| `deployment_failure_threshold` | `int` | `3` | Unreachable polls before a deployment degrades (1–20). |

---

## API Keys

Prefix: `/api/api-keys`

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/` | Create a new API key (permanent or temporary). |
| `GET` | `/` | List all active (non-expired) API keys. |
| `PATCH` | `/{key_id}` | Update a permanent key's label or deployment scope. |
| `DELETE` | `/{key_id}` | Revoke and delete an API key. |

**CreateApiKeyRequest** (POST `/`):

| Field | Type | Description |
| --- | --- | --- |
| `label` | `string` | Human-readable name for this key (1–128 chars). |
| `ttl_seconds` | `int` | Key lifetime in seconds (1–86400); omit for a permanent key. |
| `deployment_ids` | `int[]` | Restrict this key to specific deployment IDs. Null = all deployments. Required for temporary keys. |

**UpdateApiKeyRequest** (PATCH `/{key_id}`):

| Field | Type | Description |
| --- | --- | --- |
| `label` | `string` | Updated label. |
| `deployment_ids` | `int[]` | Updated deployment scope. Null = all deployments. |

---

## Gateway

Prefix: `/v1`

All gateway endpoints require a Bearer API key when permanent keys exist. See [Gateway → Authentication](gateway.md#authentication).

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/chat/completions` | Proxy an OpenAI-compatible chat completion request to the appropriate vLLM instance. |
| `POST` | `/completions` | Proxy an OpenAI-compatible text completion request. |
| `POST` | `/embeddings` | Proxy an embeddings request. |
| `GET` | `/models` | Return available models in OpenAI `/v1/models` format, filtered by API key scope. |

The gateway routes by the `model` field in the request body. Streaming is supported via `"stream": true`. Error responses follow the OpenAI error format:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

| Code | Meaning |
| --- | --- |
| `model_not_found` | No running deployment serves this model name. |
| `model_ambiguous` | Multiple deployments match — use a specific `served_model_name`. |
| `model_loading` | Model is still starting up; retry shortly. |
| `deployment_unreachable` | The node hosting this model is not responding. |
| `missing_model` | Request body is missing the required `model` field. |
| `invalid_json` | Request body is not valid JSON. |

---

## Admin

Prefix: `/api/admin`

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/purge` | Delete database records by category; running containers are untouched. |

**PurgeRequest** (POST `/purge`):

| Field | Type | Description |
| --- | --- | --- |
| `targets` | `string[]` | Categories to purge: `deployments`, `nodes`, `metrics`, `configs`. Omit for all. Purging `nodes` automatically includes `deployments` and `metrics`. |

---

## WebSocket

| Method | Path | Description |
| --- | --- | --- |
| `WebSocket` | `/ws` | Accept a WebSocket connection for live state-change notifications. |

The backend pushes JSON messages to notify clients of state changes. See [Architecture → WebSocket events](architecture.md#websocket-events) for the message types and payload format.
