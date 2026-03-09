# Deployments

This page covers how to deploy models through the dashboard, including vLLM version selection, extra packages, plugins, GPU assignment, and deployment lifecycle.

## Creating a deployment

From the dashboard, select a target node, fill in the deployment form, and click **Deploy**. The required fields are:

| Field | Description |
| --- | --- |
| **Model Name** | Hugging Face model ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`). |
| **Port** | Port the vLLM OpenAI-compatible server will listen on. |
| **GPU Memory Fraction** | Fraction of GPU memory to allocate (0.0–1.0). |

## vLLM version

Every deployment runs inside its own isolated virtual environment. You can control which vLLM version is installed.

| Input | Behavior |
| --- | --- |
| *(blank)* | Installs the **latest stable release** from GitHub automatically. |
| `0.8.5` | Installs that specific release (`vllm==0.8.5`). |
| `nightly` | Installs the latest nightly build from `wheels.vllm.ai/nightly`. |
| 40-character hex string | Installs from a specific commit via `wheels.vllm.ai/<commit>`. |

The version field placeholder dynamically shows the current latest release so you always know what "blank" resolves to.

!!! note
    The resolved version is stored in the database and displayed in the deployments table, even when you leave the field blank. This way you always know exactly which vLLM version a deployment is running.

### How venvs work

- Each deployment gets its own venv, keyed by a hash of the version string and deployment key (`model:port`).
- Venvs are cached and reused if the same version + key combination is deployed again.
- When a deployment is stopped, its venv is automatically cleaned up.
- `uv` is used for venv creation and package installation for speed and reliability.

## GPU selection

Select which GPUs to use with the toggle buttons in the deploy form. Each button corresponds to a GPU index reported by the node. You can select one or more GPUs.

- If no GPUs are selected, vLLM uses its default GPU assignment.
- Selecting multiple GPUs sets `CUDA_VISIBLE_DEVICES` accordingly.

### Tensor parallel

When using multiple GPUs for a single model, set **Tensor Parallel Size** to the number of GPUs. This tells vLLM to shard the model across the selected GPUs.

## Extra packages

Expand the **Add Extra Packages** section to install additional pip packages into the deployment's venv. Enter one package per line, using standard pip syntax:

```
transformers>=4.40
flash-attn
```

These packages are installed after vLLM, into the same isolated venv.

## Plugins

vLLM supports Python plugins passed as CLI arguments (e.g. `--reasoning-parser-plugin my_plugin.py`). To use a plugin:

1. Click the upload button and select a `.py` file.
2. The file is uploaded to the client node and stored as-is.
3. The file path is automatically added to the **Extra Args** field.

Supported upload formats:

| Format | Behavior |
| --- | --- |
| `.py` | Stored as a plugin file. Path added to extra args. |
| `.whl` | Stored as a wheel. Path added to extra packages for pip install. |
| `.tar.gz` / `.zip` | Extracted and path added to extra packages. |

## Extra args

The **Extra Args** field lets you pass additional CLI flags to the vLLM server. These are appended directly to the `vllm.entrypoints.openai.api_server` command. Examples:

```
--max-model-len 4096
--reasoning-parser-plugin /path/to/plugin.py
--enforce-eager
```

## Environment variables

Add environment variables for the deployment under the **Env Vars** section. Common use cases:

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for gated models (e.g. Llama).
- `VLLM_ATTENTION_BACKEND` to override the attention backend.

!!! tip
    Sensitive values (tokens, keys, passwords) are masked in the deployment logs for security.

## Deployment lifecycle

A deployment goes through these states:

| Status | Meaning |
| --- | --- |
| **loading** | Venv is being created and/or the vLLM server is starting up. |
| **running** | The vLLM server is healthy and responding to requests. |
| **stopping** | A stop was requested and the process is shutting down. |
| **stopped** | The process has exited cleanly. |
| **error** | The process exited unexpectedly. Check logs for details. |
| **unreachable** | The host backend cannot reach the client node. |

### Readiness detection

After starting, the backend polls the vLLM server's `/health` and `/v1/models` endpoints every 2 seconds. The deployment transitions from `loading` to `running` once either endpoint returns HTTP 200.

### Logs

Click the terminal icon on any deployment to stream its logs in real time. Logs include:

- Venv creation and package installation output (`[uv]` prefixed lines).
- vLLM server startup and runtime output.
- ANSI escape codes are automatically stripped for clean display.

Logs are buffered (up to 2000 lines per deployment) and available as long as the deployment exists.

## Saved configurations

You can save and load deployment configurations from the dashboard. A saved config stores all deployment settings (model, port, GPU selection, version, packages, extra args, env vars) so you can redeploy with one click.

## Deployment recovery

If the host backend is restarted while deployments are still running on client nodes, the sync loop automatically rediscovers them and re-creates the database entries. Running deployments are never interrupted by a backend restart.
