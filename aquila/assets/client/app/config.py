from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    node_name: str | None = None
    host: str = "0.0.0.0"
    advertise_host: str | None = None
    port: int = 9000

    consul_http_addr: str = "http://localhost:47528"

    # Docker runtime settings. vLLM deployments run as official vllm/vllm-openai
    # containers; the agent itself no longer installs vLLM.
    vllm_image_repo: str = "vllm/vllm-openai"
    # Host directory mounted into every vLLM container as the HuggingFace cache so
    # model weights are downloaded once and shared across deployments.
    hf_cache_dir: str = "~/.local/share/aquila/models"
    # Root directory for deployment logs, uploaded packages, local models, and
    # compile caches.
    vllm_client_root: str = "~/.vllm-client"

    # Default crash-loop breaker threshold: stop a deployment that restarts
    # this many times without ever becoming ready. Overridable per deployment.
    max_failed_restarts: int = 3

    # Comma-separated host directories that may be served as local models /
    # LoRA adapters. Each is mounted read-only into vLLM containers at the
    # same path. Empty = local paths rejected.
    model_dirs: str = ""

    # Persistent deployment logs ({vllm_client_root}/.logs): rotate a deployment's
    # log file once it exceeds this size, and delete files untouched for this
    # many days.
    log_max_mb: int = 50
    log_retention_days: int = 14


settings = Settings()
