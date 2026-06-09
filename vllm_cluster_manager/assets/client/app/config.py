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
    hf_cache_dir: str = "~/.cache/huggingface"


settings = Settings()
