from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    node_name: str | None = None
    host: str = "0.0.0.0"
    advertise_host: str | None = None
    port: int = 9000

    consul_http_addr: str = "http://localhost:47528"


settings = Settings()
