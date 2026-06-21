"""Backend configuration loaded from environment variables.

These values are the *defaults* for runtime settings -- once a value is
saved in the dashboard's Settings dialog, it takes precedence.  Infra
values (Postgres, Consul, bind addresses) are env-only.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    postgres_host: str = "localhost"
    postgres_port: int = 5757
    postgres_db: str = "vllm_admin"
    postgres_user: str = "vllm"
    postgres_password: str = "change-me"

    consul_http_addr: str = "http://localhost:47528"

    admin_api_host: str = "0.0.0.0"
    admin_api_port: int = 8000

    satellite_port: int = 9000

    # Seconds a deployment may sit in starting/loading without the client
    # reporting it before the watchdog marks it as errored.
    start_timeout_seconds: int = 1800

    # How long node metric samples are kept for the history charts.
    node_metrics_retention_hours: int = 48

    # Read timeout for non-streaming gateway requests (streams have none).
    gateway_timeout_seconds: int = 600

    # Optional webhook for deployment lifecycle notifications. Slack incoming
    # webhooks get Slack formatting; anything else receives generic JSON.
    webhook_url: str | None = None

    # Warn this many minutes before a deployment auto-expires.
    expiry_warning_minutes: int = 30

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
