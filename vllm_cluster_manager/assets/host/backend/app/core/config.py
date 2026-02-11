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

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
