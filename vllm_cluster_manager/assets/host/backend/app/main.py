from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI

from app.api.router import api_router
from app.api.ws import router as ws_router
from app.db.session import engine
from app.models.base import Base
from app.services.sync import sync_deployments_from_clients, sync_nodes_from_consul
from sqlalchemy import text

logger = logging.getLogger(__name__)


async def wait_for_db(retries: int = 30, delay: float = 1.0) -> None:
    for attempt in range(retries):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)


async def ensure_node_gpu_column() -> None:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        statement = (
            "ALTER TABLE nodes "
            "ADD COLUMN IF NOT EXISTS gpu_usage JSONB DEFAULT '[]'::jsonb"
        )
    else:
        statement = "ALTER TABLE nodes ADD COLUMN gpu_usage JSON"

    async with engine.begin() as conn:
        try:
            await conn.execute(text(statement))
        except Exception:
            # Best-effort for existing databases without migrations.
            pass


async def ensure_node_port_column() -> None:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        statement = "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS port INTEGER"
    else:
        statement = "ALTER TABLE nodes ADD COLUMN port INTEGER"

    async with engine.begin() as conn:
        try:
            await conn.execute(text(statement))
        except Exception:
            pass


async def ensure_deployment_gpu_columns() -> None:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        statements = [
            "ALTER TABLE deployments ADD COLUMN IF NOT EXISTS gpu_ids JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE deployments ADD COLUMN IF NOT EXISTS tensor_parallel_size INTEGER",
            "ALTER TABLE deployments ADD COLUMN IF NOT EXISTS extra_args JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE deployments ADD COLUMN IF NOT EXISTS env_vars JSONB DEFAULT '[]'::jsonb",
        ]
    else:
        statements = [
            "ALTER TABLE deployments ADD COLUMN gpu_ids JSON",
            "ALTER TABLE deployments ADD COLUMN tensor_parallel_size INTEGER",
            "ALTER TABLE deployments ADD COLUMN extra_args JSON",
            "ALTER TABLE deployments ADD COLUMN env_vars JSON",
        ]

    async with engine.begin() as conn:
        for statement in statements:
            try:
                await conn.execute(text(statement))
            except Exception:
                pass


async def ensure_deployment_pip_packages_column() -> None:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        statement = (
            "ALTER TABLE deployments "
            "ADD COLUMN IF NOT EXISTS pip_packages JSONB DEFAULT '[]'::jsonb"
        )
    else:
        statement = "ALTER TABLE deployments ADD COLUMN pip_packages JSON"

    async with engine.begin() as conn:
        try:
            await conn.execute(text(statement))
        except Exception:
            pass


async def ensure_deployment_config_table() -> None:
    statement = (
        "CREATE TABLE IF NOT EXISTS deployment_configs ("
        "id SERIAL PRIMARY KEY,"
        "name VARCHAR(255) UNIQUE NOT NULL,"
        "payload JSONB NOT NULL,"
        "created_at TIMESTAMPTZ DEFAULT NOW()"
        ")"
    )
    if engine.dialect.name != "postgresql":
        statement = (
            "CREATE TABLE IF NOT EXISTS deployment_configs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "name VARCHAR(255) UNIQUE NOT NULL,"
            "payload JSON NOT NULL,"
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP"
            ")"
        )

    async with engine.begin() as conn:
        try:
            await conn.execute(text(statement))
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    await wait_for_db()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_node_gpu_column()
    await ensure_node_port_column()
    await ensure_deployment_gpu_columns()
    await ensure_deployment_pip_packages_column()
    await ensure_deployment_config_table()
    task = asyncio.create_task(sync_nodes_from_consul())
    deploy_task = asyncio.create_task(sync_deployments_from_clients())
    yield
    task.cancel()
    deploy_task.cancel()


app = FastAPI(title="vLLM Cluster Manager", lifespan=lifespan)
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)
