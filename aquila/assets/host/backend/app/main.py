"""Aquila host backend.

FastAPI application that orchestrates multi-node vLLM deployments.
Manages the deployment lifecycle, syncs node state via Consul, serves
the OpenAI-compatible gateway, and pushes live updates over WebSocket.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import logging

from fastapi import FastAPI

from alembic import command
from alembic.config import Config as AlembicConfig

from app.api.gateway import close_gateway_client, router as gateway_router
from app.api.router import api_router
from app.api.ws import router as ws_router
from app.db.session import SessionLocal, engine
from app.services import api_keys, runtime_settings
from app.services.client_api import close_client
from app.services.sync import (
    sync_deployments_from_clients,
    sync_nodes_from_consul,
    enforce_deployment_expiry,
)
from sqlalchemy import text

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[1]


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


def _run_migrations() -> None:
    cfg = AlembicConfig(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await wait_for_db()
    # Alembic owns the schema; the baseline revision absorbs databases
    # created by the old create_all + ensure_*_column() path.
    await asyncio.to_thread(_run_migrations)
    logger.info("Database migrations applied")
    # Load dashboard-saved setting overrides (env values are the defaults).
    async with SessionLocal() as session:
        await runtime_settings.load(session)
        await api_keys.load(session)
        await api_keys.ensure_default_key(session)
    task = asyncio.create_task(sync_nodes_from_consul())
    deploy_task = asyncio.create_task(sync_deployments_from_clients())
    expiry_task = asyncio.create_task(enforce_deployment_expiry())
    flush_task = asyncio.create_task(api_keys.flush_last_used_loop(SessionLocal))
    yield
    task.cancel()
    deploy_task.cancel()
    expiry_task.cancel()
    flush_task.cancel()
    await close_client()
    await close_gateway_client()


app = FastAPI(title="Aquila", lifespan=lifespan)
app.include_router(api_router, prefix="/api")
# OpenAI-compatible gateway lives at /v1 (not under /api) so the standard
# `base_url=http://host:8000/v1` convention works with OpenAI clients.
app.include_router(gateway_router, prefix="/v1", tags=["gateway"])
app.include_router(ws_router)
