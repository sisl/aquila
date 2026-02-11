from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.deployment_config import DeploymentConfig
from app.schemas.deployment_config import DeploymentConfigCreate, DeploymentConfigRead

router = APIRouter()


@router.get("/", response_model=list[DeploymentConfigRead])
async def list_configs(
    session: AsyncSession = Depends(get_session),
) -> list[DeploymentConfigRead]:
    result = await session.execute(select(DeploymentConfig).order_by(DeploymentConfig.name))
    return list(result.scalars().all())


@router.post("/", response_model=DeploymentConfigRead)
async def create_config(
    payload: DeploymentConfigCreate, session: AsyncSession = Depends(get_session)
) -> DeploymentConfigRead:
    existing = await session.execute(
        select(DeploymentConfig).where(DeploymentConfig.name == payload.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Config name already exists")

    config = DeploymentConfig(**payload.model_dump())
    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config


@router.delete("/{config_id}")
async def delete_config(
    config_id: int, session: AsyncSession = Depends(get_session)
) -> dict[str, str]:
    config = await session.get(DeploymentConfig, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    await session.delete(config)
    await session.commit()
    return {"status": "deleted"}
