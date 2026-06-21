from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.db.session import get_session
from app.models.api_key import ApiKey
from app.services import api_keys
from app.ws.manager import manager

router = APIRouter()


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=128)
    ttl_seconds: int | None = Field(None, ge=1, le=86400)
    deployment_ids: list[int] | None = None


class UpdateApiKeyRequest(BaseModel):
    label: str | None = Field(None, min_length=1, max_length=128)
    deployment_ids: list[int] | None = None


@router.post("")
async def create_api_key(
    payload: CreateApiKeyRequest, session: AsyncSession = Depends(get_session)
):
    if payload.ttl_seconds is not None and payload.deployment_ids is None:
        raise HTTPException(
            status_code=422,
            detail="Temporary keys must specify deployment_ids.",
        )
    raw, prefix, key_hash = api_keys.generate()
    expires_at = None
    if payload.ttl_seconds is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=payload.ttl_seconds)
    row = ApiKey(
        label=payload.label,
        prefix=prefix,
        key_hash=key_hash,
        expires_at=expires_at,
        allowed_deployment_ids=payload.deployment_ids,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    api_keys.add_hash(
        key_hash, expires_at=expires_at, deployment_ids=payload.deployment_ids
    )
    await manager.broadcast({"type": "api_keys_changed"})
    return {
        "id": row.id,
        "label": row.label,
        "prefix": row.prefix,
        "key": raw,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "allowed_deployment_ids": row.allowed_deployment_ids,
    }


@router.get("")
async def list_api_keys(session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(ApiKey)
        .where(or_(ApiKey.expires_at.is_(None), ApiKey.expires_at > func.now()))
        .order_by(ApiKey.created_at.desc())
    )
    return [
        {
            "id": row.id,
            "label": row.label,
            "prefix": row.prefix,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            "allowed_deployment_ids": row.allowed_deployment_ids,
        }
        for row in result.scalars().all()
    ]


@router.patch("/{key_id}")
async def update_api_key(
    key_id: int,
    payload: UpdateApiKeyRequest,
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(ApiKey, key_id)
    if not row:
        raise HTTPException(status_code=404, detail="API key not found.")
    if row.expires_at is not None:
        raise HTTPException(
            status_code=409, detail="Cannot edit temporary keys."
        )
    if payload.label is not None:
        row.label = payload.label
    row.allowed_deployment_ids = payload.deployment_ids
    await session.commit()
    await session.refresh(row)
    api_keys.update_hash(row.key_hash, deployment_ids=payload.deployment_ids)
    await manager.broadcast({"type": "api_keys_changed"})
    return {
        "id": row.id,
        "label": row.label,
        "prefix": row.prefix,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        "expires_at": None,
        "allowed_deployment_ids": row.allowed_deployment_ids,
    }


@router.delete("/{key_id}", status_code=204)
async def delete_api_key(
    key_id: int, session: AsyncSession = Depends(get_session)
):
    row = await session.get(ApiKey, key_id)
    if not row:
        raise HTTPException(status_code=404, detail="API key not found.")
    api_keys.remove_hash(row.key_hash)
    await session.delete(row)
    await session.commit()
    await manager.broadcast({"type": "api_keys_changed"})
