from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.api_key import ApiKey
from app.services import api_keys
from app.ws.manager import manager

router = APIRouter()


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=128)


@router.post("")
async def create_api_key(
    payload: CreateApiKeyRequest, session: AsyncSession = Depends(get_session)
):
    raw, prefix, key_hash = api_keys.generate()
    row = ApiKey(label=payload.label, prefix=prefix, key_hash=key_hash)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    api_keys.add_hash(key_hash)
    await manager.broadcast({"type": "api_keys_changed"})
    return {
        "id": row.id,
        "label": row.label,
        "prefix": row.prefix,
        "key": raw,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("")
async def list_api_keys(session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(ApiKey).order_by(ApiKey.created_at.desc())
    )
    return [
        {
            "id": row.id,
            "label": row.label,
            "prefix": row.prefix,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        }
        for row in result.scalars().all()
    ]


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
