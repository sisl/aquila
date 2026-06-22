"""Gateway API key management with in-memory hash cache.

Keys are validated against a cached dict of SHA-256 hashes so gateway
requests never hit the database.  CRUD operations mutate the cache
inline.  ``last_used_at`` is flushed to the DB in batches every 60 s,
and expired keys are purged in the same loop.
"""

import asyncio
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, select, update

from app.models.api_key import ApiKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CachedKey:
    expires_at: datetime | None
    deployment_ids: frozenset[int] | None  # None = global access


_key_hashes: dict[str, _CachedKey] = {}
_last_used_updates: dict[str, datetime] = {}


def generate() -> tuple[str, str, str]:
    """Return ``(raw_key, prefix, key_hash)``."""
    raw = "ath-" + secrets.token_hex(16)
    prefix = raw[:8]
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, prefix, key_hash


async def load(session) -> None:
    """Populate the hash cache from the ``api_keys`` table (startup)."""
    result = await session.execute(
        select(ApiKey.key_hash, ApiKey.expires_at, ApiKey.allowed_deployment_ids)
    )
    _key_hashes.clear()
    for key_hash, expires_at, dep_ids in result.all():
        _key_hashes[key_hash] = _CachedKey(
            expires_at=expires_at,
            deployment_ids=frozenset(dep_ids) if dep_ids is not None else None,
        )
    if _key_hashes:
        logger.info("Loaded %d API key hash(es)", len(_key_hashes))


async def ensure_default_key(session) -> None:
    """Create a default 'admin' API key on first run so the gateway is never unprotected."""
    if has_keys():
        return
    raw, prefix, key_hash = generate()
    row = ApiKey(label="admin", prefix=prefix, key_hash=key_hash)
    session.add(row)
    await session.commit()
    add_hash(key_hash)
    logger.info(
        "Created default API key (label='admin'). "
        "Store this key — it will not be shown again: %s",
        raw,
    )


def has_keys() -> bool:
    """True when at least one permanent key exists."""
    return any(v.expires_at is None for v in _key_hashes.values())


def validate(raw_key: str) -> bool:
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    cached = _key_hashes.get(key_hash)
    if cached is None:
        return False
    if cached.expires_at is not None and cached.expires_at <= datetime.now(timezone.utc):
        _key_hashes.pop(key_hash, None)
        return False
    _last_used_updates[key_hash] = datetime.now(timezone.utc)
    return True


def get_allowed_deployments(raw_key: str) -> set[int] | None:
    """Return the set of deployment IDs this key may access, or None for global."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    cached = _key_hashes.get(key_hash)
    if cached is None:
        return set()
    return set(cached.deployment_ids) if cached.deployment_ids is not None else None


def add_hash(
    key_hash: str,
    expires_at: datetime | None = None,
    deployment_ids: list[int] | None = None,
) -> None:
    _key_hashes[key_hash] = _CachedKey(
        expires_at=expires_at,
        deployment_ids=frozenset(deployment_ids) if deployment_ids is not None else None,
    )


def update_hash(
    key_hash: str,
    deployment_ids: list[int] | None = None,
) -> None:
    """Update the cached scope for a key (used by PATCH)."""
    cached = _key_hashes.get(key_hash)
    if cached is None:
        return
    _key_hashes[key_hash] = _CachedKey(
        expires_at=cached.expires_at,
        deployment_ids=frozenset(deployment_ids) if deployment_ids is not None else None,
    )


def remove_hash(key_hash: str) -> None:
    _key_hashes.pop(key_hash, None)
    _last_used_updates.pop(key_hash, None)


def remove_deployment_from_scope(deployment_id: int) -> None:
    """Remove a deployment ID from all cached keys (called on deployment delete)."""
    for key_hash, cached in list(_key_hashes.items()):
        if cached.deployment_ids is not None and deployment_id in cached.deployment_ids:
            new_ids = cached.deployment_ids - {deployment_id}
            _key_hashes[key_hash] = _CachedKey(
                expires_at=cached.expires_at,
                deployment_ids=new_ids if new_ids else frozenset(),
            )


async def flush_last_used_loop(session_factory) -> None:
    """Batch-update ``last_used_at`` every 60 s and purge expired keys."""
    while True:
        await asyncio.sleep(60)
        try:
            async with session_factory() as session:
                if _last_used_updates:
                    batch = dict(_last_used_updates)
                    _last_used_updates.clear()
                    for key_hash, ts in batch.items():
                        await session.execute(
                            update(ApiKey)
                            .where(ApiKey.key_hash == key_hash)
                            .values(last_used_at=ts)
                        )

                now = datetime.now(timezone.utc)
                expired = [
                    h for h, cached in _key_hashes.items()
                    if cached.expires_at is not None and cached.expires_at <= now
                ]
                for h in expired:
                    _key_hashes.pop(h, None)
                if expired:
                    await session.execute(
                        delete(ApiKey).where(ApiKey.expires_at <= now)
                    )

                await session.commit()
        except Exception:
            logger.warning("Failed to flush API key updates", exc_info=True)
