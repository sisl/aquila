"""Gateway API key management with in-memory hash cache.

Keys are validated against a cached set of SHA-256 hashes so gateway
requests never hit the database.  CRUD operations mutate the cache
inline.  ``last_used_at`` is flushed to the DB in batches every 60 s.
"""

import asyncio
import hashlib
import logging
import secrets
from datetime import datetime, timezone

from sqlalchemy import select, update

from app.models.api_key import ApiKey

logger = logging.getLogger(__name__)

_key_hashes: set[str] = set()
_last_used_updates: dict[str, datetime] = {}


def generate() -> tuple[str, str, str]:
    """Return ``(raw_key, prefix, key_hash)``."""
    raw = "vcm-" + secrets.token_hex(16)
    prefix = raw[:8]
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, prefix, key_hash


async def load(session) -> None:
    """Populate the hash cache from the ``api_keys`` table (startup)."""
    result = await session.execute(select(ApiKey.key_hash))
    _key_hashes.clear()
    for (h,) in result.all():
        _key_hashes.add(h)
    if _key_hashes:
        logger.info("Loaded %d API key hash(es)", len(_key_hashes))


def has_keys() -> bool:
    return len(_key_hashes) > 0


def validate(raw_key: str) -> bool:
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    if key_hash not in _key_hashes:
        return False
    _last_used_updates[key_hash] = datetime.now(timezone.utc)
    return True


def add_hash(key_hash: str) -> None:
    _key_hashes.add(key_hash)


def remove_hash(key_hash: str) -> None:
    _key_hashes.discard(key_hash)
    _last_used_updates.pop(key_hash, None)


async def flush_last_used_loop(session_factory) -> None:
    """Batch-update ``last_used_at`` every 60 s."""
    while True:
        await asyncio.sleep(60)
        if not _last_used_updates:
            continue
        batch = dict(_last_used_updates)
        _last_used_updates.clear()
        try:
            async with session_factory() as session:
                for key_hash, ts in batch.items():
                    await session.execute(
                        update(ApiKey)
                        .where(ApiKey.key_hash == key_hash)
                        .values(last_used_at=ts)
                    )
                await session.commit()
        except Exception:
            logger.warning("Failed to flush last_used_at updates", exc_info=True)
