from sqlalchemy import DateTime, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base


class ApiKey(Base):
    """Gateway API key stored as a SHA-256 hash (zero-knowledge).

    Permanent keys (expires_at=NULL) protect the gateway; temporary keys
    are auto-created for endpoint code snippets and expire after a TTL.
    Keys can be scoped to specific deployments via allowed_deployment_ids.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    prefix: Mapped[str] = mapped_column(String(12))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    allowed_deployment_ids: Mapped[list[int] | None] = mapped_column(
        JSON, nullable=True, default=None
    )
