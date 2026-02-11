from sqlalchemy import DateTime, JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base


class DeploymentConfig(Base):
    __tablename__ = "deployment_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
