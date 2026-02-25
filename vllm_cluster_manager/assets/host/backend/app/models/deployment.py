from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(primary_key=True)
    node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id"))
    model_name: Mapped[str] = mapped_column(String(255), index=True)
    port: Mapped[int] = mapped_column(Integer)
    gpu_memory_fraction: Mapped[float]
    gpu_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    tensor_parallel_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_args: Mapped[list[str]] = mapped_column(JSON, default=list)
    env_vars: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    pip_packages: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="stopped")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
