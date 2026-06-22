from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base


class NodeMetric(Base):
    """One sample per node per sync cycle; pruned after a retention window."""

    __tablename__ = "node_metrics"
    __table_args__ = (
        Index("ix_node_metrics_node_recorded", "node_id", "recorded_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    recorded_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    gpus: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
