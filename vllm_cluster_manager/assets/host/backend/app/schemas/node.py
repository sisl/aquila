from datetime import datetime
from pydantic import BaseModel


class NodeBase(BaseModel):
    hostname: str
    ip_address: str
    port: int | None = None
    status: str = "unknown"
    gpu_usage: list[dict[str, object]] | None = None
    default_pip_packages: list[str] | None = None
    installed_packages: list[str] | None = None


class NodeCreate(NodeBase):
    pass


class NodeRead(NodeBase):
    id: int
    last_heartbeat_at: datetime | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class DiscoveredNode(BaseModel):
    node: str | None = None
    address: str | None = None
    port: int | None = None
    service_id: str | None = None
