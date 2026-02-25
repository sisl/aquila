from datetime import datetime
from pydantic import BaseModel


class DeploymentBase(BaseModel):
    node_id: int
    model_name: str
    port: int
    gpu_memory_fraction: float
    gpu_ids: list[int] | None = None
    tensor_parallel_size: int | None = None
    extra_args: list[str] | None = None
    env_vars: list[dict[str, str]] | None = None
    pip_packages: list[str] | None = None
    status: str = "stopped"


class DeploymentCreate(DeploymentBase):
    pass


class DeploymentRead(DeploymentBase):
    id: int
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class DeploymentStart(DeploymentBase):
    pass
