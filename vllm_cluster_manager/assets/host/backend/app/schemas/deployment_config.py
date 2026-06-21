from datetime import datetime
from pydantic import BaseModel, Field


class DeploymentConfigBase(BaseModel):
    name: str = Field(description="Unique name for this saved configuration.")
    payload: dict[str, object] = Field(description="Deployment parameters snapshot (model_name, port, gpu_memory_fraction, etc.).")


class DeploymentConfigCreate(DeploymentConfigBase):
    pass


class DeploymentConfigRead(DeploymentConfigBase):
    id: int
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
