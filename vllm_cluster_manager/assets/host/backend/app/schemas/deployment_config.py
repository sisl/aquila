from datetime import datetime
from pydantic import BaseModel


class DeploymentConfigBase(BaseModel):
    name: str
    payload: dict[str, object]


class DeploymentConfigCreate(DeploymentConfigBase):
    pass


class DeploymentConfigRead(DeploymentConfigBase):
    id: int
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
