from fastapi import APIRouter

from app.api import configs, deployments, health, nodes

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
api_router.include_router(deployments.router, prefix="/deployments", tags=["deployments"])
api_router.include_router(configs.router, prefix="/configs", tags=["configs"])
