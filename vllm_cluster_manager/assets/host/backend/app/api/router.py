from fastapi import APIRouter

from app.api import admin, api_keys, configs, deployments, health, nodes, settings as settings_api

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
api_router.include_router(deployments.router, prefix="/deployments", tags=["deployments"])
api_router.include_router(configs.router, prefix="/configs", tags=["configs"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(settings_api.router, prefix="/settings", tags=["settings"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])
