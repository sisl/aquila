"""WebSocket endpoint for live dashboard updates.

Clients connect to /ws and receive JSON messages when server state
changes.  Message types: deployments_changed, nodes_changed,
settings_changed, api_keys_changed.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.manager import manager

router = APIRouter()


@router.websocket("/ws")
@router.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept a WebSocket connection for live dashboard state-change notifications."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
