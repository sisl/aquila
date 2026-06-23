import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts state-change events.

    The broadcast method sends a JSON message to all connected clients,
    silently dropping connections that have gone away.
    """

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        # A single dead socket must not break the callers (sync loops).
        for websocket in list(self.active):
            try:
                await websocket.send_json(message)
            except Exception as exc:
                logger.debug("Dropping dead websocket: %s", exc)
                self.disconnect(websocket)


manager = ConnectionManager()
