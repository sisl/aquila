export type WsMessage = {
  type: string;
  payload?: unknown;
};

export function connectWebSocket(onMessage: (msg: WsMessage) => void): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsPath = withBase("ws/");
  const socket = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);

  socket.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as WsMessage;
      onMessage(parsed);
    } catch {
      onMessage({ type: "raw", payload: event.data });
    }
  };

  return socket;
}

function withBase(path: string): string {
  const base = import.meta.env.BASE_URL || "/";
  const normalized = base.endsWith("/") ? base : `${base}/`;
  return `${normalized}${path}`.replace(/\/{2,}/g, "/").replace(/:\//, "://");
}
