export type WsMessage = {
  type: string;
  payload?: unknown;
};

export function connectWebSocket(onMessage: (msg: WsMessage) => void): WebSocket {
  const socket = new WebSocket(`ws://${window.location.host}/ws`);

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
