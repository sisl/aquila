import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendHost = env.VITE_BACKEND_HOST || "localhost";
  const backendPort = env.VITE_BACKEND_PORT || "8000";
  const backendHttp = `http://${backendHost}:${backendPort}`;
  const backendWs = `ws://${backendHost}:${backendPort}`;

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      allowedHosts: true,
      proxy: {
        "/api": backendHttp,
        "/ws": {
          target: backendWs,
          ws: true
        }
      }
    }
  };
});
