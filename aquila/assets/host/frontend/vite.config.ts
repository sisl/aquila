import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendHost = env.VITE_BACKEND_HOST || "localhost";
  const backendPort = env.VITE_BACKEND_PORT || "8000";
  const baseRaw = env.VITE_BASE_PATH || "/";
  const base = normalizeBase(baseRaw);
  const baseNoTrail = base === "/" ? "" : base.slice(0, -1);
  const apiPath = `${baseNoTrail}/api`;
  const wsPath = `${baseNoTrail}/ws`;
  // OpenAI-compatible gateway lives at /v1 on the backend (not under /api).
  const gatewayPath = `${baseNoTrail}/v1`;
  const backendHttp = `http://${backendHost}:${backendPort}`;
  const backendWs = `ws://${backendHost}:${backendPort}`;

  return {
    base,
    plugins: [react()],
    build: {
      chunkSizeWarningLimit: 750,
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
      allowedHosts: true,
      proxy: {
        [apiPath]: {
          target: backendHttp,
          changeOrigin: true,
          rewrite: (path) => (baseNoTrail ? path.replace(new RegExp(`^${baseNoTrail}`), "") : path)
        },
        [wsPath]: {
          target: backendWs,
          ws: true,
          changeOrigin: true,
          rewrite: (path) => (baseNoTrail ? path.replace(new RegExp(`^${baseNoTrail}`), "") : path)
        },
        [gatewayPath]: {
          target: backendHttp,
          changeOrigin: true,
          rewrite: (path) => (baseNoTrail ? path.replace(new RegExp(`^${baseNoTrail}`), "") : path)
        }
      }
    }
  };
});

function normalizeBase(value: string): string {
  let base = value.trim();
  if (!base.startsWith("/")) {
    base = `/${base}`;
  }
  if (!base.endsWith("/")) {
    base = `${base}/`;
  }
  if (base === "//") {
    return "/";
  }
  return base;
}
