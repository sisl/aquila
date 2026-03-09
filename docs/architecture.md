# Architecture

## Overview
vLLM Cluster Manager runs three host services and a client agent on each GPU node.

<div class="grid-2">
  <div>
    <strong>Host services</strong>
    <ul>
      <li>Infra: Postgres + Consul via Docker Compose</li>
      <li>Backend: FastAPI orchestration API</li>
      <li>Frontend: React + Vite admin dashboard</li>
    </ul>
  </div>
  <div>
    <strong>Client agent</strong>
    <ul>
      <li>Python service that registers with Consul</li>
      <li>Creates isolated per-deployment venvs with <code>uv</code></li>
      <li>Executes vLLM workloads on the node</li>
      <li>Reports deployment status, version, and GPU metrics</li>
    </ul>
  </div>
</div>

## Service discovery
Consul provides service discovery so the UI and backend can list connected clients.

## Data flow
1. Client registers with Consul.
2. Backend discovers clients and stores state in Postgres.
3. UI calls the backend API and subscribes to WebSocket streams for logs and status.
4. On deploy, the backend proxies the request to the client, which creates an isolated venv, installs the requested vLLM version and extra packages, then starts the vLLM server.
5. The sync loop periodically polls clients for deployment status and GPU metrics, updating the database. If the backend restarts, it rediscovers running deployments from clients automatically.

## Ports
| Service | Default | Purpose |
| --- | --- | --- |
| Frontend | 5173 | Web UI (Vite dev server). |
| Backend | 8000 | API + WebSockets. |
| Consul | 47528 | Host port mapped to Consul HTTP API (container port 8500). |
| Postgres | 5757 | Host port mapped to Postgres (container port 5432). |
| Client | 9000 | Client agent HTTP server. |
