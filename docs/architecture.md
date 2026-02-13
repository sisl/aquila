# Architecture

## Overview
vLLM Cluster Manager runs three host services and a client agent on each GPU node.

Host services:
- Infra: Postgres + Consul via Docker Compose
- Backend: FastAPI orchestration API
- Frontend: React + Vite admin dashboard

Client agent:
- Python service that registers with Consul and executes vLLM workloads

## Service discovery
Consul provides service discovery so the UI and backend can list connected clients.

## Data flow
1. Client registers with Consul.
2. Backend discovers clients and stores state in Postgres.
3. UI calls the backend API and subscribes to WebSocket streams for logs and status.
