# CLAUDE.md

## Project structure

```
vllm_cluster_manager/
├── assets/
│   ├── host/
│   │   ├── backend/     # FastAPI backend (Python, SQLAlchemy, Alembic)
│   │   └── frontend/    # React + Vite + MUI dashboard (TypeScript)
│   └── client/
│       └── app/          # Node agent (FastAPI, Docker/Podman management)
├── docs/                 # MkDocs documentation site
├── docker-compose.yml    # Postgres + Consul infrastructure
└── pyproject.toml
```

## Running tests

Backend and client tests require a uv venv:

```bash
source .venv/bin/activate

# Backend tests
cd assets/host
CONSUL_HTTP_ADDR=127.0.0.1:1 .venv/bin/python -m pytest tests/ -q

# Client tests
cd assets/client
CONSUL_HTTP_ADDR=127.0.0.1:1 .venv/bin/python -m pytest tests/ -q
```

Set `CONSUL_HTTP_ADDR=127.0.0.1:1` to avoid registering bogus nodes against the live Consul instance.

Frontend type checking:

```bash
cd assets/host/frontend
npx tsc --noEmit
```

## Key constraints

- Do NOT SSH into client nodes — scope is limited to the host machine.
- Do NOT kill the live host backend (port 8000) without explicit request.
- Do NOT run `npm audit fix --force` in the frontend.
- Alembic ignores `DATABASE_URL`; it reads `POSTGRES_*` env vars. The defaults point at the live Postgres on port 5757.
- The client and backend both have an `app` package; tests for one may need sys.modules cleanup to avoid import collisions.

## Database

PostgreSQL 16 via Docker Compose, port 5757, database `vllm_admin`, user `vllm`. Migrations run automatically at backend startup.
