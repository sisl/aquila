# CLI Reference

The CLI command is `aquila`.

Run `--help` to see command and flag details:
```bash
aquila --help
aquila host up --help
aquila client up --help
```

## Host (`host up`)

| Flag | Default | Description |
| --- | --- | --- |
| `--service` | `false` | Run as a persistent systemd service. |
| `--host-ip` | `127.0.0.1` | Bind host for the backend API and UI backend target. |
| `--host-frontend-port` | `5173` | UI port. |
| `--host-discover-port` | `47528` | Discovery port used by clients. |
| `--host-backend-port` | `8000` | Backend API port. |
| `--postgres-host` | `127.0.0.1` | Postgres host. |
| `--postgres-port` | `5757` | Postgres port. |
| `--postgres-db` | `aquila` | Postgres database name. |
| `--postgres-user` | `vllm` | Postgres user. |
| `--postgres-password` | `change-me` | Postgres password. |
| `--base-path` | `/` | Base path for the UI (reverse proxy subpath). |

## Host (`host down`)

| Flag | Default | Description |
| --- | --- | --- |
| `--purge` | `false` | Also delete the Postgres data volume (wipes all deployments, nodes, and history). Without it, data persists and is restored on the next `host up`. |

## Client (`client up`)

| Flag | Default | Description |
| --- | --- | --- |
| `--service` | `false` | Run as a persistent systemd service. |
| `--host-ip` | `127.0.0.1` | Host IP for discovery. |
| `--host-discover-port` | `47528` | Host discovery port. |
| `--client-host` | `0.0.0.0` | Client bind host. |
| `--client-port` | `9000` | Client bind port. |
| `--node-name` | `<hostname>` | Node name used for registration. |

## Examples

Launch host services in the foreground:
```bash
aquila host up --host-ip 127.0.0.1 --host-frontend-port 5173 --host-discover-port 47528
```

Reverse proxy under `/vllm/`:
```bash
aquila host up --base-path /vllm/
```

Launch a client node:
```bash
aquila client up --host-ip 127.0.0.1 --host-discover-port 47528 --node-name node-1
```

### Service mode
Service mode writes systemd unit files and enables them so they start automatically on boot:
```bash
aquila host up --service
aquila client up --service
```

!!! warning
    Service mode requires systemd and appropriate permissions. Use `sudo` when needed.
