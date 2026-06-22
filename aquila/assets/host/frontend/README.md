# Frontend

React + TypeScript dashboard for Aquila, built with Vite and Material UI.

## Development

```bash
npm install
npm run dev
```

The dev server runs on port 5173 and proxies `/api`, `/ws`, and `/v1` to the backend (port 8000).

## Production build

```bash
npm run build
npm run preview
```

The `host up` CLI command handles the build and preview server automatically.

## Type checking

```bash
npx tsc --noEmit
```

## Project structure

```
src/
├── pages/
│   └── Dashboard.tsx        # Main page — node table, deployment table, deploy dialog
├── components/
│   ├── NodeTable.tsx         # Node list with health, metrics, maintenance status
│   ├── DeploymentTable.tsx   # Deployment list with status, usage, actions
│   ├── DeploymentActions.tsx # Per-deployment action buttons (settings, logs, pause, stop, etc.)
│   ├── DeployDialog.tsx      # Model deployment form
│   ├── SettingsDialog.tsx    # Global settings + API key management
│   ├── EndpointDialog.tsx    # Connection snippets + temporary API key creation
│   └── ...                   # Confirmation dialogs, shared UI primitives
├── services/
│   └── api.ts               # API client — types, fetch functions, WebSocket helpers
└── hooks/                    # Custom React hooks (column visibility, etc.)
```

## Key patterns

- **React Query** for all server state — queries auto-invalidate on WebSocket `*_changed` events.
- **MUI** components with a custom dark theme defined in `App.tsx`.
- **WebSocket** connection to `/ws` pushes `deployments_changed`, `nodes_changed`, `settings_changed`, and `api_keys_changed` events to trigger cache invalidation.
