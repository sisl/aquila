# Documentation Screenshots

Captures anonymized screenshots of the Aquila dashboard for the docs site. Real hostnames, IPs, ports, and user names are automatically replaced with fake values.

## Prerequisites

- Node.js 18+
- A running Aquila frontend (`aquila host up` or `npm run dev` in the frontend directory)
- At least one node and one deployment active (dialogs that require data are skipped otherwise)

## Usage

```bash
cd docs/scripts
npm install                        # first time only
npx playwright install chromium    # first time only
node update-screenshots.mjs
```

Screenshots are written to `docs/assets/img/`.

## Options

| Flag    | Default                  | Description              |
|---------|--------------------------|--------------------------|
| `--url` | `http://localhost:5173`  | Frontend URL to capture  |
| `--out` | `../assets/img`          | Output directory for PNGs |

## How it works

1. Fetches `/api/nodes/` and `/api/deployments/` to discover real values
2. Builds a replacement map (hostnames, IPs, ports, owners) to anonymized equivalents
3. Launches headless Chrome via Playwright, intercepting all API responses to apply the replacements
4. Navigates the dashboard and takes element-level screenshots of each panel and dialog
5. Saves 8 PNGs matching the filenames referenced in the MkDocs markdown files
