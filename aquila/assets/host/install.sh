#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  require_cmd curl
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

prompt_default() {
  local prompt="$1"
  local default="$2"
  local value
  read -r -p "$prompt [$default]: " value
  if [[ -z "$value" ]]; then
    echo "$default"
  else
    echo "$value"
  fi
}

require_cmd npm
require_cmd docker
require_cmd systemctl
ensure_uv

DOCKER_BIN="$(command -v docker)"
NPM_BIN="$(command -v npm)"
NODE_BIN="$(command -v node)"
COMPOSE_CMD=""

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="$(command -v docker-compose)"
elif "$DOCKER_BIN" compose version >/dev/null 2>&1; then
  COMPOSE_CMD="$DOCKER_BIN compose"
else
  echo "Missing docker compose (install docker compose plugin or docker-compose)." >&2
  exit 1
fi

echo "Athanor installer"
echo "Working directory: $ROOT_DIR"
echo

CONSUL_PORT="$(prompt_default "Consul HTTP port (host port)" "47528")"
FRONTEND_PORT="$(prompt_default "Frontend exposed port" "5173")"
ADMIN_API_HOST="$(prompt_default "Backend bind host" "127.0.0.1")"
ADMIN_API_PORT="$(prompt_default "Backend bind port" "8000")"
POSTGRES_HOST="$(prompt_default "Postgres host" "127.0.0.1")"
POSTGRES_PORT="$(prompt_default "Postgres host port" "5757")"
POSTGRES_DB="$(prompt_default "Postgres database" "athanor")"
POSTGRES_USER="$(prompt_default "Postgres user" "vllm")"
POSTGRES_PASSWORD="$(prompt_default "Postgres password" "change-me")"

CONSUL_HTTP_ADDR="http://127.0.0.1:${CONSUL_PORT}"

echo
echo "Writing docker compose environment..."
cat > "$ROOT_DIR/.env" <<EOF
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_PORT=${POSTGRES_PORT}
CONSUL_PORT=${CONSUL_PORT}
EOF

echo
echo "Setting up backend virtual environment..."
uv venv --python=3.12 "$ROOT_DIR/backend/.venv"
# shellcheck disable=SC1091
source "$ROOT_DIR/backend/.venv/bin/activate"
uv pip install -r "$ROOT_DIR/backend/requirements.txt"
deactivate

echo
echo "Writing backend environment..."
cat > "$ROOT_DIR/backend/.env" <<EOF
ADMIN_API_HOST=${ADMIN_API_HOST}
ADMIN_API_PORT=${ADMIN_API_PORT}
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
CONSUL_HTTP_ADDR=${CONSUL_HTTP_ADDR}
EOF

echo
echo "Installing frontend dependencies..."
(
  cd "$ROOT_DIR/frontend"
  "$NPM_BIN" install
)

echo
echo "Writing frontend environment..."
cat > "$ROOT_DIR/frontend/.env" <<EOF
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=${FRONTEND_PORT}
VITE_BACKEND_HOST=localhost
VITE_BACKEND_PORT=${ADMIN_API_PORT}
EOF

INFRA_SERVICE="/etc/systemd/system/vllm-cluster-infra.service"
BACKEND_SERVICE="/etc/systemd/system/vllm-cluster-backend.service"
FRONTEND_SERVICE="/etc/systemd/system/vllm-cluster-frontend.service"

echo
echo "Creating systemd services (requires sudo)..."
sudo tee "$INFRA_SERVICE" >/dev/null <<EOF
[Unit]
Description=Athanor Infra (Postgres + Consul)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=$ROOT_DIR
EnvironmentFile=$ROOT_DIR/.env
ExecStart=$COMPOSE_CMD up -d
ExecStop=$COMPOSE_CMD down
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo tee "$BACKEND_SERVICE" >/dev/null <<EOF
[Unit]
Description=Athanor Backend API
After=network.target vllm-cluster-infra.service
Requires=vllm-cluster-infra.service

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR/backend
EnvironmentFile=$ROOT_DIR/backend/.env
ExecStart=$ROOT_DIR/backend/.venv/bin/uvicorn app.main:app --host \${ADMIN_API_HOST} --port \${ADMIN_API_PORT}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo tee "$FRONTEND_SERVICE" >/dev/null <<EOF
[Unit]
Description=Athanor Frontend
After=network.target vllm-cluster-backend.service
Requires=vllm-cluster-backend.service

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR/frontend
EnvironmentFile=$ROOT_DIR/frontend/.env
Environment=PATH=$(dirname "$NODE_BIN"):$(dirname "$NPM_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$NPM_BIN run dev -- --host \${FRONTEND_HOST} --port \${FRONTEND_PORT}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd and enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-cluster-infra.service
sudo systemctl enable --now vllm-cluster-backend.service
sudo systemctl enable --now vllm-cluster-frontend.service

echo
echo "Done."
echo "Check status with:"
echo "  sudo systemctl status vllm-cluster-infra.service"
echo "  sudo systemctl status vllm-cluster-backend.service"
echo "  sudo systemctl status vllm-cluster-frontend.service"
