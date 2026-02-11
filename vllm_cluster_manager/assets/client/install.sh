#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="vllm-cluster-client"

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
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation completed, but uv is still not on PATH." >&2
    echo "Try opening a new shell or add ~/.local/bin to PATH." >&2
    exit 1
  fi
}

require_debian_pkg() {
  local pkg="$1"
  if ! command -v dpkg-query >/dev/null 2>&1; then
    echo "dpkg-query not found; cannot verify package $pkg." >&2
    echo "Ensure required system packages are installed and retry." >&2
    exit 1
  fi
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    echo "Missing required system package: $pkg" >&2
    echo "Install with: sudo apt install -y $pkg" >&2
    exit 1
  fi
}

check_system_requirements() {
  require_debian_pkg python3.12-dev
  require_debian_pkg build-essential
}

detect_cuda_version() {
  local cuda_ver=""
  if command -v nvcc >/dev/null 2>&1; then
    cuda_ver="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n 1)"
  fi
  if [[ -z "$cuda_ver" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_ver="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1)"
  fi
  if [[ -z "$cuda_ver" ]]; then
    echo "Unable to detect CUDA version. Ensure nvcc or nvidia-smi is available on PATH." >&2
    exit 1
  fi
  echo "$cuda_ver"
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

ensure_uv
require_cmd systemctl
require_cmd curl

echo "VLLM Cluster Client installer"
echo "Working directory: $ROOT_DIR"
echo

check_system_requirements

CUDA_VERSION_RAW="$(detect_cuda_version)"
CUDA_MAJOR="${CUDA_VERSION_RAW%%.*}"
CUDA_MINOR="${CUDA_VERSION_RAW##*.}"
CUDA_VERSION="$((CUDA_MAJOR * 10 + CUDA_MINOR))"
CPU_ARCH="$(uname -m)"
VLLM_VERSION="$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | sed -n 's/.*"tag_name":[[:space:]]*"v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1)"
if [[ -z "$VLLM_VERSION" ]]; then
  echo "Unable to determine latest vLLM version from GitHub releases." >&2
  exit 1
fi

VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl"
if ! curl -sSfI "$VLLM_WHEEL_URL" >/dev/null; then
  echo "No vLLM wheel found for CUDA ${CUDA_VERSION_RAW} (cu${CUDA_VERSION}) on ${CPU_ARCH}." >&2
  echo "Checked: $VLLM_WHEEL_URL" >&2
  exit 1
fi

API_HOST="$(prompt_default "Client URI (bind host)" "0.0.0.0")"
API_PORT="$(prompt_default "Client port" "9000")"
echo "Note: Client port must be reachable via a TCP firewall rule."
CONSUL_HOST="$(prompt_default "Host URI (IP/DNS)" "127.0.0.1")"
CONSUL_PORT="$(prompt_default "Host port" "8500")"
NODE_NAME="$(prompt_default "Node name (used in Consul)" "$(hostname)")"

CONSUL_HTTP_ADDR="http://${CONSUL_HOST}:${CONSUL_PORT}"

echo
echo "Creating virtual environment..."
uv venv --python=3.12 "$ROOT_DIR/.venv"
# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

echo "Installing client dependencies..."
REQ_NO_VLLM="$ROOT_DIR/.requirements-no-vllm.txt"
grep -v -E '^[[:space:]]*vllm([[:space:]]|$)' "$ROOT_DIR/requirements.txt" > "$REQ_NO_VLLM"
uv pip install -r "$REQ_NO_VLLM"
rm -f "$REQ_NO_VLLM"

echo "Installing vLLM matching detected CUDA version..."
uv pip install "$VLLM_WHEEL_URL" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"

ENV_FILE="$ROOT_DIR/.env"
echo
echo "Writing $ENV_FILE..."
cat > "$ENV_FILE" <<EOF
NODE_NAME=$NODE_NAME
HOST=$API_HOST
PORT=$API_PORT
CONSUL_HTTP_ADDR=$CONSUL_HTTP_ADDR
EOF

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
echo
echo "Creating systemd service at $SERVICE_FILE (requires sudo)..."
sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=VLLM Cluster Client
After=network.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$ROOT_DIR/.venv/bin/python -m app.main
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo
echo "Done."
echo "Check status with: sudo systemctl status $SERVICE_NAME"
