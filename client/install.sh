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

find_highest_available_cuda() {
  local vllm_version="$1"
  local cpu_arch="$2"
  local max_search=200  # Reasonable upper limit to prevent infinite loop
  local highest_found=""
  
  # Search from cu100 up to max_search
  for cu_ver in $(seq 100 "$max_search"); do
    local wheel_url="https://github.com/vllm-project/vllm/releases/download/v${vllm_version}/vllm-${vllm_version}+cu${cu_ver}-cp38-abi3-manylinux_2_35_${cpu_arch}.whl"
    if curl -sSfI "$wheel_url" >/dev/null 2>&1; then
      highest_found="$cu_ver"
    else
      # If we've found at least one and now hit a miss, check a few more to be sure
      if [[ -n "$highest_found" ]]; then
        # Check if we've hit a gap of 5 consecutive misses
        local consecutive_misses=1
        local check_ahead=5
        for offset in $(seq 1 "$check_ahead"); do
          local next_cu=$((cu_ver + offset))
          local next_url="https://github.com/vllm-project/vllm/releases/download/v${vllm_version}/vllm-${vllm_version}+cu${next_cu}-cp38-abi3-manylinux_2_35_${cpu_arch}.whl"
          if curl -sSfI "$next_url" >/dev/null 2>&1; then
            highest_found="$next_cu"
            consecutive_misses=0
            break
          else
            ((consecutive_misses++))
          fi
        done
        
        # If we had consecutive misses, we've likely found the highest
        if [[ $consecutive_misses -ge $check_ahead ]]; then
          break
        fi
      fi
    fi
  done
  
  if [[ -n "$highest_found" ]]; then
    echo "$highest_found"
    return 0
  fi
  
  return 1
}

prompt_yes_no() {
  local prompt="$1"
  local response
  while true; do
    read -r -p "$prompt (y/n): " response
    case "$response" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      [Nn]|[Nn][Oo]) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
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

# Check if exact match exists
if ! curl -sSfI "$VLLM_WHEEL_URL" >/dev/null 2>&1; then
  echo "No vLLM wheel found for exact CUDA version ${CUDA_VERSION_RAW} (cu${CUDA_VERSION})." >&2
  
  # Try to find the highest available CUDA version
  echo "Searching for highest available CUDA version wheel..." >&2
  HIGHEST_CUDA="$(find_highest_available_cuda "$VLLM_VERSION" "$CPU_ARCH")"
  
  if [[ -z "$HIGHEST_CUDA" ]]; then
    echo "No compatible vLLM wheel found for any CUDA version on ${CPU_ARCH}." >&2
    exit 1
  fi
  
  HIGHEST_CUDA_MAJOR="$((HIGHEST_CUDA / 10))"
  HIGHEST_CUDA_MINOR="$((HIGHEST_CUDA % 10))"
  HIGHEST_CUDA_VERSION_RAW="${HIGHEST_CUDA_MAJOR}.${HIGHEST_CUDA_MINOR}"
  
  echo
  echo "========================================================================"
  echo "                     CUDA VERSION MISMATCH WARNING"
  echo "========================================================================"
  echo "Your CUDA version:           ${CUDA_VERSION_RAW} (cu${CUDA_VERSION})"
  echo "Highest available wheel:     ${HIGHEST_CUDA_VERSION_RAW} (cu${HIGHEST_CUDA})"
  echo
  
  if [[ $CUDA_VERSION -gt $HIGHEST_CUDA ]]; then
    echo "Your CUDA version is NEWER than the available wheel."
    echo "This may work due to CUDA forward compatibility, but is not guaranteed."
  elif [[ $CUDA_VERSION -lt $HIGHEST_CUDA ]]; then
    echo "Your CUDA version is OLDER than the available wheel."
    echo "This wheel may not be compatible with your CUDA installation."
  fi
  
  echo "========================================================================"
  echo
  
  if ! prompt_yes_no "Do you want to continue with CUDA ${HIGHEST_CUDA_VERSION_RAW} wheel?"; then
    echo "Installation cancelled by user." >&2
    exit 1
  fi
  
  # Update to use the highest available version
  CUDA_VERSION="$HIGHEST_CUDA"
  VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl"
  echo
  echo "Proceeding with CUDA ${HIGHEST_CUDA_VERSION_RAW} wheel..."
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

echo "Installing vLLM matching CUDA version cu${CUDA_VERSION}..."
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