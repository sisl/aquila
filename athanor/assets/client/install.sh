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

# vLLM now runs inside the official vllm/vllm-openai Docker containers, so the
# client node needs a working Docker engine with GPU access via the NVIDIA
# Container Toolkit. We verify these prerequisites and print precise install
# instructions if they are missing — we never mutate the system ourselves.
check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Docker is not installed. The client runs vLLM in official Docker containers.

Install Docker Engine, e.g. on Ubuntu:
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"   # then log out/in so the group takes effect

Docs: https://docs.docker.com/engine/install/
EOF
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    cat >&2 <<EOF
Docker is installed but the daemon is not reachable by user '$(id -un)'.

Fixes:
  - Start the daemon:    sudo systemctl enable --now docker
  - Grant access:        sudo usermod -aG docker "$USER"  (then log out/in)
EOF
    exit 1
  fi
}

check_nvidia_runtime() {
  # Authoritative check: can Docker actually expose GPUs to a container? This
  # covers both the legacy 'nvidia' runtime and modern CDI setups, and avoids
  # false negatives from only inspecting docker info.
  local test_img=""
  for img in ubuntu:22.04 ubuntu:latest busybox:latest; do
    if docker image inspect "$img" >/dev/null 2>&1; then test_img="$img"; break; fi
  done
  [[ -z "$test_img" ]] && test_img="busybox:latest"  # tiny; pulled if absent
  if docker run --rm --gpus all "$test_img" true >/dev/null 2>&1; then
    return
  fi
  # Fallback heuristic in case the test image could not be pulled (offline):
  # the NVIDIA Container Toolkit registers an 'nvidia' runtime and/or nvidia-ctk.
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
     || command -v nvidia-ctk >/dev/null 2>&1 \
     || command -v nvidia-container-runtime >/dev/null 2>&1; then
    return
  fi
  cat >&2 <<'EOF'
The NVIDIA Container Toolkit was not detected. vLLM containers need GPU access.

Install it, e.g. on Ubuntu:
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

Verify with:
  docker run --rm --gpus all ubuntu nvidia-smi

Docs: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
EOF
  exit 1
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

echo "Checking Docker + NVIDIA Container Toolkit..."
check_docker
check_nvidia_runtime
echo "Docker with GPU access detected."
echo

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
uv pip install -r "$ROOT_DIR/requirements.txt"

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
echo "Note: the service runs as $(id -un); this user must be able to use Docker"
echo "      (member of the 'docker' group or root)."
sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=VLLM Cluster Client
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=$(id -un)
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
