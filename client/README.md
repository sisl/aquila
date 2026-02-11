# Client (GPU Node Agent)

The client runs on each GPU node and registers itself with the host via Consul. It launches vLLM workloads using the model configurations you define in the admin UI.

## Prerequisites
- NVIDIA GPU with a CUDA toolkit installed.
- `nvcc` or `nvidia-smi` available on PATH (used to detect CUDA version).
- Python 3.12 + `python3.12-dev` and `build-essential` (Debian/Ubuntu).
- `systemctl` (systemd) and `curl`.

On Debian/Ubuntu:
```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

Note: The installer downloads a vLLM wheel that matches your detected CUDA version. If a matching wheel does not exist for your CUDA version/architecture, the installer will stop and report the URL it checked.

## Install (systemd)
Run the installer from the `client` directory:

```bash
cd client
bash install.sh
```

You will be prompted for:
- Client bind host/port (must be reachable by the host)
- Host (Consul) URI/port
- Node name (used for registration)

The installer then:
- Creates a venv at `client/.venv`.
- Installs dependencies and the matching vLLM wheel.
- Writes `client/.env`.
- Creates and enables `vllm-cluster-client.service`.

### Check status
```bash
sudo systemctl status vllm-cluster-client.service
```

### View logs
```bash
journalctl -u vllm-cluster-client.service -f
```

## Configuration
The client uses `client/.env`:
- `NODE_NAME`
- `HOST`
- `PORT`
- `CONSUL_HTTP_ADDR`

After editing, restart the service:
```bash
sudo systemctl restart vllm-cluster-client.service
```

## Firewall notes
- Ensure the client port is reachable from the host.
- Ensure the host Consul port is reachable from the client.

## Uninstall (systemd)
```bash
sudo systemctl disable --now vllm-cluster-client.service
sudo rm -f /etc/systemd/system/vllm-cluster-client.service
sudo systemctl daemon-reload
```
