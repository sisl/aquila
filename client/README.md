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

## Install (systemd via CLI)
Install the CLI and start the client:

```bash
uv pip install vllm_cluster_manager
vllm_cluster_manager client up --host_ip 127.0.0.1 --host_discover_port 47528
```
This runs in the foreground without sudo. Use `client service install` for a persistent systemd service.

Stop the client with:
```bash
vllm_cluster_manager client down
```

If you only want to install the systemd unit without starting it immediately:
```bash
vllm_cluster_manager client service install
```

Remove the systemd unit:
```bash
vllm_cluster_manager client service remove
```

## Run without sudo (foreground)
`client up` already runs in the foreground. If you want to run it manually from the runtime directory:
```bash
cd ~/.local/share/vllm_cluster_manager/client
source .venv/bin/activate
python -m app.main
```

## CLI flags
| Command | Flags |
| --- | --- |
| `client up` | `--host_ip`, `--host_discover_port`, `--client_host`, `--client_port`, `--node_name` |
| `client down` | None |
| `client service install` | Same as `client up` |
| `client service remove` | None |

Key options:
- `--client_host` / `--client_port`: bind host and port.
- `--host_ip` / `--host_discover_port`: discovery host/port.
- `--node_name`: name used for registration.

The CLI then:
- Creates a venv under the CLI runtime directory.
- Installs dependencies and the matching vLLM wheel.
- Writes the client `.env`.
- Creates and enables `vllm-cluster-client.service`.

Runtime files are stored under `~/.local/share/vllm_cluster_manager/client`.

### Check status
```bash
sudo systemctl status vllm-cluster-client.service
```

### View logs
```bash
journalctl -u vllm-cluster-client.service -f
```

## Configuration
The client uses `~/.local/share/vllm_cluster_manager/client/.env`:
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

Network paths to allow:
- Host → Client agent: TCP `client_port`.
- Client → Host port: TCP `host_discover_port`.

## Uninstall (systemd)
```bash
sudo systemctl disable --now vllm-cluster-client.service
sudo rm -f /etc/systemd/system/vllm-cluster-client.service
sudo systemctl daemon-reload
```
