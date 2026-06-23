"""Tests for the CLI helper functions (cli.py)."""

import argparse
import hashlib
import os
import socket
from pathlib import Path
from unittest import mock

import pytest

from aquila.cli import (
    HostConfig,
    ClientConfig,
    CheckStatus,
    PreflightResult,
    build_host_config,
    build_client_config,
    format_kv,
    load_env_file,
    merge_env,
    file_sha256,
    needs_install,
    write_hash_marker,
    write_pid,
    remove_pid,
    stop_pid,
    _check_python,
    _check_node,
    _check_npm,
    _check_docker,
    _check_compose,
    _check_docker_client,
    _check_podman_client,
    _check_nvidia_smi,
    _check_nvidia_ctk,
    _check_port,
    preflight_host,
    preflight_client,
    run_preflight,
)


# ---------------------------------------------------------------------------
# build_host_config / build_client_config
# ---------------------------------------------------------------------------


def test_build_host_config():
    ns = argparse.Namespace(
        host_ip="10.0.0.1",
        host_frontend_port=3000,
        host_backend_port=8000,
        host_discover_port=47528,
        postgres_host="127.0.0.1",
        postgres_port=5757,
        postgres_db="testdb",
        postgres_user="user",
        postgres_password="pw",
        base_path="/app",
    )
    cfg = build_host_config(ns)
    assert isinstance(cfg, HostConfig)
    assert cfg.host_ip == "10.0.0.1"
    assert cfg.frontend_port == 3000
    assert cfg.admin_api_port == 8000
    assert cfg.consul_port == 47528
    assert cfg.postgres_db == "testdb"
    assert cfg.base_path == "/app"


def test_build_client_config():
    ns = argparse.Namespace(
        host_ip="10.0.0.1",
        host_discover_port=47528,
        client_host="0.0.0.0",
        client_port=9000,
        node_name="gpu-node-1",
    )
    cfg = build_client_config(ns)
    assert isinstance(cfg, ClientConfig)
    assert cfg.host_ip == "10.0.0.1"
    assert cfg.consul_port == 47528
    assert cfg.node_name == "gpu-node-1"


# ---------------------------------------------------------------------------
# format_kv
# ---------------------------------------------------------------------------


def test_format_kv():
    result = format_kv({"a": 1, "b": "hello"})
    assert "a=1" in result
    assert "b=hello" in result


def test_format_kv_empty():
    assert format_kv({}) == ""


# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------


def test_load_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KEY1=value1\nKEY2=value2\n# comment\n\nKEY3=has=equals\n",
        encoding="utf-8",
    )
    result = load_env_file(env_file)
    assert result == {"KEY1": "value1", "KEY2": "value2", "KEY3": "has=equals"}


def test_load_env_file_missing(tmp_path):
    result = load_env_file(tmp_path / "nonexistent")
    assert result == {}


def test_load_env_file_strips_whitespace(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("  KEY = value  \n", encoding="utf-8")
    result = load_env_file(env_file)
    assert result == {"KEY": "value"}


# ---------------------------------------------------------------------------
# merge_env
# ---------------------------------------------------------------------------


def test_merge_env():
    with mock.patch.dict(os.environ, {"EXISTING": "1"}, clear=True):
        result = merge_env({"NEW": "2"})
        assert result["EXISTING"] == "1"
        assert result["NEW"] == "2"


def test_merge_env_overrides():
    with mock.patch.dict(os.environ, {"KEY": "old"}, clear=True):
        result = merge_env({"KEY": "new"})
        assert result["KEY"] == "new"


# ---------------------------------------------------------------------------
# file_sha256 / needs_install / write_hash_marker
# ---------------------------------------------------------------------------


def test_file_sha256(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world\n", encoding="utf-8")
    expected = hashlib.sha256(b"hello world\n").hexdigest()
    assert file_sha256(f) == expected


def test_needs_install_no_requirements(tmp_path):
    assert needs_install(tmp_path / "missing.txt", tmp_path / "marker") is False


def test_needs_install_no_marker(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("fastapi\n", encoding="utf-8")
    assert needs_install(req, tmp_path / "marker") is True


def test_needs_install_matching_marker(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("fastapi\n", encoding="utf-8")
    marker = tmp_path / "marker"
    write_hash_marker(req, marker)
    assert needs_install(req, marker) is False


def test_needs_install_changed_requirements(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("fastapi\n", encoding="utf-8")
    marker = tmp_path / "marker"
    write_hash_marker(req, marker)
    req.write_text("fastapi\nuvicorn\n", encoding="utf-8")
    assert needs_install(req, marker) is True


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------


def test_write_and_remove_pid(tmp_path):
    pid_file = tmp_path / "test.pid"
    write_pid(pid_file, 12345)
    assert pid_file.read_text() == "12345"
    remove_pid(pid_file)
    assert not pid_file.exists()


def test_remove_pid_missing(tmp_path):
    remove_pid(tmp_path / "missing.pid")


def test_stop_pid_missing(tmp_path):
    stop_pid(tmp_path / "missing.pid")


def test_stop_pid_invalid_content(tmp_path):
    pid_file = tmp_path / "test.pid"
    pid_file.write_text("not-a-number", encoding="utf-8")
    stop_pid(pid_file)


def test_stop_pid_no_such_process(tmp_path):
    pid_file = tmp_path / "test.pid"
    pid_file.write_text("999999999", encoding="utf-8")
    stop_pid(pid_file)
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# write_host_env_files / write_client_env_file
# ---------------------------------------------------------------------------


def test_write_host_env_files(tmp_path):
    from aquila.cli import write_host_env_files

    cfg = HostConfig(
        host_ip="10.0.0.1",
        frontend_port=3000,
        admin_api_port=8000,
        consul_port=47528,
        postgres_host="127.0.0.1",
        postgres_port=5757,
        postgres_db="testdb",
        postgres_user="user",
        postgres_password="pw",
        base_path="/",
    )
    (tmp_path / "backend").mkdir()
    (tmp_path / "frontend").mkdir()
    write_host_env_files(tmp_path, cfg)

    root_env = load_env_file(tmp_path / ".env")
    assert root_env["POSTGRES_DB"] == "testdb"
    assert root_env["CONSUL_PORT"] == "47528"

    backend_env = load_env_file(tmp_path / "backend" / ".env")
    assert backend_env["ADMIN_API_HOST"] == "10.0.0.1"
    assert backend_env["POSTGRES_HOST"] == "127.0.0.1"

    frontend_env = load_env_file(tmp_path / "frontend" / ".env")
    assert frontend_env["VITE_BACKEND_PORT"] == "8000"


def test_write_client_env_file(tmp_path):
    from aquila.cli import write_client_env_file

    cfg = ClientConfig(
        host_ip="10.0.0.1",
        consul_port=47528,
        client_host="0.0.0.0",
        client_port=9000,
        node_name="gpu-1",
    )
    write_client_env_file(tmp_path, cfg)

    env = load_env_file(tmp_path / ".env")
    assert env["NODE_NAME"] == "gpu-1"
    assert env["PORT"] == "9000"
    assert "47528" in env["CONSUL_HTTP_ADDR"]


# ---------------------------------------------------------------------------
# run_clean
# ---------------------------------------------------------------------------


def test_run_clean_removes_working_dirs(tmp_path, monkeypatch):
    from aquila.cli import run_clean

    data_root = tmp_path / "share" / "aquila"
    # Only the client subtree (no host dir) so clean doesn't shell out to docker.
    (data_root / "client" / ".venv").mkdir(parents=True)
    client_root = tmp_path / ".vllm-client"
    (client_root / ".packages").mkdir(parents=True)

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("VLLM_CLIENT_ROOT", str(client_root))

    run_clean(remove_docker=False, assume_yes=True)

    assert not data_root.exists()
    assert not client_root.exists()


def test_run_clean_nothing_to_do(tmp_path, monkeypatch, capsys):
    from aquila.cli import run_clean

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty-share"))
    monkeypatch.setenv("VLLM_CLIENT_ROOT", str(tmp_path / "empty-client"))

    run_clean(remove_docker=False, assume_yes=True)

    assert "Nothing to clean." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Postgres volume persistence (stop_infra / host down --purge)
# ---------------------------------------------------------------------------


class TestInfraPersistence:
    def test_stop_infra_keeps_volumes_by_default(self, tmp_path):
        from aquila import cli as cli_mod

        with mock.patch.object(
            cli_mod, "detect_compose_cmd", return_value="docker compose"
        ), mock.patch.object(cli_mod, "run") as run_cmd:
            cli_mod.stop_infra(tmp_path)
        cmd = run_cmd.call_args.args[0]
        assert cmd[-1] == "down"
        assert "-v" not in cmd

    def test_stop_infra_purge_removes_volumes(self, tmp_path):
        from aquila import cli as cli_mod

        with mock.patch.object(
            cli_mod, "detect_compose_cmd", return_value="docker compose"
        ), mock.patch.object(cli_mod, "run") as run_cmd:
            cli_mod.stop_infra(tmp_path, purge=True)
        cmd = run_cmd.call_args.args[0]
        assert cmd[-2:] == ["down", "-v"]

    @pytest.mark.parametrize("purge", [False, True])
    def test_run_host_down_threads_purge(self, tmp_path, purge):
        from aquila import cli as cli_mod

        with mock.patch.object(
            cli_mod, "runtime_dir_path", return_value=tmp_path
        ), mock.patch.object(cli_mod, "stop_infra") as stop_infra, mock.patch.object(
            cli_mod, "stop_pid"
        ), mock.patch.object(cli_mod, "remove_host_service"), mock.patch.object(
            cli_mod, "remove_runtime_dir"
        ):
            cli_mod.run_host_down(purge=purge)
        stop_infra.assert_called_once_with(tmp_path, purge=purge)

    def test_infra_service_execstop_keeps_volumes(self, tmp_path):
        from aquila import cli as cli_mod

        units: dict[str, str] = {}

        with mock.patch.object(
            cli_mod, "ensure_runtime_dir", return_value=tmp_path
        ), mock.patch.object(
            cli_mod.shutil, "which", return_value="/usr/bin/tool"
        ), mock.patch.object(
            cli_mod,
            "write_systemd_service",
            side_effect=lambda path, content: units.__setitem__(path, content),
        ), mock.patch.object(cli_mod, "systemctl"):
            cli_mod.install_host_service(
                HostConfig(
                    host_ip="127.0.0.1",
                    frontend_port=5173,
                    admin_api_port=8000,
                    consul_port=47528,
                    postgres_host="127.0.0.1",
                    postgres_port=5757,
                    postgres_db="db",
                    postgres_user="u",
                    postgres_password="p",
                    base_path="/",
                )
            )
        infra_unit = next(text for path, text in units.items() if "infra" in path)
        assert "down -v" not in infra_unit
        assert "docker compose down" in infra_unit

    def test_run_clean_purges_volumes(self, tmp_path, monkeypatch):
        from aquila import cli as cli_mod

        data_root = tmp_path / "share" / "aquila"
        (data_root / "host").mkdir(parents=True)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
        monkeypatch.setenv("VLLM_CLIENT_ROOT", str(tmp_path / "no-client"))

        with mock.patch.object(cli_mod, "stop_infra") as stop_infra, mock.patch.object(
            cli_mod, "stop_pid"
        ):
            cli_mod.run_clean(remove_docker=False, assume_yes=True)
        stop_infra.assert_called_once_with(data_root / "host", purge=True)


# ---------------------------------------------------------------------------
# Asset refresh (_refresh_tree / copy_assets) — content-only, no metadata
# ---------------------------------------------------------------------------


def test_refresh_tree_updates_content_and_skips_ignored(tmp_path):
    from aquila import cli as cli_mod

    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("new")
    (src / "sub" / "b.txt").write_text("nested")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_text("junk")
    (src / "mod.pyc").write_text("bytecode")

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.txt").write_text("stale")  # already present -> overwritten

    cli_mod._refresh_tree(src, dest)

    assert (dest / "a.txt").read_text() == "new"
    assert (dest / "sub" / "b.txt").read_text() == "nested"
    # Ignore patterns are honored (no stale bytecode in the runtime tree).
    assert not (dest / "__pycache__").exists()
    assert not (dest / "mod.pyc").exists()


def test_refresh_tree_does_not_touch_directory_metadata(tmp_path, monkeypatch):
    """Regression: a runtime subdir chowned by a container (e.g. Consul takes
    infra/consul as uid 100) must not break a re-copy. copytree failed there via
    copystat -> os.utime (EPERM); _refresh_tree must never call those."""
    from aquila import cli as cli_mod

    src = tmp_path / "src"
    (src / "infra" / "consul").mkdir(parents=True)
    (src / "infra" / "consul" / "consul.hcl").write_text("config v2")
    dest = tmp_path / "dest"
    (dest / "infra" / "consul").mkdir(parents=True)
    (dest / "infra" / "consul" / "consul.hcl").write_text("config v1 stale")

    def _forbidden(name):
        def _boom(*_a, **_k):
            pytest.fail(f"{name} must not be called by _refresh_tree")

        return _boom

    monkeypatch.setattr(cli_mod.shutil, "copystat", _forbidden("shutil.copystat"))
    monkeypatch.setattr(cli_mod.shutil, "copy2", _forbidden("shutil.copy2"))
    monkeypatch.setattr(cli_mod.os, "utime", _forbidden("os.utime"))

    cli_mod._refresh_tree(src, dest)

    assert (dest / "infra" / "consul" / "consul.hcl").read_text() == "config v2"


def test_copy_assets_subdir_missing_raises(tmp_path):
    from aquila import cli as cli_mod

    with pytest.raises(RuntimeError, match="Missing packaged assets"):
        cli_mod.copy_assets_subdir("host", "definitely-not-a-real-subdir", tmp_path / "d")


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


class TestCheckPython:
    def test_pass(self):
        result = _check_python()
        assert result.status == CheckStatus.PASS
        assert "Python" in result.message

    def test_fail_on_old_version(self, monkeypatch):
        monkeypatch.setattr("sys.version_info", (3, 9, 0, "final", 0))
        result = _check_python()
        assert result.status == CheckStatus.FAIL
        assert result.hint


class TestCheckNode:
    def test_pass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/node" if cmd == "node" else None)
        fake = mock.MagicMock()
        fake.stdout = "v23.6.0\n"
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_node()
        assert result.status == CheckStatus.PASS
        assert "v23.6.0" in result.message

    def test_fail_old_version(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/node" if cmd == "node" else None)
        fake = mock.MagicMock()
        fake.stdout = "v18.0.0\n"
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_node()
        assert result.status == CheckStatus.FAIL

    def test_fail_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_node()
        assert result.status == CheckStatus.FAIL
        assert "Not found" in result.message


class TestCheckNpm:
    def test_pass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/npm" if cmd == "npm" else None)
        result = _check_npm()
        assert result.status == CheckStatus.PASS

    def test_fail(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_npm()
        assert result.status == CheckStatus.FAIL


class TestCheckDocker:
    def test_pass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
        fake = mock.MagicMock()
        fake.stdout = "27.0.3\n"
        fake.stderr = ""
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_docker()
        assert result.status == CheckStatus.PASS
        assert "27.0.3" in result.message

    def test_fail_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_docker()
        assert result.status == CheckStatus.FAIL

    def test_fail_permission_denied(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
        fake = mock.MagicMock()
        fake.stdout = ""
        fake.stderr = "Got permission denied while trying to connect"
        fake.returncode = 1
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_docker()
        assert result.status == CheckStatus.FAIL
        assert "docker group" in result.hint


class TestCheckCompose:
    def test_pass_plugin(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
        fake = mock.MagicMock()
        fake.stdout = "2.29.1\n"
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_compose()
        assert result.status == CheckStatus.PASS

    def test_fail(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_compose()
        assert result.status == CheckStatus.FAIL


class TestCheckDockerClient:
    def test_pass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
        fake = mock.MagicMock()
        fake.stdout = "27.0.3\n"
        fake.stderr = ""
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_docker_client()
        assert result.status == CheckStatus.PASS
        assert "27.0.3" in result.message

    def test_warn_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_docker_client()
        assert result.status == CheckStatus.WARN

    def test_warn_permission_denied(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
        fake = mock.MagicMock()
        fake.stdout = ""
        fake.stderr = "Got permission denied while trying to connect"
        fake.returncode = 1
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_docker_client()
        assert result.status == CheckStatus.WARN
        assert "docker group" in result.hint


class TestCheckPodmanClient:
    def test_pass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/podman" if cmd == "podman" else None)
        fake = mock.MagicMock()
        fake.stdout = "5.4.1\n"
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_podman_client()
        assert result.status == CheckStatus.PASS
        assert "5.4.1" in result.message

    def test_warn_old_version(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/podman" if cmd == "podman" else None)
        fake = mock.MagicMock()
        fake.stdout = "4.9.3\n"
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_podman_client()
        assert result.status == CheckStatus.WARN
        assert "CDI" in result.hint

    def test_warn_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_podman_client()
        assert result.status == CheckStatus.WARN


class TestCheckNvidiaSmi:
    def test_warn_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = _check_nvidia_smi()
        assert result.status == CheckStatus.WARN

    def test_pass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None)
        fake = mock.MagicMock()
        fake.stdout = "NVIDIA H100\nNVIDIA H100\n"
        fake.returncode = 0
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake)
        result = _check_nvidia_smi()
        assert result.status == CheckStatus.PASS
        assert "2 GPU" in result.message


class TestCheckNvidiaCtk:
    def test_pass_binary(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/nvidia-ctk" if cmd == "nvidia-ctk" else None)
        result = _check_nvidia_ctk()
        assert result.status == CheckStatus.PASS

    def test_warn_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        result = _check_nvidia_ctk()
        assert result.status == CheckStatus.WARN


class TestCheckPort:
    def test_pass_available(self):
        result = _check_port(0, "test")
        assert result.status == CheckStatus.PASS

    def test_fail_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 0))
            s.listen(1)
            port = s.getsockname()[1]
            result = _check_port(port, "test", "--some-flag")
            assert result.status == CheckStatus.FAIL
            assert "--some-flag" in result.hint


class TestPreflightOrchestrators:
    def test_preflight_host_returns_results(self):
        config = HostConfig(
            host_ip="127.0.0.1", frontend_port=0, admin_api_port=0,
            consul_port=0, postgres_host="127.0.0.1", postgres_port=0,
            postgres_db="db", postgres_user="u", postgres_password="p",
            base_path="/",
        )
        with mock.patch("aquila.cli._check_port", return_value=PreflightResult("Port", CheckStatus.PASS, "ok")):
            results = preflight_host(config)
        assert len(results) == 9
        assert all(isinstance(r, PreflightResult) for r in results)

    def test_preflight_client_returns_results(self):
        config = ClientConfig(
            host_ip="127.0.0.1", consul_port=0,
            client_host="0.0.0.0", client_port=0, node_name="n",
        )
        with mock.patch("aquila.cli._check_port", return_value=PreflightResult("Port", CheckStatus.PASS, "ok")):
            results = preflight_client(config)
        assert len(results) == 6
        assert all(isinstance(r, PreflightResult) for r in results)

    def test_preflight_client_fails_when_no_runtime(self):
        config = ClientConfig(
            host_ip="127.0.0.1", consul_port=0,
            client_host="0.0.0.0", client_port=0, node_name="n",
        )
        with mock.patch("aquila.cli._check_docker_client",
                         return_value=PreflightResult("Docker", CheckStatus.WARN, "Not found")), \
             mock.patch("aquila.cli._check_podman_client",
                         return_value=PreflightResult("Podman", CheckStatus.WARN, "Not found")), \
             mock.patch("aquila.cli._check_port",
                         return_value=PreflightResult("Port", CheckStatus.PASS, "ok")):
            results = preflight_client(config)
        docker_r = next(r for r in results if r.label == "Docker")
        podman_r = next(r for r in results if r.label == "Podman")
        assert docker_r.status == CheckStatus.FAIL
        assert podman_r.status == CheckStatus.FAIL

    def test_preflight_client_warns_when_one_runtime(self):
        config = ClientConfig(
            host_ip="127.0.0.1", consul_port=0,
            client_host="0.0.0.0", client_port=0, node_name="n",
        )
        with mock.patch("aquila.cli._check_docker_client",
                         return_value=PreflightResult("Docker", CheckStatus.PASS, "Docker 27.0.3")), \
             mock.patch("aquila.cli._check_podman_client",
                         return_value=PreflightResult("Podman", CheckStatus.WARN, "Not found")), \
             mock.patch("aquila.cli._check_port",
                         return_value=PreflightResult("Port", CheckStatus.PASS, "ok")):
            results = preflight_client(config)
        docker_r = next(r for r in results if r.label == "Docker")
        podman_r = next(r for r in results if r.label == "Podman")
        assert docker_r.status == CheckStatus.PASS
        assert podman_r.status == CheckStatus.WARN


class TestRunPreflight:
    def test_all_pass(self, capsys):
        results = [
            PreflightResult("Check A", CheckStatus.PASS, "ok"),
            PreflightResult("Check B", CheckStatus.PASS, "ok"),
        ]
        run_preflight(results)
        out = capsys.readouterr().out
        assert "Preflight checks:" in out
        assert "✓" in out

    def test_warn_does_not_exit(self, capsys):
        results = [
            PreflightResult("Check A", CheckStatus.PASS, "ok"),
            PreflightResult("Check B", CheckStatus.WARN, "maybe", hint="try this"),
        ]
        run_preflight(results)
        out = capsys.readouterr().out
        assert "1 warning(s)" in out

    def test_fail_exits_non_tty(self, monkeypatch):
        results = [
            PreflightResult("Check A", CheckStatus.PASS, "ok"),
            PreflightResult("Check B", CheckStatus.FAIL, "bad", hint="fix it"),
        ]
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
        with pytest.raises(SystemExit) as exc_info:
            run_preflight(results)
        assert exc_info.value.code == 1

    def test_fail_shows_hint(self, capsys, monkeypatch):
        results = [
            PreflightResult("Check A", CheckStatus.FAIL, "bad", hint="do this"),
        ]
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
        with pytest.raises(SystemExit):
            run_preflight(results)
        out = capsys.readouterr().out
        assert "do this" in out

    def test_fail_exits_on_decline(self, monkeypatch):
        results = [
            PreflightResult("Check A", CheckStatus.FAIL, "bad", hint="fix it"),
        ]
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("builtins.input", lambda _: "n")
        with pytest.raises(SystemExit) as exc_info:
            run_preflight(results)
        assert exc_info.value.code == 1

    def test_fail_continues_on_confirm(self, capsys, monkeypatch):
        results = [
            PreflightResult("Check A", CheckStatus.PASS, "ok"),
            PreflightResult("Check B", CheckStatus.FAIL, "bad", hint="fix it"),
        ]
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("builtins.input", lambda _: "y")
        run_preflight(results)
        out = capsys.readouterr().out
        assert "1 check(s) failed" in out
