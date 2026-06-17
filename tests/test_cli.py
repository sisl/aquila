"""Tests for the CLI helper functions (cli.py)."""

import argparse
import hashlib
import os
from pathlib import Path
from unittest import mock

import pytest

from vllm_cluster_manager.cli import (
    HostConfig,
    ClientConfig,
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
    from vllm_cluster_manager.cli import write_host_env_files

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
    from vllm_cluster_manager.cli import write_client_env_file

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
    from vllm_cluster_manager.cli import run_clean

    data_root = tmp_path / "share" / "vllm_cluster_manager"
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
    from vllm_cluster_manager.cli import run_clean

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty-share"))
    monkeypatch.setenv("VLLM_CLIENT_ROOT", str(tmp_path / "empty-client"))

    run_clean(remove_docker=False, assume_yes=True)

    assert "Nothing to clean." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Postgres volume persistence (stop_infra / host down --purge)
# ---------------------------------------------------------------------------


class TestInfraPersistence:
    def test_stop_infra_keeps_volumes_by_default(self, tmp_path):
        from vllm_cluster_manager import cli as cli_mod

        with mock.patch.object(
            cli_mod, "detect_compose_cmd", return_value="docker compose"
        ), mock.patch.object(cli_mod, "run") as run_cmd:
            cli_mod.stop_infra(tmp_path)
        cmd = run_cmd.call_args.args[0]
        assert cmd[-1] == "down"
        assert "-v" not in cmd

    def test_stop_infra_purge_removes_volumes(self, tmp_path):
        from vllm_cluster_manager import cli as cli_mod

        with mock.patch.object(
            cli_mod, "detect_compose_cmd", return_value="docker compose"
        ), mock.patch.object(cli_mod, "run") as run_cmd:
            cli_mod.stop_infra(tmp_path, purge=True)
        cmd = run_cmd.call_args.args[0]
        assert cmd[-2:] == ["down", "-v"]

    @pytest.mark.parametrize("purge", [False, True])
    def test_run_host_down_threads_purge(self, tmp_path, purge):
        from vllm_cluster_manager import cli as cli_mod

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
        from vllm_cluster_manager import cli as cli_mod

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
        from vllm_cluster_manager import cli as cli_mod

        data_root = tmp_path / "share" / "vllm_cluster_manager"
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
    from vllm_cluster_manager import cli as cli_mod

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
    from vllm_cluster_manager import cli as cli_mod

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
    from vllm_cluster_manager import cli as cli_mod

    with pytest.raises(RuntimeError, match="Missing packaged assets"):
        cli_mod.copy_assets_subdir("host", "definitely-not-a-real-subdir", tmp_path / "d")
