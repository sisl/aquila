"""Tests for the CLI helper functions (cli.py)."""

import argparse
import hashlib
import os
import textwrap
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
    parse_nvcc_version,
    parse_smi_version,
    _vllm_wheel_url,
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
# CUDA version parsers
# ---------------------------------------------------------------------------


def test_parse_nvcc_version():
    output = textwrap.dedent("""\
        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2024 NVIDIA Corporation
        Cuda compilation tools, release 12.4, V12.4.131
    """)
    assert parse_nvcc_version(output) == "12.4"


def test_parse_nvcc_version_no_match():
    assert parse_nvcc_version("random output") is None


def test_parse_smi_version():
    output = textwrap.dedent("""\
        +-----------------------------------------------------------------------------------------+
        | NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
        +-----------------------------------------------------------------------------------------+
    """)
    assert parse_smi_version(output) == "12.4"


def test_parse_smi_version_no_match():
    assert parse_smi_version("no cuda here") is None


# ---------------------------------------------------------------------------
# _vllm_wheel_url
# ---------------------------------------------------------------------------


def test_vllm_wheel_url():
    url = _vllm_wheel_url("0.8.5", 124, "x86_64")
    assert "v0.8.5" in url
    assert "cu124" in url
    assert "x86_64" in url
    assert url.endswith(".whl")


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
