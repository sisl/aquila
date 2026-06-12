"""Tests for the client (satellite) application."""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add client app to sys.path.
_CLIENT_DIR = Path(__file__).resolve().parent.parent / "vllm_cluster_manager" / "assets" / "client"
if str(_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIENT_DIR))

# Stub out consul registration before importing the app (it runs on import via lifespan).
with mock.patch.dict("sys.modules", {"app.consul": mock.MagicMock()}):
    import app.main as client_main
    from app.main import (
        _resolve_image_tag,
        _derived_image_tag,
        StartRequest,
        StopRequest,
        app,
    )

from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
def _isolate_runtimes():
    """Keep unit tests off the machine's real container daemons.

    The runtime probe is reset around every test and Podman is unreachable by
    default (tests that exercise Podman paths patch `_podman` themselves,
    which takes precedence inside their own `with` blocks).
    """
    client_main._runtime_probe = (0.0, [])
    with mock.patch.object(
        client_main, "_podman", side_effect=RuntimeError("isolated in tests")
    ):
        yield
    client_main._runtime_probe = (0.0, [])


# ---------------------------------------------------------------------------
# _resolve_image_tag
# ---------------------------------------------------------------------------


class TestResolveImageTag:
    def test_release_version(self):
        image, resolved = _resolve_image_tag("0.8.5")
        assert image == "vllm/vllm-openai:v0.8.5"
        assert resolved == "0.8.5"

    def test_nightly(self):
        image, resolved = _resolve_image_tag("nightly")
        assert image == "vllm/vllm-openai:nightly"
        assert resolved == "nightly"

    def test_commit_hash(self):
        commit = "a" * 40
        image, resolved = _resolve_image_tag(commit)
        assert image == f"vllm/vllm-openai:nightly-{commit}"
        assert resolved == commit

    def test_blank_uses_latest(self):
        with mock.patch.object(client_main, "_get_latest_vllm_version", return_value="1.2.3"):
            image, resolved = _resolve_image_tag("")
        assert image == "vllm/vllm-openai:v1.2.3"
        assert resolved == "1.2.3"

    def test_none_uses_latest(self):
        with mock.patch.object(client_main, "_get_latest_vllm_version", return_value="1.2.3"):
            image, _ = _resolve_image_tag(None)
        assert image == "vllm/vllm-openai:v1.2.3"

    def test_blank_falls_back_to_latest_tag_when_unresolved(self):
        # GitHub unreachable -> _get_latest_vllm_version returns "" -> fall back
        # to the moving :latest tag instead of raising.
        with mock.patch.object(client_main, "_get_latest_vllm_version", return_value=""):
            image, resolved = _resolve_image_tag("")
        assert image == "vllm/vllm-openai:latest"
        assert resolved == "latest"


class TestEnsureImageMovingTags:
    def test_nightly_always_pulls_even_when_cached(self):
        fake = mock.MagicMock()  # images.get would succeed (image is cached)
        with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.object(
            client_main, "_pull_image"
        ) as pull:
            result = client_main._ensure_image("vllm/vllm-openai:nightly", None, lambda _m: None)
        assert result == "vllm/vllm-openai:nightly"
        pull.assert_called_once()

    def test_latest_always_pulls_even_when_cached(self):
        fake = mock.MagicMock()
        with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.object(
            client_main, "_pull_image"
        ) as pull:
            client_main._ensure_image("vllm/vllm-openai:latest", None, lambda _m: None)
        pull.assert_called_once()

    def test_immutable_release_uses_cache(self):
        fake = mock.MagicMock()  # images.get succeeds -> cached, no pull
        with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.object(
            client_main, "_pull_image"
        ) as pull:
            result = client_main._ensure_image("vllm/vllm-openai:v0.8.5", None, lambda _m: None)
        assert result == "vllm/vllm-openai:v0.8.5"
        pull.assert_not_called()


class TestDerivedImageTag:
    def test_order_independent(self):
        a = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["transformers", "numpy"])
        b = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["numpy", "transformers"])
        assert a == b

    def test_varies_by_packages(self):
        a = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["transformers"])
        b = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["numpy"])
        assert a != b

    def test_repo_prefix(self):
        tag = _derived_image_tag("vllm/vllm-openai:v0.8.5", ["transformers"])
        assert tag.startswith("vllm-cluster-manager/local:")


# ---------------------------------------------------------------------------
# StartRequest / StopRequest Pydantic models
# ---------------------------------------------------------------------------


class TestStartRequest:
    def test_minimal(self):
        r = StartRequest(model_name="llama", port=8000, gpu_memory_fraction=0.9)
        assert r.gpu_ids is None
        assert r.vllm_version is None
        assert r.extra_packages is None

    def test_full(self):
        r = StartRequest(
            model_name="llama",
            port=8000,
            gpu_memory_fraction=0.9,
            gpu_ids=[0, 1],
            tensor_parallel_size=2,
            extra_args=["--max-model-len", "4096"],
            env_vars=[{"key": "HF_TOKEN", "value": "tok"}],
            vllm_version="0.8.5",
            extra_packages=["transformers"],
        )
        assert r.gpu_ids == [0, 1]
        assert r.vllm_version == "0.8.5"


class TestStopRequest:
    def test_basic(self):
        r = StopRequest(key="llama:8000")
        assert r.key == "llama:8000"


# ---------------------------------------------------------------------------
# Container runtimes (docker / podman)
# ---------------------------------------------------------------------------


@pytest.fixture
def _reset_runtime_probe():
    client_main._runtime_probe = (0.0, [])
    yield
    client_main._runtime_probe = (0.0, [])


class TestAvailableRuntimes:
    def test_docker_only(self, _reset_runtime_probe):
        fake = mock.MagicMock()
        with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.object(
            client_main, "_podman", side_effect=RuntimeError("no socket")
        ):
            assert client_main._available_runtimes() == ["docker"]

    def test_both_runtimes(self, _reset_runtime_probe):
        fake = mock.MagicMock()
        with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.object(
            client_main, "_podman", return_value=fake
        ):
            assert client_main._available_runtimes() == ["docker", "podman"]

    def test_none_available(self, _reset_runtime_probe):
        with mock.patch.object(
            client_main, "_docker", side_effect=RuntimeError("down")
        ), mock.patch.object(client_main, "_podman", side_effect=RuntimeError("down")):
            assert client_main._available_runtimes() == []

    def test_probe_is_cached(self, _reset_runtime_probe):
        fake = mock.MagicMock()
        with mock.patch.object(
            client_main, "_docker", return_value=fake
        ) as docker_factory, mock.patch.object(
            client_main, "_podman", side_effect=RuntimeError("no socket")
        ):
            client_main._available_runtimes()
            client_main._available_runtimes()
        assert docker_factory.call_count == 1  # second call served from cache

    def test_runtime_client_dispatch(self):
        with mock.patch.object(client_main, "_docker", return_value="D"), mock.patch.object(
            client_main, "_podman", return_value="P"
        ):
            assert client_main._runtime_client("docker") == "D"
            assert client_main._runtime_client("podman") == "P"
        with pytest.raises(ValueError):
            client_main._runtime_client("lxc")

    def test_podman_socket_candidates_prefer_env(self):
        with mock.patch.dict(
            os.environ, {"PODMAN_SOCK": "/custom/podman.sock"}, clear=False
        ):
            candidates = client_main._podman_socket_candidates()
        assert candidates[0] == "/custom/podman.sock"
        assert "/run/podman/podman.sock" in candidates

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores dir permissions")
    def test_socket_exists_tolerates_unreadable_dir(self, tmp_path):
        # On machines without Podman, /run/podman can be a root-only dir;
        # Path.exists() raises PermissionError there instead of returning
        # False, which used to crash every endpoint that probes runtimes.
        locked = tmp_path / "locked"
        locked.mkdir()
        sock = locked / "podman.sock"
        locked.chmod(0)
        try:
            assert client_main._socket_exists(str(sock)) is False
        finally:
            locked.chmod(0o755)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores dir permissions")
    def test_probe_survives_unreadable_socket_dir(self, _reset_runtime_probe, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        sock = str(locked / "podman.sock")
        locked.chmod(0)
        try:
            with mock.patch.object(
                client_main, "_docker", side_effect=RuntimeError("not installed")
            ), mock.patch.object(
                client_main, "_podman", side_effect=RuntimeError("not installed")
            ), mock.patch.object(
                client_main, "_podman_socket_candidates", return_value=[sock]
            ):
                assert client_main._available_runtimes() == []
        finally:
            locked.chmod(0o755)


class TestStartRuntimeSelection:
    @pytest.mark.anyio
    async def test_unavailable_runtime_rejected(self):
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker"]
        ), mock.patch.dict(client_main._statuses, {}, clear=True), mock.patch.dict(
            client_main._containers, {}, clear=True
        ), mock.patch.dict(client_main._logs, {}, clear=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/start",
                    json={
                        "model_name": "org/model",
                        "port": 38475,
                        "gpu_memory_fraction": 0.5,
                        "vllm_version": "0.9.1",
                        "skip_resource_check": True,
                        "container_runtime": "podman",
                    },
                )
        assert resp.status_code == 409
        assert "podman" in resp.json()["detail"]
        assert "docker" in resp.json()["detail"]

    @pytest.mark.anyio
    async def test_no_runtime_at_all_rejected(self):
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=[]
        ), mock.patch.dict(client_main._statuses, {}, clear=True), mock.patch.dict(
            client_main._containers, {}, clear=True
        ), mock.patch.dict(client_main._logs, {}, clear=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/start",
                    json={
                        "model_name": "org/model",
                        "port": 38476,
                        "gpu_memory_fraction": 0.5,
                        "vllm_version": "0.9.1",
                        "skip_resource_check": True,
                    },
                )
        assert resp.status_code == 409
        assert "none" in resp.json()["detail"]


class TestPodmanGpuPassthrough:
    def test_device_requests_docker_unchanged(self):
        reqs = client_main._device_requests(None)
        assert reqs[0]["Count"] == -1
        assert reqs[0]["Capabilities"] == [["gpu"]]
        reqs = client_main._device_requests([0, 2], "docker")
        assert reqs[0]["DeviceIDs"] == ["0", "2"]

    def test_device_requests_podman_uses_cdi(self):
        # Podman's compat API ignores capability-based requests; only
        # driver="cdi" with qualified device names reaches the container.
        reqs = client_main._device_requests(None, "podman")
        assert reqs[0]["Driver"] == "cdi"
        assert reqs[0]["DeviceIDs"] == ["nvidia.com/gpu=all"]
        reqs = client_main._device_requests([1, 3], "podman")
        assert reqs[0]["DeviceIDs"] == ["nvidia.com/gpu=1", "nvidia.com/gpu=3"]

    def test_podman_server_version_parsing(self):
        fake = mock.MagicMock()
        fake.version.return_value = {"Version": "5.8.2"}
        with mock.patch.object(client_main, "_podman", return_value=fake):
            assert client_main._podman_server_version() == (5, 8, 2)
        fake.version.return_value = {
            "Version": "",
            "Components": [{"Name": "Podman Engine", "Version": "4.9.4-rhel"}],
        }
        with mock.patch.object(client_main, "_podman", return_value=fake):
            assert client_main._podman_server_version() == (4, 9, 4)
        fake.version.side_effect = RuntimeError("socket gone")
        with mock.patch.object(client_main, "_podman", return_value=fake):
            assert client_main._podman_server_version() is None

    def test_cdi_spec_detection(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with mock.patch.object(client_main, "_CDI_SPEC_DIRS", (str(empty),)):
            assert client_main._nvidia_cdi_specs_present() is False
        (empty / "nvidia.yaml").write_text("devices:\n- name: all\nkind: nvidia.com/gpu\n")
        with mock.patch.object(client_main, "_CDI_SPEC_DIRS", (str(empty),)):
            assert client_main._nvidia_cdi_specs_present() is True
        with mock.patch.object(
            client_main, "_CDI_SPEC_DIRS", (str(tmp_path / "missing"),)
        ):
            assert client_main._nvidia_cdi_specs_present() is False

    def test_gpu_error_old_podman(self):
        with mock.patch.object(
            client_main, "_podman_server_version", return_value=(4, 9, 4)
        ):
            reason = client_main._podman_gpu_error()
        assert reason is not None and "5.4" in reason

    def test_gpu_error_missing_cdi_spec(self):
        with mock.patch.object(
            client_main, "_podman_server_version", return_value=(5, 8, 2)
        ), mock.patch.object(
            client_main, "_nvidia_cdi_specs_present", return_value=False
        ):
            reason = client_main._podman_gpu_error()
        assert reason is not None and "nvidia-ctk cdi generate" in reason

    def test_gpu_error_all_good(self):
        with mock.patch.object(
            client_main, "_podman_server_version", return_value=(5, 8, 2)
        ), mock.patch.object(
            client_main, "_nvidia_cdi_specs_present", return_value=True
        ):
            assert client_main._podman_gpu_error() is None

    def test_gpu_error_unknown_version_not_blocking(self):
        # An unparsable version must not block deployments on its own.
        with mock.patch.object(
            client_main, "_podman_server_version", return_value=None
        ), mock.patch.object(
            client_main, "_nvidia_cdi_specs_present", return_value=True
        ):
            assert client_main._podman_gpu_error() is None

    @pytest.mark.anyio
    async def test_start_rejected_when_podman_cannot_pass_gpus(self):
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["podman"]
        ), mock.patch.object(
            client_main,
            "_podman_gpu_error",
            return_value="No NVIDIA CDI spec found in /etc/cdi",
        ), mock.patch.dict(client_main._statuses, {}, clear=True), mock.patch.dict(
            client_main._containers, {}, clear=True
        ), mock.patch.dict(client_main._logs, {}, clear=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/start",
                    json={
                        "model_name": "org/model",
                        "port": 38477,
                        "gpu_memory_fraction": 0.5,
                        "vllm_version": "0.9.1",
                        "skip_resource_check": True,
                        "container_runtime": "podman",
                    },
                )
        assert resp.status_code == 409
        assert "CDI" in resp.json()["detail"]


class TestCrossRuntimeEnumeration:
    def test_reconcile_unions_runtimes(self, logs_dir):
        docker_container = mock.MagicMock()
        docker_container.labels = {
            client_main._LABEL_KEY: "m1:8001",
            client_main._LABEL_PORT: "8001",
            client_main._LABEL_RUNTIME: "docker",
        }
        docker_container.status = "running"
        docker_container.id = "c1"
        docker_container.image.tags = []
        podman_container = mock.MagicMock()
        podman_container.labels = {
            client_main._LABEL_KEY: "m2:8002",
            client_main._LABEL_PORT: "8002",
            client_main._LABEL_RUNTIME: "podman",
        }
        podman_container.status = "running"
        podman_container.id = "c2"
        podman_container.image.tags = []

        docker_client = mock.MagicMock()
        docker_client.containers.list.return_value = [docker_container]
        podman_client = mock.MagicMock()
        podman_client.containers.list.return_value = [podman_container]

        def fake_runtime_client(runtime):
            return docker_client if runtime == "docker" else podman_client

        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(
            client_main, "_runtime_client", side_effect=fake_runtime_client
        ), mock.patch.object(
            client_main, "_image_digest", return_value="sha256:x"
        ), mock.patch.object(
            client_main, "_stream_container_logs", mock.MagicMock()
        ), mock.patch.object(
            client_main, "_monitor_container", mock.MagicMock()
        ), mock.patch("asyncio.create_task"), mock.patch.dict(
            client_main._statuses, {}, clear=True
        ), mock.patch.dict(client_main._containers, {}, clear=True), mock.patch.dict(
            client_main._logs, {}, clear=True
        ):
            client_main._reconcile_containers()
            statuses = dict(client_main._statuses)

        assert statuses["m1:8001"]["container_runtime"] == "docker"
        assert statuses["m2:8002"]["container_runtime"] == "podman"

    @pytest.mark.anyio
    async def test_images_tagged_with_runtime(self):
        image = mock.MagicMock()
        image.tags = ["vllm/vllm-openai:v0.9.1"]
        image.short_id = "sha256:abc"
        image.attrs = {"Size": 1024**3}
        docker_client = mock.MagicMock()
        docker_client.images.list.return_value = [image]

        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker"]
        ), mock.patch.object(client_main, "_runtime_client", return_value=docker_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/images")
        (entry,) = resp.json()["images"]
        assert entry["runtime"] == "docker"

    @pytest.mark.anyio
    async def test_metrics_reports_available_runtimes(self):
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_gpu_metrics", return_value=[]), mock.patch.object(
            client_main, "_disk_metrics", return_value=None
        ), mock.patch.object(
            client_main.psutil, "cpu_percent", return_value=1.0
        ), mock.patch.object(
            client_main.psutil, "virtual_memory", return_value=_fake_vmem()
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/metrics")
        assert resp.json()["available_runtimes"] == ["docker", "podman"]


# ---------------------------------------------------------------------------
# Launch manifest (container label for host re-adoption)
# ---------------------------------------------------------------------------


class TestLaunchManifest:
    def test_contents(self):
        payload = StartRequest(
            model_name="org/model",
            port=8001,
            gpu_memory_fraction=0.5,
            gpu_ids=[0, 1],
            tensor_parallel_size=2,
            extra_args=["--seed", "7"],
            env_vars=[{"key": "HF_TOKEN", "value": "tok"}],
            engine_args={"max_model_len": 4096},
            lora_modules=[{"name": "ad", "path": "org/adapter"}],
            extra_packages=["transformers"],
            max_failed_restarts=5,
            owner="alice",
            duration_seconds=3600,
            expires_at="2026-06-11T12:00:00+00:00",
        )
        manifest = client_main._launch_manifest(payload)
        assert manifest["version"] == 1
        assert manifest["owner"] == "alice"
        assert manifest["duration_seconds"] == 3600
        assert manifest["expires_at"] == "2026-06-11T12:00:00+00:00"
        assert manifest["extra_args"] == ["--seed", "7"]
        # env vars travel verbatim — they are already on the container's Env.
        assert manifest["env_vars"] == [{"key": "HF_TOKEN", "value": "tok"}]
        assert manifest["engine_args"] == {"max_model_len": 4096}
        assert manifest["lora_modules"] == [{"name": "ad", "path": "org/adapter"}]
        assert manifest["extra_packages"] == ["transformers"]
        assert manifest["gpu_memory_fraction"] == 0.5
        assert manifest["gpu_ids"] == [0, 1]
        assert manifest["tensor_parallel_size"] == 2
        assert manifest["max_failed_restarts"] == 5

    def test_old_host_payload_still_validates(self):
        # An old host that doesn't send the metadata fields must keep working.
        r = StartRequest(model_name="llama", port=8000, gpu_memory_fraction=0.9)
        manifest = client_main._launch_manifest(r)
        assert manifest["owner"] is None
        assert manifest["duration_seconds"] is None
        assert manifest["extra_args"] == []


def _fake_managed_container(labels: dict, status: str = "running"):
    c = mock.MagicMock()
    c.labels = labels
    c.status = status
    c.id = "cid123"
    c.image.tags = ["vllm/vllm-openai:v0.9.1"]
    return c


class TestReconcileContainers:
    def _reconcile(self, container):
        fake = mock.MagicMock()
        fake.containers.list.return_value = [container]
        with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.object(
            client_main, "_image_digest", return_value="sha256:dgst"
        ), mock.patch.object(
            client_main, "_stream_container_logs", mock.MagicMock()
        ), mock.patch.object(
            client_main, "_monitor_container", mock.MagicMock()
        ), mock.patch(
            "asyncio.create_task"
        ), mock.patch.dict(
            client_main._statuses, {}, clear=True
        ), mock.patch.dict(
            client_main._containers, {}, clear=True
        ), mock.patch.dict(
            client_main._logs, {}, clear=True
        ):
            client_main._reconcile_containers()
            return dict(client_main._statuses)

    def test_restores_manifest(self):
        import json as json_mod

        manifest = {
            "version": 1,
            "owner": "alice",
            "duration_seconds": 3600,
            "expires_at": "2026-06-11T12:00:00+00:00",
            "extra_args": ["--seed", "7"],
            "env_vars": [],
            "engine_args": {"max_model_len": 4096},
            "lora_modules": [{"name": "ad", "path": "p"}],
            "extra_packages": [],
            "gpu_memory_fraction": 0.5,
            "gpu_ids": [0],
            "tensor_parallel_size": 2,
            "max_failed_restarts": 5,
        }
        container = _fake_managed_container(
            {
                client_main._LABEL_KEY: "org/model:8001",
                client_main._LABEL_PORT: "8001",
                client_main._LABEL_VERSION: "0.9.1",
                client_main._LABEL_LAUNCH: json_mod.dumps(manifest),
            }
        )
        statuses = self._reconcile(container)
        status = statuses["org/model:8001"]
        assert status["launch_manifest"] == manifest
        assert status["gpu_memory_fraction"] == 0.5
        assert status["gpu_ids"] == [0]
        assert status["tensor_parallel_size"] == 2
        assert status["engine_args"] == {"max_model_len": 4096}
        assert status["lora_modules"] == [{"name": "ad", "path": "p"}]
        assert status["max_failed_restarts"] == 5

    def test_tolerates_corrupt_manifest(self):
        container = _fake_managed_container(
            {
                client_main._LABEL_KEY: "org/model:8001",
                client_main._LABEL_PORT: "8001",
                client_main._LABEL_LAUNCH: "{not json",
            }
        )
        statuses = self._reconcile(container)
        status = statuses["org/model:8001"]
        assert "launch_manifest" not in status
        assert status["port"] == 8001

    def test_no_manifest_label_keeps_minimal_status(self):
        container = _fake_managed_container(
            {
                client_main._LABEL_KEY: "org/model:8001",
                client_main._LABEL_PORT: "8001",
            }
        )
        statuses = self._reconcile(container)
        assert "launch_manifest" not in statuses["org/model:8001"]


# ---------------------------------------------------------------------------
# Managed local models (/local-models endpoints)
# ---------------------------------------------------------------------------


@pytest.fixture
def managed_models(tmp_path):
    models = tmp_path / ".models"
    with mock.patch.object(client_main, "_MODELS_DIR", models), mock.patch.object(
        client_main, "_MODELS_TMP", models / ".tmp"
    ), mock.patch.dict(client_main._upload_sessions, {}, clear=True), mock.patch.dict(
        client_main._transfers, {}, clear=True
    ):
        yield models


class TestLocalModelUpload:
    @pytest.mark.anyio
    async def test_full_lifecycle(self, managed_models):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/local-models/upload/begin",
                json={"name": "my-ft", "total_bytes": 8, "file_count": 2},
            )
            assert resp.status_code == 200
            sid = resp.json()["session_id"]

            resp = await client.put(
                f"/local-models/upload/{sid}/file",
                params={"path": "config.json"},
                content=b"{}",
            )
            assert resp.status_code == 200
            resp = await client.put(
                f"/local-models/upload/{sid}/file",
                params={"path": "sub/weights.safetensors"},
                content=b"weight",
            )
            assert resp.status_code == 200
            assert resp.json()["received_bytes"] == 8

            resp = await client.post(f"/local-models/upload/{sid}/finish", json={})
            assert resp.status_code == 200
            body = resp.json()
            assert body["name"] == "my-ft"
            assert body["warnings"] == []

        target = managed_models / "my-ft"
        assert (target / "config.json").read_bytes() == b"{}"
        assert (target / "sub" / "weights.safetensors").read_bytes() == b"weight"
        assert sid not in client_main._upload_sessions

    @pytest.mark.anyio
    async def test_begin_conflict_and_bad_name(self, managed_models):
        (managed_models / "taken").mkdir(parents=True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/local-models/upload/begin", json={"name": "taken"}
            )
            assert resp.status_code == 409
            resp = await client.post(
                "/local-models/upload/begin", json={"name": "../escape"}
            )
            assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_begin_insufficient_disk(self, managed_models):
        usage = mock.MagicMock(free=10)
        with mock.patch.object(client_main.shutil, "disk_usage", return_value=usage):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/local-models/upload/begin",
                    json={"name": "big", "total_bytes": 10**12},
                )
        assert resp.status_code == 507

    @pytest.mark.anyio
    async def test_file_put_unknown_session_and_traversal(self, managed_models):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/local-models/upload/nope/file", params={"path": "a"}, content=b"x"
            )
            assert resp.status_code == 404

            begin = await client.post(
                "/local-models/upload/begin", json={"name": "m"}
            )
            sid = begin.json()["session_id"]
            resp = await client.put(
                f"/local-models/upload/{sid}/file",
                params={"path": "../escape"},
                content=b"x",
            )
            assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_abort_removes_staging(self, managed_models):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            begin = await client.post("/local-models/upload/begin", json={"name": "m"})
            sid = begin.json()["session_id"]
            staging = Path(str(client_main._upload_sessions[sid]["staging"]))
            assert staging.exists()
            resp = await client.post(f"/local-models/upload/{sid}/abort")
            assert resp.status_code == 200
        assert not staging.exists()
        assert sid not in client_main._upload_sessions


def _targz_bytes(entries: dict[str, bytes], root: str = "ckpt") -> bytes:
    import io
    import tarfile as tarfile_mod

    buf = io.BytesIO()
    with tarfile_mod.open(fileobj=buf, mode="w:gz") as tar:
        for rel, data in entries.items():
            info = tarfile_mod.TarInfo(f"{root}/{rel}" if root else rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestLocalModelArchive:
    @pytest.mark.anyio
    async def test_archive_extracts_and_flattens(self, managed_models):
        body = _targz_bytes({"config.json": b"{}", "w.safetensors": b"w"})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/local-models/archive",
                params={"name": "arch", "filename": "ckpt.tar.gz"},
                content=body,
            )
        assert resp.status_code == 200
        target = managed_models / "arch"
        # Single-root archive is flattened: config.json at the model root.
        assert (target / "config.json").exists()
        assert resp.json()["warnings"] == []

    @pytest.mark.anyio
    async def test_archive_bad_extension(self, managed_models):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/local-models/archive",
                params={"name": "x", "filename": "model.rar"},
                content=b"x",
            )
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_archive_traversal_member_cleaned_up(self, managed_models):
        body = _targz_bytes({"evil": b"x"}, root="..")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/local-models/archive",
                params={"name": "evil", "filename": "e.tar.gz"},
                content=body,
            )
        assert resp.status_code == 400
        assert not (managed_models / "evil").exists()
        # Staging and the temp archive are cleaned up.
        tmp = managed_models / ".tmp"
        assert not tmp.exists() or list(tmp.iterdir()) == []


class TestLocalModelListAndDelete:
    @pytest.mark.anyio
    async def test_list_includes_managed_and_external(self, managed_models, tmp_path):
        (managed_models / "mine").mkdir(parents=True)
        (managed_models / "mine" / "config.json").write_text("{}")
        external = tmp_path / "shared"
        (external / "other-ckpt").mkdir(parents=True)
        with mock.patch.object(client_main.settings, "model_dirs", str(external)):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/local-models")
        assert resp.status_code == 200
        models = {m["name"]: m for m in resp.json()["models"]}
        assert models["mine"]["source"] == "managed"
        assert models["mine"]["deletable"] is True
        assert models["other-ckpt"]["deletable"] is False

    @pytest.mark.anyio
    async def test_single_gguf_path_points_at_file(self, managed_models):
        model = managed_models / "tiny"
        model.mkdir(parents=True)
        (model / "tiny.gguf").write_bytes(b"g")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/local-models")
        (entry,) = resp.json()["models"]
        assert entry["path"].endswith("tiny/tiny.gguf")

    @pytest.mark.anyio
    async def test_delete_ok_404_and_in_use(self, managed_models):
        model = managed_models / "served"
        model.mkdir(parents=True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with mock.patch.dict(
                client_main._statuses,
                {"k:8000": {"model_name": str(model)}},
                clear=True,
            ), mock.patch.dict(client_main._containers, {"k:8000": object()}, clear=True):
                resp = await client.delete("/local-models/served")
                assert resp.status_code == 409

            resp = await client.delete("/local-models/served")
            assert resp.status_code == 200
            assert not model.exists()

            resp = await client.delete("/local-models/served")
            assert resp.status_code == 404


class TestLocalModelPull:
    @pytest.mark.anyio
    async def test_pull_single_file_and_status(self, managed_models):
        chunks = [b"abc", b"def"]

        class _FakeStreamResponse:
            status_code = 200
            headers = {"content-length": "6"}

            async def aiter_bytes(self):
                for chunk in chunks:
                    yield chunk

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                return _FakeStreamResponse()

        with mock.patch.object(client_main.httpx, "AsyncClient", _FakeAsyncClient):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/local-models/pull",
                    json={"url": "http://example.com/w.safetensors", "name": "pulled"},
                )
                assert resp.status_code == 200
                # The pull task runs on the same loop; poll until terminal.
                for _ in range(50):
                    status_resp = await client.get("/local-models/transfers")
                    (transfer,) = status_resp.json()["transfers"]
                    if transfer["status"] in ("done", "error"):
                        break
                    await asyncio.sleep(0.01)
        assert transfer["status"] == "done"
        assert (managed_models / "pulled" / "w.safetensors").read_bytes() == b"abcdef"

    @pytest.mark.anyio
    async def test_pull_http_error_reports_and_cleans(self, managed_models):
        class _FakeStreamResponse:
            status_code = 404
            headers = {}

            async def aiter_bytes(self):
                if False:  # pragma: no cover
                    yield b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                return _FakeStreamResponse()

        with mock.patch.object(client_main.httpx, "AsyncClient", _FakeAsyncClient):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/local-models/pull",
                    json={"url": "http://example.com/gone.tar.gz"},
                )
                assert resp.status_code == 200
                assert resp.json()["name"] == "gone"
                for _ in range(50):
                    status_resp = await client.get("/local-models/transfers")
                    (transfer,) = status_resp.json()["transfers"]
                    if transfer["status"] in ("done", "error"):
                        break
                    await asyncio.sleep(0.01)
        assert transfer["status"] == "error"
        assert "404" in transfer["error"]
        assert not (managed_models / "gone").exists()

    @pytest.mark.anyio
    async def test_pull_rejects_non_http(self, managed_models):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/local-models/pull", json={"url": "ftp://example.com/x"}
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GPU metrics (unified-memory devices)
# ---------------------------------------------------------------------------


def _fake_vmem(used_mb: int = 51200, total_mb: int = 131072):
    return mock.MagicMock(
        used=used_mb * 1024 * 1024, total=total_mb * 1024 * 1024, percent=39.1
    )


class TestGpuMetricsUnifiedMemory:
    def test_smi_int(self):
        assert client_main._smi_int("42") == 42
        assert client_main._smi_int("42.7") == 42
        assert client_main._smi_int("[N/A]") is None

    def test_parse_keeps_utilization_when_memory_na(self):
        output = "0, NVIDIA GB10, 37, [N/A], [N/A]\n"
        (gpu,) = client_main._parse_nvidia_smi_gpus(output)
        assert gpu["utilization"] == 37
        assert gpu["memory_total_mb"] is None

    def test_parse_normal_row_and_junk_lines(self):
        output = "garbage line\n0, NVIDIA H100, 64, 73728, 81920\n"
        (gpu,) = client_main._parse_nvidia_smi_gpus(output)
        assert gpu == {
            "index": 0,
            "name": "NVIDIA H100",
            "source": "nvidia-smi",
            "utilization": 64,
            "memory_used_mb": 73728,
            "memory_total_mb": 81920,
        }

    def test_substitute_fills_ram_keeps_compute(self):
        gpus = [
            {"index": 0, "utilization": 37, "memory_used_mb": None, "memory_total_mb": None},
            {"index": 1, "utilization": 64, "memory_used_mb": 100, "memory_total_mb": 200},
        ]
        with mock.patch.object(client_main.psutil, "virtual_memory", return_value=_fake_vmem()):
            result = client_main._substitute_unified_memory(gpus)
        assert result[0]["utilization"] == 37  # real compute preserved
        assert result[0]["memory_total_mb"] == 131072  # RAM substituted
        assert result[0]["source"] == "unified"
        assert result[1]["memory_total_mb"] == 200  # dedicated VRAM untouched
        assert "source" not in result[1]

    def test_gpu_metrics_unified_device_via_nvidia_smi(self):
        # DGX Spark-style device: real utilization.gpu, [N/A] VRAM fields.
        broken_nvml = mock.MagicMock()
        broken_nvml.nvmlInit.side_effect = RuntimeError("no NVML")
        with mock.patch.dict("sys.modules", {"pynvml": broken_nvml}), mock.patch.object(
            client_main.shutil, "which", return_value="/usr/bin/nvidia-smi"
        ), mock.patch.object(
            client_main.subprocess,
            "check_output",
            return_value="0, NVIDIA GB10, 37, [N/A], [N/A]\n",
        ), mock.patch.object(
            client_main.psutil, "virtual_memory", return_value=_fake_vmem()
        ):
            (gpu,) = client_main._gpu_metrics()
        assert gpu["utilization"] == 37  # compute, NOT the memory percentage
        assert gpu["memory_used_mb"] == 51200
        assert gpu["memory_total_mb"] == 131072
        assert gpu["source"] == "unified"

    def test_gpu_metrics_no_tooling_reports_unknown_compute(self):
        broken_nvml = mock.MagicMock()
        broken_nvml.nvmlInit.side_effect = RuntimeError("no NVML")
        with mock.patch.dict("sys.modules", {"pynvml": broken_nvml}), mock.patch.object(
            client_main.shutil, "which", return_value=None
        ), mock.patch.object(
            client_main.psutil, "virtual_memory", return_value=_fake_vmem()
        ):
            (gpu,) = client_main._gpu_metrics()
        # Without any GPU tooling, compute is unknown — never the RAM percent.
        assert gpu["utilization"] is None
        assert gpu["memory_total_mb"] == 131072


# ---------------------------------------------------------------------------
# Image pull progress
# ---------------------------------------------------------------------------


class TestPullTracker:
    def test_accumulates_per_layer(self):
        tracker = client_main._PullTracker()
        tracker.update({"id": "a", "status": "Downloading", "progressDetail": {"current": 10, "total": 100}})
        downloaded, total = tracker.update(
            {"id": "b", "status": "Downloading", "progressDetail": {"current": 5, "total": 50}}
        )
        assert (downloaded, total) == (15, 150)
        downloaded, total = tracker.update(
            {"id": "a", "status": "Downloading", "progressDetail": {"current": 60, "total": 100}}
        )
        assert (downloaded, total) == (65, 150)

    def test_download_complete_snaps_to_total(self):
        tracker = client_main._PullTracker()
        tracker.update({"id": "a", "status": "Downloading", "progressDetail": {"current": 10, "total": 100}})
        downloaded, total = tracker.update({"id": "a", "status": "Download complete"})
        assert (downloaded, total) == (100, 100)

    def test_ignores_chunks_without_progress(self):
        tracker = client_main._PullTracker()
        assert tracker.update({"status": "Pulling from vllm/vllm-openai"}) == (0, 0)
        assert tracker.update({"id": "a", "status": "Already exists"}) == (0, 0)
        assert tracker.update({"id": "a", "status": "Pulling fs layer"}) == (0, 0)


class TestPullImageProgress:
    def _fake_client(self, chunks):
        fake = mock.MagicMock()
        fake.api.pull.return_value = iter(chunks)
        return fake

    def test_progress_cb_receives_aggregates(self):
        chunks = [
            {"id": "a", "status": "Pulling fs layer"},
            {"id": "a", "status": "Downloading", "progressDetail": {"current": 10, "total": 100}},
            {"id": "b", "status": "Downloading", "progressDetail": {"current": 20, "total": 100}},
            {"id": "a", "status": "Download complete"},
            {"id": "b", "status": "Download complete"},
        ]
        reports: list[tuple[int, int]] = []
        client_main._pull_image(
            self._fake_client(chunks),
            "vllm/vllm-openai:v0.9.1",
            lambda _m: None,
            lambda d, t: reports.append((d, t)),
        )
        assert reports[-1] == (200, 200)
        assert all(reports[i][0] <= reports[i + 1][0] for i in range(len(reports) - 1))

    def test_without_progress_cb_still_works(self):
        logged: list[str] = []
        client_main._pull_image(
            self._fake_client([{"id": "a", "status": "Pull complete"}]),
            "vllm/vllm-openai:v0.9.1",
            logged.append,
        )
        assert logged[0].startswith("[docker] Pulling")
        assert logged[-1].startswith("[docker] Pulled")


class TestStartProvisionalStatus:
    @pytest.mark.anyio
    async def test_pull_progress_visible_then_replaced(self, logs_dir):
        key = "org/model:38473"
        seen_during_pull: dict[str, object] = {}

        def fake_ensure(image_ref, extra_packages, log, progress_cb=None, runtime="docker"):
            assert client_main._statuses[key]["status"] == "starting"
            assert runtime == "docker"
            progress_cb(50 * 1024 * 1024, 100 * 1024 * 1024)
            seen_during_pull.update(client_main._statuses[key])
            return image_ref

        container = mock.MagicMock()
        container.id = "cid"
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker"]
        ), mock.patch.object(client_main, "_ensure_image", fake_ensure), mock.patch.object(
            client_main, "_run_container", return_value=container
        ), mock.patch.object(client_main, "_image_digest", return_value="sha256:d"), mock.patch.object(
            client_main, "_stream_container_logs", mock.MagicMock()
        ), mock.patch.object(
            client_main, "_monitor_container", mock.MagicMock()
        ), mock.patch(
            "asyncio.create_task"
        ), mock.patch.dict(
            client_main._statuses, {}, clear=True
        ), mock.patch.dict(client_main._containers, {}, clear=True), mock.patch.dict(
            client_main._logs, {}, clear=True
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/start",
                    json={
                        "model_name": "org/model",
                        "port": 38473,
                        "gpu_memory_fraction": 0.5,
                        "vllm_version": "0.9.1",
                        "skip_resource_check": True,
                    },
                )
            assert resp.status_code == 200
            assert seen_during_pull["phase"] == "pulling image"
            assert seen_during_pull["pull_progress"] == {
                "downloaded_mb": 50,
                "total_mb": 100,
                "percent": 50.0,
            }
            # After start, the provisional entry is replaced: no progress left.
            final = client_main._statuses[key]
            assert final["status"] == "loading"
            assert "pull_progress" not in final

    @pytest.mark.anyio
    async def test_failed_pull_leaves_no_ghost_status(self, logs_dir):
        def fake_ensure(image_ref, extra_packages, log, progress_cb=None, runtime="docker"):
            raise RuntimeError("registry unreachable")

        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker"]
        ), mock.patch.object(client_main, "_ensure_image", fake_ensure), mock.patch.dict(
            client_main._statuses, {}, clear=True
        ), mock.patch.dict(client_main._containers, {}, clear=True), mock.patch.dict(
            client_main._logs, {}, clear=True
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/start",
                    json={
                        "model_name": "org/model",
                        "port": 38474,
                        "gpu_memory_fraction": 0.5,
                        "vllm_version": "0.9.1",
                        "skip_resource_check": True,
                    },
                )
            assert resp.status_code == 500
            assert "org/model:38474" not in client_main._statuses


# ---------------------------------------------------------------------------
# Persistent deployment logs
# ---------------------------------------------------------------------------


@pytest.fixture
def logs_dir(tmp_path):
    d = tmp_path / ".logs"
    with mock.patch.object(client_main, "_LOGS_DIR", d), mock.patch.dict(
        client_main._log_state, {}, clear=True
    ), mock.patch.dict(client_main._logs, {}, clear=True):
        yield d
        for key in list(client_main._log_state):
            client_main._close_log_file(key)


class TestLogLineHelpers:
    def test_split_docker_ts(self):
        ts, rest = client_main._split_docker_ts(
            "2026-06-11T08:00:32.123456789Z INFO hello"
        )
        assert ts == "2026-06-11T08:00:32.123456789Z"
        assert rest == "INFO hello"

    def test_split_docker_ts_absent(self):
        ts, rest = client_main._split_docker_ts("no timestamp here")
        assert ts is None
        assert rest == "no timestamp here"

    def test_format_log_line(self):
        line = client_main._format_log_line(
            "2026-06-11T08:00:32.123456789Z", "INFO hello"
        )
        assert line == "[2026-06-11 08:00:32] INFO hello"

    def test_format_log_line_without_ts(self):
        assert client_main._format_log_line(None, "raw") == "raw"

    @pytest.mark.parametrize(
        "line",
        [
            '(APIServer pid=1) INFO:     127.0.0.1:53710 - "GET /metrics HTTP/1.1" 200 OK',
            'INFO:     10.0.0.5:1234 - "GET /health HTTP/1.1" 200 OK',
            'INFO:     10.0.0.5:1234 - "HEAD /health HTTP/1.1" 200 OK',
        ],
    )
    def test_noise_lines_dropped(self, line):
        assert client_main._is_noise_line(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            'INFO:     10.0.0.21:51424 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
            "INFO 06-11 [metrics.py:417] Avg prompt throughput: 1843.2 tokens/s",
            "torch.OutOfMemoryError: CUDA out of memory.",
            "[docker] Started container vllm-cluster-x",
            "INFO: Application startup complete.",
        ],
    )
    def test_relevant_lines_kept(self, line):
        assert client_main._is_noise_line(line) is False


class TestAppendLogLine:
    def test_writes_deque_and_file_and_sidecar(self, logs_dir):
        key = "org/model:8001"
        client_main._append_log_line(
            key, "[2026-06-11 08:00:32] hello", "2026-06-11T08:00:32.000000001Z"
        )
        client_main._append_log_line(key, "[2026-06-11 08:00:33] world")
        path = client_main._log_file_for(key)
        client_main._close_log_file(key)
        assert list(client_main._logs[key]) == [
            "[2026-06-11 08:00:32] hello",
            "[2026-06-11 08:00:33] world",
        ]
        assert path.read_text().splitlines() == [
            "[2026-06-11 08:00:32] hello",
            "[2026-06-11 08:00:33] world",
        ]
        sidecar = client_main._log_sidecar_for(path)
        assert sidecar.read_text() == "2026-06-11T08:00:32.000000001Z"

    def test_rotation_at_size_cap(self, logs_dir):
        key = "m:1"
        big_line = "first " + "x" * (1024 * 1024 + 100)  # exceeds the 1 MB cap
        with mock.patch.object(client_main.settings, "log_max_mb", 1):
            client_main._append_log_line(key, big_line)
            client_main._append_log_line(key, "second line")
        path = client_main._log_file_for(key)
        client_main._close_log_file(key)
        rotated = Path(str(path) + ".1")
        assert rotated.read_text().startswith("first ")
        assert path.read_text().splitlines() == ["second line"]

    def test_rotate_on_fresh_run(self, logs_dir):
        key = "m:1"
        client_main._append_log_line(key, "old run")
        client_main._open_log_file(key, rotate=True)
        client_main._append_log_line(key, "new run")
        path = client_main._log_file_for(key)
        client_main._close_log_file(key)
        assert path.read_text().splitlines() == ["new run"]
        assert "old run" in Path(str(path) + ".1").read_text()


class TestGcOldLogs:
    def test_removes_only_old_files(self, logs_dir):
        logs_dir.mkdir(parents=True)
        old = logs_dir / "ancient-8000.log"
        old.write_text("x")
        os.utime(old, (0, 0))
        fresh = logs_dir / "fresh-8001.log"
        fresh.write_text("y")
        client_main._gc_old_logs()
        assert not old.exists()
        assert fresh.exists()


@pytest.mark.anyio
async def test_stream_resume_skips_persisted_lines(logs_dir):
    key = "org/model:8001"
    path = client_main._log_file_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[2026-06-11 08:00:30] already persisted\n")
    client_main._log_sidecar_for(path).write_text("2026-06-11T08:00:30.000000005Z")

    container = mock.MagicMock()
    container.logs.return_value = [
        b"2026-06-11T08:00:30.000000005Z duplicate line\n",
        b"2026-06-11T08:00:31.000000001Z fresh engine line\n",
        b'2026-06-11T08:00:32.000000001Z INFO:     127.0.0.1:1 - "GET /metrics HTTP/1.1" 200 OK\n',
    ]
    with mock.patch.dict(client_main._statuses, {key: {}}, clear=True):
        await client_main._stream_container_logs(key, container, resume=True)
    client_main._close_log_file(key)

    content = path.read_text().splitlines()
    assert content == [
        "[2026-06-11 08:00:30] already persisted",
        "[2026-06-11 08:00:31] fresh engine line",
    ]
    # Resume used since= derived from the sidecar.
    assert container.logs.call_args.kwargs.get("since") is not None
    assert container.logs.call_args.kwargs.get("timestamps") is True


class TestLogDownloadEndpoint:
    @pytest.mark.anyio
    async def test_download_and_404(self, logs_dir):
        key = "org/model:8001"
        path = client_main._log_file_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[2026-06-11 08:00:32] hello\n")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/deployments/logs/download", params={"key": key})
            assert resp.status_code == 200
            assert resp.text == "[2026-06-11 08:00:32] hello\n"
            resp = await client.get(
                "/deployments/logs/download", params={"key": "missing:1"}
            )
            assert resp.status_code == 404


@pytest.mark.anyio
async def test_status_endpoint_reports_launch_manifest():
    manifest = {"version": 1, "owner": "alice"}
    with mock.patch.dict(
        client_main._statuses,
        {"org/model:8001": {"model_name": "org/model", "port": 8001, "launch_manifest": manifest}},
        clear=True,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/deployments/status")
    assert resp.status_code == 200
    (dep,) = resp.json()["deployments"]
    assert dep["launch_manifest"] == manifest


# ---------------------------------------------------------------------------
# Client HTTP endpoints (no external dependencies)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_deployments_list_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data


@pytest.mark.anyio
async def test_deployment_status_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "deployments" in data


@pytest.mark.anyio
async def test_deployment_logs_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/deployments/logs", params={"key": "nonexistent:1234"})
        assert resp.status_code == 404


@pytest.mark.anyio
async def test_stop_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/deployments/stop", json={"key": "no-such:1234"})
        assert resp.status_code == 404


@pytest.mark.anyio
async def test_images_list():
    fake = mock.MagicMock()
    fake.images.list.return_value = []
    with mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/images")
            assert resp.status_code == 200
            assert "images" in resp.json()


@pytest.mark.anyio
async def test_delete_image_not_found():
    # Use the ImageNotFound class the module itself imported so the endpoint's
    # `except` matches it regardless of how docker was imported under test.
    # Runtime detection is pinned so the test never probes a real daemon.
    fake = mock.MagicMock()
    fake.images.remove.side_effect = client_main.ImageNotFound("nope")
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["docker"]
    ), mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/images/nonexistent")
            assert resp.status_code == 404


@pytest.mark.anyio
async def test_packages_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/packages")
        assert resp.status_code == 200
        data = resp.json()
        assert "packages" in data


@pytest.mark.anyio
async def test_delete_package_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/packages/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Container management endpoints (/containers, /containers/{id}/stop)
# ---------------------------------------------------------------------------


def _fake_container(
    *, name, short_id, status, labels, image_tags=None, image_id="sha256:img"
):
    c = mock.MagicMock()
    c.name = name
    c.short_id = short_id
    c.status = status
    c.labels = labels
    c.image.tags = image_tags or []
    c.image.short_id = image_id
    c.image.id = image_id
    return c


@pytest.mark.anyio
async def test_list_containers_filters_and_tracks():
    tracked = _fake_container(
        name="vllm-tracked",
        short_id="aaa111",
        status="running",
        labels={"vllm-cluster-manager.managed": "true", "vllm-cluster-manager.key": "trk:8000"},
        image_tags=["vllm/vllm-openai:v0.8.5"],
    )
    rogue = _fake_container(
        name="vllm-rogue",
        short_id="bbb222",
        status="exited",
        labels={"vllm-cluster-manager.managed": "true", "vllm-cluster-manager.key": "rog:8001"},
        image_tags=["vllm/vllm-openai:v0.8.5"],
    )
    unrelated = _fake_container(
        name="postgres",
        short_id="ccc333",
        status="running",
        labels={},
        image_tags=["postgres:16"],
    )
    fake = mock.MagicMock()
    fake.containers.list.return_value = [tracked, rogue, unrelated]

    with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.dict(
        client_main._containers, {"trk:8000": object()}, clear=True
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/containers")

    assert resp.status_code == 200
    containers = resp.json()["containers"]
    by_id = {c["id"]: c for c in containers}
    # The unrelated postgres container is filtered out (not vLLM-related).
    assert set(by_id) == {"aaa111", "bbb222"}
    assert by_id["aaa111"]["tracked"] is True
    assert by_id["bbb222"]["tracked"] is False
    assert by_id["bbb222"]["image"] == "vllm/vllm-openai:v0.8.5"


@pytest.mark.anyio
async def test_stop_container_guards_tracked():
    tracked = _fake_container(
        name="vllm-tracked",
        short_id="aaa111",
        status="running",
        labels={"vllm-cluster-manager.managed": "true", "vllm-cluster-manager.key": "trk:8000"},
        image_tags=["vllm/vllm-openai:v0.8.5"],
    )
    fake = mock.MagicMock()
    fake.containers.get.return_value = tracked

    with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.dict(
        client_main._containers, {"trk:8000": object()}, clear=True
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/containers/aaa111/stop")

    assert resp.status_code == 409
    tracked.stop.assert_not_called()


@pytest.mark.anyio
async def test_stop_container_removes_rogue():
    rogue = _fake_container(
        name="vllm-rogue",
        short_id="bbb222",
        status="exited",
        labels={"vllm-cluster-manager.managed": "true", "vllm-cluster-manager.key": "rog:8001"},
        image_tags=["vllm/vllm-openai:v0.8.5"],
    )
    fake = mock.MagicMock()
    fake.containers.get.return_value = rogue

    with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.dict(
        client_main._containers, {}, clear=True
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/containers/bbb222/stop")

    assert resp.status_code == 200
    assert resp.json() == {"status": "removed", "id": "bbb222"}
    rogue.stop.assert_called_once()
    rogue.remove.assert_called_once()


@pytest.mark.anyio
async def test_stop_container_not_found():
    fake = mock.MagicMock()
    fake.containers.get.side_effect = client_main.NotFound("nope")
    with mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/containers/missing/stop")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Image prune endpoint (/images/prune)
# ---------------------------------------------------------------------------


def _fake_image(*, image_id, tags, size_mb):
    img = mock.MagicMock()
    img.id = image_id
    img.short_id = image_id
    img.tags = tags
    img.attrs = {"Size": size_mb * 1024 * 1024}
    return img


@pytest.mark.anyio
async def test_prune_images_skips_in_use():
    in_use = _fake_image(image_id="sha256:aaa", tags=["vllm/vllm-openai:v0.8.5"], size_mb=2048)
    free = _fake_image(image_id="sha256:bbb", tags=["vllm-cluster-manager/local:x"], size_mb=1024)
    unrelated = _fake_image(image_id="sha256:ccc", tags=["postgres:16"], size_mb=500)

    using_container = mock.MagicMock()
    using_container.image.id = "sha256:aaa"

    def _list(*args, **kwargs):
        # The dangling-derived sweep passes filters={"dangling": True, ...}.
        if kwargs.get("filters", {}).get("dangling"):
            return []
        return [in_use, free, unrelated]

    fake = mock.MagicMock()
    fake.containers.list.return_value = [using_container]
    fake.images.list.side_effect = _list

    with mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/images/prune")

    assert resp.status_code == 200
    body = resp.json()
    assert body["removed"] == ["sha256:bbb"]
    assert body["skipped"] == ["sha256:aaa"]
    assert body["freed_mb"] == 1024
    fake.images.remove.assert_called_once_with("sha256:bbb")


@pytest.mark.anyio
async def test_prune_images_sweeps_dangling_derived():
    free = _fake_image(image_id="sha256:bbb", tags=["vllm-cluster-manager/local:x"], size_mb=1024)
    dangling = _fake_image(image_id="sha256:ddd", tags=[], size_mb=2048)

    def _list(*args, **kwargs):
        if kwargs.get("filters", {}).get("dangling"):
            return [dangling]  # labeled <none> leftover from a moving-tag rebuild
        return [free]

    fake = mock.MagicMock()
    fake.containers.list.return_value = []
    fake.images.list.side_effect = _list

    with mock.patch.object(client_main, "_docker", return_value=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/images/prune")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body["removed"]) == {"sha256:bbb", "sha256:ddd"}
    assert body["freed_mb"] == 1024 + 2048


# ---------------------------------------------------------------------------
# Model cache deletion (DELETE /models/cache/{name})
# ---------------------------------------------------------------------------


def _make_cached_model(cache_dir: Path, name: str = "org/model"):
    """Create a fake hub cache entry plus its .locks twin; return (hub, target, locks)."""
    dir_name = f"models--{name.replace('/', '--')}"
    hub = cache_dir / "hub"
    target = hub / dir_name
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"x")
    locks = hub / ".locks" / dir_name
    locks.mkdir(parents=True)
    return hub, target, locks


class TestDeleteCachedModel:
    @pytest.mark.anyio
    async def test_deletes_model_and_locks(self, tmp_path):
        _, target, locks = _make_cached_model(tmp_path)
        with mock.patch.object(client_main.settings, "hf_cache_dir", str(tmp_path)):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/models/cache/org/model")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "name": "org/model"}
        assert not target.exists()
        assert not locks.exists()

    @pytest.mark.anyio
    async def test_not_found(self, tmp_path):
        with mock.patch.object(client_main.settings, "hf_cache_dir", str(tmp_path)):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/models/cache/org/missing")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_refuses_model_in_use(self, tmp_path):
        _, target, _ = _make_cached_model(tmp_path)
        with mock.patch.object(
            client_main.settings, "hf_cache_dir", str(tmp_path)
        ), mock.patch.dict(
            client_main._statuses, {"org-model:8000": {"model_name": "org/model"}}, clear=True
        ), mock.patch.dict(
            client_main._containers, {"org-model:8000": object()}, clear=True
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/models/cache/org/model")
        assert resp.status_code == 409
        assert target.exists()

    @pytest.mark.anyio
    async def test_permission_error_falls_back_to_docker(self, tmp_path):
        # Containers download models as root, so the agent's native rmtree can
        # hit PermissionError; the endpoint must retry via a root container.
        hub, target, locks = _make_cached_model(tmp_path)

        fake_shutil = mock.MagicMock()
        fake_shutil.rmtree.side_effect = PermissionError("denied")

        fake_docker = mock.MagicMock()

        def _root_delete(image, command, volumes, remove):
            shutil.rmtree(target)
            shutil.rmtree(locks)

        fake_docker.containers.run.side_effect = _root_delete

        with mock.patch.object(
            client_main.settings, "hf_cache_dir", str(tmp_path)
        ), mock.patch.object(client_main, "shutil", fake_shutil), mock.patch.object(
            client_main, "_docker", return_value=fake_docker
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/models/cache/org/model")

        assert resp.status_code == 200
        assert not target.exists()
        assert not locks.exists()
        fake_docker.containers.run.assert_called_once()
        kwargs = fake_docker.containers.run.call_args.kwargs
        args = fake_docker.containers.run.call_args.args
        assert args[0] == client_main._CLEANUP_IMAGE
        assert args[1] == ["rm", "-rf", "/hub/models--org--model", "/hub/.locks/models--org--model"]
        assert kwargs["volumes"] == {str(hub): {"bind": "/hub", "mode": "rw"}}
        assert kwargs["remove"] is True

    @pytest.mark.anyio
    async def test_reports_failure_when_fallback_also_fails(self, tmp_path):
        _, target, _ = _make_cached_model(tmp_path)

        fake_shutil = mock.MagicMock()
        fake_shutil.rmtree.side_effect = PermissionError("denied")
        fake_docker = mock.MagicMock()
        fake_docker.containers.run.side_effect = RuntimeError("docker down")

        with mock.patch.object(
            client_main.settings, "hf_cache_dir", str(tmp_path)
        ), mock.patch.object(client_main, "shutil", fake_shutil), mock.patch.object(
            client_main, "_docker", return_value=fake_docker
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/models/cache/org/model")

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "denied" in detail
        assert "docker down" in detail
        assert target.exists()

    @pytest.mark.anyio
    async def test_silent_no_op_is_reported(self, tmp_path):
        # If rmtree neither raises nor removes the dir, the endpoint must not
        # claim success (regression guard for the old ignore_errors=True).
        _, target, _ = _make_cached_model(tmp_path)

        fake_shutil = mock.MagicMock()  # rmtree does nothing, raises nothing

        with mock.patch.object(
            client_main.settings, "hf_cache_dir", str(tmp_path)
        ), mock.patch.object(client_main, "shutil", fake_shutil):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/models/cache/org/model")

        assert resp.status_code == 500
        assert target.exists()
