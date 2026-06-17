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
    client_main._transfer_clients.clear()
    with mock.patch.object(
        client_main, "_podman", side_effect=RuntimeError("isolated in tests")
    ):
        yield
    client_main._runtime_probe = (0.0, [])
    client_main._transfer_clients.clear()


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
        image.id = "sha256:" + "ab" * 32
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
        assert entry["runtimes"] == ["docker"]
        # Full hex id, no sha256: prefix — Podman's compat API rejects the
        # prefixed short form on delete.
        assert entry["id"] == "ab" * 32

    @pytest.mark.anyio
    async def test_images_deduped_across_runtimes(self):
        # The same image cached in both stores is ONE logical cache entry.
        def _image(tags):
            image = mock.MagicMock()
            image.tags = tags
            image.id = "sha256:" + "ab" * 32
            image.attrs = {"Size": 1024**3}
            return image

        docker_client = mock.MagicMock()
        docker_client.images.list.return_value = [_image(["vllm/vllm-openai:v0.9.1"])]
        podman_client = mock.MagicMock()
        podman_client.images.list.return_value = [
            _image([
                "docker.io/vllm/vllm-openai:v0.9.1",
                "docker.io/vllm/vllm-openai:extra",
            ])
        ]

        def fake_runtime_client(runtime):
            return docker_client if runtime == "docker" else podman_client

        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(
            client_main, "_runtime_client", side_effect=fake_runtime_client
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/images")
        (entry,) = resp.json()["images"]
        assert entry["runtimes"] == ["docker", "podman"]
        assert entry["runtime"] == "docker"
        assert entry["tags"] == ["vllm/vllm-openai:v0.9.1", "vllm/vllm-openai:extra"]

    @pytest.mark.anyio
    async def test_images_merged_across_stores_with_different_ids(self):
        # Real-world ids for vllm/vllm-openai:v0.22.1 — Docker's containerd
        # store reports the manifest-list digest, Podman the config digest.
        # Identical layers identify them as ONE image.
        layers = ["sha256:" + "11" * 32, "sha256:" + "22" * 32]

        def _image(image_hex, tags, size):
            image = mock.MagicMock()
            image.tags = tags
            image.id = "sha256:" + image_hex
            image.attrs = {"Size": size, "RootFS": {"Layers": list(layers)}}
            return image

        docker_hex = "953d3a06d5e64ab582985cd7401289d3abf2a2c14ef2158e9a84313daeec77d7"
        podman_hex = "0204ba447a71c42f300e5218e1fe236f539f8b44da74198132366c3baec5887b"
        docker_client = mock.MagicMock()
        docker_client.images.list.return_value = [
            _image(docker_hex, ["vllm/vllm-openai:v0.22.1"], 9 * 1024**3)
        ]
        podman_client = mock.MagicMock()
        podman_client.images.list.return_value = [
            _image(podman_hex, ["docker.io/vllm/vllm-openai:v0.22.1"], 24 * 1024**3)
        ]

        def fake_runtime_client(runtime):
            return docker_client if runtime == "docker" else podman_client

        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(
            client_main, "_runtime_client", side_effect=fake_runtime_client
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/images")
        (entry,) = resp.json()["images"]
        assert entry["runtimes"] == ["docker", "podman"]
        assert entry["tags"] == ["vllm/vllm-openai:v0.22.1"]
        assert entry["id"] == docker_hex  # first-seen store's id
        assert entry["size_mb"] == 24 * 1024  # max(): the unpacked footprint

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

    @pytest.mark.anyio
    async def test_start_returns_offloaded_list(self, logs_dir):
        # A warm-mode deploy that auto-offloads reports what it moved so the host
        # can surface it transparently.
        def fake_ensure(image_ref, extra_packages, log, progress_cb=None, runtime="docker"):
            return image_ref

        container = mock.MagicMock()
        container.id = "cid"
        offloaded = [{"key": "old:8000", "model_name": "old", "tier": "ram"}]
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker"]
        ), mock.patch.object(client_main, "_ensure_image", fake_ensure), \
            mock.patch.object(client_main, "_run_container", return_value=container), \
            mock.patch.object(client_main, "_image_digest", return_value="sha256:d"), \
            mock.patch.object(client_main, "_effective_runtime", return_value="docker"), \
            mock.patch.object(client_main, "_docker_gpu_error", return_value=None), \
            mock.patch.object(
                client_main, "_ensure_fit", return_value=(True, offloaded)
            ), mock.patch.object(client_main, "_start_proxy", mock.AsyncMock()), \
            mock.patch("asyncio.create_task"), mock.patch.dict(
                client_main._node_policy, {"warm_offload_enabled": True}, clear=False
            ), mock.patch.dict(client_main._statuses, {}, clear=True), \
            mock.patch.dict(client_main._containers, {}, clear=True), \
            mock.patch.dict(client_main._logs, {}, clear=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/start",
                    json={
                        "model_name": "org/model",
                        "port": 38477,
                        "gpu_memory_fraction": 0.5,
                        "vllm_version": "0.9.1",
                        "gpu_ids": [0],
                    },
                )
            assert resp.status_code == 200
            assert resp.json()["offloaded"] == offloaded


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


class TestAgentLogRuntimeTag:
    def test_podman_deployment_lines_tagged_podman(self, logs_dir):
        key = "org/model:8002"
        with mock.patch.dict(
            client_main._statuses, {key: {"container_runtime": "podman"}}, clear=False
        ):
            client_main._append_agent_log(key, "[docker] Started container vllm-cluster-x")
            client_main._append_agent_log(key, "[agent] Probable cause: GPU OOM")
        client_main._close_log_file(key)
        lines = list(client_main._logs[key])
        assert "[podman] Started container vllm-cluster-x" in lines[0]
        # Non-runtime tags pass through untouched.
        assert "[agent] Probable cause" in lines[1]

    def test_docker_deployment_lines_unchanged(self, logs_dir):
        key = "org/model:8003"
        with mock.patch.dict(
            client_main._statuses, {key: {"container_runtime": "docker"}}, clear=False
        ):
            client_main._append_agent_log(key, "[docker] Pulled vllm/vllm-openai:v1")
        client_main._close_log_file(key)
        assert "[docker] Pulled" in client_main._logs[key][0]


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

    def test_size_cap_rolls_into_same_run_overflow_part(self, logs_dir):
        key = "m:1"
        big_line = "first " + "x" * (1024 * 1024 + 100)  # exceeds the 1 MB cap
        with mock.patch.object(client_main.settings, "log_max_mb", 1):
            client_main._append_log_line(key, big_line)
            client_main._append_log_line(key, "second line")
        path = client_main._log_file_for(key)
        client_main._close_log_file(key)
        overflow = client_main._log_overflow_for(path)
        assert overflow.read_text().startswith("first ")
        assert path.read_text().splitlines() == ["second line"]
        # The previous-run slot is untouched by same-run overflow.
        assert not Path(str(path) + ".1").exists()

    def test_repeated_overflow_appends_to_same_part(self, logs_dir):
        key = "m:1"
        big = "x" * (1024 * 1024 + 100)
        with mock.patch.object(client_main.settings, "log_max_mb", 1):
            client_main._append_log_line(key, "one " + big)
            client_main._append_log_line(key, "two " + big)
            client_main._append_log_line(key, "tail line")
        path = client_main._log_file_for(key)
        client_main._close_log_file(key)
        overflow_lines = client_main._log_overflow_for(path).read_text().splitlines()
        assert overflow_lines[0].startswith("one ")
        assert overflow_lines[1].startswith("two ")
        assert path.read_text().splitlines() == ["tail line"]

    def test_rotate_on_fresh_run(self, logs_dir):
        key = "m:1"
        client_main._append_log_line(key, "old run")
        path = client_main._log_file_for(key)
        client_main._log_overflow_for(path).write_text("old overflow\n")
        client_main._open_log_file(key, rotate=True)
        client_main._append_log_line(key, "new run")
        client_main._close_log_file(key)
        assert path.read_text().splitlines() == ["new run"]
        assert "old run" in Path(str(path) + ".1").read_text()
        # Same-run overflow belongs to the previous run; a fresh run drops it.
        assert not client_main._log_overflow_for(path).exists()


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
    # No re-attach: the deployment is not desired-running.
    assert container.logs.call_count == 1


@pytest.mark.anyio
async def test_stream_resume_without_sidecar_backfills_full_history(logs_dir):
    """No resume point -> fetch the container's complete history (no tail),
    setting aside any partial file so the fresh one holds the whole run."""
    key = "org/model:8001"
    path = client_main._log_file_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[2026-06-11 08:00:30] partial 200-line backfill\n")

    container = mock.MagicMock()
    container.logs.return_value = [
        b"2026-06-11T07:00:01.000000001Z engine boot line\n",
        b"2026-06-11T08:00:30.000000001Z partial 200-line backfill\n",
    ]
    with mock.patch.dict(client_main._statuses, {key: {}}, clear=True):
        await client_main._stream_container_logs(key, container, resume=True)
    client_main._close_log_file(key)

    # Full history requested: neither tail nor since.
    assert "tail" not in container.logs.call_args.kwargs
    assert "since" not in container.logs.call_args.kwargs
    # The fresh file is exactly the streamed history; the partial copy was
    # set aside as the previous-run file.
    assert path.read_text().splitlines() == [
        "[2026-06-11 07:00:01] engine boot line",
        "[2026-06-11 08:00:30] partial 200-line backfill",
    ]
    assert "partial 200-line backfill" in Path(str(path) + ".1").read_text()


@pytest.mark.anyio
async def test_stream_reattaches_while_desired_running(logs_dir):
    """A follow stream that ends mid-run is re-attached with since= so later
    lines (e.g. restart attempts) keep landing in the same run log."""
    key = "org/model:8001"
    container = mock.MagicMock()
    container.logs.side_effect = [
        [b"2026-06-11T08:00:01.000000001Z first attempt line\n"],
        [
            b"2026-06-11T08:00:01.000000001Z first attempt line\n",  # overlap
            b"2026-06-11T08:00:05.000000001Z after restart line\n",
        ],
    ]
    # First stream end: container still there -> re-attach; second: gone.
    container.reload.side_effect = [None, client_main.NotFound("gone")]
    with mock.patch.dict(
        client_main._statuses, {key: {"desired_state": "running"}}, clear=True
    ), mock.patch.dict(
        client_main._containers, {key: container}, clear=True
    ), mock.patch.object(client_main.time, "sleep"):
        await client_main._stream_container_logs(key, container)
    client_main._close_log_file(key)

    assert container.logs.call_count == 2
    # The re-attach resumed from the last seen timestamp...
    assert container.logs.call_args.kwargs.get("since") is not None
    # ...and the overlap line was deduped by the skip-guard.
    path = client_main._log_file_for(key)
    assert path.read_text().splitlines() == [
        "[2026-06-11 08:00:01] first attempt line",
        "[2026-06-11 08:00:05] after restart line",
    ]


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
    async def test_download_concatenates_overflow_part(self, logs_dir):
        """A run that hit the size cap spans <file>.0 + <file>; the download
        holds both, in order, so the run is complete from the beginning."""
        key = "org/model:8001"
        path = client_main._log_file_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        client_main._log_overflow_for(path).write_text("[ts] start of run\n")
        path.write_text("[ts] newest lines\n")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/deployments/logs/download", params={"key": key})
        assert resp.status_code == 200
        assert resp.text == "[ts] start of run\n[ts] newest lines\n"


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


class TestDeleteImageEverywhere:
    """Plain deletes treat both stores as one cache — no surviving copies."""

    _IMAGE_HEX = "ab" * 32
    _LAYERS = ["sha256:" + "11" * 32, "sha256:" + "22" * 32]

    def _clients(self, docker_tags=None, podman_tags=None):
        def _client(tags, image_hex=None):
            client = mock.MagicMock()
            image = mock.MagicMock()
            image.tags = ["vllm/vllm-openai:v0.9.1"] if tags is None else tags
            image.id = "sha256:" + (image_hex or self._IMAGE_HEX)
            image.attrs = {"Size": 0, "RootFS": {"Layers": list(self._LAYERS)}}
            client.images.get.return_value = image
            return client

        docker_client = _client(docker_tags)
        podman_client = _client(podman_tags)

        def fake_runtime_client(runtime):
            return docker_client if runtime == "docker" else podman_client

        return docker_client, podman_client, fake_runtime_client

    @pytest.mark.anyio
    async def test_delete_removes_from_all_runtimes(self):
        docker_client, podman_client, fake = self._clients()
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/images/sha256:abc")
        assert resp.status_code == 200
        assert resp.json()["runtimes"] == ["docker", "podman"]
        # Removal uses each store's own full-hex id (the requested id may be
        # the other store's identity scheme).
        docker_client.images.remove.assert_called_once_with(self._IMAGE_HEX)
        podman_client.images.remove.assert_called_once_with(self._IMAGE_HEX)

    @pytest.mark.anyio
    async def test_delete_with_runtime_param_targets_one(self):
        docker_client, podman_client, fake = self._clients()
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/images/sha256:abc?runtime=podman")
        assert resp.status_code == 200
        assert resp.json()["runtimes"] == ["podman"]
        docker_client.images.remove.assert_not_called()
        podman_client.images.remove.assert_called_once()

    @pytest.mark.anyio
    async def test_partial_in_use_still_succeeds_with_skip_note(self):
        docker_client, podman_client, fake = self._clients()
        podman_client.images.remove.side_effect = client_main.APIError("in use")
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/images/sha256:abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["runtimes"] == ["docker"]
        assert "podman" in body["skipped"]

    @pytest.mark.anyio
    async def test_foreign_tags_protected(self):
        # Docker's copy of the id also carries a user's own tag: only the
        # vLLM ref is untagged; podman's single-tag copy is removed by id.
        docker_client, podman_client, fake = self._clients(
            docker_tags=["vllm/vllm-openai:v0.9.1", "alpine:3"]
        )
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/images/sha256:abc")
        assert resp.status_code == 200
        assert resp.json()["runtimes"] == ["docker", "podman"]
        docker_client.images.remove.assert_called_once_with("vllm/vllm-openai:v0.9.1")
        podman_client.images.remove.assert_called_once_with(self._IMAGE_HEX)

    @pytest.mark.anyio
    async def test_id_held_only_under_foreign_tags_skipped(self):
        docker_client, podman_client, fake = self._clients(
            docker_tags=["alpine:3"]
        )
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/images/sha256:abc")
        assert resp.status_code == 200
        assert resp.json()["runtimes"] == ["podman"]
        docker_client.images.remove.assert_not_called()

    @pytest.mark.anyio
    async def test_delete_resolves_other_store_by_content(self):
        # Containerd-store Docker ids the image by its manifest digest, Podman
        # by the config digest — same image, different ids. Deleting by one id
        # must still clear the other store via the layer-content match.
        docker_hex = "95" * 32
        podman_hex = "02" * 32
        layers = list(self._LAYERS)

        def _image(image_hex):
            image = mock.MagicMock()
            image.tags = ["vllm/vllm-openai:v0.22.1"]
            image.id = "sha256:" + image_hex
            image.attrs = {"Size": 0, "RootFS": {"Layers": layers}}
            return image

        docker_client = mock.MagicMock()
        docker_client.images.get.return_value = _image(docker_hex)
        podman_client = mock.MagicMock()
        podman_client.images.get.side_effect = client_main.ImageNotFound("missing")
        podman_client.images.list.return_value = [_image(podman_hex)]

        def fake(runtime):
            return docker_client if runtime == "docker" else podman_client

        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete(f"/images/{docker_hex}")
        assert resp.status_code == 200
        assert resp.json()["runtimes"] == ["docker", "podman"]
        docker_client.images.remove.assert_called_once_with(docker_hex)
        # Podman's copy was found by content and removed under ITS id.
        podman_client.images.remove.assert_called_once_with(podman_hex)

    @pytest.mark.anyio
    async def test_all_copies_in_use_conflicts(self):
        docker_client, podman_client, fake = self._clients()
        docker_client.images.remove.side_effect = client_main.APIError("in use")
        podman_client.images.remove.side_effect = client_main.APIError("in use")
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(client_main, "_runtime_client", side_effect=fake):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/images/sha256:abc")
        assert resp.status_code == 409


class TestCrossRuntimeImageCopy:
    """_ensure_image transfers locally instead of re-pulling from the registry."""

    def _clients(self, target_has_image=False):
        target = mock.MagicMock()
        if not target_has_image:
            # get() fails until a load() succeeds, then the post-load check passes.
            target.images.get.side_effect = [
                client_main.ImageNotFound("missing"),
                mock.MagicMock(),
            ]
        source = mock.MagicMock()  # images.get succeeds -> source has the image

        def fake_runtime_client(runtime):
            return target if runtime == "podman" else source

        return target, source, fake_runtime_client

    def test_copies_from_other_runtime_instead_of_pulling(self):
        target, source, fake = self._clients()
        logged = []
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(
            client_main, "_runtime_client", side_effect=fake
        ), mock.patch.object(
            client_main, "_transfer_client", side_effect=fake
        ), mock.patch.object(client_main, "_pull_image") as pull:
            ref = client_main._ensure_image(
                "vllm/vllm-openai:v0.9.1", None, logged.append, runtime="podman"
            )
        assert ref == "vllm/vllm-openai:v0.9.1"
        pull.assert_not_called()
        source.api.get_image.assert_called_once_with("vllm/vllm-openai:v0.9.1")
        target.images.load.assert_called_once()
        assert any("Copying image" in line for line in logged)

    def test_failed_copy_falls_back_to_pull(self):
        target, source, fake = self._clients()
        target.images.load.side_effect = client_main.APIError("load failed")
        target.images.get.side_effect = client_main.ImageNotFound("missing")
        logged = []
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(
            client_main, "_runtime_client", side_effect=fake
        ), mock.patch.object(
            client_main, "_transfer_client", side_effect=fake
        ), mock.patch.object(client_main, "_pull_image") as pull:
            client_main._ensure_image(
                "vllm/vllm-openai:v0.9.1", None, logged.append, runtime="podman"
            )
        pull.assert_called_once()
        assert any("falling back" in line for line in logged)

    def test_moving_tags_never_copy(self):
        target, source, fake = self._clients()
        with mock.patch.object(
            client_main, "_available_runtimes", return_value=["docker", "podman"]
        ), mock.patch.object(
            client_main, "_runtime_client", side_effect=fake
        ), mock.patch.object(
            client_main, "_transfer_client", side_effect=fake
        ), mock.patch.object(client_main, "_pull_image") as pull:
            client_main._ensure_image(
                "vllm/vllm-openai:latest", None, lambda _msg: None, runtime="podman"
            )
        pull.assert_called_once()
        source.api.get_image.assert_not_called()

    def test_transfer_clients_use_long_timeout(self):
        # The default 60s read timeout aborts multi-GB save->load transfers
        # mid-stream; transfer clients must be built with the long timeout.
        client_main._transfer_clients.clear()
        with mock.patch.object(
            client_main, "_runtime_client", return_value=mock.MagicMock()
        ), mock.patch.object(
            client_main.docker, "from_env", return_value=mock.MagicMock()
        ) as from_env, mock.patch.object(
            client_main.docker, "DockerClient", return_value=mock.MagicMock()
        ) as docker_client, mock.patch.object(
            client_main, "_podman_base_url", "unix:///run/x/podman.sock"
        ):
            client_main._transfer_client("docker")
            client_main._transfer_client("podman")
            # Cached: a second call must not build a new client.
            client_main._transfer_client("docker")
        client_main._transfer_clients.clear()
        from_env.assert_called_once_with(timeout=client_main._TRANSFER_TIMEOUT_S)
        docker_client.assert_called_once_with(
            base_url="unix:///run/x/podman.sock",
            timeout=client_main._TRANSFER_TIMEOUT_S,
        )
        assert client_main._TRANSFER_TIMEOUT_S >= 1800


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
    *,
    name,
    short_id,
    status,
    labels,
    image_tags=None,
    image_id="sha256:img",
    repo_digests=None,
    layers=None,
):
    c = mock.MagicMock()
    c.name = name
    c.short_id = short_id
    c.status = status
    c.labels = labels
    c.image.tags = image_tags or []
    c.image.short_id = image_id
    c.image.id = image_id
    # Real dict so the recognition path can read RepoDigests / RootFS layers
    # (a bare MagicMock would mis-iterate).
    c.image.attrs = {
        "RepoDigests": repo_digests or [],
        "RootFS": {"Layers": layers or []},
    }
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
async def test_list_containers_recognizes_untagged_vllm_by_content():
    """An untagged vLLM image (lost its tag in a transfer/re-pull) is still rogue."""
    layers = ["sha256:layerA", "sha256:layerB"]
    # Unmanaged, no tags, no repo-digests: only layer identity gives it away.
    orphan = _fake_container(
        name="orphan",
        short_id="ddd444",
        status="running",
        labels={},
        image_tags=[],
        image_id="sha256:orphan",
        layers=layers,
    )
    # Unrelated container whose layers do NOT match a vLLM image stays filtered.
    unrelated = _fake_container(
        name="postgres",
        short_id="eee555",
        status="running",
        labels={},
        image_tags=["postgres:16"],
        image_id="sha256:pg",
        layers=["sha256:pg1"],
    )
    cached_vllm = _fake_image(
        image_id="sha256:vllm",
        tags=["vllm/vllm-openai:v0.9.1"],
        size_mb=2048,
        layers=layers,
    )
    fake = mock.MagicMock()
    fake.containers.list.return_value = [orphan, unrelated]
    fake.images.list.return_value = [cached_vllm]

    with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.dict(
        client_main._containers, {}, clear=True
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/containers")

    assert resp.status_code == 200
    by_id = {c["id"]: c for c in resp.json()["containers"]}
    assert set(by_id) == {"ddd444"}
    assert by_id["ddd444"]["tracked"] is False


@pytest.mark.anyio
async def test_list_containers_recognizes_vllm_tag_not_first():
    """A vLLM image carrying another tag first is still recognized."""
    multi = _fake_container(
        name="multitag",
        short_id="fff666",
        status="running",
        labels={},
        image_tags=["myrepo/foo:1", "vllm/vllm-openai:v0.8.5"],
    )
    fake = mock.MagicMock()
    fake.containers.list.return_value = [multi]

    with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.dict(
        client_main._containers, {}, clear=True
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/containers")

    assert resp.status_code == 200
    by_id = {c["id"]: c for c in resp.json()["containers"]}
    assert set(by_id) == {"fff666"}
    assert by_id["fff666"]["tracked"] is False


@pytest.mark.anyio
async def test_list_containers_recognizes_digest_pinned_vllm():
    """A digest-pinned vLLM container (no tag, only RepoDigests) is recognized."""
    bydigest = _fake_container(
        name="bydigest",
        short_id="ggg777",
        status="running",
        labels={},
        image_tags=[],
        repo_digests=["vllm/vllm-openai@sha256:abc123"],
    )
    fake = mock.MagicMock()
    fake.containers.list.return_value = [bydigest]

    with mock.patch.object(client_main, "_docker", return_value=fake), mock.patch.dict(
        client_main._containers, {}, clear=True
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/containers")

    assert resp.status_code == 200
    by_id = {c["id"]: c for c in resp.json()["containers"]}
    assert set(by_id) == {"ggg777"}
    assert by_id["ggg777"]["tracked"] is False


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
# Rogue vLLM GPU process detection + container child-process reaping
# ---------------------------------------------------------------------------


def _fake_proc(*, name="VLLM::EngineCore", cmdline=None, create_time=100.0):
    proc = mock.MagicMock()
    proc.name.return_value = name
    proc.cmdline.return_value = cmdline if cmdline is not None else ["python", "-m", "vllm"]
    proc.create_time.return_value = create_time
    return proc


class TestLooksLikeVllm:
    def test_matches_engine_core_name(self):
        assert client_main._looks_like_vllm("VLLM::EngineCore") is True

    def test_matches_vllm_in_cmdline(self):
        proc = _fake_proc(name="python", cmdline=["python", "-m", "vllm.entrypoints"])
        with mock.patch.object(client_main.psutil, "Process", return_value=proc):
            assert client_main._looks_like_vllm("python", 4242) is True

    def test_rejects_unrelated(self):
        proc = _fake_proc(name="postgres", cmdline=["postgres", "-D", "/data"])
        with mock.patch.object(client_main.psutil, "Process", return_value=proc):
            assert client_main._looks_like_vllm("postgres", 4242) is False


def test_gpu_processes_smi_parses():
    csv = "111, 40869, VLLM::EngineCore\n222, 1024, python -m something, weird\n"
    with mock.patch.object(
        client_main.shutil, "which", return_value="/usr/bin/nvidia-smi"
    ), mock.patch.object(client_main.subprocess, "check_output", return_value=csv):
        procs = client_main._gpu_processes_smi()
    assert procs[0] == {
        "pid": 111,
        "gpu_index": None,
        "gpu_memory_mb": 40869,
        "process_name": "VLLM::EngineCore",
    }
    assert procs[1]["pid"] == 222
    # process_name keeps its tail intact even when it contains commas.
    assert procs[1]["process_name"] == "python -m something, weird"


def test_gpu_processes_nvml_parses():
    info = mock.MagicMock(pid=111, usedGpuMemory=40869 * 1024 * 1024)
    fake_nvml = mock.MagicMock()
    fake_nvml.nvmlDeviceGetCount.return_value = 1
    fake_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [info]
    with mock.patch.dict("sys.modules", {"pynvml": fake_nvml}), mock.patch.object(
        client_main, "_proc_name", return_value="VLLM::EngineCore"
    ):
        procs = client_main._gpu_processes_nvml()
    assert procs == [
        {
            "pid": 111,
            "gpu_index": 0,
            "gpu_memory_mb": 40869,
            "process_name": "VLLM::EngineCore",
        }
    ]


@pytest.mark.anyio
async def test_list_gpu_processes_marks_rogue_and_tracked():
    procs = [
        {"pid": 111, "gpu_index": 1, "gpu_memory_mb": 40000, "process_name": "VLLM::EngineCore"},
        {"pid": 222, "gpu_index": 0, "gpu_memory_mb": 16000, "process_name": "VLLM::Worker"},
        {"pid": 333, "gpu_index": 0, "gpu_memory_mb": 500, "process_name": "postgres"},
    ]
    with mock.patch.object(
        client_main, "_gpu_compute_processes", return_value=procs
    ), mock.patch.object(
        client_main,
        "_looks_like_vllm",
        side_effect=lambda name, pid=None: "vllm" in name.lower(),
    ), mock.patch.object(
        client_main, "_tracked_gpu_pids", return_value={222: "model:abc"}
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/gpu-processes")
    assert resp.status_code == 200
    items = {p["pid"]: p for p in resp.json()["processes"]}
    # postgres (not a vLLM process) is filtered out entirely.
    assert set(items) == {111, 222}
    assert items[111]["tracked"] is False
    assert items[111]["key"] is None
    assert items[222]["tracked"] is True
    assert items[222]["key"] == "model:abc"


@pytest.mark.anyio
async def test_kill_gpu_process_success():
    proc = _fake_proc()
    with mock.patch.object(
        client_main.psutil, "Process", return_value=proc
    ), mock.patch.object(
        client_main, "_tracked_gpu_pids", return_value={}
    ), mock.patch.object(client_main, "_looks_like_vllm", return_value=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/111/kill")
    assert resp.status_code == 200
    assert resp.json() == {"status": "killed", "pid": 111}
    proc.terminate.assert_called_once()


@pytest.mark.anyio
async def test_kill_gpu_process_escalates_to_sigkill():
    proc = _fake_proc()
    proc.wait.side_effect = client_main.psutil.TimeoutExpired(1)
    with mock.patch.object(
        client_main.psutil, "Process", return_value=proc
    ), mock.patch.object(
        client_main, "_tracked_gpu_pids", return_value={}
    ), mock.patch.object(client_main, "_looks_like_vllm", return_value=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/111/kill")
    assert resp.status_code == 200
    proc.kill.assert_called_once()


@pytest.mark.anyio
async def test_kill_gpu_process_refuses_tracked():
    proc = _fake_proc()
    with mock.patch.object(
        client_main.psutil, "Process", return_value=proc
    ), mock.patch.object(client_main, "_tracked_gpu_pids", return_value={111: "k"}):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/111/kill")
    assert resp.status_code == 409
    proc.terminate.assert_not_called()


@pytest.mark.anyio
async def test_kill_gpu_process_rejects_non_vllm():
    proc = _fake_proc(name="postgres")
    with mock.patch.object(
        client_main.psutil, "Process", return_value=proc
    ), mock.patch.object(
        client_main, "_tracked_gpu_pids", return_value={}
    ), mock.patch.object(client_main, "_looks_like_vllm", return_value=False):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/111/kill")
    assert resp.status_code == 404
    proc.terminate.assert_not_called()


@pytest.mark.anyio
async def test_kill_gpu_process_permission_denied_and_container_fails():
    # Agent can't signal the root-owned process AND the root-container fallback
    # also can't kill it → 403 with manual-cleanup guidance.
    proc = _fake_proc()
    proc.terminate.side_effect = client_main.psutil.AccessDenied()
    with mock.patch.object(
        client_main.psutil, "Process", return_value=proc
    ), mock.patch.object(
        client_main, "_tracked_gpu_pids", return_value={}
    ), mock.patch.object(
        client_main, "_looks_like_vllm", return_value=True
    ), mock.patch.object(client_main, "_kill_pid_via_container", return_value=False):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/111/kill")
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_kill_gpu_process_escalates_to_root_container():
    # A root-owned orphan (rootful Docker): native kill is denied, but the
    # root-container fallback succeeds → 200.
    proc = _fake_proc()
    proc.terminate.side_effect = client_main.psutil.AccessDenied()
    with mock.patch.object(
        client_main.psutil, "Process", return_value=proc
    ), mock.patch.object(
        client_main, "_tracked_gpu_pids", return_value={}
    ), mock.patch.object(
        client_main, "_looks_like_vllm", return_value=True
    ), mock.patch.object(
        client_main, "_kill_pid_via_container", return_value=True
    ) as escalate:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/111/kill")
    assert resp.status_code == 200
    assert resp.json() == {"status": "killed", "pid": 111, "via": "container"}
    escalate.assert_called_once_with(111)


def _clients_by_runtime(**clients):
    """side_effect for `_runtime_client` returning a distinct mock per runtime."""
    def _get(runtime):
        return clients[runtime]

    return _get


def _assert_host_kill(client, pid):
    client.containers.run.assert_called_once()
    args, kwargs = client.containers.run.call_args
    assert args[0] == client_main._CLEANUP_IMAGE
    assert args[1] == ["kill", "-9", str(pid)]
    assert kwargs["pid_mode"] == "host"  # share the host PID namespace
    assert kwargs["remove"] is True


def test_kill_pid_via_container_prefers_docker():
    docker_client = mock.MagicMock()
    podman_client = mock.MagicMock()
    # Docker listed last to prove it is still tried first (it is the rootful
    # runtime that produces root-owned orphans).
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["podman", "docker"]
    ), mock.patch.object(
        client_main,
        "_runtime_client",
        side_effect=_clients_by_runtime(docker=docker_client, podman=podman_client),
    ), mock.patch.object(client_main.psutil, "pid_exists", return_value=False):
        assert client_main._kill_pid_via_container(2840605) is True
    _assert_host_kill(docker_client, 2840605)
    podman_client.containers.run.assert_not_called()  # Docker cleared it first


def test_kill_pid_via_container_podman_only():
    podman_client = mock.MagicMock()
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["podman"]
    ), mock.patch.object(
        client_main,
        "_runtime_client",
        side_effect=_clients_by_runtime(podman=podman_client),
    ), mock.patch.object(client_main.psutil, "pid_exists", return_value=False):
        assert client_main._kill_pid_via_container(777) is True
    # On a Podman-only host the escalation still runs, via Podman.
    _assert_host_kill(podman_client, 777)


def test_kill_pid_via_container_falls_back_from_docker_to_podman():
    docker_client = mock.MagicMock()
    # Docker can't do it (e.g. daemon refuses the request); the loop moves on.
    docker_client.containers.run.side_effect = RuntimeError("docker kill failed")
    podman_client = mock.MagicMock()
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["docker", "podman"]
    ), mock.patch.object(
        client_main,
        "_runtime_client",
        side_effect=_clients_by_runtime(docker=docker_client, podman=podman_client),
    ), mock.patch.object(
        # Still alive after the Docker attempt, gone after the Podman one.
        client_main.psutil,
        "pid_exists",
        side_effect=[True, False],
    ):
        assert client_main._kill_pid_via_container(2840605) is True
    docker_client.containers.run.assert_called_once()
    _assert_host_kill(podman_client, 2840605)


@pytest.mark.anyio
async def test_kill_gpu_process_not_found():
    with mock.patch.object(
        client_main.psutil, "Process", side_effect=client_main.psutil.NoSuchProcess(999)
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/gpu-processes/999/kill")
    assert resp.status_code == 404


def test_remove_container_reaps_surviving_children():
    container = mock.MagicMock()
    container.id = "abc123def456"
    snapshot = {1234: 100.0}
    with mock.patch.object(
        client_main, "_container_process_tree", return_value=snapshot
    ), mock.patch.object(client_main, "_kill_pids") as kill:
        client_main._remove_container(container)
    container.stop.assert_called_once()
    container.remove.assert_called_once_with(force=True)
    kill.assert_called_once()
    assert kill.call_args.args[0] == snapshot


def test_kill_pids_kills_unchanged_process():
    proc = _fake_proc(create_time=100.0)
    with mock.patch.object(client_main.psutil, "Process", return_value=proc):
        client_main._kill_pids({555: 100.0})
    proc.kill.assert_called_once()


def test_kill_pids_skips_recycled_pid():
    proc = _fake_proc(create_time=999.0)  # create_time mismatch => pid recycled
    with mock.patch.object(client_main.psutil, "Process", return_value=proc):
        client_main._kill_pids({555: 100.0})
    proc.kill.assert_not_called()


def test_kill_pids_ignores_gone_process():
    with mock.patch.object(
        client_main.psutil, "Process", side_effect=client_main.psutil.NoSuchProcess(555)
    ):
        client_main._kill_pids({555: 100.0})  # must not raise


def test_run_container_uses_init():
    fake = mock.MagicMock()
    fake.containers.get.side_effect = client_main.NotFound("none")
    sentinel = object()
    fake.containers.run.return_value = sentinel
    with mock.patch.object(
        client_main, "_runtime_client", return_value=fake
    ), mock.patch.object(client_main, "_volumes", return_value={}):
        result = client_main._run_container(
            "img:1", ["--model", "m"], "vllm-x", {}, [], {"k": "v"}
        )
    assert result is sentinel
    kwargs = fake.containers.run.call_args.kwargs
    assert kwargs["init"] is True
    assert kwargs["network_mode"] == "host"
    assert kwargs["ipc_mode"] == "host"


# ---------------------------------------------------------------------------
# Image prune endpoint (/images/prune)
# ---------------------------------------------------------------------------


def _fake_image(*, image_id, tags, size_mb, layers=None, repo_digests=None):
    img = mock.MagicMock()
    img.id = image_id
    img.short_id = image_id
    img.tags = tags
    attrs = {"Size": size_mb * 1024 * 1024}
    if layers is not None:
        attrs["RootFS"] = {"Layers": list(layers)}
    if repo_digests is not None:
        attrs["RepoDigests"] = repo_digests
    img.attrs = attrs
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


# ---------------------------------------------------------------------------
# Warm cache: pause/resume, eviction, rogue artifacts
# ---------------------------------------------------------------------------


def _meta(**over):
    base = dict(
        model_name="m",
        port=8000,
        internal_port=41000,
        gpu_memory_fraction=0.5,
        gpu_ids=[0],
        status="running",
        pause_tier=None,
        pinned=False,
        paused_ram_mb=0,
        last_active_at=0.0,
        requests_running=0,
        engine_args={},
        warm=True,
    )
    base.update(over)
    return base


class TestWarmFor:
    def test_node_enabled(self):
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": True}):
            assert client_main._warm_for(StartRequest(model_name="m", port=8000, gpu_memory_fraction=0.5))

    def test_payload_flag(self):
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": False}):
            req = StartRequest(model_name="m", port=8000, gpu_memory_fraction=0.5, warm_offload=True)
            assert client_main._warm_for(req)

    def test_disabled_everywhere(self):
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": False}):
            assert not client_main._warm_for(StartRequest(model_name="m", port=8000, gpu_memory_fraction=0.5))

    def test_opt_out_overrides(self):
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": True}):
            req = StartRequest(
                model_name="m", port=8000, gpu_memory_fraction=0.5,
                engine_args={"disable_sleep_mode": True},
            )
            assert not client_main._warm_for(req)


def test_allocate_internal_port_skips_used():
    with mock.patch.dict(
        client_main._statuses, {"a:1": {"internal_port": 41000}}, clear=True
    ):
        port = client_main._allocate_internal_port({41001})
    assert port in client_main._INTERNAL_PORT_RANGE
    assert port not in (41000, 41001)


class TestIsBusy:
    def test_in_flight_requests(self):
        assert client_main._is_busy(_meta(requests_running=2)) is True

    def test_recent_activity(self):
        assert client_main._is_busy(_meta(last_active_at=client_main.time.monotonic())) is True

    def test_idle(self):
        assert client_main._is_busy(_meta(last_active_at=0.0, requests_running=0)) is False


def test_pick_victim_skips_pinned_and_busy_picks_lru():
    statuses = {
        "lru:8000": _meta(last_active_at=100.0),
        "fresh:8001": _meta(port=8001, last_active_at=200.0),
        "pinned:8002": _meta(port=8002, pinned=True, last_active_at=1.0),
        "busy:8003": _meta(port=8003, requests_running=3, last_active_at=1.0),
    }
    containers = {k: object() for k in statuses}
    with mock.patch.dict(client_main._statuses, statuses, clear=True), mock.patch.dict(
        client_main._containers, containers, clear=True
    ):
        victim = client_main._pick_victim([0], requester_key="new:9000")
    assert victim == "lru:8000"


def test_pick_victim_none_when_all_protected():
    statuses = {"pinned:8002": _meta(port=8002, pinned=True)}
    with mock.patch.dict(client_main._statuses, statuses, clear=True), mock.patch.dict(
        client_main._containers, {"pinned:8002": object()}, clear=True
    ):
        assert client_main._pick_victim([0], requester_key="new:9000") is None


def test_pick_victim_skips_non_warm():
    # A model deployed before warm cache was enabled (no sleep mode, no loopback
    # internal_port + proxy) is not evictable — stopping it would leave it
    # un-resumable, so it must never be chosen as a victim.
    statuses = {"legacy:8000": _meta(warm=False, internal_port=None, last_active_at=1.0)}
    with mock.patch.dict(client_main._statuses, statuses, clear=True), mock.patch.dict(
        client_main._containers, {"legacy:8000": object()}, clear=True
    ):
        assert client_main._pick_victim([0], requester_key="new:9000") is None


class TestEnsureFit:
    @pytest.mark.anyio
    async def test_warm_disabled_returns_none(self):
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": False}):
            assert await client_main._ensure_fit([0], 0.5, "new:9000") is None

    @pytest.mark.anyio
    async def test_fits_without_eviction(self):
        statuses = {"small:8000": _meta(gpu_memory_fraction=0.2)}
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": True}), \
            mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"small:8000": object()}, clear=True):
            assert await client_main._ensure_fit([0], 0.5, "new:9000") == (True, [])

    @pytest.mark.anyio
    async def test_evicts_lru_then_fits(self):
        statuses = {"old:8000": _meta(gpu_memory_fraction=0.9, last_active_at=1.0)}

        async def fake_pause(key, tier=None):
            client_main._statuses[key]["pause_tier"] = tier or "ram"

        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": True}), \
            mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"old:8000": object()}, clear=True), \
            mock.patch.object(client_main, "_pause", side_effect=fake_pause) as paused:
            ok, offloaded = await client_main._ensure_fit([0], 0.5, "new:9000")
        assert ok is True
        # The planner assigns an explicit tier (RAM here, unlimited budget).
        paused.assert_awaited_once_with("old:8000", tier="ram")
        assert offloaded == [{"key": "old:8000", "model_name": "m", "tier": "ram"}]

    @pytest.mark.anyio
    async def test_no_eligible_victim_returns_false(self):
        statuses = {"pinned:8000": _meta(gpu_memory_fraction=0.9, pinned=True)}
        with mock.patch.dict(client_main._node_policy, {"warm_offload_enabled": True}), \
            mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"pinned:8000": object()}, clear=True):
            assert await client_main._ensure_fit([0], 0.5, "new:9000") == (False, [])

    @pytest.mark.anyio
    async def test_non_warm_victim_never_sleeps(self):
        # GPU 0 holds a non-warm legacy model and a warm one. Only the warm model
        # is eligible; the legacy model must be left untouched (no /sleep to a
        # None internal port, no 500).
        statuses = {
            "legacy:8000": _meta(
                gpu_memory_fraction=0.5, warm=False, internal_port=None, last_active_at=1.0
            ),
            "warm:8001": _meta(
                port=8001, internal_port=41001, gpu_memory_fraction=0.5, last_active_at=2.0
            ),
        }
        with mock.patch.dict(
            client_main._node_policy,
            {"warm_offload_enabled": True, "ram_cache_limit_mb": None},
            clear=False,
        ), mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(
                client_main._containers, {k: object() for k in statuses}, clear=True
            ), mock.patch.object(client_main, "_vllm_sleep") as sleep, \
            mock.patch.object(client_main, "_ram_estimate", return_value=4096.0):
            ok, offloaded = await client_main._ensure_fit([0], 0.5, "new:9000")
            # The legacy model must be left untouched (asserted inside the patch).
            assert client_main._statuses["legacy:8000"].get("pause_tier") is None
        assert ok is True
        assert [o["key"] for o in offloaded] == ["warm:8001"]
        sleep.assert_awaited_once()  # the warm victim, not the legacy one


class TestPlanEviction:
    def test_fits_without_eviction(self):
        statuses = {"small:8000": _meta(gpu_memory_fraction=0.2)}
        with mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"small:8000": object()}, clear=True):
            plan = client_main._plan_eviction([0], 0.5, "new:9000")
        assert plan["fits"] is True
        assert plan["plan"] == []
        assert plan["reason"] is None

    def test_warm_victim_to_ram_unlimited(self):
        statuses = {"old:8000": _meta(gpu_memory_fraction=0.9, last_active_at=1.0)}
        with mock.patch.dict(
            client_main._node_policy, {"ram_cache_limit_mb": None}, clear=False
        ), mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"old:8000": object()}, clear=True):
            plan = client_main._plan_eviction([0], 0.5, "new:9000")
        assert plan["fits"] is True
        assert [(s["key"], s["tier"]) for s in plan["plan"]] == [("old:8000", "ram")]

    def test_warm_victim_to_disk_over_budget(self):
        statuses = {"old:8000": _meta(gpu_memory_fraction=0.9, last_active_at=1.0)}
        with mock.patch.dict(
            client_main._node_policy, {"ram_cache_limit_mb": 100}, clear=False
        ), mock.patch.object(client_main, "_ram_estimate", return_value=5000.0), \
            mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"old:8000": object()}, clear=True):
            plan = client_main._plan_eviction([0], 0.5, "new:9000")
        assert plan["fits"] is True
        assert [(s["key"], s["tier"]) for s in plan["plan"]] == [("old:8000", "disk")]

    def test_cascade_multiple_victims(self):
        statuses = {
            "a:8000": _meta(gpu_memory_fraction=0.6, last_active_at=1.0),
            "b:8001": _meta(port=8001, gpu_memory_fraction=0.6, last_active_at=2.0),
        }
        with mock.patch.dict(
            client_main._node_policy, {"ram_cache_limit_mb": None}, clear=False
        ), mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(
                client_main._containers, {k: object() for k in statuses}, clear=True
            ):
            plan = client_main._plan_eviction([0], 0.5, "new:9000")
        # reserved 1.2; evict LRU a (->0.6, still over with +0.5), then b (->0.0).
        assert plan["fits"] is True
        assert [s["key"] for s in plan["plan"]] == ["a:8000", "b:8001"]

    def test_nothing_eligible_sets_reason(self):
        statuses = {
            "legacy:8000": _meta(
                gpu_memory_fraction=0.9, warm=False, internal_port=None
            )
        }
        with mock.patch.dict(
            client_main._node_policy, {"warm_offload_enabled": True}, clear=False
        ), mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"legacy:8000": object()}, clear=True):
            plan = client_main._plan_eviction([0], 0.5, "new:9000")
        assert plan["fits"] is False
        assert plan["over_gpus"] == [0]
        assert plan["reason"]


class TestPlanEndpoint:
    @pytest.mark.anyio
    async def test_returns_plan_and_mutates_nothing(self):
        statuses = {"old:8000": _meta(gpu_memory_fraction=0.9, last_active_at=1.0)}
        with mock.patch.dict(
            client_main._node_policy,
            {"warm_offload_enabled": True, "ram_cache_limit_mb": None},
            clear=False,
        ), mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.dict(client_main._containers, {"old:8000": object()}, clear=True):
            before = dict(client_main._statuses["old:8000"])
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/plan",
                    json={
                        "model_name": "new",
                        "port": 9000,
                        "gpu_memory_fraction": 0.5,
                        "gpu_ids": [0],
                    },
                )
            after = dict(client_main._statuses["old:8000"])
        assert resp.status_code == 200
        body = resp.json()
        assert body["warm_enabled"] is True
        assert body["fits"] is True
        assert [(s["key"], s["tier"]) for s in body["plan"]] == [("old:8000", "ram")]
        assert before == after  # dry-run is read-only

    @pytest.mark.anyio
    async def test_warm_disabled_is_trivial(self):
        with mock.patch.dict(
            client_main._node_policy, {"warm_offload_enabled": False}, clear=False
        ), mock.patch.dict(client_main._statuses, {}, clear=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/deployments/plan",
                    json={
                        "model_name": "new",
                        "port": 9000,
                        "gpu_memory_fraction": 0.5,
                        "gpu_ids": [0],
                    },
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["warm_enabled"] is False
        assert body["fits"] is True
        assert body["plan"] == []


class TestTierSelection:
    def test_ram_unlimited(self):
        with mock.patch.dict(client_main._node_policy, {"ram_cache_limit_mb": None}):
            assert client_main._ram_limit_mb() is None

    @pytest.mark.anyio
    async def test_auto_tier_disk_when_over_budget(self):
        # Limit smaller than the model estimate -> can't fit RAM -> disk.
        with mock.patch.dict(
            client_main._node_policy, {"ram_cache_limit_mb": 100}, clear=False
        ), mock.patch.object(client_main, "_ram_estimate", return_value=5000.0), \
            mock.patch.dict(client_main._statuses, {}, clear=True):
            tier = await client_main._auto_tier(_meta())
        assert tier == "disk"

    @pytest.mark.anyio
    async def test_auto_tier_ram_when_fits(self):
        with mock.patch.dict(
            client_main._node_policy, {"ram_cache_limit_mb": 100000}, clear=False
        ), mock.patch.object(client_main, "_ram_estimate", return_value=5000.0), \
            mock.patch.dict(client_main._statuses, {}, clear=True):
            tier = await client_main._auto_tier(_meta())
        assert tier == "ram"

    @pytest.mark.anyio
    async def test_enforce_ram_budget_demotes_lru(self):
        statuses = {
            "a:8000": _meta(pause_tier="ram", paused_ram_mb=8000.0, last_active_at=1.0),
            "b:8001": _meta(port=8001, pause_tier="ram", paused_ram_mb=8000.0, last_active_at=2.0),
        }
        demoted = []

        async def fake_disk(meta, key):
            meta["pause_tier"] = "disk"
            meta["paused_ram_mb"] = 0
            demoted.append(key)

        with mock.patch.dict(
            client_main._node_policy, {"ram_cache_limit_mb": 20000}, clear=False
        ), mock.patch.dict(client_main._statuses, statuses, clear=True), \
            mock.patch.object(client_main, "_pause_to_disk", side_effect=fake_disk):
            await client_main._enforce_ram_budget(5000.0)
        # Used was 16 GB; adding 5 GB (21 GB) exceeds the 20 GB limit, so demote
        # just the LRU sleeper (a); afterwards 8+5 GB fits.
        assert demoted == ["a:8000"]


class TestPause:
    @pytest.mark.anyio
    async def test_pause_to_ram_sleeps_and_marks(self):
        meta = _meta()
        with mock.patch.object(client_main, "_vllm_sleep") as sleep, \
            mock.patch.object(client_main, "_ram_estimate", return_value=4096.0):
            await client_main._pause_to_ram(meta, "m:8000")
        sleep.assert_awaited_once()
        assert meta["pause_tier"] == "ram"
        assert meta["status"] == "paused_ram"
        assert meta["paused_ram_mb"] == 4096.0

    @pytest.mark.anyio
    async def test_pause_to_disk_removes_container(self):
        meta = _meta()
        container = object()
        with mock.patch.dict(client_main._statuses, {"m:8000": meta}, clear=True), \
            mock.patch.dict(client_main._containers, {"m:8000": container}, clear=True), \
            mock.patch.object(client_main, "_remove_container") as remove:
            await client_main._pause_to_disk(meta, "m:8000")
        remove.assert_called_once_with(container)
        assert meta["pause_tier"] == "disk"
        assert meta["status"] == "paused_disk"
        assert "m:8000" not in client_main._containers


class TestEnsureActive:
    @pytest.mark.anyio
    async def test_ram_resume_wakes_and_runs(self):
        meta = _meta(pause_tier="ram", status="paused_ram")
        with mock.patch.dict(client_main._statuses, {"m:8000": meta}, clear=True), \
            mock.patch.object(client_main, "_ensure_fit", return_value=(True, [])), \
            mock.patch.object(client_main, "_vllm_wake") as wake, \
            mock.patch.object(client_main, "_wait_awake", return_value=True):
            ok = await client_main._ensure_active("m:8000")
        assert ok is True
        wake.assert_awaited_once()
        assert meta["pause_tier"] is None
        assert meta["status"] == "running"

    @pytest.mark.anyio
    async def test_already_active_is_noop(self):
        meta = _meta(pause_tier=None, status="running")
        with mock.patch.dict(client_main._statuses, {"m:8000": meta}, clear=True):
            assert await client_main._ensure_active("m:8000") is True

    @pytest.mark.anyio
    async def test_returns_false_when_cannot_fit(self):
        meta = _meta(pause_tier="ram", status="paused_ram")
        with mock.patch.dict(client_main._statuses, {"m:8000": meta}, clear=True), \
            mock.patch.object(client_main, "_ensure_fit", return_value=(False, [])):
            assert await client_main._ensure_active("m:8000") is False
        assert meta["pause_tier"] == "ram"  # stays paused


def test_orphan_compile_caches(tmp_path):
    tracked_key = "m:8000"
    sub = client_main._cache_subdir(tracked_key)
    (tmp_path / sub).mkdir()
    (tmp_path / "orphan123").mkdir()
    (tmp_path / "orphan123" / "f").write_bytes(b"x" * 2048)
    with mock.patch.object(client_main, "_COMPILE_CACHE_DIR", tmp_path), \
        mock.patch.dict(client_main._statuses, {tracked_key: _meta()}, clear=True):
        orphans = client_main._orphan_compile_caches()
    names = {o["name"] for o in orphans}
    assert names == {"orphan123"}
    assert sub not in names


def test_ram_sleeper_candidates_flags_untracked(tmp_path):
    class P:
        def __init__(self, pid, name, rss):
            self.info = {"pid": pid, "name": name, "memory_info": mock.MagicMock(rss=rss)}

    procs = [
        P(111, "VLLM::EngineCore", 4 * 1024**3),   # rogue sleeper
        P(222, "postgres", 4 * 1024**3),           # not vLLM
        P(333, "VLLM::EngineCore", 10 * 1024**2),  # below RSS floor
    ]
    with mock.patch.object(client_main.psutil, "process_iter", return_value=procs), \
        mock.patch.object(client_main, "_tracked_gpu_pids", return_value={}), \
        mock.patch.object(client_main, "_looks_like_vllm", side_effect=lambda n, pid=None: "VLLM" in n):
        sleepers = client_main._ram_sleeper_candidates()
    assert [s["pid"] for s in sleepers] == [111]


@pytest.mark.anyio
async def test_config_endpoint_updates_policy_and_pins():
    statuses = {"a:8000": _meta(pinned=False), "b:8001": _meta(port=8001, pinned=True)}
    with mock.patch.dict(client_main._node_policy, {}, clear=False), \
        mock.patch.dict(client_main._statuses, statuses, clear=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/config",
                json={"warm_offload_enabled": True, "ram_cache_limit_mb": 20480, "pins": ["a:8000"]},
            )
        assert resp.status_code == 200
        assert client_main._node_policy["warm_offload_enabled"] is True
        assert client_main._node_policy["ram_cache_limit_mb"] == 20480
        assert client_main._statuses["a:8000"]["pinned"] is True
        assert client_main._statuses["b:8001"]["pinned"] is False


@pytest.mark.anyio
async def test_pause_endpoint_requires_warm(logs_dir):
    with mock.patch.dict(
        client_main._statuses, {"m:8000": _meta(warm=False)}, clear=True
    ), mock.patch.dict(client_main._containers, {"m:8000": object()}, clear=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/deployments/pause", json={"key": "m:8000"})
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_stop_deletes_compile_cache_and_proxy(logs_dir):
    container = mock.MagicMock()
    container.id = "cid"
    with mock.patch.dict(
        client_main._statuses, {"m:8000": _meta()}, clear=True
    ), mock.patch.dict(client_main._containers, {"m:8000": container}, clear=True), \
        mock.patch.object(client_main, "_remove_container"), \
        mock.patch.object(client_main, "_stop_proxy") as stop_proxy, \
        mock.patch.object(client_main, "_delete_compile_cache") as del_cache:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/deployments/stop", json={"key": "m:8000"})
    assert resp.status_code == 200
    del_cache.assert_called_once_with("m:8000")
    stop_proxy.assert_awaited_once_with("m:8000")


# ---------------------------------------------------------------------------
# Docker GPU pre-flight guard + no-GPU failure classification
# ---------------------------------------------------------------------------


class TestDockerGpuProbe:
    @staticmethod
    def _client_raising(exc):
        client = mock.MagicMock()
        client.containers.run.side_effect = exc
        return client

    def test_true_when_device_present(self):
        client = mock.MagicMock()  # containers.run returns normally
        with mock.patch.object(client_main, "_docker_gpu_probe_cache", (0.0, None)), \
            mock.patch.object(client_main, "_runtime_client", return_value=client):
            assert client_main._docker_gpu_probe() is True
            assert client_main._docker_gpu_error() is None

    def test_false_on_container_error(self):
        exc = client_main.ContainerError(object(), 1, ["test"], "alpine:3", b"")
        with mock.patch.object(client_main, "_docker_gpu_probe_cache", (0.0, None)), \
            mock.patch.object(
                client_main, "_runtime_client", return_value=self._client_raising(exc)
            ):
            assert client_main._docker_gpu_probe() is False
            msg = client_main._docker_gpu_error()
            assert msg and "nvidia-ctk" in msg

    def test_false_on_device_driver_apierror(self):
        exc = client_main.APIError(
            'could not select device driver "" with capabilities: [[gpu]]'
        )
        with mock.patch.object(client_main, "_docker_gpu_probe_cache", (0.0, None)), \
            mock.patch.object(
                client_main, "_runtime_client", return_value=self._client_raising(exc)
            ):
            assert client_main._docker_gpu_probe() is False

    def test_inconclusive_when_probe_image_missing(self):
        exc = client_main.ImageNotFound("alpine:3 not found")
        with mock.patch.object(client_main, "_docker_gpu_probe_cache", (0.0, None)), \
            mock.patch.object(
                client_main, "_runtime_client", return_value=self._client_raising(exc)
            ):
            # Inconclusive must NOT block deploys.
            assert client_main._docker_gpu_probe() is None
            assert client_main._docker_gpu_error() is None

    def test_result_is_cached(self):
        client = mock.MagicMock()
        with mock.patch.object(client_main, "_docker_gpu_probe_cache", (0.0, None)), \
            mock.patch.object(client_main, "_runtime_client", return_value=client):
            client_main._docker_gpu_probe()
            client_main._docker_gpu_probe()
        assert client.containers.run.call_count == 1  # second call hit the cache


@pytest.mark.anyio
async def test_start_rejected_when_docker_cannot_pass_gpus(logs_dir):
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["docker"]
    ), mock.patch.object(
        client_main, "_docker_gpu_error",
        return_value="Docker cannot pass GPUs ... sudo nvidia-ctk runtime configure ...",
    ), mock.patch.dict(client_main._statuses, {}, clear=True), \
        mock.patch.dict(client_main._containers, {}, clear=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/deployments/start",
                json={
                    "model_name": "org/model",
                    "port": 38911,
                    "gpu_memory_fraction": 0.5,
                    "vllm_version": "0.9.1",
                },
            )
    assert resp.status_code == 409
    assert "nvidia-ctk" in resp.json()["detail"]
    # Fail-fast: no provisional status, no container, port left free for retry.
    assert "org/model:38911" not in client_main._statuses
    assert "org/model:38911" not in client_main._containers


@pytest.mark.anyio
async def test_start_skip_resource_check_bypasses_gpu_guard(logs_dir):
    # The operator override must skip the GPU pre-flight entirely.
    probe = mock.MagicMock()
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["docker"]
    ), mock.patch.object(client_main, "_docker_gpu_error", probe), \
        mock.patch.object(client_main, "_ensure_image", side_effect=RuntimeError("stop here")), \
        mock.patch.dict(client_main._statuses, {}, clear=True), \
        mock.patch.dict(client_main._containers, {}, clear=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/deployments/start",
                json={
                    "model_name": "org/model",
                    "port": 38912,
                    "gpu_memory_fraction": 0.5,
                    "vllm_version": "0.9.1",
                    "skip_resource_check": True,
                },
            )
    probe.assert_not_called()


class TestClassifyNoGpu:
    def test_failed_to_infer_device_type(self):
        result = client_main._classify_failure(
            ["RuntimeError: Failed to infer device type, please set VLLM_LOGGING_LEVEL=DEBUG"]
        )
        assert result is not None
        code, msg = result
        assert code == "no_gpu"
        assert "nvidia-ctk" in msg

    def test_no_cuda_runtime(self):
        result = client_main._classify_failure(
            ["W0616 torch/utils/cpp_extension.py:140] No CUDA runtime is found, using CUDA_HOME=..."]
        )
        assert result and result[0] == "no_gpu"

    def test_could_not_select_device_driver(self):
        result = client_main._classify_failure(
            ['docker: could not select device driver "" with capabilities: [[gpu]].']
        )
        assert result and result[0] == "no_gpu"


# ---------------------------------------------------------------------------
# Daemon-flavor detection (Docker vs Podman behind a "docker" endpoint)
# ---------------------------------------------------------------------------


class TestDaemonFlavor:
    def test_detects_podman_via_components(self):
        client = mock.MagicMock()
        client.version.return_value = {
            "Components": [{"Name": "Podman Engine", "Version": "5.4.0"}]
        }
        assert client_main._daemon_flavor(client) == "podman"

    def test_detects_docker(self):
        client = mock.MagicMock()
        client.version.return_value = {
            "Components": [{"Name": "Engine", "Version": "27.0"}],
            "Platform": {"Name": "Docker Engine - Community"},
        }
        assert client_main._daemon_flavor(client) == "docker"

    def test_detects_podman_via_platform_name(self):
        client = mock.MagicMock()
        client.version.return_value = {"Platform": {"Name": "podman"}}
        assert client_main._daemon_flavor(client) == "podman"

    def test_defaults_docker_when_version_unavailable(self):
        client = mock.MagicMock()
        client.version.side_effect = RuntimeError("boom")
        assert client_main._daemon_flavor(client) == "docker"


class TestEffectiveRuntime:
    def test_docker_label_actually_podman(self):
        with mock.patch.object(client_main, "_runtime_client", return_value=mock.MagicMock()), \
            mock.patch.object(client_main, "_daemon_flavor", return_value="podman"):
            assert client_main._effective_runtime("docker") == "podman"

    def test_matching_label_passthrough(self):
        with mock.patch.object(client_main, "_runtime_client", return_value=mock.MagicMock()), \
            mock.patch.object(client_main, "_daemon_flavor", return_value="docker"):
            assert client_main._effective_runtime("docker") == "docker"

    def test_returns_label_when_client_unavailable(self):
        with mock.patch.object(
            client_main, "_runtime_client", side_effect=RuntimeError("down")
        ):
            assert client_main._effective_runtime("docker") == "docker"


@pytest.mark.anyio
async def test_start_uses_podman_gpu_path_when_docker_is_really_podman(logs_dir):
    # DOCKER_HOST-style misroute: the "docker" endpoint is actually Podman, so
    # the launch must use the Podman (CDI) GPU guard, not the Docker one.
    with mock.patch.object(
        client_main, "_available_runtimes", return_value=["docker"]
    ), mock.patch.object(
        client_main, "_effective_runtime", return_value="podman"
    ), mock.patch.object(
        client_main, "_podman_gpu_error",
        return_value="No NVIDIA CDI spec found ... nvidia-ctk cdi generate ...",
    ) as pod_err, mock.patch.object(
        client_main, "_docker_gpu_error"
    ) as dock_err, mock.patch.dict(
        client_main._statuses, {}, clear=True
    ), mock.patch.dict(client_main._containers, {}, clear=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/deployments/start",
                json={
                    "model_name": "org/model",
                    "port": 38913,
                    "gpu_memory_fraction": 0.5,
                    "vllm_version": "0.9.1",
                },
            )
    assert resp.status_code == 409
    assert "cdi" in resp.json()["detail"].lower()
    pod_err.assert_called_once()
    dock_err.assert_not_called()  # never use the Docker GPU form against Podman
