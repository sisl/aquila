"""Tests for client-side failure classification, engine args, and GPU pre-checks."""

import sys
from pathlib import Path
from unittest import mock

import pytest

# The client and the host backend both ship a regular package named `app`, so
# whichever is imported first wins for the whole test session. Import the
# client's package in a clean slate, bind the names we need, then restore the
# previous `app` modules (the backend's) for later test modules.
_CLIENT_DIR = Path(__file__).resolve().parent.parent / "aquila" / "assets" / "client"
_client_path = str(_CLIENT_DIR)
if _client_path in sys.path:
    sys.path.remove(_client_path)
sys.path.insert(0, _client_path)

_saved_app_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == "app" or name.startswith("app.")
}
for _name in _saved_app_modules:
    del sys.modules[_name]
sys.modules["app.consul"] = mock.MagicMock()

import app.main as client_main  # noqa: E402

StartRequest = client_main.StartRequest
_classify_failure = client_main._classify_failure
_check_gpu_resources = client_main._check_gpu_resources
_engine_args_to_cli = client_main._engine_args_to_cli
_phase_for_line = client_main._phase_for_line
_restart_threshold = client_main._restart_threshold

for _name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
    del sys.modules[_name]
sys.modules.update(_saved_app_modules)


# ---------------------------------------------------------------------------
# _classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_gpu_oom(self):
        result = _classify_failure(["...", "torch.OutOfMemoryError: CUDA out of memory."])
        assert result is not None
        assert result[0] == "gpu_oom"

    def test_kv_cache(self):
        result = _classify_failure(
            ["ValueError: No available memory for the cache blocks."]
        )
        assert result is not None
        assert result[0] == "kv_cache_too_small"

    def test_hf_auth(self):
        result = _classify_failure(
            ["huggingface_hub.errors.GatedRepoError: 403 Client Error."]
        )
        assert result is not None
        assert result[0] == "hf_auth"

    def test_model_not_found(self):
        result = _classify_failure(
            ["huggingface_hub.errors.RepositoryNotFoundError: 404"]
        )
        assert result is not None
        assert result[0] == "model_not_found"

    def test_port_conflict(self):
        result = _classify_failure(["OSError: [Errno 98] address already in use"])
        assert result is not None
        assert result[0] == "port_conflict"

    def test_bad_args(self):
        result = _classify_failure(["api_server.py: error: argument --dtype: invalid choice"])
        assert result is not None
        assert result[0] == "bad_args"

    def test_unknown_returns_none(self):
        assert _classify_failure(["INFO: everything is fine"]) is None

    def test_only_tail_is_searched(self):
        lines = ["CUDA out of memory"] + ["noise"] * 400
        assert _classify_failure(lines) is None


# ---------------------------------------------------------------------------
# _phase_for_line
# ---------------------------------------------------------------------------


class TestPhaseForLine:
    def test_phases(self):
        assert _phase_for_line("Downloading model.safetensors: 42%") == "downloading"
        assert _phase_for_line("Loading weights took 12s") == "loading_weights"
        assert _phase_for_line("Capturing CUDA graph shapes") == "compiling"
        assert _phase_for_line("INFO: Application startup complete.") == "ready"
        assert _phase_for_line("some unrelated line") is None


# ---------------------------------------------------------------------------
# _engine_args_to_cli
# ---------------------------------------------------------------------------


class TestEngineArgsToCli:
    def test_empty(self):
        assert _engine_args_to_cli(None) == []
        assert _engine_args_to_cli({}) == []

    def test_value_flags(self):
        tokens = _engine_args_to_cli({"max_model_len": 4096, "dtype": "bfloat16"})
        assert tokens == ["--max-model-len", "4096", "--dtype", "bfloat16"]

    def test_boolean_flags(self):
        assert _engine_args_to_cli({"enforce_eager": True}) == ["--enforce-eager"]
        assert _engine_args_to_cli({"enforce_eager": False}) == []

    def test_unknown_keys_dropped(self):
        assert _engine_args_to_cli({"rm_rf": "/", "max_num_seqs": 8}) == [
            "--max-num-seqs",
            "8",
        ]

    def test_empty_values_dropped(self):
        assert _engine_args_to_cli({"dtype": "", "quantization": None}) == []


# ---------------------------------------------------------------------------
# _check_gpu_resources
# ---------------------------------------------------------------------------


def _gpu(index, total_mb=24576, used_mb=2048):
    return {
        "index": index,
        "name": "Test GPU",
        "source": "nvml",
        "utilization": 5,
        "memory_used_mb": used_mb,
        "memory_total_mb": total_mb,
    }


def _request(**overrides):
    payload = {
        "model_name": "test/model",
        "port": 8000,
        "gpu_memory_fraction": 0.5,
    }
    payload.update(overrides)
    return StartRequest(**payload)


class TestCheckGpuResources:
    def setup_method(self):
        self._statuses = mock.patch.dict(client_main._statuses, clear=True)
        self._containers = mock.patch.dict(client_main._containers, clear=True)
        self._statuses.start()
        self._containers.start()

    def teardown_method(self):
        self._statuses.stop()
        self._containers.stop()

    def test_fits(self):
        with mock.patch.object(client_main, "_gpu_metrics", return_value=[_gpu(0)]):
            assert _check_gpu_resources(_request()) is None

    def test_not_enough_free_memory(self):
        # 0.9 x 24 GiB needed but only ~4 GiB free.
        gpus = [_gpu(0, used_mb=24576 - 4096)]
        with mock.patch.object(client_main, "_gpu_metrics", return_value=gpus):
            reason = _check_gpu_resources(_request(gpu_memory_fraction=0.9))
        assert reason is not None
        assert "GPU 0" in reason
        assert "skip_resource_check" in reason

    def test_cumulative_overcommit(self):
        client_main._containers["other:8001"] = object()
        client_main._statuses["other:8001"] = {
            "gpu_memory_fraction": 0.6,
            "gpu_ids": [0],
        }
        with mock.patch.object(client_main, "_gpu_metrics", return_value=[_gpu(0)]):
            reason = _check_gpu_resources(_request(gpu_memory_fraction=0.5, gpu_ids=[0]))
        assert reason is not None
        assert "over-committed" in reason

    def test_unified_memory_skips(self):
        gpus = [{**_gpu(0), "source": "unified"}]
        with mock.patch.object(client_main, "_gpu_metrics", return_value=gpus):
            assert _check_gpu_resources(_request(gpu_memory_fraction=1.0)) is None

    def test_untracked_status_ignored(self):
        # Status entries without a live container don't count against capacity.
        client_main._statuses["dead:8002"] = {
            "gpu_memory_fraction": 0.9,
            "gpu_ids": [0],
        }
        with mock.patch.object(client_main, "_gpu_metrics", return_value=[_gpu(0)]):
            assert _check_gpu_resources(_request(gpu_memory_fraction=0.5)) is None


# ---------------------------------------------------------------------------
# _restart_threshold
# ---------------------------------------------------------------------------


class TestRestartThreshold:
    def test_default(self):
        with mock.patch.dict(client_main._statuses, clear=True):
            assert _restart_threshold("missing:8000") == client_main.settings.max_failed_restarts

    def test_override(self):
        with mock.patch.dict(
            client_main._statuses, {"m:8000": {"max_failed_restarts": 7}}, clear=True
        ):
            assert _restart_threshold("m:8000") == 7

    def test_invalid_override_falls_back(self):
        with mock.patch.dict(
            client_main._statuses, {"m:8000": {"max_failed_restarts": 0}}, clear=True
        ):
            assert _restart_threshold("m:8000") == client_main.settings.max_failed_restarts
