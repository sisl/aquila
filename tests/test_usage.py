"""Tests for usage accounting (counter accumulation + Prometheus parsing)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = (
    Path(__file__).resolve().parent.parent
    / "vllm_cluster_manager"
    / "assets"
    / "host"
    / "backend"
)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import app.services.sync as sync


def _dep(dep_id=1):
    return SimpleNamespace(
        id=dep_id,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_requests=0,
    )


@pytest.fixture(autouse=True)
def _clean_state():
    sync._usage_last_seen.clear()
    sync.live_usage.clear()
    yield
    sync._usage_last_seen.clear()
    sync.live_usage.clear()


class TestAccumulateUsage:
    def test_first_sample_counts_fully(self):
        dep = _dep()
        sync._accumulate_usage(
            dep, {"prompt_tokens": 100, "generation_tokens": 50, "requests": 5}
        )
        assert dep.total_prompt_tokens == 100
        assert dep.total_completion_tokens == 50
        assert dep.total_requests == 5

    def test_delta_accumulation(self):
        dep = _dep()
        sync._accumulate_usage(
            dep, {"prompt_tokens": 100, "generation_tokens": 50, "requests": 5}
        )
        sync._accumulate_usage(
            dep, {"prompt_tokens": 160, "generation_tokens": 80, "requests": 8}
        )
        assert dep.total_prompt_tokens == 160
        assert dep.total_completion_tokens == 80
        assert dep.total_requests == 8

    def test_counter_reset_counts_new_value(self):
        # Container restarted: counters drop; the new value is the delta.
        dep = _dep()
        sync._accumulate_usage(
            dep, {"prompt_tokens": 100, "generation_tokens": 50, "requests": 5}
        )
        sync._accumulate_usage(
            dep, {"prompt_tokens": 20, "generation_tokens": 10, "requests": 1}
        )
        assert dep.total_prompt_tokens == 120
        assert dep.total_completion_tokens == 60
        assert dep.total_requests == 6

    def test_unchanged_counters_add_nothing(self):
        dep = _dep()
        snapshot = {"prompt_tokens": 100, "generation_tokens": 50, "requests": 5}
        sync._accumulate_usage(dep, snapshot)
        sync._accumulate_usage(dep, dict(snapshot))
        assert dep.total_prompt_tokens == 100

    def test_non_numeric_values_ignored(self):
        dep = _dep()
        sync._accumulate_usage(dep, {"prompt_tokens": "nan", "requests": 3})
        assert dep.total_prompt_tokens == 0
        assert dep.total_requests == 3

    def test_independent_deployments(self):
        a, b = _dep(1), _dep(2)
        sync._accumulate_usage(a, {"prompt_tokens": 100})
        sync._accumulate_usage(b, {"prompt_tokens": 7})
        assert a.total_prompt_tokens == 100
        assert b.total_prompt_tokens == 7


class TestLiveUsage:
    def test_live_values_stored_for_running_metrics(self):
        sync._update_live_usage(
            1,
            {
                "prompt_tokens": 100,
                "tokens_per_second": 42.5,
                "requests_running": 3,
                "requests_waiting": 1,
            },
        )
        assert sync.live_usage[1] == {
            "tokens_per_second": 42.5,
            "requests_running": 3,
            "requests_waiting": 1,
        }

    def test_entry_removed_when_no_live_values(self):
        sync._update_live_usage(1, {"tokens_per_second": 10.0})
        sync._update_live_usage(1, {"prompt_tokens": 100})
        assert 1 not in sync.live_usage

    def test_non_numeric_live_values_ignored(self):
        sync._update_live_usage(1, {"tokens_per_second": "fast", "requests_running": 2})
        assert sync.live_usage[1] == {"requests_running": 2}


# Captured shape of vLLM's /metrics output (labels vary by engine version).
_PROM_SAMPLE = """\
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 12345.0
vllm:prompt_tokens_total{model_name="my-lora"} 100.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 6789.0
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 41.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2.0
# unrelated metric
python_gc_objects_collected_total{generation="0"} 9999.0
"""


class TestPrometheusRegex:
    def _parse(self, text):
        # Mirror the client-side summing logic against the shared regex.
        from tests.test_failure_classification import client_main

        totals = {
            "prompt_tokens_total": 0.0,
            "generation_tokens_total": 0.0,
            "request_success_total": 0.0,
        }
        matched = False
        for line in text.splitlines():
            match = client_main._VLLM_COUNTER_RE.match(line)
            if not match:
                continue
            totals[match.group(1)] += float(match.group(2))
            matched = True
        return totals if matched else None

    def test_sums_across_label_sets(self):
        totals = self._parse(_PROM_SAMPLE)
        assert totals["prompt_tokens_total"] == 12445.0
        assert totals["generation_tokens_total"] == 6789.0
        assert totals["request_success_total"] == 42.0

    def test_no_vllm_metrics_returns_none(self):
        assert self._parse("python_gc_objects_collected_total 1\n") is None

    def test_gauge_regex_matches_queue_depths(self):
        from tests.test_failure_classification import client_main

        gauges = {}
        for line in _PROM_SAMPLE.splitlines():
            match = client_main._VLLM_GAUGE_RE.match(line)
            if match:
                gauges[match.group(1)] = gauges.get(match.group(1), 0.0) + float(
                    match.group(2)
                )
        assert gauges == {"num_requests_running": 3.0, "num_requests_waiting": 2.0}


class TestTokensPerSecond:
    @pytest.fixture(autouse=True)
    def _clean_rate_state(self):
        from tests.test_failure_classification import client_main

        client_main._usage_rate_state.clear()
        yield
        client_main._usage_rate_state.clear()

    def test_first_sample_has_no_rate(self):
        from tests.test_failure_classification import client_main

        assert client_main._tokens_per_second("d-1", 1000) is None

    def test_rate_from_consecutive_samples(self):
        from tests.test_failure_classification import client_main
        from unittest import mock

        with mock.patch.object(client_main.time, "monotonic", side_effect=[100.0, 110.0]):
            assert client_main._tokens_per_second("d-1", 1000) is None
            assert client_main._tokens_per_second("d-1", 1500) == 50.0

    def test_counter_reset_yields_no_rate(self):
        from tests.test_failure_classification import client_main
        from unittest import mock

        with mock.patch.object(client_main.time, "monotonic", side_effect=[100.0, 110.0]):
            client_main._tokens_per_second("d-1", 1000)
            assert client_main._tokens_per_second("d-1", 200) is None
