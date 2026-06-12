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
                "prompt_tps": 1800.0,
                "generation_tps": 52.3,
                "prompt_throughput": 410.0,
                "generation_throughput": 12.5,
                "requests_running": 3,
                "requests_waiting": 1,
            },
        )
        assert sync.live_usage[1] == {
            "prompt_tps": 1800.0,
            "generation_tps": 52.3,
            "prompt_throughput": 410.0,
            "generation_throughput": 12.5,
            "requests_running": 3,
            "requests_waiting": 1,
        }

    def test_entry_removed_when_no_live_values(self):
        sync._update_live_usage(1, {"generation_tps": 10.0})
        sync._update_live_usage(1, {"prompt_tokens": 100})
        assert 1 not in sync.live_usage

    def test_non_numeric_live_values_ignored(self):
        sync._update_live_usage(1, {"generation_tps": "fast", "requests_running": 2})
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


def _snapshot(**overrides):
    base = {
        "prompt_tokens": 0.0,
        "generation_tokens": 0.0,
        "prefill_sum": 0.0,
        "decode_sum": 0.0,
        "ttft_sum": 0.0,
        "tpot_sum": 0.0,
        "tpot_count": 0.0,
    }
    base.update(overrides)
    return base


class TestComputeUsageRates:
    @pytest.fixture(autouse=True)
    def _clean_rate_state(self):
        from tests.test_failure_classification import client_main

        client_main._usage_snapshots.clear()
        yield
        client_main._usage_snapshots.clear()

    def _rates(self, monotonic_values, snapshots):
        from unittest import mock

        from tests.test_failure_classification import client_main

        results = []
        with mock.patch.object(
            client_main.time, "monotonic", side_effect=monotonic_values
        ):
            for snapshot in snapshots:
                results.append(client_main._compute_usage_rates("d-1", snapshot))
        return results

    def test_first_sample_has_no_rates(self):
        (rates,) = self._rates([100.0], [_snapshot(prompt_tokens=1000)])
        assert rates == {}

    def test_split_rates_from_time_histograms(self):
        first, second = self._rates(
            [100.0, 115.0],
            [
                _snapshot(),
                _snapshot(
                    # 3000 prompt tokens prefilled in 2s of processing time,
                    # 300 generated tokens over 6s of decode time — inside a
                    # 15s wall window. Idle time must not dilute the speeds.
                    prompt_tokens=3000.0,
                    generation_tokens=300.0,
                    prefill_sum=2.0,
                    decode_sum=6.0,
                ),
            ],
        )
        assert first == {}
        assert second["prompt_tps"] == 1500.0  # 3000 / 2s processing
        assert second["generation_tps"] == 50.0  # 300 / 6s decode
        assert second["prompt_throughput"] == 200.0  # 3000 / 15s wall
        assert second["generation_throughput"] == 20.0  # 300 / 15s wall

    def test_idle_window_yields_no_rates(self):
        active = _snapshot(prompt_tokens=1000.0, prefill_sum=1.0)
        _, _, idle = self._rates(
            [100.0, 115.0, 130.0], [_snapshot(), active, dict(active)]
        )
        assert idle == {}  # nothing moved: no stale speeds reported

    def test_counter_reset_skips_window(self):
        _, reset = self._rates(
            [100.0, 115.0],
            [_snapshot(prompt_tokens=5000.0, prefill_sum=3.0), _snapshot(prompt_tokens=10.0)],
        )
        assert reset == {}

    def test_ttft_tpot_fallback_for_older_engines(self):
        _, rates = self._rates(
            [100.0, 110.0],
            [
                _snapshot(),
                _snapshot(
                    prompt_tokens=2000.0,
                    generation_tokens=100.0,
                    ttft_sum=4.0,  # no prefill/decode sums exposed
                    tpot_sum=2.5,
                    tpot_count=100.0,
                ),
            ],
        )
        assert rates["prompt_tps"] == 500.0  # 2000 / 4s TTFT
        assert rates["generation_tps"] == 40.0  # 100 tokens / 2.5s TPOT

    def test_partial_availability(self):
        _, rates = self._rates(
            [100.0, 110.0],
            [
                _snapshot(),
                _snapshot(generation_tokens=100.0, decode_sum=2.0),
            ],
        )
        assert rates["generation_tps"] == 50.0
        assert "prompt_tps" not in rates
        assert "prompt_throughput" not in rates
        assert rates["generation_throughput"] == 10.0


class TestHistogramRegex:
    def test_matches_time_histograms_with_labels(self):
        from tests.test_failure_classification import client_main

        sample = (
            'vllm:request_prefill_time_seconds_sum{model_name="x"} 12.5\n'
            'vllm:request_prefill_time_seconds_count{model_name="x"} 40\n'
            'vllm:request_decode_time_seconds_sum{model_name="x"} 90.0\n'
            "vllm:time_to_first_token_seconds_sum 4.25\n"
            "vllm:time_per_output_token_seconds_sum 2.5\n"
            "vllm:time_per_output_token_seconds_count 100\n"
            'vllm:time_per_output_token_seconds_bucket{le="0.025"} 31\n'
        )
        hist = {}
        for line in sample.splitlines():
            match = client_main._VLLM_HIST_RE.match(line)
            if match:
                key = f"{match.group(1)}_{match.group(2)}"
                hist[key] = hist.get(key, 0.0) + float(match.group(3))
        assert hist == {
            "request_prefill_time_seconds_sum": 12.5,
            "request_prefill_time_seconds_count": 40.0,
            "request_decode_time_seconds_sum": 90.0,
            "time_to_first_token_seconds_sum": 4.25,
            "time_per_output_token_seconds_sum": 2.5,
            "time_per_output_token_seconds_count": 100.0,
        }
