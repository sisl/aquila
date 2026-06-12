"""Tests for the OpenAI-compatible gateway (model routing + error shape)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Add the host backend app to sys.path.
_BACKEND_DIR = (
    Path(__file__).resolve().parent.parent
    / "vllm_cluster_manager"
    / "assets"
    / "host"
    / "backend"
)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.api.gateway import (
    _match_deployment,
    _model_aliases,
    _openai_error,
)


def _dep(model_name, served=None, lora=None, **overrides):
    base = dict(
        id=1,
        model_name=model_name,
        engine_args={"served_model_name": served} if served else {},
        lora_modules=[{"name": n, "path": "p"} for n in (lora or [])],
        status="running",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestModelMatching:
    def test_served_name_beats_model_name(self):
        by_served = _dep("org/model-a", served="alias", id=1)
        by_model = _dep("alias", id=2)
        assert _match_deployment([by_model, by_served], "alias").id == 1

    def test_model_name_match(self):
        d = _dep("org/model-a")
        assert _match_deployment([d], "org/model-a") is d

    def test_lora_match_is_last_resort(self):
        with_lora = _dep("base", lora=["adapter"], id=1)
        direct = _dep("adapter", id=2)
        assert _match_deployment([with_lora, direct], "adapter").id == 2
        assert _match_deployment([with_lora], "adapter").id == 1

    def test_no_match(self):
        assert _match_deployment([_dep("a")], "b") is None

    def test_aliases_order_and_dedup(self):
        d = _dep("base", served="alias", lora=["l1", "alias"])
        assert _model_aliases(d) == ["alias", "base", "l1"]


class TestOpenAIError:
    def test_shape(self):
        resp = _openai_error(404, "nope", "invalid_request_error", "model_not_found")
        assert resp.status_code == 404
        payload = json.loads(resp.body)
        assert payload["error"]["message"] == "nope"
        assert payload["error"]["code"] == "model_not_found"


