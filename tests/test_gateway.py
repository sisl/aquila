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
    _choose,
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
        outcome, dep = _choose([by_model, by_served], "alias")
        assert outcome == "ok" and dep.id == 1

    def test_model_name_match(self):
        d = _dep("org/model-a")
        assert _choose([d], "org/model-a") == ("ok", d)

    def test_lora_match_is_last_resort(self):
        with_lora = _dep("base", lora=["adapter"], id=1)
        direct = _dep("adapter", id=2)
        outcome, dep = _choose([with_lora, direct], "adapter")
        assert outcome == "ok" and dep.id == 2
        outcome, dep = _choose([with_lora], "adapter")
        assert outcome == "ok" and dep.id == 1

    def test_no_match(self):
        assert _choose([_dep("a")], "b") == ("none", None)

    def test_shared_model_name_is_ambiguous(self):
        # Same base model served twice under distinct served names: the bare
        # model name is ambiguous and reports the served names to use.
        a = _dep("org/model", served="alpha", id=1)
        b = _dep("org/model", served="beta", id=2)
        outcome, names = _choose([a, b], "org/model")
        assert outcome == "ambiguous"
        assert names == ["alpha", "beta"]

    def test_served_name_unique_even_when_model_name_shared(self):
        # Addressing by the (unique) served name stays deterministic.
        a = _dep("org/model", served="alpha", id=1)
        b = _dep("org/model", served="beta", id=2)
        assert _choose([a, b], "alpha")[1].id == 1
        assert _choose([a, b], "beta")[1].id == 2

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


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return self._items


class _FakeSession:
    def __init__(self, deployments):
        self._deployments = deployments

    async def execute(self, *args, **kwargs):
        return _FakeResult(self._deployments)


class TestResolveAmbiguous:
    def test_shared_model_name_returns_model_ambiguous(self):
        import asyncio

        from app.api.gateway import _resolve

        a = _dep("org/model", served="alpha", id=1, node_id=1)
        b = _dep("org/model", served="beta", id=2, node_id=1)
        resp = asyncio.run(_resolve(_FakeSession([a, b]), "org/model"))
        assert resp.status_code == 400
        payload = json.loads(resp.body)
        assert payload["error"]["code"] == "model_ambiguous"
        assert "alpha" in payload["error"]["message"]
        assert "beta" in payload["error"]["message"]


