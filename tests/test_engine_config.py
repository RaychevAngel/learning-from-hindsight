"""Offline configuration boundary checks; no corpus or model loading."""

from pathlib import Path

import pytest

from hindsight.experiment import Experiment, read_json

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("overrides, expected", [
    ({}, None), ({"enable_thinking": None}, None),
    ({"enable_thinking": False}, False), ({"enable_thinking": True}, True),
])
def test_get_engine_forwards_thinking_without_changing_other_options(
    monkeypatch, overrides, expected
):
    experiment = Experiment.__new__(Experiment)
    experiment.config = {**read_json(CONFIGS / "initial.json"), **overrides}
    experiment.device, experiment.engine = "cpu", None
    experiment.check_time = lambda: None
    experiment.event = lambda *args, **kwargs: None
    calls = []
    sentinel = object()

    def constructor(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr("hindsight.model.HFEngine", constructor)
    assert experiment.get_engine() is sentinel
    assert experiment.get_engine() is sentinel
    assert calls == [{
        "model_name": experiment.config["model_name"], "revision": experiment.config["revision"],
        "device": "cpu", "max_generation_context": 1536, "max_teacher_context": 4096,
        "budget_check": experiment.check_time, "enable_thinking": expected,
    }]


def test_14b_calibration_config_changes_only_instrument_identity_and_thinking():
    original = read_json(CONFIGS / "initial.json")
    assert read_json(CONFIGS / "calibration-14b.json") == {
        **original, "model_name": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "enable_thinking": False,
    }


def test_generation_precheck_uses_the_same_explicit_thinking_option():
    from types import SimpleNamespace

    calls = []
    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a, **kw: calls.append(kw) or [1]),
        generate=lambda *a, **kw: ["result"],
    )
    experiment = Experiment.__new__(Experiment)
    experiment.config = {"max_context_tokens": 128, "enable_thinking": False}
    experiment.check_time = lambda: None
    experiment.get_engine = lambda: engine
    assert experiment.generate(["task"], max_new_tokens=4, sample=False, seed=1) == ["result"]
    assert calls == [{"tokenize": True, "add_generation_prompt": True,
                      "return_dict": False, "enable_thinking": False}]


def test_qwen35_config_keeps_the_bounded_comparison_and_original_budgets():
    original = read_json(CONFIGS / "initial.json")
    new = read_json(CONFIGS / "qwen35-4b.json")
    for name in ("source_tokens", "evaluation_tokens", "student_steps", "learning_rate",
                 "seed", "absolute_deadline_utc", "initial_cost_cap_usd"):
        assert new[name] == original[name]
    assert new["model_name"] == "Qwen/Qwen3.5-4B"
    assert new["enable_thinking"] is False
    assert len(new["diagnostic_source_ids"]) == len(set(new["diagnostic_source_ids"])) == 4
