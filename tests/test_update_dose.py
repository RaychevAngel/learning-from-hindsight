"""Offline fixed-dose orchestration checks; the actual Experiment methods perform evaluation."""

import importlib.util
from pathlib import Path

import pytest
import torch
from test_external_lessons import DIAGNOSTIC as EXTERNAL
from test_external_lessons import RecordingEngine, diagnostic_inputs

from hindsight.experiment import read_json, rows, write_json
from hindsight.model import Generation, forward_kl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_update_dose.py"
SPEC = importlib.util.spec_from_file_location("check_update_dose", SCRIPT)
DOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOSE)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    values = diagnostic_inputs.__wrapped__(tmp_path, monkeypatch)
    reference_engine = RecordingEngine(values.output)
    assert EXTERNAL.run(values.sources, values.lessons, values.data, values.output,
                        engine=reference_engine) == 0
    values.reference = values.output
    values.output = tmp_path / "dose"
    return values


class DoseEngine(RecordingEngine):
    def generate(self, prompts, **kwargs):
        assert len(rows(self.output / "expected.jsonl")) == 48
        assert all(prompt.startswith("V task") for prompt in prompts)
        assert not kwargs["sample"] and kwargs["adapter"] == "student"
        self.calls.append((prompts, kwargs))
        if len(self.calls) == self.fail_on_call:
            raise TimeoutError("controlled deadline")
        self.counters["generation_sequences"] += len(prompts)
        return [Generation(prompt, [2], [9, 1], "FINAL: [1]", False) for prompt in prompts]

    def score_rollout(self, generation, feedback, **kwargs):
        self.scored.append((generation, feedback, kwargs))
        self.counters["scoring_forwards"] += 1
        if kwargs.get("adapter") == "student":
            assert "pi" not in kwargs
            return torch.tensor([[0.6, 0.4]] * len(generation.response_ids)).log()
        assert "pi" in kwargs
        self.target = torch.tensor([[0.7, 0.3]] * len(generation.response_ids)).log()
        return self.target

    def adapt(self, generation, target, steps, lr, **kwargs):
        assert target is self.target
        assert {"steps": steps, "lr": lr} in [
            {"steps": arm["steps"], "lr": arm["lr"]} for arm in DOSE.ARMS
        ]
        self.adapted.append((generation, steps, lr))
        self.counters["student_optimizer_steps"] += steps
        return [{"optimizer_steps": 1, "loss": 0.1} for _ in range(steps)]


def invoke(values, engine):
    return DOSE.run(values.sources, values.lessons, values.data, values.reference,
                    values.output, engine=engine)


def test_exact_new_work_v_only_original_sources_and_post_update_kl(inputs):
    engine = DoseEngine(inputs.output)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["work"] == DOSE.WORK
    assert receipt["actual_observations"] == 48 and receipt["missing_observations"] == 0
    assert len(engine.resets) == 12 and len(set(engine.resets)) == 1
    assert {row["role"] for row in rows(inputs.output / "observations.jsonl")} == {"V"}
    assert {row["role"] for row in read_json(inputs.output / "reference-v.json")} == {"V"}
    assert len(read_json(inputs.output / "reference-v.json")) == 48
    events = [row for row in rows(inputs.output / "events.jsonl")
              if row["event"] == "student_adapted"]
    expected_kl = forward_kl(torch.tensor([[0.6, 0.4]]).log(),
                             torch.tensor([[0.7, 0.3]]).log()).item()
    assert len(events) == 12
    assert all(row["post_update_source_kl_target_to_student"] == pytest.approx(expected_kl)
               for row in events)
    for generation, _, _ in engine.adapted:
        source_id = generation.prompt_ids[-1] - 5
        original = read_json(inputs.sources / "sources" / f"ep-{source_id}.json")["generation"]
        assert generation.prompt_ids == original["prompt_ids"]
        assert generation.response_ids == original["response_ids"]
    assert not (inputs.output / "selections.jsonl").exists()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)


@pytest.mark.parametrize("defect", ["model", "source", "lesson", "core_code", "incomplete"])
def test_reference_identity_mismatch_rejected_before_compute(inputs, defect):
    if defect in ("model", "core_code"):
        path = inputs.reference / "identity.json"
        value = read_json(path)
        if defect == "model":
            value["config"]["model_name"] = "changed-model"
        else:
            value["code"]["model.py"] = "changed-code"
        write_json(path, value)
    elif defect == "source":
        path = inputs.reference / "sources" / "ep-0.json"
        value = read_json(path)
        value["generation"]["response_ids"] = [8, 1]
        write_json(path, value)
    elif defect == "lesson":
        value = read_json(inputs.lessons)
        value["rows"][0]["lesson"] = "Another lesson."
        write_json(inputs.lessons, value)
    else:
        path = inputs.reference / "receipt.json"
        value = read_json(path)
        value["status"] = "failed"
        write_json(path, value)
    engine = DoseEngine(inputs.output)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "ValueError"
    assert not engine.calls and not engine.scored and not engine.resets


def test_partial_failure_records_actual_work_and_no_hidden_retries(inputs):
    engine = DoseEngine(inputs.output, fail_on_call=2)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["error"] == {"type": "TimeoutError", "message": "controlled deadline"}
    assert len(engine.calls) == 2 and receipt["work"]["generation_sequences"] == 4
    assert receipt["work"]["student_optimizer_steps"] == 5
    assert receipt["work"]["scoring_forwards"] == 3
    assert receipt["actual_observations"] == 4 and receipt["missing_observations"] == 44
