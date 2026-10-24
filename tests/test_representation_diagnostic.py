"""Offline calibration-only checks using the real lossless representation and grader."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight.data import Task, grade
from hindsight.experiment import append_json, read_json, seed_for, write_json
from hindsight.model import MODEL_NAME, MODEL_REVISION, Generation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_representation.py"
SPEC = importlib.util.spec_from_file_location("check_representation", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    baseline, data, output = tmp_path / "baseline", tmp_path / "data", tmp_path / "rle.json"
    manifest = {"fixture": "unchanged sixteen calibration tasks"}
    config = {
        "model_name": MODEL_NAME, "revision": MODEL_REVISION, "source_tokens": 128,
        "seed": 7301, "max_context_tokens": 1536, "max_teacher_context_tokens": 4096,
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00", "max_invocation_seconds": 60,
    }
    write_json(data / "manifest.json", manifest)
    write_json(baseline / "identity.json", {"config": config, "dataset_manifest": manifest})
    prompt = "Infer the common transformation and apply it to the query.\n" + "\n".join(
        f"Example {i} input: [0, 1, 1]\nExample {i} output: [1, 1, 0]" for i in range(1, 5)
    ) + "\nQuery input: [0, 1, 1]\n\n" + "\n".join([
        "Respond in exactly two lines.",
        "First line: one rule sentence of at most 12 words.",
        "Second line: FINAL: followed by the complete output grid as a JSON array.",
        "Do not restate demonstrations, show steps, or add any other text.",
    ])
    tasks = [Task(f"cal-{i}", prompt, (1, 1, 0), "fixture") for i in range(16)]
    write_json(baseline / "calibration-expected.json",
               [{"task_id": task.task_id, "family": task.family} for task in tasks])
    for task in tasks:
        text = "Move the block left.\nFINAL: [1, 1, 0]"
        append_json(baseline / "calibration.jsonl", {
            "task_id": task.task_id, "family": task.family, **grade(text, task),
            "text": text, "cap_hit": False, "generated_tokens": 20,
        })
    monkeypatch.setattr(CHECK, "load_corpus", lambda path: ([object()], tasks))
    return SimpleNamespace(baseline=baseline, data=data, output=output, tasks=tasks, config=config)


class Engine:
    def __init__(self, values, fail_on=None, invalid=False):
        self.values, self.fail_on, self.invalid = values, fail_on, invalid
        self.calls = []
        self.counters = dict.fromkeys(CHECK.WORK, 0)
        self.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [2] * 20,
            encode=lambda text, **kwargs: [2] * len(text.split()),
        )

    def generate(self, prompts, **kwargs):
        plan = read_json(self.values.output)
        assert len(plan["expected"]) == 16
        task = self.values.tasks[len(self.calls)]
        assert kwargs == {"max_new_tokens": 128, "sample": True, "adapter": None,
                          "seed": seed_for(7301, task.task_id, "calibration")}
        assert prompts == [CHECK.representation.render_rle_prompt(task.prompt)[0]]
        self.calls.append((prompts, kwargs))
        if len(self.calls) == self.fail_on:
            raise TimeoutError("controlled calibration deadline")
        self.counters["generation_sequences"] += 1
        text = "Rule.\nFINAL: [[1, 2], [0, 1]]" if not self.invalid else "FINAL: [[1, 0]]"
        return [Generation(prompts[0], [2, 3], [4, 1], text, False)]


def invoke(values, engine):
    return CHECK.run(values.baseline, values.data, values.output, engine=engine)


def test_original_policy_and_exact_calibration_tasks_with_no_updates(inputs):
    originals = [(task.task_id, task.prompt, task.answer) for task in inputs.tasks]
    engine = Engine(inputs)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output)
    assert receipt["status"] == "completed" and receipt["work"] == CHECK.WORK
    assert receipt["actual_observations"] == 16 and receipt["missing_observations"] == 0
    assert receipt["summary"]["raw_successes"] == receipt["summary"]["rle_successes"] == 16
    assert [(task.task_id, task.prompt, task.answer) for task in inputs.tasks] == originals
    for row in receipt["rows"]:
        assert row["decoded_answer"] == [1, 1, 0]
        assert row["rle_grade"]["success"] and row["rle_grade"]["rle_format_ok"]
        assert row["generation"]["response_ids"] == [4, 1]
        assert row["original_prompt_sha256"] != row["rle_prompt_sha256"]
    assert "unavailable" in receipt["baseline_generation_ids"]
    before = inputs.output.read_bytes()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)
    assert inputs.output.read_bytes() == before


def test_malformed_rle_is_recorded_as_failure_not_retried(inputs):
    engine = Engine(inputs, invalid=True)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output)
    assert len(engine.calls) == 16
    assert receipt["summary"]["rle_successes"] == 0
    assert receipt["summary"]["rle_format_failures"] == 16
    assert all(row["decoded_answer"] is None for row in receipt["rows"])


@pytest.mark.parametrize("defect", ["ids", "manifest", "grade"])
def test_changed_baseline_or_oracle_rejected_before_generation(inputs, defect):
    if defect == "ids":
        write_json(inputs.baseline / "calibration-expected.json", [])
    elif defect == "manifest":
        write_json(inputs.data / "manifest.json", {"changed": True})
    else:
        inputs.tasks[0] = Task(
            inputs.tasks[0].task_id, inputs.tasks[0].prompt, (0, 0, 0), "fixture"
        )
    engine = Engine(inputs)
    assert invoke(inputs, engine) == 1
    assert not engine.calls
    assert read_json(inputs.output)["error"]["type"] == "ValueError"


def test_partial_failure_preserves_per_task_receipt_and_counts(inputs):
    engine = Engine(inputs, fail_on=3)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output)
    assert receipt["work"]["generation_sequences"] == receipt["actual_observations"] == 2
    assert receipt["missing_observations"] == 14
    assert receipt["error"]["type"] == "TimeoutError" and len(engine.calls) == 3


def test_reference_output_overflow_fails_before_any_generation(inputs):
    engine = Engine(inputs)
    engine.tokenizer.encode = lambda *args, **kwargs: [2] * 129
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output)
    assert not engine.calls and receipt["work"]["generation_sequences"] == 0
    assert len(receipt["token_preflight"]) == 16
    assert "token cap" in receipt["error"]["message"]
