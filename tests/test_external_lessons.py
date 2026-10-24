"""Exercise the diagnostic's real Experiment methods with an offline recording engine."""

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight.data import Episode, Task
from hindsight.experiment import read_json, rows, write_json
from hindsight.model import Generation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_external_lessons.py"
SPEC = importlib.util.spec_from_file_location("check_external_lessons", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


@pytest.fixture
def diagnostic_inputs(tmp_path, monkeypatch):
    sources, data = tmp_path / "prior", tmp_path / "data"
    lessons, output = tmp_path / "lessons.json", tmp_path / "diagnostic"
    manifest = {"fixture": "unchanged corpus"}
    write_json(data / "manifest.json", manifest)
    config = {
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "max_invocation_seconds": 60, "seed": 7301, "student_steps": 4,
        "learning_rate": 1e-4, "generation_batch_size": 8, "evaluation_tokens": 160,
        "max_context_tokens": 1536, "max_teacher_context_tokens": 4096,
    }
    write_json(sources / "identity.json", {"config": config, "dataset_manifest": manifest})
    episodes, lesson_rows = [], []
    for index in range(4):
        def task(role, number):
            return Task(f"{index}-{role}-{number}", f"{role} task {index}-{number}", (1,), "f")
        ep = Episode(f"ep-{index}", "development", "f", task("SOURCE", 0),
                     tuple(task("V", i) for i in range(4)),
                     tuple(task("Q", i) for i in range(8)))
        episodes.append(ep)
        generation = Generation(ep.source.prompt, [2, 4, index + 5], [9, 1],
                                "Presentation text deliberately differs from tokens", False)
        source_path = sources / "sources" / f"{ep.episode_id}.json"
        write_json(source_path, {"task_id": ep.source.task_id,
                                "generation": dataclasses.asdict(generation),
                                "feedback": "incorrect", "grade": {"success": False}})
        lesson_rows.append({"episode_id": ep.episode_id,
                            "source_sha256": DIAGNOSTIC.sha256(source_path.read_bytes()),
                            "lesson": f"Fixed source-only lesson {index}."})
    write_json(lessons, {"provenance": {"kind": "offline fixture"}, "rows": lesson_rows})
    monkeypatch.setattr("hindsight.data.load_corpus", lambda path: (episodes, []))
    return SimpleNamespace(sources=sources, data=data, lessons=lessons, output=output,
                           episodes=episodes)


class RecordingEngine:
    def __init__(self, output, fail_on_call=None, expected_count=144):
        self.output, self.fail_on_call = output, fail_on_call
        self.expected_count = expected_count
        self.counters = dict.fromkeys(DIAGNOSTIC.PLANNED_WORK, 0)
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [2])
        self.calls, self.scored, self.adapted, self.resets = [], [], [], []

    def generate(self, prompts, **kwargs):
        assert len(rows(self.output / "expected.jsonl")) == self.expected_count
        assert all("SOURCE" not in prompt for prompt in prompts)
        self.calls.append((prompts, kwargs))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("controlled interruption")
        self.counters["generation_sequences"] += len(prompts)
        return [Generation(prompt, [2], [9, 1], "FINAL: [1]", False) for prompt in prompts]

    def score_rollout(self, generation, feedback, **kwargs):
        self.scored.append((generation, feedback, kwargs))
        self.counters["scoring_forwards"] += 1
        self.target = object()
        return self.target

    def reset_student(self, seed):
        self.resets.append(seed)

    def adapt(self, generation, target, steps, lr, **kwargs):
        assert target is self.target and steps == 4 and lr == 1e-4
        self.adapted.append(generation)
        self.counters["student_optimizer_steps"] += steps
        return [{"optimizer_steps": 1, "loss": 0.1} for _ in range(steps)]


def invoke(inputs, engine):
    return DIAGNOSTIC.run(inputs.sources, inputs.lessons, inputs.data, inputs.output, engine=engine)


def test_fixed_plan_reuses_exact_sources_and_existing_experiment_methods(diagnostic_inputs):
    values = diagnostic_inputs
    engine = RecordingEngine(values.output)
    assert invoke(values, engine) == 0
    receipt = read_json(values.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["actual_observations"] == 144
    assert receipt["missing_observations"] == 0
    assert receipt["work"] == DIAGNOSTIC.PLANNED_WORK
    assert len(engine.resets) == 4 and len(set(engine.resets)) == 1
    assert len(receipt["source_provenance"]) == 4
    assert not (values.output / "selections.jsonl").exists()
    assert not (values.output / "proposals").exists()
    for index, (generation, feedback, kwargs) in enumerate(engine.scored):
        original = values.sources / "sources" / f"ep-{index}.json"
        assert (values.output / "sources" / original.name).read_bytes() == original.read_bytes()
        assert dataclasses.asdict(generation) == read_json(original)["generation"]
        assert engine.adapted[index] is generation
        assert feedback == "incorrect" and kwargs == {"pi": f"Fixed source-only lesson {index}."}
    observations = rows(values.output / "observations.jsonl")
    assert {row["partition"] for row in observations} == {"development"}
    assert {(row["condition"], row["candidate"]) for row in observations} == set(
        DIAGNOSTIC.CONDITIONS
    )
    for prompts, kwargs in engine.calls:
        assert kwargs["sample"] is False
        if kwargs["adapter"] == "student":
            assert all("hindsight_advice" not in prompt for prompt in prompts)
    before = (values.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(values, engine)
    assert (values.output / "receipt.json").read_bytes() == before


@pytest.mark.parametrize("defect", [
    "hash", "prompt", "final", "duplicate", "empty", "thought", "answer",
])
def test_bad_inputs_fail_before_any_model_work(diagnostic_inputs, defect):
    values = diagnostic_inputs
    lessons = read_json(values.lessons)
    first = lessons["rows"][0]
    if defect == "hash":
        first["source_sha256"] = "0" * 64
    elif defect == "prompt":
        path = values.sources / "sources" / "ep-0.json"
        source = read_json(path)
        source["generation"]["prompt"] = "A different public task"
        write_json(path, source)
        first["source_sha256"] = DIAGNOSTIC.sha256(path.read_bytes())
    elif defect == "final":
        values.episodes[0] = dataclasses.replace(values.episodes[0], partition="final_known")
    elif defect == "duplicate":
        lessons["rows"][1]["episode_id"] = first["episode_id"]
    elif defect == "empty":
        first["lesson"] = "  "
    elif defect == "thought":
        first["lesson"] = "<think>Unfinished reasoning"
    elif defect == "answer":
        first["lesson"] = "Use the rule. FINAL:\n[1, 0]"
    write_json(values.lessons, lessons)
    engine = RecordingEngine(values.output)
    assert invoke(values, engine) == 1
    receipt = read_json(values.output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "ValueError"
    assert not engine.calls and not engine.scored and not engine.resets
    assert all(value == 0 for value in receipt["work"].values())


def test_format_advice_is_preserved_verbatim(diagnostic_inputs):
    values = diagnostic_inputs
    lessons = read_json(values.lessons)
    lesson = "Preserve length. Write FINAL: followed by the complete JSON array."
    lessons["rows"][0]["lesson"] = lesson
    write_json(values.lessons, lessons)
    engine = RecordingEngine(values.output)
    assert invoke(values, engine) == 0
    assert engine.scored[0][2] == {"pi": lesson}
    assert (values.output / "lessons.json").read_bytes() == values.lessons.read_bytes()


def test_partial_failure_preserves_counts_plan_and_exception(diagnostic_inputs):
    values = diagnostic_inputs
    engine = RecordingEngine(values.output, fail_on_call=2)
    assert invoke(values, engine) == 1
    receipt = read_json(values.output / "receipt.json")
    assert receipt["error"] == {"type": "RuntimeError", "message": "controlled interruption"}
    assert receipt["actual_observations"] == receipt["work"]["generation_sequences"] == 4
    assert receipt["missing_observations"] == 140
    assert len(rows(values.output / "expected.jsonl")) == 144
    assert receipt["completed_episode_ids"] == []


@pytest.fixture
def balanced_inputs(diagnostic_inputs, monkeypatch):
    values = diagnostic_inputs
    episodes, lesson_rows = [], []
    for index, family in enumerate(("translation", "denoising", "size_recoloring", "gravity") * 2):
        def task(role, number):
            return Task(f"fresh-{index}-{role}-{number}", f"{role} task {index}-{number}",
                        (1,), family)
        ep = Episode(f"fresh-{index}", "development", family, task("SOURCE", 0),
                     tuple(task("V", n) for n in range(8)), tuple(task("Q", n) for n in range(8)))
        episodes.append(ep)
        source_path = values.sources / "sources" / f"{ep.episode_id}.json"
        generation = Generation(ep.source.prompt, [20, 40, 50 + index], [90, 1],
                                "Fresh native model attempt", False)
        write_json(source_path, {"task_id": ep.source.task_id,
                                "generation": dataclasses.asdict(generation),
                                "feedback": "incorrect", "grade": {"success": False}})
        lesson_rows.append({"episode_id": ep.episode_id,
                            "source_sha256": DIAGNOSTIC.sha256(source_path.read_bytes()),
                            "lesson": f"Source-only unchanged lesson {index}."})
    manifest = {"protocol_version": "balanced-development-v1",
                "version": "arc1d-balanced-development-v1", "episodes": 8, "tasks": 136,
                "calibration_tasks": 0, "partition_episodes": {"development": 8}}
    write_json(values.data / "manifest.json", manifest)
    parent = read_json(values.sources / "identity.json")
    parent["dataset_manifest"] = manifest
    parent["config"].update(
        diagnostic_source_ids=[ep.episode_id for ep in episodes],
        dataset_manifest_sha256=DIAGNOSTIC.sha256((values.data / "manifest.json").read_bytes()),
    )
    write_json(values.sources / "identity.json", parent)
    write_json(values.lessons, {"provenance": {"kind": "offline source-only"}, "rows": lesson_rows})
    values.episodes = episodes
    monkeypatch.setattr("hindsight.data.load_corpus", lambda path: (episodes, []))
    return values


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_balanced_opt_in_exact_work_and_frozen_batch_policy(balanced_inputs, batch_size):
    values = balanced_inputs
    engine = RecordingEngine(values.output, expected_count=384)
    assert DIAGNOSTIC.run(values.sources, values.lessons, values.data, values.output,
                          balanced_development=True, generation_batch_size=batch_size,
                          engine=engine) == 0
    receipt = read_json(values.output / "receipt.json")
    assert receipt["work"] == {**DIAGNOSTIC.PLANNED_WORK, "generation_sequences": 384,
                               "scoring_forwards": 8, "student_optimizer_steps": 32}
    assert receipt["actual_observations"] == 384 and receipt["missing_observations"] == 0
    assert receipt["generation_batch_size_override"] == batch_size
    effective = read_json(values.output / "identity.json")["config"]
    assert effective["generation_batch_size"] == batch_size
    assert read_json(values.sources / "identity.json")["config"]["generation_batch_size"] == 8
    assert all(len(prompts) == batch_size for prompts, _ in engine.calls)
    assert len(engine.scored) == len(engine.adapted) == len(engine.resets) == 8
    assert len(set(engine.resets)) == 1
    assert not (values.output / "selections.jsonl").exists()
    assert not (values.output / "proposals").exists()
    for index, (generation, feedback, kwargs) in enumerate(engine.scored):
        assert generation.prompt_ids == [20, 40, 50 + index]
        assert generation.response_ids == [90, 1]
        assert feedback == "incorrect"
        assert kwargs == {"pi": f"Source-only unchanged lesson {index}."}


@pytest.mark.parametrize("defect", [
    "no_opt_in", "pin", "protocol", "ids", "final", "family", "roles", "batch", "batch_bool",
])
def test_balanced_external_rejects_before_model_work(balanced_inputs, defect):
    values = balanced_inputs
    parent = read_json(values.sources / "identity.json")
    if defect == "pin":
        parent["config"]["dataset_manifest_sha256"] = "0" * 64
    elif defect == "protocol":
        parent["dataset_manifest"]["protocol_version"] = "wrong"
        write_json(values.data / "manifest.json", parent["dataset_manifest"])
        parent["config"]["dataset_manifest_sha256"] = DIAGNOSTIC.sha256(
            (values.data / "manifest.json").read_bytes()
        )
    elif defect == "ids":
        parent["config"]["diagnostic_source_ids"] = list(
            reversed(parent["config"]["diagnostic_source_ids"])
        )
    elif defect == "final":
        values.episodes[0] = dataclasses.replace(values.episodes[0], partition="final_known")
    elif defect == "family":
        values.episodes[0] = dataclasses.replace(values.episodes[0], family="other")
    elif defect == "roles":
        values.episodes[0] = dataclasses.replace(values.episodes[0], validation=())
    write_json(values.sources / "identity.json", parent)
    engine = RecordingEngine(values.output, expected_count=384)
    override = True if defect == "batch_bool" else 2 if defect == "batch" else None
    assert DIAGNOSTIC.run(values.sources, values.lessons, values.data, values.output,
                          balanced_development=defect != "no_opt_in",
                          generation_batch_size=override, engine=engine) == 1
    assert not engine.calls and not engine.scored and not engine.adapted
    assert not any(read_json(values.output / "receipt.json")["work"].values())
