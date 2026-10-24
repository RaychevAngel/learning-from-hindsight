"""Frozen fit/evaluation separation and exact work using a tiny recording engine."""

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hindsight.data import Episode, Task
from hindsight.experiment import Experiment, read_json, rows, write_json
from hindsight.model import Generation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_hindsight_teacher.py"
SPEC = importlib.util.spec_from_file_location("check_hindsight_teacher", SCRIPT)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    fit_run, sources = tmp_path / "fit-run", tmp_path / "sources"
    fit_data, data = tmp_path / "fit-data", tmp_path / "data"
    lessons, output = tmp_path / "lessons.json", tmp_path / "output"
    corpus = {}
    for kind, root, dataset, lesson_file in (
        ("fit", fit_run, fit_data, fit_run / "lessons.json"),
        ("eval", sources, data, lessons),
    ):
        episodes, lesson_rows = [], []
        for index, family in enumerate(D.FAMILIES * 2):
            def task(role, number):
                name = f"{kind}-{index}-{role}-{number}"
                return Task(name, name, (1,), family)
            ep = Episode(f"{kind}-{index}", "development", family, task("SOURCE", 0),
                         tuple(task("V", n) for n in range(8)),
                         tuple(task("Q", n) for n in range(8)))
            episodes.append(ep)
            generation = Generation(ep.source.prompt, [3, 2], [2, 1], "attempt", False)
            path = root / "sources" / f"{ep.episode_id}.json"
            write_json(path, dict(task_id=ep.source.task_id,
                                 generation=dataclasses.asdict(generation),
                                 feedback="incorrect"))
            lesson_rows.append(dict(episode_id=ep.episode_id, source_sha256=D.sha(path),
                                    lesson=f"Source lesson {kind}-{index}"))
        manifest = dict(protocol_version="balanced-development-v1", kind=kind)
        write_json(dataset / "manifest.json", manifest)
        config = dict(model_name="pinned", revision="fixed", enable_thinking=False,
                      deterministic_execution=True, seed=7301, source_tokens=128,
                      evaluation_tokens=160, student_steps=4, learning_rate=1e-4,
                      max_context_tokens=1536, max_teacher_context_tokens=4096,
                      absolute_deadline_utc="2099-01-01T00:00:00+00:00",
                      max_invocation_seconds=3900, generation_batch_size=1,
                      dataset_manifest_sha256=D.sha(dataset / "manifest.json"),
                      diagnostic_source_ids=[ep.episode_id for ep in episodes])
        write_json(root / "identity.json", dict(config=config, dataset_manifest=manifest,
                                                code=Experiment.code_identity(None)))
        write_json(lesson_file, dict(provenance={"source_only": True}, rows=lesson_rows))
        corpus[dataset] = episodes
    observed = []
    for index, ep in enumerate(corpus[fit_data]):
        for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
            for condition, candidate in (("baseline", None), ("direct_pi", 0), ("adapted_pi", 0)):
                observed.extend(dict(episode_id=ep.episode_id, partition=ep.partition,
                                     family=ep.family, role=role, condition=condition,
                                     candidate=candidate, task_id=task.task_id,
                                     success=condition == "adapted_pi" and index < 6)
                                for task in tasks)
    from hindsight.experiment import append_json
    for row in observed:
        append_json(fit_run / "observations.jsonl", row)
        append_json(fit_run / "expected.jsonl", row)
    write_json(fit_run / "receipt.json", dict(status="completed", actual_observations=384,
                                             lessons_sha256=D.sha(fit_run / "lessons.json")))
    write_json(sources / "receipt.json", dict(status="completed", actual_sources=8))
    monkeypatch.setattr(D, "load_corpus", lambda path: (corpus[path], []))
    monkeypatch.setattr("hindsight.data.load_corpus", lambda path: (corpus[path], []))
    monkeypatch.setattr(D, "canonical_inputs", lambda path: {str(path)})
    return SimpleNamespace(fit_run=fit_run, fit_data=fit_data, sources=sources, lessons=lessons,
                           data=data, output=output, corpus=corpus)


class Engine:
    def __init__(self, output, fail=False):
        self.output, self.fail = output, fail
        self.counters = dict.fromkeys(D.PLANNED_WORK, 0)
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1])
        self.calls, self.adaptations, self.resets, self.fits = [], [], [], []
        self.distribution = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 2).log()

    def generate(self, prompts, adapter=None, **kwargs):
        assert len(rows(self.output / "expected.jsonl")) == 512
        assert len(prompts) == 1
        assert "hindsight_advice" not in prompts[0] and "SOURCE" not in prompts[0]
        self.calls.append((prompts, adapter))
        if self.fail:
            raise RuntimeError("bounded failure")
        self.counters["generation_sequences"] += 1
        answer = 1 if adapter == "student" else 0
        return [Generation(prompts[0], [1], [2, 1], f"FINAL: [{answer}]", False)]

    def score_rollout(self, generation, feedback, **kwargs):
        assert len(rows(self.output / "expected.jsonl")) == 512
        self.counters["scoring_forwards"] += 1
        return self.distribution.clone()

    def reset_student(self, seed):
        self.resets.append(seed)

    def adapt(self, generation, target, steps, lr, initial_logprobs_callback=None):
        assert steps == 4 and lr == 1e-4
        self.adaptations.append((generation, target.clone()))
        self.counters["student_optimizer_steps"] += steps
        if initial_logprobs_callback:
            initial_logprobs_callback(self.distribution.clone())
        return [{"loss": 0.1}] * steps

    def fit_teacher(self, examples, **kwargs):
        assert len(examples) == 8 and kwargs["epochs"] == 32 and kwargs["batch_size"] == 4
        self.fits.append(kwargs)
        self.counters["teacher_optimizer_steps"] += 64
        self.counters["teacher_example_exposures"] += 256
        return dict(optimizer_steps=64, example_exposures=256)

    def save_teacher(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tiny": self.distribution}, path)


def qualify(engine, generation, feedback, target, output):
    output.mkdir(parents=True, exist_ok=False)
    for key, count in D.QUALIFICATION_WORK.items():
        engine.counters[key] += count
    write_json(output / "qualification.json", {"status": "passed"})
    return {"status": "passed_teacher_qualification"}


def invoke(values, engine):
    return D.run(values.fit_run, values.fit_data, values.sources, values.lessons, values.data,
                 values.output, engine=engine, qualifier=qualify)


def test_frozen_plan_exact_counts_resets_neutrality_and_all_controls(inputs):
    engine = Engine(inputs.output)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["work"] == D.PLANNED_WORK
    assert receipt["status"] == "completed" and receipt["actual_observations"] == 512
    assert len(receipt["completed_episode_ids"]) == 8
    assert len(engine.adaptations) == len(engine.resets) == 40 and len(set(engine.resets)) == 1
    assert len(engine.fits) == 1
    assert sum(row["selected_candidate"] is None
               for row in receipt["fit_selections"].values()) == 2
    observations = rows(inputs.output / "observations.jsonl")
    assert len({D.observation_key(row) for row in observations}) == 512
    assert {row["role"] for row in observations if row["condition"] in D.EXTRAS} == {"Q"}
    assert len(list((inputs.output / "fit-fidelity-before").glob("*.json"))) == 8
    assert len(list((inputs.output / "fit-fidelity-after").glob("*.json"))) == 8
    assert len(list((inputs.output / "evaluation-fidelity").rglob("*.json"))) == 16
    assert D.sha(inputs.output / "teacher/checkpoint.pt") == receipt["teacher_checkpoint_sha256"]
    before = (inputs.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)
    assert (inputs.output / "receipt.json").read_bytes() == before


def test_fit_selection_cannot_be_changed_by_query_scores(inputs):
    _, prepared = D.admit(inputs.fit_run, inputs.fit_run / "lessons.json", inputs.fit_data)
    before = D.fit_selections(inputs.fit_run, prepared)
    observed = rows(inputs.fit_run / "observations.jsonl")
    from hindsight.experiment import append_json
    (inputs.fit_run / "observations.jsonl").unlink()
    for row in observed:
        if row["role"] == "Q":
            row["success"] = not row["success"]
        append_json(inputs.fit_run / "observations.jsonl", row)
    assert D.fit_selections(inputs.fit_run, prepared) == before


@pytest.mark.parametrize("defect", ["source_hash", "policy", "overlap", "missing_v", "code"])
def test_bad_inputs_fail_before_gpu_work(inputs, monkeypatch, defect):
    if defect == "source_hash":
        supplied = read_json(inputs.lessons)
        supplied["rows"][0]["source_sha256"] = "0" * 64
        write_json(inputs.lessons, supplied)
    elif defect in ("policy", "code"):
        parent = read_json(inputs.sources / "identity.json")
        if defect == "policy":
            parent["config"]["learning_rate"] = 1e-3
        else:
            parent["code"]["model.py"] = "changed"
        write_json(inputs.sources / "identity.json", parent)
    elif defect == "overlap":
        monkeypatch.setattr(D, "canonical_inputs", lambda path: {"duplicate"})
    else:
        original = rows(inputs.fit_run / "observations.jsonl")
        (inputs.fit_run / "observations.jsonl").write_text(
            "\n".join(__import__("json").dumps(row) for row in original[1:]) + "\n")
    engine = Engine(inputs.output)
    assert invoke(inputs, engine) == 1
    assert not any(engine.counters.values())
    assert read_json(inputs.output / "receipt.json")["status"] == "failed"


def test_failure_preserves_actual_work_and_missing_rows(inputs):
    engine = Engine(inputs.output, fail=True)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["actual_observations"] == 0 and receipt["missing_observations"] == 512
    assert receipt["work"]["teacher_optimizer_steps"] == 66
    assert receipt["work"]["scoring_forwards"] == 29
    assert receipt["error"] == dict(type="RuntimeError", message="bounded failure")


def test_checkpoint_tampering_is_fatal_at_real_experiment_boundary(inputs):
    class TamperingEngine(Engine):
        def generate(self, prompts, **kwargs):
            result = super().generate(prompts, **kwargs)
            (self.output / "teacher/checkpoint.pt").write_bytes(b"changed checkpoint")
            return result

    engine = TamperingEngine(inputs.output)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["error"]["message"] == "Teacher checkpoint bytes changed"
    assert len(engine.fits) == 1 and receipt["actual_observations"] == 32
    assert not any(row["condition"] == "adapted_bidirectional"
                   for row in rows(inputs.output / "observations.jsonl"))
