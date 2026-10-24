"""Source preparation tests use real Experiment.source and no model/provider calls."""

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight.data import Episode, Task
from hindsight.experiment import read_json, seed_for, write_json
from hindsight.model import Generation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_model_sources.py"
SPEC = importlib.util.spec_from_file_location("prepare_model_sources", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data, output, config_path = tmp_path / "data", tmp_path / "output", tmp_path / "config.json"
    write_json(data / "manifest.json", {"fixture": "original v2"})
    monkeypatch.setattr(PREPARE, "DATASET_MANIFEST_SHA256",
                        PREPARE.sha256((data / "manifest.json").read_bytes()))
    episodes = []
    for index in range(4):
        task = Task(f"source-{index}", f"PUBLIC SOURCE {index}", (1,), "PRIVATE_FAMILY")
        hidden = Task(f"hidden-{index}", "NEVER_SHOW_V_OR_Q", (7,), "PRIVATE_FAMILY")
        episodes.append(Episode(f"ep-{index}", "development", "PRIVATE_FAMILY", task,
                                (hidden,), (hidden,)))
    monkeypatch.setattr("hindsight.data.load_corpus", lambda path: (episodes, []))
    config = {
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "max_invocation_seconds": 60, "seed": 7301, "source_tokens": 128,
        "max_context_tokens": 1536, "max_teacher_context_tokens": 4096,
        "model_name": "Qwen/Qwen3.5-offline-fixture", "revision": "a" * 40,
        "enable_thinking": False, "diagnostic_source_ids": [ep.episode_id for ep in episodes],
    }
    write_json(config_path, config)
    return SimpleNamespace(data=data, output=output, config_path=config_path,
                           config=config, episodes=episodes)


class OfflineEngine:
    def __init__(self, output, fail_on_call=None, source_count=4):
        self.output, self.fail_on_call = output, fail_on_call
        self.source_count = source_count
        self.calls = []
        self.counters = dict.fromkeys(PREPARE.PLANNED_WORK, 0)
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [900])

    def generate(self, prompts, **kwargs):
        receipt = read_json(self.output / "receipt.json")
        assert receipt["planned_work"] == {
            **PREPARE.PLANNED_WORK, "generation_sequences": self.source_count,
        }
        assert len(receipt["expected"]) == self.source_count
        assert len(prompts) == 1 and prompts[0].startswith("PUBLIC SOURCE")
        assert "PRIVATE" not in prompts[0] and "NEVER_SHOW" not in prompts[0]
        self.calls.append((prompts, kwargs))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("controlled interruption")
        self.counters["generation_sequences"] += 1
        text = "FINAL: [1]" if len(self.calls) % 2 else "FINAL: [0]"
        return [Generation(prompts[0], [900, 901], [1000 + len(self.calls)], text, False)]


def invoke(values, engine):
    return PREPARE.run(values.config_path, values.data, values.output, engine=engine)


def test_exactly_four_fresh_sources_and_filtered_inputs(inputs):
    engine = OfflineEngine(inputs.output)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["work"] == PREPARE.PLANNED_WORK
    assert receipt["actual_sources"] == 4 and receipt["missing_sources"] == 0
    assert read_json(inputs.output / "identity.json")["config"] == inputs.config
    assert not (inputs.output / "proposals").exists()
    assert not (inputs.output / "observations.jsonl").exists()
    for index, (ep, call) in enumerate(zip(inputs.episodes, engine.calls, strict=True)):
        source = read_json(inputs.output / "sources" / f"{ep.episode_id}.json")
        safe = read_json(inputs.output / "proposer-inputs" / f"{ep.episode_id}.json")
        assert set(safe) == {"record_id", "task", "attempt", "feedback"}
        assert safe == {"record_id": ep.episode_id, "task": ep.source.prompt,
                        "attempt": source["generation"]["text"], "feedback": source["feedback"]}
        assert source["generation"]["response_ids"] == [1001 + index]
        assert source["generation"]["prompt_ids"] == [900, 901]
        assert source["grade"]["success"] is (index % 2 == 0)
        assert source["feedback"] == (
            "The submitted answer was correct." if index % 2 == 0
            else "The submitted answer was incorrect."
        )
        assert call[1] == {
            "max_new_tokens": 128, "sample": True,
            "seed": seed_for(7301, ep.episode_id, "source"), "adapter": None,
            "context_limit": 1536,
        }
        provenance = receipt["source_provenance"][index]
        assert provenance["source_sha256"] == PREPARE.sha256(
            (inputs.output / "sources" / f"{ep.episode_id}.json").read_bytes()
        )
    before = (inputs.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)
    assert (inputs.output / "receipt.json").read_bytes() == before


@pytest.mark.parametrize("defect", ["ids_missing", "ids_duplicate", "ids_count", "unknown",
                                   "final", "revision", "model", "manifest"])
def test_bad_admission_fails_before_engine_work(inputs, defect):
    config = dict(inputs.config)
    if defect == "ids_missing":
        del config["diagnostic_source_ids"]
    elif defect == "ids_duplicate":
        config["diagnostic_source_ids"] = ["ep-0"] * 4
    elif defect == "ids_count":
        config["diagnostic_source_ids"] = ["ep-0"]
    elif defect == "unknown":
        config["diagnostic_source_ids"][-1] = "unknown"
    elif defect == "final":
        inputs.episodes[-1] = dataclasses.replace(inputs.episodes[-1], partition="final_known")
    elif defect == "revision":
        config["revision"] = "main"
    elif defect == "model":
        config["model_name"] = ""
    elif defect == "manifest":
        write_json(inputs.data / "manifest.json", {"fixture": "different corpus"})
    write_json(inputs.config_path, config)
    engine = OfflineEngine(inputs.output)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "ValueError"
    assert engine.calls == [] and all(value == 0 for value in receipt["work"].values())


def test_interruption_records_actual_counts_without_retry_or_resume(inputs):
    engine = OfflineEngine(inputs.output, fail_on_call=2)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["work"] == {**dict.fromkeys(PREPARE.PLANNED_WORK, 0),
                               "generation_sequences": 1}
    assert receipt["actual_sources"] == 1 and receipt["missing_sources"] == 3
    assert receipt["completed_episode_ids"] == ["ep-0"]
    assert receipt["error"] == {"type": "RuntimeError", "message": "controlled interruption"}
    assert len(list((inputs.output / "sources").glob("*.json"))) == 1
    assert len(list((inputs.output / "proposer-inputs").glob("*.json"))) == 1


def test_nonfresh_engine_is_rejected_without_generation(inputs):
    engine = OfflineEngine(inputs.output)
    engine.counters["generation_sequences"] = 1
    assert invoke(inputs, engine) == 1
    assert not engine.calls
    assert "fresh unused engine" in read_json(inputs.output / "receipt.json")["error"]["message"]


@pytest.fixture
def balanced_inputs(inputs, monkeypatch):
    episodes = []
    for index, family in enumerate(("translation", "denoising", "size_recoloring", "gravity") * 2):
        task = Task(f"fresh-S-{index}", f"PUBLIC SOURCE {index}", (1,), family)
        def hidden(role, n):
            return Task(f"fresh-{role}-{index}-{n}", "NEVER_SHOW_V_OR_Q", (7,), family)
        episodes.append(Episode(f"fresh-{index}", "development", family, task,
                                tuple(hidden("V", n) for n in range(8)),
                                tuple(hidden("Q", n) for n in range(8))))
    manifest = {"protocol_version": "balanced-development-v1",
                "version": "arc1d-balanced-development-v1", "episodes": 8, "tasks": 136,
                "calibration_tasks": 0, "partition_episodes": {"development": 8}}
    write_json(inputs.data / "manifest.json", manifest)
    inputs.config.update(
        diagnostic_source_ids=[ep.episode_id for ep in episodes],
        dataset_manifest_sha256=PREPARE.sha256((inputs.data / "manifest.json").read_bytes()),
    )
    write_json(inputs.config_path, inputs.config)
    inputs.episodes = episodes
    monkeypatch.setattr("hindsight.data.load_corpus", lambda path: (episodes, []))
    return inputs


def test_balanced_opt_in_generates_only_eight_native_sources(balanced_inputs):
    values = balanced_inputs
    engine = OfflineEngine(values.output, source_count=8)
    assert PREPARE.run(values.config_path, values.data, values.output,
                       balanced_development=True, engine=engine) == 0
    receipt = read_json(values.output / "receipt.json")
    assert receipt["work"] == {**PREPARE.PLANNED_WORK, "generation_sequences": 8}
    assert receipt["actual_sources"] == 8 and receipt["missing_sources"] == 0
    assert receipt["admission_protocol"] == "balanced-development-v1"
    assert receipt["episode_ids"] == values.config["diagnostic_source_ids"]
    assert read_json(values.output / "identity.json")["config"] == values.config
    assert not (values.output / "observations.jsonl").exists()
    assert not (values.output / "proposals").exists()
    for ep in values.episodes:
        safe = read_json(values.output / "proposer-inputs" / f"{ep.episode_id}.json")
        source = read_json(values.output / "sources" / f"{ep.episode_id}.json")
        assert safe == {"record_id": ep.episode_id, "task": ep.source.prompt,
                        "attempt": source["generation"]["text"], "feedback": source["feedback"]}


@pytest.mark.parametrize("defect", [
    "no_opt_in", "pin_missing", "pin_wrong", "protocol", "tasks", "ids_duplicate",
    "final", "family", "roles",
])
def test_balanced_source_admission_before_any_model_call(balanced_inputs, defect):
    values = balanced_inputs
    if defect == "pin_missing":
        del values.config["dataset_manifest_sha256"]
    elif defect == "pin_wrong":
        values.config["dataset_manifest_sha256"] = "0" * 64
    elif defect in ("protocol", "tasks"):
        manifest = read_json(values.data / "manifest.json")
        manifest["protocol_version" if defect == "protocol" else "tasks"] = "wrong"
        write_json(values.data / "manifest.json", manifest)
        values.config["dataset_manifest_sha256"] = PREPARE.sha256(
            (values.data / "manifest.json").read_bytes()
        )
    elif defect == "ids_duplicate":
        values.config["diagnostic_source_ids"][-1] = values.config["diagnostic_source_ids"][0]
    elif defect == "final":
        values.episodes[0] = dataclasses.replace(values.episodes[0], partition="final_known")
    elif defect == "family":
        values.episodes[0] = dataclasses.replace(values.episodes[0], family="other")
    elif defect == "roles":
        values.episodes[0] = dataclasses.replace(values.episodes[0], validation=())
    write_json(values.config_path, values.config)
    engine = OfflineEngine(values.output, source_count=8)
    assert PREPARE.run(values.config_path, values.data, values.output,
                       balanced_development=defect != "no_opt_in", engine=engine) == 1
    assert engine.calls == []
    assert not any(read_json(values.output / "receipt.json")["work"].values())
