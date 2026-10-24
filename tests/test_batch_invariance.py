"""Forensic batching uses the real Experiment generator boundary with an offline engine."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight.data import Episode, Task
from hindsight.experiment import Experiment, append_json, read_json, rows, write_json
from hindsight.model import Generation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_batch_invariance.py"
SPEC = importlib.util.spec_from_file_location("check_batch_invariance", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    sources, reference, data, output = [
        tmp_path / name for name in ("sources", "reference", "data", "out")
    ]
    episodes, lessons = [], []
    for number in range(4):

        def task(role, index):
            tid = f"ep{number}-{role}{index}"
            return Task(tid, f"TASK {tid}", (1,), "fixture")

        ep = Episode(
            f"ep{number}",
            "development",
            "fixture",
            task("S", 0),
            tuple(task("V", i) for i in range(4)),
            tuple(task("Q", i) for i in range(8)),
        )
        episodes.append(ep)
        source = {
            "task_id": ep.source.task_id,
            "generation": dataclasses.asdict(
                Generation(ep.source.prompt, [2], [3, 1], "original source", False)
            ),
        }
        for folder in (sources, reference):
            write_json(folder / "sources" / f"{ep.episode_id}.json", source)
        lessons.append(
            {
                "episode_id": ep.episode_id,
                "lesson": f"Fixed lesson {number}.",
                "source_sha256": CHECK.sha256(
                    (sources / "sources" / f"{ep.episode_id}.json").read_bytes()
                ),
            }
        )
        for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
            for condition in CHECK.CONDITIONS:
                for t in tasks:
                    append_json(
                        reference / "observations.jsonl",
                        {
                            "episode_id": ep.episode_id,
                            "task_id": t.task_id,
                            "role": role,
                            "partition": "development",
                            "condition": condition,
                            "candidate": 0 if condition == "direct_pi" else None,
                            "text": f"{t.task_id}\nFINAL: [1]",
                            "cap_hit": False,
                            "success": True,
                            "format_ok": True,
                            "prediction": [1],
                        },
                    )
    config = {
        "model_name": "offline",
        "revision": "a" * 40,
        "seed": 7301,
        "deterministic_execution": True,
        "generation_batch_size": 8,
        "evaluation_tokens": 160,
        "max_context_tokens": 1536,
        "max_teacher_context_tokens": 4096,
        "max_invocation_seconds": 1200,
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "diagnostic_source_ids": [ep.episode_id for ep in episodes],
    }
    manifest = {"fixture": "immutable corpus"}
    code = {name: "unchanged" for name in ("model.py", "experiment.py", "data.py")}
    write_json(data / "manifest.json", manifest)
    write_json(
        sources / "identity.json", {"config": config, "dataset_manifest": manifest, "code": code}
    )
    write_json(reference / "lessons.json", {"provenance": {"kind": "fixture"}, "rows": lessons})
    diagnostic = {
        "parent_identity_sha256": CHECK.sha256((sources / "identity.json").read_bytes()),
        "lessons_sha256": CHECK.sha256((reference / "lessons.json").read_bytes()),
        "episode_ids": config["diagnostic_source_ids"],
    }
    write_json(
        reference / "identity.json",
        {
            "config": {**config, "external_lessons_diagnostic": diagnostic},
            "dataset_manifest": manifest,
            "code": code,
        },
    )
    write_json(
        reference / "receipt.json",
        {"status": "completed", "completed_episode_ids": config["diagnostic_source_ids"]},
    )
    monkeypatch.setattr("hindsight.data.load_corpus", lambda path: (episodes, []))
    monkeypatch.setattr(Experiment, "code_identity", lambda self: code)
    # Actual strict numeric configuration is separately tested; avoid process-global changes here.
    monkeypatch.setattr("hindsight.model.configure_deterministic_execution", lambda enabled: None)
    return SimpleNamespace(
        sources=sources, reference=reference, data=data, output=output, episodes=episodes
    )


class Engine:
    def __init__(self, output, *, sensitive=False, fail_on_call=None):
        self.output, self.sensitive, self.fail_on_call = output, sensitive, fail_on_call
        self.calls = []
        self.counters = dict.fromkeys(CHECK.PLANNED_WORK, 0)
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [2])

    def generate(self, prompts, **kwargs):
        plan = read_json(self.output / "plan.json")
        assert sum(len(batch["tasks"]) for batch in plan["batches"]) == 384
        assert not any("-S0" in prompt for prompt in prompts)
        assert kwargs == {
            "max_new_tokens": 160,
            "sample": False,
            "seed": 7301,
            "adapter": None,
            "context_limit": 1536,
        }
        self.calls.append(prompts)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("controlled interruption")
        self.counters["generation_sequences"] += len(prompts)
        result = []
        for prompt in prompts:
            tid = prompt.rsplit("TASK ", 1)[1]
            marker = len(prompts) if self.sensitive else 0
            text = f"{tid}\nFINAL: [1]"
            # Same decoded text/grade but different token IDs must still fail invariance.
            response = [int(CHECK.sha256(tid.encode())[:6], 16), marker, 1]
            result.append(Generation(prompt, [2, len(prompt)], response, text, False))
        return result


def invoke(values, engine, **kwargs):
    return CHECK.run(
        values.sources, values.reference, values.data, values.output, engine=engine, **kwargs
    )


def test_complete_384_sequence_plan_pairing_and_unchanged_sources(inputs):
    engine = Engine(inputs.output)
    original = {p.name: p.read_bytes() for p in (inputs.sources / "sources").iterdir()}
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["invariance"] is True
    assert receipt["work"] == CHECK.PLANNED_WORK and receipt["missing_records"] == 0
    assert len(engine.calls) == 152
    records = rows(inputs.output / "generations.jsonl")
    assert len(records) == 384
    for layout in CHECK.LAYOUTS:
        selected = [r for r in records if r["layout"] == layout]
        assert len({CHECK.key(r) for r in selected}) == 96
        for record in selected:
            assert record["generation"]["text"] == f"{record['task_id']}\nFINAL: [1]"
            assert record["batch_companions"][record["batch_index"]] == {
                "task_id": record["task_id"],
                "role": record["role"],
            }
            assert ("hindsight_advice" in record["generation"]["prompt"]) == (
                record["condition"] == "direct_pi"
            )
    mixed = [r for r in records if r["layout"] == "batch8" and r["batch_size"] == 8]
    assert all([t["role"] for t in r["batch_companions"]] == ["V"] * 4 + ["Q"] * 4 for r in mixed)
    matrix = read_json(inputs.output / "mismatch-matrix.json")
    assert len(matrix["fresh_pairwise"]) == 6 and len(matrix["saved_replay"]) == 4
    assert all(r["compared_tasks"] == 96 for group in matrix.values() for r in group)
    assert all(not r["token_comparison_available"] for r in matrix["saved_replay"])
    assert original == {p.name: p.read_bytes() for p in (inputs.sources / "sources").iterdir()}
    before = (inputs.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)
    assert (inputs.output / "receipt.json").read_bytes() == before


def test_token_mismatch_is_completed_not_failed_or_retried(inputs):
    engine = Engine(inputs.output, sensitive=True)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["invariance"] is False
    assert receipt["batch_invariance"] is False
    assert receipt["original_role_replay_invariance"] is True
    matrix = read_json(inputs.output / "mismatch-matrix.json")
    pair = next(
        r for r in matrix["fresh_pairwise"] if r["left"] == "batch1" and r["right"] == "batch4"
    )
    assert pair["mismatch_counts"]["response_ids"] == 96
    assert pair["mismatch_counts"]["text"] == pair["mismatch_counts"]["grade"] == 0
    assert receipt["work"] == CHECK.PLANNED_WORK


def test_old_raw_text_mismatch_is_reported_without_claiming_token_comparison(inputs):
    path = inputs.reference / "observations.jsonl"
    original = rows(path)
    original[0]["text"] = "Previously different explanation\nFINAL: [1]"
    path.write_text("\n".join(json.dumps(row) for row in original) + "\n")
    assert invoke(inputs, Engine(inputs.output)) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["batch_invariance"] is True
    assert receipt["original_role_replay_invariance"] is False
    comparison = read_json(inputs.output / "mismatch-matrix.json")["saved_replay"][0]
    assert comparison["mismatch_counts"]["text"] == 1
    assert "response_ids" not in comparison["mismatch_counts"]


@pytest.mark.parametrize("defect", ["source", "lessons", "config", "final", "missing", "duplicate"])
def test_bad_inputs_stop_before_generation(inputs, defect):
    if defect == "source":
        write_json(inputs.sources / "sources" / "ep0.json", {"changed": True})
    elif defect == "lessons":
        value = read_json(inputs.reference / "lessons.json")
        value["rows"][0]["lesson"] = "changed"
        write_json(inputs.reference / "lessons.json", value)
    elif defect == "config":
        value = read_json(inputs.reference / "identity.json")
        value["config"]["evaluation_tokens"] = 161
        write_json(inputs.reference / "identity.json", value)
    elif defect == "final":
        inputs.episodes[0] = dataclasses.replace(inputs.episodes[0], partition="final_known")
    else:
        path = inputs.reference / "observations.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[1:] if defect == "missing" else lines + lines[:1]) + "\n")
    engine = Engine(inputs.output)
    assert invoke(inputs, engine) == 1
    assert not engine.calls
    assert read_json(inputs.output / "receipt.json")["status"] == "failed"


def test_interruption_keeps_partial_work_without_retry(inputs):
    engine = Engine(inputs.output, fail_on_call=3)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["actual_records"] == 12 and receipt["work"]["generation_sequences"] == 12
    assert receipt["invariance"] is None
    assert receipt["error"]["message"] == "controlled interruption"


def test_bounded_deadline_stops_before_model_generation(inputs):
    engine = Engine(inputs.output)
    assert invoke(inputs, engine, max_seconds=1e-12) == 1
    assert not engine.calls
    assert read_json(inputs.output / "receipt.json")["error"]["type"] == "TimeoutError"
