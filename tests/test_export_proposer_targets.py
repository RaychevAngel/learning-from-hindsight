"""CPU-only export admission, native target lineage and exact work accounting."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_proposer_training import FakeEngine, invoke
from test_proposer_training import cohort as proposer_cohort

from hindsight.experiment import read_json, write_json
from hindsight.shared import SharedRecipe

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


D = load_script("export_proposer_targets")
SHARED = load_script("check_shared_learning")


@pytest.fixture
def cohort(tmp_path):
    return proposer_cohort.__wrapped__(tmp_path)


class NativeTrainingEngine(FakeEngine):
    def generate(self, *args, **kwargs):
        results = super().generate(*args, **kwargs)
        for value in results:
            value.response_ids = [ord(c) for c in value.text]
        return results

    def sample_proposal(self, *args, **kwargs):
        value = super().sample_proposal(*args, **kwargs)
        value.generation.response_ids = [ord(c) for c in value.generation.text]
        return value


class ExportEngine:
    def __init__(self, *, fail_at=None):
        self.model_name, self.revision = "offline", "pinned"
        self.counters = dict.fromkeys(D.COUNTERS, 0)
        self.calls, self.fail_at = [], fail_at
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))

    def encode_prompt(self, prompt):
        return [ord(c) for c in prompt]

    def score_rollout(self, generation, feedback, **kwargs):
        self.budget_check()
        self.calls.append((generation, feedback, kwargs))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("Controlled scoring failure")
        self.counters["scoring_forwards"] += 1
        return torch.zeros(len(generation.response_ids), 272).log_softmax(-1)

    def _validate_target(self, generation, target):
        assert tuple(target.shape) == (len(generation.response_ids), 272)

    def _scoring_input(self, generation, feedback, pi, mode):
        prefix = self.encode_prompt((pi or "") + generation.prompt)
        return (
            prefix + generation.response_ids,
            list(range(len(prefix) - 1, len(prefix) + len(generation.response_ids) - 1)),
            False,
        )


def prepare(cohort, *, ties=False):
    assert invoke(cohort, NativeTrainingEngine(cohort, ties=ties)) == 0


def export(cohort, engine, **kwargs):
    return D.run(
        cohort.output,
        cohort.data,
        cohort.output.parent / "export",
        engine=engine,
        loader=cohort.loader,
        deadline_utc="2099-01-01T00:00:00+00:00",
        **kwargs,
    )


@pytest.mark.parametrize("ties", [False, True])
def test_exact_one_score_per_source_native_artifact_and_shared_loader_compatibility(cohort, ties):
    prepare(cohort, ties=ties)
    engine = ExportEngine()
    assert export(cohort, engine) == 0
    output = cohort.output.parent / "export"
    receipt, manifest = read_json(output / "receipt.json"), read_json(output / "manifest.json")
    assert receipt["status"] == "completed"
    assert receipt["work"] == {**dict.fromkeys(D.COUNTERS, 0), "scoring_forwards": 2}
    assert manifest["selection_validation_indices"] == [0, 2, 4, 6]
    assert manifest["selection_condition"] == "candidate"
    assert [item["selected_candidate"] for item in manifest["examples"]] == [
        None if ties else 1
    ] * 2
    assert [call[2]["pi"] for call in engine.calls] == [None if ties else "Lesson 1"] * 2
    assert all(call[2]["adapter"] is None and call[2]["mode"] == "causal" for call in engine.calls)
    for entry in manifest["examples"]:
        payload = torch.load(output / entry["target_path"], weights_only=True)
        assert payload["response_ids"] == payload["source_generation"]["response_ids"]
        assert [
            payload["input_ids"][position + 1] for position in payload["predecessor_positions"]
        ] == payload["response_ids"]
        assert not payload["target"].requires_grad
        assert D.sha(output / entry["target_path"]) == entry["target_sha256"]
    examples, loaded = SHARED.load_teaching(output / "manifest.json", SharedRecipe())
    assert [item.episode_id for item in examples] == ["e0", "e1"]
    assert loaded == manifest


@pytest.mark.parametrize(
    "defect", ["incomplete", "dataset", "feedback", "selection", "policy", "native", "core"]
)
def test_export_rejects_frozen_evidence_defects_without_any_score(cohort, defect):
    prepare(cohort)
    if defect in ("incomplete", "dataset", "core"):
        path = cohort.output / "receipt.json"
        value = read_json(path)
        if defect == "incomplete":
            value["status"] = "failed"
        elif defect == "dataset":
            value["config"]["dataset_manifest_sha256"] = "different"
        else:
            key = next(key for key in value["frozen_files"] if key.endswith("/model.py"))
            value["frozen_files"][key] = "different"
    elif defect in ("feedback", "native"):
        path = cohort.output / "sources/e0.json"
        value = read_json(path)
        if defect == "feedback":
            value["feedback"] = "incorrect"
        else:
            value["generation"]["response_ids"][0] += 1
    elif defect == "policy":
        path = cohort.output / "proposals/e0-1.json"
        value = read_json(path)
        value["proposal"]["seed"] += 1
    else:
        path = cohort.output / "groups.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["selected_candidate"] = None
        # This is test-artifact setup; no production lineage is rewritten.
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        value = None
    if value is not None:
        write_json(path, value)
    engine = ExportEngine()
    assert export(cohort, engine) == 1
    assert engine.counters["scoring_forwards"] == 0
    assert not (cohort.output.parent / "export/manifest.json").exists()


def test_scoring_failure_preserves_terminal_partial_without_retry(cohort):
    prepare(cohort)
    engine = ExportEngine(fail_at=2)
    assert export(cohort, engine) == 1
    output = cohort.output.parent / "export"
    receipt = read_json(output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["completed_episode_ids"] == ["e0"]
    assert receipt["work"]["scoring_forwards"] == 1 and len(engine.calls) == 2
    assert (output / "manifest.partial.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        export(cohort, engine)
