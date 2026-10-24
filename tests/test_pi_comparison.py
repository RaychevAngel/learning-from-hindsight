import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch
from test_proposer_training import FakeEngine
from test_proposer_training import cohort as proposer_cohort

from hindsight.experiment import read_json, rows
from hindsight.model import Generation

SPEC = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).parents[1] / "scripts/compare_pi.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    value = proposer_cohort.__wrapped__(tmp_path)
    value.episodes = [
        dataclasses.replace(ep, partition="development", family=f"family{i}")
        for i, ep in enumerate(value.episodes[:2])
    ]
    value.config.update(partition="development", expected_sources=2, enable_thinking=False)
    monkeypatch.setattr(D, "load_cohort", lambda path: (value.episodes, []))
    return value


class Engine(FakeEngine):
    def generate(self, prompts, *, adapter, **kwargs):
        self.budget_check()
        self.calls.append((prompts, adapter, kwargs))
        values = []
        for prompt in prompts:
            success = (
                "-S0" in prompt
                or "Lesson 2" in prompt
                or (adapter == "student" and self.student_choice == 1)
            )
            values.append(
                Generation(
                    prompt,
                    self.encode_prompt(prompt),
                    [10, 1],
                    "FINAL: [1]" if success else "FINAL: [0]",
                    False,
                )
            )
        self.counters["generation_sequences"] += len(values)
        return values

    def score_rollout(self, generation, feedback, *, pi=None):
        self.counters["scoring_forwards"] += 1
        return torch.tensor([float(pi.split()[-1]) if pi else -1.0])

    def adapt(self, generation, target, steps, lr):
        return super().adapt(generation, target, steps=steps, lr=lr)


def test_complete_matrix_direct_ranking_not_actual_update_ranking(cohort):
    engine = Engine(cohort)
    receipt = D.run(cohort.config, cohort.data, cohort.output, engine=engine)
    assert receipt["status"] == "completed"
    assert receipt["work"]["generation_sequences"] == 362
    assert receipt["work"]["student_optimizer_steps"] == 32
    assert receipt["work"]["scoring_forwards"] == 8
    assert receipt["work"]["proposer_optimizer_steps"] == 0
    selections = rows(cohort.output / "selections.jsonl")
    assert {r["selected_candidate"] for r in selections} == {1}
    assert {r["direct_selected_candidate"] for r in selections} == {2}
    manifest = read_json(cohort.output / "teaching.json")
    assert manifest["selection_validation_indices"] == list(range(8))
    assert len(manifest["examples"]) == 2
    assert manifest["selection_observations_sha256"] == D.IO.sha(
        cohort.output / "observations.jsonl"
    )
    assert len(rows(cohort.output / "observations.jsonl")) == 352
    assert len(engine.proposals) == 8


def test_invalid_pool_records_explicit_neutral_reuse_without_inventing_work(cohort):
    engine = Engine(cohort, invalid=True)
    receipt = D.run(cohort.config, cohort.data, cohort.output, engine=engine)
    assert receipt["status"] == "completed"
    assert receipt["work"]["generation_sequences"] == 74
    assert receipt["work"]["student_optimizer_steps"] == 0
    assert receipt["work"]["scoring_forwards"] == 2
    assert all(r["selected_candidate"] is None for r in rows(cohort.output / "selections.jsonl"))


@pytest.mark.parametrize(
    "field,value",
    [("dataset_manifest_sha256", "wrong"), ("expected_sources", 3), ("candidate_count", 3)],
)
def test_bad_identity_or_budget_fails_before_generation(cohort, field, value):
    cohort.config[field] = value
    engine = Engine(cohort)
    with pytest.raises(ValueError):
        D.run(cohort.config, cohort.data, cohort.output, engine=engine)
    assert not engine.calls and not engine.proposals
    assert read_json(cohort.output / "receipt.json")["status"] == "failed"
