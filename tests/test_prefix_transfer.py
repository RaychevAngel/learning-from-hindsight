"""Cached-target transfer has no target/source generation or hidden arm selection."""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_prefix_interventions import Engine as PrefixEngine
from test_prefix_interventions import fixture as prefix_fixture

from hindsight.data import Episode, Task
from hindsight.experiment import read_json, rows, write_json
from hindsight.model import Generation

SPEC = importlib.util.spec_from_file_location(
    "prefix_transfer", Path(__file__).parents[1] / "scripts/check_prefix_transfer.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class Engine(PrefixEngine):
    def __init__(self, output, fail_at=None, mutate=False):
        super().__init__(output)
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(named_buffers=lambda: [])
        self.resets, self.adaptations, self.generated = [], [], []
        self.fail_at, self.mutate = fail_at, mutate

    def _adapter_parameters(self, adapter):
        assert adapter == "student"
        return {"weight": self.weight}

    def reset_student(self, seed):
        self.resets.append(seed)
        self.weight.data.zero_()
        self._student_optimizer = torch.optim.AdamW([self.weight], lr=1e-4, weight_decay=0)

    def adapt(self, generation, target, steps, lr):
        self.budget_check()
        assert steps == 16 and lr == 1e-4 and not target.requires_grad
        assert self.weight.item() == 0 and not self._student_optimizer.state
        self.adaptations.append((generation, target.clone()))
        self.weight.data.add_(target.exp()[0, 0])
        self.counters["student_optimizer_steps"] += steps
        return [
            dict(step=step, optimizer_steps=1, example_exposures=1, loss=0.1, grad_norm=0.01, lr=lr)
            for step in range(1, steps + 1)
        ]

    def generate(self, prompts, *, max_new_tokens, sample, seed, adapter):
        self.budget_check()
        assert len(prompts) == 8 and not sample and seed == 7301 and max_new_tokens == 160
        assert all(prompt.startswith("validation ") for prompt in prompts)
        self.generated.append((prompts, adapter))
        if len(self.generated) == self.fail_at:
            raise RuntimeError("controlled failure")
        self.counters["generation_sequences"] += len(prompts)
        if self.mutate and adapter == "student":
            self.weight.data.add_(1)
        text = "FINAL: [1]"
        return [
            Generation(prompt, self.encode_prompt(prompt), list(map(ord, text)), text, False)
            for prompt in prompts
        ]

    def score_rollout(self, *args, **kwargs):
        raise AssertionError("Transfer must never score a teacher or student")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 6, 8, 0, tzinfo=timezone.utc)

    original = prefix_fixture.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(D, "datetime", Clock)
    monkeypatch.setattr(D.PREFIX, "datetime", Clock)
    path = original.fidelity / "receipt.json"
    receipt = read_json(path)
    receipt["inherited_config"].update(seed=7301, evaluation_tokens=160)
    write_json(path, receipt)
    framing = original.output
    assert D.PREFIX.run(original.fidelity, original.data, framing, engine=original.engine) == 0
    episodes = []
    for i, ep_id in enumerate(receipt["episode_ids"]):
        source = Task(f"task-{i}", f"public task {i}", (1,), "family")
        validation = tuple(
            Task(f"v-{i}-{j}", f"validation {i} {j}", (1,), "family") for j in range(8)
        )
        queries = tuple(
            Task(f"q-{i}-{j}", f"QUERY NEVER GENERATED {i} {j}", (9,), "family") for j in range(8)
        )
        episodes.append(Episode(ep_id, "development", "family", source, validation, queries))
    monkeypatch.setattr(D, "load_corpus", lambda path: (episodes, []))
    qualified = tmp_path / "qualified.json"
    write_json(
        qualified,
        dict(
            status="completed",
            selected_batch_size=8,
            repeat_stability={"8": True},
            dataset_sha256="different-qualified-corpus",
            config=dict(model_name="offline", revision="pinned"),
        ),
    )
    config = dict(
        receipt["inherited_config"],
        generation_batch_size=8,
        student_steps=16,
        learning_rate=1e-4,
        max_invocation_seconds=900,
        absolute_deadline_utc=D.PREFIX.LATEST_DEADLINE,
        batch_qualification=dict(path=str(qualified), sha256=D.sha(qualified)),
    )
    return SimpleNamespace(
        data=original.data,
        fidelity=original.fidelity,
        framing=framing,
        output=tmp_path / "transfer",
        config=config,
        episodes=episodes,
    )


def invoke(fixture, **kwargs):
    engine = Engine(fixture.output, **kwargs)
    status = D.run(
        fixture.config,
        fixture.data,
        fixture.fidelity,
        fixture.framing,
        fixture.output,
        engine=engine,
    )
    return status, engine


def test_all_three_cached_targets_exact_work_native_matrix_and_matched_resets(fixture):
    status, engine = invoke(fixture)
    assert status == 0
    receipt = read_json(fixture.output / "receipt.json")
    assert receipt["work"] == D.WORK and receipt["status"] == "completed"
    assert len(engine.generated) == 32 and len(engine.adaptations) == len(engine.resets) == 24
    assert len(set(engine.resets)) == 1
    assert [adapter for _, adapter in engine.generated] == [
        None,
        "student",
        "student",
        "student",
    ] * 8
    observations = rows(fixture.output / "observations.jsonl")
    assert len(observations) == 256 and {row["role"] for row in observations} == {"V"}
    assert all(row["success"] and row["generation"]["response_ids"] for row in observations)
    assert receipt["validation_totals"] == {
        arm: dict(successes=64, denominator=64, invalid=0, caps=0) for arm in D.ARMS
    }
    assert not (fixture.output / "recipe.json").exists()
    assert not (fixture.output / "selections.jsonl").exists()
    # The three values differ, proving the fake did not accidentally reuse old PI for B/C.
    first_targets = [target for _, target in engine.adaptations[:3]]
    assert not torch.equal(first_targets[0], first_targets[1])
    assert not torch.equal(first_targets[1], first_targets[2])


@pytest.mark.parametrize(
    "defect", ["incomplete", "tensor", "model", "recipe", "qualification", "deadline", "budget"]
)
def test_frozen_admission_and_runtime_fail_before_model_work(fixture, defect):
    if defect == "incomplete":
        path = fixture.framing / "receipt.json"
        value = read_json(path)
        value["status"] = "failed"
        write_json(path, value)
    elif defect == "tensor":
        (fixture.framing / "distributions/source-0.pt").write_bytes(b"corrupt")
    elif defect == "model":
        fixture.config["revision"] = "other"
    elif defect == "recipe":
        fixture.config["student_steps"] = 4
    elif defect == "qualification":
        fixture.config["batch_qualification"]["sha256"] = "changed"
    elif defect == "deadline":
        fixture.config["absolute_deadline_utc"] = "2026-09-06T16:00:00+00:00"
    else:
        fixture.config["max_invocation_seconds"] = 1801
    status, engine = invoke(fixture)
    assert status == 1 and not engine.generated and not engine.adaptations
    assert read_json(fixture.output / "receipt.json")["work"] == dict.fromkeys(D.WORK, 0)


def test_changed_native_alignment_rejected_before_generation(fixture):
    path = fixture.framing / "distributions/source-0.pt"
    value = torch.load(path, weights_only=True)
    value["alignment"]["B"]["predecessor_positions"][0] -= 1
    torch.save(value, path)
    sidecar = read_json(path.with_suffix(".json"))
    sidecar["artifact"]["sha256"] = D.sha(path)
    write_json(path.with_suffix(".json"), sidecar)
    status, engine = invoke(fixture)
    assert status == 1 and not engine.generated and not engine.adaptations
    assert "alignment" in read_json(fixture.output / "receipt.json")["error"]["message"]


def test_read_only_generation_state_boundary_fails_closed(fixture):
    status, engine = invoke(fixture, mutate=True)
    assert status == 1 and len(engine.generated) == 2
    receipt = read_json(fixture.output / "receipt.json")
    assert "mutated" in receipt["error"]["message"]
    assert receipt["work"]["student_optimizer_steps"] == 16


def test_generation_failure_is_durable_partial_with_no_retry_or_overwrite(fixture):
    status, engine = invoke(fixture, fail_at=5)
    assert status == 1 and len(engine.generated) == 5
    receipt = read_json(fixture.output / "receipt.json")
    assert receipt["completed_episode_ids"] == ["source-0"]
    assert receipt["work"]["generation_sequences"] == 32
    assert receipt["work"]["student_optimizer_steps"] == 48
    with pytest.raises(FileExistsError):
        invoke(fixture)
