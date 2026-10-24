"""Source fitting uses the real coordinator and bounded tiny fake distributions."""

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_order_and_pi import inputs as order_inputs

from hindsight.experiment import read_json, rows, seed_for, write_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_source_fidelity", ROOT / "scripts/check_source_fidelity.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture
def inputs(tmp_path):
    values = order_inputs.__wrapped__(tmp_path)
    # Turn the copied source files into small, internally consistent native-ID fixtures.
    supplied = read_json(values.lessons)
    for lesson in supplied["rows"]:
        name = lesson["episode_id"] + ".json"
        source = read_json(values.sources / "sources" / name)
        source["generation"].update(response_ids=[1, 2, 3], prompt_ids=[4, 5])
        for folder in (values.sources, values.reference):
            write_json(folder / "sources" / name, source)
        lesson["source_sha256"] = D.FROZEN.sha(values.sources / "sources" / name)
    for path in (values.lessons, values.reference / "lessons.json"):
        write_json(path, supplied)
    identity = read_json(values.reference / "identity.json")
    identity["config"]["hindsight_comparison"]["lessons_sha256"] = D.FROZEN.sha(values.lessons)
    write_json(values.reference / "identity.json", identity)
    receipt = read_json(values.reference / "receipt.json")
    receipt["lessons_sha256"] = D.FROZEN.sha(values.lessons)
    write_json(values.reference / "receipt.json", receipt)
    shutil.copy2(
        ROOT / "results/hindsight-teacher-qwen35-023/events.jsonl",
        values.reference / "events.jsonl",
    )
    return values


class Engine:
    def __init__(
        self, inputs, *, mismatch=False, fail_score=None, missing_callback=False, mutate_score=False
    ):
        self.inputs, self.mismatch = inputs, mismatch
        self.fail_score, self.missing_callback = fail_score, missing_callback
        self.mutate_score = mutate_score
        self.counters = dict.fromkeys(D.PLANNED_WORK, 0)
        self.scores, self.resets, self.adaptations = [], [], []
        self.target = torch.tensor([[0.1, 0.4, 0.2, 0.2, 0.1]] * 3).log()
        self.model = torch.nn.Linear(1, 1)
        self.model.register_buffer("sample_buffer", torch.tensor([0.0]))
        self._student_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda values: [str(v) for v in values]
        )
        self.initial = torch.tensor([[0.2] * 5] * 3).log().requires_grad_()
        self.post = self.target.clone()
        inherited, self.prepared, _, _ = D.FROZEN.admit(
            inputs.sources, inputs.reference, inputs.lessons, inputs.data
        )
        self.historical = D.reference_metrics(
            inputs.reference, [ep.episode_id for ep, *_ in self.prepared]
        )

    def _adapter_parameters(self, name):
        assert name == "student"
        return dict(self.model.named_parameters())

    def _scoring_input(self, generation, feedback, pi, mode):
        assert mode == "causal"
        prefix = generation.prompt_ids if pi is None else [4, 5, 6, 7]
        return (
            prefix + generation.response_ids,
            [len(prefix) + i - 1 for i in range(len(generation.response_ids))],
            False,
        )

    def generate(self, *args, **kwargs):
        raise AssertionError("No generation is allowed")

    def score_rollout(self, generation, feedback, **kwargs):
        assert len(read_json(self.inputs.output / "plan.json")["sources"]) == 8
        self.scores.append((generation, feedback, kwargs))
        if len(self.scores) == self.fail_score:
            raise RuntimeError("controlled score failure")
        self.counters["scoring_forwards"] += 1
        if "pi" in kwargs:
            assert "adapter" not in kwargs
            return self.target.clone()
        assert kwargs == {"mode": "causal", "adapter": "student"} and feedback == ""
        if self.mutate_score:
            self.model.sample_buffer.add_(1)
        return self.post.clone()

    def reset_student(self, seed):
        self.resets.append(seed)

    def adapt(self, generation, target, steps, lr, initial_logprobs_callback):
        assert steps == 4 and lr == 1e-4 and not target.requires_grad
        assert generation.response_ids == [1, 2, 3]
        self.adaptations.append(generation)
        if not self.missing_callback:
            initial_logprobs_callback(self.initial)
        self.counters["student_optimizer_steps"] += steps
        ep = next(ep for ep, *_ in self.prepared if ep.source.prompt == generation.prompt)
        metrics = [{**row, "elapsed_seconds": 123.0} for row in self.historical[ep.episode_id]]
        if self.mismatch:
            metrics[0]["loss"] += 1e-7
        return metrics


def invoke(inputs, engine, **kwargs):
    return D.run(
        inputs.sources,
        inputs.reference,
        inputs.lessons,
        inputs.data,
        inputs.output,
        engine=engine,
        deadline_utc="2099-01-01T00:00:00+00:00",
        **kwargs,
    )


def test_exact_frozen_eight_source_work_native_artifacts_and_independent_metrics(inputs):
    engine = Engine(inputs)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["work"] == D.PLANNED_WORK
    assert receipt["actual_sources"] == 8 and receipt["missing_sources"] == 0
    assert receipt["frozen_inputs_unchanged"]
    assert len(engine.scores) == 16 and len(engine.adaptations) == len(engine.resets) == 8
    assert engine.resets == [seed_for(7301, "student_initialization")] * 8
    for index, (ep, generation, _, lesson) in enumerate(engine.prepared):
        assert engine.scores[index * 2][2] == {"pi": lesson}
        assert engine.scores[index * 2 + 1][2] == {"mode": "causal", "adapter": "student"}
        assert engine.scores[index * 2][0] is engine.scores[index * 2 + 1][0]
        artifact = inputs.output / "distributions" / f"{ep.episode_id}.pt"
        saved = torch.load(artifact, weights_only=True)
        assert saved["response_ids"] == generation.response_ids
        for name in ("target", "initial", "post_fourth"):
            assert saved[name].dtype == torch.float32 and not saved[name].requires_grad
        metrics = read_json(artifact.with_suffix(".json"))["metrics"]
        independently_computed = (saved["target"].exp() * (saved["target"] - saved["initial"])).sum(
            -1
        )
        for row, expected in zip(metrics["positions"], independently_computed, strict=True):
            assert row["initial_forward_kl_teacher_to_student"] == pytest.approx(
                expected.item(), abs=1e-7
            )
            assert row["post_fourth_forward_kl_teacher_to_student"] == pytest.approx(0, abs=1e-7)
            assert row["post_fourth_total_variation"] == 0
            assert row["post_fourth_argmax_agreement"] == 1
        assert (
            D.FROZEN.sha(artifact) == read_json(artifact.with_suffix(".json"))["artifact"]["sha256"]
        )
    assert all(row["identical"] for row in rows(inputs.output / "reconstruction.jsonl"))
    before = (inputs.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)
    assert (inputs.output / "receipt.json").read_bytes() == before


def test_numeric_step_mismatch_stops_before_postscore_and_saves_partial(inputs):
    engine = Engine(inputs, mismatch=True)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["work"] == {
        **dict.fromkeys(D.PLANNED_WORK, 0),
        "scoring_forwards": 1,
        "student_optimizer_steps": 4,
    }
    assert receipt["actual_sources"] == 0 and len(engine.scores) == 1
    assert rows(inputs.output / "reconstruction.jsonl")[0]["identical"] is False
    saved = torch.load(next((inputs.output / "distributions").glob("*.pt")), weights_only=True)
    assert "initial" in saved and "target" in saved and "post_fourth" not in saved


def test_failed_postscore_is_not_retried_or_counted_as_completed_source(inputs):
    engine = Engine(inputs, fail_score=2)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["actual_sources"] == 0 and len(engine.scores) == 2
    assert receipt["work"]["scoring_forwards"] == 1
    assert receipt["error"]["message"] == "controlled score failure"


def test_callback_is_required_without_extra_initial_scoring(inputs):
    engine = Engine(inputs, missing_callback=True)
    assert invoke(inputs, engine) == 1
    assert len(engine.scores) == 1
    assert (
        read_json(inputs.output / "receipt.json")["error"]["message"]
        == "Missing unique initial causal distribution callback"
    )


def test_final_score_is_read_only_and_mutation_fails(inputs):
    engine = Engine(inputs, mutate_score=True)
    assert invoke(inputs, engine) == 1
    assert rows(inputs.output / "student-state.jsonl")[0]["unchanged"] is False
    assert len(engine.scores) == 2


@pytest.mark.parametrize("defect", ["events", "lessons", "core_identity"])
def test_immutable_input_defects_fail_before_work(inputs, defect):
    engine = Engine(inputs)
    if defect == "events":
        path = inputs.reference / "events.jsonl"
        entries = rows(path)
        entries = [row for row in entries if row.get("condition") != "adapted_pi"]
        path.write_text("\n".join(__import__("json").dumps(row) for row in entries) + "\n")
    elif defect == "lessons":
        value = read_json(inputs.lessons)
        value["rows"][0]["lesson"] += "Changed"
        write_json(inputs.lessons, value)
    else:
        value = read_json(inputs.reference / "identity.json")
        value["code"]["model.py"] = "changed"
        write_json(inputs.reference / "identity.json", value)
    assert invoke(inputs, engine) == 1
    assert not any(engine.counters.values())


@pytest.mark.parametrize("max_seconds", [0, -1, 301, float("inf"), 1e-12])
def test_positive_bounded_runtime_before_model_work(inputs, max_seconds):
    engine = Engine(inputs)
    assert invoke(inputs, engine, max_seconds=max_seconds) == 1
    assert not any(engine.counters.values())
