"""Real immutable 023 admission/coordinator with a deterministic tiny CPU engine."""

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hindsight.data import load_corpus
from hindsight.experiment import read_json, rows, seed_for, write_json
from hindsight.model import Generation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_order_and_pi", ROOT / "scripts/check_order_and_pi.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture
def inputs(tmp_path):
    sources, reference, data = (tmp_path / name for name in ("sources", "reference", "data"))
    originals = (
        ROOT / "results/sources-balanced-qwen35-022",
        ROOT / "results/hindsight-teacher-qwen35-023",
        ROOT / "data/balanced-transfer-v1",
    )
    for original, destination in zip(originals, (sources, reference, data), strict=True):
        destination.mkdir()
        for name in (
            "identity.json",
            "receipt.json",
            "lessons.json",
            "observations.jsonl",
            "expected.jsonl",
        ):
            if (original / name).exists():
                shutil.copy2(original / name, destination / name)
        if (original / "sources").is_dir():
            shutil.copytree(original / "sources", destination / "sources")
    for path in originals[2].iterdir():
        if path.is_file():
            shutil.copy2(path, data / path.name)
    lessons = tmp_path / "lessons.json"
    shutil.copy2(reference / "lessons.json", lessons)
    return SimpleNamespace(
        sources=sources, reference=reference, lessons=lessons, data=data, output=tmp_path / "output"
    )


class Engine:
    def __init__(self, inputs, *, mismatch=None, mutate=None, fail_on=None):
        self.inputs, self.mismatch, self.mutate, self.fail_on = inputs, mismatch, mutate, fail_on
        self.counters = dict.fromkeys(D.PLANNED_WORK, 0)
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1])
        self.model = torch.nn.Module()
        self.model.register_parameter("base", torch.nn.Parameter(torch.tensor([1.0])))
        self.model.register_parameter("lora_student", torch.nn.Parameter(torch.tensor([2.0])))
        self.model.register_buffer("state_buffer", torch.tensor([3.0]))
        self.model.generation_config = SimpleNamespace(cache_implementation=None)
        self._student_optimizer = torch.optim.AdamW([self.model.lora_student], lr=1e-4)
        self.calls, self.resets, self.adaptations, self.scores = [], [], [], []
        self.episodes = load_corpus(inputs.data)[0]
        self.tasks = {
            task.prompt: task for ep in self.episodes for task in (*ep.validation, *ep.queries)
        }
        self.old = {
            row["task_id"]: row
            for row in rows(inputs.reference / "observations.jsonl")
            if row["condition"] == "adapted_pi"
        }

    def _adapter_parameters(self, name):
        assert name == "student"
        return {"lora_student": self.model.lora_student}

    def reset_student(self, seed):
        self.resets.append(seed)
        self.model.lora_student.data.fill_(2)
        self._student_optimizer = torch.optim.AdamW([self.model.lora_student], lr=1e-4)

    def score_rollout(self, generation, feedback, **kwargs):
        assert "pi" in kwargs
        self.scores.append((generation, feedback, kwargs))
        self.counters["scoring_forwards"] += 1
        return torch.tensor([[0.5, 0.5]]).log()

    def adapt(self, generation, targets, steps, lr, initial_logprobs_callback=None):
        assert steps == 4 and lr == 1e-4
        self.adaptations.append(generation)
        self.model.lora_student.data.add_(1)
        self.counters["student_optimizer_steps"] += 4
        return [dict(loss=0.1)] * 4

    def generate(self, prompts, *, adapter, **kwargs):
        assert len(read_json(self.inputs.output / "plan.json")["generations"]) == 224
        assert len(prompts) == 1 and kwargs["sample"] is False and kwargs["seed"] == 7301
        prompt = prompts[0]
        task = next(task for plain, task in self.tasks.items() if prompt.endswith(plain))
        self.calls.append(dict(task_id=task.task_id, prompt=prompt, adapter=adapter))
        number = len(self.calls)
        if number == self.fail_on:
            raise RuntimeError("controlled interruption")
        old = self.old[task.task_id]
        response_ids = list(range(old["generated_tokens"]))
        text = old["text"]
        if self.mismatch == "tokens" and 16 < number <= 32:
            response_ids[0] += 999
        if self.mismatch == "history":
            text += " "
        if self.mutate == "student":
            self.model.lora_student.data.add_(1)
        elif self.mutate == "buffer":
            self.model.state_buffer.add_(1)
        elif self.mutate == "base":
            self.model.base.data.add_(1)
        elif self.mutate == "input":
            write_json(self.inputs.lessons, {"changed": True})
        self.counters["generation_sequences"] += 1
        return [Generation(prompt, [1, 2], response_ids, text, old["cap_hit"])]


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


def test_exact_work_orders_fixed_sources_and_no_optimizer_between_layouts(inputs):
    engine = Engine(inputs)
    before = D.frozen_files(inputs.sources, inputs.reference, inputs.lessons, inputs.data)
    assert invoke(inputs, engine) == 0
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["work"] == D.PLANNED_WORK and receipt["actual_records"] == 224
    assert (
        receipt["order_gate"]
        and receipt["current_order_invariance"]
        and receipt["history_reproduction"]
    )
    assert receipt["missing_records"] == 0 and receipt["frozen_inputs_unchanged"]
    assert (
        receipt["inherited_config"]["absolute_deadline_utc"]
        != receipt["runtime_overrides"]["absolute_deadline_utc"]
    )
    assert len(engine.resets) == len(engine.adaptations) == len(engine.scores) == 2
    assert engine.resets == [seed_for(7301, "student_initialization")] * 2
    records = rows(inputs.output / "generations.jsonl")
    for offset, ep_id in enumerate(D.ORDER_IDS):
        ep = next(ep for ep in engine.episodes if ep.episode_id == ep_id)
        for index, layout in enumerate(D.LAYOUTS):
            segment = records[offset * 48 + index * 16 : offset * 48 + (index + 1) * 16]
            assert [(row["role"], row["task_id"]) for row in segment] == [
                (role, task.task_id) for role, task in D.task_order(ep, layout)
            ]
    assert all(
        call["adapter"] == "student" and "hindsight_advice" not in call["prompt"]
        for call in engine.calls[:96]
    )
    assert all(
        call["adapter"] is None and "hindsight_advice" in call["prompt"]
        for call in engine.calls[96:]
    )
    assert len(rows(inputs.output / "student-state.jsonl")) == 6
    assert all(row["unchanged"] for row in rows(inputs.output / "student-state.jsonl"))
    assert D.frozen_files(inputs.sources, inputs.reference, inputs.lessons, inputs.data) == before
    receipt_bytes = (inputs.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(inputs, engine)
    assert (inputs.output / "receipt.json").read_bytes() == receipt_bytes


@pytest.mark.parametrize("mismatch", ["tokens", "history"])
def test_mismatch_gate_persists_and_stops_before_direct_pi_without_retry(inputs, mismatch):
    engine = Engine(inputs, mismatch=mismatch)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["order_gate"] is False and not receipt["direct_pi_started"]
    assert receipt["actual_records"] == 96 and len(engine.calls) == 96
    assert receipt["work"] == {**D.PLANNED_WORK, "generation_sequences": 96}
    assert receipt["current_order_invariance"] == (mismatch == "history")
    assert receipt["history_reproduction"] == (mismatch == "tokens")


@pytest.mark.parametrize("mutate", ["student", "buffer", "base", "input"])
def test_state_or_input_mutation_is_terminal_and_blocks_direct_pi(inputs, mutate):
    engine = Engine(inputs, mutate=mutate)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert not receipt["direct_pi_started"] and receipt["status"] == "failed"
    if mutate in ("student", "buffer"):
        assert len(engine.calls) == 16
        assert rows(inputs.output / "student-state.jsonl")[0]["unchanged"] is False
    if mutate == "input":
        assert receipt["frozen_inputs_unchanged"] is False


@pytest.mark.parametrize("defect", ["source", "lessons", "config", "code", "missing"])
def test_bad_frozen_inputs_fail_before_engine_work(inputs, defect):
    if defect == "source":
        write_json(inputs.sources / "sources" / f"{D.ORDER_IDS[0]}.json", {"changed": True})
    elif defect == "lessons":
        supplied = read_json(inputs.lessons)
        supplied["rows"][0]["lesson"] += "changed"
        write_json(inputs.lessons, supplied)
    elif defect in ("config", "code"):
        parent = read_json(inputs.reference / "identity.json")
        if defect == "config":
            parent["config"]["evaluation_tokens"] += 1
        else:
            parent["code"]["model.py"] = "changed"
        write_json(inputs.reference / "identity.json", parent)
    else:
        path = inputs.reference / "observations.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    engine = Engine(inputs)
    assert invoke(inputs, engine) == 1
    assert not engine.calls and not any(engine.counters.values())


def test_failure_preserves_partial_counts_and_never_retries(inputs):
    engine = Engine(inputs, fail_on=3)
    assert invoke(inputs, engine) == 1
    receipt = read_json(inputs.output / "receipt.json")
    assert receipt["actual_records"] == receipt["work"]["generation_sequences"] == 2
    assert len(engine.calls) == 3 and receipt["error"]["message"] == "controlled interruption"
    assert not receipt["direct_pi_started"]


@pytest.mark.parametrize("limit", [0, -1, 1801, float("inf"), 1e-12])
def test_runtime_bound_stops_before_model_work(inputs, limit):
    engine = Engine(inputs)
    assert invoke(inputs, engine, max_seconds=limit) == 1
    assert not any(engine.counters.values())
