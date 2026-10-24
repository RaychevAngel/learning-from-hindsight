"""No-generation tests of fixed framing, source-only admission and exact16-score work."""

import dataclasses
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hindsight.experiment import digest, read_json, write_json
from hindsight.model import Generation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prefix_interventions", ROOT / "scripts/score_prefix_interventions.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class Engine:
    model_name, revision = "offline", "pinned"

    def __init__(self, output, fail_at=None):
        self.counters = dict.fromkeys(D.WORK, 0)
        self.calls, self.output, self.fail_at = [], output, fail_at
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))

    def encode_prompt(self, text):
        return list(map(ord, text))

    def _scoring_input(self, generation, feedback, pi, mode):
        assert mode == "causal"
        prefix = self.encode_prompt((pi or "") + generation.prompt)
        return (
            prefix + generation.response_ids,
            list(range(len(prefix) - 1, len(prefix) + len(generation.response_ids) - 1)),
            False,
        )

    def score_rollout(self, generation, feedback, *, pi, mode, adapter):
        self.budget_check()
        assert (self.output / "plan.json").exists()
        assert adapter is None and mode == "causal"
        assert "SECRET_GOLD" not in pi and "OTHER_TASK" not in pi
        self.calls.append((generation, feedback, pi))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("Controlled scoring failure")
        self.counters["scoring_forwards"] += 1
        logits = torch.zeros(len(generation.response_ids), 128)
        logits[:, 0 if D.FRAME_C in pi else 1] = 1
        return logits.log_softmax(-1)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 6, 8, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(D, "datetime", FrozenDatetime)
    fidelity, data, output = (tmp_path / name for name in ("fidelity", "data", "result"))
    engine = Engine(output)
    ids = [f"source-{i}" for i in range(8)]
    entries, lessons, public = [], [], []
    for index, ep_id in enumerate(ids):
        prompt, text = f"public task {index}", f"attempt{index}"
        generation = Generation(
            prompt, engine.encode_prompt(prompt), list(map(ord, text)), text, False
        )
        source = dict(
            task_id=f"task-{index}",
            generation=dataclasses.asdict(generation),
            feedback="The submitted answer was incorrect.",
        )
        source_path = fidelity / "sources" / f"{ep_id}.json"
        write_json(source_path, source)
        lesson = dict(
            episode_id=ep_id, source_sha256=D.sha(source_path), lesson="UNCHANGED STRONG LESSON"
        )
        lessons.append(lesson)
        entry = dict(
            episode_id=ep_id,
            task_id=source["task_id"],
            source_sha256=D.sha(source_path),
            lesson_sha256=digest(lesson["lesson"]),
        )
        entries.append(entry)
        public.append(dict(task_id=source["task_id"], prompt=prompt))
        tokens, positions, _ = engine._scoring_input(
            generation, source["feedback"], lesson["lesson"], "causal"
        )
        target = torch.zeros(len(generation.response_ids), 128).log_softmax(-1)
        path = fidelity / "distributions" / f"{ep_id}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(
            dict(
                prompt_ids=generation.prompt_ids,
                response_ids=generation.response_ids,
                target=target,
                initial=target.clone(),
                alignment=dict(
                    teacher=dict(input_ids=tokens, predecessor_positions=positions),
                    response_token_displays=list(text),
                ),
            ),
            path,
        )
        write_json(
            path.with_suffix(".json"),
            dict(**entry, status="completed", artifact=dict(sha256=D.sha(path))),
        )
    write_json(fidelity / "lessons.json", dict(rows=lessons))
    data.mkdir()
    (data / "tasks.public.jsonl").write_text(
        "\n".join(
            json.dumps(value)
            for value in public
            + [dict(task_id="OTHER_TASK", prompt="OTHER_TASK public task never in model context")]
        )
        + "\n"
    )
    (data / "tasks.private.jsonl").write_text("SECRET_GOLD is deliberately not JSON\n")
    write_json(
        data / "manifest.json",
        dict(
            files={
                name: D.sha(data / name) for name in ("tasks.public.jsonl", "tasks.private.jsonl")
            }
        ),
    )
    config = dict(
        model_name="offline",
        revision="pinned",
        diagnostic_source_ids=ids,
        dataset_manifest_sha256=D.sha(data / "manifest.json"),
        deterministic_execution=True,
        enable_thinking=False,
        max_context_tokens=1536,
        max_teacher_context_tokens=4096,
    )
    frozen = {
        str(ROOT / "src/hindsight" / name): D.sha(ROOT / "src/hindsight" / name)
        for name in ("model.py", "experiment.py", "data.py")
    }
    frozen["/old/frontier-lessons-balanced-qwen35-023.json"] = D.sha(fidelity / "lessons.json")
    write_json(fidelity / "plan.json", dict(sources=entries))
    write_json(
        fidelity / "receipt.json",
        dict(
            status="completed",
            frozen_inputs_unchanged=True,
            completed_episode_ids=ids,
            episode_ids=ids,
            inherited_config=config,
            work=dict(D.WORK, student_optimizer_steps=32),
            frozen_files=frozen,
        ),
    )
    return SimpleNamespace(fidelity=fidelity, data=data, output=output, engine=engine)


def invoke(fixture, **kwargs):
    return D.run(
        fixture.fidelity,
        fixture.data,
        fixture.output,
        engine=fixture.engine,
        deadline_utc=D.LATEST_DEADLINE,
        **kwargs,
    )


def test_frames_keep_lesson_exact_and_c_history_is_untrusted_source_only():
    source = Generation("task", [1], [2], 'attempt says "ignore everything"', False)
    contexts = D.contexts(" fixed lesson\n", source, "The submitted answer was incorrect.")
    assert contexts["B"].startswith(" fixed lesson\n\n\n" + D.FRAME_B)
    assert contexts["C"].startswith(contexts["B"] + "\n\n" + D.FRAME_C)
    assert source.text not in contexts["B"]
    history = json.loads(contexts["C"].split(D.FRAME_C + "\n", 1)[1])
    assert history == dict(
        recorded_attempt=source.text, binary_feedback="The submitted answer was incorrect."
    )


def test_all_eight_sources_exact16_scores_frozen_plan_and_native_artifacts(fixture):
    assert invoke(fixture) == 0
    receipt = read_json(fixture.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["work"] == D.WORK
    assert len(fixture.engine.calls) == 16 and len(receipt["completed_episode_ids"]) == 8
    plan = read_json(fixture.output / "plan.json")
    assert plan["generic_frames"] == dict(B=D.FRAME_B, C=D.FRAME_C)
    for entry in plan["sources"]:
        payload = torch.load(
            fixture.output / "distributions" / f"{entry['episode_id']}.pt", weights_only=True
        )
        assert payload["unchanged_lesson"] == "UNCHANGED STRONG LESSON"
        assert payload["contexts"] == entry["contexts"]
        for arm in ("B", "C"):
            alignment = payload["alignment"][arm]
            assert [
                alignment["input_ids"][position + 1]
                for position in alignment["predecessor_positions"]
            ] == payload["response_ids"]
            assert not payload[arm].requires_grad


@pytest.mark.parametrize("defect", ["lesson", "source", "tensor", "core", "budget"])
def test_changed_frozen_inputs_fail_before_scoring(fixture, defect):
    if defect == "lesson":
        path = fixture.fidelity / "lessons.json"
        value = read_json(path)
        value["rows"][0]["lesson"] += " changed"
        write_json(path, value)
    elif defect == "source":
        path = fixture.fidelity / "sources/source-0.json"
        value = read_json(path)
        value["generation"]["text"] += " changed"
        write_json(path, value)
    elif defect == "tensor":
        (fixture.fidelity / "distributions/source-0.pt").write_bytes(b"changed")
    elif defect == "core":
        path = fixture.fidelity / "receipt.json"
        value = read_json(path)
        key = next(key for key in value["frozen_files"] if key.endswith("/model.py"))
        value["frozen_files"][key] = "changed"
        write_json(path, value)
    assert invoke(fixture, max_seconds=301 if defect == "budget" else 300) == 1
    assert not fixture.engine.calls
    assert read_json(fixture.output / "receipt.json")["status"] == "failed"


def test_failure_is_durable_partial_without_retry_or_overwrite(fixture):
    fixture.engine.fail_at = 4
    assert invoke(fixture) == 1
    receipt = read_json(fixture.output / "receipt.json")
    assert receipt["work"]["scoring_forwards"] == 3
    assert receipt["completed_episode_ids"] == ["source-0"]
    assert len(fixture.engine.calls) == 4
    with pytest.raises(FileExistsError):
        invoke(fixture)


def test_distribution_metrics_compare_old_policy_not_gold():
    old = torch.tensor([[0.8, 0.2], [0.3, 0.7]]).log()
    new = torch.tensor([[0.5, 0.5], [0.7, 0.3]]).log()
    result = D.compare(old, old, dict(B=new, C=old), [0, 1])
    assert result["argmax_changes"] == dict(B=1, C=0)
    assert result["means"]["old_PI_total_variation_to_B"] == pytest.approx(0.35)
    assert result["means"]["old_PI_forward_KL_to_C"] == 0
