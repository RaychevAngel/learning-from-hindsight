"""Offline scalar-token probability and immutable source-prefix artifact audits."""

import dataclasses
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_prefix_interventions import D as PREFIX_TEST
from test_prefix_interventions import Engine as PrefixEngine
from test_prefix_interventions import fixture as prefix_fixture

from hindsight.data import Episode, Task, grade
from hindsight.experiment import digest, read_json, write_json
from hindsight.model import Generation

SPEC = importlib.util.spec_from_file_location(
    "analyze_prefix", Path(__file__).parents[1] / "scripts/analyze_prefix_interventions.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class Targets(PrefixEngine):
    def score_rollout(self, generation, feedback, *, pi, mode, adapter):
        assert adapter is None and mode == "causal"
        self.counters["scoring_forwards"] += 1
        logits = torch.zeros(len(generation.response_ids), 128)
        logits[:, ord("1") if D.PREFIX.FRAME_C in pi else ord("0")] = 3
        return logits.log_softmax(-1)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original = prefix_fixture.__wrapped__(tmp_path, monkeypatch)
    engine = Targets(original.output)
    receipt = read_json(original.fidelity / "receipt.json")
    plan = read_json(original.fidelity / "plan.json")
    lessons = read_json(original.fidelity / "lessons.json")
    episodes = []
    for index, ep_id in enumerate(receipt["episode_ids"]):
        prompt, text = f"public task {index}", "FINAL: [1, 0]"
        task = Task(f"task-{index}", prompt, (0, 0) if index < 4 else (1, 0), "family")
        episodes.append(Episode(ep_id, "development", "family", task, (), ()))
        generation = Generation(
            prompt, engine.encode_prompt(prompt), list(map(ord, text)), text, False
        )
        regrade = grade(text, task)
        feedback = f"The submitted answer was {'correct' if regrade['success'] else 'incorrect'}."
        source_path = original.fidelity / "sources" / f"{ep_id}.json"
        write_json(
            source_path,
            dict(
                task_id=task.task_id,
                generation=dataclasses.asdict(generation),
                feedback=feedback,
                grade=regrade,
            ),
        )
        plan["sources"][index]["source_sha256"] = D.sha(source_path)
        lessons["rows"][index]["source_sha256"] = D.sha(source_path)
        logits = torch.zeros(len(generation.response_ids), 128)
        logits[:, ord("1")] = 2
        target = logits.log_softmax(-1)
        alignment = {}
        for mode, pi in (("teacher", lessons["rows"][index]["lesson"]), ("student", None)):
            tokens, positions, _ = engine._scoring_input(generation, feedback, pi, "causal")
            alignment[mode] = dict(
                input_ids=tokens, predecessor_positions=positions, input_ids_sha256=digest(tokens)
            )
        alignment["response_token_displays"] = list(text)
        path = original.fidelity / "distributions" / f"{ep_id}.pt"
        torch.save(
            dict(
                prompt_ids=generation.prompt_ids,
                response_ids=generation.response_ids,
                target=target,
                initial=target.clone(),
                alignment=alignment,
            ),
            path,
        )
        sidecar = read_json(path.with_suffix(".json"))
        sidecar.update(source_sha256=D.sha(source_path), artifact=dict(sha256=D.sha(path)))
        write_json(path.with_suffix(".json"), sidecar)
    write_json(original.fidelity / "lessons.json", lessons)
    write_json(original.fidelity / "plan.json", plan)
    receipt["frozen_files"]["/old/frontier-lessons-balanced-qwen35-023.json"] = D.sha(
        original.fidelity / "lessons.json"
    )
    write_json(original.fidelity / "receipt.json", receipt)
    monkeypatch.setattr(D.PREFIX, "datetime", PREFIX_TEST.datetime)
    assert D.PREFIX.run(original.fidelity, original.data, original.output, engine=engine) == 0
    monkeypatch.setattr(
        D.PREFIX, "HFEngine", lambda *args, **kwargs: pytest.fail("No offline model construction")
    )
    return SimpleNamespace(
        fidelity=original.fidelity, framing=original.output, data=original.data, episodes=episodes
    )


def audit(fixture):
    return D.analyze(
        fixture.fidelity, fixture.framing, fixture.data, loader=lambda path: (fixture.episodes, [])
    )


def test_eight_sources_first_wrong_all_positions_and_no_model_work(fixture):
    result = audit(fixture)
    assert result["source_count"] == 8 and result["incorrect_source_count"] == 4
    first = result["first_wrong_summary"]
    assert first["array_values"] == first["identified_scalar_values"] == 4
    assert [first["arms"][arm]["argmax_correct_scalar"] for arm in D.ARMS] == [0, 4, 0]
    all_values = result["pooled_array_positions_descriptive"]
    assert all_values["array_values"] == all_values["native_positions"] == 16
    assert all_values["arms"]["B"]["argmax_correct_scalar"] == 12
    assert result["new_model_work"] == dict(
        generation_sequences=0, scoring_forwards=0, optimizer_steps=0
    )
    assert len(result["source_equal_array_means"]) == 3
    for source in result["sources"][:4]:
        row = source["first_wrong_array_value"]
        assert row["array_index"] == 0 and row["earlier_array_values_correct"]
        assert row["alignment_kind"] == "single_scalar_token"


def test_multitoken_and_compound_values_do_not_get_fabricated_scalar_probabilities():
    spans = D.native_values(
        ["FINAL", ": [", "1", "2", ", ", "0", "]"], [10, 11, 1, 2, 12, 0, 13], [12, 0], [0, 0]
    )
    assert spans[0]["native_positions"] == [2, 3]
    assert spans[0]["alignment_kind"] == "multi_token_value"
    assert not spans[0]["scalar_probability_identified"] and spans[0]["correct_token_id"] is None
    assert spans[0]["first_wrong_array_value"] and not spans[1]["earlier_array_values_correct"]
    compound = D.native_values(["FINAL: [", "1,", " 0", "]"], [5, 6, 7, 8], [1, 0], [0, 0])
    assert compound[0]["alignment_kind"] == "compound_or_multidigit_token"
    assert compound[0]["correct_token_id"] is None
    missing = D.native_values(["FINAL: [", "1", "]"], [5, 1, 8], [1], [0])
    assert missing[0]["single_scalar_token"] and not missing[0]["scalar_probability_identified"]


def test_probability_and_full_vocabulary_distance_math():
    old, new = torch.tensor([[0.8, 0.2]]).log(), torch.tensor([[0.3, 0.7]]).log()
    spans = [dict(native_positions=[0], correct_token_id=1)]
    row = D.policy_metrics(dict(old_PI=old, B=new, C=old), [0], spans)[0]["positions"][0]["arms"][
        "B"
    ]
    assert row["correct_scalar_probability"] == pytest.approx(0.7)
    assert row["observed_token_probability"] == pytest.approx(0.3)
    assert row["total_variation_to_old_PI"] == pytest.approx(0.5)
    assert row["forward_KL_old_PI_to_arm"] == pytest.approx(
        0.8 * math.log(0.8 / 0.3) + 0.2 * math.log(0.2 / 0.7)
    )


@pytest.mark.parametrize("defect", ["incomplete", "tensor", "alignment", "gold"])
def test_lineage_native_alignment_and_original_regrade_are_required(fixture, defect):
    if defect == "incomplete":
        path = fixture.framing / "receipt.json"
        value = read_json(path)
        value["status"] = "failed"
        write_json(path, value)
    elif defect == "gold":
        ep = fixture.episodes[4]
        fixture.episodes[4] = dataclasses.replace(
            ep, source=dataclasses.replace(ep.source, answer=(0, 0))
        )
    else:
        path = fixture.framing / "distributions/source-0.pt"
        if defect == "tensor":
            path.write_bytes(b"changed")
        else:
            value = torch.load(path, weights_only=True)
            value["alignment"]["B"]["predecessor_positions"][0] -= 1
            torch.save(value, path)
            sidecar = read_json(path.with_suffix(".json"))
            sidecar["artifact"].update(sha256=D.sha(path), bytes=path.stat().st_size)
            write_json(path.with_suffix(".json"), sidecar)
    with pytest.raises(ValueError):
        audit(fixture)
