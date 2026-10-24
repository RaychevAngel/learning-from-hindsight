"""CPU numerical and whole-run evidence for matched teacher controls."""

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_model import TinyTokenizer

from hindsight.data import Episode, Task
from hindsight.experiment import read_json, rows, write_json
from hindsight.model import Generation
from hindsight.shared import TeachingExample, adapter_values, snapshot, tensor_digest
from hindsight.teacher_controls import (
    ARMS,
    TeacherRecipe,
    distribution_metrics,
    fit_teacher_mode,
    planned_work,
    replay_target,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "teacher_runner", ROOT / "scripts/check_teacher_controls.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture(params=["qwen3", "qwen35"])
def engine(request):
    if request.param == "qwen35":
        from test_qwen35 import engine as fixture
    else:
        from test_model import engine as fixture
    return fixture.__wrapped__()


def example_for(engine, recipe, ep_id, prompt, neutral=False):
    generation = Generation(prompt, engine.encode_prompt(prompt), [100, 101, 1], "ab", False)
    target = engine.score_rollout(generation, "incorrect", pi=None if neutral else "fix the rule")
    return TeachingExample(
        ep_id,
        generation,
        "incorrect",
        target,
        {
            "selection_recipe": recipe.selection_recipe(),
            "derived_from": "post_update_V",
            "selected_candidate": None if neutral else 0,
        },
    )


def test_matched_fits_share_initial_tensors_schedule_work_and_leave_actor_unchanged(engine):
    recipe = TeacherRecipe(student_steps=1, teacher_steps=2, teacher_batch_size=2)
    examples = [
        example_for(engine, recipe, f"training-{i}", f"training prompt {i}") for i in range(2)
    ]
    initial = snapshot(engine)
    evidence = []
    fits = [
        fit_teacher_mode(engine, examples, recipe, initial, mode, 11, evidence)
        for mode in ("bidirectional", "raw_causal")
    ]
    assert fits[0]["before_sha256"] == fits[1]["before_sha256"] == tensor_digest(initial)
    assert [row["source_ids"] for row in fits[0]["metrics"]] == [
        row["source_ids"] for row in fits[1]["metrics"]
    ]
    assert all(fit["after_sha256"] != fit["before_sha256"] for fit in fits)
    assert fits[0]["after_sha256"] != fits[1]["after_sha256"]
    assert tensor_digest(adapter_values(engine)) == tensor_digest(initial)
    assert engine.counters["teacher_optimizer_steps"] == 4
    assert engine.counters["teacher_example_exposures"] == 8
    assert all(row["adapter"] == "teacher" and row["forward_calls"] > 0 for row in evidence)
    with pytest.raises(ValueError, match="actual-V"):
        fit_teacher_mode(
            engine, examples, replace(recipe, student_steps=8), initial, "raw_causal", 11, evidence
        )


def test_exact_replay_and_finite_or_infinite_KL_are_explicit():
    generation = Generation("prompt", [1], [0, 1], "unused", True)
    target = replay_target(generation, 2)
    assert torch.equal(target.exp(), torch.eye(2))
    distribution = torch.tensor([[0.8, 0.2], [0.4, 0.6]]).log()
    finite = distribution_metrics(target, distribution)
    assert finite["forward_kl_reference_to_prediction"] == pytest.approx(
        -torch.log(torch.tensor(0.48)).item() / 2
    )
    infinite = distribution_metrics(distribution, target)
    assert (
        infinite["forward_kl_reference_to_prediction"] is None
        and infinite["forward_kl_is_infinite"]
    )
    assert infinite["per_position_forward_kl"] == [None, None]
    same = distribution_metrics(target, target)
    assert same["forward_kl_reference_to_prediction"] == 0
    assert not same["forward_kl_is_infinite"]
    with pytest.raises(ValueError, match="vocabulary"):
        replay_target(generation, 1)
    with pytest.raises(ValueError, match="normalized"):
        distribution_metrics(torch.ones(2, 2), distribution)


def test_action_disagreement_records_reference_direction_not_gold_correctness():
    reference = torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.5, 0.5, 0.0]]).log()
    predicted = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.2, 0.6, 0.2]]).log()
    result = distribution_metrics(reference, predicted, [0, 0, 1])
    detail = result["reference_action_disagreement"]
    assert detail["positions"] == 2 and detail["reference_strict_preference_positions"] == 1
    assert detail["reference_tied_probability_positions"] == 1
    assert detail["prediction_argmax_retained_positions"] == 0
    assert detail["prediction_argmax_copies_observed_positions"] == 2
    assert detail["strict_direction_retention_fraction"] == 0
    assert [row["response_position"] for row in detail["per_position"]] == [0, 2]
    first = detail["per_position"][0]
    assert first["observed_token_id"] == 0 and first["reference_argmax_token_id"] == 1
    assert first["reference_observed_probability"] == pytest.approx(0.2)
    assert first["prediction_observed_probability"] == pytest.approx(0.6)
    assert first["prediction_reference_argmax_probability"] == pytest.approx(0.3)
    assert detail["mean_forward_kl"] == pytest.approx(
        sum(result["per_position_forward_kl"][i] for i in (0, 2)) / 2
    )
    retained = distribution_metrics(reference, reference, [0, 0, 1])[
        "reference_action_disagreement"
    ]
    assert retained["prediction_argmax_retained_positions"] == 2
    assert retained["strict_direction_retention_fraction"] == 1
    assert retained["mean_forward_kl"] == 0
    assert "not a ground-truth error" in detail["interpretation"]


def test_empty_disagreement_and_infinite_disagreement_kl_are_not_silently_zero():
    reference = torch.tensor([[0.2, 0.8], [0.7, 0.3]]).log()
    empty = distribution_metrics(reference, reference, [1, 0])["reference_action_disagreement"]
    assert empty["positions"] == 0 and empty["per_position"] == []
    assert empty["mean_forward_kl"] is None and empty["strict_direction_retention_fraction"] is None
    replay = torch.eye(2).log()
    infinite = distribution_metrics(reference, replay, [0, 1])["reference_action_disagreement"]
    assert infinite["mean_forward_kl"] is None and infinite["forward_kl_is_infinite"]
    assert all(row["forward_kl_is_infinite"] for row in infinite["per_position"])
    for invalid in ([0], [0, 2], [False, 1]):
        with pytest.raises(ValueError, match="native IDs"):
            distribution_metrics(reference, reference, invalid)


def test_direction_is_preserved_when_reported_probabilities_underflow():
    reference = torch.tensor([[0.5, 0.2, 0.3]]).log()
    prediction = torch.tensor([[-110.0, -120.0, 0.0]])
    detail = distribution_metrics(reference, prediction, [1])["reference_action_disagreement"]
    row = detail["per_position"][0]
    assert (
        row["prediction_observed_probability"]
        == row["prediction_reference_argmax_probability"]
        == 0
    )
    assert row["prediction_reference_argmax_logprob"] == -110
    assert row["prediction_observed_logprob"] == -120
    assert row["prediction_strictly_prefers_reference_argmax"]
    assert not row["prediction_ties_reference_argmax_and_observed"]
    assert detail["strict_direction_retention_fraction"] == 1


@pytest.mark.parametrize("batch_size", [1, 4])
def test_seven_arm_run_has_fresh_queries_fixed_sources_fresh_KL_and_exact_work(
    engine, tmp_path, batch_size
):
    recipe = TeacherRecipe(student_steps=1, teacher_steps=2, teacher_batch_size=2)
    training = [example_for(engine, recipe, f"train-{i}", f"training {i}") for i in range(2)]
    evaluation = [
        example_for(engine, recipe, f"eval-{i}", f"fresh source {i}", neutral=i == 0)
        for i in range(2)
    ]
    episodes = [
        Episode(
            item.episode_id,
            "development",
            "toy",
            Task(f"source-{i}", item.generation.prompt, (0,), "toy"),
            (),
            tuple(Task(f"query-{i}-{j}", f"fresh query {i}-{j}", (0,), "toy") for j in range(4)),
        )
        for i, item in enumerate(evaluation)
    ]
    engine.counters = dict.fromkeys(engine.counters, 0)
    config = {
        "recipe": {"student_steps": 1, "teacher_steps": 2, "teacher_batch_size": 2},
        "seeds": [7],
        "max_invocation_seconds": 120,
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "model_name": engine.model_name,
        "revision": engine.revision,
        "evaluation_tokens": 2,
        "generation_batch_size": batch_size,
        "dataset_manifest_sha256": "test-corpus",
    }
    qualification = tmp_path / "qualification.json"
    write_json(
        qualification,
        {
            "status": "completed",
            "selected_batch_size": batch_size,
            "repeat_stability": {str(batch_size): True},
            "config": {"model_name": engine.model_name, "revision": engine.revision},
            "dataset_sha256": "test-corpus",
        },
    )
    config["batch_qualification"] = {
        "path": str(qualification),
        "sha256": RUNNER.sha(qualification),
    }
    output = tmp_path / "result"
    receipt = RUNNER.run(config, training, evaluation, episodes, output, engine=engine)
    assert receipt["status"] == "completed", receipt.get("error")
    assert receipt["work"] == planned_work(2, 2, 8, 1, recipe)
    assert receipt["actual_generations"] == 56
    assert receipt["base_before_sha256"] == receipt["base_after_sha256"]
    observed = rows(output / "observations.jsonl")
    assert all(
        row["role"] == "Q" and row["generation_batch_size"] == batch_size for row in observed
    )
    assert {row["condition"] for row in observed} == set(ARMS)
    cases = rows(output / "cases.jsonl")
    assert len(cases) == 14
    assert len({case["actor_before_sha256"] for case in cases}) == 1
    assert all(case["buffers_before_sha256"] == case["buffers_after_sha256"] for case in cases)
    assert all("actor_after_fidelity" in case for case in cases if case["condition"] != "baseline")
    neutral = next(
        case
        for case in cases
        if case["episode_id"] == "eval-0" and case["condition"] == "adapted_pi"
    )
    assert not neutral["actor_parameters_changed"]
    assert neutral["metrics"][0]["loss"] == 0.0
    assert len(rows(output / "teacher-steps.jsonl")) == 4
    assert len(rows(output / "fit-fidelity.jsonl")) == 4
    for record in rows(output / "fit-fidelity.jsonl"):
        for boundary in ("before", "after"):
            assert "reference_action_disagreement" in record[boundary]
    for case in cases:
        if case["condition"] != "baseline":
            for boundary in ("target_vs_selected_PI", "actor_after_vs_selected_PI"):
                assert "reference_action_disagreement" in case[boundary]
    assert read_json(output / "plan.json")["work"] == receipt["work"]
    with pytest.raises(FileExistsError):
        RUNNER.run(config, training, evaluation, episodes, output, engine=engine)


def test_declared_default_work_and_explicit_teacher_rate():
    recipe = TeacherRecipe()
    assert planned_work(32, 16, 128, 1, recipe) == {
        "generation_sequences": 896,
        "scoring_forwards": 288,
        "student_optimizer_steps": 384,
        "teacher_optimizer_steps": 128,
        "teacher_example_exposures": 512,
    }
    assert recipe.selection_recipe() == {
        "student_steps": 4,
        "learning_rate": 1e-4,
        "loss": "forward_kl",
    }
    with pytest.raises(ValueError):
        TeacherRecipe(teacher_learning_rate=0)
    with pytest.raises(ValueError):
        planned_work(3, 16, 128, 1, recipe)


def test_failure_before_engine_preserves_receipt_without_extra_work(tmp_path):
    result = RUNNER.run({}, [], [], [], tmp_path / "failure")
    assert result["status"] == "failed" and result["actual_generations"] == 0
    assert read_json(tmp_path / "failure/receipt.json")["status"] == "failed"


@pytest.mark.parametrize("indices", [[0, 2, 4, 6], list(range(8))])
def test_manifest_cohort_admission_uses_only_declared_V_subset_and_fresh_partition(
    indices, monkeypatch
):
    from hindsight.cohorts import load_cohort

    episodes, _ = load_cohort(ROOT / "data/overnight-v2")
    train_eps = [ep for ep in episodes if ep.partition == "meta_train"]
    eval_eps = [ep for ep in episodes if ep.partition == "development"]

    def example(ep):
        return TeachingExample(
            ep.episode_id,
            Generation(ep.source.prompt, [], [100], "invalid", True),
            "The submitted answer was incorrect.",
            torch.tensor([[0.0]]),
            {
                "source_task_id": ep.source.task_id,
                "selection_validation_indices": indices,
                "validation_task_ids": [ep.validation[index].task_id for index in indices],
            },
        )

    training, evaluation = list(map(example, train_eps)), list(map(example, eval_eps))
    config = {
        "training_episode_ids": [ep.episode_id for ep in train_eps],
        "evaluation_episode_ids": [ep.episode_id for ep in eval_eps],
        "evaluation_partition": "development",
    }
    assert len(RUNNER.admit(config, training, evaluation, episodes)) == 16
    corrupted = replace(
        evaluation[0],
        provenance={
            **evaluation[0].provenance,
            "validation_task_ids": [eval_eps[0].queries[0].task_id],
        },
    )
    with pytest.raises(ValueError, match="provenance"):
        RUNNER.admit(config, training, [corrupted, *evaluation[1:]], episodes)
    with pytest.raises(ValueError, match="provenance"):
        RUNNER.admit(
            {**config, "evaluation_partition": "confirmation"}, training, evaluation, episodes
        )

    def forbidden_grade(*args, **kwargs):
        pytest.fail("Structural preflight derived feedback before tokenizer admission")

    monkeypatch.setattr(RUNNER, "grade", forbidden_grade)
    assert len(RUNNER.admit(
        config, training, evaluation, episodes, check_feedback=False
    )) == 16


def test_same_hindsight_information_is_present_in_fitted_causal_control(engine):
    from hindsight.shared import alignment

    example = example_for(engine, TeacherRecipe(), "source", "observed source")
    raw = alignment(engine, example.generation, "incorrect", "raw_causal")
    teacher = alignment(engine, example.generation, "incorrect", "bidirectional")
    assert not raw["future_aware_attention"] and teacher["future_aware_attention"]
    assert len(raw["input_ids"]) != len(teacher["input_ids"])
    # Full observed attempt + feedback precede every causally scored action.
    prefix = engine.tokenizer.decode(raw["input_ids"][: raw["predecessor_positions"][0] + 1])
    assert example.generation.text in prefix and "incorrect" in prefix


@pytest.mark.parametrize("corrupt_partition", ["training", "evaluation"])
def test_custom_manifest_source_mismatch_is_rejected_before_any_fitting(
    tmp_path, monkeypatch, corrupt_partition
):
    tokenizer = TinyTokenizer()
    recipe = TeacherRecipe(student_steps=1, teacher_steps=1, teacher_batch_size=1)

    def item(name):
        text = "ab"
        return TeachingExample(
            name, Generation(name, [100], tokenizer.encode(text) + [1], text, False),
            "incorrect", torch.zeros((3, 1)),
            {"selection_recipe": recipe.selection_recipe(), "derived_from": "post_update_V"},
        )

    training, evaluation = [item("training")], [item("evaluation")]
    corrupt = training if corrupt_partition == "training" else evaluation
    corrupt[0] = replace(corrupt[0], generation=replace(corrupt[0].generation, text="different"))
    episodes = [Episode(
        "evaluation", "development", "toy", Task("s", "evaluation", (0,), "toy"), (),
        (Task("q", "query", (0,), "toy"),),
    )]
    config = {
        "recipe": {"student_steps": 1, "teacher_steps": 1, "teacher_batch_size": 1},
        "seeds": [7], "model_name": "fake", "revision": "pinned",
        "max_invocation_seconds": 60, "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "max_context_tokens": 128, "max_teacher_context_tokens": 256,
    }
    fake = SimpleNamespace(
        tokenizer=tokenizer, model_name="fake", revision="pinned",
        counters=dict.fromkeys(planned_work(1, 1, 1, 1, recipe), 0),
    )

    def forbidden_work(*args, **kwargs):
        pytest.fail("Inconsistent manifest reached model fitting/scoring/adaptation")

    for name in ("fit_teacher_mode", "score", "adapt_shared", "grade"):
        monkeypatch.setattr(RUNNER, name, forbidden_work)
    constructed = []

    def construct(**kwargs):
        constructed.append(kwargs)
        return fake

    monkeypatch.setattr(RUNNER, "HFEngine", construct)
    receipt = RUNNER.run(config, training, evaluation, episodes, tmp_path / "run", cohort=episodes)
    assert len(constructed) == 1
    assert receipt["status"] == "failed"
    assert "tokenizer-decoded native response IDs" in receipt["error"]["message"]
    assert not any(fake.counters.values()) and receipt["actual_generations"] == 0
    assert not (tmp_path / "run/teacher-steps.jsonl").exists()
