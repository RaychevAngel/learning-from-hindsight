"""Real tiny Qwen CPU proof of shared storage, both objectives and five-arm flow."""

import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_model import TinyTokenizer
from test_model import engine as original_engine

from hindsight.data import Episode, Task
from hindsight.experiment import append_json, read_json, rows, write_json
from hindsight.model import Generation
from hindsight.shared import (
    ARMS,
    SharedRecipe,
    TeachingExample,
    adapt_shared,
    adapter_values,
    alignment,
    audit_forward,
    distances,
    fit_shared,
    planned_work,
    restore,
    score,
    snapshot,
    storage_map,
    tensor_digest,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shared_runner", ROOT / "scripts/check_shared_learning.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture(params=["qwen3", "qwen35"])
def engine(request):
    if request.param == "qwen35":
        from test_qwen35 import engine as hybrid_engine

        return hybrid_engine.__wrapped__()
    return original_engine.__wrapped__()


def examples_for(engine, recipe, count=2):
    examples = []
    for index in range(count):
        prompt = f"training {index}"
        generation = Generation(prompt, engine.encode_prompt(prompt), [100, 101, 1], "ab", False)
        target = engine.score_rollout(generation, "incorrect", pi="correct the rule")
        examples.append(
            TeachingExample(
                f"train-{index}",
                generation,
                "incorrect",
                target,
                {"selection_recipe": recipe.selection_recipe(), "derived_from": "post_update_V"},
            )
        )
    return examples


def test_hindsight_and_causal_training_use_same_live_parameters_and_keep_control_isolated(engine):
    recipe = SharedRecipe(student_steps=2, teacher_steps=2, teacher_batch_size=2)
    examples = examples_for(engine, recipe)
    initial = snapshot(engine)
    teacher = snapshot(engine, "teacher")
    pointers = storage_map(engine)
    evidence = []
    fitted = fit_shared(engine, examples, recipe, 7, evidence)
    assert fitted["before_sha256"] != fitted["after_sha256"]
    assert tensor_digest(teacher) == tensor_digest(adapter_values(engine, "teacher"))
    example = examples[0]
    target = score(
        engine,
        example.generation,
        example.feedback,
        mode="bidirectional",
        adapter="student",
        evidence=evidence,
    )
    fixed_target = target.clone()
    update = adapt_shared(engine, example.generation, target, recipe, evidence)
    assert update["before_sha256"] == fitted["after_sha256"]
    assert update["before_sha256"] != update["after_sha256"]
    assert torch.equal(target, fixed_target) and not target.requires_grad
    assert storage_map(engine) == pointers
    assert all(
        row["adapter"] == "student" and row["forward_calls"] > 0 and row["same_parameter_storage"]
        for row in evidence
    )
    restore(engine, initial)
    assert tensor_digest(adapter_values(engine)) == tensor_digest(initial)
    assert not engine._student_optimizer.state


def test_generation_hook_observes_underlying_peft_generate_calls(engine):
    evidence = []
    with audit_forward(engine, "student", evidence):
        engine.generate(["hello"], max_new_tokens=2, sample=False, seed=1, adapter="student")
    assert evidence[0]["forward_calls"] >= 1
    with pytest.raises(RuntimeError, match="Forward adapter"):
        with audit_forward(engine, "student", evidence):
            engine.generate(["hello"], max_new_tokens=2, sample=False, seed=1, adapter="teacher")


def test_shared_future_visibility_causal_locality_and_native_alignment(engine):
    recipe = SharedRecipe(teacher_steps=1, teacher_batch_size=1)
    example = examples_for(engine, recipe, count=1)[0]
    changed = replace(example.generation, response_ids=[110, 101, 1])
    evidence = []
    causal = score(
        engine, example.generation, "", mode="causal", adapter="student", evidence=evidence
    )
    causal_changed = score(engine, changed, "", mode="causal", adapter="student", evidence=evidence)
    future = score(
        engine,
        example.generation,
        "wrong",
        mode="bidirectional",
        adapter="student",
        evidence=evidence,
    )
    future_changed = score(
        engine, changed, "wrong", mode="bidirectional", adapter="student", evidence=evidence
    )
    assert torch.equal(causal[0], causal_changed[0])
    assert not torch.equal(future[0], future_changed[0])
    for mode in ("causal", "bidirectional"):
        result = alignment(engine, example.generation, "wrong", mode)
        assert [
            result["input_ids"][i + 1] for i in result["predecessor_positions"]
        ] == example.generation.response_ids
    with pytest.raises(ValueError, match="prompt IDs"):
        alignment(engine, replace(example.generation, prompt_ids=[1, 2]), "", "causal")


def test_wrong_recipe_labels_and_non_V_provenance_fail_before_updates(engine):
    recipe = SharedRecipe(teacher_steps=1, teacher_batch_size=1)
    example = examples_for(engine, recipe, 1)[0]
    before = snapshot(engine)
    with pytest.raises(ValueError, match="another student recipe"):
        fit_shared(engine, [example], replace(recipe, student_steps=8), 0, [])
    with pytest.raises(ValueError, match="post-update V"):
        fit_shared(
            engine,
            [replace(example, provenance={**example.provenance, "derived_from": "Q"})],
            recipe,
            0,
            [],
        )
    assert tensor_digest(before) == tensor_digest(adapter_values(engine))


def test_snapshot_cross_adapter_copy_is_exact_but_storage_distinct(engine):
    recipe = SharedRecipe(teacher_steps=2, teacher_batch_size=1)
    example = examples_for(engine, recipe, 1)
    fit_shared(engine, example, recipe, 0, [])
    fitted = snapshot(engine)
    restore(engine, fitted, "teacher")
    assert tensor_digest(adapter_values(engine)) == tensor_digest(adapter_values(engine, "teacher"))
    assert {x["data_ptr"] for x in storage_map(engine).values()}.isdisjoint(
        x["data_ptr"] for x in storage_map(engine, "teacher").values()
    )
    with pytest.raises(ValueError, match="parameter names"):
        restore(engine, {})


@pytest.mark.parametrize("batch_size", [1, 4])
def test_five_arms_complete_and_count_real_model_work(engine, tmp_path, batch_size):
    recipe = SharedRecipe(student_steps=1, teacher_steps=2, teacher_batch_size=2)
    examples = examples_for(engine, recipe)
    engine.counters = dict.fromkeys(engine.counters, 0)
    episodes = [
        Episode(
            f"fresh-{i}",
            "development",
            "toy",
            Task(f"source-{i}", f"fresh source {i}", (0,), "toy"),
            (),
            tuple(Task(f"query-{i}-{j}", f"fresh query {i} {j}", (0,), "toy") for j in range(4)),
        )
        for i in range(2)
    ]
    config = {
        "recipe": {"student_steps": 1, "teacher_steps": 2, "teacher_batch_size": 2},
        "seeds": [7],
        "max_invocation_seconds": 120,
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "model_name": engine.model_name,
        "revision": engine.revision,
        "source_tokens": 2,
        "evaluation_tokens": 2,
        "generation_batch_size": batch_size,
        "dataset_manifest_sha256": "test-corpus",
    }
    qualification = tmp_path / "qualification.json"
    write_json(
        qualification,
        {
            "status": "completed",
            "selected_batch_size": 4,
            "repeat_stability": {"4": True},
            "config": {"model_name": engine.model_name, "revision": engine.revision},
            "dataset_sha256": config["dataset_manifest_sha256"],
        },
    )
    config["batch_qualification"] = {
        "path": str(qualification),
        "sha256": RUNNER.sha(qualification),
    }
    output = tmp_path / "run"
    receipt = RUNNER.run(config, examples, episodes, output, engine=engine)
    assert receipt["status"] == "completed", receipt.get("error")
    assert receipt["work"] == planned_work(2, 2, 8, 1, recipe)
    assert receipt["actual_generations"] == 50 and receipt["missing_generations"] == 0
    assert receipt["base_before_sha256"] == receipt["base_after_sha256"]
    assert receipt["same_storage_audits_passed"] and receipt["audited_forward_calls"] > 0
    cases = rows(output / "cases.jsonl")
    assert len(cases) == 10 and {row["arm"] for row in cases} == set(ARMS)
    assert all(row["teacher_adapter"] == "student" for row in cases if row["arm"] == "shared")
    assert all(row["teacher_adapter"] == "teacher" for row in cases if row["arm"] == "separate")
    assert all(row["buffers_before_sha256"] == row["buffers_after_sha256"] for row in cases)
    assert all(
        row["teacher_distribution_drift"]["forward_kl_reference_to_prediction"] == 0
        for row in cases
        if row["arm"] == "separate"
    )
    outcomes = rows(output / "generations.jsonl")
    assert all("hindsight_advice" not in row["generation"]["prompt"] for row in outcomes)
    assert all(row["generation_batch_size"] == batch_size for row in outcomes if row["role"] == "Q")
    assert len(read_json(output / "plan.json")["generations"]) == 50
    with pytest.raises(FileExistsError):
        RUNNER.run(config, examples, episodes, output, engine=engine)
    if batch_size == 4:
        config["batch_qualification"]["sha256"] = "wrong"
        with pytest.raises(ValueError, match="checksum"):
            RUNNER.run(config, examples, episodes, tmp_path / "bad-receipt", engine=engine)


def teaching_manifest(tmp_path, engine, recipe):
    example = examples_for(engine, recipe, 1)[0]
    source_path = tmp_path / "source.json"
    target_path = tmp_path / "target.pt"
    observations_path = tmp_path / "observations.jsonl"
    write_json(
        source_path,
        {"generation": asdict(example.generation), "feedback": "incorrect", "task_id": "source-id"},
    )
    torch.save(example.target, target_path)
    for candidate, successes in [(None, (False, False)), (0, (True, False)), (1, (True, True))]:
        for index, success in enumerate(successes):
            append_json(
                observations_path,
                {
                    "episode_id": example.episode_id,
                    "task_id": f"validation-{index}",
                    "role": "V",
                    "condition": "baseline" if candidate is None else "adapted_pi",
                    "candidate": candidate,
                    "success": success,
                },
            )
    # A Q-only apparent win must never select the inferior candidate.
    append_json(
        observations_path,
        {"episode_id": example.episode_id, "role": "Q", "condition": "adapted_pi", "candidate": 0},
    )
    manifest = {
        "format": "shared-teaching-v1",
        "selection_recipe": recipe.selection_recipe(),
        "selection_observations_path": observations_path.name,
        "selection_observations_sha256": RUNNER.sha(observations_path),
        "examples": [
            {
                "episode_id": example.episode_id,
                "selected_candidate": 1,
                "source_path": source_path.name,
                "source_sha256": RUNNER.sha(source_path),
                "target_path": target_path.name,
                "target_sha256": RUNNER.sha(target_path),
            }
        ],
    }
    path = tmp_path / "teaching.json"
    write_json(path, manifest)
    return path, manifest


def test_target_manifest_admits_actual_V_selection_and_ignores_Q(engine, tmp_path):
    recipe = SharedRecipe()
    path, manifest = teaching_manifest(tmp_path, engine, recipe)
    examples, _ = RUNNER.load_teaching(path, recipe)
    assert examples[0].provenance["candidate_successes"] == {0: 1, 1: 2}
    assert examples[0].provenance["source_task_id"] == "source-id"
    for selected in (0, None, True):
        manifest["examples"][0]["selected_candidate"] = selected
        write_json(path, manifest)
        with pytest.raises(ValueError, match="selection|candidate"):
            RUNNER.load_teaching(path, recipe)


def test_target_manifest_recipe_checksum_and_neutral_tie_fail_closed(engine, tmp_path):
    recipe = SharedRecipe()
    path, manifest = teaching_manifest(tmp_path, engine, recipe)
    with pytest.raises(ValueError, match="another student recipe"):
        RUNNER.load_teaching(path, replace(recipe, student_steps=8))
    manifest["selection_observations_sha256"] = "changed"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="checksum"):
        RUNNER.load_teaching(path, recipe)
    neutral = tmp_path / "neutral.jsonl"
    for row in rows(tmp_path / manifest["selection_observations_path"]):
        append_json(neutral, {**row, "success": False})
    manifest.update(
        selection_observations_path=neutral.name,
        selection_observations_sha256=RUNNER.sha(neutral),
    )
    manifest["examples"][0]["selected_candidate"] = None
    write_json(path, manifest)
    assert RUNNER.load_teaching(path, recipe)[0][0].provenance["selected_candidate"] is None
    manifest["examples"][0]["selected_candidate"] = 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match="no-update tie"):
        RUNNER.load_teaching(path, recipe)


def test_work_plan_and_distance_math():
    assert planned_work(8, 8, 64, 2, SharedRecipe()) == {
        "generation_sequences": 720,
        "scoring_forwards": 208,
        "student_optimizer_steps": 192,
        "teacher_optimizer_steps": 128,
        "teacher_example_exposures": 512,
    }
    q = torch.tensor([[0.8, 0.2]]).log()
    p = torch.tensor([[0.5, 0.5]]).log()
    result = distances(q, p)
    assert result["forward_kl_reference_to_prediction"] == pytest.approx(
        float((q.exp() * (q - p)).sum()), abs=1e-7
    )
    assert result["mean_total_variation"] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        SharedRecipe(loss="unspecified_new_objective")


@pytest.mark.parametrize("indices", [[0, 2, 4, 6], list(range(8))])
def test_actual_cohort_admits_declared_V4_or_V8_but_no_other_tasks(indices, monkeypatch):
    from hindsight.cohorts import load_cohort

    episodes, _ = load_cohort(ROOT / "data/overnight-v2")
    training = next(ep for ep in episodes if ep.partition == "meta_train")
    evaluation = [ep.episode_id for ep in episodes if ep.partition == "development"][:8]
    example = TeachingExample(
        training.episode_id,
        Generation(training.source.prompt, [], [100], "invalid", True),
        "The submitted answer was incorrect.",
        torch.tensor([[0.0]]),
        {
            "source_task_id": training.source.task_id,
            "selection_validation_indices": indices,
            "validation_task_ids": [training.validation[i].task_id for i in indices],
        },
    )
    selected = RUNNER.admit_cohort([example], episodes, evaluation)
    assert len(selected) == 8 and all(ep.partition == "development" for ep in selected)
    bad = replace(
        example,
        provenance={**example.provenance, "validation_task_ids": [training.queries[0].task_id]},
    )
    with pytest.raises(ValueError, match="source/V provenance"):
        RUNNER.admit_cohort([bad], episodes, evaluation)
    with pytest.raises(ValueError, match="source grade"):
        RUNNER.admit_cohort([replace(example, feedback="a leaked answer")], episodes, evaluation)

    def forbidden_grade(*args, **kwargs):
        pytest.fail("Structural preflight derived feedback before tokenizer admission")

    monkeypatch.setattr(RUNNER, "grade", forbidden_grade)
    assert len(RUNNER.admit_cohort(
        [example], episodes, evaluation, check_feedback=False
    )) == 8


def test_same_named_adapter_with_replaced_parameter_storage_is_rejected(engine):
    name, value = next(iter(engine._adapter_parameters("student").items()))
    module_name, key = name.rsplit(".", 1)
    module = engine.model.get_submodule(module_name)
    evidence = []
    try:
        with pytest.raises(RuntimeError, match="live parameter storage"):
            with audit_forward(engine, "student", evidence):
                module._parameters[key] = torch.nn.Parameter(value.detach().clone())
                engine.generate(
                    ["hello"], max_new_tokens=1, sample=False, seed=1, adapter="student"
                )
    finally:
        module._parameters[key] = value
    assert not evidence[0]["same_parameter_storage"]


def test_custom_training_source_mismatch_is_rejected_before_scoring_or_fitting(
    tmp_path, monkeypatch
):
    tokenizer = TinyTokenizer()
    recipe = SharedRecipe(student_steps=1, teacher_steps=1, teacher_batch_size=1)
    example = TeachingExample(
        "training", Generation("training", [100], tokenizer.encode("ab") + [1], "different", False),
        "incorrect", torch.zeros((3, 1)),
        {"selection_recipe": recipe.selection_recipe(), "derived_from": "post_update_V"},
    )
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

    for name in ("fit_shared", "score", "adapt_shared", "grade"):
        monkeypatch.setattr(RUNNER, name, forbidden_work)
    constructed = []

    def construct(**kwargs):
        constructed.append(kwargs)
        return fake

    monkeypatch.setattr(RUNNER, "HFEngine", construct)
    receipt = RUNNER.run(config, [example], episodes, tmp_path / "run", cohort=episodes)
    assert len(constructed) == 1
    assert receipt["status"] == "failed"
    assert "tokenizer-decoded native response IDs" in receipt["error"]["message"]
    assert not any(fake.counters.values()) and receipt["actual_generations"] == 0


def test_new_shared_evaluation_source_is_checked_before_feedback_and_teaching(
    engine, tmp_path, monkeypatch
):
    recipe = SharedRecipe(student_steps=1, teacher_steps=1, teacher_batch_size=1)
    examples = examples_for(engine, recipe, 1)
    engine.counters = dict.fromkeys(engine.counters, 0)
    episodes = [Episode(
        "evaluation", "development", "toy", Task("s", "evaluation", (0,), "toy"), (),
        (Task("q", "query", (0,), "toy"),),
    )]
    config = {
        "recipe": {"student_steps": 1, "teacher_steps": 1, "teacher_batch_size": 1},
        "seeds": [7], "model_name": engine.model_name, "revision": engine.revision,
        "source_tokens": 2, "evaluation_tokens": 2,
        "max_invocation_seconds": 120, "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
    }
    generate = engine.generate

    def inconsistent(*args, **kwargs):
        # Preserve actual tiny forward calls, but make the returned presentation
        # contradict those actions, as an erroneous custom generator could.
        return [
            replace(value, text=value.text + "different")
            for value in generate(*args, **kwargs)
        ]

    def forbidden_work(*args, **kwargs):
        pytest.fail("Inconsistent new source reached grading or adaptation")

    monkeypatch.setattr(engine, "generate", inconsistent)
    monkeypatch.setattr(RUNNER, "grade", forbidden_work)
    monkeypatch.setattr(RUNNER, "adapt_shared", forbidden_work)
    receipt = RUNNER.run(config, examples, episodes, tmp_path / "run", engine=engine)
    assert receipt["status"] == "failed"
    assert "tokenizer-decoded native response IDs" in receipt["error"]["message"]
    assert receipt["actual_generations"] == 0 and receipt["completed_cases"] == []
    assert receipt["work"]["student_optimizer_steps"] == 0
    assert not (tmp_path / "run/generations.jsonl").exists()
