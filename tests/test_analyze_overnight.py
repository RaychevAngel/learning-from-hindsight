"""Offline exact-matrix audits and source-level statistics for all four run kinds."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pi_comparison import D as COMPARISON
from test_pi_comparison import Engine as ComparisonEngine
from test_proposer_training import FakeEngine, invoke
from test_proposer_training import cohort as proposer_cohort

from hindsight.analysis import paired_interval
from hindsight.data import grade
from hindsight.experiment import digest, read_json, rows, write_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "overnight_analysis", ROOT / "scripts/analyze_overnight.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture
def cohort(tmp_path):
    return proposer_cohort.__wrapped__(tmp_path)


def report(kind, cohort):
    return D.analyze(
        kind,
        cohort.output,
        cohort.data,
        loader=lambda path: (cohort.episodes, []),
        bootstrap_samples=100,
        bootstrap_seed=73,
    )


def comparison_run(cohort, monkeypatch, invalid=False):
    cohort.episodes = [
        dataclasses.replace(ep, partition="development", family=f"f{i}")
        for i, ep in enumerate(cohort.episodes[:2])
    ]
    cohort.config.update(partition="development", expected_sources=2, enable_thinking=False)
    monkeypatch.setattr(COMPARISON, "load_cohort", lambda path: (cohort.episodes, []))
    assert (
        COMPARISON.run(
            cohort.config,
            cohort.data,
            cohort.output,
            engine=ComparisonEngine(cohort, invalid=invalid),
        )["status"]
        == "completed"
    )


def test_complete_comparison_uses_full_pool_and_actual_update_selection_on_q(cohort, monkeypatch):
    comparison_run(cohort, monkeypatch)
    result = report("comparison", cohort)
    assert result["denominator_records"] == 352 and result["source_count"] == 2
    decision = result["per_seed"]["7301"]["decisions"]
    assert decision["selector_disagreements"] == 2 and decision["tie_only_disagreements"] == 0
    assert [row["actual_update_choice"] for row in decision["candidate_pool"]] == [1, 1]
    assert [row["direct_choice"] for row in decision["candidate_pool"]] == [2, 2]
    e3 = decision["paired_decision_contrasts"][-1]
    assert e3["left"]["correct"] == 16 and e3["right"]["correct"] == 0
    assert e3["episodes"] == 2 and e3["left"]["denominator"] == 16
    assert e3["gains"] == 16 and e3["losses"] == 0
    assert e3["ci95"] == [1.0, 1.0]
    for source in decision["candidate_pool"]:
        assert source["direct_selected_actual_V_regret_successes"] == 8
        assert source["direct_selected_actual_V_regret_fraction"] == 1
        assert source["direct_V_selected_student_Q_successes"] == 0
        assert source["actual_V_selected_student_Q_successes"] == 8
        assert source["actual_minus_direct_selected_student_Q_successes"] == 8
        assert len(source["within_source_rankings"]["direct_V_vs_student_V"]["pairs"]) == 6
    assert decision["ranking_summary"]["direct_V_vs_student_V"]["sources"] == 2
    assert decision["direct_selected_actual_V_regret"]["mean_fraction"] == 1
    assert decision["direct_selected_actual_V_regret"]["positive_regret_sources"] == 2
    assert "improved_episodes" not in decision["direct_selected_actual_V_regret"]


def test_invalid_candidate_matrix_is_literal_baseline_reuse_and_neutral_selection(
    cohort, monkeypatch
):
    comparison_run(cohort, monkeypatch, invalid=True)
    result = report("comparison", cohort)
    assert result["work"]["student_optimizer_steps"] == 0
    assert result["work"]["scoring_forwards"] == 2
    assert all(
        row["actual_update_choice"] is None
        for row in result["per_seed"]["7301"]["decisions"]["candidate_pool"]
    )
    decisions = result["per_seed"]["7301"]["decisions"]
    assert decisions["direct_selected_actual_V_regret"]["mean_fraction"] == 0
    for summary in decisions["ranking_summary"].values():
        assert summary["defined_sources"] == 0 and summary["undefined_sources"] == 2
        assert summary["mean_within_source_spearman"] is None


def test_within_source_rankings_retain_ties_and_all_six_candidate_pairs():
    positive = D.candidate_ranking([0, 1, 2, 3], [1, 2, 3, 4])
    negative = D.candidate_ranking([0, 1, 2, 3], [4, 3, 2, 1])
    assert positive["spearman"] == 1 and positive["pair_counts"]["concordant"] == 6
    assert negative["spearman"] == -1 and negative["pair_counts"]["discordant"] == 6
    tied = D.candidate_ranking([0, 0, 1, 2], [0, 1, 1, 2])
    assert tied["left_average_ranks"] == [1.5, 1.5, 3, 4]
    assert tied["spearman"] == pytest.approx(5 / 6)
    assert tied["pair_counts"] == dict(
        concordant=4, discordant=0, both_tied=0, left_tied=1, right_tied=1
    )
    assert tied["non_tied_order_agreement"] == 1
    assert tied["exact_order_agreement"] == pytest.approx(4 / 6)


@pytest.mark.parametrize(
    "right,reason", [([1, 2, 3, 4], "left_constant"), ([1] * 4, "both_constant")]
)
def test_constant_candidate_scores_are_undefined_not_zero_correlation(right, reason):
    ranking = D.candidate_ranking([1] * 4, right)
    assert ranking["spearman"] is None and ranking["spearman_undefined_reason"] == reason
    assert ranking["non_tied_pair_count"] == 0
    assert ranking["non_tied_order_agreement"] is None
    assert sum(ranking["pair_counts"].values()) == 6


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "grade",
        "native",
        "work",
        "selected_by_q",
        "wrong_selected",
        "pi_leak",
    ],
)
def test_corrupted_comparison_fails_instead_of_summarizing_partial_or_changed_results(
    cohort, monkeypatch, defect
):
    comparison_run(cohort, monkeypatch)
    path = cohort.output / "observations.jsonl"
    values = rows(path)
    if defect == "missing":
        values.pop()
    elif defect == "duplicate":
        values.append(values[-1])
    elif defect == "grade":
        values[-1]["success"] = not values[-1]["success"]
    elif defect == "native":
        values[-1]["generation"]["response_ids"] = []
    elif defect == "pi_leak":
        values[-1]["generation"]["prompt"] += " privileged advice"
    elif defect == "work":
        path = cohort.output / "receipt.json"
        value = read_json(path)
        value["work"]["generation_sequences"] += 1
        write_json(path, value)
        values = None
    else:
        path = cohort.output / "selections.jsonl"
        values = rows(path)
        if defect == "selected_by_q":
            values[0]["derived_from"] = "Q"
        else:
            values[0]["selected_candidate"] = 2
    if values is not None:
        path.write_text("\n".join(json.dumps(value) for value in values) + "\n")
    with pytest.raises(ValueError):
        report("comparison", cohort)


def test_proposer_frozen_initial_and_trained_are_paired_without_training_counts(cohort, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    write_json(
        checkpoint,
        dict(
            weight=9,
            metadata=dict(
                dataset_manifest_sha256=D.sha(cohort.data / "manifest.json"),
                train_episode_ids=["e0", "e1"],
                student_recipe={
                    key: cohort.config[key] for key in ("student_steps", "learning_rate", "seed")
                },
                proposer_temperature=0.7,
                proposer_module_sha256=D.sha(ROOT / "src/hindsight/proposer.py"),
            ),
        ),
    )
    assert invoke(cohort, FakeEngine(cohort), phase="evaluate", checkpoint=checkpoint) == 0
    result = report("proposer", cohort)
    assert result["source_count"] == 1 and result["denominator_records"] == 48
    assert result["work"]["proposer_optimizer_steps"] == 0
    q = result["per_seed"]["7301"]["trained_minus_initial"]["Q"]
    assert q["episodes"] == 1 and q["ci95"] is None and q["gains"] == 8


def synthetic_multiseed(cohort, kind):
    cohort.episodes = [
        dataclasses.replace(ep, partition="development", family=f"f{i}")
        for i, ep in enumerate(cohort.episodes[:2])
    ]
    config = dict(
        cohort.config,
        evaluation_episode_ids=[ep.episode_id for ep in cohort.episodes],
        training_episode_ids=["train-a", "train-b"],
        seeds=[7, 9],
        recipe=dict(student_steps=2, teacher_steps=2, teacher_batch_size=2),
    )
    arms = D.TEACHER_ARMS if kind == "teachers" else D.SHARED_ARMS
    recipe = (D.TeacherRecipe if kind == "teachers" else D.SharedRecipe)(**config["recipe"])
    work = (D.teacher_work if kind == "teachers" else D.shared_work)(2, 2, 16, 2, recipe)
    values, cases, teacher_sources = [], [], {}
    if kind == "teachers":
        for ep in cohort.episodes:
            generation = dict(
                prompt=ep.source.prompt,
                prompt_ids=[2, 3],
                response_ids=[10, 1],
                text="FINAL: [0]",
                cap_hit=False,
            )
            teacher_sources[ep.episode_id] = generation
            write_json(
                cohort.output / "sources" / f"{ep.episode_id}.json",
                dict(generation=generation, feedback="The submitted answer was incorrect."),
            )
    for seed in config["seeds"]:
        for ep in cohort.episodes:
            for arm in arms:
                roles = ("Q",) if kind == "teachers" else ("S", "Q")
                for role in roles:
                    for task in [ep.source] if role == "S" else ep.queries:
                        good = seed == 7 and arm != "baseline"
                        generation = dict(
                            prompt=task.prompt,
                            prompt_ids=[2, 3],
                            response_ids=[10, 1],
                            text="FINAL: [1]" if good else "FINAL: [0]",
                            cap_hit=False,
                        )
                        row = dict(
                            seed=seed,
                            episode_id=ep.episode_id,
                            task_id=task.task_id,
                            family=ep.family,
                            role=role,
                            generation=generation,
                        )
                        row.update(
                            dict(condition=arm, **grade(generation["text"], task))
                            if kind == "teachers"
                            else dict(arm=arm, grade=grade(generation["text"], task))
                        )
                        values.append(row)
                updated = arm != "baseline" if kind == "teachers" else arm in D.UPDATED_ARMS
                case = dict(seed=seed, episode_id=ep.episode_id)
                case.update(
                    buffers_before_sha256="buffers",
                    buffers_after_sha256="buffers",
                    actor_before_sha256="initial",
                    actor_after_sha256="updated",
                    actor_after_evaluation_sha256="updated" if updated else "initial",
                )
                case["condition" if kind == "teachers" else "arm"] = arm
                if kind == "shared":
                    source = next(
                        row
                        for row in values
                        if row["seed"] == seed
                        and row["episode_id"] == ep.episode_id
                        and row.get("arm") == arm
                        and row["role"] == "S"
                    )
                    case["source_digest"] = digest(source["generation"])
                else:
                    case["source_digest"] = digest(teacher_sources[ep.episode_id])
                case["metrics" if kind == "teachers" else "student_metrics"] = (
                    [dict(step=i + 1, loss=0.1) for i in range(2)] if updated else []
                )
                cases.append(case)
    write_json(
        cohort.output / "receipt.json",
        dict(
            status="completed",
            config=config,
            base_before_sha256="same",
            base_after_sha256="same",
            same_storage_audits_passed=True,
            frozen_files_unchanged=True,
            work=work,
            planned_work=work,
        ),
    )
    write_json(
        cohort.output / "identity.json",
        dict(
            config=config,
            training_examples=[dict(episode_id=key) for key in config["training_episode_ids"]],
        ),
    )
    write_json(
        cohort.output / "plan.json",
        dict(
            work=work,
            generations=[
                {
                    key: row[key]
                    for key in (
                        "seed",
                        "episode_id",
                        "task_id",
                        "role",
                        "condition" if kind == "teachers" else "arm",
                    )
                }
                for row in values
            ],
        ),
    )
    (
        cohort.output / ("observations.jsonl" if kind == "teachers" else "generations.jsonl")
    ).write_text("\n".join(json.dumps(value) for value in values) + "\n")
    (cohort.output / "cases.jsonl").write_text(
        "\n".join(json.dumps(value) for value in cases) + "\n"
    )
    teacher_steps = [
        dict(
            seed=seed,
            mode=mode,
            step=step,
            loss=0.1,
            grad_norm=0.2,
            source_ids=["train-a", "train-b"],
        )
        for seed in config["seeds"]
        for mode in (("bidirectional", "raw_causal") if kind == "teachers" else ("bidirectional",))
        for step in (1, 2)
    ]
    if kind == "teachers":
        (cohort.output / "teacher-steps.jsonl").write_text(
            "\n".join(json.dumps(value) for value in teacher_steps) + "\n"
        )
    else:
        events = [
            dict(
                event="shared_teacher_fit_completed",
                seed=seed,
                fit=dict(metrics=[row for row in teacher_steps if row["seed"] == seed]),
            )
            for seed in config["seeds"]
        ]
        (cohort.output / "events.jsonl").write_text(
            "\n".join(json.dumps(value) for value in events) + "\n"
        )
    return SimpleNamespace(config=config, values=values, cases=cases)


@pytest.mark.parametrize("kind", ["teachers", "shared"])
def test_teacher_and_shared_seeds_are_never_pooled_as_independent_sources(cohort, kind):
    synthetic_multiseed(cohort, kind)
    result = report(kind, cohort)
    assert result["seeds"] == [7, 9] and result["source_count"] == 2
    assert set(result["per_seed"]) == {"7", "9"}
    for seed in (7, 9):
        q = next(
            row for row in result["per_seed"][str(seed)]["paired_vs_baseline"] if row["role"] == "Q"
        )
        assert q["episodes"] == 2 and q["left"]["denominator"] == 16
        assert q["mean_difference"] == (1.0 if seed == 7 else 0.0)


def test_contrast_bootstraps_sources_not_individual_task_tokens():
    left, right = [], []
    for ep_id, good in (("e0", True), ("e1", False)):
        for task in range(8):
            value = dict(
                episode_id=ep_id,
                task_id=f"{ep_id}-{task}",
                family="f",
                format_ok=True,
                cap_hit=False,
                reused_baseline=False,
            )
            left.append(dict(value, success=good))
            right.append(dict(value, success=False))
    result = D.contrast(left, right, samples=100, seed=73)
    expected = paired_interval([1.0, 0.0], bootstrap_samples=100, seed=73)
    assert result["episodes"] == 2 and result["left"]["denominator"] == 16
    assert all(result[key] == value for key, value in expected.items())


@pytest.mark.parametrize("kind", ["teachers", "shared"])
@pytest.mark.parametrize(
    "defect",
    [
        "plan",
        "teacher_steps",
        "student_steps",
        "buffers",
        "actor",
        "training_membership",
        "source_digest",
    ],
)
def test_multiseed_plans_and_actual_update_evidence_are_audited(cohort, kind, defect):
    synthetic_multiseed(cohort, kind)
    if defect == "plan":
        path = cohort.output / "plan.json"
        value = read_json(path)
        value["generations"].pop()
        write_json(path, value)
    else:
        path = cohort.output / (
            "cases.jsonl"
            if defect in ("student_steps", "buffers", "actor", "source_digest")
            else "teacher-steps.jsonl"
            if kind == "teachers"
            else "events.jsonl"
        )
        values = rows(path)
        if defect == "student_steps":
            key = "metrics" if kind == "teachers" else "student_metrics"
            next(row for row in values if row[key])[key].pop()
        elif defect == "source_digest":
            values[0]["source_digest"] = "different-source"
        elif defect in ("buffers", "actor"):
            values[0][
                "buffers_after_sha256" if defect == "buffers" else "actor_after_evaluation_sha256"
            ] = "changed"
        elif defect == "training_membership":
            target = values[0] if kind == "teachers" else values[0]["fit"]["metrics"][0]
            target["source_ids"][0] = "not-a-training-source"
        else:
            values.pop()
        path.write_text("\n".join(json.dumps(value) for value in values) + "\n")
    with pytest.raises(ValueError):
        report(kind, cohort)
