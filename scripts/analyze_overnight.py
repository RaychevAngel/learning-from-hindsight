"""Offline complete-run regrading and source-paired overnight comparisons."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from hindsight.analysis import paired_interval
from hindsight.cohorts import load_cohort
from hindsight.data import grade
from hindsight.experiment import GENERIC, digest, proposal_prompt, read_json, rows, seed_for
from hindsight.model import Generation, with_pi
from hindsight.proposer import PROPOSER_COUNTERS, Proposal, usable_lesson
from hindsight.shared import ARMS as SHARED_ARMS
from hindsight.shared import UPDATED_ARMS, SharedRecipe
from hindsight.shared import planned_work as shared_work
from hindsight.teacher_controls import ARMS as TEACHER_ARMS
from hindsight.teacher_controls import TeacherRecipe
from hindsight.teacher_controls import planned_work as teacher_work

CORE = (
    "generation_sequences",
    "scoring_forwards",
    "student_optimizer_steps",
    "teacher_optimizer_steps",
    "teacher_example_exposures",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def identity(row):
    return tuple(
        row[key] for key in ("seed", "episode_id", "role", "task_id", "condition", "candidate")
    )


def native(value, cap):
    require(
        isinstance(value["text"], str) and type(value["cap_hit"]) is bool,
        "Invalid native generation text/cap metadata",
    )
    for field in ("prompt_ids", "response_ids"):
        require(
            isinstance(value[field], list)
            and value[field]
            and all(type(token) is int and token >= 0 for token in value[field]),
            "Missing or invalid native token IDs",
        )
    require(len(value["response_ids"]) <= cap, "Native response exceeds declared token budget")


def counts(values):
    by_source = defaultdict(list)
    for row in values:
        by_source[row["episode_id"]].append(row)
    return dict(
        sources=len(by_source),
        denominator=len(values),
        correct=sum(row["success"] for row in values),
        source_mean_success=mean(
            mean(row["success"] for row in group) for group in by_source.values()
        )
        if values
        else None,
        invalid=sum(not row["format_ok"] for row in values),
        caps=sum(row["cap_hit"] for row in values),
        reused_baseline=sum(row["reused_baseline"] for row in values),
    )


def contrast(left, right, *, samples, seed):
    def key(row):
        return row["episode_id"], row["task_id"]

    a, b = {key(row): row for row in left}, {key(row): row for row in right}
    require(
        len(a) == len(left) and len(b) == len(right) and a.keys() == b.keys(),
        "Unpaired or duplicate contrast task identities",
    )
    sources = []
    for ep_id in sorted({pair[0] for pair in a}):
        keys = [pair for pair in a if pair[0] == ep_id]
        gains = sum(a[pair]["success"] and not b[pair]["success"] for pair in keys)
        losses = sum(b[pair]["success"] and not a[pair]["success"] for pair in keys)
        sources.append(
            dict(
                episode_id=ep_id,
                family=a[keys[0]]["family"],
                denominator=len(keys),
                gains=gains,
                losses=losses,
                difference=(gains - losses) / len(keys),
            )
        )
    return dict(
        left=counts(left),
        right=counts(right),
        gains=sum(row["gains"] for row in sources),
        losses=sum(row["losses"] for row in sources),
        per_source=sources,
        by_family={
            family: dict(
                sources=len(group),
                mean_difference=mean(row["difference"] for row in group),
                gains=sum(row["gains"] for row in group),
                losses=sum(row["losses"] for row in group),
            )
            for family in sorted({row["family"] for row in sources})
            if (group := [row for row in sources if row["family"] == family])
        },
        **paired_interval(
            [row["difference"] for row in sources], bootstrap_samples=samples, seed=seed
        ),
    )


def describe(values, *, samples, seed):
    groups = defaultdict(list)
    for row in values:
        groups[(row["role"], row["condition"], row["candidate"])].append(row)
    conditions, comparisons = [], []
    for (role, condition, candidate), group in sorted(
        groups.items(), key=lambda pair: repr(pair[0])
    ):
        label = dict(role=role, condition=condition, candidate=candidate)
        conditions.append(
            dict(
                **label,
                **counts(group),
                by_family={
                    family: counts([row for row in group if row["family"] == family])
                    for family in sorted({row["family"] for row in group})
                },
                per_source={
                    ep_id: counts([row for row in group if row["episode_id"] == ep_id])
                    for ep_id in sorted({row["episode_id"] for row in group})
                },
            )
        )
        if role != "S" and condition != "baseline":
            comparisons.append(
                dict(
                    **label,
                    **contrast(group, groups[(role, "baseline", None)], samples=samples, seed=seed),
                )
            )
    return dict(conditions=conditions, paired_vs_baseline=comparisons)


def candidate_ranking(left, right):
    """Within-source K4 ranks; ties are retained, not broken by selection priority."""
    require(
        len(left) == len(right) == 4 and all(math.isfinite(value) for value in (*left, *right)),
        "Ranking requires four finite matched candidate scores",
    )

    def ranks(values):
        return [
            1
            + sum(other < value for other in values)
            + (sum(other == value for other in values) - 1) / 2
            for value in values
        ]

    a, b = ranks(left), ranks(right)
    centered_a, centered_b = [v - mean(a) for v in a], [v - mean(b) for v in b]
    variance_a = sum(v * v for v in centered_a)
    variance_b = sum(v * v for v in centered_b)
    rho = (
        sum(x * y for x, y in zip(centered_a, centered_b, strict=True))
        / math.sqrt(variance_a * variance_b)
        if variance_a and variance_b
        else None
    )
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            x = (left[i] > left[j]) - (left[i] < left[j])
            y = (right[i] > right[j]) - (right[i] < right[j])
            category = (
                "both_tied"
                if x == y == 0
                else "left_tied"
                if x == 0
                else "right_tied"
                if y == 0
                else "concordant"
                if x == y
                else "discordant"
            )
            pairs.append(dict(candidates=[i, j], left_order=x, right_order=y, category=category))
    totals = {
        key: sum(row["category"] == key for row in pairs)
        for key in ("concordant", "discordant", "both_tied", "left_tied", "right_tied")
    }
    comparable = totals["concordant"] + totals["discordant"]
    return dict(
        candidate_count=4,
        pairs=pairs,
        pair_counts=totals,
        left_average_ranks=a,
        right_average_ranks=b,
        spearman=rho,
        spearman_undefined_reason=(
            "both_constant"
            if not variance_a and not variance_b
            else "left_constant"
            if not variance_a
            else "right_constant"
            if not variance_b
            else None
        ),
        non_tied_pair_count=comparable,
        non_tied_order_agreement=totals["concordant"] / comparable if comparable else None,
        exact_order_agreement=(totals["concordant"] + totals["both_tied"]) / 6,
    )


def decision_analysis(values, selections, config, *, samples, seed):
    groups = defaultdict(list)
    for row in values:
        groups[(row["episode_id"], row["role"], row["condition"], row["candidate"])].append(row)
    require(
        len(selections) == len({row["episode_id"] for row in values})
        and len({row["episode_id"] for row in selections}) == len(selections),
        "Missing or duplicate V selections",
    )
    derived, ordering = [], []
    for selection in selections:
        ep_id = selection["episode_id"]
        require(selection["derived_from"] == "V", "Q-derived selection is prohibited")
        base = sum(row["success"] for row in groups[(ep_id, "V", "baseline", None)])
        scores = {
            condition: [
                sum(row["success"] for row in groups[(ep_id, "V", condition, candidate)])
                for candidate in range(4)
            ]
            for condition in ("direct_pi", "adapted_pi")
        }
        tie_seed = seed_for(config["seed"], ep_id, "pi_tie_order")
        order = sorted(
            range(4), key=lambda candidate: seed_for(tie_seed, "candidate_priority", candidate)
        )
        require(
            selection.get("candidate_tie_order", order) == order,
            "Recorded candidate tie order changed",
        )
        chosen = {}
        for condition, field in (
            ("direct_pi", "direct_selected_candidate"),
            ("adapted_pi", "selected_candidate"),
        ):
            score = scores[condition]
            winner = (
                None
                if max(score) <= base
                else next(candidate for candidate in order if score[candidate] == max(score))
            )
            require(
                selection[field] == winner,
                "Recorded choice differs from actual V-only neutral/tie rule",
            )
            require(
                selection["direct_successes" if condition == "direct_pi" else "adapted_successes"]
                == score
                and selection["baseline_successes"] == base,
                "Recorded selection scores differ from regraded V",
            )
            chosen[condition] = winner
        a, b = chosen["direct_pi"], chosen["adapted_pi"]
        query_scores = [
            sum(row["success"] for row in groups[(ep_id, "Q", "adapted_pi", candidate)])
            for candidate in range(4)
        ]
        query_baseline = sum(row["success"] for row in groups[(ep_id, "Q", "baseline", None)])
        validation_n = len(groups[(ep_id, "V", "baseline", None)])
        query_n = len(groups[(ep_id, "Q", "baseline", None)])
        actual_v_regret = max([base, *scores["adapted_pi"]]) - (
            base if a is None else scores["adapted_pi"][a]
        )
        direct_choice_q = query_baseline if a is None else query_scores[a]
        actual_choice_q = query_baseline if b is None else query_scores[b]

        def value(condition, candidate):
            return base if candidate is None else scores[condition][candidate]

        tie_only = a != b and all(
            value(condition, a) == value(condition, b) for condition in scores
        )
        ordering.append(
            dict(
                episode_id=ep_id,
                candidate_ids=list(range(4)),
                baseline_successes=base,
                direct_V_successes=scores["direct_pi"],
                adapted_V_successes=scores["adapted_pi"],
                direct_choice=a,
                actual_update_choice=b,
                selectors_disagree=a != b,
                tie_only_disagreement=tie_only,
                direct_choice_is_actual_V_optimal=value("adapted_pi", a)
                == max([base, *scores["adapted_pi"]]),
                candidate_tie_order=order,
                adapted_Q_successes=query_scores,
                baseline_Q_successes=query_baseline,
                within_source_rankings={
                    "direct_V_vs_student_V": candidate_ranking(
                        scores["direct_pi"], scores["adapted_pi"]
                    ),
                    "direct_V_vs_student_Q": candidate_ranking(scores["direct_pi"], query_scores),
                },
                direct_selected_actual_V_regret_successes=actual_v_regret,
                direct_selected_actual_V_regret_fraction=actual_v_regret / validation_n,
                validation_denominator=validation_n,
                query_denominator=query_n,
                direct_V_selected_student_Q_successes=direct_choice_q,
                actual_V_selected_student_Q_successes=actual_choice_q,
                actual_minus_direct_selected_student_Q_successes=actual_choice_q - direct_choice_q,
            )
        )
        for label, condition, candidate in (
            ("baseline", "baseline", None),
            ("fixed_direct", "direct_pi", 0),
            ("fixed_adapted", "adapted_pi", 0),
            ("V_selected_direct", "direct_pi", a),
            ("V_selected_adapted", "adapted_pi", b),
            ("direct_V_selected_adapted", "adapted_pi", a),
        ):
            source = (
                groups[(ep_id, "Q", "baseline", None)]
                if candidate is None
                else groups[(ep_id, "Q", condition, candidate)]
            )
            derived.extend({**row, "condition": label, "candidate": None} for row in source)
    result = describe(derived, samples=samples, seed=seed)
    for left, right in (
        ("V_selected_direct", "fixed_direct"),
        ("V_selected_adapted", "fixed_adapted"),
        ("V_selected_adapted", "direct_V_selected_adapted"),
    ):
        result.setdefault("paired_decision_contrasts", []).append(
            dict(
                left_policy=left,
                right_policy=right,
                **contrast(
                    [row for row in derived if row["condition"] == left],
                    [row for row in derived if row["condition"] == right],
                    samples=samples,
                    seed=seed,
                ),
            )
        )
    rankings = {}
    for name in ("direct_V_vs_student_V", "direct_V_vs_student_Q"):
        defined = [
            row["within_source_rankings"][name]["spearman"]
            for row in ordering
            if row["within_source_rankings"][name]["spearman"] is not None
        ]
        rankings[name] = dict(
            sources=len(ordering),
            defined_sources=len(defined),
            undefined_sources=len(ordering) - len(defined),
            mean_within_source_spearman=mean(defined) if defined else None,
        )
    regrets = [row["direct_selected_actual_V_regret_fraction"] for row in ordering]
    regret_interval = paired_interval(regrets, bootstrap_samples=samples, seed=seed)
    result.update(
        candidate_pool=ordering,
        selector_disagreements=sum(row["selectors_disagree"] for row in ordering),
        tie_only_disagreements=sum(row["tie_only_disagreement"] for row in ordering),
        ranking_summary=rankings,
        direct_selected_actual_V_regret=dict(
            sources=len(regrets),
            mean_fraction=regret_interval["mean_difference"],
            ci95=regret_interval["ci95"],
            positive_regret_sources=sum(value > 0 for value in regrets),
            zero_regret_sources=sum(value == 0 for value in regrets),
            bootstrap_samples=samples,
            seed=seed,
            uncertainty="Source-episode bootstrap of V selection regret; positive is worse. "
            "Conditional on the observed candidate pool, families and fitted seed.",
        ),
        ranking_scope="K4 candidates only; no-update remains in selection/regret, not ranks. "
        "Average ranks retain ties; constant vectors have undefined correlation, not zero. "
        "Six within-source pairs are descriptive and are not independent observations. "
        "Q reports fixed V choices and never selects a candidate.",
    )
    return result


def analyze(kind, run, data, *, loader=load_cohort, bootstrap_samples=10000, bootstrap_seed=7301):
    files, checksums = set(), {}

    def read(name, many=False):
        path = run / name
        if not path.exists() and many:
            return []
        files.add(path)
        checksum = sha(path)
        require(path not in checksums or checksums[path] == checksum, "Input changed during audit")
        checksums[path] = checksum
        return rows(path) if many else read_json(path)

    receipt = read("receipt.json")
    require(receipt["status"] == "completed", "Only completed runs admit outcome analysis")
    identity_file = read("identity.json") if kind in ("teachers", "shared") else {}
    config = identity_file["config"] if kind == "shared" else receipt["config"]
    require(
        sha(data / "manifest.json") == config["dataset_manifest_sha256"],
        "Cohort manifest differs from run",
    )
    episodes, _ = loader(data)
    all_episodes = {ep.episode_id: ep for ep in episodes}
    plan = read("plan.json") if kind != "comparison" else None
    if kind == "comparison":
        ids = [ep.episode_id for ep in episodes if ep.partition == config["partition"]]
        require(
            len(ids) == config["expected_sources"] and receipt["completed_episode_ids"] == ids,
            "Comparison partition is incomplete",
        )
        conditions = [(name, None) for name in ("baseline", "generic", "mismatched_pi")] + [
            (name, candidate) for name in ("direct_pi", "adapted_pi") for candidate in range(4)
        ]
        roles, seeds = ("V", "Q"), [config["seed"]]
    elif kind == "proposer":
        require(
            receipt["phase"] == "evaluate" and receipt["frozen_inputs_unchanged"],
            "Only frozen proposer evaluation is admitted",
        )
        ids = plan["evaluation_episode_ids"]
        require(
            ids == receipt["completed_episode_ids"]
            and not set(ids) & set(plan["train_episode_ids"]),
            "Proposer evaluation source/partition overlap",
        )
        conditions = [("baseline", None), ("initial_proposer", 0), ("trained_proposer", 0)]
        roles, seeds = config.get("evaluation_roles", ["V", "Q"]), [config["seed"]]
    elif kind in ("teachers", "shared"):
        ids, seeds = config["evaluation_episode_ids"], config["seeds"]
        conditions = [(arm, None) for arm in (TEACHER_ARMS if kind == "teachers" else SHARED_ARMS)]
        roles = ("Q",) if kind == "teachers" else ("S", "Q")
        require(
            receipt["base_before_sha256"] == receipt["base_after_sha256"]
            and receipt["same_storage_audits_passed"],
            "Recorded base/storage invariants failed",
        )
        if kind == "teachers":
            require(receipt["frozen_files_unchanged"], "Teacher run code/qualification changed")
    else:
        raise ValueError("Unknown overnight experiment kind")
    require(
        ids
        and len(set(ids)) == len(ids)
        and set(ids) <= all_episodes.keys()
        and seeds
        and all(type(seed) is int for seed in seeds)
        and len(set(seeds)) == len(seeds)
        and roles
        and len(set(roles)) == len(roles)
        and set(roles) <= ({"S", "Q"} if kind == "shared" else {"V", "Q"}),
        "Invalid source/seed identities",
    )
    selected = [all_episodes[key] for key in ids]
    require(
        all(len(ep.validation) == len(ep.queries) == 8 for ep in selected),
        "Require complete V8/Q8 source cohorts",
    )
    expected_partition = (
        config["partition"]
        if kind == "comparison"
        else config.get("evaluation_partition", "confirmation")
        if kind == "proposer"
        else config.get("evaluation_partition", "development")
    )
    require(
        expected_partition in ("development", "confirmation")
        and all(ep.partition == expected_partition for ep in selected),
        "Evaluation source partition differs from the frozen protocol",
    )
    expected = {
        (seed, ep.episode_id, role, task.task_id, condition, candidate)
        for seed in seeds
        for ep in selected
        for role in roles
        for task in ([ep.source] if role == "S" else ep.validation if role == "V" else ep.queries)
        for condition, candidate in conditions
    }
    original = read("generations.jsonl" if kind == "shared" else "observations.jsonl", True)
    normalized, indexed = [], {}
    proposals = {}
    if kind in ("comparison", "proposer"):
        for ep in selected:
            source = read(f"sources/{ep.episode_id}.json")
            native(source["generation"], config["source_tokens"])
            require(
                source["task_id"] == ep.source.task_id
                and source["generation"]["prompt"] == ep.source.prompt
                and source["grade"] == grade(source["generation"]["text"], ep.source),
                "Saved source identity/regrade mismatch",
            )
            feedback = "The submitted answer was " + (
                "correct." if source["grade"]["success"] else "incorrect."
            )
            require(source["feedback"] == feedback, "Source feedback differs from actual grade")
            labels = range(4) if kind == "comparison" else ("initial_proposer", "trained_proposer")
            for label in labels:
                proposal = read(f"proposals/{ep.episode_id}-{label}.json")
                value = proposal["proposal"]["generation"]
                native(value, config["pi_tokens"])
                policy = dict(proposal["proposal"])
                policy["generation"] = Generation(**value)
                require(
                    type(proposal["usable"]) is bool
                    and proposal["usable"] == usable_lesson(Proposal(**policy)),
                    "Saved proposal usability differs from its native text/cap rule",
                )
                require(
                    policy["temperature"] == config.get("proposer_temperature", 0.7),
                    "Proposal sampling temperature changed",
                )
                if kind == "proposer":
                    require(
                        policy["initial"] == (label == "initial_proposer")
                        and policy["seed"]
                        == seed_for(config["seed"], ep.episode_id, "proposer_evaluation", 0),
                        "Frozen and trained proposer budgets/seeds are unmatched",
                    )
                require(
                    value["prompt"]
                    == proposal_prompt(ep.source.prompt, source["generation"]["text"], feedback),
                    "Proposal inspected non-source context",
                )
                proposals[ep.episode_id, label] = proposal
        require(
            len(list((run / "proposals").glob("*.json"))) == len(proposals),
            "Unexpected proposal pool members",
        )
    selections = read("selections.jsonl", True) if kind == "comparison" else []
    selection_by = {row["episode_id"]: row for row in selections}
    for row in original:
        ep = all_episodes.get(row["episode_id"])
        require(ep is not None, "Outcome source absent from cohort")
        role, condition = row["role"], row.get("condition", row.get("arm"))
        members = [ep.source] if role == "S" else ep.validation if role == "V" else ep.queries
        tasks = {task.task_id: task for task in members}
        require(
            row["task_id"] in tasks
            and role in roles
            and row.get("family", ep.family) == ep.family
            and row.get("partition", ep.partition) == ep.partition,
            "Outcome task/role/family/partition mismatch",
        )
        task, generation = tasks[row["task_id"]], row["generation"]
        native(generation, config["source_tokens"] if role == "S" else config["evaluation_tokens"])
        actual = grade(generation["text"], task)
        recorded = row["grade"] if kind == "shared" else row
        require(
            all(recorded[key] == value for key, value in actual.items())
            and type(recorded["success"]) is bool
            and type(recorded["format_ok"]) is bool,
            "Independent native-text regrade mismatch",
        )
        expected_prompt = task.prompt
        reused = row.get("reused_baseline", False)
        require(type(reused) is bool, "Invalid baseline-reuse flag")
        eligible_reuse = False
        if kind == "comparison" and condition in ("direct_pi", "adapted_pi"):
            eligible_reuse = not proposals[ep.episode_id, row["candidate"]]["usable"]
        elif kind == "comparison" and condition == "mismatched_pi":
            donor = selection_by[ep.episode_id]["mismatch_episode_id"]
            expected_donor = next(
                other.episode_id for other in selected if other.family != ep.family
            )
            require(
                donor == expected_donor, "Mismatched donor differs from fixed source-only mapping"
            )
            eligible_reuse = not proposals[donor, 0]["usable"]
        elif kind == "proposer" and condition in ("initial_proposer", "trained_proposer"):
            eligible_reuse = not proposals[ep.episode_id, condition]["usable"]
        require(reused == eligible_reuse, "Baseline reuse is not eligible under the proposal rule")
        if kind == "comparison" and not reused:
            if condition == "generic":
                expected_prompt = with_pi(task.prompt, GENERIC)
            elif condition == "direct_pi":
                expected_prompt = with_pi(
                    task.prompt,
                    proposals[ep.episode_id, row["candidate"]]["proposal"]["generation"][
                        "text"
                    ].strip(),
                )
            elif condition == "mismatched_pi":
                donor = selection_by[ep.episode_id]["mismatch_episode_id"]
                require(
                    donor in ids and all_episodes[donor].family != ep.family,
                    "Mismatch control source is not a fixed other-family member",
                )
                expected_prompt = with_pi(
                    task.prompt, proposals[donor, 0]["proposal"]["generation"]["text"].strip()
                )
        require(
            generation["prompt"] == expected_prompt, "Evaluation PI/causal prompt boundary mismatch"
        )
        value = dict(
            seed=row.get("seed", config.get("seed")),
            episode_id=ep.episode_id,
            partition=ep.partition,
            family=ep.family,
            role=role,
            task_id=task.task_id,
            condition=condition,
            candidate=row.get("candidate"),
            generation=generation,
            cap_hit=generation["cap_hit"],
            reused_baseline=reused,
            **actual,
        )
        key = identity(value)
        require(
            key in expected and key not in indexed, "Missing/duplicate/unplanned outcome identity"
        )
        normalized.append(value)
        indexed[key] = value
    require(indexed.keys() == expected, "Incomplete expected outcome matrix")
    if kind in ("teachers", "shared"):
        planned = [
            dict(row, condition=row.get("condition", row.get("arm")), candidate=None)
            for row in plan["generations"]
        ]
        require(
            len(planned) == len(expected) and {identity(row) for row in planned} == expected,
            "Saved generation plan differs from the corpus-derived exact matrix",
        )
    for row in normalized:
        if row["reused_baseline"]:
            key = (row["seed"], row["episode_id"], row["role"], row["task_id"], "baseline", None)
            require(
                row["generation"] == indexed[key]["generation"],
                "Reused outcome is not the literal measured baseline",
            )
    if kind in ("comparison", "proposer"):
        adaptations = read("adaptations.jsonl", True)
        require(
            all(
                len(row["metrics"]) == config["student_steps"]
                and all(
                    math.isfinite(value)
                    for step in row["metrics"]
                    for value in step.values()
                    if isinstance(value, (int, float))
                )
                for row in adaptations
            ),
            "Invalid student step count or numerical metrics",
        )
        adapted_keys = {
            (row["episode_id"], row["condition"], row["candidate"]) for row in adaptations
        }
        needed = {
            (
                ep_id,
                "adapted_pi" if kind == "comparison" else label,
                label if kind == "comparison" else 0,
            )
            for (ep_id, label), proposal in proposals.items()
            if proposal["usable"]
        }
        require(
            len(adapted_keys) == len(adaptations) and adapted_keys == needed,
            "Adaptation identities differ from the usable candidate pool",
        )
        for adaptation in adaptations:
            source = read(f"sources/{adaptation['episode_id']}.json")
            require(
                adaptation["source_response_ids"] == source["generation"]["response_ids"],
                "Student adaptation lost original source token identity",
            )
        expected_work = dict.fromkeys((*CORE, *PROPOSER_COUNTERS), 0)
        source_count = sum(
            read(f"sources/{ep.episode_id}.json")["source_origin"] == "generated" for ep in selected
        )
        expected_work.update(
            generation_sequences=sum(not row["reused_baseline"] for row in normalized)
            + len(proposals)
            + source_count,
            proposer_generation_sequences=len(proposals),
            scoring_forwards=len(adaptations)
            + (
                sum(row["selected_candidate"] is None for row in selections)
                if kind == "comparison"
                else 0
            ),
            student_optimizer_steps=len(adaptations) * config["student_steps"],
        )
    else:
        recipe = (TeacherRecipe if kind == "teachers" else SharedRecipe)(**config["recipe"])
        n_train = (
            len(config["training_episode_ids"])
            if kind == "teachers"
            else len(identity_file["training_examples"])
        )
        expected_work = (teacher_work if kind == "teachers" else shared_work)(
            n_train, len(selected), sum(len(ep.queries) for ep in selected), len(seeds), recipe
        )
        require(
            plan["work"] == expected_work and receipt["planned_work"] == expected_work,
            "Work plan differs from explicit source/seed/recipe counts",
        )
        training_ids = set(
            config["training_episode_ids"]
            if kind == "teachers"
            else [item["episode_id"] for item in identity_file["training_examples"]]
        )
        if kind == "teachers":
            teacher_steps = read("teacher-steps.jsonl", True)
            teacher_keys = [(row["seed"], row["mode"], row["step"]) for row in teacher_steps]
            expected_teacher = {
                (seed, mode, step)
                for seed in seeds
                for mode in ("bidirectional", "raw_causal")
                for step in range(1, recipe.teacher_steps + 1)
            }
        else:
            fits = [
                row
                for row in read("events.jsonl", True)
                if row["event"] == "shared_teacher_fit_completed"
            ]
            require(
                len(fits) == len(seeds) and {row["seed"] for row in fits} == set(seeds),
                "Missing/duplicate shared teacher-fit receipts",
            )
            teacher_steps = [
                dict(step, seed=fit["seed"]) for fit in fits for step in fit["fit"]["metrics"]
            ]
            teacher_keys = [(row["seed"], row["step"]) for row in teacher_steps]
            expected_teacher = {
                (seed, step) for seed in seeds for step in range(1, recipe.teacher_steps + 1)
            }
        require(
            len(teacher_keys) == len(set(teacher_keys))
            and set(teacher_keys) == expected_teacher
            and len(teacher_steps) == expected_work["teacher_optimizer_steps"]
            and sum(len(row["source_ids"]) for row in teacher_steps)
            == expected_work["teacher_example_exposures"]
            and all(
                len(row["source_ids"]) == recipe.teacher_batch_size
                and set(row["source_ids"]) <= training_ids
                and math.isfinite(row["loss"])
                and math.isfinite(row["grad_norm"])
                for row in teacher_steps
            ),
            "Teacher optimizer/exposure evidence differs from exact planned work",
        )
        cases = read("cases.jsonl", True)
        case_keys = [
            (row["seed"], row["episode_id"], row.get("condition", row.get("arm"))) for row in cases
        ]
        require(
            len(case_keys) == len(set(case_keys))
            and set(case_keys)
            == {
                (seed, ep_id, condition)
                for seed in seeds
                for ep_id in ids
                for condition, _ in conditions
            },
            "Incomplete per-source/seed update cases",
        )
        for case in cases:
            name = case.get("condition", case.get("arm"))
            metrics = case.get("metrics", case.get("student_metrics", []))
            updated = name != "baseline" if kind == "teachers" else name in UPDATED_ARMS
            require(
                len(metrics) == (recipe.student_steps if updated else 0),
                "Case optimizer-step evidence differs from plan",
            )
            require(
                case["buffers_before_sha256"] == case["buffers_after_sha256"]
                and case["actor_after_evaluation_sha256"]
                == case["actor_after_sha256" if updated else "actor_before_sha256"],
                "Recorded per-case actor/buffer state invariant failed",
            )
            require(
                all(
                    math.isfinite(value)
                    for step in metrics
                    for value in step.values()
                    if isinstance(value, (int, float))
                ),
                "Nonfinite student metrics",
            )
            if kind == "shared":
                source = indexed[
                    (
                        case["seed"],
                        case["episode_id"],
                        "S",
                        all_episodes[case["episode_id"]].source.task_id,
                        name,
                        None,
                    )
                ]
                require(
                    case["source_digest"] == digest(source["generation"]),
                    "Shared update source differs from its arm's causal generation",
                )
            else:
                source = read(f"sources/{case['episode_id']}.json")
                source_generation = source["generation"]
                task = all_episodes[case["episode_id"]].source
                native(source_generation, config["source_tokens"])
                feedback = "The submitted answer was " + (
                    "correct."
                    if grade(source_generation["text"], task)["success"]
                    else "incorrect."
                )
                require(
                    source_generation["prompt"] == task.prompt
                    and source["feedback"] == feedback
                    and case["source_digest"] == digest(source_generation),
                    "Teacher-control case differs from its saved source/feedback",
                )
    require(receipt["work"] == expected_work, "Actual model work differs from audited evidence")
    reports = {}
    for current_seed in seeds:
        values = [row for row in normalized if row["seed"] == current_seed]
        report = describe(values, samples=bootstrap_samples, seed=bootstrap_seed)
        if kind == "comparison":
            report["decisions"] = decision_analysis(
                values, selections, config, samples=bootstrap_samples, seed=bootstrap_seed
            )
        if kind == "proposer":
            report["trained_minus_initial"] = {
                role: contrast(
                    [
                        row
                        for row in values
                        if row["role"] == role and row["condition"] == "trained_proposer"
                    ],
                    [
                        row
                        for row in values
                        if row["role"] == role and row["condition"] == "initial_proposer"
                    ],
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                for role in roles
            }
        reports[str(current_seed)] = report
    files.update(data.rglob("*.json"))
    files.update(data.rglob("*.jsonl"))
    files.update(
        Path(__file__).resolve().parents[1] / "src/hindsight" / name
        for name in ("analysis.py", "data.py", "cohorts.py")
    )
    require(
        all(sha(path) == checksum for path, checksum in checksums.items()),
        "Evidence changed during analysis",
    )
    return dict(
        status="passed_offline_terminal_audit",
        kind=kind,
        source_count=len(selected),
        source_ids=ids,
        partitions=sorted({ep.partition for ep in selected}),
        config=config,
        families=sorted({ep.family for ep in selected}),
        seeds=seeds,
        denominator_records=len(normalized),
        work=expected_work,
        per_seed=reports,
        hashes={str(path.resolve()): sha(path) for path in sorted(files)},
        analysis_code_sha256=sha(Path(__file__)),
        limitations=[
            "Source-episode bootstrap conditional on represented families and each fitted seed; "
            "no pooled-seed or token/task-independent uncertainty.",
            "Q never selects candidates or settings; development/confirmation labels are retained, "
            "not upgraded to a sealed test claim.",
            "Native IDs are structurally checked and saved text independently graded; "
            "no tokenizer/model is loaded to regenerate text or targets.",
            "Recorded parameter/hash invariants are checked where supplied; "
            "model tensors are not independently re-forwarded.",
            "Cap hits are separate; exact correctness uses the existing grader "
            "without silently changing its policy.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("comparison", "proposer", "teachers", "shared"), required=True
    )
    for name in ("run", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite analysis")
    result = analyze(args.kind, args.run, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(status=result["status"], output=str(args.output), sources=result["source_count"])
        )
    )


if __name__ == "__main__":
    main()
