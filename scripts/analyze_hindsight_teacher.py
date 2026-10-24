"""Read-only independent grading and paired-source reporting for completed 023."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from hindsight.analysis import paired_interval, summarize
from hindsight.data import grade, load_corpus
from hindsight.experiment import EXTRAS, digest, observation_key, read_json, rows

WORK = dict(generation_sequences=512, scoring_forwards=61, student_optimizer_steps=160,
            teacher_optimizer_steps=66, teacher_example_exposures=258)
QUALIFICATION = dict(generation_sequences=0, scoring_forwards=5, student_optimizer_steps=0,
                     teacher_optimizer_steps=2, teacher_example_exposures=2)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric_means(values):
    return {key: mean(numbers) for key in sorted({key for row in values for key in row})
            if (numbers := [row[key] for row in values
                            if isinstance(row.get(key), (int, float))])}


def analyze(result, data):
    receipt, identity = read_json(result / "receipt.json"), read_json(result / "identity.json")
    observed, expected = rows(result / "observations.jsonl"), rows(result / "expected.jsonl")
    episodes, calibration = load_corpus(data)
    require(receipt["status"] == "completed" and receipt["work"] == WORK
            and receipt["planned_work"] == WORK and receipt["actual_observations"] == 512
            and receipt["missing_observations"] == 0, "Incomplete or mismatched work receipt")
    require(identity["dataset_manifest"] == read_json(data / "manifest.json")
            and len(episodes) == 8 and not calibration
            and set(receipt["completed_episode_ids"]) == {ep.episode_id for ep in episodes},
            "Dataset or completed source identity differs")
    plan = set()
    tasks = {}
    for ep in episodes:
        require(ep.partition == "development" and len(ep.validation) == len(ep.queries) == 8,
                "Only balanced development V8/Q8 is admitted")
        for role, members in (("V", ep.validation), ("Q", ep.queries)):
            for task in members:
                tasks[task.task_id] = task
                for condition, candidate in [("baseline", None), ("adapted_pi", 0)] + (
                    [(condition, None) for condition in EXTRAS] if role == "Q" else []
                ):
                    plan.add((ep.episode_id, ep.partition, condition, candidate,
                              task.task_id, role))
    require(len(observed) == len(expected) == len(plan) == 512
            and {observation_key(row) for row in observed} == plan
            and {observation_key(row) for row in expected} == plan,
            "Missing, duplicate, unplanned, or misjoined observation")
    for row in observed:
        require(row["family"] == tasks[row["task_id"]].family, "Observation family mismatch")
        actual = grade(row["text"], tasks[row["task_id"]])
        require(all(row[key] == value for key, value in actual.items()),
                "Independent regrade mismatch")
    fit = read_json(result / "teacher/FIT-COMPLETE.json")
    checkpoint = sha(result / "teacher/checkpoint.pt")
    require(fit["identity_digest"] == digest(identity)
            and fit["checkpoint_files"] == {"checkpoint.pt": checkpoint}
            and receipt["teacher_checkpoint_sha256"] == checkpoint, "Teacher checkpoint mismatch")
    require(all(row.get("teacher_checkpoint_sha256") == checkpoint for row in observed
                if row["condition"] == "adapted_bidirectional"), "Outcome checkpoint mismatch")
    qualification = read_json(result / "qualification/qualification.json")
    require(qualification["status"] == "passed_teacher_qualification"
            and qualification["actual_work"] == QUALIFICATION, "Qualification not proven")
    metrics = fit["metrics"]
    require(metrics["optimizer_steps"] == 64 and metrics["example_exposures"] == 256
            and len(metrics["losses"]) == len(metrics["grad_norms"]) == 64
            and all(math.isfinite(value) for key in ("losses", "grad_norms")
                    for value in metrics[key]), "Incomplete or nonfinite fitting metrics")
    adaptations = [row for row in rows(result / "events.jsonl")
                   if row["event"] == "student_adapted"]
    require(len(adaptations) == 40 and all(len(row["metrics"]) == 4 for row in adaptations)
            and len({(row["episode_id"], row["condition"]) for row in adaptations}) == 40,
            "Student update boundaries differ from forty matched four-step adaptations")
    for row in adaptations:
        require(all(math.isfinite(value) for step in row["metrics"]
                    for value in step.values() if isinstance(value, (int, float))),
                "Nonfinite student update metric")
    selections = rows(result / "selections.jsonl")
    summary = summarize(observed, expected=expected, selections=selections,
                        bootstrap_samples=10000, seed=7301)
    groups = defaultdict(dict)
    for row in observed:
        groups[(row["episode_id"], row["role"], row["condition"])][row["task_id"]] = row
    source_counts, paired = [], {}
    for ep in episodes:
        source_counts.append(dict(episode_id=ep.episode_id, family=ep.family, conditions={
            f"{role}/{condition}": dict(correct=sum(row["success"] for row in values.values()),
                                        total=len(values))
            for (episode, role, condition), values in groups.items() if episode == ep.episode_id}))
    for control in ("baseline", "adapted_untrained_bidirectional", "adapted_pi",
                    "adapted_raw_causal", "adapted_replay"):
        per_source = []
        for ep in episodes:
            treatment = groups[(ep.episode_id, "Q", "adapted_bidirectional")]
            baseline = groups[(ep.episode_id, "Q", control)]
            gains = sum(row["success"] and not baseline[key]["success"]
                        for key, row in treatment.items())
            losses = sum(not row["success"] and baseline[key]["success"]
                         for key, row in treatment.items())
            per_source.append(dict(episode_id=ep.episode_id, family=ep.family, gains=gains,
                                   losses=losses, difference=(gains - losses) / 8))
        paired[control] = dict(per_source=per_source, gains=sum(x["gains"] for x in per_source),
                               losses=sum(x["losses"] for x in per_source),
                               **paired_interval((x["difference"] for x in per_source),
                                                 bootstrap_samples=10000, seed=7301))
    fidelity = {}
    for name in ("fit-fidelity-before", "fit-fidelity-after"):
        records = {path.stem: read_json(path) for path in sorted((result / name).glob("*.json"))}
        require(set(records) == set(receipt["fit_episode_ids"]), "Incomplete fit fidelity")
        fidelity[name] = dict(per_source=records,
                             source_equal_means=metric_means([row["metrics"]
                                                             for row in records.values()]))
    for condition in ("adapted_bidirectional", "adapted_untrained_bidirectional"):
        records = {path.stem: read_json(path) for path in sorted(
            (result / "evaluation-fidelity" / condition).glob("*.json"))}
        require(set(records) == {ep.episode_id for ep in episodes}, "Incomplete held-out fidelity")
        expected_checkpoint = checkpoint if condition == "adapted_bidirectional" else None
        require(all(row["teacher_checkpoint_sha256"] == expected_checkpoint
                    for row in records.values()), "Held-out fidelity checkpoint mismatch")
        fidelity[condition] = dict(per_source=records, source_equal_means={
            reference: metric_means([row[reference] for row in records.values()])
            for reference in ("selected_reference", "pi_reference")})
    return dict(status="verified_completed_development",
                receipt_sha256=sha(result / "receipt.json"),
                observations_sha256=sha(result / "observations.jsonl"),
                checkpoint_sha256=checkpoint, regraded_observations=512, work=receipt["work"],
                summary=summary, per_source=source_counts, fitted_q_contrasts=paired,
                fidelity=fidelity, fit_target_selection=receipt["fit_selections"],
                fit_target_kinds=dict(Counter("PI" if row["selected_candidate"] == 0 else "neutral"
                                             for row in receipt["fit_selections"].values())),
                teacher_fit_metrics=metrics, student_adaptations=adaptations,
                failures=[row for row in observed if row["cap_hit"] or not row["format_ok"]],
                interpretation="Eight fresh source clusters, four familiar families, one fit seed; "
                "development only. Lower fitting KL alone does not establish transfer. "
                "No matched fitted-causal comparator isolates bidirectionality from fitting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite analysis")
    report = analyze(args.results, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
