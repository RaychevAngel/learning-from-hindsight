"""Exploratory CPU-only V/Q split audit; no model loading or new evaluation."""

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = (
    ("021", "external-lessons-balanced-qwen35-021", "balanced-development-v1"),
    ("023", "hindsight-teacher-qwen35-023", "balanced-transfer-v1"),
)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(label, result_name, data_name):
    result, data = ROOT / "results" / result_name, ROOT / "data" / data_name
    observations = rows(result / "observations.jsonl")
    private = {row["task_id"]: row for row in rows(data / "tasks.private.jsonl")}
    assignments = rows(data / "assignment.jsonl")
    relevant = [row for row in observations if row["condition"] in ("baseline", "adapted_pi")]
    indexed = {(row["task_id"], row["condition"]): row for row in relevant}
    assert len(indexed) == len(relevant) == 256 and len(assignments) == 32
    histogram, statistic = Counter({0: 1}), 0
    serials, sources, family_counts = defaultdict(list), defaultdict(dict), defaultdict(Counter)
    for item in assignments:
        ids = item["shuffled_ids"]
        pool = sorted(ids, key=lambda task: private[task]["pool_serial"])
        Random(item["seed"]).shuffle(pool)
        assert pool == ids
        assert [private[task]["role"] for task in ids] == ["validation"] * 2 + ["queries"] * 2
        assert all(private[task]["stratum"] == item["stratum"] for task in ids)
        changes = [int(indexed[task, "adapted_pi"]["success"])
                   - int(indexed[task, "baseline"]["success"]) for task in ids]
        statistic += sum(changes[:2]) - sum(changes[2:])
        # Six equiprobable 2-of-4 allocations, conditional on each fixed pool.
        local = Counter(2 * sum(changes[i] for i in selection) - sum(changes)
                        for selection in itertools.combinations(range(4), 2))
        updated = Counter()
        for previous, weight in histogram.items():
            for increment, count in local.items():
                updated[previous + increment] += weight * count
        histogram = updated
        for task in ids:
            serials[private[task]["role"]].append(int(private[task]["pool_serial"].split("-")[-1]))
    for row in relevant:
        key = (row["episode_id"], row["family"], row["role"])
        sources[key].setdefault(row["condition"], []).append(row["success"])
        family_counts[row["family"], row["role"]][row["condition"]] += row["success"]
    denominator = sum(histogram.values())
    assert denominator == 6 ** 32
    paths = [result / "observations.jsonl", result / "identity.json", data / "manifest.json",
             data / "assignment.jsonl", data / "tasks.private.jsonl"]
    return {
        "comparison": label,
        "provenance_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths},
        "assignment_seed_reproduction": "32/32 pools match",
        "statistic_V_delta_count_minus_Q_delta_count": statistic,
        "V_minus_Q_effect_proportion": statistic / 64,
        "effect_denominator_per_split": 64,
        "null_allocations": denominator,
        "two_sided_tail": sum(n for value, n in histogram.items()
                              if abs(value) >= abs(statistic)) / denominator,
        "one_sided_V_larger_tail": sum(n for value, n in histogram.items()
                                       if value >= statistic) / denominator,
        "histogram_integer_weights": dict(sorted(histogram.items())),
        "pool_generation_serial_means": {role: mean(values) for role, values in serials.items()},
        "pool_generation_serial_counts": {role: dict(Counter(values))
                                           for role, values in serials.items()},
        "per_source": [dict(episode_id=ep, family=family, role=role,
                            baseline=sum(value["baseline"]), adapted_pi=sum(value["adapted_pi"]),
                            delta=sum(value["adapted_pi"]) - sum(value["baseline"]))
                       for (ep, family, role), value in sources.items()],
        "per_family": [dict(family=family, role=role, **counts)
                       for (family, role), counts in family_counts.items()],
        "format_or_cap_failures": [row for row in relevant
                                   if not row["format_ok"] or row["cap_hit"]],
    }


if __name__ == "__main__":
    print(json.dumps({
        "status": "exploratory_posthoc_conditional_reassignment",
        "assumptions": [
            "Within each frozen four-task stratum pool, V is two uniformly assigned tasks.",
            "Observed paired task outcomes are held fixed under reassignment.",
            "The fixed-outcomes assumption requires no V/Q execution-order effect; it is unproven.",
            "This is a posthoc diagnostic tail area, not a preregistered confirmatory p-value.",
            "No independent task/family sampling, causal diagnosis, or bug attribution is implied.",
        ],
        "comparisons": [audit(*comparison) for comparison in COMPARISONS],
    }, indent=2, sort_keys=True, allow_nan=False))
