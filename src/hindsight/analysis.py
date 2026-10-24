"""Paired episode-level reporting; query outcomes never select a candidate."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

FIELDS = ("episode_id", "partition", "family", "condition", "candidate", "task_id", "role")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(record)
    return records


def _index(rows: Iterable[Mapping[str, Any]], outcomes: bool) -> dict[tuple, dict]:
    indexed = {}
    for source in rows:
        row = dict(source)
        for field in FIELDS:
            if field not in row:
                raise ValueError(f"Missing field {field}")
            if field != "candidate" and (not isinstance(row[field], str) or not row[field]):
                raise ValueError(f"Invalid {field}: {row[field]!r}")
        candidate = row["candidate"]
        if candidate is not None and (type(candidate) is not int or candidate < 0):
            raise ValueError("candidate must be a nonnegative integer or null")
        if row["role"] not in {"V", "Q"}:
            raise ValueError("Only V/Q outcome records belong in this analysis")
        if outcomes:
            for field in ("success", "format_ok", "cap_hit"):
                if type(row.get(field)) is not bool:
                    raise ValueError(f"{field} must be an explicit boolean")
        key = tuple(row[field] for field in FIELDS)
        if key in indexed:
            raise ValueError(f"Duplicate observation: {key}")
        indexed[key] = row
    return indexed


def _percentile(values: list[float], fraction: float) -> float:
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_interval(
    episode_deltas: Iterable[float], *, bootstrap_samples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Resample source episodes, never individual query tasks or tokens."""
    values = list(episode_deltas)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite episode difference")
    interval = None
    if len(values) > 1:
        rng = random.Random(seed)
        replicates = sorted(
            mean(rng.choices(values, k=len(values))) for _ in range(bootstrap_samples)
        )
        interval = [_percentile(replicates, 0.025), _percentile(replicates, 0.975)]
    return {
        "episodes": len(values),
        "mean_difference": mean(values) if values else None,
        "ci95": interval,
        "improved_episodes": sum(value > 0 for value in values),
        "regressed_episodes": sum(value < 0 for value in values),
        "tied_episodes": sum(value == 0 for value in values),
        "uncertainty": "paired source-episode bootstrap; conditional on observed families/fit seed",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def summarize(
    records: Iterable[Mapping[str, Any]],
    *,
    expected: Iterable[Mapping[str, Any]] | None = None,
    selections: Iterable[Mapping[str, Any]] = (),
    protocol_frozen: bool = False,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize observed and planned rows without inventing missing outcomes.

    ``expected`` contains the seven FIELDS, with no outcome fields required.
    Selection events have episode_id, condition, selected_candidate, derived_from='V'.
    A null selected_candidate means the no-update baseline, not candidate zero.
    ``protocol_frozen`` attests caller-side freezing before any final Q evaluation.
    """
    observed = _index(records, outcomes=True)
    planned = _index(expected, outcomes=False) if expected is not None else observed
    unexpected = observed.keys() - planned.keys()
    if unexpected:
        raise ValueError(f"{len(unexpected)} observations outside the supplied plan")
    if not protocol_frozen and any(
        row["role"] == "Q" and row["partition"].lower().startswith("final")
        for row in observed.values()
    ):
        raise ValueError("Final Q reporting requires a frozen protocol")

    episode_meta = {}
    tasks = {}
    groups = defaultdict(dict)
    group_plan = defaultdict(set)
    for key, row in planned.items():
        episode = row["episode_id"]
        metadata = (row["partition"], row["family"])
        if episode in episode_meta and episode_meta[episode] != metadata:
            raise ValueError(f"Conflicting episode metadata: {episode}")
        episode_meta[episode] = metadata
        task_key = (episode, row["task_id"])
        if task_key in tasks and tasks[task_key] != row["role"]:
            raise ValueError("The same task cannot occupy both V and Q")
        tasks[task_key] = row["role"]
        group_key = (episode, row["role"], row["condition"], row["candidate"])
        group_plan[group_key].add(row["task_id"])
        if key in observed:
            groups[group_key][row["task_id"]] = observed[key]

    def complete(key: tuple) -> bool:
        return bool(group_plan.get(key)) and set(groups[key]) == group_plan[key]

    aggregates = defaultdict(list)
    contrasts = defaultdict(list)
    excluded_pairs = defaultdict(list)
    for key in sorted(group_plan, key=repr):
        episode, role, condition, candidate = key
        partition, family = episode_meta[episode]
        aggregate_key = (partition, role, condition, candidate)
        aggregates[aggregate_key].append((key, family))
        if condition == "baseline" and candidate is None:
            continue
        baseline = (episode, role, "baseline", None)
        if baseline in group_plan and group_plan[baseline] != group_plan[key]:
            raise ValueError(f"Unpaired task sets: {episode}/{role}/{condition}/{candidate}")
        if not complete(key) or not complete(baseline):
            excluded_pairs[aggregate_key].append(episode)
            continue
        delta = mean(row["success"] for row in groups[key].values()) - mean(
            row["success"] for row in groups[baseline].values()
        )
        contrasts[aggregate_key].append((episode, family, delta))

    def counts(items: list[tuple]) -> dict:
        rows = [row for key, _ in items for row in groups[key].values()]
        completed = [key for key, _ in items if complete(key)]
        return {
            "planned_trials": sum(len(group_plan[key]) for key, _ in items),
            "observed_trials": len(rows),
            "complete_episodes": len(completed),
            "incomplete_episodes": len(items) - len(completed),
            "source_mean_success": mean(
                mean(row["success"] for row in groups[key].values()) for key in completed
            ) if completed else None,
            "successful_observed_trials": sum(row["success"] for row in rows),
            "format_failures": sum(not row["format_ok"] for row in rows),
            "cap_hits": sum(row["cap_hit"] for row in rows),
        }

    conditions = []
    comparisons = []
    for key, items in sorted(aggregates.items(), key=lambda item: repr(item[0])):
        partition, role, condition, candidate = key
        identity = dict(partition=partition, role=role, condition=condition, candidate=candidate)
        conditions.append({
            **identity, **counts(items),
            "by_family": {
                family: counts([item for item in items if item[1] == family])
                for family in sorted({family for _, family in items})
            },
        })
        if condition == "baseline" and candidate is None:
            continue
        pairs = contrasts[key]
        comparisons.append({
            **identity,
            "status": "incomplete" if excluded_pairs[key] else "complete",
            **paired_interval((delta for _, _, delta in pairs),
                              bootstrap_samples=bootstrap_samples, seed=seed),
            "incomplete_episode_ids": excluded_pairs[key],
            "by_family": {
                family: {"episodes": len(values), "mean_difference": mean(values)}
                for family in sorted({family for _, family, _ in pairs})
                if (values := [delta for _, f, delta in pairs if f == family])
            },
        })

    ordering = []
    for episode in sorted(episode_meta):
        candidates = sorted({
            key[3] for key in group_plan
            if key[0] == episode and key[1] == "V"
            and key[2] in {"direct_pi", "adapted_pi"} and key[3] is not None
        })
        if not candidates:
            continue
        keys = [(episode, "V", condition, candidate)
                for condition in ("direct_pi", "adapted_pi") for candidate in candidates]
        item = dict(episode_id=episode, partition=episode_meta[episode][0],
                    family=episode_meta[episode][1], candidates=candidates)
        if not all(complete(key) for key in keys):
            ordering.append({**item, "status": "incomplete"})
            continue
        if len({frozenset(group_plan[key]) for key in keys}) != 1:
            raise ValueError(f"Unpaired candidate V tasks: {episode}")
        scores = {
            condition: [mean(row["success"] for row in groups[
                (episode, "V", condition, candidate)].values()) for candidate in candidates]
            for condition in ("direct_pi", "adapted_pi")
        }
        concordant = discordant = direct_ties = update_ties = both_ties = 0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                x = scores["direct_pi"][i] - scores["direct_pi"][j]
                y = scores["adapted_pi"][i] - scores["adapted_pi"][j]
                concordant += x * y > 0
                discordant += x * y < 0
                both_ties += x == 0 and y == 0
                direct_ties += x == 0 and y != 0
                update_ties += y == 0 and x != 0
        decisive = concordant + discordant
        denominator = math.sqrt((decisive + direct_ties) * (decisive + update_ties))
        ordering.append({
            **item, "status": "complete", "scores": scores,
            "concordant_pairs": concordant, "discordant_pairs": discordant,
            "direct_only_ties": direct_ties, "update_only_ties": update_ties,
            "both_ties": both_ties,
            "kendall_tau_b": (concordant - discordant) / denominator if denominator else None,
        })

    selected_pairs = defaultdict(list)
    selection_missing = defaultdict(list)
    selection_without_query_plan = []
    seen_selections = set()
    for event in selections:
        episode, condition = event["episode_id"], event["condition"]
        candidate = event["selected_candidate"]
        if condition not in {"direct_pi", "adapted_pi"}:
            raise ValueError("Unknown PI selection condition")
        if event.get("derived_from") != "V":
            raise ValueError("Selection must be declared from V, never Q")
        if candidate is not None and (type(candidate) is not int or candidate < 0):
            raise ValueError("Invalid selected_candidate")
        if (episode, condition) in seen_selections or episode not in episode_meta:
            raise ValueError("Duplicate or unknown selection episode")
        seen_selections.add((episode, condition))
        partition, _ = episode_meta[episode]
        label = (partition, condition)
        baseline = (episode, "Q", "baseline", None)
        if baseline not in group_plan:
            selection_without_query_plan.append(dict(episode_id=episode, condition=condition))
            continue
        chosen = baseline if candidate is None else (episode, "Q", condition, candidate)
        if chosen not in group_plan:
            raise ValueError("Selection references an unplanned Q condition")
        if not complete(baseline) or not complete(chosen):
            selection_missing[label].append(episode)
            continue
        if group_plan[baseline] != group_plan[chosen]:
            raise ValueError("Selected comparison has unpaired Q tasks")
        selected_pairs[label].append(mean(row["success"] for row in groups[chosen].values())
                                     - mean(row["success"] for row in groups[baseline].values()))

    missing = sorted(planned.keys() - observed.keys(), key=repr)
    return {
        "status": ("incomplete" if missing else "complete")
        if expected is not None else "plan_unverified",
        "observed_records": len(observed),
        "planned_records": len(planned) if expected is not None else None,
        "missing_records": [dict(zip(FIELDS, key)) for key in missing],
        "conditions": conditions,
        "paired_vs_baseline": comparisons,
        "within_source_V_ordering": ordering,
        "selections_without_query_plan": selection_without_query_plan,
        "frozen_V_selection_Q": [
            {"partition": partition, "condition": condition,
             **paired_interval(selected_pairs[label],
                               bootstrap_samples=bootstrap_samples, seed=seed),
             "status": "incomplete" if selection_missing[label] else "complete",
             "incomplete_episode_ids": selection_missing[label]}
            for label in sorted(selected_pairs.keys() | selection_missing.keys())
            for partition, condition in [label]
        ],
    }
