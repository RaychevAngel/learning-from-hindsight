"""Verify and reanalyse compact historical evidence with only the standard library.

No model/tokenizer imports, inference, external service access or training occur.
The default is read-only. An optional output path must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hindsight.analysis import FIELDS, paired_interval, summarize  # noqa: E402
from hindsight.data import grade, load_corpus  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root=ROOT):
    manifest = read(root / "provenance/evidence-manifest.json")
    require(manifest["schema"] == "hindsight-public-evidence-v1", "Wrong evidence manifest")
    for item in manifest["files"]:
        path = root / item["path"]
        require(
            path.is_relative_to(root) and ".." not in Path(item["path"]).parts,
            "Unsafe manifest path",
        )
        require(path.is_file() and not path.is_symlink(), f"Missing input: {item['path']}")
        require(
            path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
            f"Changed evidence input: {item['path']}",
        )
    return len(manifest["files"])


def task_index(corpus):
    episodes, calibration = load_corpus(corpus)
    index = {}
    for episode in episodes:
        for role, tasks in (
            ("S", [episode.source]),
            ("V", episode.validation),
            ("Q", episode.queries),
        ):
            for task in tasks:
                index[task.task_id] = (episode, role, task)
    require(not calibration or len(calibration) == 16, "Unexpected calibration count")
    return episodes, index


def verify_synthetic_oracles(corpus):
    """Check stored answers via rule implementations independent of the generator.

    Oracle access is deliberately confined to this offline evidence audit; this
    function is not a proposer input or a model-training call.
    """
    specification = importlib.util.spec_from_file_location(
        "public_oracle_checks", ROOT / "scripts/build_corpus.py"
    )
    builder = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(builder)
    private = {row["task_id"]: row for row in rows(corpus / "tasks.private.jsonl")}
    checked = 0
    for public in rows(corpus / "tasks.public.jsonl"):
        oracle = private[public["task_id"]]
        pairs = public["demonstrations"] + [
            {"input": public["query_input"], "output": oracle["answer"]}
        ]
        for pair in pairs:
            require(
                builder.independent_answer(oracle["generator_alias"], pair["input"])
                == pair["output"],
                "Synthetic oracle disagrees with independent rule implementation",
            )
            checked += 1
    return checked


def regrade(observations, index, expected=None):
    seen = set()
    for row in observations:
        key = tuple(row[field] for field in FIELDS)
        require(key not in seen, f"Duplicate observation: {key}")
        seen.add(key)
        require(row["task_id"] in index, "Unknown observed task")
        episode, role, task = index[row["task_id"]]
        require(
            row["episode_id"] == episode.episode_id
            and row["role"] == role
            and row["family"] == task.family
            and row["partition"] == episode.partition,
            "Observation source/task/role/family mismatch",
        )
        actual = grade(row["text"], task)
        require(
            all(row.get(key) == value for key, value in actual.items()),
            f"Stored grade differs from independent regrade: {row['task_id']}",
        )
        require(type(row["cap_hit"]) is bool, "Missing cap accounting")
    if expected is not None:
        planned = [tuple(row[field] for field in FIELDS) for row in expected]
        require(
            len(planned) == len(set(planned)) and seen == set(planned),
            "Incomplete or unplanned historical observation matrix",
        )
    return len(seen)


def group_counts(observations):
    groups = defaultdict(list)
    for row in observations:
        groups[(row["role"], row["condition"])].append(row)
    result = {}
    for (role, condition), records in sorted(groups.items()):
        by_source = defaultdict(list)
        for row in records:
            by_source[row["episode_id"]].append(row)
        result[f"{condition}/{role}"] = {
            "correct": sum(row["success"] for row in records),
            "denominator": len(records),
            "malformed": sum(not row["format_ok"] for row in records),
            "caps": sum(row["cap_hit"] for row in records),
            "per_source": {
                source: sum(row["success"] for row in values)
                for source, values in sorted(by_source.items())
            },
        }
    return result


def compare_main(observations, direct, *, bootstrap_samples=10000):
    main = {
        (r["episode_id"], r["task_id"], r["condition"]): r for r in observations if r["role"] == "Q"
    }
    # Keep the recorded episode order used by the original seeded audit. A
    # permutation leaves the exact bootstrap distribution invariant, but changes
    # which finite Monte Carlo samples a fixed seed draws.
    clusters = list(dict.fromkeys(row["episode_id"] for row in observations))
    contrasts = {}
    for control in (
        "baseline",
        "adapted_untrained_bidirectional",
        "adapted_pi",
        "adapted_raw_causal",
        "adapted_replay",
    ):
        sources = []
        for source in clusters:
            treatment = [
                r
                for r in observations
                if r["episode_id"] == source
                and r["role"] == "Q"
                and r["condition"] == "adapted_bidirectional"
            ]
            require(len(treatment) == 8, "Wrong treatment denominator")
            pairs = [
                (r["success"], main[(source, r["task_id"], control)]["success"]) for r in treatment
            ]
            gains = sum(a and not b for a, b in pairs)
            losses = sum(b and not a for a, b in pairs)
            sources.append(
                {
                    "episode_id": source,
                    "gains": gains,
                    "losses": losses,
                    "difference": (gains - losses) / 8,
                }
            )
        contrasts[control] = {
            "gains": sum(s["gains"] for s in sources),
            "losses": sum(s["losses"] for s in sources),
            "per_source": sources,
            **paired_interval(
                [s["difference"] for s in sources], bootstrap_samples=bootstrap_samples, seed=7301
            ),
        }
    contextual_pairs = [
        (r["success"], main[(r["episode_id"], r["task_id"], "adapted_pi")]["success"])
        for r in direct
        if r["role"] == "Q"
    ]
    require(len(contextual_pairs) == 64, "Wrong direct-context query denominator")
    overlap = {
        "both_correct": sum(a and b for a, b in contextual_pairs),
        "direct_only": sum(a and not b for a, b in contextual_pairs),
        "adapted_only": sum(b and not a for a, b in contextual_pairs),
        "both_wrong": sum(not a and not b for a, b in contextual_pairs),
    }
    require(
        overlap["direct_only"] == 12 and overlap["adapted_only"] == 0,
        "Historical direct-context/update contrast changed",
    )
    return contrasts, overlap


def fidelity_from_positions(data):
    require(data["status"] == "independent_saved_tensor_audit", "Wrong fidelity evidence type")
    positions = data["positions"]
    require(len(positions) == 576 and len(data["sources"]) == 8, "Incomplete native metrics")
    by_source, by_region = defaultdict(list), defaultdict(list)
    identities = set()
    for row in positions:
        identity = row["episode_id"], row["position"]
        require(identity not in identities, "Duplicate native position")
        identities.add(identity)
        by_source[row["episode_id"]].append(row)
        by_region[row["region"]].append(row)
        # Stored FP32 log-probability normalization can yield tiny negative KL
        # after the original float64 audit (minimum about -5.3e-8). Preserve it;
        # neither clamp the data nor mistake that rounding for a missing value.
        require(
            all(
                math.isfinite(row[key]) and row[key] >= -1e-6
                for key in ("initial_kl", "post_kl", "initial_tv", "post_tv")
            ),
            "Nonfinite or negative native metric",
        )
    pooled = data["pooled_positions_descriptive"]
    for native, aggregate in (
        ("initial_kl", "initial_kl_mean"),
        ("post_kl", "post_kl_mean"),
        ("initial_tv", "initial_total_variation_mean"),
        ("post_tv", "post_total_variation_mean"),
    ):
        require(
            math.isclose(
                mean(row[native] for row in positions),
                pooled[aggregate],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            "Pooled metric mismatch",
        )
        source_equal = mean(mean(row[native] for row in source) for source in by_source.values())
        require(
            math.isclose(
                source_equal, data["source_equal_means"][aggregate], rel_tol=1e-12, abs_tol=1e-12
            ),
            "Source-equal metric mismatch",
        )
    require(
        Counter(row["region"] for row in positions)
        == {"rule_text": 102, "array_value": 150, "array_format": 316, "stop_token": 8},
        "Native region coverage changed",
    )
    rule_fraction = sum(r["initial_kl"] for r in by_region["rule_text"]) / sum(
        r["initial_kl"] for r in positions
    )
    require(
        all(row["initial_argmax"] == row["target_argmax"] for row in by_region["array_value"]),
        "Historical array top choices differ",
    )
    for region, values in by_region.items():
        require(
            math.isclose(
                sum(row["initial_kl"] for row in values),
                data["regions_pooled_descriptive"][region]["initial_kl_sum"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            "Region KL mismatch",
        )
    return {
        "sources": len(by_source),
        "native_positions": len(positions),
        "array_values": len(by_region["array_value"]),
        "initial_KL_fraction_in_rule_text": rule_fraction,
        "source_equal_means": data["source_equal_means"],
        "pooled_means": {
            key: pooled[key]
            for key in (
                "initial_kl_mean",
                "post_kl_mean",
                "initial_total_variation_mean",
                "post_total_variation_mean",
            )
        },
        "scope": "Recomputed from compact native-position metrics, not full tensors",
    }


def analyze(root=ROOT, *, bootstrap_samples=10000):
    verified = verify_manifest(root)
    experiments = {}
    oracle_pairs = 0
    for label, corpus, run, expected in (
        ("initial", "corpus-v2", "initial-v3", 1728),
        (
            "earlier_transfer",
            "balanced-development-v1",
            "external-lessons-balanced-qwen35-021",
            384,
        ),
        ("main_teacher", "balanced-transfer-v1", "hindsight-teacher-qwen35-023", 512),
    ):
        _, index = task_index(root / "data" / corpus)
        oracle_pairs += verify_synthetic_oracles(root / "data" / corpus)
        folder = root / "results" / run
        observed = rows(folder / "observations.jsonl")
        require(len(observed) == expected, f"Incomplete {label} invocation")
        regrade(observed, index, rows(folder / "expected.jsonl"))
        experiments[label] = {"regraded_records": len(observed), "counts": group_counts(observed)}
        if label == "initial":
            selected = rows(folder / "selections.jsonl")
            report = summarize(
                observed,
                expected=rows(folder / "expected.jsonl"),
                selections=selected,
                bootstrap_samples=bootstrap_samples,
                seed=7301,
            )
            experiments[label]["validation_selected_query_contrasts"] = report[
                "frozen_V_selection_Q"
            ]
            # A three-candidate pool is not three independent copies of the
            # selected method. Keep candidate-level outcomes separate.
            experiments[label].pop("counts")
            experiments[label]["candidate_level_counts"] = report["conditions"]
            selected_counts = {}
            for condition in ("direct_pi", "adapted_pi"):
                selected_rows = []
                by_task = {
                    (r["episode_id"], r["task_id"], r["condition"], r["candidate"]): r
                    for r in observed
                    if r["role"] == "Q"
                }
                for decision in selected:
                    if decision["condition"] != condition:
                        continue
                    episode = decision["episode_id"]
                    candidate = decision["selected_candidate"]
                    chosen = "baseline" if candidate is None else condition
                    task_ids = [
                        r["task_id"]
                        for r in observed
                        if r["episode_id"] == episode
                        and r["role"] == "Q"
                        and r["condition"] == "baseline"
                    ]
                    selected_rows.extend(
                        by_task[(episode, task, chosen, candidate)] for task in task_ids
                    )
                selected_counts[condition] = {
                    "correct": sum(r["success"] for r in selected_rows),
                    "denominator": len(selected_rows),
                }
            require(
                selected_counts
                == {
                    "direct_pi": {"correct": 48, "denominator": 128},
                    "adapted_pi": {"correct": 47, "denominator": 128},
                },
                "Initial validation-selected method counts changed",
            )
            experiments[label]["validation_selected_query_counts"] = selected_counts
        if label == "main_teacher":
            main, main_index = observed, index
    native = rows(root / "results/order-and-pi-qwen35-024/generations.jsonl")
    require(len(native) == 224, "Incomplete follow-up invocation")
    direct, layouts = [], defaultdict(dict)
    for row in native:
        generation = row["generation"]
        episode, role, task = main_index[row["task_id"]]
        require(
            row["episode_id"] == episode.episode_id and row["role"] == role,
            "Native task identity mismatch",
        )
        require(grade(generation["text"], task) == row["grade"], "Native grade mismatch")
        require(
            len(generation["response_ids"]) == row["generated_tokens"] <= 160,
            "Native generation length mismatch",
        )
        require(
            all(
                type(x) is int and x >= 0
                for field in ("prompt_ids", "response_ids")
                for x in generation[field]
            ),
            "Invalid native token IDs",
        )
        key = row["episode_id"], row["task_id"], row["role"]
        require(key not in layouts[row["layout"]], "Duplicate native layout entry")
        layouts[row["layout"]][key] = generation
        if row["condition"] == "direct_pi":
            direct.append(
                {
                    **{key: row[key] for key in FIELDS},
                    **row["grade"],
                    "cap_hit": generation["cap_hit"],
                    "text": generation["text"],
                }
            )
    require(len(direct) == 128, "Incomplete direct-context comparison")
    regrade(direct, main_index)
    for left in ("original", "reverse", "interleaved"):
        require(
            len(layouts[left]) == 32 and layouts[left] == layouts["original"],
            "Native order replay mismatch",
        )
    contrasts, overlap = compare_main(main, direct, bootstrap_samples=bootstrap_samples)
    historical_audit = read(root / "results/analysis-hindsight-teacher-023.json")
    for control, contrast in contrasts.items():
        original = historical_audit["fitted_q_contrasts"][control]
        for key in ("gains", "losses", "episodes", "mean_difference"):
            require(contrast[key] == original[key], "Historical paired contrast changed")
        if bootstrap_samples == original["bootstrap_samples"]:
            require(contrast["ci95"] == original["ci95"], "Seeded audit interval changed")
    experiments["followup"] = {
        "regraded_records": len(native),
        "direct_context_counts": group_counts(direct),
        "native_replay_sources": 2,
        "native_replay_orderings": 3,
        "direct_context_vs_update_query_overlap": overlap,
    }
    experiments["main_teacher"]["fitted_query_contrasts"] = contrasts
    evidence_path = root / "paper/derived/evidence023024.json"
    expected = read(evidence_path)
    current_counts = group_counts(main + direct)
    for condition, value in current_counts.items():
        for key in ("correct", "denominator", "malformed", "caps", "per_source"):
            require(value[key] == expected["counts"][condition][key], "Published table mismatch")
    training = read(root / "results/proposer-028-independent-audit.json")
    framing = read(root / "results/prefix-framing-029-independent-audit.json")
    require(
        training["status"] == "passed_independent_completed_training_audit"
        and training["groups"] == training["optimizer_updates"] + training["tied_no_update_groups"]
        and training["proposal_records"]
        == training["usable_proposals"] + training["unusable_proposals"]
        and training["Q_outcomes"] == 0,
        "Training audit arithmetic mismatch",
    )
    require(
        framing["status"] == "passed_independent_completed_scoring_audit"
        and framing["source_count"] == len(framing["per_source"]) == 8,
        "Framing audit incomplete",
    )
    return {
        "status": "verified_public_evidence",
        "new_model_calls": 0,
        "verified_manifest_inputs": verified,
        "independently_checked_synthetic_pairs": oracle_pairs,
        "regraded_records": 2848,
        "experiments": experiments,
        "source_fidelity": fidelity_from_positions(
            read(root / "paper/derived/source-fidelity025-independent.json")
        ),
        "proposer_training": {
            key: training[key]
            for key in (
                "groups",
                "optimizer_updates",
                "tied_no_update_groups",
                "proposal_records",
                "usable_proposals",
                "unusable_proposals",
                "Q_outcomes",
            )
        },
        "limits": [
            "Completed development evidence, not confirmation",
            "Intervals resample source clusters conditional on observed families/fit seed",
            "Logged fitted-teacher fidelity is not a fresh model forward",
            "Compact position metrics do not replace a full-tensor audit",
            "Completed proposer updates are not held-out proposer benefit",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Save report to a new file; default prints only"
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing report")
    report = analyze()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as destination:
            destination.write(encoded)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "regraded_records": report["regraded_records"],
                    "new_model_calls": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
