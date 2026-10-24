"""Extract compact writing evidence from completed CPU audits; no model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> tuple[dict, str]:
    data = (ROOT / path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def main() -> None:
    training_path = "results/proposer-028-independent-audit.json"
    framing_path = "results/prefix-framing-029-independent-audit.json"
    training, training_hash = read(training_path)
    framing, framing_hash = read(framing_path)
    assert training["status"] == "passed_independent_completed_training_audit"
    assert framing["status"] == "passed_independent_completed_scoring_audit"
    assert training["groups"] == training["optimizer_updates"] + training["tied_no_update_groups"]
    assert (
        training["proposal_records"]
        == training["usable_proposals"] + training["unusable_proposals"]
    )
    assert training["Q_outcomes"] == 0
    assert framing["source_count"] == len(framing["per_source"]) == 8
    assert len(framing["first_wrong_values"]) == 4
    assert sum(row["array_values"] for row in framing["per_source"]) == 150
    for arm in ["old_PI", "B", "C"]:
        first_mean = (
            sum(
                row["arms"][arm]["correct_scalar_probability"]
                for row in framing["first_wrong_values"]
            )
            / 4
        )
        all_mean = sum(row["correct_probability_means"][arm] for row in framing["per_source"]) / 8
        assert abs(first_mean - framing["first_wrong_source_probability_means"][arm]) < 1e-12
        assert abs(all_mean - framing["source_equal_array_probability_means"][arm]) < 1e-12
        assert all(row["arms"][arm]["argmax_is_observed"] for row in framing["first_wrong_values"])
    c_improved = [
        row["episode_id"]
        for row in framing["first_wrong_values"]
        if row["arms"]["C"]["correct_scalar_probability"]
        > row["arms"]["old_PI"]["correct_scalar_probability"]
    ]
    assert c_improved == ["b0701b945b9e0648f9007b52"]
    result = {
        "scope": "Completed audited training028 and scoring029 only; no031/035 partial outcomes.",
        "source_hashes": {training_path: training_hash, framing_path: framing_hash},
        "training": {
            key: training[key]
            for key in [
                "source_episodes",
                "groups",
                "optimizer_updates",
                "tied_no_update_groups",
                "proposal_records",
                "usable_proposals",
                "unusable_proposals",
                "proposal_cap_hits",
                "likelihood_discrepancy_nats",
                "work",
                "Q_outcomes",
                "runtime_seconds",
                "limitations",
            ]
        },
        "framing": {
            key: framing[key]
            for key in [
                "first_wrong_values",
                "first_wrong_source_probability_means",
                "source_equal_array_probability_means",
                "source_count",
                "work",
                "runtime_seconds",
                "independently_mapped_native_scalar_values",
                "interpretation",
            ]
        },
        "C_first_error_improved_sources": c_improved,
        "limits": (
            "This extractor checks arithmetic in independent audit receipts, "
            "not model outputs anew."
        ),
    }
    output = ROOT / "paper/derived/closeout028029.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "source_hashes": result["source_hashes"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
