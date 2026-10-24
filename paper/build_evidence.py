"""Rebuild the completed 023/024 manuscript tables, optionally plot a figure.

No model calls. Inputs are immutable saved observations and fidelity records.
This recounts saved grades; it is not a substitute for the independent regrader.
Run from any directory: python paper/build_evidence.py [--plot]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "observations023": (
        "results/hindsight-teacher-qwen35-023/observations.jsonl",
        "bcf190ecf0663408a5a1564f0412fb4ba31ee1b3d7ef7d7367597d5a09f947d1",
    ),
    "generations024": (
        "results/order-and-pi-qwen35-024/generations.jsonl",
        "a7a611b7c0751cf43e7081e12faae437de275866ef86ab33bcf8d7f024e27996",
    ),
}
CONDITIONS = [
    ("baseline", "No update"),
    ("adapted_pi", "Explicit PI update"),
    ("adapted_bidirectional", "Fitted hindsight update"),
    ("adapted_untrained_bidirectional", "Unfitted hindsight update"),
    ("adapted_raw_causal", "Raw causal update"),
    ("adapted_replay", "Observed-token replay"),
]


def read_verified(name: str) -> list[dict]:
    relative, expected = INPUTS[name]
    payload = (ROOT / relative).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"Immutable input hash mismatch: {relative}")
    return [json.loads(line) for line in payload.splitlines() if line]


def keyed(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    result = {}
    for row in rows:
        key = (row["episode_id"], row["task_id"], row["role"], row["condition"])
        if key in result:
            raise ValueError(f"Duplicate observation: {key}")
        result[key] = row
    return result


def build() -> dict:
    historical = read_verified("observations023")
    current = read_verified("generations024")
    if len(historical) != 512 or len(current) != 224:
        raise ValueError("Incomplete saved invocation")
    rows = historical + [
        {
            **{k: row[k] for k in ("episode_id", "task_id", "role", "condition", "family")},
            **row["grade"],
            "cap_hit": row["generation"]["cap_hit"],
        }
        for row in current
        if row["condition"] == "direct_pi"
    ]
    index = keyed(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["role"])].append(row)
    counts = {}
    for (condition, role), group in sorted(groups.items()):
        per_source = defaultdict(list)
        for row in group:
            per_source[row["episode_id"]].append(row)
        if len(group) != 64 or len(per_source) != 8:
            raise ValueError(f"Wrong task/source denominator: {condition}/{role}")
        if any(len(source) != 8 for source in per_source.values()):
            raise ValueError("Unequal source task coverage")
        counts[f"{condition}/{role}"] = {
            "correct": sum(row["success"] for row in group),
            "denominator": len(group),
            "sources": len(per_source),
            "malformed": sum(not row["format_ok"] for row in group),
            "caps": sum(row["cap_hit"] for row in group),
            "per_source": {
                source: sum(row["success"] for row in source_rows)
                for source, source_rows in sorted(per_source.items())
            },
        }
    gains, losses = [], []
    for row in groups[("direct_pi", "Q")]:
        partner = index[(row["episode_id"], row["task_id"], "Q", "adapted_pi")]
        delta = int(row["success"]) - int(partner["success"])
        if delta:
            (gains if delta > 0 else losses).append(row["task_id"])
    if len(gains) != 12 or losses:
        raise ValueError("Current direct-versus-adapted paired contrast changed")

    fidelity = {}
    fidelity_hashes = {}
    for condition in ("adapted_untrained_bidirectional", "adapted_bidirectional"):
        folder = ROOT / "results/hindsight-teacher-qwen35-023/evaluation-fidelity" / condition
        records = {}
        for path in sorted(folder.glob("*.json")):
            payload = path.read_bytes()
            record = json.loads(payload)
            records[path.stem] = record["selected_reference"][
                "forward_kl_reference_to_bidirectional"
            ]
            fidelity_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        if len(records) != 8:
            raise ValueError("Incomplete fresh-source fidelity records")
        fidelity[condition] = {"per_source": records, "source_equal_mean": mean(records.values())}
    before = fidelity["adapted_untrained_bidirectional"]["per_source"]
    after = fidelity["adapted_bidirectional"]["per_source"]
    if before.keys() != after.keys() or not all(after[k] < before[k] for k in before):
        raise ValueError("Fresh-source fidelity comparison does not improve on every source")
    return {
        "evidence_cutoff": "Figure1 contains completed023/024 only; later diagnostics excluded",
        "inputs": {name: {"path": path, "sha256": sha} for name, (path, sha) in INPUTS.items()},
        "fidelity_file_sha256": fidelity_hashes,
        "counts": counts,
        "direct_pi_minus_adapted_pi_Q": {
            "gain_task_ids": sorted(gains),
            "loss_task_ids": sorted(losses),
        },
        "fresh_selected_fidelity": fidelity,
        "limit": "Recounts saved grades and logged fidelity; does not regrade or execute models.",
    }


def render_tables(data: dict) -> str:
    counts = data["counts"]
    lines = [
        "# Rebuilt development tables",
        "",
        "Generated by `python paper/build_evidence.py`. Counts include every saved failure.",
        "",
        "| Condition | V correct /64 | Q correct /64 |",
        "|---|---:|---:|",
    ]
    for condition, label in [
        ("baseline", "No update"),
        ("direct_pi", "Lesson in context"),
        ("adapted_pi", "Four PI-derived updates"),
    ]:
        lines.append(
            f"| {label} | {counts[condition + '/V']['correct']} "
            f"| {counts[condition + '/Q']['correct']} |"
        )
    lines += ["", "| Student condition | Q correct /64 |", "|---|---:|"]
    for condition, label in CONDITIONS:
        lines.append(f"| {label} | {counts[condition + '/Q']['correct']} |")
    lines += ["", "| Fresh selected-reference fidelity | Mean forward KL |", "|---|---:|"]
    for condition, label in [
        ("adapted_untrained_bidirectional", "Unfitted"),
        ("adapted_bidirectional", "Fitted"),
    ]:
        value = data["fresh_selected_fidelity"][condition]["source_equal_mean"]
        lines.append(f"| {label} | {value:.6f} |")
    lines += [
        "",
        "Eight source clusters, four familiar families. Descriptive development evidence.",
        "Fidelity values are logged metrics, not newly computed model predictions.",
        "",
    ]
    return "\n".join(lines)


def plot(data: dict, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 4.7), gridspec_kw={"width_ratios": [1.05, 0.8, 1.3]}
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.27, top=0.8, wspace=0.48)
    counts = data["counts"]
    colors = ["#777777", "#17668B", "#C26A34"]
    for offset, (condition, label) in enumerate(
        [("baseline", "No update"), ("direct_pi", "PI in context"), ("adapted_pi", "PI update")]
    ):
        heights = [counts[f"{condition}/{role}"]["correct"] for role in ("V", "Q")]
        xs = [i + (offset - 1) * 0.24 for i in range(2)]
        axes[0].bar(xs, heights, width=0.21, label=label, color=colors[offset])
        for x, value in zip(xs, heights, strict=True):
            axes[0].text(x, value + 1, str(value), ha="center", fontsize=9)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Validation", "Queries"],
        ylim=(0, 64),
        yticks=[0, 16, 32, 48, 64],
        ylabel="Correct tasks /64",
        title="A  Context vs. four updates",
    )
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncols=3, fontsize=8.5
    )

    fidelity = data["fresh_selected_fidelity"]
    for source, initial in fidelity["adapted_untrained_bidirectional"]["per_source"].items():
        final = fidelity["adapted_bidirectional"]["per_source"][source]
        axes[1].plot(
            [0, 1], [initial, final], color="#B8B8B8", linewidth=1.0, marker="o", markersize=3
        )
    means = [
        fidelity[c]["source_equal_mean"]
        for c in ("adapted_untrained_bidirectional", "adapted_bidirectional")
    ]
    axes[1].plot([0, 1], means, color="#17668B", linewidth=2.5, marker="o", markersize=6)
    for x, value in enumerate(means):
        axes[1].annotate(
            f"{value:.3f}",
            (x, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#17668B",
            weight="bold",
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Unfitted", "Fitted"],
        xlim=(-0.3, 1.3),
        ylim=(0, None),
        ylabel="Forward KL to selected reference",
        title="B  Fresh-source fidelity",
    )

    labels = [
        label if condition == "baseline" else label.replace(" update", "")
        for condition, label in CONDITIONS
    ]
    values = [counts[c + "/Q"]["correct"] for c, _ in CONDITIONS]
    axes[2].barh(
        list(range(6)),
        values,
        color=["#777777", "#C26A34", "#17668B", "#99BAC9", "#A8A8A8", "#C5C5C5"],
    )
    axes[2].invert_yaxis()
    axes[2].set(
        yticks=list(range(6)),
        yticklabels=labels,
        xlim=(0, 64),
        xticks=[0, 16, 32, 48, 64],
        xlabel="Correct query tasks /64",
        title="C  Causal student results",
    )
    for y, value in enumerate(values):
        axes[2].text(value + 1.0, y, str(value), va="center", fontsize=9)
    fig.suptitle(
        "Useful context and teacher fidelity do not establish reliable student gains",
        fontsize=14,
        y=0.96,
    )
    fig.text(
        0.065,
        0.02,
        "Completed development runs 023/024. Eight source clusters, four familiar families.\n"
        "Observed counts and logged KL; source-cluster uncertainty for fitted-teacher "
        "query contrasts spans zero.",
        fontsize=8.5,
        color="#444444",
    )
    fig.savefig(folder / "context-fidelity-transfer.png", dpi=190)
    svg = folder / "context-fidelity-transfer.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true", help="Require matplotlib and produce PNG/SVG figures"
    )
    args = parser.parse_args()
    data = build()
    output = ROOT / "paper/derived"
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence023024.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (output / "tables023024.md").write_text(render_tables(data))
    if args.plot:
        plot(data, output)
    print(
        json.dumps(
            {
                "status": "verified_saved_counts",
                "source_clusters": 8,
                "new_model_calls": 0,
                "figure_rendered": args.plot,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
