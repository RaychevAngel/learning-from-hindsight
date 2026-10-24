"""Render the independently audited025 distributions, without model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "paper/derived"


def main() -> None:
    path = FOLDER / "source-fidelity025-independent.json"
    data = json.loads(path.read_text())
    if data["status"] != "independent_saved_tensor_audit" or len(data["sources"]) != 8:
        raise ValueError("Require complete independently audited025 data")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.79, wspace=0.32)
    regions = data["regions_pooled_descriptive"]
    totals = data["pooled_positions_descriptive"]
    measures = ["positions", "initial_kl_sum", "post_kl_sum"]
    parts = [
        ("Rule text", ["rule_text"], "#17668B"),
        ("Array values", ["array_value"], "#C26A34"),
        ("Format / stop", ["array_format", "stop_token"], "#B5B5B5"),
    ]
    bottom = [0.0] * 3
    for label, names, color in parts:
        values = [100 * sum(regions[name][key] for name in names) / totals[key] for key in measures]
        axes[0].bar(range(3), values, bottom=bottom, width=0.6, label=label, color=color)
        if label == "Rule text":
            for i, value in enumerate(values):
                axes[0].text(i, value / 2, f"{value:.1f}%", ha="center", va="center", color="white")
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    axes[0].set(
        xticks=range(3),
        xticklabels=["Native positions", "Initial KL", "Post-update KL"],
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
        ylabel="Share of total (%)",
        title="A  KL is concentrated in rule text",
    )
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncols=3, fontsize=8
    )

    errors = [
        source["first_incorrect_array_position"]
        for source in data["sources"]
        if source["first_incorrect_array_position"] is not None
    ]
    expected_ids = [
        "b5146ba4da8c49d8068f68b6",
        "b0701b945b9e0648f9007b52",
        "b3a7a8f892234e68d05c5e72",
        "eb2e08ce9a30ca1da12fc3f8",
    ]
    if [row["episode_id"] for row in errors] != expected_ids:
        raise ValueError("Incorrect-source examples changed")
    for offset, (label, key, color) in enumerate(
        [
            ("Initial student", "initial_expected_value_probability", "#777777"),
            ("PI teacher", "teacher_expected_value_probability", "#17668B"),
            ("After four updates", "post_expected_value_probability", "#C26A34"),
        ]
    ):
        xs = [i + (offset - 1) * 0.25 for i in range(4)]
        values = [100 * row[key] for row in errors]
        axes[1].bar(xs, values, width=0.22, label=label, color=color)
        for x, value in zip(xs, values, strict=True):
            display = "<0.1" if 0 < value < 0.1 else f"{value:.1f}"
            axes[1].text(x, value + 0.6, display, ha="center", fontsize=8)
    axes[1].set(
        xticks=range(4),
        xticklabels=["Translation 1", "Translation 2", "Recoloring 1", "Recoloring 2"],
        ylim=(0, 32),
        yticks=[0, 10, 20, 30],
        ylabel="Probability of correct value (%)",
        title="B  First wrong source-array value",
    )
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncols=3, fontsize=8
    )
    fig.suptitle(
        "Source imitation improves, but its task-relevant signal is uneven", fontsize=14, y=0.96
    )
    fig.text(
        0.07,
        0.045,
        "025: eight original source trajectories; explicit-PI targets, "
        "not the fitted hindsight teacher.\n"
        "Source gold is used offline only. All four displayed PI targets "
        "still prefer an incorrect next value; "
        "these are not free-generation accuracy scores.",
        fontsize=8,
        color="#444444",
    )
    fig.savefig(FOLDER / "source-fidelity-localization025.png", dpi=190)
    svg = FOLDER / "source-fidelity-localization025.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance = {
        "input": str(path.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "source_order_panel_b": expected_ids,
        "new_model_calls": 0,
    }
    (FOLDER / "source-fidelity-localization025.provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
