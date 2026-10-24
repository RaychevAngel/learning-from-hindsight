"""Print-sized layouts of the same independently checked figure data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build_evidence import CONDITIONS, build

FOLDER = Path(__file__).resolve().parent / "derived"


def save(fig, name: str) -> None:
    fig.savefig(FOLDER / f"{name}-print.png", dpi=240)
    svg = FOLDER / f"{name}-print.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def context_figure() -> None:
    data = build()
    fig = plt.figure(figsize=(7.2, 5.8))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.97,
        bottom=0.09,
        top=0.95,
        wspace=0.5,
        hspace=0.72,
        height_ratios=[1.15, 1],
    )
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    counts = data["counts"]
    colors = ["#777777", "#17668B", "#C26A34"]
    for offset, (condition, label) in enumerate(
        [("baseline", "No update"), ("direct_pi", "PI context"), ("adapted_pi", "PI update")]
    ):
        xs = [i + (offset - 1) * 0.24 for i in range(2)]
        values = [counts[f"{condition}/{role}"]["correct"] for role in ("V", "Q")]
        a.bar(xs, values, width=0.21, color=colors[offset], label=label)
        for x, value in zip(xs, values, strict=True):
            a.text(x, value + 1.2, str(value), ha="center", fontsize=8)
    a.set(
        xticks=[0, 1],
        xticklabels=["Validation", "Queries"],
        ylim=(0, 64),
        yticks=[0, 16, 32, 48, 64],
        ylabel="Correct tasks /64",
        title="A  Context vs. four updates",
    )
    a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncols=3,
        frameon=False,
        fontsize=7,
        handlelength=1.1,
        columnspacing=0.8,
    )
    fidelity = data["fresh_selected_fidelity"]
    first, last = "adapted_untrained_bidirectional", "adapted_bidirectional"
    for source, initial in fidelity[first]["per_source"].items():
        b.plot(
            [0, 1],
            [initial, fidelity[last]["per_source"][source]],
            color="#B8B8B8",
            linewidth=1,
            marker="o",
            markersize=3,
        )
    means = [fidelity[key]["source_equal_mean"] for key in [first, last]]
    b.plot([0, 1], means, color="#17668B", linewidth=2, marker="o")
    for x, value in enumerate(means):
        b.annotate(
            f"{value:.3f}",
            (x, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#17668B",
            weight="bold",
            fontsize=8,
        )
    b.set(
        xticks=[0, 1],
        xticklabels=["Unfitted", "Fitted"],
        xlim=(-0.2, 1.2),
        ylim=(0, 0.48),
        ylabel="Forward KL to reference",
        title="B  Fresh-source teacher fit",
    )
    labels = [
        label if key == "baseline" else label.replace(" update", "") for key, label in CONDITIONS
    ]
    values = [counts[key + "/Q"]["correct"] for key, _ in CONDITIONS]
    c.barh(
        range(6), values, color=["#777777", "#C26A34", "#17668B", "#99BAC9", "#A8A8A8", "#C5C5C5"]
    )
    c.invert_yaxis()
    c.set(
        yticks=range(6),
        yticklabels=labels,
        xlim=(0, 64),
        xticks=[0, 16, 32, 48, 64],
        xlabel="Correct query tasks /64",
        title="C  Causal student outcomes",
    )
    for y, value in enumerate(values):
        c.text(value + 0.7, y, str(value), va="center", fontsize=8)
    # The longer condition labels need more room than the top-left plot's tick labels.
    c.set_position([0.25, 0.09, 0.72, 0.30])
    save(fig, "context-fidelity-transfer")


def fidelity_figure() -> None:
    data = json.loads((FOLDER / "source-fidelity025-independent.json").read_text())
    if data["status"] != "independent_saved_tensor_audit" or len(data["sources"]) != 8:
        raise ValueError("Require complete025 audit")
    # Public-report print layout refinement only; all historical values unchanged.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.95, hspace=0.74)
    regions, total = data["regions_pooled_descriptive"], data["pooled_positions_descriptive"]
    measures = ["positions", "initial_kl_sum", "post_kl_sum"]
    bottom = [0.0] * 3
    for label, keys, color in [
        ("Rule text", ["rule_text"], "#17668B"),
        ("Array values", ["array_value"], "#C26A34"),
        ("Format / stop", ["array_format", "stop_token"], "#B5B5B5"),
    ]:
        values = [
            100 * sum(regions[key][metric] for key in keys) / total[metric] for metric in measures
        ]
        axes[0].bar(range(3), values, bottom=bottom, width=0.5, color=color, label=label)
        if label == "Rule text":
            for x, value in enumerate(values):
                axes[0].text(x, value / 2, f"{value:.1f}%", color="white", ha="center", va="center")
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    axes[0].set(
        xticks=range(3),
        xticklabels=["Native positions", "Initial KL", "Post-update KL"],
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
        ylabel="Share of total (%)",
        title="A  Source KL is concentrated in rule text",
    )
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=3, frameon=False, fontsize=8
    )
    errors = [
        source["first_incorrect_array_position"]
        for source in data["sources"]
        if source["first_incorrect_array_position"] is not None
    ]
    for offset, (label, key, color) in enumerate(
        [
            ("Initial student", "initial_expected_value_probability", "#777777"),
            ("PI teacher", "teacher_expected_value_probability", "#17668B"),
            ("After four updates", "post_expected_value_probability", "#C26A34"),
        ]
    ):
        values = [100 * row[key] for row in errors]
        xs = [i + (offset - 1) * 0.25 for i in range(4)]
        axes[1].bar(xs, values, width=0.22, color=color, label=label)
        for x, value in zip(xs, values, strict=True):
            axes[1].text(
                x,
                value + 0.8,
                "<0.1" if 0 < value < 0.1 else f"{value:.1f}",
                ha="center",
                fontsize=8,
            )
    axes[1].set(
        xticks=range(4),
        xticklabels=["Translation 1", "Translation 2", "Recoloring 1", "Recoloring 2"],
        ylim=(0, 32),
        yticks=[0, 10, 20, 30],
        ylabel="P(correct value), %",
        title="B  First wrong source-array value",
    )
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=3, frameon=False, fontsize=8
    )
    save(fig, "source-fidelity-localization025")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    context_figure()
    fidelity_figure()
    print("Rendered two print-layout figure pairs from the unchanged verified data.")


if __name__ == "__main__":
    main()
