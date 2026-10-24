"""Independent CPU audit of saved full-vocabulary source distributions from 025.

Computes in float64 from tensors, checks hashes/native alignment, and describes
signal concentration. No model, tokenizer, new labels, or GPU is instantiated.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/source-fidelity-qwen35-025"
OUT = ROOT / "paper/derived"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"positions": 0}
    initial_sum = sum(row["initial_kl"] for row in rows)
    post_sum = sum(row["post_kl"] for row in rows)
    initial_mismatch = [row for row in rows if row["initial_argmax"] != row["target_argmax"]]
    return {
        "positions": len(rows),
        "initial_kl_mean": initial_sum / len(rows),
        "post_kl_mean": post_sum / len(rows),
        "initial_kl_sum": initial_sum,
        "post_kl_sum": post_sum,
        "kl_reduction_fraction": 1 - post_sum / initial_sum if initial_sum > 0 else None,
        "initial_argmax_disagreements": len(initial_mismatch),
        "disagreements_resolved": sum(
            row["post_argmax"] == row["target_argmax"] for row in initial_mismatch
        ),
        "new_argmax_disagreements": sum(
            row["initial_argmax"] == row["target_argmax"]
            and row["post_argmax"] != row["target_argmax"]
            for row in rows
        ),
        "post_argmax_disagreements": sum(
            row["post_argmax"] != row["target_argmax"] for row in rows
        ),
        "kl_worsened_positions": sum(row["post_kl"] > row["initial_kl"] + 1e-7 for row in rows),
        "total_positive_kl_change": sum(max(0, row["post_kl"] - row["initial_kl"]) for row in rows),
        "initial_total_variation_mean": mean(row["initial_tv"] for row in rows),
        "post_total_variation_mean": mean(row["post_tv"] for row in rows),
        "max_post_kl": max(row["post_kl"] for row in rows),
    }


def region(display: str, position: int, final_start: int) -> str:
    if position < final_start:
        return "rule_text"
    readable = display.replace("Ġ", " ").replace("Ċ", "\n").strip()
    if re.fullmatch(r"[0-9]+", readable):
        return "array_value"
    if readable.startswith("<|"):
        return "stop_token"
    return "array_format"


def audit() -> dict:
    torch.set_num_threads(2)
    receipt = json.loads((RUN / "receipt.json").read_text())
    if receipt["status"] != "completed" or receipt["actual_sources"] != 8:
        raise ValueError("Require completed eight-source receipt")
    expected_work = {
        "generation_sequences": 0,
        "scoring_forwards": 16,
        "student_optimizer_steps": 32,
        "teacher_optimizer_steps": 0,
        "teacher_example_exposures": 0,
    }
    if receipt["work"] != expected_work:
        raise ValueError("Unexpected scientific work counts")
    metrics_rows = lines(RUN / "source-metrics.jsonl")
    reconstruct = lines(RUN / "reconstruction.jsonl")
    if len(metrics_rows) != 8 or len(reconstruct) != 8:
        raise ValueError("Incomplete metadata")
    expected_ids = set(receipt["episode_ids"])
    if len(expected_ids) != 8 or {row["episode_id"] for row in metrics_rows} != expected_ids:
        raise ValueError("Source identity matrix does not match the completed receipt")
    if {row["episode_id"] for row in reconstruct} != expected_ids:
        raise ValueError("Reconstruction source identity matrix differs")
    history_path = ROOT / "results/hindsight-teacher-qwen35-023/events.jsonl"
    if sha(history_path) != "8d2c9367aa5f4fb5ac26bbe44678a7fad9967388f8dfa8f32623eea49e5c31e2":
        raise ValueError("Original023 training-event identity changed")
    historical_events = [
        row
        for row in lines(history_path)
        if row.get("event") == "student_adapted" and row.get("condition") == "adapted_pi"
    ]
    if (
        len(historical_events) != 8
        or {row["episode_id"] for row in historical_events} != expected_ids
    ):
        raise ValueError("Historical update event matrix differs")
    historical_by_source = {row["episode_id"]: row["metrics"] for row in historical_events}
    for entry in reconstruct:
        if not entry["identical"] or entry["actual_metrics"] != entry["historical_metrics"]:
            raise ValueError("Historical four-step reconstruction did not match")
        original_metrics = [
            {key: value for key, value in metric.items() if key != "elapsed_seconds"}
            for metric in historical_by_source[entry["episode_id"]]
        ]
        if entry["historical_metrics"] != original_metrics or len(original_metrics) != 4:
            raise ValueError("Reconstruction historical metrics disagree with original023 events")
    missing = [
        row["artifact"]["path"]
        for row in metrics_rows
        if not (RUN / "distributions" / row["artifact"]["path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Await complete backup: {missing}")

    original_sources = ROOT / "results/sources-balanced-qwen35-022/sources"
    lessons_path = ROOT / "results/frontier-lessons-balanced-qwen35-023.json"
    if sha(lessons_path) != "3810766140284b967ce11e28e5872f2725d3132393f4063b5e47e01e16b6fee9":
        raise ValueError("Frozen023 lesson file hash mismatch")
    lessons = {
        row["episode_id"]: row["lesson"] for row in json.loads(lessons_path.read_text())["rows"]
    }
    transfer_path = ROOT / "results/hindsight-teacher-qwen35-023/observations.jsonl"
    direct_path = ROOT / "results/order-and-pi-qwen35-024/generations.jsonl"
    for path, expected in [
        (transfer_path, "bcf190ecf0663408a5a1564f0412fb4ba31ee1b3d7ef7d7367597d5a09f947d1"),
        (direct_path, "a7a611b7c0751cf43e7081e12faae437de275866ef86ab33bcf8d7f024e27996"),
    ]:
        if sha(path) != expected:
            raise ValueError("Reused behavioral evidence identity changed")
    transfer = lines(transfer_path)
    direct = lines(direct_path)
    private_path = ROOT / "data/balanced-transfer-v1/tasks.private.jsonl"
    private_sha = sha(private_path)
    if private_sha != "b2c74b125b1b39605f8d3dd18b64faf6c141f8f0f21e0abd4dd28ac4b8e5b6da":
        raise ValueError("Frozen source gold identity changed")
    source_gold = {
        row["task_id"]: row["answer"] for row in lines(private_path) if row["role"] == "source"
    }
    result = {
        "status": "independent_saved_tensor_audit",
        "receipt_sha256": sha(RUN / "receipt.json"),
        "source_metrics_sha256": sha(RUN / "source-metrics.jsonl"),
        "reconstruction_sha256": sha(RUN / "reconstruction.jsonl"),
        "historical_training_events023_sha256": sha(history_path),
        "offline_source_gold_file_sha256": private_sha,
        "reused_observations023_sha256": sha(transfer_path),
        "reused_generations024_sha256": sha(direct_path),
        "method": "CPU float64 sum(exp(target) * (target - student)); original native positions",
        "work": receipt["work"],
        "all_reconstructed_updates_exact": True,
        "sources": [],
        "positions": [],
        "warnings": [
            "Finite source imitation is not downstream task performance.",
            "Token positions are correlated within sources; no token-level significance test.",
            "Text regions classify saved native token displays without re-tokenization.",
            "Top residual positions are posthoc descriptive examples, not training selection.",
            "Source gold is used only for offline analysis after scoring, never for new targets.",
            "Gold comparisons after an incorrect array prefix are not teacher task accuracy.",
        ],
    }
    for meta in metrics_rows:
        source_id = meta["episode_id"]
        lesson_sha = hashlib.sha256(
            json.dumps(lessons[source_id], sort_keys=True).encode()
        ).hexdigest()
        if lesson_sha != meta["lesson_sha256"]:
            raise ValueError(f"Per-source lesson identity mismatch: {source_id}")
        artifact = RUN / "distributions" / meta["artifact"]["path"]
        actual_sha = sha(artifact)
        if actual_sha != meta["artifact"]["sha256"]:
            raise ValueError(f"Tensor hash mismatch: {source_id}")
        if artifact.stat().st_size != meta["artifact"]["bytes"]:
            raise ValueError(f"Tensor size mismatch: {source_id}")
        source_path = original_sources / f"{source_id}.json"
        if sha(source_path) != meta["source_sha256"]:
            raise ValueError(f"Original source hash mismatch: {source_id}")
        original = json.loads(source_path.read_text())
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        if payload["format"] != "source-fidelity-full-vocabulary-logprobs-v1":
            raise ValueError("Unknown tensor format")
        ids = payload["response_ids"]
        if ids != original["generation"]["response_ids"]:
            raise ValueError("Source action IDs changed")
        if payload["prompt_ids"] != original["generation"]["prompt_ids"]:
            raise ValueError("Source prompt IDs changed")
        for mode in ("teacher", "student"):
            aligned = payload["alignment"][mode]
            actual = [aligned["input_ids"][p + 1] for p in aligned["predecessor_positions"]]
            if actual != ids:
                raise ValueError("Scoring/predecessor alignment mismatch")
            input_sha = hashlib.sha256(
                json.dumps(aligned["input_ids"], sort_keys=True).encode()
            ).hexdigest()
            if input_sha != aligned["input_ids_sha256"]:
                raise ValueError("Saved scoring input hash mismatch")
        displays = payload["alignment"]["response_token_displays"]
        starts = [i for i, display in enumerate(displays) if "FINAL" in display]
        if len(starts) != 1 or len(displays) != len(ids):
            raise ValueError("Cannot classify the saved original response regions")
        labels = dict(zip(ids, displays, strict=True))
        array_positions = [
            i
            for i, display in enumerate(displays)
            if region(display, i, starts[0]) == "array_value"
        ]
        historical_values = [int(displays[i].replace("Ġ", " ").strip()) for i in array_positions]
        if historical_values != original["grade"]["prediction"]:
            raise ValueError("Native array token values disagree with the original parsed answer")
        expected_values = source_gold[original["task_id"]]
        if len(historical_values) != len(expected_values):
            raise ValueError("Cannot compare arrays of different lengths positionwise")
        value_ids = {}
        for i, value in zip(array_positions, historical_values, strict=True):
            if value in value_ids and value_ids[value] != ids[i]:
                raise ValueError("Array values have non-unique native token identities")
            value_ids[value] = ids[i]
        if not set(expected_values).issubset(value_ids):
            raise ValueError("Gold value token identity is not present in the original response")
        array_lookup = {position: i for i, position in enumerate(array_positions)}
        wrong_indices = [
            i
            for i, (historical, expected) in enumerate(
                zip(historical_values, expected_values, strict=True)
            )
            if historical != expected
        ]
        all_tensors = [payload[key] for key in ("target", "initial", "post_fourth")]
        if any(t.dtype != torch.float32 or t.shape != (len(ids), 248320) for t in all_tensors):
            raise ValueError("Unexpected distribution dtype or shape")
        saved = json.loads((RUN / "distributions" / f"{source_id}.json").read_text())
        saved_positions = saved["metrics"]["positions"]
        rows = []
        max_metric_error = 0.0
        for offset in range(0, len(ids), 4):
            teacher, initial, post = [value[offset : offset + 4].double() for value in all_tensors]
            for value in (teacher, initial, post):
                if not torch.isfinite(value).all() or value.logsumexp(-1).abs().max() > 1e-4:
                    raise ValueError("Invalid saved distribution")
            q, p0, p4 = teacher.exp(), initial.exp(), post.exp()
            kl0 = (q * (teacher - initial)).sum(-1)
            kl4 = (q * (teacher - post)).sum(-1)
            tv0 = (q - p0).abs().sum(-1) / 2
            tv4 = (q - p4).abs().sum(-1) / 2
            target_top = q.argmax(-1)
            initial_top = p0.argmax(-1)
            post_top = p4.argmax(-1)
            desired = q - p0
            actual = p4 - p0
            desired_norm = (desired * desired).sum(-1)
            actual_norm = (actual * actual).sum(-1)
            inner = (desired * actual).sum(-1)
            for j in range(teacher.shape[0]):
                position = offset + j
                historical_id = ids[position]
                target_id = int(target_top[j])
                initial_id = int(initial_top[j])
                post_id = int(post_top[j])
                row = {
                    "episode_id": source_id,
                    "family": meta["family"],
                    "position": position,
                    "region": region(displays[position], position, starts[0]),
                    "historical_token_id": historical_id,
                    "historical_token_display": displays[position],
                    "native_display_window": displays[max(0, position - 3) : position + 4],
                    "target_argmax": target_id,
                    "initial_argmax": initial_id,
                    "post_argmax": post_id,
                    "target_argmax_display_if_observed": labels.get(target_id),
                    "initial_argmax_display_if_observed": labels.get(initial_id),
                    "post_argmax_display_if_observed": labels.get(post_id),
                    "initial_kl": float(kl0[j]),
                    "post_kl": float(kl4[j]),
                    "initial_tv": float(tv0[j]),
                    "post_tv": float(tv4[j]),
                    "teacher_historical_probability": float(q[j, historical_id]),
                    "initial_historical_probability": float(p0[j, historical_id]),
                    "post_historical_probability": float(p4[j, historical_id]),
                    "teacher_preferred_probability": float(q[j, target_id]),
                    "initial_teacher_preferred_probability": float(p0[j, target_id]),
                    "post_teacher_preferred_probability": float(p4[j, target_id]),
                    "probability_shift_projection": (
                        float(inner[j] / desired_norm[j]) if desired_norm[j] > 1e-16 else None
                    ),
                    "probability_shift_cosine": (
                        float(inner[j] / (desired_norm[j] * actual_norm[j]).sqrt())
                        if desired_norm[j] * actual_norm[j] > 1e-24
                        else None
                    ),
                }
                if position in array_lookup:
                    array_index = array_lookup[position]
                    expected_value = expected_values[array_index]
                    expected_id = value_ids[expected_value]
                    row.update(
                        {
                            "array_index": array_index,
                            "expected_array_value_offline": expected_value,
                            "expected_array_value_token_id": expected_id,
                            "historical_array_prefix_correct": not any(
                                i < array_index for i in wrong_indices
                            ),
                            "first_incorrect_array_position": bool(wrong_indices)
                            and array_index == wrong_indices[0],
                            "teacher_expected_value_probability": float(q[j, expected_id]),
                            "initial_expected_value_probability": float(p0[j, expected_id]),
                            "post_expected_value_probability": float(p4[j, expected_id]),
                        }
                    )
                for observed, recorded in [
                    (row["initial_kl"], "initial_forward_kl_teacher_to_student"),
                    (row["post_kl"], "post_fourth_forward_kl_teacher_to_student"),
                    (row["initial_tv"], "initial_total_variation"),
                    (row["post_tv"], "post_fourth_total_variation"),
                ]:
                    error = abs(observed - saved_positions[position][recorded])
                    max_metric_error = max(error, max_metric_error)
                    if error > 1e-5:
                        raise ValueError(
                            f"Independent metric mismatch {source_id}/{position}/{recorded}"
                        )
                rows.append(row)
        summary = summarize(rows)
        region_summaries = {
            name: summarize([row for row in rows if row["region"] == name])
            for name in ("rule_text", "array_value", "array_format", "stop_token")
        }
        query = {
            condition: sum(
                row["success"]
                for row in transfer
                if row["episode_id"] == source_id
                and row["role"] == "Q"
                and row["condition"] == condition
            )
            for condition in ("baseline", "adapted_pi")
        }
        query["direct_pi"] = sum(
            row["grade"]["success"]
            for row in direct
            if row["episode_id"] == source_id
            and row["role"] == "Q"
            and row["condition"] == "direct_pi"
        )
        largest = sorted(rows, key=lambda row: row["post_kl"], reverse=True)
        result["sources"].append(
            {
                "episode_id": source_id,
                "family": meta["family"],
                "source_correct": original["grade"]["success"],
                "tensor_sha256": actual_sha,
                "tensor_bytes": artifact.stat().st_size,
                "summary": summary,
                "regions": region_summaries,
                "maximum_independent_metric_difference": max_metric_error,
                "query_counts_reused_out_of_8": query,
                "top_five_post_kl_fraction": sum(row["post_kl"] for row in largest[:5])
                / summary["post_kl_sum"],
                "largest_five_residual_positions": largest[:5],
                "first_incorrect_array_position": next(
                    (row for row in rows if row.get("first_incorrect_array_position")), None
                ),
            }
        )
        result["positions"].extend(rows)
        del payload, all_tensors, teacher, initial, post, q, p0, p4, desired, actual

    result["pooled_positions_descriptive"] = summarize(result["positions"])
    result["source_equal_means"] = {
        metric: mean(source["summary"][metric] for source in result["sources"])
        for metric in (
            "initial_kl_mean",
            "post_kl_mean",
            "initial_total_variation_mean",
            "post_total_variation_mean",
        )
    }
    result["regions_pooled_descriptive"] = {
        name: summarize([row for row in result["positions"] if row["region"] == name])
        for name in ("rule_text", "array_value", "array_format", "stop_token")
    }
    return result


def write_report(data: dict) -> str:
    avg = data["source_equal_means"]
    pooled = data["pooled_positions_descriptive"]
    rules = data["regions_pooled_descriptive"]["rule_text"]
    initial_rule_share = 100 * rules["initial_kl_sum"] / pooled["initial_kl_sum"]
    post_rule_share = 100 * rules["post_kl_sum"] / pooled["post_kl_sum"]
    lines_out = [
        "# Independent source-fidelity025 tensor audit",
        "",
        "CPU-only analysis of existing saved distributions. "
        "No model or tokenizer calls; no new tasks.",
        "The target here is the fixed explicit-PI-conditioned causal teacher, "
        "not the fitted future-aware hindsight teacher.",
        "All eight tensor hashes and source hashes match; "
        "native action IDs and predecessor positions agree.",
        "All recorded reconstructed-update metrics match the original023 training events exactly "
        "after excluding wall-clock durations.",
        "",
        f"Source-equal mean KL: **{avg['initial_kl_mean']:.6f} -> {avg['post_kl_mean']:.6f}**.",
        f"Source-equal mean total variation: **{avg['initial_total_variation_mean']:.6f} "
        f"-> {avg['post_total_variation_mean']:.6f}**.",
        "These are source imitation measurements; "
        "existing Q outcomes are reused only for interpretation.",
        "",
        "| Source | Family | Correct source | Initial KL | Post-fourth KL "
        "| Reduction | Top5 residual share | Q baseline/PI update/direct PI |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for source in data["sources"]:
        s = source["summary"]
        q = source["query_counts_reused_out_of_8"]
        lines_out.append(
            f"| `{source['episode_id']}` | {source['family']} | {source['source_correct']} "
            f"| {s['initial_kl_mean']:.6f} | {s['post_kl_mean']:.6f} "
            f"| {100 * s['kl_reduction_fraction']:.1f}% "
            f"| {100 * source['top_five_post_kl_fraction']:.1f}% "
            f"| {q['baseline']}/{q['adapted_pi']}/{q['direct_pi']} |"
        )
    lines_out += [
        "",
        "## Localized signal",
        "",
        f"Across {pooled['positions']} correlated native positions, "
        "the teacher's argmax differs from the initial student's at "
        f"{pooled['initial_argmax_disagreements']} positions.",
        f"The update resolves {pooled['disagreements_resolved']} of those "
        f"and introduces {pooled['new_argmax_disagreements']} new disagreements; "
        f"{pooled['post_argmax_disagreements']} remain afterward.",
        f"KL increases by more than 1e-7 at {pooled['kl_worsened_positions']} positions, "
        "despite lower mean KL. The sum of all positive per-position increases is "
        f"{pooled['total_positive_kl_change']:.4f}; their count alone is not their importance.",
        f"Rule text contributes {initial_rule_share:.2f}% of initial KL and {post_rule_share:.2f}% "
        "of post-update KL, when summing over native positions.",
        "",
        "| Original response region | Positions | Initial mean KL | Post mean KL "
        "| Initial top1 disagreements | Resolved | New |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for region_name, row in data["regions_pooled_descriptive"].items():
        if row["positions"]:
            lines_out.append(
                f"| {region_name} | {row['positions']} | {row['initial_kl_mean']:.6f} "
                f"| {row['post_kl_mean']:.6f} | {row['initial_argmax_disagreements']} "
                f"| {row['disagreements_resolved']} | {row['new_argmax_disagreements']} |"
            )
    lines_out += [
        "",
        "Regions use native saved token displays: tokens before FINAL are rule text; "
        "digit tokens afterward are array values; "
        "other final-line tokens are format or stopping tokens.",
        "This is an explicit descriptive classification, "
        "not a new tokenizer alignment or an attribution of internal reasoning.",
        "",
        "## Largest remaining residuals",
        "",
        "Posthoc examples sorted by residual KL. Token IDs and distributions, "
        "not visible text alone, define the comparisons.",
        "",
        "| Source | Position | Region | Historical token display | Initial KL | Post KL "
        "| Teacher/initial/post top1 IDs |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in sorted(data["positions"], key=lambda row: row["post_kl"], reverse=True)[:16]:
        token = row["historical_token_display"].replace("|", "\\|")
        lines_out.append(
            f"| `{row['episode_id']}` | {row['position']} | {row['region']} | `{token}` "
            f"| {row['initial_kl']:.4f} | {row['post_kl']:.4f} "
            f"| {row['target_argmax']}/{row['initial_argmax']}/{row['post_argmax']} |"
        )
    lines_out += [
        "",
        "## First erroneous source array position",
        "",
        "For each incorrect source, compare the first wrong array value, "
        "whose earlier array entries are still correct. The preceding rule sentence "
        "is nevertheless the original one and can itself be wrong. Gold comes from "
        "the existing private source record and is used only in this offline audit; "
        "no distribution is rescored or target changed.",
        "",
        "| Source | Native position / array index | Historical / expected value "
        "| P(expected) initial / teacher / post | Teacher / post top1 is expected |",
        "|---|---:|---|---|---|",
    ]
    for source in data["sources"]:
        row = source["first_incorrect_array_position"]
        if row is None:
            continue
        expected_id = row["expected_array_value_token_id"]
        lines_out.append(
            f"| `{source['episode_id']}` | {row['position']} / {row['array_index']} "
            f"| {row['historical_token_display']} / {row['expected_array_value_offline']} "
            f"| {row['initial_expected_value_probability']:.4f} / "
            f"{row['teacher_expected_value_probability']:.4f} / "
            f"{row['post_expected_value_probability']:.4f} "
            f"| {row['target_argmax'] == expected_id} / {row['post_argmax'] == expected_id} |"
        )
    lines_out += [
        "",
        "## Interpretation boundary",
        "",
        "The update substantially fits the actual original-prefix targets. "
        "That does not certify those targets as useful, "
        "prove that the remaining local error is harmless, "
        "or show fidelity on corrected/unvisited prefixes or new tasks.",
        "Teacher argmax agreement is a descriptive proxy; full-distribution residuals "
        "and teacher-promoted probability shifts are retained in the JSON. "
        "Probability-shift projections are unbounded descriptive ratios, "
        "not percent task benefit.",
        "No arbitrary KL cutoff is declared sufficient, "
        "no causal explanation is inferred from eight source-level associations, "
        "and no recipe change follows from this audit alone.",
        "",
        "Reproduce: `.venv-qwen35/bin/python paper/audit_source_fidelity025.py`.",
        "Detailed per-position values and every tensor SHA256: "
        "`paper/derived/source-fidelity025-independent.json`.",
        "",
    ]
    return "\n".join(lines_out)


def main() -> None:
    data = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source-fidelity025-independent.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    (ROOT / "writing/source-fidelity025-independent.md").write_text(write_report(data))
    print(
        json.dumps(
            {
                "status": data["status"],
                "sources": len(data["sources"]),
                "positions": len(data["positions"]),
                "source_equal_means": data["source_equal_means"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
