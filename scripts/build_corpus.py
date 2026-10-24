"""Generate the fixed ARC-1D experiment corpus without model calls.

Admission checks are independent of any model/treatment outcomes. Source code
is pinned and vendored. This program refuses to overwrite existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from itertools import groupby
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
SOURCE_HASH = "6f5a06a909149e5eebe2b285eb334e79d3ca27a6f77e0f31fff9188f709688aa"
FAMILIES = {
    "translation": ("move_2pix_colorful_right", "move_2pix_colorful_left"),
    "denoising": ("block_and_noise_remove",),
    "size_recoloring": ("recolor_blocks_by_size",),
    "gravity": ("gravity_right", "gravity_left"),
    "endpoint_filling": ("two_points_and_fill", "two_points_and_fill_inv"),
    "copying_to_markers": ("copy_block_to_dots_colors",),
    "pattern_repetition": ("repeat_pattern_full",),
    "block_toward_marker": ("block_touch_dot",),
    "reflection": ("reflect_block_around_dot",),
    "block_size_sorting": ("sort_blocks_by_size",),
    "collision_filling": ("fill_until_collision",),
    "half_block_coloring": ("color_left_half_blocks",),
}
KNOWN = tuple(FAMILIES)[:8]
RESERVED = tuple(FAMILIES)[8:]
VERSION = "arc1d-hindsight-v1"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def runs(row):
    offset, result = 0, []
    for color, values in groupby(row):
        width = len(list(values))
        if color:
            result.append((offset, width, color))
        offset += width
    return result


def independent_answer(alias, row):
    """Independent input-to-output implementations, not generator trace replay.

    None means an input lies outside a supported transform's stated domain.
    These functions also supply the finite witness hypothesis library.
    """
    n, blocks = len(row), runs(row)
    nonzero = [i for i, x in enumerate(row) if x]
    if not nonzero:
        return None
    out = [0] * n
    if alias.startswith("shift_"):
        amount = int(alias[6:])
        if any(not 0 <= i + amount < n for i in nonzero):
            return None
        for i in nonzero:
            out[i + amount] = row[i]
        return out
    if alias.startswith("move_2pix_colorful"):
        return independent_answer("shift_2" if alias.endswith("right") else "shift_-2", row)
    if alias == "identity":
        return list(row)
    if alias == "reverse":
        return list(reversed(row))
    if alias == "block_and_noise_remove":
        if sum(width > 1 for _, width, _ in blocks) != 1:
            return None
        for start, width, color in blocks:
            if width > 1:
                out[start : start + width] = [color] * width
        return out
    if alias == "recolor_blocks_by_size":
        if len(blocks) != 2 or blocks[0][1] == blocks[1][1]:
            return None
        largest = max(width for _, width, _ in blocks)
        for start, width, _ in blocks:
            out[start : start + width] = [1 if width == largest else 2] * width
        return out
    if alias in ("gravity_right", "gravity_left"):
        # Pinned upstream names are counterintuitive: gravity_right compacts LEFT.
        pixels = [row[i] for i in nonzero]
        return (
            pixels + [0] * (n - len(pixels))
            if alias == "gravity_right"
            else [0] * (n - len(pixels)) + pixels
        )
    if alias == "two_points_and_fill":
        if len(nonzero) != 2 or row[nonzero[0]] != row[nonzero[1]]:
            return None
        out[nonzero[0] : nonzero[-1] + 1] = [row[nonzero[0]]] * (nonzero[-1] - nonzero[0] + 1)
        return out
    if alias == "two_points_and_fill_inv":
        if len(blocks) != 1 or blocks[0][1] < 2:
            return None
        out[nonzero[0]] = out[nonzero[-1]] = row[nonzero[0]]
        return out
    if alias == "copy_block_to_dots_colors":
        if not blocks or blocks[0][0] != 0 or blocks[0][1] not in (3, 5):
            return None
        width = blocks[0][1]
        if any(length != 1 for _, length, _ in blocks[1:]):
            return None
        out[:width] = row[:width]
        for start, _, color in blocks[1:]:
            left, right = start - width // 2, start + width // 2 + 1
            if left < width or right > n:
                return None
            out[left:right] = [color] * width
        return out
    if alias == "repeat_pattern_full":
        length = nonzero[-1] + 1
        if nonzero != list(range(length)) or length % 2 or length // 2 not in (2, 3, 4, 5):
            return None
        period = length // 2
        if row[:period] != row[period:length]:
            return None
        return [row[i % period] for i in range(n)]
    if alias in ("block_touch_dot", "reflect_block_around_dot"):
        marker = 1 if alias == "block_touch_dot" else 2
        markers = [i for i, x in enumerate(row) if x == marker]
        other = [(s, w, c) for s, w, c in blocks if c != marker]
        if len(markers) != 1 or len(other) != 1:
            return None
        dot = markers[0]
        start, width, color = other[0]
        out[dot] = marker
        if alias == "reflect_block_around_dot":
            destinations = [2 * dot - i for i in range(start, start + width)]
        else:
            dest = dot - width if start < dot else dot + 1
            destinations = list(range(dest, dest + width))
        if any(not 0 <= i < n or i == dot for i in destinations):
            return None
        for i in destinations:
            out[i] = color
        return out
    if alias == "sort_blocks_by_size":
        if len({color for _, _, color in blocks}) != 1 or len(blocks) < 2:
            return None
        cursor = 0
        for _, width, color in sorted(blocks, key=lambda b: b[1]):
            if cursor + width > n:
                return None
            out[cursor : cursor + width] = [color] * width
            cursor += width + 1
        return out
    if alias == "fill_until_collision":
        markers = [i for i, color in enumerate(row) if color == 5]
        if len(markers) != 1 or markers[0] not in (0, n - 1):
            return None
        out = list(row)
        points = [i for i in nonzero if i != markers[0]]
        if len(points) < 2:
            return None
        if markers[0] == 0:
            for i in range(1, max(points)):
                if not row[i]:
                    out[i] = row[min(p for p in points if p > i)]
        else:
            for i in range(min(points) + 1, n - 1):
                if not row[i]:
                    out[i] = row[max(p for p in points if p < i)]
        return out
    if alias == "color_left_half_blocks":
        if any(c != 2 or w < 2 for _, w, c in blocks) or len(blocks) < 2:
            return None
        out = list(row)
        for start, width, _ in blocks:
            out[start : start + width // 2] = [8] * (width // 2)
        return out
    raise ValueError(f"Unknown independent transform: {alias}")


HYPOTHESES = tuple(
    dict.fromkeys(
        [alias for aliases in FAMILIES.values() for alias in aliases]
        + ["identity", "reverse"]
        + [f"shift_{i}" for i in range(-4, 5) if i]
    )
)


def normalized(rows, reverse=False):
    palette = {0: 0}
    result = []
    for row in rows:
        row = row[::-1] if reverse else row
        for color in row:
            if color not in palette:
                palette[color] = len(palette)
        result.append([palette[color] for color in row])
    return result


def pair_signature(pair):
    rows = [pair["input"], pair["output"]]
    return hashlib.sha256(
        min(encoded(normalized(rows)), encoded(normalized(rows, True)))
    ).hexdigest()


def witness_check(pairs):
    """No consistent supported transform may disagree on this query.

    This finite-library check is not universal inductive identifiability.
    """
    consistent = []
    for hypothesis in HYPOTHESES:
        if all(independent_answer(hypothesis, p["input"]) == p["output"] for p in pairs[:4]):
            consistent.append(hypothesis)
            if independent_answer(hypothesis, pairs[4]["input"]) != pairs[4]["output"]:
                return None
    return consistent or None


def prompt_for(demos, query):
    lines = [
        ("Infer the common transformation from the four demonstrations and apply it to the query."),
        "Give a brief explanation, then exactly one final line: FINAL: [0, 1, ...].",
        "The FINAL field must be a JSON array of integers with the same length as the query.",
    ]
    for i, pair in enumerate(demos, 1):
        lines.extend(
            [
                f"Example {i} input: {json.dumps(pair['input'])}",
                f"Example {i} output: {json.dumps(pair['output'])}",
            ]
        )
    lines.append(f"Query input: {json.dumps(query)}")
    return "\n".join(lines)


def build(output: Path, seed: int = 20260905):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = ROOT / "third_party/reasoning_gym/arc_1d_tasks.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Vendored generator hash mismatch")
    spec = importlib.util.spec_from_file_location("pinned_arc_1d", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operators = module.ARC_1D_TASKS
    exclusion_file = ROOT / "third_party/reasoning_gym/preview_exclusions.json"
    exclusions = json.loads(exclusion_file.read_text())
    preview_pairs = set(exclusions["canonical_pair_sha256"])
    public, private, episodes = [], [], []
    seen_pairs, seen_exact, seen_tasks, seeds = set(), set(), set(), set()
    rejected = Counter()

    def task(family, alias, partition, episode_id, role, index):
        key = [VERSION, seed, partition, family, episode_id, role, index]
        task_id = digest(key)[:24]
        task_seed = int(digest(key + ["rng"])[:16], 16)
        if task_seed in seeds:
            raise ValueError("Seed collision")
        seeds.add(task_seed)
        rng = Random(task_seed)
        generator, kwargs = operators[alias]
        max_size = 32 if family == "pattern_repetition" else 24
        for attempt in range(500):
            size = rng.randint(16, max_size)
            pairs, signatures = [], set()
            for _ in range(2000):
                pair = generator(rng, size, **kwargs)
                if pair is None or pair["input"] == pair["output"]:
                    rejected["empty_or_identity"] += 1
                    continue
                if (
                    set(pair) != {"input", "output"}
                    or any(len(row) != size for row in pair.values())
                    or any(
                        type(v) is not int or not 0 <= v <= 9 for row in pair.values() for v in row
                    )
                ):
                    raise ValueError(f"Invalid generator pair from {alias}")
                if independent_answer(alias, pair["input"]) != pair["output"]:
                    raise ValueError(f"Independent oracle disagreement: {alias}, {pair}")
                signature = pair_signature(pair)
                if signature in preview_pairs:
                    rejected["planning_preview_overlap"] += 1
                    continue
                if signature in seen_pairs or signature in signatures:
                    rejected["canonical_pair_duplicate"] += 1
                    continue
                pairs.append(pair)
                signatures.add(signature)
                if len(pairs) == 5:
                    break
            if len(pairs) != 5:
                rejected["task_pair_budget"] += 1
                continue
            hypotheses = witness_check(pairs)
            if hypotheses is None:
                rejected["finite_witness_disagreement"] += 1
                continue
            canonical_task = digest([pair_signature(p) for p in pairs])
            if canonical_task in seen_tasks:
                rejected["canonical_task_duplicate"] += 1
                continue
            seen_tasks.add(canonical_task)
            seen_pairs.update(signatures)
            seen_exact.update(digest(p) for p in pairs)
            demos, query = pairs[:4], pairs[4]
            record = {
                "task_id": task_id,
                "prompt": prompt_for(demos, query["input"]),
                "demonstrations": demos,
                "query_input": query["input"],
            }
            public.append(record)
            private.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "partition": partition,
                    "role": role,
                    "family": family,
                    "generator_alias": alias,
                    "seed": task_seed,
                    "size": size,
                    "answer": query["output"],
                    "public_sha256": digest(record),
                    "oracle_sha256": digest(query["output"]),
                    "canonical_task_sha256": canonical_task,
                    "pair_sha256": [digest(p) for p in pairs],
                    "canonical_pair_sha256": [pair_signature(p) for p in pairs],
                    "consistent_supported_hypotheses": hypotheses,
                    "admission_attempt": attempt + 1,
                }
            )
            return task_id
        raise RuntimeError(f"Could not admit task {task_id} / {alias} within generation cap")

    for family in KNOWN:
        aliases = FAMILIES[family]
        for i in range(2):
            task(family, aliases[i % len(aliases)], "calibration", None, "calibration", i)
    for partition, families, count in (
        ("meta_train", KNOWN, 8),
        ("development", KNOWN, 2),
        ("final_known", KNOWN, 2),
        ("final_unseen", RESERVED, 4),
    ):
        for family in families:
            aliases = FAMILIES[family]
            for i in range(count):
                episode_id = digest([VERSION, seed, partition, family, i])[:24]
                alias = aliases[i % len(aliases)]
                episode = {"episode_id": episode_id, "partition": partition, "family": family}
                for role, amount in (("source", 1), ("validation", 4), ("queries", 8)):
                    ids = [
                        task(family, alias, partition, episode_id, role, j) for j in range(amount)
                    ]
                    episode[role] = ids[0] if role == "source" else ids
                episodes.append(episode)
    expected = (112, 1472, 7360)
    if (len(episodes), len(public), len(seen_pairs)) != expected or len(seen_exact) != 7360:
        raise ValueError("Final corpus counts or pair uniqueness failed")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, records in (
        ("tasks.public.jsonl", public),
        ("tasks.private.jsonl", private),
        ("episodes.jsonl", episodes),
    ):
        raw = b"".join(encoded(record) + b"\n" for record in records)
        (output / name).write_bytes(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    registry = {}
    for family, aliases in FAMILIES.items():
        registry[family] = {
            "partition_scope": "known" if family in KNOWN else "reserved",
            "admitted_aliases": aliases,
            "length_range": [16, 32 if family == "pattern_repetition" else 24],
        }
    manifest = {
        "version": VERSION,
        "seed": seed,
        "generator_revision": REVISION,
        "generator_sha256": SOURCE_HASH,
        "status": "data_qualified_model_unqualified",
        "episodes": 112,
        "tasks": 1472,
        "calibration_tasks": 16,
        "demonstrations_per_task": 4,
        "validation_per_episode": 4,
        "queries_per_episode": 8,
        "partition_episodes": dict(Counter(e["partition"] for e in episodes)),
        "family_registry": registry,
        "files": files,
        "checks": {
            "independently_verified_pairs": 7360,
            "exact_unique_pairs": 7360,
            "mirror_color_canonical_unique_pairs": 7360,
            "canonical_unique_tasks": 1472,
            "unique_task_seeds": len(seeds),
            "finite_hypothesis_witness_tasks": 1472,
            "model_calls": 0,
        },
        "finite_witness_hypotheses": HYPOTHESES,
        "planning_preview_excluded_pairs": len(preview_pairs),
        "planning_preview_exclusion_sha256": hashlib.sha256(
            exclusion_file.read_bytes()
        ).hexdigest(),
        "rejections": dict(rejected),
        "alias_split_policy": (
            "All unadmitted aliases of a semantic family are excluded, not assigned to "
            "another split. Mirrored/parameter aliases belong to the same family."
        ),
        "qualification_limits": [
            (
                "Witnesses distinguish query predictions only among the listed supported "
                "transforms; they do not prove universal identification from finite examples."
            ),
            (
                "Task difficulty, token limits, answer formatting and model behavior require "
                "development-only model preflight."
            ),
            "No model outcome or treatment response was used to admit or reject data.",
            "Whole semantic operator families are held out; ARC-1D remains a shared domain.",
            (
                "Calibration tasks are newly generated using the preview's familiar families; "
                "the prior preview remains excluded from final data."
            ),
        ],
        "development_revisions": [
            (
                "Root approved repetition-only lengths16-32: period2-5 generators at lengths16-24 "
                "have insufficient globally color/mirror-distinct pairs for the fixed allocation. "
                "Other families stay16-24; tokenizer preflight remains required."
            ),
            (
                "Endpoint filling includes forward and inverse aliases within the same familiar "
                "family to avoid structural exhaustion; each episode fixes its alias across S/V/Q."
            ),
        ],
        "provenance": {
            "generator_url": f"https://raw.githubusercontent.com/open-thought/reasoning-gym/{REVISION}/reasoning_gym/arc/arc_1d_tasks.py",
            "license": "Apache-2.0",
            "license_file": "third_party/reasoning_gym/LICENSE",
            "notice_file": "third_party/reasoning_gym/NOTICE.txt",
            "copied_from_verified_development_preview": True,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    manifest = build(args.output, args.seed)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": manifest["episodes"],
                "tasks": manifest["tasks"],
                "checks": manifest["checks"],
                "rejections": manifest["rejections"],
            }
        )
    )


if __name__ == "__main__":
    main()
