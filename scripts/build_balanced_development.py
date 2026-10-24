"""Fresh source-only learning episodes with outcome-blind, balanced V/Q assignment.

Only public old-corpus inputs are read for overlap exclusion. No old private
answer or model-result file is opened. This is development, not a new final set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
VERSION = "arc1d-balanced-development-v1"
SEED = 2026090601
ALIASES = {
    "translation": "move_2pix_colorful_right",
    "denoising": "block_and_noise_remove",
    "size_recoloring": "recolor_blocks_by_size",
    "gravity": "gravity_right",
}
STRATA = {
    "length": "short=16..19; long=20..24",
    "translation": "small=nonzero block length <=7; large=>=8",
    "denoising": "small=longest nonzero run <=7; large=>=8",
    "size_recoloring": "small=longer block on left; large=longer block on right",
    "gravity": "small=nonzero count <=10; large=>=11",
}
BINS = tuple(product(("short", "long"), ("small", "large")))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


base = module("balanced_base", ROOT / "scripts/build_corpus.py")
prompt_revision = module("balanced_prompt", ROOT / "scripts/revise_prompt.py")


def input_signature(row):
    return hashlib.sha256(min(
        base.encoded(base.normalized([row])),
        base.encoded(base.normalized([row], True)),
    )).hexdigest()


def stratum(family, row):
    lengths = [width for _, width, _ in base.runs(row)]
    if family == "size_recoloring":
        if len(lengths) != 2 or lengths[0] == lengths[1]:
            raise ValueError("Recoloring requires two differently sized blocks")
        small = lengths[0] > lengths[1]
    elif family == "denoising":
        small = max(lengths) <= 7
    else:
        small = sum(value != 0 for value in row) <= (7 if family == "translation" else 10)
    return ("short" if len(row) <= 19 else "long", "small" if small else "large")


def old_public_exclusions(corpus):
    manifest_raw = (corpus / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    public_raw = (corpus / "tasks.public.jsonl").read_bytes()
    if hashlib.sha256(public_raw).hexdigest() != manifest["files"]["tasks.public.jsonl"]:
        raise ValueError("Old public corpus integrity failure")
    inputs = set()
    for line in public_raw.splitlines():
        row = json.loads(line)
        for values in [pair["input"] for pair in row["demonstrations"]] + [row["query_input"]]:
            inputs.add(input_signature(values))
    return inputs, hashlib.sha256(manifest_raw).hexdigest()


def build(output: Path, old_corpus: Path, seed: int = SEED, additional_corpora=()):
    output, old_corpus = Path(output), Path(old_corpus)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    generator_path = ROOT / "third_party/reasoning_gym/arc_1d_tasks.py"
    if hashlib.sha256(generator_path.read_bytes()).hexdigest() != base.SOURCE_HASH:
        raise ValueError("Vendored generator hash mismatch")
    operators = module("balanced_generator", generator_path).ARC_1D_TASKS
    excluded_inputs, parent_hash = old_public_exclusions(old_corpus)
    additional_exclusions = []
    for corpus in additional_corpora:
        inputs, manifest_hash = old_public_exclusions(Path(corpus))
        additional_exclusions.append({
            "manifest_sha256": manifest_hash,
            "canonical_inputs": len(inputs),
            "newly_excluded_canonical_inputs": len(inputs - excluded_inputs),
        })
        excluded_inputs.update(inputs)
    seen_inputs, seen_pairs = set(excluded_inputs), set()
    preview = json.loads((ROOT / "third_party/reasoning_gym/preview_exclusions.json").read_text())
    seen_pairs.update(preview["canonical_pair_sha256"])
    public, private, episodes, assignment = [], [], [], []
    rejected = Counter()

    def make_task(family, episode_id, serial, wanted=None):
        # No V/Q role enters task generation, rejection, or RNG seeding.
        task_id = base.digest([VERSION, seed, episode_id, serial])[:24]
        task_seed = int(base.digest([task_id, "generation"])[0:16], 16)
        rng = Random(task_seed)
        alias = ALIASES[family]
        generator, kwargs = operators[alias]
        for attempt in range(500):
            limits = (16, 24) if wanted is None else (
                (16, 19) if wanted[0] == "short" else (20, 24))
            size = rng.randint(*limits)
            pairs, inputs, signatures = [], set(), set()
            for _ in range(2000):
                pair = generator(rng, size, **kwargs)
                if pair is None or pair["input"] == pair["output"]:
                    rejected["empty_or_identity"] += 1
                    continue
                if (set(pair) != {"input", "output"}
                        or any(len(row) != size for row in pair.values())
                        or any(type(v) is not int or not 0 <= v <= 9
                               for row in pair.values() for v in row)
                        or base.independent_answer(alias, pair["input"]) != pair["output"]):
                    raise ValueError("Invalid generator pair or independent oracle disagreement")
                input_hash = input_signature(pair["input"])
                signature = base.pair_signature(pair)
                if input_hash in seen_inputs or input_hash in inputs:
                    rejected["canonical_input_overlap"] += 1
                    continue
                if signature in seen_pairs or signature in signatures:
                    rejected["canonical_pair_overlap"] += 1
                    continue
                if (len(pairs) == 4 and wanted is not None
                        and stratum(family, pair["input"]) != wanted):
                    rejected["query_outside_predeclared_stratum"] += 1
                    continue
                pairs.append(pair)
                inputs.add(input_hash)
                signatures.add(signature)
                if len(pairs) == 5:
                    break
            if len(pairs) != 5:
                rejected["pair_budget"] += 1
                continue
            hypotheses = base.witness_check(pairs)
            if hypotheses is None:
                rejected["finite_witness_disagreement"] += 1
                continue
            seen_inputs.update(inputs)
            seen_pairs.update(signatures)
            demos, query = pairs[:4], pairs[4]
            record = {"task_id": task_id, "demonstrations": demos, "query_input": query["input"],
                      "prompt": base.prompt_for(demos, query["input"])}
            record["prompt"] = prompt_revision.revise_prompt(record)
            public.append(record)
            secret = {
                "task_id": task_id, "episode_id": episode_id, "partition": "development",
                "role": None, "family": family, "generator_alias": alias, "seed": task_seed,
                "size": size, "answer": query["output"], "public_sha256": base.digest(record),
                "oracle_sha256": base.digest(query["output"]), "admission_attempt": attempt + 1,
                "pair_sha256": [base.digest(pair) for pair in pairs],
                "canonical_pair_sha256": [base.pair_signature(pair) for pair in pairs],
                "canonical_input_sha256": [input_signature(pair["input"]) for pair in pairs],
                "consistent_supported_hypotheses": hypotheses,
                "stratum": list(stratum(family, query["input"])), "pool_serial": serial,
            }
            private.append(secret)
            return secret
        raise RuntimeError(f"Could not fill predeclared stratum {family}/{wanted}")

    for family in ALIASES:
        for index in range(2):
            episode_id = base.digest([VERSION, seed, family, index])[:24]
            source = make_task(family, episode_id, "source")
            source["role"] = "source"
            roles = defaultdict(list)
            for bin_index, wanted in enumerate(BINS):
                pool = [make_task(family, episode_id, f"pool-{bin_index}-{j}", wanted)
                        for j in range(4)]
                assignment_seed = int(base.digest([episode_id, wanted, "assignment"])[0:16], 16)
                Random(assignment_seed).shuffle(pool)
                for j, row in enumerate(pool):
                    role = "validation" if j < 2 else "queries"
                    row["role"] = role
                    roles[role].append(row["task_id"])
                assignment.append({"episode_id": episode_id, "stratum": list(wanted),
                                   "seed": assignment_seed,
                                   "shuffled_ids": [r["task_id"] for r in pool]})
            episodes.append({"episode_id": episode_id, "partition": "development", "family": family,
                             "source": source["task_id"], **roles})
    counts = Counter((row["episode_id"], row["role"], tuple(row["stratum"]))
                     for row in private if row["role"] != "source")
    if (len(episodes) != 8 or len(public) != 136 or len(counts) != 64
            or set(counts.values()) != {2} or len(seen_inputs) - len(excluded_inputs) != 680):
        raise ValueError("Balanced development corpus counts or uniqueness failed")
    payloads = {name: b"".join(base.encoded(row) + b"\n" for row in rows) for name, rows in (
        ("tasks.public.jsonl", public), ("tasks.private.jsonl", private),
        ("episodes.jsonl", episodes), ("assignment.jsonl", assignment),
    )}
    manifest = {
        "version": VERSION, "protocol_version": "balanced-development-v1",
        "seed": seed, "status": "development_only_outcome_blind",
        "episodes": 8, "tasks": 136, "calibration_tasks": 0,
        "partition_episodes": {"development": 8},
        "role_counts": {"source": 1, "validation": 8, "queries": 8},
        "declared_families": list(ALIASES),
        "source_per_episode": 1, "validation_per_episode": 8, "queries_per_episode": 8,
        "demonstrations_per_task": 4, "aliases": ALIASES, "strata": STRATA,
        "tasks_per_role_per_stratum": 2, "model_calls": 0,
        "generator_revision": base.REVISION, "generator_sha256": base.SOURCE_HASH,
        "parent_manifest_sha256": parent_hash, "excluded_canonical_inputs": len(excluded_inputs),
        "new_unique_inputs": 680, "rejections": dict(rejected),
        "assignment": "Common independent four-task pool per stratum; seeded 2V/2Q split",
        "limitations": ["Strata balance observable query features, not exact task difficulty.",
                        "Global uniqueness filtering means tasks are not strictly IID.",
                        "Separate development replication; not an untouched final test."],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
    }
    if additional_exclusions:
        manifest["additional_excluded_corpora"] = additional_exclusions
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (output / name).write_bytes(raw)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-corpus", type=Path, default=ROOT / "data/corpus-v2")
    parser.add_argument("--exclude-corpus", type=Path, action="append", default=[],
                        help="Also exclude this corpus's verified public inputs; repeatable")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.old_corpus, args.seed,
                           additional_corpora=args.exclude_corpus), indent=2))


if __name__ == "__main__":
    main()
