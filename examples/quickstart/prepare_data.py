"""Build a self-contained demo cohort, without historical data/results or model calls."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from hindsight.cohorts import load_cohort
from hindsight.experiment import write_json

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "quickstart_builder", ROOT / "scripts/build_balanced_development.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def build(output, exclusions=()):
    output.mkdir(parents=True, exist_ok=False)
    support = output / "support"
    original, revised = support / "corpus-v1", support / "corpus-v2"
    BUILDER.base.build(original)
    BUILDER.prompt_revision.revise(original, revised)
    prior = list(exclusions)
    chunks = []
    for index, partition in enumerate(
        ["meta_train"] * 4 + ["development"] * 2 + ["confirmation"] * 4
    ):
        path = output / f"chunk-{index:02d}"
        BUILDER.build(path, revised, seed=202609070000 + index, additional_corpora=prior)
        prior.append(path)
        chunks.append(
            {
                "path": path.name,
                "partition": partition,
                "sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
            }
        )
        print(json.dumps({"completed_chunk": index, "partition": partition}), flush=True)
    manifest = {
        "protocol_version": "hindsight-cohort-v1",
        "episodes": 80,
        "partition_episodes": {"meta_train": 32, "development": 16, "confirmation": 32},
        "unique_inputs": 6800,
        "role_counts": {"source": 1, "validation": 8, "queries": 8},
        "chunks": chunks,
        "model_calls": 0,
        "purpose": "New demo cohort, not historical overnight-v2 or original final data",
        "seed_base": 202609070000,
        "explicit_exclusions": [str(path.resolve()) for path in exclusions],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)
    episodes, _ = load_cohort(output)
    return {
        "episodes": len(episodes),
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-corpus", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.exclude_corpus)))
