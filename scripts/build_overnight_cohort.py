"""Generate outcome-blind meta-train/dev/confirmation chunks with frozen role assignment."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from hindsight.cohorts import load_cohort

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "balanced", ROOT / "scripts/build_balanced_development.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    exclusions = [ROOT / "data/balanced-development-v1", ROOT / "data/balanced-transfer-v1"]
    chunks = []
    partitions = ["meta_train"] * 4 + ["development"] * 2 + ["confirmation"] * 4
    for index, partition in enumerate(partitions):
        destination = output / f"chunk-{index:02d}"
        BUILDER.build(
            destination,
            ROOT / "data/corpus-v2",
            seed=202609061000 + index,
            additional_corpora=exclusions,
        )
        exclusions.append(destination)
        chunks.append(
            {
                "path": destination.name,
                "partition": partition,
                "sha256": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(),
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
        "partition_authority": (
            "This collection assigns fresh chunks before any model outcomes; "
            "child development labels are legacy storage format only."
        ),
        "confirmation_rule": (
            "No confirmation outcomes may select model, recipe, checkpoint, "
            "proposer or global hyperparameters."
        ),
        "limitations": [
            "Four familiar operator families; not unseen-family generalization.",
            "Global exclusion means not strictly IID; "
            "fixed strata do not guarantee equal difficulty.",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    episodes, _ = load_cohort(output)
    print(
        json.dumps(
            {
                "validated_episodes": len(episodes),
                "manifest_sha256": hashlib.sha256(
                    (output / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
