"""Generate the explicitly admitted fresh development sources for a pinned student."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hindsight.experiment import Experiment, write_json

DATASET_MANIFEST_SHA256 = "9554c4a805967ed7f7fb52edbb94c5c246512a20c70ff673be6d812ca9552065"
PLANNED_WORK = {
    "generation_sequences": 4, "scoring_forwards": 0,
    "student_optimizer_steps": 0, "teacher_optimizer_steps": 0,
    "teacher_example_exposures": 0,
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(
    config_path: Path, data: Path, output: Path, *, engine=None,
    balanced_development: bool = False,
) -> int:
    # A new model requires fresh native token IDs, never cached old-model sources.
    output.mkdir(parents=True, exist_ok=False)
    source_count = 8 if balanced_development else 4
    planned_work = {**PLANNED_WORK, "generation_sequences": source_count}
    receipt = {
        "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "data_directory": str(data.resolve()), "planned_work": planned_work,
        "expected_sources": source_count, "actual_sources": 0, "missing_sources": source_count,
        "completed_episode_ids": [], "source_provenance": [],
        "selection": ("eight" if balanced_development else "four")
        + " predeclared development source IDs; no outcome-based selection",
        "proposer_input_fields": ["record_id", "task", "attempt", "feedback"],
        "proposal_generation_sequences": 0, "evaluation_generation_sequences": 0,
    }
    write_json(output / "receipt.json", receipt)
    experiment = None
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        ids = config.get("diagnostic_source_ids")
        if (not isinstance(ids, list) or len(ids) != source_count
                or any(not isinstance(key, str) or not key for key in ids)
                or len(set(ids)) != source_count):
            raise ValueError(f"Require {source_count} unique predeclared diagnostic_source_ids")
        if (not isinstance(config.get("model_name"), str) or not config["model_name"].strip()
                or not isinstance(config.get("revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", config["revision"])):
            raise ValueError("Require model_name and a full immutable model revision")
        manifest_bytes = (data / "manifest.json").read_bytes()
        manifest_hash = sha256(manifest_bytes)
        if balanced_development:
            manifest = json.loads(manifest_bytes)
            pin = config.get("dataset_manifest_sha256")
            if (not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin)
                    or pin != manifest_hash):
                raise ValueError("Require the exact fresh dataset_manifest_sha256 in config")
            if (manifest.get("protocol_version") != "balanced-development-v1"
                    or manifest.get("episodes") != 8 or manifest.get("tasks") != 136
                    or manifest.get("calibration_tasks") != 0
                    or manifest.get("partition_episodes") != {"development": 8}):
                raise ValueError("Require the bounded balanced-development-v1 manifest")
            receipt["admission_protocol"] = "balanced-development-v1"
        elif manifest_hash != DATASET_MANIFEST_SHA256:
            raise ValueError("Require the unchanged original corpus-v2 manifest")
        receipt.update(
            config_sha256=sha256(config_bytes), model_name=config["model_name"],
            revision=config["revision"], episode_ids=ids, manifest_sha256=manifest_hash,
        )
        experiment = Experiment(config, data, output, engine=engine)
        admitted = {ep.episode_id: ep for ep in experiment.episodes
                    if ep.partition == "development"}
        if any(key not in admitted for key in ids):
            raise ValueError("Every declared source must belong to development")
        selected = [admitted[key] for key in ids]
        if len({ep.source.task_id for ep in selected}) != source_count:
            raise ValueError(f"Require {source_count} distinct source task identities")
        if balanced_development and (
            len(experiment.episodes) != 8 or set(ids) != set(admitted)
            or experiment.calibration
            or any(len(ep.validation) != 8 or len(ep.queries) != 8 for ep in selected)
            or Counter(ep.family for ep in selected) != dict.fromkeys(
                ("translation", "denoising", "size_recoloring", "gravity"), 2
            )
        ):
            raise ValueError("Require exactly two development S1/V8/Q8 episodes per fixed family")
        receipt["expected"] = [
            {"episode_id": ep.episode_id, "task_id": ep.source.task_id,
             "partition": "development", "prompt_sha256": sha256(ep.source.prompt.encode())}
            for ep in selected
        ]
        # Write the complete admission/limit receipt before model loading or generation.
        write_json(output / "receipt.json", receipt)
        backend = experiment.get_engine()
        if backend.counters != dict.fromkeys(PLANNED_WORK, 0):
            raise ValueError("Source preparation requires a fresh unused engine")
        for index, ep in enumerate(selected):
            source_path = output / "sources" / f"{ep.episode_id}.json"
            if source_path.exists():
                raise ValueError("Refusing cached source reuse in fresh model preparation")
            receipt["current_episode_id"] = ep.episode_id
            generation, feedback = experiment.source(ep)
            expected_work = {**dict.fromkeys(PLANNED_WORK, 0),
                             "generation_sequences": index + 1}
            if backend.counters != expected_work:
                raise ValueError("Source generation exceeded its declared per-call work")
            proposer_path = output / "proposer-inputs" / f"{ep.episode_id}.json"
            write_json(proposer_path, {
                "record_id": ep.episode_id, "task": ep.source.prompt,
                "attempt": generation.text, "feedback": feedback,
            })
            receipt["source_provenance"].append({
                "episode_id": ep.episode_id, "task_id": ep.source.task_id,
                "source_sha256": sha256(source_path.read_bytes()),
                "proposer_input_path": str(proposer_path.resolve()),
                "proposer_input_sha256": sha256(proposer_path.read_bytes()),
                "prompt_tokens": len(generation.prompt_ids),
                "response_tokens": len(generation.response_ids), "cap_hit": generation.cap_hit,
            })
            receipt["completed_episode_ids"].append(ep.episode_id)
            receipt["actual_sources"] = len(receipt["completed_episode_ids"])
            receipt["missing_sources"] = source_count - receipt["actual_sources"]
            receipt["work"] = dict(backend.counters)
            write_json(output / "receipt.json", receipt)
        if backend.counters != planned_work:
            raise ValueError(
                f"Expected exactly {source_count} source generations and no other work"
            )
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        backend = experiment.engine if experiment is not None else engine
        receipt["work"] = (dict(backend.counters) if backend is not None
                           else dict.fromkeys(PLANNED_WORK, 0))
        write_json(output / "receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--balanced-development", action="store_true")
    args = parser.parse_args()
    return run(args.config, args.data, args.output,
               balanced_development=args.balanced_development)


if __name__ == "__main__":
    raise SystemExit(main())
