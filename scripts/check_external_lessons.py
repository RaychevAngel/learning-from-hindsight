"""Fixed development diagnostic of externally supplied source-only lessons."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hindsight.experiment import Experiment, write_json
from hindsight.model import Generation

CONDITIONS = (("baseline", None), ("direct_pi", 0), ("adapted_pi", 0))
PLANNED_WORK = {
    "generation_sequences": 144, "scoring_forwards": 4,
    "student_optimizer_steps": 16, "teacher_optimizer_steps": 0,
    "teacher_example_exposures": 0,
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(
    sources: Path, lessons: Path, data: Path, output: Path, *, engine=None,
    balanced_development: bool = False, generation_batch_size: int | None = None,
) -> int:
    # A diagnostic is never resumed, overwritten, or selected using its outcomes.
    output.mkdir(parents=True, exist_ok=False)
    source_count, validation_count = (8, 8) if balanced_development else (4, 4)
    observation_count = source_count * (validation_count + 8) * len(CONDITIONS)
    planned_work = {**PLANNED_WORK, "generation_sequences": observation_count,
                    "scoring_forwards": source_count, "student_optimizer_steps": source_count * 4}
    receipt = {
        "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
        "sources_directory": str(sources.resolve()), "lessons_file": str(lessons.resolve()),
        "data_directory": str(data.resolve()), "script_sha256": sha256(Path(__file__).read_bytes()),
        "selection": "none; " + ("eight" if balanced_development else "four")
        + " supplied development episodes, one fixed lesson each",
        "expected_observations": observation_count, "actual_observations": 0,
        "missing_observations": observation_count, "planned_work": planned_work,
        "source_generation_sequences": 0, "proposal_generation_sequences": 0,
        "source_provenance": [], "completed_episode_ids": [],
    }
    write_json(output / "receipt.json", receipt)
    experiment = None
    try:
        if generation_batch_size is not None and (
            type(generation_batch_size) is not int or generation_batch_size not in (1, 4, 8)
        ):
            raise ValueError("generation_batch_size override must be 1, 4, or 8")
        lesson_bytes = lessons.read_bytes()
        supplied = json.loads(lesson_bytes)
        lesson_rows = supplied["rows"]
        if (not isinstance(supplied.get("provenance"), dict)
                or not isinstance(lesson_rows, list) or len(lesson_rows) != source_count):
            raise ValueError(f"Require provenance and exactly {source_count} lesson rows")
        episode_ids = [row["episode_id"] for row in lesson_rows]
        if len(set(episode_ids)) != source_count:
            raise ValueError(f"Require {source_count} unique development episode IDs")
        receipt.update(lessons_sha256=sha256(lesson_bytes),
                       lesson_provenance=supplied["provenance"], episode_ids=episode_ids)
        (output / "lessons.json").write_bytes(lesson_bytes)
        identity_bytes = (sources / "identity.json").read_bytes()
        parent = json.loads(identity_bytes)
        receipt["parent_identity_sha256"] = sha256(identity_bytes)
        config = dict(parent["config"])
        if generation_batch_size is not None:
            receipt["generation_batch_size_override"] = generation_batch_size
            config["generation_batch_size"] = generation_batch_size
        if config["student_steps"] != 4:
            raise ValueError("This diagnostic requires the original four student steps")
        manifest_bytes = (data / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if parent["dataset_manifest"] != manifest:
            raise ValueError("Dataset manifest differs from the saved source run")
        if balanced_development:
            if (manifest.get("protocol_version") != "balanced-development-v1"
                    or manifest.get("episodes") != 8 or manifest.get("tasks") != 136
                    or manifest.get("calibration_tasks") != 0
                    or manifest.get("partition_episodes") != {"development": 8}
                    or config.get("dataset_manifest_sha256") != sha256(manifest_bytes)
                    or config.get("diagnostic_source_ids") != episode_ids):
                raise ValueError("Require the pinned eight-source balanced-development-v1 plan")
            receipt["admission_protocol"] = "balanced-development-v1"
        config["external_lessons_diagnostic"] = {
            "lessons_sha256": receipt["lessons_sha256"], "episode_ids": episode_ids,
            "parent_identity_sha256": receipt["parent_identity_sha256"],
            "lesson_provenance": supplied["provenance"], "planned_work": planned_work,
        }
        experiment = Experiment(config, data, output, engine=engine)
        admitted = {ep.episode_id: ep for ep in experiment.episodes
                    if ep.partition == "development"}
        if balanced_development and (
            len(experiment.episodes) != 8 or set(episode_ids) != set(admitted)
            or experiment.calibration
            or Counter(ep.family for ep in admitted.values()) != dict.fromkeys(
                ("translation", "denoising", "size_recoloring", "gravity"), 2
            )
        ):
            raise ValueError("Require exactly two development episodes per fixed family")
        prepared = []
        for row in lesson_rows:
            episode_id, lesson = row["episode_id"], row["lesson"]
            receipt["current_episode_id"] = episode_id
            if episode_id not in admitted:
                raise ValueError("Only development episodes are allowed; no final partition")
            ep = admitted[episode_id]
            if len(ep.validation) != validation_count or len(ep.queries) != 8:
                raise ValueError(f"Each specified episode must contain V{validation_count}/Q8")
            if not isinstance(lesson, str) or not lesson.strip():
                raise ValueError("Every fixed lesson must be a nonempty string")
            if (any(marker in lesson for marker in ("<think>", "</think>"))
                    or re.search(r"FINAL:\s*\[", lesson)):
                raise ValueError("Supply only the validated lesson, not reasoning or an answer")
            source_path = sources / "sources" / f"{episode_id}.json"
            source_bytes = source_path.read_bytes()
            if sha256(source_bytes) != row["source_sha256"]:
                raise ValueError(f"Saved source SHA256 mismatch: {episode_id}")
            source = json.loads(source_bytes)
            generation = Generation(**source["generation"])
            if (source["task_id"] != ep.source.task_id
                    or generation.prompt != ep.source.prompt):
                raise ValueError("Saved source task or prompt differs from the admitted task")
            for tokens in (generation.prompt_ids, generation.response_ids):
                if (not isinstance(tokens, list) or not tokens
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError("Saved original token IDs must be nonempty integer lists")
            if not isinstance(source["feedback"], str):
                raise ValueError("Saved source feedback must be a string")
            (output / "sources").mkdir(exist_ok=True)
            (output / "sources" / source_path.name).write_bytes(source_bytes)
            receipt["source_provenance"].append({
                "episode_id": episode_id, "partition": "development",
                "task_id": source["task_id"], "source_path": str(source_path.resolve()),
                "source_sha256": row["source_sha256"],
                "prompt_sha256": sha256(generation.prompt.encode()),
                "lesson_sha256": sha256(lesson.encode()),
                "prompt_tokens": len(generation.prompt_ids),
                "response_tokens": len(generation.response_ids),
            })
            prepared.append((ep, generation, source["feedback"], lesson))
        # Admit every input and declare every outcome before the first model call.
        for ep, _, _, _ in prepared:
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                for condition, candidate in CONDITIONS:
                    for task in tasks:
                        experiment.expect(experiment.metadata(ep, task, role, condition, candidate))
        if len(experiment.expected) != observation_count:
            raise ValueError(f"Diagnostic requires exactly {observation_count} unique outcome rows")
        write_json(output / "receipt.json", receipt)
        experiment.check_time()
        for ep, generation, feedback, lesson in prepared:
            receipt["current_episode_id"] = ep.episode_id
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                experiment.evaluate(ep, tasks, role, "baseline")
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                experiment.evaluate(ep, tasks, role, "direct_pi", candidate=0, pi=lesson)
            target = experiment.score(generation, feedback, pi=lesson)
            experiment.adaptation(ep, generation, target, "adapted_pi", candidate=0)
            del target
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                experiment.evaluate(ep, tasks, role, "adapted_pi", candidate=0, adapter="student")
            receipt["completed_episode_ids"].append(ep.episode_id)
            write_json(output / "receipt.json", receipt)
        if experiment.expected != set(experiment.observations):
            raise ValueError("Diagnostic observations do not exactly match the declared plan")
        if experiment.get_engine().counters != planned_work:
            raise ValueError("Actual model work differs from the fixed diagnostic work plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["work"] = (dict(engine.counters) if engine is not None
                           else dict.fromkeys(PLANNED_WORK, 0))
        if experiment is not None:
            receipt["actual_observations"] = len(experiment.observations)
            receipt["missing_observations"] = observation_count - len(experiment.observations)
            receipt["generated_evaluation_tokens"] = sum(
                row["generated_tokens"] for row in experiment.observations.values()
            )
            if experiment.engine is not None:
                receipt["work"] = dict(experiment.engine.counters)
        write_json(output / "receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "lessons", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--balanced-development", action="store_true")
    parser.add_argument("--generation-batch-size", type=int, choices=(1, 4, 8))
    args = parser.parse_args()
    return run(args.sources, args.lessons, args.data, args.output,
               balanced_development=args.balanced_development,
               generation_batch_size=args.generation_batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
