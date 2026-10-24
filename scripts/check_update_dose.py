"""Three fixed, weaker student-update arms; development V only, no Q generation or selection."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from hindsight.experiment import Experiment, observation_key, read_json, rows, seed_for, write_json
from hindsight.model import Generation, forward_kl

ARMS = [
    {"condition": "dose_lr1e-5_steps1", "lr": 1e-5, "steps": 1},
    {"condition": "dose_lr1e-5_steps4", "lr": 1e-5, "steps": 4},
    {"condition": "dose_lr1e-4_steps1", "lr": 1e-4, "steps": 1},
]
WORK = {"generation_sequences": 48, "scoring_forwards": 16,
        "student_optimizer_steps": 24, "teacher_optimizer_steps": 0,
        "teacher_example_exposures": 0}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(sources: Path, lessons: Path, data: Path, reference: Path, output: Path,
        *, engine=None) -> int:
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__).read_bytes()), "arms": ARMS,
        "planned_work": WORK, "expected_observations": 48, "actual_observations": 0,
        "missing_observations": 48, "selection": "none; all three fixed arms reported on V only",
        "source_generation_sequences": 0, "proposal_generation_sequences": 0,
        "query_generation_sequences": 0, "source_provenance": [], "completed_arms": [],
        "sources_directory": str(sources.resolve()),
        "reference_directory": str(reference.resolve()),
    }
    write_json(output / "receipt.json", receipt)
    experiment = None
    try:
        lesson_bytes = lessons.read_bytes()
        supplied = json.loads(lesson_bytes)
        lesson_rows = supplied["rows"]
        if (not isinstance(supplied.get("provenance"), dict)
                or not isinstance(lesson_rows, list) or len(lesson_rows) != 4):
            raise ValueError("Require provenance and exactly four fixed lesson rows")
        episode_ids = [row["episode_id"] for row in lesson_rows]
        if len(set(episode_ids)) != 4:
            raise ValueError("Require four unique development episodes")
        parent_bytes = (sources / "identity.json").read_bytes()
        parent = json.loads(parent_bytes)
        config = dict(parent["config"])
        if config["student_steps"] != 4 or config["learning_rate"] != 1e-4:
            raise ValueError("Parent must use the original 1e-4/four-step update")
        if parent["dataset_manifest"] != read_json(data / "manifest.json"):
            raise ValueError("Dataset manifest differs from the saved source run")
        ref_identity = read_json(reference / "identity.json")
        ref_receipt = read_json(reference / "receipt.json")
        ref_config = dict(ref_identity["config"])
        ref_diagnostic = ref_config.pop("external_lessons_diagnostic", None)
        expected_diagnostic = {
            "lessons_sha256": sha256(lesson_bytes), "episode_ids": episode_ids,
            "parent_identity_sha256": sha256(parent_bytes),
            "lesson_provenance": supplied["provenance"],
            "planned_work": {**WORK, "generation_sequences": 144,
                             "scoring_forwards": 4, "student_optimizer_steps": 16},
        }
        if (ref_config != config or ref_diagnostic != expected_diagnostic
                or ref_identity["dataset_manifest"] != parent["dataset_manifest"]
                or ref_receipt.get("status") != "completed"
                or ref_receipt.get("work") != expected_diagnostic["planned_work"]
                or ref_receipt.get("actual_observations") != 144
                or ref_receipt.get("missing_observations") != 0
                or ref_receipt.get("lessons_sha256") != sha256(lesson_bytes)
                or ref_receipt.get("episode_ids") != episode_ids
                or (reference / "lessons.json").read_bytes() != lesson_bytes):
            raise ValueError("Reference model/config/source/lesson identity or completion mismatch")
        receipt.update(
            episode_ids=episode_ids, lessons_sha256=sha256(lesson_bytes),
            parent_identity_sha256=sha256(parent_bytes), lesson_provenance=supplied["provenance"],
            reference_hashes={name: sha256((reference / name).read_bytes()) for name in
                              ("identity.json", "receipt.json", "observations.jsonl")},
        )
        config["update_dose_diagnostic"] = {
            key: receipt[key] for key in ("arms", "planned_work", "episode_ids", "lessons_sha256",
                                         "parent_identity_sha256", "reference_hashes")
        }
        experiment = Experiment(config, data, output, engine=engine)
        if experiment.identity["code"] != ref_identity["code"]:
            raise ValueError("Core implementation differs from the fixed reference")
        admitted = {ep.episode_id: ep for ep in experiment.episodes
                    if ep.partition == "development"}
        prepared = []
        reference_sources = {row["episode_id"]: row for row in ref_receipt["source_provenance"]}
        for row in lesson_rows:
            episode_id, lesson = row["episode_id"], row["lesson"]
            if episode_id not in admitted:
                raise ValueError("Only development episodes are allowed; no final partition")
            ep = admitted[episode_id]
            if len(ep.validation) != 4 or len(ep.queries) != 8:
                raise ValueError("Require the original V4/Q8 episode shape; Q remains unevaluated")
            if (not isinstance(lesson, str) or not lesson.strip()
                    or any(marker in lesson for marker in ("<think>", "</think>", "FINAL:"))):
                raise ValueError("Supply only a nonempty validated lesson")
            source_path = sources / "sources" / f"{episode_id}.json"
            source_bytes = source_path.read_bytes()
            if (sha256(source_bytes) != row["source_sha256"]
                    or (reference / "sources" / source_path.name).read_bytes() != source_bytes
                    or reference_sources[episode_id]["source_sha256"] != row["source_sha256"]
                    or reference_sources[episode_id]["lesson_sha256"] != sha256(lesson.encode())):
                raise ValueError("Original/reference source or lesson hash mismatch")
            source = json.loads(source_bytes)
            generation = Generation(**source["generation"])
            if source["task_id"] != ep.source.task_id or generation.prompt != ep.source.prompt:
                raise ValueError("Saved source task/prompt differs from the admitted task")
            for tokens in (generation.prompt_ids, generation.response_ids):
                if not isinstance(tokens, list) or not tokens or any(
                    type(token) is not int or token < 0 for token in tokens
                ):
                    raise ValueError("Saved original token IDs must be nonempty integer lists")
            if not isinstance(source["feedback"], str):
                raise ValueError("Saved feedback must be a string")
            (output / "sources").mkdir(exist_ok=True)
            (output / "sources" / source_path.name).write_bytes(source_bytes)
            receipt["source_provenance"].append(reference_sources[episode_id])
            prepared.append((ep, generation, source["feedback"], lesson))
        (output / "lessons.json").write_bytes(lesson_bytes)
        # Only V records enter this diagnostic; Q records are neither compared nor selected.
        reference_v = [row for row in rows(reference / "observations.jsonl") if row["role"] == "V"]
        expected_reference = {
            observation_key(experiment.metadata(ep, task, "V", condition, candidate))
            for ep, _, _, _ in prepared for task in ep.validation
            for condition, candidate in (("baseline", None), ("direct_pi", 0), ("adapted_pi", 0))
        }
        if (len(reference_v) != 48
                or {observation_key(row) for row in reference_v} != expected_reference):
            raise ValueError("Reference must contain exactly the 48 matched V observations")
        write_json(output / "reference-v.json", reference_v)
        for ep, _, _, _ in prepared:
            for arm in ARMS:
                for task in ep.validation:
                    experiment.expect(experiment.metadata(ep, task, "V", arm["condition"], 0))
        if len(experiment.expected) != 48:
            raise ValueError("The declared V-only plan must contain exactly 48 rows")
        write_json(output / "receipt.json", receipt)
        for ep, generation, feedback, lesson in prepared:
            target = experiment.score(generation, feedback, pi=lesson)
            for arm in ARMS:
                current = experiment.get_engine()
                current.reset_student(seed_for(config["seed"], "student_initialization"))
                metrics = current.adapt(generation, target, steps=arm["steps"], lr=arm["lr"])
                post = experiment.score(generation, feedback, adapter="student")
                experiment.event(
                    "student_adapted", episode_id=ep.episode_id, candidate=0,
                    condition=arm["condition"], lr=arm["lr"], steps=arm["steps"], metrics=metrics,
                    post_update_source_kl_target_to_student=float(forward_kl(post, target)),
                )
                del post
                experiment.evaluate(ep, ep.validation, "V", arm["condition"], 0, adapter="student")
                receipt["completed_arms"].append({"episode_id": ep.episode_id, **arm})
                write_json(output / "receipt.json", receipt)
            del target
        if experiment.expected != set(experiment.observations):
            raise ValueError("Actual observations differ from the predeclared V-only plan")
        if experiment.get_engine().counters != WORK:
            raise ValueError("Actual compute differs from the fixed work plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["work"] = dict(engine.counters) if engine is not None else dict.fromkeys(WORK, 0)
        if experiment is not None:
            receipt["actual_observations"] = len(experiment.observations)
            receipt["missing_observations"] = 48 - len(experiment.observations)
            if experiment.engine is not None:
                receipt["work"] = dict(experiment.engine.counters)
        write_json(output / "receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "lessons", "data", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    return run(args.sources, args.lessons, args.data, args.reference, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
