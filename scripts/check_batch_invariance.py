"""Frozen-model, fixed-input batch invariance check for four executed development episodes."""

import argparse
import dataclasses
import hashlib
import itertools
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from hindsight.data import grade
from hindsight.experiment import Experiment, append_json, digest, read_json, rows, write_json
from hindsight.model import with_pi

LAYOUTS = ("original_roles", "batch1", "batch4", "batch8")
CONDITIONS = ("baseline", "direct_pi")
PLANNED_WORK = {
    "generation_sequences": 384,
    "scoring_forwards": 0,
    "student_optimizer_steps": 0,
    "teacher_optimizer_steps": 0,
    "teacher_example_exposures": 0,
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def key(row):
    return tuple(row[name] for name in ("episode_id", "task_id", "role", "condition"))


def batch_groups(ep, layout):
    ordered = [("V", task) for task in ep.validation] + [("Q", task) for task in ep.queries]
    if layout == "original_roles":
        return [ordered[:4], ordered[4:]]
    size = int(layout.removeprefix("batch"))
    return [ordered[offset : offset + size] for offset in range(0, len(ordered), size)]


def compare(left_name, right_name, left, right, *, tokens=True):
    if left.keys() != right.keys():
        raise ValueError("Comparison task/role/condition identities do not match")
    fields = ["text", "cap_hit", "grade"] + (["prompt_ids", "response_ids"] if tokens else [])
    counts = dict.fromkeys(fields, 0)
    mismatches = []
    for identity, a in left.items():
        b = right[identity]
        changed = []
        for field in fields:
            av = a["grade"] if field == "grade" else a["generation"][field]
            bv = b["grade"] if field == "grade" else b["generation"][field]
            if av != bv:
                counts[field] += 1
                changed.append(field)
        if changed:
            mismatches.append(
                {
                    **dict(
                        zip(("episode_id", "task_id", "role", "condition"), identity, strict=True)
                    ),
                    "fields": changed,
                }
            )
    return {
        "left": left_name,
        "right": right_name,
        "compared_tasks": len(left),
        "token_comparison_available": tokens,
        "mismatch_counts": counts,
        "mismatches": mismatches,
        "invariant": not mismatches,
    }


def run(
    sources: Path,
    reference: Path,
    data: Path,
    output: Path,
    *,
    max_seconds: float = 1200,
    engine=None,
) -> int:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources_directory": str(sources.resolve()),
        "reference_directory": str(reference.resolve()),
        "data_directory": str(data.resolve()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "max_seconds": max_seconds,
        "planned_work": PLANNED_WORK,
        "expected_records": 384,
        "actual_records": 0,
        "completed_batches": 0,
        "source_provenance": [],
        "invariance": None,
        "selection": "none; identical fixed inputs, no adaptation or outcome-based selection",
        "old_reference_limit": "Saved018 has raw text, grades and caps but no response token IDs",
    }
    write_json(output / "receipt.json", receipt)
    experiment = None
    generated_rows = []
    try:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Require a positive finite invocation limit")
        parent = read_json(sources / "identity.json")
        previous = read_json(reference / "identity.json")
        reference_receipt = read_json(reference / "receipt.json")
        lessons_bytes = (reference / "lessons.json").read_bytes()
        lessons = read_json(reference / "lessons.json")
        parent_hash = sha256((sources / "identity.json").read_bytes())
        lesson_hash = sha256(lessons_bytes)
        config = dict(previous["config"])
        diagnostic = config.pop("external_lessons_diagnostic")
        ids = config.get("diagnostic_source_ids")
        if (
            reference_receipt["status"] != "completed"
            or config != parent["config"]
            or diagnostic["parent_identity_sha256"] != parent_hash
            or diagnostic["lessons_sha256"] != lesson_hash
        ):
            raise ValueError("Reference config, source identity or fixed lessons do not match")
        if (
            config.get("deterministic_execution") is not True
            or config.get("generation_batch_size") != 8
            or not isinstance(ids, list)
            or len(ids) != 4
            or len(set(ids)) != 4
            or [row["episode_id"] for row in lessons["rows"]] != ids
            or diagnostic["episode_ids"] != ids
            or reference_receipt["completed_episode_ids"] != ids
        ):
            raise ValueError("Require strict deterministic original grouping and four executed IDs")
        manifest = read_json(data / "manifest.json")
        if parent["dataset_manifest"] != manifest or previous["dataset_manifest"] != manifest:
            raise ValueError("Dataset manifest changed")
        receipt["identity"] = {
            "model_name": config["model_name"],
            "revision": config["revision"],
            "source_identity_sha256": parent_hash,
            "lessons_sha256": lesson_hash,
            "reference_identity_sha256": sha256((reference / "identity.json").read_bytes()),
            "reference_observations_sha256": sha256(
                (reference / "observations.jsonl").read_bytes()
            ),
            "manifest_sha256": sha256((data / "manifest.json").read_bytes()),
            "config": config,
            "config_sha256": digest(config),
        }
        experiment = Experiment(config, data, output, engine=engine)
        for name in ("model.py", "experiment.py", "data.py"):
            if previous["code"][name] != experiment.identity["code"][name]:
                raise ValueError(f"Core implementation differs from saved018: {name}")
        original_check = experiment.check_time

        def check_time():
            original_check()
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Batch diagnostic invocation deadline reached")

        experiment.check_time = check_time
        admitted = {
            ep.episode_id: ep for ep in experiment.episodes if ep.partition == "development"
        }
        selected = []
        for lesson in lessons["rows"]:
            if lesson["episode_id"] not in admitted:
                raise ValueError("Only the executed development episodes are allowed")
            ep = admitted[lesson["episode_id"]]
            if len(ep.validation) != 4 or len(ep.queries) != 8 or not lesson["lesson"].strip():
                raise ValueError("Require V4/Q8 and a nonempty fixed lesson")
            source_path = sources / "sources" / f"{ep.episode_id}.json"
            source_bytes = source_path.read_bytes()
            if (
                sha256(source_bytes) != lesson["source_sha256"]
                or source_bytes != (reference / "sources" / source_path.name).read_bytes()
            ):
                raise ValueError("Original source bytes differ from lesson/reference identity")
            source = read_json(source_path)
            if (
                source["task_id"] != ep.source.task_id
                or source["generation"]["prompt"] != ep.source.prompt
            ):
                raise ValueError("Saved source task/prompt differs from current corpus")
            receipt["source_provenance"].append(
                {
                    "episode_id": ep.episode_id,
                    "task_id": ep.source.task_id,
                    "source_sha256": lesson["source_sha256"],
                    "lesson_sha256": sha256(lesson["lesson"].encode()),
                }
            )
            selected.append((ep, lesson["lesson"]))
        (output / "lessons.json").write_bytes(lessons_bytes)
        expected_keys = {
            (ep.episode_id, task.task_id, role, condition)
            for ep, _ in selected
            for role, tasks in (("V", ep.validation), ("Q", ep.queries))
            for task in tasks
            for condition in CONDITIONS
        }
        old = {}
        for row in rows(reference / "observations.jsonl"):
            if row["condition"] not in CONDITIONS:
                continue
            identity = key(row)
            if (
                identity not in expected_keys
                or identity in old
                or row["partition"] != "development"
                or row["candidate"] != (0 if row["condition"] == "direct_pi" else None)
            ):
                raise ValueError("Saved replay observations have unexpected/duplicate identities")
            old[identity] = {
                "generation": {"text": row["text"], "cap_hit": row["cap_hit"]},
                "grade": {name: row[name] for name in ("success", "format_ok", "prediction")},
            }
        if len(expected_keys) != 96 or old.keys() != expected_keys:
            raise ValueError("Require exactly 96 saved baseline/direct-PI observations")
        plan = []
        for layout in LAYOUTS:
            for ep, lesson in selected:
                for condition in CONDITIONS:
                    for batch in batch_groups(ep, layout):
                        plan.append((ep, lesson, layout, condition, batch))
        write_json(
            output / "plan.json",
            {
                "layouts": list(LAYOUTS),
                "conditions": list(CONDITIONS),
                "planned_work": PLANNED_WORK,
                "task_order": "Within each episode: original V order followed by original Q order",
                "batches": [
                    {
                        "batch_id": index,
                        "layout": layout,
                        "episode_id": ep.episode_id,
                        "condition": condition,
                        "tasks": [{"task_id": t.task_id, "role": role} for role, t in batch],
                    }
                    for index, (ep, _, layout, condition, batch) in enumerate(plan)
                ],
            },
        )
        write_json(output / "receipt.json", receipt)
        check_time()
        backend = experiment.get_engine()
        if backend.counters != dict.fromkeys(PLANNED_WORK, 0):
            raise ValueError("Require a fresh unused engine")
        for batch_id, (ep, lesson, layout, condition, batch) in enumerate(plan):
            prompts = [
                with_pi(task.prompt, lesson) if condition == "direct_pi" else task.prompt
                for _, task in batch
            ]
            before = time.monotonic()
            values = experiment.generate(
                prompts,
                max_new_tokens=config["evaluation_tokens"],
                sample=False,
                seed=config["seed"],
                adapter=None,
            )
            elapsed = time.monotonic() - before
            companions = [{"task_id": t.task_id, "role": role} for role, t in batch]
            for index, ((role, task), prompt, value) in enumerate(
                zip(batch, prompts, values, strict=True)
            ):
                if value.prompt != prompt:
                    raise ValueError("Returned generation prompt does not match requested task")
                record = {
                    **experiment.metadata(
                        ep, task, role, condition, 0 if condition == "direct_pi" else None
                    ),
                    "layout": layout,
                    "batch_id": batch_id,
                    "batch_index": index,
                    "batch_size": len(batch),
                    "batch_companions": companions,
                    "batch_elapsed_seconds": elapsed,
                    "prompt_sha256": sha256(prompt.encode()),
                    "generation": dataclasses.asdict(value),
                    "grade": grade(value.text, task),
                }
                append_json(output / "generations.jsonl", record)
                generated_rows.append(record)
            receipt.update(
                actual_records=len(generated_rows),
                completed_batches=batch_id + 1,
                work=dict(backend.counters),
                elapsed_seconds=time.monotonic() - started,
            )
            write_json(output / "receipt.json", receipt)
        if backend.counters != PLANNED_WORK or len(generated_rows) != 384:
            raise ValueError("Actual work differs from the fixed 384-sequence plan")
        grouped = {
            layout: {key(row): row for row in generated_rows if row["layout"] == layout}
            for layout in LAYOUTS
        }
        if any(set(group) != expected_keys for group in grouped.values()):
            raise ValueError("Fresh comparison identities do not match the declared tasks")
        fresh = [
            compare(a, b, grouped[a], grouped[b]) for a, b in itertools.combinations(LAYOUTS, 2)
        ]
        replay = [
            compare("saved018", layout, old, grouped[layout], tokens=False) for layout in LAYOUTS
        ]
        write_json(
            output / "mismatch-matrix.json", {"fresh_pairwise": fresh, "saved_replay": replay}
        )
        receipt.update(
            status="completed",
            batch_invariance=all(row["invariant"] for row in fresh),
            original_role_replay_invariance=replay[0]["invariant"],
            invariance=all(row["invariant"] for row in fresh + replay),
        )
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        backend = experiment.engine if experiment is not None else engine
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            actual_records=len(generated_rows),
            missing_records=384 - len(generated_rows),
            elapsed_seconds=time.monotonic() - started,
            work=dict(backend.counters) if backend is not None else dict.fromkeys(PLANNED_WORK, 0),
        )
        write_json(output / "receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "reference", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    return run(args.sources, args.reference, args.data, args.output, max_seconds=args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
