"""Frozen five-arm shared-parameter experiment; no provider operations or retries."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import grade
from hindsight.experiment import append_json, digest, read_json, rows, seed_for, write_json
from hindsight.model import Generation, HFEngine, configure_deterministic_execution
from hindsight.shared import (
    ARMS,
    UPDATED_ARMS,
    SharedRecipe,
    TeachingExample,
    adapt_shared,
    adapter_values,
    alignment,
    audit_forward,
    distances,
    fit_shared,
    planned_work,
    restore,
    score,
    snapshot,
    storage_map,
    tensor_digest,
)

SOURCE_SPEC = importlib.util.spec_from_file_location(
    "shared_source_admission", Path(__file__).with_name("source_admission.py")
)
SOURCE_ADMISSION = importlib.util.module_from_spec(SOURCE_SPEC)
SOURCE_SPEC.loader.exec_module(SOURCE_ADMISSION)


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_teaching(manifest_path: Path, recipe: SharedRecipe):
    """Selected targets must have real validation-only evidence for this recipe."""
    manifest = read_json(manifest_path)
    if manifest.get("format") != "shared-teaching-v1":
        raise ValueError("Expected shared-teaching-v1 target manifest")
    if manifest.get("selection_recipe") != recipe.selection_recipe():
        raise ValueError("Cached target labels belong to another student recipe")
    selection_indices = manifest.get("selection_validation_indices", list(range(8)))
    if (
        not selection_indices
        or len(set(selection_indices)) != len(selection_indices)
        or any(type(index) is not int or not 0 <= index < 8 for index in selection_indices)
    ):
        raise ValueError("Selection validation indices must uniquely identify the V subset")

    def admitted_path(value, expected):
        path = (manifest_path.parent / value).resolve()
        if sha(path) != expected:
            raise ValueError(f"Pinned teaching artifact checksum differs: {path}")
        return path

    observations_path = admitted_path(
        manifest["selection_observations_path"], manifest["selection_observations_sha256"]
    )
    observed = rows(observations_path)
    examples = []
    ids = set()
    for entry in manifest["examples"]:
        ep_id = entry["episode_id"]
        if ep_id in ids:
            raise ValueError("Duplicate teacher training source")
        ids.add(ep_id)
        baseline = [
            row
            for row in observed
            if row["episode_id"] == ep_id and row["role"] == "V" and row["condition"] == "baseline"
        ]
        candidates = [
            row
            for row in observed
            if row["episode_id"] == ep_id
            and row["role"] == "V"
            and row["condition"] == manifest.get("selection_condition", "adapted_pi")
        ]
        baseline_ids = {row["task_id"] for row in baseline}
        if (
            not baseline
            or len(baseline_ids) != len(baseline)
            or any(type(row["success"]) is not bool for row in baseline + candidates)
        ):
            raise ValueError("Missing or invalid actual post-update validation evidence")
        scores = {}
        for candidate in {row["candidate"] for row in candidates}:
            group = [row for row in candidates if row["candidate"] == candidate]
            if (
                type(candidate) is not int
                or candidate < 0
                or len(group) != len(baseline)
                or {row["task_id"] for row in group} != baseline_ids
            ):
                raise ValueError("Candidate validation rows do not match baseline")
            scores[candidate] = sum(row["success"] for row in group)
        selected = entry["selected_candidate"]
        if selected is not None and type(selected) is not int:
            raise ValueError("Selected candidate must be an integer ID or neutral null")
        base_score = sum(row["success"] for row in baseline)
        best = max(scores.values(), default=base_score)
        if (selected is None and best > base_score) or (
            selected is not None
            and (selected not in scores or scores[selected] != best or best <= base_score)
        ):
            raise ValueError("Target choice violates actual-V selection or no-update tie rule")
        source_path = admitted_path(entry["source_path"], entry["source_sha256"])
        target_path = admitted_path(entry["target_path"], entry["target_sha256"])
        source = read_json(source_path)
        generation = Generation(**source["generation"])
        target = torch.load(target_path, map_location="cpu", weights_only=True)
        if isinstance(target, dict):
            target = target[entry.get("target_key", "target")]
        if not isinstance(target, torch.Tensor):
            raise ValueError("Selected target artifact must contain a tensor")
        provenance = {
            **entry,
            "derived_from": "post_update_V",
            "selection_recipe": recipe.selection_recipe(),
            "baseline_successes": base_score,
            "candidate_successes": scores,
            "selection_observations_sha256": manifest["selection_observations_sha256"],
            "source_task_id": source["task_id"],
            "validation_task_ids": sorted(baseline_ids),
            "selection_validation_indices": selection_indices,
            "target_lineage_verification": "Manifest-bound file bytes; no independent rescoring",
        }
        examples.append(TeachingExample(ep_id, generation, source["feedback"], target, provenance))
    if not examples:
        raise ValueError("Empty teacher training manifest")
    return examples, manifest


def admit_cohort(examples, episodes, ids, *, check_feedback=True):
    """Admit cohort structure; CLI defers grading until tokenizer admission."""
    admitted = {ep.episode_id: ep for ep in episodes}
    if len(ids) != 8 or len(set(ids)) != 8 or not set(ids) <= admitted.keys():
        raise ValueError("Initial E7 cut requires exactly eight known evaluation sources")
    if any(admitted[ep_id].partition != "development" for ep_id in ids) or sorted(
        Counter(admitted[ep_id].family for ep_id in ids).values()
    ) != [2, 2, 2, 2]:
        raise ValueError("Initial E7 sources must be two per family from development only")
    for item in examples:
        ep = admitted.get(item.episode_id)
        indices = item.provenance["selection_validation_indices"]
        if (
            ep is None
            or ep.partition != "meta_train"
            or len(ep.validation) != 8
            or item.generation.prompt != ep.source.prompt
            or item.provenance["source_task_id"] != ep.source.task_id
            or set(item.provenance["validation_task_ids"])
            != {ep.validation[index].task_id for index in indices}
        ):
            raise ValueError("Teacher source/V provenance does not match this meta-train cohort")
        if check_feedback:
            actual_feedback = "The submitted answer was " + (
                "correct." if grade(item.generation.text, ep.source)["success"] else "incorrect."
            )
            if item.feedback != actual_feedback:
                raise ValueError("Teacher source feedback differs from its actual source grade")
    return [admitted[ep_id] for ep_id in ids]


def run(config, examples, episodes, output: Path, *, engine=None, cohort=None):
    """Small callable runner; CLI additionally enforces the initial eight-source/two-seed cut."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    recipe = SharedRecipe(**config["recipe"])
    seeds = config["seeds"]
    maximum = config["max_invocation_seconds"]
    deadline = datetime.fromisoformat(config["absolute_deadline_utc"])
    batch_size = config.get("generation_batch_size", 1)
    if type(batch_size) is not int or batch_size not in (1, 4, 8):
        raise ValueError("Supported evaluation generation batch sizes are 1, 4 and 8")
    qualification_path = None
    if batch_size != 1:
        qualification = config["batch_qualification"]
        qualification_path = Path(qualification["path"])
        if sha(qualification_path) != qualification["sha256"]:
            raise ValueError("Batch qualification receipt checksum changed")
        qualified = read_json(qualification_path)
        if (
            qualified["status"] != "completed"
            or qualified["selected_batch_size"] != batch_size
            or not qualified["repeat_stability"].get(str(batch_size), False)
            or qualified["config"]["model_name"] != config["model_name"]
            or qualified["config"]["revision"] != config["revision"]
            or qualified["dataset_sha256"] != config["dataset_manifest_sha256"]
        ):
            raise ValueError("Batch qualification does not admit this model/dataset/batch")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int for seed in seeds)
        or not math.isfinite(maximum)
        or maximum <= 0
        or deadline.tzinfo is None
        or not episodes
        or len({ep.episode_id for ep in episodes}) != len(episodes)
    ):
        raise ValueError("Require unique seeds/sources and explicit bounded runtime")
    if {example.episode_id for example in examples} & {ep.episode_id for ep in episodes} or {
        example.generation.prompt for example in examples
    } & {task.prompt for ep in episodes for task in (ep.source, *ep.validation, *ep.queries)}:
        raise ValueError("Teacher training and evaluation sources/prompts overlap")
    work = planned_work(
        len(examples), len(episodes), sum(len(ep.queries) for ep in episodes), len(seeds), recipe
    )
    expected = [
        {
            "seed": seed,
            "arm": arm,
            "episode_id": ep.episode_id,
            "task_id": task.task_id,
            "role": role,
            "family": ep.family,
        }
        for seed in seeds
        for arm in ARMS
        for ep in episodes
        for role, tasks in (("S", [ep.source]), ("Q", ep.queries))
        for task in tasks
    ]
    receipt = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "planned_work": work,
        "expected_generations": len(expected),
        "actual_generations": 0,
        "completed_cases": [],
        "arms": list(ARMS),
        "seeds": seeds,
        "shared_active_adapter": "student",
        "separate_control_teacher_adapter": "teacher",
        "target_storage": (
            "Hashes plus pinned actor checkpoints/native sources; "
            "per-case full tensors not persisted"
        ),
        "selection": "All predetermined arms/seeds; evaluation Q never selects",
        "generation_batch_size": batch_size,
        "batch_semantics": "New matched protocol; not assumed equivalent to singleton decoding",
    }
    write_json(output / "receipt.json", receipt)
    source_root = Path(__file__).resolve().parents[1]
    code_files = [
        source_root / "src/hindsight" / name
        for name in ("shared.py", "model.py", "experiment.py", "data.py")
    ]
    code_files += [Path(__file__), Path(SOURCE_ADMISSION.__file__)]
    identity = {
        "config": config,
        "recipe": dataclasses.asdict(recipe),
        "code": {str(path.relative_to(source_root)): sha(path) for path in code_files},
        "training_examples": [
            {
                "episode_id": item.episode_id,
                "provenance": item.provenance,
                "source_digest": digest(dataclasses.asdict(item.generation)),
                "target_tensor_sha256": tensor_digest({"target": item.target}),
            }
            for item in examples
        ],
        "evaluation_episode_ids": [ep.episode_id for ep in episodes],
    }
    write_json(output / "identity.json", identity)
    write_json(output / "plan.json", {"generations": expected, "work": work})
    actual = []
    evidence = []

    def check():
        if datetime.now(timezone.utc) >= deadline or time.monotonic() - started >= maximum:
            raise TimeoutError("Shared learning absolute/invocation deadline reached")

    def event(kind, **fields):
        append_json(
            output / "events.jsonl",
            {"event": kind, "time": datetime.now(timezone.utc).isoformat(), **fields},
        )

    def persist_checkpoint(name, values):
        check()
        path = output / "checkpoints" / name
        path.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "format": "shared-adapter-v1",
                "parameters": values,
                "model_name": config["model_name"],
                "revision": config["revision"],
            },
            path,
        )
        return {
            "path": str(path.relative_to(output)),
            "sha256": sha(path),
            "parameter_sha256": tensor_digest(values, check),
        }

    def generate(ep, tasks, role, seed, arm):
        check()
        if role == "S" and len(tasks) != 1:
            raise ValueError("Each arm must draw its own singleton source")
        with audit_forward(engine, "student", evidence):
            values = engine.generate(
                [task.prompt for task in tasks],
                max_new_tokens=config["source_tokens"]
                if role == "S"
                else config["evaluation_tokens"],
                sample=role == "S",
                seed=seed_for(seed, ep.episode_id, "source") if role == "S" else seed,
                adapter="student",
            )
        generated = []
        for task, value in zip(tasks, values, strict=True):
            if role == "S":
                SOURCE_ADMISSION.validate_source_text(engine, value)
            record = {
                "seed": seed,
                "arm": arm,
                "episode_id": ep.episode_id,
                "task_id": task.task_id,
                "role": role,
                "family": ep.family,
                "sequence": len(actual),
                "generation_batch_size": len(tasks),
                "generation": dataclasses.asdict(value),
                "grade": grade(value.text, task),
            }
            append_json(output / "generations.jsonl", record)
            actual.append(record)
            generated.append((value, record["grade"]))
        receipt.update(actual_generations=len(actual), work=dict(engine.counters))
        write_json(output / "receipt.json", receipt)
        return generated

    try:
        check()
        configure_deterministic_execution(True)
        if engine is None:
            engine = HFEngine(
                model_name=config["model_name"],
                revision=config["revision"],
                device=config.get("device", "cuda"),
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"],
                enable_thinking=config.get("enable_thinking"),
                budget_check=check,
            )
        else:
            engine.budget_check = check
        if engine.counters != dict.fromkeys(work, 0):
            raise ValueError("Shared comparison requires a fresh unused engine")
        if engine.model_name != config["model_name"] or engine.revision != config["revision"]:
            raise ValueError("Engine model identity differs from protocol")
        for item in examples:
            SOURCE_ADMISSION.validate_source_text(engine, item.generation)
        if cohort is not None:
            admit_cohort(examples, cohort, config["evaluation_episode_ids"])
        base = {
            name: value for name, value in engine.model.named_parameters() if "lora_" not in name
        }
        receipt["base_before_sha256"] = tensor_digest(base, check)
        receipt["architecture"] = {
            "model_type": engine.model_type,
            "layer_types": list(engine.model.config.layer_types),
            "attention": (
                "Unrestricted full-attention layers; DeltaNet recurrence/convolution remain causal"
            ),
        }
        for seed in seeds:
            check()
            engine.reset_student(seed_for(seed, "shared_initialization"))
            initial = snapshot(engine)
            initial_checkpoint = persist_checkpoint(f"{seed}-initial.pt", initial)
            storage = storage_map(engine)
            write_json(output / f"storage-{seed}.json", storage)
            before_fit = {}
            for item in examples:
                before_fit[item.episode_id] = (
                    score(
                        engine,
                        item.generation,
                        item.feedback,
                        mode="bidirectional",
                        adapter="student",
                        evidence=evidence,
                    ),
                    score(
                        engine,
                        item.generation,
                        "",
                        mode="causal",
                        adapter="student",
                        evidence=evidence,
                    ),
                )
            event("shared_teacher_fit_started", seed=seed)
            fit = fit_shared(engine, examples, recipe, seed_for(seed, "teacher_fit"), evidence)
            fitted = snapshot(engine)
            fitted_checkpoint = persist_checkpoint(f"{seed}-fitted.pt", fitted)
            for item in examples:
                old_teacher, old_actor = before_fit.pop(item.episode_id)
                new_teacher = score(
                    engine,
                    item.generation,
                    item.feedback,
                    mode="bidirectional",
                    adapter="student",
                    evidence=evidence,
                )
                new_actor = score(
                    engine, item.generation, "", mode="causal", adapter="student", evidence=evidence
                )
                append_json(
                    output / "fit-interference.jsonl",
                    {
                        "seed": seed,
                        "episode_id": item.episode_id,
                        "teacher_before": distances(item.target, old_teacher),
                        "teacher_after": distances(item.target, new_teacher),
                        "causal_distribution_drift": distances(old_actor, new_actor),
                    },
                )
                del old_teacher, old_actor, new_teacher, new_actor
            event(
                "shared_teacher_fit_completed",
                seed=seed,
                fit=fit,
                initial_checkpoint=initial_checkpoint,
                fitted_checkpoint=fitted_checkpoint,
            )
            restore(engine, fitted, "teacher")
            if (
                tensor_digest(adapter_values(engine, "teacher"), check)
                != fitted_checkpoint["parameter_sha256"]
            ):
                raise RuntimeError("Separate control did not receive the identical fitted weights")
            for arm in ARMS:
                for ep in episodes:
                    check()
                    restore(engine, fitted if arm in ("shared", "teacher_only") else initial)
                    before_hash = tensor_digest(adapter_values(engine), check)
                    buffers_before = tensor_digest(dict(engine.model.named_buffers()), check)
                    generation, source_grade = generate(ep, [ep.source], "S", seed, arm)[0]
                    if before_hash != tensor_digest(adapter_values(engine), check):
                        raise RuntimeError("Source generation changed actor weights")
                    feedback = "The submitted answer was " + (
                        "correct." if source_grade["success"] else "incorrect."
                    )
                    case = {
                        "seed": seed,
                        "arm": arm,
                        "episode_id": ep.episode_id,
                        "source_digest": digest(dataclasses.asdict(generation)),
                        "source_feedback": feedback,
                        "actor_before_sha256": before_hash,
                        "source_generated_by_this_arm": True,
                    }
                    if arm in UPDATED_ARMS:
                        teacher_adapter = "teacher" if arm == "separate" else "student"
                        teacher_hash = tensor_digest(adapter_values(engine, teacher_adapter), check)
                        target = score(
                            engine,
                            generation,
                            feedback,
                            mode="bidirectional",
                            adapter=teacher_adapter,
                            evidence=evidence,
                        )
                        if before_hash != tensor_digest(adapter_values(engine), check):
                            raise RuntimeError(
                                "Source generation or target scoring changed actor weights"
                            )
                        update = adapt_shared(engine, generation, target, recipe, evidence)
                        actor_after = score(
                            engine,
                            generation,
                            "",
                            mode="causal",
                            adapter="student",
                            evidence=evidence,
                        )
                        teacher_after = score(
                            engine,
                            generation,
                            feedback,
                            mode="bidirectional",
                            adapter=teacher_adapter,
                            evidence=evidence,
                        )
                        case.update(
                            teacher_adapter=teacher_adapter,
                            teacher_snapshot_sha256=teacher_hash,
                            target_tensor_sha256=update["target_sha256"],
                            actor_after_sha256=update["after_sha256"],
                            actor_before_fidelity=distances(target, update["initial"]),
                            actor_after_fidelity=distances(target, actor_after),
                            causal_distribution_drift=distances(update["initial"], actor_after),
                            teacher_distribution_drift=distances(target, teacher_after),
                            student_metrics=update["metrics"],
                            alignment={
                                mode: alignment(
                                    engine,
                                    generation,
                                    feedback if mode == "bidirectional" else "",
                                    mode,
                                )
                                for mode in ("causal", "bidirectional")
                            },
                        )
                        if arm == "separate" and (
                            not torch.equal(target, teacher_after)
                            or teacher_hash
                            != tensor_digest(adapter_values(engine, "teacher"), check)
                        ):
                            raise RuntimeError("Separate actor update changed its frozen teacher")
                        del target, update, actor_after, teacher_after
                    expected_actor = case.get("actor_after_sha256", before_hash)
                    for start in range(0, len(ep.queries), batch_size):
                        generate(ep, ep.queries[start : start + batch_size], "Q", seed, arm)
                    actor_after_evaluation = tensor_digest(adapter_values(engine), check)
                    buffers_after = tensor_digest(dict(engine.model.named_buffers()), check)
                    if expected_actor != actor_after_evaluation or buffers_before != buffers_after:
                        raise RuntimeError("Evaluation changed actor weights or model buffers")
                    case.update(
                        actor_after_evaluation_sha256=actor_after_evaluation,
                        buffers_before_sha256=buffers_before,
                        buffers_after_sha256=buffers_after,
                    )
                    append_json(output / "cases.jsonl", case)
                    if storage_map(engine) != storage:
                        raise RuntimeError(
                            "Shared adapter storage changed between its acting/scoring phases"
                        )
                    receipt["completed_cases"].append(
                        {"seed": seed, "arm": arm, "episode_id": ep.episode_id}
                    )
                    write_json(output / "receipt.json", receipt)
            del initial, fitted
        receipt["base_after_sha256"] = tensor_digest(base, check)
        if receipt["base_before_sha256"] != receipt["base_after_sha256"]:
            raise RuntimeError("Frozen base parameters changed")
        if engine.counters != work or len(actual) != len(expected):
            raise RuntimeError("Actual work differs from declared shared-comparison work")
        if [{key: row[key] for key in expected[0]} for row in actual] != expected:
            raise RuntimeError("Generated outcome identities/order differ from frozen plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            actual_generations=len(actual),
            missing_generations=len(expected) - len(actual),
            work=dict(engine.counters) if engine is not None else dict.fromkeys(work, 0),
            forward_audits=len(evidence),
            audited_forward_calls=sum(row["forward_calls"] for row in evidence),
            same_storage_audits_passed=all(row["same_parameter_storage"] for row in evidence),
        )
        write_json(output / "forward-audits.json", evidence)
        for path in code_files:
            if sha(path) != identity["code"][str(path.relative_to(source_root))]:
                receipt.update(status="failed", code_identity_changed=True)
        if (
            qualification_path is not None
            and sha(qualification_path) != config["batch_qualification"]["sha256"]
        ):
            receipt.update(status="failed", batch_qualification_changed=True)
        write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "teacher-manifest", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    recipe = SharedRecipe(**config["recipe"])
    examples, teaching = load_teaching(args.teacher_manifest, recipe)
    from hindsight.cohorts import load_cohort

    episodes, _ = load_cohort(args.data)
    ids = config["evaluation_episode_ids"]
    if len(config["seeds"]) != 2:
        raise ValueError("Initial E7 cut requires exactly two seeds")
    selected = admit_cohort(examples, episodes, ids, check_feedback=False)
    if (
        teaching["model_name"] != config["model_name"]
        or teaching["revision"] != config["revision"]
        or sha(args.data / "manifest.json") != config["dataset_manifest_sha256"]
    ):
        raise ValueError("Teaching model or evaluation corpus identity differs")
    config["teacher_manifest_sha256"] = sha(args.teacher_manifest)
    result = run(config, examples, selected, args.output, cohort=episodes)
    print(json.dumps({"status": result["status"], "work": result["work"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
