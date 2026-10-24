"""E5/E6: matched teacher fits, frozen sources, and seven explicit fresh-Q arms."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import grade
from hindsight.experiment import append_json, digest, read_json, seed_for, write_json
from hindsight.model import HFEngine, configure_deterministic_execution
from hindsight.shared import (
    adapt_shared,
    adapter_values,
    alignment,
    audit_forward,
    restore,
    score,
    snapshot,
    storage_map,
    tensor_digest,
)
from hindsight.teacher_controls import (
    ARMS,
    TeacherRecipe,
    distribution_metrics,
    fit_teacher_mode,
    planned_work,
    replay_target,
)

# Reuse the already-tested target admission contract without moving frozen E7 code.
SPEC = importlib.util.spec_from_file_location(
    "shared_target_admission", Path(__file__).with_name("check_shared_learning.py")
)
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)
sha = ADMISSION.sha


def admit(config, training, evaluation, episodes, *, check_feedback=True):
    """Admit cohort structure; CLI defers grading until tokenizer admission."""
    admitted = {ep.episode_id: ep for ep in episodes}
    train_ids = [item.episode_id for item in training]
    eval_ids = [item.episode_id for item in evaluation]
    partition = config["evaluation_partition"]
    if (
        partition not in ("development", "confirmation")
        or train_ids != config["training_episode_ids"]
        or eval_ids != config["evaluation_episode_ids"]
        or len(set(train_ids)) != len(train_ids)
        or len(set(eval_ids)) != len(eval_ids)
        or set(train_ids) & set(eval_ids)
        or not (set(train_ids) | set(eval_ids)) <= admitted.keys()
    ):
        raise ValueError("Require exact disjoint declared training/evaluation source identities")
    for values, expected_partition in ((training, "meta_train"), (evaluation, partition)):
        for item in values:
            ep = admitted[item.episode_id]
            indices = item.provenance["selection_validation_indices"]
            if (
                ep.partition != expected_partition
                or item.generation.prompt != ep.source.prompt
                or item.provenance["source_task_id"] != ep.source.task_id
                or len(ep.validation) != 8
                or set(item.provenance["validation_task_ids"])
                != {ep.validation[index].task_id for index in indices}
            ):
                raise ValueError("Source/V/feedback provenance differs from admitted cohort")
            if check_feedback:
                feedback = "The submitted answer was " + (
                    "correct."
                    if grade(item.generation.text, ep.source)["success"]
                    else "incorrect."
                )
                if item.feedback != feedback:
                    raise ValueError("Source/V/feedback provenance differs from admitted cohort")
    selected = [admitted[ep_id] for ep_id in eval_ids]
    sizes = Counter(ep.family for ep in selected)
    if len(sizes) != 4 or len(set(sizes.values())) != 1:
        raise ValueError("Evaluation must retain all four equally represented families")
    return selected


def run(config, training, evaluation, episodes, output: Path, *, engine=None, cohort=None):
    """No generation reuse: all seven arms answer every declared Q under one schedule."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "arms": list(ARMS),
        "sources": "Reused exact native frozen causal source; no new source/proposal draws",
        "neutral_pi": (
            "Execute declared KL updates; a zero target loss may leave weights unchanged"
        ),
        "selection": "All arms/seeds reported; Q never selects recipe, teacher or checkpoints",
        "target_lineage": (
            "Manifest-bound bytes, source/V identity; no independent target rescoring"
        ),
        "teacher_comparison": (
            "Matched targets/weights/work and completed response/feedback information; "
            "raw-causal serialization differs in token positions/context length; "
            "not a pure mask ablation"
        ),
        "completed_cases": [],
        "actual_generations": 0,
    }
    write_json(output / "receipt.json", receipt)
    work, evidence, outcomes, files = {}, [], [], {}
    try:
        recipe = TeacherRecipe(**config["recipe"])
        seeds = config["seeds"]
        limit = config["max_invocation_seconds"]
        deadline = datetime.fromisoformat(config["absolute_deadline_utc"])
        size = config.get("generation_batch_size", 1)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int for seed in seeds)
            or not math.isfinite(limit)
            or limit <= 0
            or deadline.tzinfo is None
            or type(size) is not int
            or size not in (1, 4, 8)
            or [item.episode_id for item in evaluation] != [ep.episode_id for ep in episodes]
            or len({ep.episode_id for ep in episodes}) != len(episodes)
            or {item.episode_id for item in training} & {ep.episode_id for ep in episodes}
        ):
            raise ValueError("Invalid bounded seed/source/batch protocol")
        if any(
            item.provenance.get("selection_recipe") != recipe.selection_recipe()
            or item.provenance.get("derived_from") != "post_update_V"
            for item in training + evaluation
        ):
            raise ValueError("All cached targets must use actual V under the selected recipe")
        if {item.generation.prompt for item in training} & {
            task.prompt for ep in episodes for task in (ep.source, *ep.validation, *ep.queries)
        }:
            raise ValueError("Teacher training/evaluation prompts overlap")

        def check():
            if datetime.now(timezone.utc) >= deadline or time.monotonic() - started >= limit:
                raise TimeoutError("Teacher-control absolute/invocation deadline reached")

        def event(kind, **values):
            append_json(output / "events.jsonl", {"event": kind, **values})

        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "src/hindsight" / name
            for name in (
                "teacher_controls.py",
                "shared.py",
                "model.py",
                "experiment.py",
                "data.py",
                "cohorts.py",
            )
        ] + [
            Path(__file__),
            Path(__file__).with_name("check_shared_learning.py"),
            Path(ADMISSION.SOURCE_ADMISSION.__file__),
        ]
        if size != 1:
            declared = config["batch_qualification"]
            path = Path(declared["path"])
            qualified = read_json(path)
            if (
                sha(path) != declared["sha256"]
                or qualified["status"] != "completed"
                or qualified["selected_batch_size"] != size
                or not qualified["repeat_stability"].get(str(size), False)
                or qualified["config"]["model_name"] != config["model_name"]
                or qualified["config"]["revision"] != config["revision"]
                or qualified["dataset_sha256"] != config["dataset_manifest_sha256"]
            ):
                raise ValueError("Batch qualification does not admit this exact schedule")
            paths.append(path)
        files = {str(path.resolve()): sha(path) for path in paths}
        for path, expected_hash in config.get("frozen_input_files", {}).items():
            if sha(path) != expected_hash:
                raise ValueError("Declared source/target/corpus/config artifact changed")
            files[str(Path(path).resolve())] = expected_hash
        work = planned_work(
            len(training),
            len(episodes),
            sum(len(ep.queries) for ep in episodes),
            len(seeds),
            recipe,
        )
        expected = [
            {
                "seed": seed,
                "condition": arm,
                "episode_id": ep.episode_id,
                "task_id": task.task_id,
                "role": "Q",
            }
            for seed in seeds
            for ep in episodes
            for arm in ARMS
            for task in ep.queries
        ]
        write_json(
            output / "plan.json", {"work": work, "generations": expected, "frozen_files": files}
        )
        identity = {
            "config_sha256": digest(config),
            "training": [
                {
                    "episode_id": item.episode_id,
                    "provenance": item.provenance,
                    "source_digest": digest(dataclasses.asdict(item.generation)),
                    "target_tensor_sha256": tensor_digest({"target": item.target}, check),
                }
                for item in training
            ],
            "evaluation": [
                {
                    "episode_id": item.episode_id,
                    "provenance": item.provenance,
                    "source_digest": digest(dataclasses.asdict(item.generation)),
                    "target_tensor_sha256": tensor_digest({"target": item.target}, check),
                }
                for item in evaluation
            ],
        }
        write_json(output / "identity.json", identity)
        for item in evaluation:
            write_json(
                output / "sources" / f"{item.episode_id}.json",
                {
                    "generation": dataclasses.asdict(item.generation),
                    "feedback": item.feedback,
                    "provenance": item.provenance,
                    "new_generation": False,
                },
            )
        receipt.update(planned_work=work, expected_generations=len(expected))
        write_json(output / "receipt.json", receipt)
        check()
        configure_deterministic_execution(True)
        if engine is None:
            engine = HFEngine(
                model_name=config["model_name"],
                revision=config["revision"],
                device=config.get("device", "cuda"),
                enable_thinking=config.get("enable_thinking"),
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"],
                budget_check=check,
            )
        else:
            engine.budget_check = check
        if engine.counters != dict.fromkeys(work, 0) or (engine.model_name, engine.revision) != (
            config["model_name"],
            config["revision"],
        ):
            raise ValueError("Require a fresh engine matching the pinned model identity")
        for item in training + evaluation:
            ADMISSION.SOURCE_ADMISSION.validate_source_text(engine, item.generation)
        if cohort is not None:
            admit(config, training, evaluation, cohort)
        for item in training + evaluation:
            engine._validate_target(item.generation, item.target)
            alignment(engine, item.generation, item.feedback, "bidirectional")
        base = {
            name: value for name, value in engine.model.named_parameters() if "lora_" not in name
        }
        receipt["base_before_sha256"] = tensor_digest(base, check)
        receipt["architecture"] = {
            "model_type": engine.model_type,
            "layer_types": list(engine.model.config.layer_types),
            "claim": "Future-aware hybrid; DeltaNet remains causal",
        }

        def checkpoint(name, values):
            path = output / "checkpoints" / name
            path.parent.mkdir(exist_ok=True)
            torch.save(
                {
                    "parameters": values,
                    "model_name": engine.model_name,
                    "revision": engine.revision,
                },
                path,
            )
            return {
                "path": str(path.relative_to(output)),
                "sha256": sha(path),
                "parameter_sha256": tensor_digest(values, check),
            }

        for seed in seeds:
            engine.reset_student(seed_for(seed, "student_initialization"))
            initial = snapshot(engine)
            initial_sha = tensor_digest(initial, check)
            actor_storage = storage_map(engine)
            write_json(output / f"storage-{seed}.json", actor_storage)
            receipt.setdefault("checkpoints", {})[str(seed)] = {
                "initial": checkpoint(f"{seed}-initial.pt", initial)
            }
            fitted = {}
            fit_receipts = []
            for mode in ("bidirectional", "raw_causal"):
                restore(engine, initial, "teacher")
                before_fit = {}
                for item in training:
                    q = score(
                        engine,
                        item.generation,
                        item.feedback,
                        mode=mode,
                        adapter="teacher",
                        evidence=evidence,
                    )
                    before_fit[item.episode_id] = distribution_metrics(
                        item.target, q, item.generation.response_ids
                    )
                fit = fit_teacher_mode(
                    engine,
                    training,
                    recipe,
                    initial,
                    mode,
                    seed_for(seed, "matched_teacher_fit"),
                    evidence,
                    on_step=lambda row, s=seed: append_json(
                        output / "teacher-steps.jsonl", {"seed": s, **row}
                    ),
                )
                fitted[mode] = snapshot(engine, "teacher")
                saved = checkpoint(f"{seed}-{mode}.pt", fitted[mode])
                receipt["checkpoints"][str(seed)][mode] = saved
                for item in training:
                    q = score(
                        engine,
                        item.generation,
                        item.feedback,
                        mode=mode,
                        adapter="teacher",
                        evidence=evidence,
                    )
                    append_json(
                        output / "fit-fidelity.jsonl",
                        {
                            "seed": seed,
                            "mode": mode,
                            "episode_id": item.episode_id,
                            "before": before_fit[item.episode_id],
                            "after": distribution_metrics(
                                item.target, q, item.generation.response_ids
                            ),
                        },
                    )
                fit_receipts.append(fit)
                event("teacher_fit_completed", seed=seed, fit=fit, checkpoint=saved)
            if fit_receipts[0]["before_sha256"] != fit_receipts[1]["before_sha256"] or [
                row["source_ids"] for row in fit_receipts[0]["metrics"]
            ] != [row["source_ids"] for row in fit_receipts[1]["metrics"]]:
                raise RuntimeError("Matched teacher initial states or source schedules diverged")
            for item, ep in zip(evaluation, episodes, strict=True):
                check()
                restore(engine, initial)
                buffer_before = tensor_digest(dict(engine.model.named_buffers()), check)
                targets = {
                    "adapted_pi": item.target.detach().cpu(),
                    "adapted_replay": replay_target(item.generation, item.target.shape[-1]),
                }
                for arm, mode, weights in (
                    ("adapted_fitted_bidirectional", "bidirectional", fitted["bidirectional"]),
                    ("adapted_unfitted_bidirectional", "bidirectional", initial),
                    ("adapted_fitted_raw_causal", "raw_causal", fitted["raw_causal"]),
                    ("adapted_raw_causal", "raw_causal", initial),
                ):
                    restore(engine, weights, "teacher")
                    targets[arm] = score(
                        engine,
                        item.generation,
                        item.feedback,
                        mode=mode,
                        adapter="teacher",
                        evidence=evidence,
                    )
                if initial_sha != tensor_digest(adapter_values(engine), check):
                    raise RuntimeError("Target scoring changed frozen actor initialization")
                teacher_sha = tensor_digest(adapter_values(engine, "teacher"), check)
                for arm in ARMS:
                    restore(engine, initial)
                    case = {
                        "seed": seed,
                        "episode_id": ep.episode_id,
                        "condition": arm,
                        "source_digest": digest(dataclasses.asdict(item.generation)),
                        "actor_before_sha256": initial_sha,
                        "selected_candidate": item.provenance.get("selected_candidate"),
                    }
                    expected_actor = initial_sha
                    if arm != "baseline":
                        target = targets[arm]
                        update = adapt_shared(engine, item.generation, target, recipe, evidence)
                        post = score(
                            engine,
                            item.generation,
                            "",
                            mode="causal",
                            adapter="student",
                            evidence=evidence,
                        )
                        expected_actor = update["after_sha256"]
                        case.update(
                            target_tensor_sha256=update["target_sha256"],
                            target_vs_selected_PI=distribution_metrics(
                                item.target, target, item.generation.response_ids
                            ),
                            actor_after_vs_selected_PI=distribution_metrics(
                                item.target, post, item.generation.response_ids
                            ),
                            actor_before_fidelity=distribution_metrics(target, update["initial"]),
                            actor_after_fidelity=distribution_metrics(target, post),
                            actor_after_sha256=expected_actor,
                            actor_parameters_changed=expected_actor != initial_sha,
                            metrics=update["metrics"],
                        )
                        del update, post
                    for start in range(0, len(ep.queries), size):
                        check()
                        batch = ep.queries[start : start + size]
                        with audit_forward(engine, "student", evidence):
                            generated = engine.generate(
                                [task.prompt for task in batch],
                                max_new_tokens=config["evaluation_tokens"],
                                sample=False,
                                seed=seed,
                                adapter="student",
                            )
                        for task, value in zip(batch, generated, strict=True):
                            row = {
                                "seed": seed,
                                "episode_id": ep.episode_id,
                                "condition": arm,
                                "task_id": task.task_id,
                                "role": "Q",
                                "family": ep.family,
                                "sequence": len(outcomes),
                                "generation": dataclasses.asdict(value),
                                "generation_batch_size": len(batch),
                                **grade(value.text, task),
                            }
                            append_json(output / "observations.jsonl", row)
                            outcomes.append(row)
                    actor_after = tensor_digest(adapter_values(engine), check)
                    buffers_after = tensor_digest(dict(engine.model.named_buffers()), check)
                    if (
                        actor_after != expected_actor
                        or buffers_after != buffer_before
                        or teacher_sha != tensor_digest(adapter_values(engine, "teacher"), check)
                        or storage_map(engine) != actor_storage
                    ):
                        raise RuntimeError(
                            "Actor/teacher/buffer/storage invariant failed after evaluation"
                        )
                    case.update(
                        actor_after_evaluation_sha256=actor_after,
                        buffers_before_sha256=buffer_before,
                        buffers_after_sha256=buffers_after,
                    )
                    append_json(output / "cases.jsonl", case)
                    receipt["completed_cases"].append(
                        {"seed": seed, "episode_id": ep.episode_id, "condition": arm}
                    )
                    receipt.update(actual_generations=len(outcomes), work=dict(engine.counters))
                    write_json(output / "receipt.json", receipt)
                del targets
        receipt["base_after_sha256"] = tensor_digest(base, check)
        if receipt["base_before_sha256"] != receipt["base_after_sha256"]:
            raise RuntimeError("Frozen base parameters changed")
        if (
            engine.counters != work
            or [{key: row[key] for key in expected[0]} for row in outcomes] != expected
        ):
            raise RuntimeError("Observed work/outcome identities differ from frozen plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        unchanged = all(
            Path(path).exists() and sha(path) == expected for path, expected in files.items()
        )
        if not unchanged:
            receipt["status"] = "failed"
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            actual_generations=len(outcomes),
            frozen_files_unchanged=unchanged,
            work=dict(engine.counters) if engine is not None else dict.fromkeys(work, 0),
            same_storage_audits_passed=all(row["same_parameter_storage"] for row in evidence),
        )
        write_json(output / "forward-audits.json", evidence)
        write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "training-manifest", "evaluation-manifest", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    recipe = TeacherRecipe(**config["recipe"])
    training, train_manifest = ADMISSION.load_teaching(args.training_manifest, recipe)
    evaluation, eval_manifest = ADMISSION.load_teaching(args.evaluation_manifest, recipe)
    from hindsight.cohorts import load_cohort

    episodes, _ = load_cohort(args.data)
    if (
        any(
            (manifest["model_name"], manifest["revision"])
            != (config["model_name"], config["revision"])
            for manifest in (train_manifest, eval_manifest)
        )
        or sha(args.data / "manifest.json") != config["dataset_manifest_sha256"]
    ):
        raise ValueError("Manifest model/corpus identity differs from frozen protocol")
    selected = admit(config, training, evaluation, episodes, check_feedback=False)
    config.update(
        training_manifest_sha256=sha(args.training_manifest),
        evaluation_manifest_sha256=sha(args.evaluation_manifest),
    )
    inputs = [args.config, args.training_manifest, args.evaluation_manifest]
    inputs += list(args.data.rglob("*.json")) + list(args.data.rglob("*.jsonl"))
    for path, manifest in (
        (args.training_manifest, train_manifest),
        (args.evaluation_manifest, eval_manifest),
    ):
        inputs.append(path.parent / manifest["selection_observations_path"])
        for entry in manifest["examples"]:
            inputs.extend(path.parent / entry[key] for key in ("source_path", "target_path"))
    config["frozen_input_files"] = {str(path.resolve()): sha(path) for path in inputs}
    result = run(config, training, evaluation, selected, args.output, cohort=episodes)
    print(json.dumps({"status": result["status"], "work": result["work"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
