"""One frozen development fit and fresh-source hindsight transfer comparison."""

import argparse
import hashlib
import importlib.util
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import load_corpus
from hindsight.experiment import (
    EXTRAS,
    Experiment,
    choose_candidate,
    digest,
    observation_key,
    read_json,
    rows,
    seed_for,
    teacher_fidelity,
    write_json,
)
from hindsight.model import Generation

CORE_WORK = dict(generation_sequences=512, scoring_forwards=56,
                 student_optimizer_steps=160, teacher_optimizer_steps=64,
                 teacher_example_exposures=256)
QUALIFICATION_WORK = dict(generation_sequences=0, scoring_forwards=5,
                          student_optimizer_steps=0, teacher_optimizer_steps=2,
                          teacher_example_exposures=2)
PLANNED_WORK = {key: count + QUALIFICATION_WORK[key] for key, count in CORE_WORK.items()}
MATCHED_CONFIG = (
    "model_name", "revision", "enable_thinking", "deterministic_execution", "seed",
    "source_tokens", "evaluation_tokens", "student_steps", "learning_rate",
    "max_context_tokens", "max_teacher_context_tokens", "absolute_deadline_utc",
)
FAMILIES = ("translation", "denoising", "size_recoloring", "gravity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_inputs(data):
    signature = script_module("build_balanced_development").input_signature
    return {signature(grid) for row in rows(data / "tasks.public.jsonl")
            for grid in [row["query_input"], *(pair["input"] for pair in row["demonstrations"])]}


def admit(sources, lesson_file, data):
    parent = read_json(sources / "identity.json")
    manifest = read_json(data / "manifest.json")
    episodes, calibration = load_corpus(data)
    supplied = read_json(lesson_file)
    ids = [ep.episode_id for ep in episodes]
    if (parent["dataset_manifest"] != manifest
            or parent["config"]["dataset_manifest_sha256"] != sha(data / "manifest.json")
            or manifest.get("protocol_version") != "balanced-development-v1"
            or len(episodes) != 8 or calibration
            or Counter(ep.family for ep in episodes) != dict.fromkeys(FAMILIES, 2)
            or any(ep.partition != "development" or len(ep.validation) != 8
                   or len(ep.queries) != 8 for ep in episodes)
            or parent["config"]["diagnostic_source_ids"] != ids
            or not isinstance(supplied.get("provenance"), dict)
            or [row["episode_id"] for row in supplied["rows"]] != ids):
        raise ValueError("Require all eight pinned balanced development sources and lessons")
    prepared = []
    for ep, lesson in zip(episodes, supplied["rows"]):
        path = sources / "sources" / f"{ep.episode_id}.json"
        source = read_json(path)
        generation = Generation(**source["generation"])
        if (sha(path) != lesson["source_sha256"] or source["task_id"] != ep.source.task_id
                or generation.prompt != ep.source.prompt
                or not isinstance(source["feedback"], str)
                or not isinstance(lesson["lesson"], str) or not lesson["lesson"].strip()
                or any(not isinstance(tokens, list) or not tokens
                       or any(type(token) is not int or token < 0 for token in tokens)
                       for tokens in (generation.prompt_ids, generation.response_ids))):
            raise ValueError("Invalid native source, lesson, or immutable source join")
        prepared.append((ep, generation, source["feedback"], lesson["lesson"]))
    return parent, prepared


def fit_selections(fit_run, prepared):
    receipt = read_json(fit_run / "receipt.json")
    observed, expected = rows(fit_run / "observations.jsonl"), rows(fit_run / "expected.jsonl")
    if (receipt["status"] != "completed" or receipt["actual_observations"] != 384
            or len(observed) != 384 or len(expected) != 384
            or len({observation_key(row) for row in observed}) != 384
            or {observation_key(row) for row in observed}
            != {observation_key(row) for row in expected}
            or receipt["lessons_sha256"] != sha(fit_run / "lessons.json")):
        raise ValueError("Fitting requires the complete immutable 021 comparison")
    selected = {}
    for ep, *_ in prepared:
        groups = []
        for condition, candidate in (("baseline", None), ("adapted_pi", 0)):
            group = [row for row in observed if row["episode_id"] == ep.episode_id
                     and row["role"] == "V" and row["condition"] == condition
                     and row["candidate"] == candidate]
            if (len(group) != 8 or {row["task_id"] for row in group}
                    != {task.task_id for task in ep.validation}
                    or any(type(row["success"]) is not bool or row["partition"] != "development"
                           or row["family"] != ep.family for row in group)):
                raise ValueError("Incomplete or mismatched fit validation joins")
            groups.append(group)
        winner = choose_candidate(groups[0], [groups[1]], seed=0)
        selected[ep.episode_id] = dict(selected_candidate=winner, derived_from="post_update_V",
                                     baseline_successes=sum(row["success"] for row in groups[0]),
                                     adapted_successes=sum(row["success"] for row in groups[1]))
    if sum(value["selected_candidate"] == 0 for value in selected.values()) != 6:
        raise ValueError("Frozen 021 selection must contain six PI and two neutral targets")
    return selected


def save_target(path, target, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target.detach().cpu(), path)
    write_json(path.with_suffix(".json"), {**provenance, "tensor_sha256": sha(path)})


def run(fit_run, fit_data, sources, lessons, data, output, *, engine=None, qualifier=None):
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="started", started_at=datetime.now(timezone.utc).isoformat(),
                   expected_observations=512, planned_work=PLANNED_WORK,
                   core_work=CORE_WORK, qualification_work=QUALIFICATION_WORK,
                   completed_episode_ids=[], script_sha256=sha(__file__))
    write_json(output / "receipt.json", receipt)
    experiment = None
    try:
        fit_parent, fitting = admit(fit_run, fit_run / "lessons.json", fit_data)
        parent, evaluating = admit(sources, lessons, data)
        source_receipt = read_json(sources / "receipt.json")
        if source_receipt["status"] != "completed" or source_receipt["actual_sources"] != 8:
            raise ValueError("Fresh source generation must be completed before comparison")
        selected = fit_selections(fit_run, fitting)
        config = dict(parent["config"])
        if (any(config[key] != fit_parent["config"][key] for key in MATCHED_CONFIG)
                or config["student_steps"] != 4 or config["learning_rate"] != 1e-4
                or fit_parent["config"]["generation_batch_size"] != 1
                or set(ep.episode_id for ep, *_ in fitting)
                & set(ep.episode_id for ep, *_ in evaluating)
                or canonical_inputs(fit_data) & canonical_inputs(data)):
            raise ValueError("Fit/evaluation identity, policy, or input separation violated")
        core = Path(__file__).resolve().parents[1] / "src/hindsight"
        if any(parent["code"][name] != sha(core / name)
               or fit_parent["code"][name] != sha(core / name)
               for name in ("model.py", "experiment.py")):
            raise ValueError("Core model/experiment code changed from source and fit evidence")
        config.update(generation_batch_size=1, teacher_epochs=32, teacher_batch_size=4,
                      max_invocation_seconds=min(config["max_invocation_seconds"], 3900))
        provenance = dict(fit_identity_sha256=sha(fit_run / "identity.json"),
                          fit_observations_sha256=sha(fit_run / "observations.jsonl"),
                          fit_lessons_sha256=sha(fit_run / "lessons.json"),
                          source_identity_sha256=sha(sources / "identity.json"),
                          lessons_sha256=sha(lessons), fit_selections=selected,
                          fit_episode_ids=[ep.episode_id for ep, *_ in fitting],
                          evaluation_episode_ids=[ep.episode_id for ep, *_ in evaluating],
                          qualification_script_sha256=sha(
                              Path(__file__).with_name("qualify_hindsight_teacher.py")))
        config["hindsight_comparison"] = provenance
        receipt.update(provenance)
        experiment = Experiment(config, data, output, engine=engine)
        for directory, prepared, location in (("fit-sources", fitting, fit_run),
                                              ("sources", evaluating, sources)):
            for ep, *_ in prepared:
                path = output / directory / f"{ep.episode_id}.json"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes((location / "sources" / path.name).read_bytes())
        (output / "lessons.json").write_bytes(lessons.read_bytes())
        (output / "fit-lessons.json").write_bytes((fit_run / "lessons.json").read_bytes())
        for ep, *_ in evaluating:
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                conditions = [("baseline", None), ("adapted_pi", 0)]
                if role == "Q":
                    conditions += [(condition, None) for condition in EXTRAS]
                for condition, candidate in conditions:
                    for task in tasks:
                        experiment.expect(experiment.metadata(ep, task, role, condition, candidate))
        if len(experiment.expected) != 512:
            raise ValueError("Expected exactly 512 distinct evaluation observations")
        write_json(output / "receipt.json", receipt)
        targets = []
        for ep, generation, feedback, lesson in fitting:
            selection = selected[ep.episode_id]
            target = experiment.score(generation, feedback,
                                      pi=lesson if selection["selected_candidate"] == 0 else None)
            targets.append((generation, feedback, target))
            target_provenance = {**selection, "episode_id": ep.episode_id,
                                 "source_sha256": sha(
                                     fit_run / "sources" / f"{ep.episode_id}.json"),
                                 "lesson_sha256": hashlib.sha256(lesson.encode()).hexdigest()}
            save_target(output / "targets" / f"{ep.episode_id}.pt", target,
                        target_provenance)
            untrained = experiment.score(generation, feedback, mode="bidirectional")
            write_json(output / "fit-fidelity-before" / f"{ep.episode_id}.json",
                       dict(provenance=target_provenance, metrics=teacher_fidelity(
                           target, untrained, generation.response_ids)))
        active = experiment.get_engine()
        qualifier = qualifier or script_module("qualify_hindsight_teacher").qualify_teacher
        qualified = qualifier(active, *targets[0], output / "qualification")
        if qualified["status"] != "passed_teacher_qualification":
            raise ValueError("Teacher qualification did not pass")
        metrics = active.fit_teacher(targets, epochs=32, batch_size=4, lr=1e-4,
                                     seed=seed_for(config["seed"], "teacher_fit"))
        if metrics["optimizer_steps"] != 64 or metrics["example_exposures"] != 256:
            raise ValueError("Teacher fitting diverged from the fixed 64-step schedule")
        checkpoint = output / "teacher/checkpoint.pt"
        active.save_teacher(checkpoint)
        write_json(checkpoint.parent / "FIT-COMPLETE.json",
                   dict(identity_digest=digest(experiment.identity), metrics=metrics,
                        checkpoint_files={checkpoint.name: sha(checkpoint)}))
        experiment.event("teacher_fit_completed", metrics=metrics)
        receipt["teacher_checkpoint_sha256"] = experiment.validate_teacher()
        for (ep, *_), (generation, feedback, target) in zip(fitting, targets):
            fitted = experiment.score(generation, feedback, mode="bidirectional", adapter="teacher")
            write_json(output / "fit-fidelity-after" / f"{ep.episode_id}.json",
                       dict(episode_id=ep.episode_id, selection=selected[ep.episode_id],
                            target_sha256=sha(output / "targets" / f"{ep.episode_id}.pt"),
                            teacher_checkpoint_sha256=receipt["teacher_checkpoint_sha256"],
                            metrics=teacher_fidelity(target, fitted, generation.response_ids)))
        del targets
        for ep, generation, feedback, lesson in evaluating:
            baseline = experiment.evaluate(ep, ep.validation, "V", "baseline")
            experiment.evaluate(ep, ep.queries, "Q", "baseline")
            pi_target = experiment.score(generation, feedback, pi=lesson)
            initial = []
            experiment.adaptation(ep, generation, pi_target, "adapted_pi", 0,
                                  initial_logprobs_callback=initial.append)
            adapted = experiment.evaluate(ep, ep.validation, "V", "adapted_pi", 0,
                                          adapter="student")
            experiment.evaluate(ep, ep.queries, "Q", "adapted_pi", 0, adapter="student")
            winner = experiment.selection(ep, baseline, [adapted], "adapted_pi")
            if len(initial) != 1:
                raise ValueError("Missing matched initial causal distribution")
            reference = pi_target if winner == 0 else initial[0]
            lineage = experiment.reference_identity(ep, generation, feedback, winner)
            save_target(output / "fidelity-references" / f"{ep.episode_id}.pt", reference, lineage)
            for condition in EXTRAS:
                if condition == "adapted_replay":
                    target = torch.full_like(pi_target, -torch.inf)
                    target.scatter_(1, torch.tensor(generation.response_ids).unsqueeze(1), 0.0)
                else:
                    mode = "raw_causal" if condition == "adapted_raw_causal" else "bidirectional"
                    adapter = "teacher" if condition == "adapted_bidirectional" else None
                    target = experiment.score(generation, feedback, mode=mode, adapter=adapter)
                    if mode == "bidirectional":
                        path = output / "evaluation-fidelity" / condition / f"{ep.episode_id}.json"
                        checkpoint_hash = receipt["teacher_checkpoint_sha256"] if adapter else None
                        write_json(path,
                                   dict(provenance=lineage,
                                        teacher_checkpoint_sha256=checkpoint_hash,
                                        selected_reference=teacher_fidelity(
                                       reference, target, generation.response_ids),
                                       pi_reference=teacher_fidelity(
                                           pi_target, target, generation.response_ids)))
                experiment.adaptation(ep, generation, target, condition)
                experiment.evaluate(ep, ep.queries, "Q", condition, adapter="student")
            receipt["completed_episode_ids"].append(ep.episode_id)
            write_json(output / "receipt.json", receipt)
        if (experiment.expected != set(experiment.observations)
                or active.counters != PLANNED_WORK):
            raise ValueError("Actual observations or model work differ from the frozen plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        active = experiment.engine if experiment is not None else engine
        actual = len(experiment.observations) if experiment is not None else 0
        receipt.update(completed_at=datetime.now(timezone.utc).isoformat(),
                       actual_observations=actual, missing_observations=512 - actual,
                       work=dict(active.counters) if active is not None
                       else dict.fromkeys(PLANNED_WORK, 0))
        write_json(output / "receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit-run", "fit-data", "sources", "lessons", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return run(args.fit_run, args.fit_data, args.sources, args.lessons, args.data, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
