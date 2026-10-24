"""Bounded actual-post-update-V proposer RL and a matched frozen-initialization evaluation."""

import argparse
import dataclasses
import hashlib
import importlib.util
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from hindsight.data import grade
from hindsight.experiment import (
    append_json,
    digest,
    proposal_prompt,
    read_json,
    rows,
    seed_for,
    write_json,
)
from hindsight.model import Generation, configure_deterministic_execution
from hindsight.proposer import PROPOSER_COUNTERS, ProposerEngine, usable_lesson

SOURCE_SPEC = importlib.util.spec_from_file_location(
    "proposer_source_admission", Path(__file__).with_name("source_admission.py")
)
SOURCE_ADMISSION = importlib.util.module_from_spec(SOURCE_SPEC)
SOURCE_SPEC.loader.exec_module(SOURCE_ADMISSION)

CORE_COUNTERS = (
    "generation_sequences",
    "scoring_forwards",
    "student_optimizer_steps",
    "teacher_optimizer_steps",
    "teacher_example_exposures",
)
REWARD_INDICES = (0, 2, 4, 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(ep, task, role, condition, candidate=None):
    return dict(
        episode_id=ep.episode_id,
        partition=ep.partition,
        family=ep.family,
        task_id=task.task_id,
        role=role,
        condition=condition,
        candidate=candidate,
    )


def evaluate_tasks(
    engine, ep, tasks, role, condition, config, output, *, candidate=None, adapter=None
):
    """Every new answer is task-only greedy and saved with its native IDs."""
    results = []
    size = config.get("generation_batch_size", 1)
    for offset in range(0, len(tasks), size):
        batch = tasks[offset : offset + size]
        values = engine.generate(
            [task.prompt for task in batch],
            max_new_tokens=config["evaluation_tokens"],
            sample=False,
            seed=config["seed"],
            adapter=adapter,
        )
        for task, value in zip(batch, values, strict=True):
            if value.prompt != task.prompt:
                raise ValueError("Evaluation generation prompt differs from the task")
            result = dict(
                **metadata(ep, task, role, condition, candidate),
                **grade(value.text, task),
                generation=dataclasses.asdict(value),
                generation_batch_size=len(batch),
                reused_baseline=False,
            )
            append_json(output / "observations.jsonl", result)
            results.append(result)
    return results


def source_for(engine, ep, config, output, sources=None):
    path = None if sources is None else sources / "sources" / f"{ep.episode_id}.json"
    if path is not None:
        supplied = read_json(path)
        value = Generation(**supplied["generation"])
        if supplied["task_id"] != ep.source.task_id:
            raise ValueError("Saved source task identity differs")
    else:
        value = engine.generate(
            [ep.source.prompt],
            max_new_tokens=config["source_tokens"],
            sample=True,
            seed=seed_for(config["seed"], ep.episode_id, "source"),
            adapter=None,
        )[0]
    if (
        value.prompt != ep.source.prompt
        or value.prompt_ids != engine.encode_prompt(value.prompt)
        or not value.response_ids
        or any(type(token) is not int or token < 0 for token in value.response_ids)
    ):
        raise ValueError("Source generation must retain exact prompt and native response IDs")
    SOURCE_ADMISSION.validate_source_text(engine, value)
    result = grade(value.text, ep.source)
    feedback = "The submitted answer was " + ("correct." if result["success"] else "incorrect.")
    if path is not None and supplied["feedback"] != feedback:
        raise ValueError("Saved source feedback differs from its actual grade")
    record = dict(
        task_id=ep.source.task_id,
        generation=dataclasses.asdict(value),
        grade=result,
        feedback=feedback,
        source_origin="generated" if path is None else str(path.resolve()),
        parent_source_sha256=None if path is None else sha(path),
    )
    write_json(output / "sources" / f"{ep.episode_id}.json", record)
    return value, feedback


def adapt_from_lesson(
    engine, ep, generation, feedback, lesson, config, output, condition, candidate
):
    target = engine.score_rollout(generation, feedback, pi=lesson)
    engine.reset_student(seed=seed_for(config["seed"], "student_initialization"))
    metrics = engine.adapt(
        generation, target, steps=config["student_steps"], lr=config["learning_rate"]
    )
    append_json(
        output / "adaptations.jsonl",
        dict(
            episode_id=ep.episode_id,
            condition=condition,
            candidate=candidate,
            reset_seed=seed_for(config["seed"], "student_initialization"),
            source_response_ids=generation.response_ids,
            metrics=metrics,
        ),
    )
    return metrics


def reuse_baseline(baseline, ep, condition, output, candidate=None):
    results = []
    for row in baseline:
        record = {
            **row,
            "condition": condition,
            "candidate": candidate,
            "reused_baseline": True,
            "reason": "Unusable proposal maps to no update",
        }
        append_json(output / "observations.jsonl", record)
        results.append(record)
    return results


def train_group(engine, ep, generation, feedback, config, output):
    """Portable one-source RL group; Q is never evaluated or passed to the proposer."""
    count = config.get("reward_tasks", 4)
    if count != 4 or len(ep.validation) != 8:
        raise ValueError("Training rewards require the declared four stratified V tasks")
    tasks = [ep.validation[index] for index in REWARD_INDICES]
    baseline = evaluate_tasks(engine, ep, tasks, "V", "baseline", config, output)
    prompt = proposal_prompt(ep.source.prompt, generation.text, feedback)
    policy_hash = engine.proposer_state_sha256()
    proposals = []
    for candidate in range(4):
        proposal = engine.sample_proposal(
            prompt,
            max_new_tokens=config["pi_tokens"],
            temperature=config.get("proposer_temperature", 0.7),
            seed=seed_for(config["seed"], ep.episode_id, "proposer_train", candidate),
        )
        proposals.append(proposal)
        write_json(
            output / "proposals" / f"{ep.episode_id}-{candidate}.json",
            dict(
                episode_id=ep.episode_id,
                candidate=candidate,
                policy_sha256=policy_hash,
                usable=usable_lesson(proposal),
                proposal=dataclasses.asdict(proposal),
            ),
        )
    rewards, candidates = [], []
    for candidate, proposal in enumerate(proposals):
        record = dict(
            episode_id=ep.episode_id,
            candidate=candidate,
            policy_sha256=policy_hash,
            usable=usable_lesson(proposal),
            proposal=dataclasses.asdict(proposal),
        )
        write_json(output / "proposals" / f"{ep.episode_id}-{candidate}.json", record)
        if record["usable"]:
            adapt_from_lesson(
                engine,
                ep,
                generation,
                feedback,
                proposal.generation.text.strip(),
                config,
                output,
                "candidate",
                candidate,
            )
            observed = evaluate_tasks(
                engine,
                ep,
                tasks,
                "V",
                "candidate",
                config,
                output,
                candidate=candidate,
                adapter="student",
            )
        else:
            observed = reuse_baseline(baseline, ep, "candidate", output, candidate)
        reward = sum(row["success"] for row in observed) / len(tasks)
        rewards.append(reward)
        candidates.append(dict(candidate=candidate, usable=record["usable"], reward=reward))
    if engine.proposer_state_sha256() != policy_hash:
        raise ValueError("Proposer changed while gathering its on-policy reward group")
    before = dict(engine.counters)
    metrics = engine.reinforce_group(proposals, rewards)
    if any(engine.counters[key] != before[key] for key in CORE_COUNTERS):
        raise ValueError("Proposer update changed unrelated model-work counters")
    baseline_reward = sum(row["success"] for row in baseline) / len(tasks)
    best = max(rewards)
    winners = [index for index, value in enumerate(rewards) if value == best]
    selected = (
        None
        if best <= baseline_reward
        else min(
            winners, key=lambda index: seed_for(config["seed"], ep.episode_id, "selection", index)
        )
    )
    result = dict(
        episode_id=ep.episode_id,
        policy_sha256_before=policy_hash,
        policy_sha256_after=engine.proposer_state_sha256(),
        reward_definition="Mean exact success on V indices 0,2,4,6 after matched source update",
        invalid_proposal_rule="No update; reuse the measured deterministic baseline",
        baseline_reward=baseline_reward,
        candidates=candidates,
        selected_candidate=selected,
        selection_uses="actual_post_update_V_no_update_wins_ties",
        metrics=metrics,
    )
    append_json(output / "groups.jsonl", result)
    return result


def evaluate_proposers(engine, ep, generation, feedback, config, output):
    roles = config.get("evaluation_roles", ["V", "Q"])
    if not roles or len(set(roles)) != len(roles) or set(roles) - {"V", "Q"}:
        raise ValueError("Evaluation roles must be V and/or Q without duplicates")
    groups = [(role, ep.validation if role == "V" else ep.queries) for role in roles]
    baseline = {
        role: evaluate_tasks(engine, ep, tasks, role, "baseline", config, output)
        for role, tasks in groups
    }
    prompt = proposal_prompt(ep.source.prompt, generation.text, feedback)
    for condition, initial in (("initial_proposer", True), ("trained_proposer", False)):
        proposal = engine.sample_proposal(
            prompt,
            max_new_tokens=config["pi_tokens"],
            temperature=config.get("proposer_temperature", 0.7),
            seed=seed_for(config["seed"], ep.episode_id, "proposer_evaluation", 0),
            initial=initial,
        )
        usable = usable_lesson(proposal)
        write_json(
            output / "proposals" / f"{ep.episode_id}-{condition}.json",
            dict(
                episode_id=ep.episode_id,
                condition=condition,
                usable=usable,
                proposal=dataclasses.asdict(proposal),
            ),
        )
        if usable:
            adapt_from_lesson(
                engine,
                ep,
                generation,
                feedback,
                proposal.generation.text.strip(),
                config,
                output,
                condition,
                0,
            )
        for role, tasks in groups:
            if usable:
                evaluate_tasks(
                    engine,
                    ep,
                    tasks,
                    role,
                    condition,
                    config,
                    output,
                    candidate=0,
                    adapter="student",
                )
            else:
                reuse_baseline(baseline[role], ep, condition, output, 0)


def validate_accounting(engine, output, selected, config, phase, sources):
    observations = rows(output / "observations.jsonl")
    adaptations = rows(output / "adaptations.jsonl")
    groups = rows(output / "groups.jsonl")
    proposals = list((output / "proposals").glob("*.json"))
    roles = config.get("evaluation_roles", ["V", "Q"])
    expected_rows = (
        len(selected) * 20
        if phase == "train"
        else 3
        * sum(
            len(ep.validation if role == "V" else ep.queries) for ep in selected for role in roles
        )
    )
    identities = [
        tuple(row[key] for key in ("episode_id", "condition", "candidate", "role", "task_id"))
        for row in observations
    ]
    if len(observations) != expected_rows or len(set(identities)) != expected_rows:
        raise ValueError("Missing or duplicate E4 outcome identities")
    expected = dict.fromkeys((*CORE_COUNTERS, *PROPOSER_COUNTERS), 0)
    expected["proposer_generation_sequences"] = len(selected) * (4 if phase == "train" else 2)
    if len(proposals) != expected["proposer_generation_sequences"]:
        raise ValueError("Native proposer records do not match the declared budget")
    expected["generation_sequences"] = (
        sum(not row["reused_baseline"] for row in observations)
        + len(proposals)
        + (len(selected) if sources is None else 0)
    )
    expected["scoring_forwards"] = len(adaptations)
    expected["student_optimizer_steps"] = len(adaptations) * config["student_steps"]
    if any(len(row["metrics"]) != config["student_steps"] for row in adaptations):
        raise ValueError("Student adaptation metrics do not match the selected recipe")
    expected["proposer_groups"] = len(groups)
    expected["proposer_optimizer_steps"] = sum(row["metrics"]["optimizer_steps"] for row in groups)
    expected["proposer_zero_advantage_groups"] = sum(
        row["metrics"]["optimizer_steps"] == 0 for row in groups
    )
    expected["proposer_backward_sequences"] = expected["proposer_optimizer_steps"] * 4
    expected["proposer_scoring_forwards"] = expected["proposer_backward_sequences"]
    if engine.counters != expected:
        raise ValueError("Actual work differs from persisted E4 evidence")
    return expected


def run(
    config, data, output, *, phase="train", sources=None, checkpoint=None, engine=None, loader=None
):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        phase=phase,
        started_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_sha256=digest(config),
        completed_episode_ids=[],
        sampling="Full categorical at declared temperature; no top-k/top-p truncation",
        objective="Fresh-group leave-one-out REINFORCE using actual post-update V success",
        hidden_retries=0,
    )
    write_json(output / "receipt.json", receipt)
    snapshot, active = None, engine
    try:
        if phase not in ("train", "evaluate"):
            raise ValueError("Phase must be train or evaluate")
        limit = config["max_invocation_seconds"]
        deadline = datetime.fromisoformat(config["absolute_deadline_utc"])
        if not math.isfinite(limit) or limit <= 0 or deadline.tzinfo is None:
            raise ValueError("Require positive finite runtime and timezone-aware deadline")
        if (
            config.get("proposer_k", 4) != 4
            or config.get("reward_tasks", 4) != 4
            or type(config.get("generation_batch_size", 1)) is not int
            or config.get("generation_batch_size", 1) not in (1, 4, 8)
            or type(config["student_steps"]) is not int
            or config["student_steps"] < 1
            or not math.isfinite(config["learning_rate"])
            or config["learning_rate"] <= 0
        ):
            raise ValueError("Require frozen K4/V4, declared batch size and valid student recipe")
        if (
            type(config.get("checkpoint_every", 8)) is not int
            or config.get("checkpoint_every", 8) < 1
        ):
            raise ValueError("Checkpoint interval must be a positive integer")
        if sha(data / "manifest.json") != config["dataset_manifest_sha256"]:
            raise ValueError("Cohort manifest checksum differs from the declared dataset")
        qualification_path = None
        batch_size = config.get("generation_batch_size", 1)
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
        if loader is None:
            from hindsight.cohorts import load_cohort

            loader = load_cohort
        episodes, calibration = loader(data)
        admitted = {ep.episode_id: ep for ep in episodes}
        if len(admitted) != len(episodes):
            raise ValueError("Duplicate cohort episode identities")
        train_ids = config.get(
            "train_episode_ids", [ep.episode_id for ep in episodes if ep.partition == "meta_train"]
        )
        eval_partition = config.get("evaluation_partition", "confirmation")
        eval_ids = config.get(
            "evaluation_episode_ids",
            [ep.episode_id for ep in episodes if ep.partition == eval_partition],
        )
        if (
            not train_ids
            or not eval_ids
            or set(train_ids) & set(eval_ids)
            or len(set(train_ids)) != len(train_ids)
            or len(set(eval_ids)) != len(eval_ids)
            or not (set(train_ids) | set(eval_ids)) <= admitted.keys()
        ):
            raise ValueError("Require explicit disjoint nonempty train/evaluation sources")
        ids = train_ids if phase == "train" else eval_ids
        selected = [admitted[key] for key in ids]
        if (
            any(len(ep.validation) != 8 or not ep.queries for ep in selected)
            or any(admitted[key].partition != "meta_train" for key in train_ids)
            or eval_partition not in ("development", "confirmation")
            or any(admitted[key].partition != eval_partition for key in eval_ids)
        ):
            raise ValueError("Require V8 and disjoint meta_train/evaluation partition identities")
        if (phase == "evaluate") != (checkpoint is not None):
            raise ValueError(
                "Evaluation requires a frozen checkpoint; training starts from initialization"
            )
        root = Path(__file__).resolve().parents[1]
        paths = list(data.rglob("*.json")) + list(data.rglob("*.jsonl"))
        paths += [
            root / "src/hindsight" / name
            for name in ("model.py", "experiment.py", "data.py", "proposer.py")
        ]
        paths += [Path(__file__), Path(SOURCE_ADMISSION.__file__)]
        if (root / "src/hindsight/cohorts.py").exists():
            paths.append(root / "src/hindsight/cohorts.py")
        if sources is not None:
            paths += [sources / "sources" / f"{ep_id}.json" for ep_id in ids]
        if checkpoint is not None:
            paths.append(checkpoint)
        if qualification_path is not None:
            paths.append(qualification_path)
        snapshot = {str(path.resolve()): sha(path) for path in paths}

        def check():
            if time.monotonic() - started >= limit or datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Bounded proposer execution deadline reached")

        def verify_inputs():
            if any(sha(path) != expected for path, expected in snapshot.items()):
                raise ValueError("Frozen E4 source/config/code/dataset/checkpoint bytes changed")

        evaluation_roles = config.get("evaluation_roles", ["V", "Q"])
        task_total = (
            len(selected) * 4
            if phase == "train"
            else sum(
                len(ep.validation if role == "V" else ep.queries)
                for ep in selected
                for role in evaluation_roles
            )
        )
        proposal_total = len(selected) * (4 if phase == "train" else 2)
        ceiling = dict(
            generation_sequences=(len(selected) if sources is None else 0)
            + proposal_total
            + task_total * (5 if phase == "train" else 3),
            scoring_forwards=proposal_total,
            student_optimizer_steps=proposal_total * config["student_steps"],
            proposer_optimizer_steps=len(selected) if phase == "train" else 0,
            teacher_optimizer_steps=0,
        )
        write_json(
            output / "plan.json",
            dict(
                phase=phase,
                episode_ids=ids,
                train_episode_ids=train_ids,
                evaluation_episode_ids=eval_ids,
                source_generation="one per source"
                if sources is None
                else "reuse native saved sources",
                proposal_budget_per_source=4 if phase == "train" else 2,
                reward_task_ids={
                    ep.episode_id: [ep.validation[i].task_id for i in REWARD_INDICES]
                    for ep in selected
                },
                student_recipe={
                    key: config[key] for key in ("student_steps", "learning_rate", "seed")
                },
                work_ceiling=ceiling,
                frozen_files=snapshot,
            ),
        )
        receipt.update(
            episode_ids=ids, frozen_files=snapshot, expected_sources=len(ids), work_ceiling=ceiling
        )
        write_json(output / "receipt.json", receipt)
        checkpoint_metadata = dict(
            config_sha256=digest(config),
            dataset_manifest_sha256=sha(data / "manifest.json"),
            train_episode_ids=train_ids,
            student_recipe={key: config[key] for key in ("student_steps", "learning_rate", "seed")},
            proposer_temperature=config.get("proposer_temperature", 0.7),
            proposer_module_sha256=sha(root / "src/hindsight/proposer.py"),
        )
        check()
        configure_deterministic_execution(config.get("deterministic_execution", False))
        if active is None:
            active = ProposerEngine(
                model_name=config["model_name"],
                revision=config["revision"],
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"],
                enable_thinking=config.get("enable_thinking"),
                budget_check=check,
            )
        else:
            active.budget_check = check
        if any(active.counters.values()):
            raise ValueError("Require a fresh unused engine")
        active.initialize_proposer(
            seed=config.get("proposer_seed", seed_for(config["seed"], "proposer_initialization")),
            lr=config.get("proposer_learning_rate", 1e-5),
        )
        if checkpoint is not None:
            payload = active.load_proposer(checkpoint)
            if any(
                payload["metadata"].get(key) != value
                for key, value in checkpoint_metadata.items()
                if key != "config_sha256"
            ):
                raise ValueError(
                    "Checkpoint learning recipe, training sources or policy implementation changed"
                )
        initial_hash = active.proposer_state_sha256(initial=True)
        trained_hash = active.proposer_state_sha256()
        receipt.update(initial_proposer_sha256=initial_hash, starting_proposer_sha256=trained_hash)
        if phase == "train":
            active.save_proposer(output / "checkpoints/initial.pt", metadata=checkpoint_metadata)
        for ep in selected:
            check()
            verify_inputs()
            receipt["current_episode_id"] = ep.episode_id
            generation, feedback = source_for(active, ep, config, output, sources)
            if phase == "train":
                train_group(active, ep, generation, feedback, config, output)
            else:
                evaluate_proposers(active, ep, generation, feedback, config, output)
            if active.proposer_state_sha256(initial=True) != initial_hash:
                raise ValueError("Frozen proposer initialization changed")
            if phase == "evaluate" and active.proposer_state_sha256() != trained_hash:
                raise ValueError("Evaluation changed the trained proposer")
            receipt["completed_episode_ids"].append(ep.episode_id)
            receipt.update(work=dict(active.counters), elapsed_seconds=time.monotonic() - started)
            write_json(output / "receipt.json", receipt)
            if (
                phase == "train"
                and len(receipt["completed_episode_ids"]) % config.get("checkpoint_every", 8) == 0
            ):
                active.save_proposer(
                    output / f"checkpoints/group-{len(receipt['completed_episode_ids']):04}.pt",
                    metadata=checkpoint_metadata,
                )
        verify_inputs()
        if phase == "train":
            active.save_proposer(output / "checkpoints/final.pt", metadata=checkpoint_metadata)
            receipt["checkpoint_sha256"] = sha(output / "checkpoints/final.pt")
            if active.counters["proposer_groups"] != len(selected):
                raise ValueError("Actual proposer groups differ from the declared source count")
        elif any(
            active.counters[key]
            for key in (
                "proposer_optimizer_steps",
                "proposer_backward_sequences",
                "proposer_groups",
            )
        ):
            raise ValueError("Evaluation performed proposer training")
        if any(
            active.counters[key] for key in ("teacher_optimizer_steps", "teacher_example_exposures")
        ):
            raise ValueError("E4 must not fit a teacher")
        validate_accounting(active, output, selected, config, phase, sources)
        receipt.update(status="completed", final_proposer_sha256=active.proposer_state_sha256())
        receipt["proposer_differs_from_initialization"] = (
            receipt["final_proposer_sha256"] != initial_hash
        )
        if phase == "train":
            receipt["nonzero_advantage_groups"] = active.counters["proposer_optimizer_steps"]
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            actual_sources=len(receipt["completed_episode_ids"]),
            elapsed_seconds=time.monotonic() - started,
            work=dict(active.counters)
            if active is not None
            else dict.fromkeys((*CORE_COUNTERS, *PROPOSER_COUNTERS), 0),
        )
        if snapshot is not None:
            try:
                receipt["frozen_inputs_unchanged"] = all(
                    sha(path) == value for path, value in snapshot.items()
                )
                if not receipt["frozen_inputs_unchanged"]:
                    receipt["status"] = "failed"
            except Exception as error:
                receipt.update(
                    status="failed",
                    frozen_inputs_unchanged=False,
                    integrity_error=dict(type=type(error).__name__, message=str(error)),
                )
        write_json(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    return run(
        read_json(args.config),
        args.data,
        args.output,
        phase=args.phase,
        sources=args.sources,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
