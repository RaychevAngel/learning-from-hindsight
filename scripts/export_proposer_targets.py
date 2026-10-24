"""Export actual-V-selected E4 targets with one deterministic source score each."""

import argparse
import dataclasses
import hashlib
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import grade
from hindsight.experiment import proposal_prompt, read_json, rows, seed_for, write_json
from hindsight.model import Generation, HFEngine, configure_deterministic_execution
from hindsight.proposer import Proposal, usable_lesson

COUNTERS = (
    "generation_sequences",
    "scoring_forwards",
    "student_optimizer_steps",
    "teacher_optimizer_steps",
    "teacher_example_exposures",
)
INDICES = (0, 2, 4, 6)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def admit(training, data, loader):
    prior = read_json(training / "receipt.json")
    config = prior["config"]
    plan = read_json(training / "plan.json")
    episodes, _ = loader(data)
    admitted = {ep.episode_id: ep for ep in episodes}
    ids = plan["train_episode_ids"]
    if (
        prior["status"] != "completed"
        or prior["phase"] != "train"
        or not ids
        or len(set(ids)) != len(ids)
        or ids != prior["completed_episode_ids"]
        or ids != prior["episode_ids"]
        or prior["actual_sources"] != len(ids)
        or not prior["frozen_inputs_unchanged"]
        or sha(data / "manifest.json") != config["dataset_manifest_sha256"]
        or any(admitted[key].partition != "meta_train" for key in ids)
    ):
        raise ValueError("Require complete immutable E4 training on this exact meta-train cohort")
    observed, groups = rows(training / "observations.jsonl"), rows(training / "groups.jsonl")
    if [group["episode_id"] for group in groups] != ids or len(observed) != 20 * len(ids):
        raise ValueError("E4 selection evidence has missing or extra source/task rows")
    paths = [
        training / name
        for name in ("receipt.json", "plan.json", "observations.jsonl", "groups.jsonl")
    ]
    examples = []
    for version, (key, group) in enumerate(zip(ids, groups, strict=True)):
        ep = admitted[key]
        tasks = [ep.validation[index] for index in INDICES]
        if len(ep.validation) != 8 or plan["reward_task_ids"][key] != [t.task_id for t in tasks]:
            raise ValueError("Selection does not use the frozen stratified V4 task identities")
        source_path = training / "sources" / f"{key}.json"
        source = read_json(source_path)
        generation = Generation(**source["generation"])
        expected_grade = grade(generation.text, ep.source)
        feedback = "The submitted answer was " + (
            "correct." if expected_grade["success"] else "incorrect."
        )
        if (
            source["task_id"] != ep.source.task_id
            or generation.prompt != ep.source.prompt
            or source["grade"] != expected_grade
            or source["feedback"] != feedback
        ):
            raise ValueError("Source identity, original grade or binary feedback differs")
        paths.append(source_path)
        rewards, proposals = [], []
        baseline = None
        for candidate in (None, 0, 1, 2, 3):
            condition = "baseline" if candidate is None else "candidate"
            records = [
                row
                for row in observed
                if row["episode_id"] == key
                and row["condition"] == condition
                and row["candidate"] == candidate
            ]
            if len(records) != 4 or [row["task_id"] for row in records] != [
                task.task_id for task in tasks
            ]:
                raise ValueError("Candidate selection rows differ from exact V4 baseline")
            for row, task in zip(records, tasks, strict=True):
                if (
                    row["role"] != "V"
                    or row["generation"]["prompt"] != task.prompt
                    or type(row["success"]) is not bool
                    or row["success"] != grade(row["generation"]["text"], task)["success"]
                ):
                    raise ValueError("Selection rewards must be actual graded task-only V answers")
            reward = sum(row["success"] for row in records) / 4
            if candidate is None:
                baseline = reward
                continue
            proposal_path = training / "proposals" / f"{key}-{candidate}.json"
            record = read_json(proposal_path)
            value = dict(record["proposal"])
            value["generation"] = Generation(**value["generation"])
            proposal = Proposal(**value)
            if (
                record["episode_id"] != key
                or record["candidate"] != candidate
                or record["usable"] != usable_lesson(proposal)
                or proposal.generation.prompt
                != proposal_prompt(generation.prompt, generation.text, feedback)
                or proposal.initial
                or proposal.policy_version != version
                or proposal.temperature != config.get("proposer_temperature", 0.7)
                or proposal.seed != seed_for(config["seed"], key, "proposer_train", candidate)
                or any(row["reused_baseline"] == record["usable"] for row in records)
                or (not record["usable"] and reward != baseline)
            ):
                raise ValueError(
                    "Proposal source/native policy or invalid-proposal evidence differs"
                )
            paths.append(proposal_path)
            proposals.append(proposal)
            rewards.append(reward)
        winners = [index for index, value in enumerate(rewards) if value == max(rewards)]
        selected = (
            None
            if max(rewards) <= baseline
            else min(winners, key=lambda index: seed_for(config["seed"], key, "selection", index))
        )
        if (
            group["selected_candidate"] != selected
            or group["baseline_reward"] != baseline
            or group["metrics"]["rewards"] != rewards
            or [candidate["reward"] for candidate in group["candidates"]] != rewards
        ):
            raise ValueError("Selected target violates actual V rewards or neutral-wins-ties rule")
        examples.append((ep, source, proposals, selected))
    return config, examples, paths


def run(training, data, output, *, max_seconds=300, deadline_utc, engine=None, loader=None):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        completed_episode_ids=[],
        runtime_overrides=dict(max_seconds=max_seconds, deadline_utc=deadline_utc),
        hidden_retries=0,
    )
    write_json(output / "receipt.json", receipt)
    active, frozen = engine, {}
    try:
        deadline = datetime.fromisoformat(deadline_utc)
        if not math.isfinite(max_seconds) or max_seconds <= 0 or deadline.tzinfo is None:
            raise ValueError("Require positive finite runtime and timezone-aware deadline")

        def check():
            if time.monotonic() - started >= max_seconds or datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Target export deadline reached")

        check()
        if loader is None:
            from hindsight.cohorts import load_cohort

            loader = load_cohort
        config, examples, paths = admit(training, data, loader)
        root = Path(__file__).resolve().parents[1]
        prior = read_json(training / "receipt.json")
        for name in ("model.py", "experiment.py", "data.py", "proposer.py", "cohorts.py"):
            current = root / "src/hindsight" / name
            matches = [
                value
                for key, value in prior["frozen_files"].items()
                if key.endswith("/src/hindsight/" + name)
            ]
            if matches != [sha(current)]:
                raise ValueError("E4 core/scientific implementation changed before target export")
            paths.append(current)
        paths += list(data.rglob("*.json")) + list(data.rglob("*.jsonl")) + [Path(__file__)]
        frozen = {str(path.resolve()): sha(path) for path in paths}
        expected = dict.fromkeys(COUNTERS, 0)
        expected["scoring_forwards"] = len(examples)
        receipt.update(
            config=config,
            frozen_files=frozen,
            planned_work=expected,
            training_receipt_sha256=sha(training / "receipt.json"),
        )
        write_json(output / "receipt.json", receipt)
        configure_deterministic_execution(config.get("deterministic_execution", False))
        if active is None:
            active = HFEngine(
                model_name=config["model_name"],
                revision=config["revision"],
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"],
                enable_thinking=config.get("enable_thinking"),
                budget_check=check,
            )
        active.budget_check = check
        if (
            active.counters != dict.fromkeys(COUNTERS, 0)
            or active.model_name != config["model_name"]
            or active.revision != config["revision"]
        ):
            raise ValueError("Require unused exact-model engine for source-only rescoring")
        shutil.copyfile(training / "observations.jsonl", output / "selection-observations.jsonl")
        manifest = dict(
            format="shared-teaching-v1",
            model_name=config["model_name"],
            revision=config["revision"],
            dataset_manifest_sha256=config["dataset_manifest_sha256"],
            selection_validation_indices=list(INDICES),
            selection_condition="candidate",
            selection_recipe=dict(
                student_steps=config["student_steps"],
                learning_rate=config["learning_rate"],
                loss="forward_kl",
            ),
            selection_observations_path="selection-observations.jsonl",
            selection_observations_sha256=sha(output / "selection-observations.jsonl"),
            training_receipt_sha256=receipt["training_receipt_sha256"],
            examples=[],
        )
        for ep, source, proposals, selected in examples:
            check()
            generation = Generation(**source["generation"])
            for value in (generation, *(p.generation for p in proposals)):
                if (
                    value.prompt_ids != active.encode_prompt(value.prompt)
                    or not value.response_ids
                    or any(type(token) is not int or token < 0 for token in value.response_ids)
                    or value.text
                    != active.tokenizer.decode(value.response_ids, skip_special_tokens=True)
                ):
                    raise ValueError(
                        "Original native source/proposal IDs differ from pinned tokenizer"
                    )
            lesson = None if selected is None else proposals[selected].generation.text.strip()
            target = active.score_rollout(
                generation, source["feedback"], pi=lesson, mode="causal", adapter=None
            )
            active._validate_target(generation, target)
            if target.requires_grad or not torch.isfinite(target).all():
                raise ValueError(
                    "Export requires detached finite full-vocabulary log probabilities"
                )
            tokens, positions, _ = active._scoring_input(
                generation, source["feedback"], lesson, "causal"
            )
            artifact = dict(
                target=target.detach().float().cpu(),
                episode_id=ep.episode_id,
                selected_candidate=selected,
                source_generation=dataclasses.asdict(generation),
                feedback=source["feedback"],
                lesson=lesson,
                input_ids=tokens,
                predecessor_positions=positions,
                response_ids=generation.response_ids,
                target_mode="causal_base_neutral"
                if selected is None
                else "causal_base_with_selected_pi",
            )
            source_path, target_path = (
                Path("sources") / f"{ep.episode_id}.json",
                Path("targets") / f"{ep.episode_id}.pt",
            )
            write_json(output / source_path, source)
            (output / target_path).parent.mkdir(parents=True, exist_ok=True)
            check()
            torch.save(artifact, output / target_path)
            check()
            entry = dict(
                episode_id=ep.episode_id,
                selected_candidate=selected,
                source_path=str(source_path),
                source_sha256=sha(output / source_path),
                target_path=str(target_path),
                target_sha256=sha(output / target_path),
                target_key="target",
                target_mode=artifact["target_mode"],
                lesson=lesson,
                proposal_sha256=None
                if selected is None
                else sha(training / "proposals" / f"{ep.episode_id}-{selected}.json"),
            )
            manifest["examples"].append(entry)
            write_json(output / "manifest.partial.json", manifest)
            receipt["completed_episode_ids"].append(ep.episode_id)
            receipt.update(work=dict(active.counters))
            write_json(output / "receipt.json", receipt)
        if active.counters != expected:
            raise ValueError("Target export performed work beyond one score per source")
        if any(sha(path) != value for path, value in frozen.items()):
            raise ValueError("Frozen selection/source/code/corpus changed during export")
        write_json(output / "manifest.json", manifest)
        receipt.update(status="completed", manifest_sha256=sha(output / "manifest.json"))
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            work=dict(active.counters) if active is not None else dict.fromkeys(COUNTERS, 0),
        )
        write_json(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-output", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    return run(
        args.training_output,
        args.data,
        args.output,
        max_seconds=args.max_seconds,
        deadline_utc=args.deadline_utc,
    )


if __name__ == "__main__":
    raise SystemExit(main())
