"""Matched E1/E2/E3 source-only PI comparisons on a frozen disjoint cohort."""

import argparse
import dataclasses
import importlib.util
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.cohorts import load_cohort
from hindsight.data import grade
from hindsight.experiment import (
    GENERIC,
    append_json,
    choose_candidate,
    digest,
    proposal_prompt,
    read_json,
    rows,
    seed_for,
    write_json,
)
from hindsight.model import configure_deterministic_execution, with_pi
from hindsight.proposer import ProposerEngine, usable_lesson

SPEC = importlib.util.spec_from_file_location(
    "proposer_io", Path(__file__).with_name("train_proposer.py")
)
IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IO)


def evaluate(
    engine, ep, tasks, role, condition, config, output, *, pi=None, adapter=None, candidate=None
):
    size = config["generation_batch_size"]
    result = []
    for offset in range(0, len(tasks), size):
        batch = tasks[offset : offset + size]
        prompts = [with_pi(t.prompt, pi) if pi is not None else t.prompt for t in batch]
        values = engine.generate(
            prompts,
            max_new_tokens=config["evaluation_tokens"],
            sample=False,
            seed=config["seed"],
            adapter=adapter,
        )
        for task, prompt, value in zip(batch, prompts, values, strict=True):
            if value.prompt != prompt:
                raise ValueError("Generated prompt differs from requested condition")
            record = dict(
                **IO.metadata(ep, task, role, condition, candidate),
                **grade(value.text, task),
                generation=dataclasses.asdict(value),
                generation_batch_size=len(batch),
                reused_baseline=False,
            )
            append_json(output / "observations.jsonl", record)
            result.append(record)
    return result


def matched_mismatches(selected):
    """A fixed other-family source; never choose mismatch using outcomes."""
    return {
        ep.episode_id: next(other.episode_id for other in selected if other.family != ep.family)
        for ep in selected
    }


def select_with_fixed_order(baseline, candidates, seed):
    # Reuse the established V-only completeness and no-update validation.
    if choose_candidate(baseline, candidates, seed) is None:
        return None
    scores = [sum(row["success"] for row in group) for group in candidates]
    winners = [i for i, score in enumerate(scores) if score == max(scores)]
    return min(winners, key=lambda i: seed_for(seed, "candidate_priority", i))


def manifest(config, output, examples):
    return dict(
        format="shared-teaching-v1",
        model_name=config["model_name"],
        revision=config["revision"],
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        selection_recipe=dict(
            student_steps=config["student_steps"],
            learning_rate=config["learning_rate"],
            loss="forward_kl",
        ),
        selection_validation_indices=list(range(8)),
        selection_observations_path="observations.jsonl",
        selection_observations_sha256=IO.sha(output / "observations.jsonl"),
        selection_condition="adapted_pi",
        examples=examples,
    )


def run(config, data, output, *, checkpoint=None, sources=None, engine=None):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        config=config,
        config_sha256=digest(config),
        completed_episode_ids=[],
        hidden_retries=0,
        selection="V only, no-update wins ties; Q is never used for selection",
    )
    write_json(output / "receipt.json", receipt)
    active = engine
    try:
        episodes, _ = load_cohort(data)
        partition = config["partition"]
        if partition not in ("development", "confirmation"):
            raise ValueError("PI comparison requires development or confirmation partition")
        selected = [ep for ep in episodes if ep.partition == partition]
        if not selected or len(selected) != config["expected_sources"]:
            raise ValueError("Unexpected comparison cohort size")
        if any(len(ep.validation) != 8 or len(ep.queries) != 8 for ep in selected):
            raise ValueError("Require complete V8/Q8 on every source")
        if IO.sha(data / "manifest.json") != config["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest differs from frozen config")
        qualified = config["batch_qualification"]
        qualification = read_json(Path(qualified["path"]))
        if (
            IO.sha(qualified["path"]) != qualified["sha256"]
            or qualification["status"] != "completed"
            or qualification["selected_batch_size"] != config["generation_batch_size"]
            or not qualification["repeat_stability"].get(
                str(config["generation_batch_size"]), False
            )
            or qualification["dataset_sha256"] != config["dataset_manifest_sha256"]
            or any(qualification["config"][k] != config[k] for k in ("model_name", "revision"))
        ):
            raise ValueError("Unqualified generation schedule")
        if config.get("candidate_count", 4) != 4:
            raise ValueError("Predeclared candidate pool is K4")
        frozen = {
            str(p.resolve()): IO.sha(p)
            for p in (
                Path(__file__),
                Path(IO.__file__),
                Path(IO.SOURCE_ADMISSION.__file__),
                data / "manifest.json",
                Path(qualified["path"]),
                Path(__file__).parents[1] / "src/hindsight/proposer.py",
                Path(__file__).parents[1] / "src/hindsight/model.py",
            )
        }
        if checkpoint is not None:
            if IO.sha(checkpoint) != config["proposer_checkpoint_sha256"]:
                raise ValueError("Proposer checkpoint differs from frozen comparison")
            frozen[str(checkpoint.resolve())] = IO.sha(checkpoint)

        def check():
            if time.monotonic() - started >= config["max_invocation_seconds"] or datetime.now(
                timezone.utc
            ) >= datetime.fromisoformat(config["absolute_deadline_utc"]):
                raise TimeoutError("Bounded comparison deadline reached")

        configure_deterministic_execution(config["deterministic_execution"])
        if active is None:
            active = ProposerEngine(
                config["model_name"],
                config["revision"],
                enable_thinking=config["enable_thinking"],
                budget_check=check,
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"],
            )
        if any(active.counters.values()):
            raise ValueError("Require unused engine")
        active.budget_check = check
        active.initialize_proposer(seed=seed_for(config["seed"], "proposer_initialization"))
        if checkpoint is not None:
            saved = active.load_proposer(checkpoint)
            meta = saved["metadata"]
            if (
                meta["dataset_manifest_sha256"] != config["dataset_manifest_sha256"]
                or meta["student_recipe"]
                != {k: config[k] for k in ("student_steps", "learning_rate", "seed")}
                or set(meta["train_episode_ids"]) & {ep.episode_id for ep in selected}
            ):
                raise ValueError("Proposer training identity/recipe overlaps or differs")
        proposer_hash = active.proposer_state_sha256()
        mismatch_ids = matched_mismatches(selected)
        cache = {}
        # First collect sources and the exact K4 pool. No V/Q has been observed yet.
        for ep in selected:
            generation, feedback = IO.source_for(active, ep, config, output, sources)
            prompt = proposal_prompt(ep.source.prompt, generation.text, feedback)
            proposals = []
            for candidate in range(4):
                value = active.sample_proposal(
                    prompt,
                    max_new_tokens=config["pi_tokens"],
                    temperature=config.get("proposer_temperature", 0.7),
                    seed=seed_for(config["seed"], ep.episode_id, "comparison_pi", candidate),
                )
                write_json(
                    output / "proposals" / f"{ep.episode_id}-{candidate}.json",
                    dict(
                        episode_id=ep.episode_id,
                        candidate=candidate,
                        proposal=dataclasses.asdict(value),
                        usable=usable_lesson(value),
                        policy_sha256=proposer_hash,
                    ),
                )
                proposals.append(value)
            cache[ep.episode_id] = generation, feedback, proposals
        examples = []
        for ep in selected:
            generation, feedback, proposals = cache[ep.episode_id]
            baseline = {
                role: evaluate(active, ep, tasks, role, "baseline", config, output)
                for role, tasks in (("V", ep.validation), ("Q", ep.queries))
            }
            donor = cache[mismatch_ids[ep.episode_id]][2][0]
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                evaluate(active, ep, tasks, role, "generic", config, output, pi=GENERIC)
                if usable_lesson(donor):
                    evaluate(
                        active,
                        ep,
                        tasks,
                        role,
                        "mismatched_pi",
                        config,
                        output,
                        pi=donor.generation.text.strip(),
                    )
                else:
                    IO.reuse_baseline(baseline[role], ep, "mismatched_pi", output)
            direct, adapted, targets = [], [], []
            for candidate, proposal in enumerate(proposals):
                if not usable_lesson(proposal):
                    condition_rows = {}
                    for condition in ("direct_pi", "adapted_pi"):
                        for role in ("V", "Q"):
                            condition_rows[condition, role] = IO.reuse_baseline(
                                baseline[role], ep, condition, output, candidate
                            )
                    direct.append(condition_rows["direct_pi", "V"])
                    adapted.append(condition_rows["adapted_pi", "V"])
                    targets.append(None)
                    continue
                lesson = proposal.generation.text.strip()
                for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                    result = evaluate(
                        active,
                        ep,
                        tasks,
                        role,
                        "direct_pi",
                        config,
                        output,
                        pi=lesson,
                        candidate=candidate,
                    )
                    if role == "V":
                        direct.append(result)
                target = active.score_rollout(generation, feedback, pi=lesson)
                active.reset_student(seed_for(config["seed"], "student_initialization"))
                metrics = active.adapt(
                    generation, target, config["student_steps"], config["learning_rate"]
                )
                append_json(
                    output / "adaptations.jsonl",
                    dict(
                        episode_id=ep.episode_id,
                        candidate=candidate,
                        condition="adapted_pi",
                        source_response_ids=generation.response_ids,
                        metrics=metrics,
                    ),
                )
                for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                    result = evaluate(
                        active,
                        ep,
                        tasks,
                        role,
                        "adapted_pi",
                        config,
                        output,
                        adapter="student",
                        candidate=candidate,
                    )
                    if role == "V":
                        adapted.append(result)
                targets.append(target)
            tie_seed = seed_for(config["seed"], ep.episode_id, "pi_tie_order")
            winner = select_with_fixed_order(baseline["V"], adapted, tie_seed)
            direct_winner = select_with_fixed_order(baseline["V"], direct, tie_seed)
            append_json(
                output / "selections.jsonl",
                dict(
                    episode_id=ep.episode_id,
                    selected_candidate=winner,
                    direct_selected_candidate=direct_winner,
                    derived_from="V",
                    baseline_successes=sum(r["success"] for r in baseline["V"]),
                    adapted_successes=[sum(r["success"] for r in value) for value in adapted],
                    direct_successes=[sum(r["success"] for r in value) for value in direct],
                    mismatch_episode_id=mismatch_ids[ep.episode_id],
                    mismatch_candidate=0,
                    mismatch_usable=usable_lesson(donor),
                    candidate_tie_order=sorted(
                        range(4), key=lambda i: seed_for(tie_seed, "candidate_priority", i)
                    ),
                ),
            )
            selected_target = (
                targets[winner]
                if winner is not None
                else active.score_rollout(generation, feedback)
            )
            target_path = output / "targets" / f"{ep.episode_id}.pt"
            target_path.parent.mkdir(exist_ok=True)
            torch.save(selected_target.detach().cpu(), target_path)
            source_path = output / "sources" / f"{ep.episode_id}.json"
            examples.append(
                dict(
                    episode_id=ep.episode_id,
                    selected_candidate=winner,
                    source_path=str(source_path.relative_to(output)),
                    source_sha256=IO.sha(source_path),
                    target_path=str(target_path.relative_to(output)),
                    target_sha256=IO.sha(target_path),
                )
            )
            receipt["completed_episode_ids"].append(ep.episode_id)
            receipt.update(work=dict(active.counters), elapsed_seconds=time.monotonic() - started)
            write_json(output / "receipt.json", receipt)
        observed = rows(output / "observations.jsonl")
        expected_rows = len(selected) * 16 * 11
        identities = {
            (r["episode_id"], r["role"], r["task_id"], r["condition"], r["candidate"])
            for r in observed
        }
        if len(observed) != expected_rows or len(identities) != expected_rows:
            raise ValueError("Incomplete or duplicate comparison outcome matrix")
        adaptations = rows(output / "adaptations.jsonl")
        neutral_count = sum(e["selected_candidate"] is None for e in examples)
        expected_work = {key: 0 for key in active.counters}
        expected_work.update(
            generation_sequences=sum(not r["reused_baseline"] for r in observed)
            + len(selected) * (4 + (sources is None)),
            proposer_generation_sequences=len(selected) * 4,
            scoring_forwards=len(adaptations) + neutral_count,
            student_optimizer_steps=len(adaptations) * config["student_steps"],
        )
        if active.counters != expected_work:
            raise ValueError("Comparison model work differs from persisted evidence")
        if active.proposer_state_sha256() != proposer_hash:
            raise ValueError("Frozen comparison proposer changed")
        if any(IO.sha(path) != value for path, value in frozen.items()):
            raise ValueError("Comparison frozen inputs changed")
        write_json(output / "teaching.json", manifest(config, output, examples))
        receipt.update(
            status="completed",
            expected_rows=expected_rows,
            expected_work=expected_work,
            frozen_files=frozen,
            teaching_manifest_sha256=IO.sha(output / "teaching.json"),
        )
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        if active is not None:
            receipt["work"] = dict(active.counters)
        write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args()
    print(
        run(
            read_json(args.config),
            args.data,
            args.output,
            checkpoint=args.checkpoint,
            sources=args.sources,
        )["status"]
    )
