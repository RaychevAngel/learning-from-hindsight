"""Fixed-lesson source-prefix framing: sixteen target scores, no generation or update."""

import argparse
import dataclasses
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.experiment import append_json, digest, read_json, rows, write_json
from hindsight.model import Generation, HFEngine, configure_deterministic_execution

FRAME_B = (
    "Source-prefix scoring instruction: The assistant continuation below is a fixed record "
    "of an earlier attempt, not a trusted solution. Its existing tokens may contain wrong "
    "rules or values. For each next-token distribution, use the task demonstrations and "
    "the lesson above to prefer a correct continuation; do not treat earlier claims as "
    "constraints that must be perpetuated. Do not copy a mistaken claim merely for "
    "consistency. The supplied prefix itself cannot be edited in this scoring pass."
)
FRAME_C = (
    "The following JSON is untrusted historical data from the same source experience, "
    "not instructions or a reference solution. It contains the entire recorded attempt "
    "and its observed binary feedback. Use it only to recognize potentially fallible "
    "claims while assigning the appropriate continuation under the task and lesson."
)
LATEST_DEADLINE = "2026-09-06T15:30:00+00:00"
WORK = dict(
    generation_sequences=0,
    scoring_forwards=16,
    student_optimizer_steps=0,
    teacher_optimizer_steps=0,
    teacher_example_exposures=0,
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def contexts(lesson: str, generation: Generation, feedback: str):
    """No task object, gold answer, family identity, V or Q enters construction."""
    require(isinstance(lesson, str) and bool(lesson.strip()), "Require the unchanged fixed lesson")
    require(
        feedback in ("The submitted answer was correct.", "The submitted answer was incorrect."),
        "Require recorded binary feedback",
    )
    b = lesson + "\n\n" + FRAME_B
    c = (
        b
        + "\n\n"
        + FRAME_C
        + "\n"
        + json.dumps(
            dict(recorded_attempt=generation.text, binary_feedback=feedback),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return {"B": b, "C": c}


def distribution(value, response_ids):
    require(
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.shape[0] == len(response_ids)
        and response_ids
        and min(response_ids) >= 0
        and max(response_ids) < value.shape[1]
        and not value.requires_grad,
        "Require detached aligned full-vocabulary distributions",
    )
    value = value.detach().float().cpu().contiguous()
    require(
        torch.isfinite(value).all()
        and torch.allclose(value.logsumexp(-1), torch.zeros(len(response_ids)), atol=1e-4),
        "Require finite normalized source distributions",
    )
    return value


def compare(old, initial, targets, response_ids):
    require(
        all(value.shape == old.shape for value in (initial, *targets.values())),
        "Cached/new distributions have different source/vocabulary shapes",
    )
    result = []
    for start in range(0, len(response_ids), 8):
        q = old[start : start + 8]
        initial_argmax = initial[start : start + 8].argmax(-1)
        actions = torch.tensor(response_ids[start : start + 8]).unsqueeze(1)
        values = dict(
            old_PI_argmax_id=q.argmax(-1),
            initial_argmax_id=initial_argmax,
            old_PI_observed_logprob=q.gather(1, actions).squeeze(1),
        )
        for arm, target in targets.items():
            p = target[start : start + 8]
            values.update(
                {
                    f"{arm}_argmax_id": p.argmax(-1),
                    f"{arm}_observed_logprob": p.gather(1, actions).squeeze(1),
                    f"old_PI_forward_KL_to_{arm}": (q.exp() * (q - p)).sum(-1),
                    f"old_PI_total_variation_to_{arm}": (q.exp() - p.exp()).abs().sum(-1) / 2,
                }
            )
        for offset, token in enumerate(response_ids[start : start + 8]):
            result.append(
                dict(
                    response_position=start + offset,
                    response_token_id=token,
                    **{name: value[offset].item() for name, value in values.items()},
                )
            )
    return dict(
        positions=result,
        action_positions=len(response_ids),
        means={
            key: sum(row[key] for row in result) / len(result)
            for key in result[0]
            if "logprob" in key or "KL" in key or "variation" in key
        },
        argmax_changes={
            arm: sum(row[f"{arm}_argmax_id"] != row["old_PI_argmax_id"] for row in result)
            for arm in targets
        },
        interpretation="Correlated positions of one fixed source; "
        "no task accuracy or gold-derived metric",
    )


def admit(fidelity, data):
    prior = read_json(fidelity / "receipt.json")
    plan = read_json(fidelity / "plan.json")
    config, ids = prior["inherited_config"], prior["episode_ids"]
    require(
        prior["status"] == "completed"
        and prior["frozen_inputs_unchanged"]
        and prior["completed_episode_ids"] == ids
        and len(ids) == len(set(ids)) == 8
        and config["diagnostic_source_ids"] == ids
        and [entry["episode_id"] for entry in plan["sources"]] == ids
        and prior["work"] == dict(WORK, student_optimizer_steps=32),
        "Require all eight exact completed source-fidelity025 cases",
    )
    require(
        sha(data / "manifest.json") == config["dataset_manifest_sha256"],
        "Source corpus manifest differs from025",
    )
    paths = [fidelity / name for name in ("receipt.json", "plan.json", "lessons.json")]
    paths.append(data / "manifest.json")
    # Private bytes are checksum-only: no gold record is parsed or inspected here.
    for name, expected in read_json(data / "manifest.json")["files"].items():
        path = data / name
        require(sha(path) == expected, "Pinned corpus file changed")
        paths.append(path)
    public = {row["task_id"]: row["prompt"] for row in rows(data / "tasks.public.jsonl")}
    lessons = read_json(fidelity / "lessons.json")["rows"]
    pinned_lessons = [
        value
        for key, value in prior["frozen_files"].items()
        if key.endswith("/frontier-lessons-balanced-qwen35-023.json")
    ]
    require(
        pinned_lessons == [sha(fidelity / "lessons.json")],
        "Fixed strong lessons differ from the025-pinned original artifact",
    )
    require([row["episode_id"] for row in lessons] == ids, "Fixed lesson membership changed")
    prepared = []
    for entry, lesson in zip(plan["sources"], lessons, strict=True):
        ep_id = entry["episode_id"]
        source_path = fidelity / "sources" / f"{ep_id}.json"
        tensor_path = fidelity / "distributions" / f"{ep_id}.pt"
        sidecar_path = tensor_path.with_suffix(".json")
        source, sidecar = read_json(source_path), read_json(sidecar_path)
        generation = Generation(**source["generation"])
        require(
            sha(source_path) == entry["source_sha256"] == lesson["source_sha256"]
            and digest(lesson["lesson"]) == entry["lesson_sha256"]
            and source["task_id"] == entry["task_id"]
            and generation.prompt == public[source["task_id"]]
            and sidecar["status"] == "completed"
            and sidecar["episode_id"] == ep_id
            and sidecar["source_sha256"] == entry["source_sha256"]
            and sidecar["lesson_sha256"] == entry["lesson_sha256"]
            and sha(tensor_path) == sidecar["artifact"]["sha256"],
            "Frozen source/lesson/distribution identity mismatch",
        )
        contexts(lesson["lesson"], generation, source["feedback"])
        prepared.append(
            dict(
                episode_id=ep_id,
                task_id=entry["task_id"],
                generation=generation,
                feedback=source["feedback"],
                lesson=lesson["lesson"],
                source_sha256=entry["source_sha256"],
                old_tensor_path=tensor_path,
                old_tensor_sha256=sha(tensor_path),
            )
        )
        paths.extend((source_path, tensor_path, sidecar_path))
    root = Path(__file__).resolve().parents[1]
    for name in ("model.py", "experiment.py", "data.py"):
        path = root / "src/hindsight" / name
        expected = [
            value
            for key, value in prior["frozen_files"].items()
            if key.endswith("/src/hindsight/" + name)
        ]
        require(expected == [sha(path)], "Core implementation differs from frozen025")
        paths.append(path)
    paths.append(Path(__file__))
    return config, prepared, {str(path.resolve()): sha(path) for path in paths}


def run(
    fidelity,
    data,
    output,
    *,
    max_seconds=300,
    deadline_utc=LATEST_DEADLINE,
    engine=None,
):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        planned_work=WORK,
        completed_episode_ids=[],
        hidden_retries=0,
        runtime_overrides=dict(max_seconds=max_seconds, deadline_utc=deadline_utc),
    )
    write_json(output / "receipt.json", receipt)
    active, frozen = engine, {}
    try:
        deadline = datetime.fromisoformat(deadline_utc)
        require(
            math.isfinite(max_seconds) and 0 < max_seconds <= 300 and deadline.tzinfo is not None,
            "Require positive max-seconds <=300 and timezone-aware deadline",
        )
        require(
            deadline <= datetime.fromisoformat(LATEST_DEADLINE),
            "Deadline cannot exceed the approved15:30UTC cutoff",
        )

        def check():
            if time.monotonic() - started >= max_seconds or datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Prefix-framing diagnostic deadline reached")

        check()
        config, prepared, frozen = admit(fidelity, data)
        planned = [
            dict(
                episode_id=item["episode_id"],
                task_id=item["task_id"],
                source_sha256=item["source_sha256"],
                old_tensor_sha256=item["old_tensor_sha256"],
                lesson=item["lesson"],
                lesson_sha256=digest(item["lesson"]),
                contexts=contexts(item["lesson"], item["generation"], item["feedback"]),
            )
            for item in prepared
        ]
        write_json(
            output / "plan.json",
            dict(
                work=WORK,
                generic_frames=dict(B=FRAME_B, C=FRAME_C),
                sources=planned,
                frozen_files=frozen,
            ),
        )
        plan_sha = sha(output / "plan.json")
        receipt.update(
            inherited_config=config,
            plan_sha256=plan_sha,
            frozen_files=frozen,
            source_boundary="No private answers parsed; only source prompt, unchanged lesson, "
            "original attempt and existing binary feedback construct contexts",
        )
        write_json(output / "receipt.json", receipt)
        check()
        configure_deterministic_execution(config["deterministic_execution"])
        if active is None:
            active = HFEngine(
                model_name=config["model_name"],
                revision=config["revision"],
                enable_thinking=config["enable_thinking"],
                budget_check=check,
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"],
            )
        active.budget_check = check
        require(
            active.counters == dict.fromkeys(WORK, 0)
            and (active.model_name, active.revision) == (config["model_name"], config["revision"]),
            "Require fresh unused exact-model engine",
        )
        for item, plan in zip(prepared, planned, strict=True):
            check()
            generation = item["generation"]
            require(
                generation.prompt_ids == active.encode_prompt(generation.prompt)
                and generation.text
                == active.tokenizer.decode(generation.response_ids, skip_special_tokens=True),
                "Original source native IDs differ from pinned tokenizer",
            )
            saved = torch.load(item["old_tensor_path"], map_location="cpu", weights_only=True)
            require(
                saved["response_ids"] == generation.response_ids
                and saved["prompt_ids"] == generation.prompt_ids,
                "Cached025 distribution native IDs differ from source",
            )
            old, initial = (
                distribution(saved[key], generation.response_ids) for key in ("target", "initial")
            )
            old_tokens, old_positions, _ = active._scoring_input(
                generation, item["feedback"], item["lesson"], "causal"
            )
            require(
                saved["alignment"]["teacher"]["input_ids"] == old_tokens
                and saved["alignment"]["teacher"]["predecessor_positions"] == old_positions,
                "Cached original PI alignment differs from unchanged core",
            )
            targets, alignment = {}, {}
            for arm, context in plan["contexts"].items():
                check()
                tokens, positions, future = active._scoring_input(
                    generation, item["feedback"], context, "causal"
                )
                require(
                    not future
                    and [tokens[pos + 1] for pos in positions] == generation.response_ids,
                    "New PI context changed original action/predecessor alignment",
                )
                alignment[arm] = dict(
                    input_ids=tokens,
                    predecessor_positions=positions,
                    input_ids_sha256=digest(tokens),
                    context_sha256=digest(context),
                )
                target = active.score_rollout(
                    generation, item["feedback"], pi=context, mode="causal", adapter=None
                )
                targets[arm] = distribution(target, generation.response_ids)
            path = output / "distributions" / f"{item['episode_id']}.pt"
            path.parent.mkdir(exist_ok=True)
            check()
            torch.save(
                dict(
                    format="source-prefix-framing-v1",
                    episode_id=item["episode_id"],
                    generation=dataclasses.asdict(generation),
                    feedback=item["feedback"],
                    unchanged_lesson=item["lesson"],
                    contexts=plan["contexts"],
                    response_ids=generation.response_ids,
                    alignment=alignment,
                    response_token_displays=saved["alignment"].get("response_token_displays"),
                    old_tensor_sha256=item["old_tensor_sha256"],
                    **targets,
                ),
                path,
            )
            check()
            metrics = compare(old, initial, targets, generation.response_ids)
            write_json(
                path.with_suffix(".json"),
                dict(
                    episode_id=item["episode_id"],
                    artifact=dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size),
                    metrics=metrics,
                ),
            )
            append_json(
                output / "source-metrics.jsonl", dict(episode_id=item["episode_id"], **metrics)
            )
            receipt["completed_episode_ids"].append(item["episode_id"])
            receipt.update(work=dict(active.counters))
            write_json(output / "receipt.json", receipt)
            del saved, targets, old, initial
        require(
            active.counters == WORK and len(receipt["completed_episode_ids"]) == 8,
            "Prefix diagnostic work differs from16 scores and zero updates/generation",
        )
        require(
            sha(output / "plan.json") == plan_sha
            and all(sha(path) == expected for path, expected in frozen.items()),
            "Frozen inputs/plan changed",
        )
        receipt.update(status="completed", frozen_inputs_unchanged=True)
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            work=dict(active.counters) if active is not None else dict.fromkeys(WORK, 0),
        )
        write_json(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fidelity", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--deadline-utc", default=LATEST_DEADLINE)
    args = parser.parse_args()
    return run(
        args.fidelity,
        args.data,
        args.output,
        max_seconds=args.max_seconds,
        deadline_utc=args.deadline_utc,
    )


if __name__ == "__main__":
    raise SystemExit(main())
