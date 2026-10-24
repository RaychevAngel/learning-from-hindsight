"""Reconstruct all eight fixed 023 PI students and audit source-distribution fitting."""

import argparse
import importlib.util
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from hindsight.experiment import Experiment, append_json, digest, rows, write_json

SPEC = importlib.util.spec_from_file_location(
    "check_order_and_pi", Path(__file__).with_name("check_order_and_pi.py")
)
FROZEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FROZEN)
PLANNED_WORK = dict(
    generation_sequences=0,
    scoring_forwards=16,
    student_optimizer_steps=32,
    teacher_optimizer_steps=0,
    teacher_example_exposures=0,
)
METRIC_FIELDS = {"step", "loss", "grad_norm", "lr", "optimizer_steps", "example_exposures"}


def numeric_metrics(metrics):
    """Exclude only elapsed time; require the exact original four-step numeric record."""
    if not isinstance(metrics, list) or len(metrics) != 4:
        raise ValueError("Require exactly four historical adaptation metric rows")
    normalized = [
        {key: value for key, value in row.items() if key != "elapsed_seconds"} for row in metrics
    ]
    for index, row in enumerate(normalized):
        if (
            set(row) != METRIC_FIELDS
            or row["step"] != index + 1
            or row["optimizer_steps"] != 1
            or row["example_exposures"] != 1
            or row["lr"] != 1e-4
            or any(
                type(value) not in (float, int) or not math.isfinite(value)
                for value in row.values()
            )
        ):
            raise ValueError("Invalid historical four-step numeric metrics")
    return normalized


def reference_metrics(reference, episode_ids):
    selected = [
        row
        for row in rows(reference / "events.jsonl")
        if row.get("event") == "student_adapted" and row.get("condition") == "adapted_pi"
    ]
    if (
        len(selected) != 8
        or [row["episode_id"] for row in selected] != episode_ids
        or any(row.get("candidate") != 0 for row in selected)
    ):
        raise ValueError("Require one ordered historical PI adaptation for all eight sources")
    return {row["episode_id"]: numeric_metrics(row["metrics"]) for row in selected}


def detached_distribution(value, response_ids):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("Expected a full-vocabulary log-probability matrix")
    value = value.detach().cpu().float().contiguous()
    if (
        value.shape[0] != len(response_ids)
        or not response_ids
        or min(response_ids) < 0
        or max(response_ids) >= value.shape[1]
        or not torch.isfinite(value).all()
        or not torch.allclose(value.logsumexp(-1), torch.zeros(len(response_ids)), atol=1e-4)
    ):
        raise ValueError("Invalid or unaligned source log-probability distribution")
    return value


def fidelity_metrics(target, initial, post, response_ids):
    """FP32 full-vocabulary distances at every original response-token position."""
    if target.shape != initial.shape or target.shape != post.shape:
        raise ValueError("Initial, target and post-fourth distributions must align exactly")
    position_rows = []
    # Bound temporary CPU memory independently of source response length.
    for offset in range(0, len(response_ids), 8):
        teacher, before, after = (value[offset : offset + 8] for value in (target, initial, post))
        actions = torch.tensor(response_ids[offset : offset + 8]).unsqueeze(1)
        teacher_mass = teacher.exp()
        metrics = {"teacher_observed_logprob": teacher.gather(1, actions).squeeze(1)}
        for name, student in (("initial", before), ("post_fourth", after)):
            student_mass = student.exp()
            midpoint_log = torch.logaddexp(teacher, student) - math.log(2)
            metrics.update(
                {
                    f"{name}_observed_logprob": student.gather(1, actions).squeeze(1),
                    f"{name}_forward_kl_teacher_to_student": F.kl_div(
                        student, teacher_mass, reduction="none"
                    ).sum(-1),
                    f"{name}_reverse_kl_student_to_teacher": F.kl_div(
                        teacher, student_mass, reduction="none"
                    ).sum(-1),
                    f"{name}_total_variation": (teacher_mass - student_mass).abs().sum(-1) / 2,
                    f"{name}_argmax_agreement": (teacher.argmax(-1) == student.argmax(-1)).float(),
                    f"{name}_jensen_shannon": (
                        F.kl_div(midpoint_log, teacher_mass, reduction="none").sum(-1)
                        + F.kl_div(midpoint_log, student_mass, reduction="none").sum(-1)
                    )
                    / 2,
                }
            )
        for index, token in enumerate(response_ids[offset : offset + 8]):
            position_rows.append(
                dict(
                    response_position=offset + index,
                    response_token_id=token,
                    **{key: float(value[index]) for key, value in metrics.items()},
                )
            )
    names = [
        key for key in position_rows[0] if key not in ("response_position", "response_token_id")
    ]
    means = {key: sum(row[key] for row in position_rows) / len(position_rows) for key in names}
    means["forward_kl_reduction"] = (
        means["initial_forward_kl_teacher_to_student"]
        - means["post_fourth_forward_kl_teacher_to_student"]
    )
    return dict(
        action_positions=len(response_ids),
        vocabulary_size=target.shape[1],
        source_means=means,
        positions=position_rows,
        aggregation="Positions are correlated within one source, not independent samples",
    )


def save_distributions(path, generation, *, alignment, **distributions):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format="source-fidelity-full-vocabulary-logprobs-v1",
        prompt_ids=generation.prompt_ids,
        response_ids=generation.response_ids,
        alignment=alignment,
        **distributions,
    )
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return dict(
        path=str(path.name),
        sha256=FROZEN.sha(path),
        bytes=path.stat().st_size,
        distributions=list(distributions),
        dtype="torch.float32",
    )


def run(
    sources, reference, lessons, data, output, *, max_seconds=300, deadline_utc=None, engine=None
):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        started_at=datetime.now(timezone.utc).isoformat(),
        planned_work=PLANNED_WORK,
        expected_sources=8,
        completed_episode_ids=[],
        actual_sources=0,
        source_generation_sequences=0,
        proposal_generation_sequences=0,
        script_sha256=FROZEN.sha(__file__),
        initial_distribution="Existing step-one callback before the first optimizer update",
        post_distribution="Explicit causal student score after the fourth optimizer update",
        historical_losses="023 step losses describe states before updates 1, 2, 3 and 4",
        selection="All eight frozen development sources in their existing order",
    )
    write_json(output / "receipt.json", receipt)
    experiment, snapshot = None, None
    try:
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 300:
            raise ValueError("Require positive --max-seconds <= 300")
        inherited, prepared, _, _ = FROZEN.admit(sources, reference, lessons, data)
        episode_ids = [ep.episode_id for ep, *_ in prepared]
        historical = reference_metrics(reference, episode_ids)
        snapshot = FROZEN.frozen_files(sources, reference, lessons, data)
        snapshot.update(
            {
                str(path.resolve()): FROZEN.sha(path)
                for path in (reference / "events.jsonl", Path(__file__))
            }
        )
        config = dict(inherited)
        overrides = {"max_invocation_seconds": max_seconds}
        if deadline_utc is not None:
            deadline = datetime.fromisoformat(deadline_utc)
            if deadline.tzinfo is None:
                raise ValueError("Deadline override must include timezone")
            overrides["absolute_deadline_utc"] = deadline.isoformat()
        config.update(overrides)
        receipt.update(
            inherited_config=inherited,
            inherited_config_sha256=digest(inherited),
            runtime_overrides=overrides,
            frozen_files=snapshot,
            episode_ids=episode_ids,
        )
        experiment = Experiment(config, data, output, engine=engine)
        original_check = experiment.check_time

        def check_time():
            original_check()
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Source-fidelity diagnostic invocation deadline reached")

        experiment.check_time = check_time
        plan = [
            dict(
                episode_id=ep.episode_id,
                task_id=ep.source.task_id,
                family=ep.family,
                source_sha256=FROZEN.sha(sources / "sources" / f"{ep.episode_id}.json"),
                lesson_sha256=digest(lesson),
                action_positions=len(generation.response_ids),
                historical_metrics=historical[ep.episode_id],
            )
            for ep, generation, _, lesson in prepared
        ]
        write_json(output / "plan.json", dict(planned_work=PLANNED_WORK, sources=plan))
        (output / "lessons.json").write_bytes(lessons.read_bytes())
        for ep, *_ in prepared:
            path = output / "sources" / f"{ep.episode_id}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((sources / "sources" / path.name).read_bytes())
        write_json(output / "receipt.json", receipt)
        active = experiment.get_engine()
        if active.counters != dict.fromkeys(PLANNED_WORK, 0):
            raise ValueError("Require a fresh unused engine")
        for entry, (ep, generation, feedback, lesson) in zip(plan, prepared, strict=True):
            receipt["current_episode_id"] = ep.episode_id
            FROZEN.verify_frozen(snapshot)
            alignment = {
                "response_token_displays": active.tokenizer.convert_ids_to_tokens(
                    generation.response_ids
                )
            }
            for name, pi in (("teacher", lesson), ("student", None)):
                tokens, positions, bidirectional = active._scoring_input(
                    generation, "", pi, "causal"
                )
                if (
                    bidirectional
                    or [tokens[position + 1] for position in positions] != generation.response_ids
                ):
                    raise ValueError(
                        "Source predecessor positions do not align with original response IDs"
                    )
                alignment[name] = dict(
                    input_ids=tokens,
                    predecessor_positions=positions,
                    input_ids_sha256=digest(tokens),
                )
            target = detached_distribution(
                experiment.score(generation, feedback, pi=lesson), generation.response_ids
            )
            initial = []
            experiment.adaptation(
                ep, generation, target, "adapted_pi", 0, initial_logprobs_callback=initial.append
            )
            if len(initial) != 1:
                raise ValueError("Missing unique initial causal distribution callback")
            initial = detached_distribution(initial[0], generation.response_ids)
            events = [
                row
                for row in rows(output / "events.jsonl")
                if row.get("event") == "student_adapted" and row.get("episode_id") == ep.episode_id
            ]
            if len(events) != 1:
                raise ValueError("Missing unique reconstruction metrics")
            actual = numeric_metrics(events[0]["metrics"])
            matches = actual == historical[ep.episode_id]
            comparison = dict(
                episode_id=ep.episode_id,
                identical=matches,
                historical_metrics=historical[ep.episode_id],
                actual_metrics=actual,
                excluded_fields=["elapsed_seconds"],
            )
            append_json(output / "reconstruction.jsonl", comparison)
            artifact = output / "distributions" / f"{ep.episode_id}.pt"
            if not matches:
                check_time()
                partial = save_distributions(
                    artifact, generation, alignment=alignment, target=target, initial=initial
                )
                write_json(
                    artifact.with_suffix(".json"),
                    dict(
                        **entry,
                        artifact=partial,
                        status="reconstruction_mismatch",
                        reconstruction=comparison,
                    ),
                )
                raise ValueError("Reconstructed numeric step metrics differ from 023; no retry")
            before_score = FROZEN.state_digest(active, check=check_time)
            post = detached_distribution(
                experiment.score(generation, "", mode="causal", adapter="student"),
                generation.response_ids,
            )
            after_score = FROZEN.state_digest(active, check=check_time)
            append_json(
                output / "student-state.jsonl",
                dict(
                    episode_id=ep.episode_id,
                    before_sha256=before_score,
                    after_sha256=after_score,
                    unchanged=before_score == after_score,
                ),
            )
            if before_score != after_score:
                raise ValueError("Student, buffer or optimizer state changed during final scoring")
            if target.shape != initial.shape or target.shape != post.shape:
                raise ValueError("Initial, target and post distributions do not align")
            check_time()
            saved = save_distributions(
                artifact,
                generation,
                alignment=alignment,
                target=target,
                initial=initial,
                post_fourth=post,
            )
            check_time()
            metrics = fidelity_metrics(target, initial, post, generation.response_ids)
            source_result = dict(
                **entry,
                status="completed",
                artifact=saved,
                metrics=metrics,
                reconstruction_identical=True,
            )
            write_json(artifact.with_suffix(".json"), source_result)
            append_json(
                output / "source-metrics.jsonl",
                {
                    **entry,
                    "artifact": saved,
                    **{key: value for key, value in metrics.items() if key != "positions"},
                },
            )
            del target, initial, post
            receipt["completed_episode_ids"].append(ep.episode_id)
            receipt.update(
                actual_sources=len(receipt["completed_episode_ids"]), work=dict(active.counters)
            )
            write_json(output / "receipt.json", receipt)
            check_time()
        FROZEN.verify_frozen(snapshot)
        if receipt["completed_episode_ids"] != episode_ids or active.counters != PLANNED_WORK:
            raise ValueError("Actual source identities or work differ from frozen 8/16/32/0 plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        active = experiment.engine if experiment is not None else engine
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            actual_sources=len(receipt["completed_episode_ids"]),
            missing_sources=8 - len(receipt["completed_episode_ids"]),
            elapsed_seconds=time.monotonic() - started,
            work=dict(active.counters) if active is not None else dict.fromkeys(PLANNED_WORK, 0),
        )
        if snapshot is not None:
            try:
                FROZEN.verify_frozen(snapshot)
                receipt["frozen_inputs_unchanged"] = True
            except Exception as error:
                receipt.update(
                    status="failed",
                    frozen_inputs_unchanged=False,
                    integrity_error=dict(type=type(error).__name__, message=str(error)),
                )
        write_json(output / "receipt.json", receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "reference", "lessons", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    return run(
        args.sources,
        args.reference,
        args.lessons,
        args.data,
        args.output,
        max_seconds=args.max_seconds,
        deadline_utc=args.deadline_utc,
    )


if __name__ == "__main__":
    raise SystemExit(main())
