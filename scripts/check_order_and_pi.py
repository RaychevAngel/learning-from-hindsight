"""Frozen 023 execution-order control, gated before fixed-lesson direct PI."""

import argparse
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import grade
from hindsight.experiment import (
    Experiment,
    append_json,
    digest,
    observation_key,
    read_json,
    rows,
    write_json,
)
from hindsight.model import with_pi

ORDER_IDS = ("208831d7e726ae9462d2bb3a", "eb2e08ce9a30ca1da12fc3f8")
LAYOUTS = ("original", "reverse", "interleaved")
PLANNED_WORK = dict(
    generation_sequences=224,
    scoring_forwards=2,
    student_optimizer_steps=8,
    teacher_optimizer_steps=0,
    teacher_example_exposures=0,
)


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADMISSION = helper("check_hindsight_teacher")
COMPARISON = helper("check_batch_invariance")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_order(ep, layout):
    original = [("V", task) for task in ep.validation] + [("Q", task) for task in ep.queries]
    if layout == "original":
        return original
    if layout == "reverse":
        return original[::-1]
    if layout == "interleaved":
        return [
            pair
            for v, q in zip(ep.validation, ep.queries, strict=True)
            for pair in (("V", v), ("Q", q))
        ]
    raise ValueError("Unknown frozen order")


def state_digest(engine, *, base=False, check=None):
    """Hash all student values, buffers and optimizer state without serialization noise."""
    hasher = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            if check is not None:
                check()
            tensor = value.detach().cpu().contiguous()
            hasher.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                hasher.update(repr(key).encode())
                update(value[key])
        elif isinstance(value, (list, tuple)):
            for entry in value:
                update(entry)
        else:
            hasher.update(repr(value).encode())

    if base:
        update(
            {name: value for name, value in engine.model.named_parameters() if "lora_" not in name}
        )
    else:
        update(engine._adapter_parameters("student"))
        update(dict(engine.model.named_buffers()))
        update(engine._student_optimizer.state_dict())
    return hasher.hexdigest()


def frozen_files(sources, reference, lessons, data):
    paths = [
        sources / "identity.json",
        sources / "receipt.json",
        lessons,
        reference / "identity.json",
        reference / "receipt.json",
        reference / "lessons.json",
        reference / "observations.jsonl",
        reference / "expected.jsonl",
        data / "manifest.json",
    ]
    paths += [data / name for name in read_json(data / "manifest.json")["files"]]
    for ep_id in read_json(sources / "identity.json")["config"]["diagnostic_source_ids"]:
        paths += [folder / "sources" / f"{ep_id}.json" for folder in (sources, reference)]
    core = Path(__file__).resolve().parents[1] / "src/hindsight"
    paths += [core / name for name in ("model.py", "experiment.py", "data.py")]
    paths += [
        Path(__file__).with_name(name)
        for name in (
            "check_order_and_pi.py",
            "check_hindsight_teacher.py",
            "check_batch_invariance.py",
        )
    ]
    return {str(path.resolve()): sha(path) for path in paths}


def verify_frozen(snapshot):
    if any(sha(path) != expected for path, expected in snapshot.items()):
        raise ValueError("Frozen source, lesson, corpus, reference or core bytes changed")


def admit(sources, reference, lessons, data):
    parent, prepared = ADMISSION.admit(sources, lessons, data)
    previous = read_json(reference / "identity.json")
    receipt = read_json(reference / "receipt.json")
    source_receipt = read_json(sources / "receipt.json")
    inherited = dict(previous["config"])
    provenance = inherited.pop("hindsight_comparison")
    ids = [ep.episode_id for ep, *_ in prepared]
    if (
        inherited != parent["config"]
        or source_receipt["status"] != "completed"
        or source_receipt["actual_sources"] != 8
        or receipt["status"] != "completed"
        or receipt["actual_observations"] != 512
        or receipt["completed_episode_ids"] != ids
        or previous["dataset_manifest"] != parent["dataset_manifest"]
        or provenance["source_identity_sha256"] != sha(sources / "identity.json")
        or provenance["lessons_sha256"] != sha(lessons)
        or receipt["lessons_sha256"] != sha(lessons)
        or provenance["evaluation_episode_ids"] != ids
        or lessons.read_bytes() != (reference / "lessons.json").read_bytes()
        or inherited.get("deterministic_execution") is not True
        or inherited.get("generation_batch_size") != 1
        or inherited.get("student_steps") != 4
        or inherited.get("learning_rate") != 1e-4
    ):
        raise ValueError("Reference/source/config/lesson identity differs from completed 023")
    code = Experiment.code_identity(None)
    for name in ("model.py", "experiment.py", "data.py"):
        if previous["code"][name] != code[name] or parent["code"][name] != code[name]:
            raise ValueError("Core code differs from source022/reference023")
    for ep, *_ in prepared:
        name = f"{ep.episode_id}.json"
        if (sources / "sources" / name).read_bytes() != (reference / "sources" / name).read_bytes():
            raise ValueError("Reference and source native generation bytes differ")
    selected = [entry for ep_id in ORDER_IDS for entry in prepared if entry[0].episode_id == ep_id]
    if len(selected) != 2 or [ep.family for ep, *_ in selected] != ["denoising", "size_recoloring"]:
        raise ValueError("Require the two declared diagnostic failure sources")
    observed, expected = rows(reference / "observations.jsonl"), rows(reference / "expected.jsonl")
    if (
        len(observed) != 512
        or len(expected) != 512
        or len({observation_key(row) for row in observed}) != 512
        or {observation_key(row) for row in observed} != {observation_key(row) for row in expected}
    ):
        raise ValueError("Require exactly the complete 512 reference observations")
    wanted = {
        COMPARISON.key(Experiment.metadata(None, ep, task, role, "adapted_pi", 0))
        for ep, *_ in selected
        for role, task in task_order(ep, "original")
    }
    old = {}
    for row in observed:
        if row["episode_id"] not in ORDER_IDS or row["condition"] != "adapted_pi":
            continue
        key = COMPARISON.key(row)
        if (
            key not in wanted
            or key in old
            or row["candidate"] != 0
            or row["partition"] != "development"
        ):
            raise ValueError("Invalid reference adapted-PI join")
        old[key] = dict(
            generation={name: row[name] for name in ("text", "cap_hit")},
            grade={name: row[name] for name in ("success", "format_ok", "prediction")},
            generated_tokens=row["generated_tokens"],
        )
    if set(old) != wanted or len(old) != 32:
        raise ValueError("Missing frozen reference tasks")
    return previous["config"], prepared, selected, old


def run(
    sources, reference, lessons, data, output, *, max_seconds=1800, deadline_utc=None, engine=None
):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        started_at=datetime.now(timezone.utc).isoformat(),
        planned_work=PLANNED_WORK,
        expected_records=224,
        actual_records=0,
        order_gate=None,
        direct_pi_started=False,
        order_source_ids=list(ORDER_IDS),
        selection="Two known failure sources; diagnostic only, not an effect estimate",
        old_reference_limit="023 has text, grade, cap and length; no native evaluation token IDs",
        script_sha256=sha(__file__),
        source_generation_sequences=0,
        proposal_generation_sequences=0,
    )
    write_json(output / "receipt.json", receipt)
    experiment, generated, snapshot = None, [], None
    try:
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800:
            raise ValueError("Require positive --max-seconds <= 1800")
        inherited, prepared, selected, old = admit(sources, reference, lessons, data)
        snapshot = frozen_files(sources, reference, lessons, data)
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
        )
        experiment = Experiment(config, data, output, engine=engine)
        original_check = experiment.check_time

        def check_time():
            original_check()
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Order/direct-PI diagnostic invocation deadline reached")

        experiment.check_time = check_time
        plan = [
            (ep, lesson, layout, "adapted_pi", role, task)
            for ep, _, _, lesson in selected
            for layout in LAYOUTS
            for role, task in task_order(ep, layout)
        ]
        plan += [
            (ep, lesson, "direct_pi", "direct_pi", role, task)
            for ep, _, _, lesson in prepared
            for role, task in task_order(ep, "original")
        ]
        write_json(
            output / "plan.json",
            dict(
                planned_work=PLANNED_WORK,
                layouts=list(LAYOUTS),
                generations=[
                    dict(
                        sequence=i,
                        layout=layout,
                        **experiment.metadata(ep, task, role, condition, 0),
                    )
                    for i, (ep, _, layout, condition, role, task) in enumerate(plan)
                ],
            ),
        )
        (output / "lessons.json").write_bytes(lessons.read_bytes())
        write_json(output / "receipt.json", receipt)
        check_time()
        active = experiment.get_engine()
        if active.counters != dict.fromkeys(PLANNED_WORK, 0):
            raise ValueError("Require a fresh unused engine")
        receipt["generation_cache_implementation"] = (
            active.model.generation_config.cache_implementation
        )
        receipt["cache_evidence_limit"] = (
            "Existing generate API supplies no past cache; "
            "cache object identities were not instrumented"
        )
        receipt["base_before_order_sha256"] = state_digest(active, base=True, check=check_time)

        def generate_one(ep, lesson, layout, condition, role, task):
            prompt = with_pi(task.prompt, lesson) if condition == "direct_pi" else task.prompt
            value = experiment.generate(
                [prompt],
                max_new_tokens=config["evaluation_tokens"],
                sample=False,
                seed=config["seed"],
                adapter=None if condition == "direct_pi" else "student",
            )[0]
            if value.prompt != prompt:
                raise ValueError("Returned generation prompt differs from requested prompt")
            record = dict(
                **experiment.metadata(ep, task, role, condition, 0),
                sequence=len(generated),
                layout=layout,
                generation=dataclasses.asdict(value),
                grade=grade(value.text, task),
                generated_tokens=len(value.response_ids),
            )
            append_json(output / "generations.jsonl", record)
            generated.append(record)
            receipt.update(actual_records=len(generated), work=dict(active.counters))
            write_json(output / "receipt.json", receipt)

        for ep, generation, feedback, lesson in selected:
            verify_frozen(snapshot)
            target = experiment.score(generation, feedback, pi=lesson)
            experiment.adaptation(ep, generation, target, "adapted_pi", 0)
            del target
            frozen = state_digest(active, check=check_time)
            for layout in LAYOUTS:
                check_time()
                before = state_digest(active, check=check_time)
                counts = dict(active.counters)
                if before != frozen:
                    raise ValueError("Student state changed between frozen layouts")
                for role, task in task_order(ep, layout):
                    generate_one(ep, lesson, layout, "adapted_pi", role, task)
                after = state_digest(active, check=check_time)
                state_row = dict(
                    episode_id=ep.episode_id,
                    layout=layout,
                    before_sha256=before,
                    after_sha256=after,
                    unchanged=before == after == frozen,
                )
                append_json(output / "student-state.jsonl", state_row)
                if not state_row["unchanged"]:
                    raise ValueError("Student state mutated during frozen generation layout")
                if any(
                    active.counters[key] != value
                    for key, value in counts.items()
                    if key != "generation_sequences"
                ):
                    raise ValueError("Unexpected optimization/scoring during a frozen layout")
        receipt["base_after_order_sha256"] = state_digest(active, base=True, check=check_time)
        if receipt["base_before_order_sha256"] != receipt["base_after_order_sha256"]:
            raise ValueError("Frozen base parameters changed during order control")
        grouped = {
            layout: {COMPARISON.key(row): row for row in generated if row["layout"] == layout}
            for layout in LAYOUTS
        }
        fresh = [
            COMPARISON.compare(a, b, grouped[a], grouped[b])
            for a, b in itertools.combinations(LAYOUTS, 2)
        ]
        replay = [
            COMPARISON.compare("saved023", layout, old, grouped[layout], tokens=False)
            for layout in LAYOUTS
        ]
        for layout, comparison in zip(LAYOUTS, replay, strict=True):
            length_mismatches = [
                list(key)
                for key in old
                if old[key]["generated_tokens"] != grouped[layout][key]["generated_tokens"]
            ]
            comparison.update(
                generated_tokens_mismatches=length_mismatches,
                invariant=comparison["invariant"] and not length_mismatches,
            )
        write_json(output / "mismatch-matrix.json", dict(fresh_pairwise=fresh, saved_replay=replay))
        receipt["current_order_invariance"] = all(row["invariant"] for row in fresh)
        receipt["history_reproduction"] = all(row["invariant"] for row in replay)
        receipt["order_gate"] = (
            receipt["current_order_invariance"] and receipt["history_reproduction"]
        )
        verify_frozen(snapshot)
        if not receipt["order_gate"]:
            raise ValueError("Order or saved023 replay mismatch; direct PI was not started")
        expected_order = {**PLANNED_WORK, "generation_sequences": 96}
        if active.counters != expected_order or len(generated) != 96:
            raise ValueError("Order control actual work differs from fixed 96/2/8 plan")
        receipt["direct_pi_started"] = True
        write_json(output / "receipt.json", receipt)
        for ep, _, _, lesson in prepared:
            for role, task in task_order(ep, "original"):
                generate_one(ep, lesson, "direct_pi", "direct_pi", role, task)
        verify_frozen(snapshot)
        if active.counters != PLANNED_WORK or len(generated) != 224:
            raise ValueError("Actual work differs from fixed 224/2/8 plan")
        if [
            (row["episode_id"], row["layout"], row["condition"], row["role"], row["task_id"])
            for row in generated
        ] != [
            (ep.episode_id, layout, condition, role, task.task_id)
            for ep, _, layout, condition, role, task in plan
        ]:
            raise ValueError("Generated identities/order differ from the frozen plan")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        active = experiment.engine if experiment is not None else engine
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            actual_records=len(generated),
            missing_records=224 - len(generated),
            elapsed_seconds=time.monotonic() - started,
            work=dict(active.counters) if active is not None else dict.fromkeys(PLANNED_WORK, 0),
        )
        if snapshot is not None:
            try:
                verify_frozen(snapshot)
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
    parser.add_argument("--max-seconds", type=float, default=1800)
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
