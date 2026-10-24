"""Matched V-only transfer from cached025/029 targets; no model target scoring."""

import argparse
import dataclasses
import importlib.util
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import grade, load_corpus
from hindsight.experiment import append_json, digest, read_json, rows, seed_for, write_json
from hindsight.model import HFEngine, configure_deterministic_execution


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFIX = helper("score_prefix_interventions")
FROZEN = helper("check_order_and_pi")
sha, require = PREFIX.sha, PREFIX.require
ARMS = ("baseline", "old_PI", "B", "C")
WORK = dict(
    generation_sequences=256,
    scoring_forwards=0,
    student_optimizer_steps=384,
    teacher_optimizer_steps=0,
    teacher_example_exposures=0,
)


def admit(config, data, fidelity, framing):
    inherited, prepared, frozen = PREFIX.admit(fidelity, data)
    prior, plan = read_json(framing / "receipt.json"), read_json(framing / "plan.json")
    ids = [item["episode_id"] for item in prepared]
    require(
        prior["status"] == "completed"
        and prior["frozen_inputs_unchanged"]
        and prior["work"] == PREFIX.WORK
        and prior["completed_episode_ids"] == ids
        and prior["inherited_config"] == inherited
        and prior["plan_sha256"] == sha(framing / "plan.json")
        and plan["generic_frames"] == dict(B=PREFIX.FRAME_B, C=PREFIX.FRAME_C)
        and plan["work"] == PREFIX.WORK
        and [row["episode_id"] for row in plan["sources"]] == ids
        and set(frozen.values()) <= set(prior["frozen_files"].values()),
        "Require exact completed029 lineage from the same frozen025 inputs",
    )
    for key in (
        "model_name",
        "revision",
        "seed",
        "enable_thinking",
        "deterministic_execution",
        "max_context_tokens",
        "max_teacher_context_tokens",
        "evaluation_tokens",
    ):
        require(config[key] == inherited[key], "Transfer scientific configuration differs from025")
    require(
        config["generation_batch_size"] == 8
        and config["student_steps"] == 16
        and config["learning_rate"] == 1e-4
        and config.get("loss", "forward_kl") == "forward_kl",
        "Transfer fixes B8, sixteen updates, learning rate1e-4 and full forward KL",
    )
    if "dataset_manifest_sha256" in config:
        require(config["dataset_manifest_sha256"] == sha(data / "manifest.json"), "Wrong dataset")
    qualification = config["batch_qualification"]
    path = Path(qualification["path"])
    qualified = read_json(path)
    require(
        sha(path) == qualification["sha256"]
        and qualified["status"] == "completed"
        and qualified["selected_batch_size"] == 8
        and qualified["repeat_stability"]["8"]
        and all(qualified["config"][key] == config[key] for key in ("model_name", "revision")),
        "Require pinned same-model repeat-stable B8 qualification",
    )
    episodes, _ = load_corpus(data)
    by_id = {ep.episode_id: ep for ep in episodes}
    require(set(by_id) == set(ids), "Require the unchanged eight-source development corpus")
    paths = [framing / name for name in ("receipt.json", "plan.json")]
    paths += [path, Path(__file__), Path(FROZEN.__file__)]
    for item, entry in zip(prepared, plan["sources"], strict=True):
        ep = by_id[item["episode_id"]]
        expected_contexts = PREFIX.contexts(item["lesson"], item["generation"], item["feedback"])
        require(
            ep.partition == "development"
            and len(ep.validation) == 8
            and ep.source.task_id == item["task_id"]
            and ep.source.prompt == item["generation"].prompt
            and entry["task_id"] == item["task_id"]
            and entry["source_sha256"] == item["source_sha256"]
            and entry["old_tensor_sha256"] == item["old_tensor_sha256"]
            and entry["lesson"] == item["lesson"]
            and entry["lesson_sha256"] == digest(item["lesson"])
            and entry["contexts"] == expected_contexts,
            "Source, unchanged lesson or deterministic framing identity differs",
        )
        path = framing / "distributions" / f"{ep.episode_id}.pt"
        sidecar = read_json(path.with_suffix(".json"))
        require(
            sidecar["episode_id"] == ep.episode_id and sidecar["artifact"]["sha256"] == sha(path),
            "Framing target artifact checksum differs",
        )
        item.update(
            episode=ep, contexts=expected_contexts, framing_path=path, framing_sha256=sha(path)
        )
        paths += [path, path.with_suffix(".json")]
    frozen.update({str(path.resolve()): sha(path) for path in paths})
    return inherited, prepared, frozen, qualified.get("dataset_sha256")


def targets(item, engine):
    generation = item["generation"]
    require(
        generation.prompt_ids == engine.encode_prompt(generation.prompt)
        and generation.text
        == engine.tokenizer.decode(generation.response_ids, skip_special_tokens=True),
        "Frozen source native IDs differ from exact tokenizer",
    )
    old = torch.load(item["old_tensor_path"], weights_only=True, map_location="cpu")
    framed = torch.load(item["framing_path"], weights_only=True, map_location="cpu")
    require(
        old["prompt_ids"] == generation.prompt_ids
        and old["response_ids"] == generation.response_ids
        and framed["format"] == "source-prefix-framing-v1"
        and framed["episode_id"] == item["episode_id"]
        and framed["generation"] == dataclasses.asdict(generation)
        and framed["response_ids"] == generation.response_ids
        and framed["feedback"] == item["feedback"]
        and framed["unchanged_lesson"] == item["lesson"]
        and framed["old_tensor_sha256"] == item["old_tensor_sha256"]
        and framed["contexts"] == item["contexts"],
        "Cached target/native source lineage differs",
    )
    contexts = dict(old_PI=item["lesson"], **item["contexts"])
    for arm, context in contexts.items():
        tokens, positions, future = engine._scoring_input(
            generation, item["feedback"], context, "causal"
        )
        alignment = old["alignment"]["teacher"] if arm == "old_PI" else framed["alignment"][arm]
        require(
            not future
            and alignment["input_ids"] == tokens
            and alignment["predecessor_positions"] == positions
            and [tokens[position + 1] for position in positions] == generation.response_ids,
            "Cached target action/predecessor alignment differs",
        )
    result = {"old_PI": PREFIX.distribution(old["target"], generation.response_ids)}
    result.update(
        {arm: PREFIX.distribution(framed[arm], generation.response_ids) for arm in ("B", "C")}
    )
    require(
        len({tuple(value.shape) for value in result.values()}) == 1, "Target vocabularies differ"
    )
    return result


def run(config, data, fidelity, framing, output, *, engine=None):
    output.mkdir(parents=True, exist_ok=False)
    started, active = time.monotonic(), engine
    receipt = dict(
        status="started",
        config=config,
        planned_work=WORK,
        completed_episode_ids=[],
        hidden_retries=0,
        selection="None; report all four fixed arms on development V only",
    )
    write_json(output / "receipt.json", receipt)
    try:
        seconds, deadline = (
            config["max_invocation_seconds"],
            datetime.fromisoformat(config["absolute_deadline_utc"]),
        )
        require(
            math.isfinite(seconds)
            and 0 < seconds <= 1800
            and deadline.tzinfo is not None
            and deadline <= datetime.fromisoformat(PREFIX.LATEST_DEADLINE),
            "Require positive runtime <=1800 seconds and deadline no later than15:30UTC",
        )

        def check():
            if time.monotonic() - started >= seconds or datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Prefix transfer diagnostic deadline reached")

        check()
        inherited, prepared, frozen, qualified_dataset = admit(config, data, fidelity, framing)
        expected = [
            dict(episode_id=item["episode_id"], task_id=task.task_id, condition=arm, role="V")
            for item in prepared
            for arm in ARMS
            for task in item["episode"].validation
        ]
        plan = dict(
            work=WORK,
            expected=expected,
            sources=[
                dict(
                    episode_id=item["episode_id"],
                    source_sha256=item["source_sha256"],
                    generation=dataclasses.asdict(item["generation"]),
                    old_tensor_sha256=item["old_tensor_sha256"],
                    framing_sha256=item["framing_sha256"],
                    lesson_sha256=digest(item["lesson"]),
                    context_sha256={arm: digest(value) for arm, value in item["contexts"].items()},
                )
                for item in prepared
            ],
        )
        write_json(output / "plan.json", plan)
        frozen[str((output / "plan.json").resolve())] = sha(output / "plan.json")
        receipt.update(
            inherited_config=inherited,
            frozen_files=frozen,
            runtime_overrides={
                key: config[key] for key in ("max_invocation_seconds", "absolute_deadline_utc")
            },
            batch_qualification_dataset_sha256=qualified_dataset,
            qualification_scope="Same-model B8 repeat stability; "
            "not singleton or old-corpus equivalence",
            source_boundary="Cached targets use only original source, fixed lesson and binary "
            "feedback; "
            "no source-gold masking or selection; only V public prompts enter new generation",
        )
        write_json(output / "receipt.json", receipt)
        configure_deterministic_execution(config["deterministic_execution"])
        check()
        if active is None:
            active = HFEngine(
                config["model_name"],
                config["revision"],
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
        initial_hash = None
        for item in prepared:
            check()
            ep, cached = item["episode"], targets(item, active)
            for arm in ARMS:
                check()
                metrics = []
                if arm != "baseline":
                    active.reset_student(seed_for(config["seed"], "student_initialization"))
                    before = FROZEN.state_digest(active, check=check)
                    initial_hash = initial_hash or before
                    require(
                        before == initial_hash,
                        "Student/optimizer reset differs across arms/sources",
                    )
                    metrics = active.adapt(item["generation"], cached[arm], 16, 1e-4)
                    require(
                        len(metrics) == 16
                        and [row["step"] for row in metrics] == list(range(1, 17))
                        and all(
                            row["optimizer_steps"] == 1
                            and row["lr"] == 1e-4
                            and math.isfinite(row["loss"])
                            and math.isfinite(row["grad_norm"])
                            for row in metrics
                        ),
                        "Incomplete or nonfinite fixed sixteen-step adaptation",
                    )
                    state = FROZEN.state_digest(active, check=check)
                generated = active.generate(
                    [task.prompt for task in ep.validation],
                    max_new_tokens=config["evaluation_tokens"],
                    sample=False,
                    seed=config["seed"],
                    adapter=None if arm == "baseline" else "student",
                )
                if arm != "baseline":
                    require(
                        state == FROZEN.state_digest(active, check=check),
                        "V generation mutated frozen student state",
                    )
                    append_json(
                        output / "adaptations.jsonl",
                        dict(
                            episode_id=ep.episode_id,
                            condition=arm,
                            metrics=metrics,
                            initial_student_sha256=before,
                            frozen_student_sha256=state,
                            source_sha256=item["source_sha256"],
                            target_artifact_sha256=item["old_tensor_sha256"]
                            if arm == "old_PI"
                            else item["framing_sha256"],
                        ),
                    )
                for task, value in zip(ep.validation, generated, strict=True):
                    require(
                        value.prompt == task.prompt
                        and value.prompt_ids == active.encode_prompt(task.prompt)
                        and value.text
                        == active.tokenizer.decode(value.response_ids, skip_special_tokens=True)
                        and 0 < len(value.response_ids) <= config["evaluation_tokens"],
                        "Generated V native prompt/text/token identity differs",
                    )
                    append_json(
                        output / "observations.jsonl",
                        dict(
                            episode_id=ep.episode_id,
                            task_id=task.task_id,
                            family=ep.family,
                            partition=ep.partition,
                            condition=arm,
                            role="V",
                            candidate=None,
                            **grade(value.text, task),
                            generation=dataclasses.asdict(value),
                            cap_hit=value.cap_hit,
                            generation_batch_size=8,
                        ),
                    )
            receipt["completed_episode_ids"].append(ep.episode_id)
            receipt["work"] = dict(active.counters)
            write_json(output / "receipt.json", receipt)
            del cached
        observed = rows(output / "observations.jsonl")
        require(
            [{key: row[key] for key in expected[0]} for row in observed] == expected
            and active.counters == WORK,
            "Actual native V matrix/work differs from frozen plan",
        )
        require(
            all(sha(path) == value for path, value in frozen.items()),
            "Frozen transfer inputs changed",
        )
        check()
        totals = {
            arm: dict(
                successes=sum(row["success"] for row in observed if row["condition"] == arm),
                denominator=64,
                invalid=sum(not row["format_ok"] for row in observed if row["condition"] == arm),
                caps=sum(row["cap_hit"] for row in observed if row["condition"] == arm),
            )
            for arm in ARMS
        }
        receipt.update(
            status="completed",
            frozen_inputs_unchanged=True,
            validation_totals=totals,
            initial_student_sha256=initial_hash,
        )
    except Exception as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            work=dict(active.counters) if active is not None else dict.fromkeys(WORK, 0),
        )
        write_json(output / "receipt.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "fidelity", "framing", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        run(read_json(args.config), args.data, args.fidelity, args.framing, args.output)
    )
