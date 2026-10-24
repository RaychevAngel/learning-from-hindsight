"""V-only 4-versus-16-step comparison on all eight immutable diagnostic sources."""

import argparse
import dataclasses
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.data import grade, load_corpus
from hindsight.experiment import append_json, read_json, rows, seed_for, write_json
from hindsight.model import Generation, HFEngine, configure_deterministic_execution, with_pi


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def select_recipe(observations):
    """Choose only between predeclared doses; the cheaper dose wins ties."""
    conditions = ("baseline", "direct_pi", "steps_4", "steps_16")
    if any(row["role"] != "V" for row in observations):
        raise ValueError("Recipe selection must not inspect Q")
    grouped = {c: [r for r in observations if r["condition"] == c] for c in conditions}
    identities = [{(r["episode_id"], r["task_id"]) for r in grouped[c]} for c in conditions]
    if not identities[0] or any(s != identities[0] for s in identities):
        raise ValueError("Every dose/control must have the same V identities")
    if any(len(grouped[c]) != len(identities[0]) for c in conditions):
        raise ValueError("Duplicate dose outcome")
    totals = {c: sum(r["success"] for r in grouped[c]) for c in conditions}
    selected = 16 if totals["steps_16"] > totals["steps_4"] else 4
    return dict(
        student_steps=selected,
        learning_rate=1e-4,
        loss="forward_kl",
        validation_successes=totals,
        denominator=len(identities[0]),
        selection="V only; choose 16 iff better than 4; ties choose 4",
    )


def run(config, data, fidelity, output, *, engine=None):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="started",
        config=config,
        completed_episode_ids=[],
        planned_work=dict(
            generation_sequences=256,
            scoring_forwards=16,
            student_optimizer_steps=160,
            teacher_optimizer_steps=0,
            teacher_example_exposures=0,
        ),
    )
    write_json(output / "receipt.json", receipt)
    active = engine
    try:
        prior = read_json(fidelity / "receipt.json")
        ids = prior["episode_ids"]
        episodes = {ep.episode_id: ep for ep in load_corpus(data)[0]}
        lessons = {r["episode_id"]: r for r in read_json(fidelity / "lessons.json")["rows"]}
        if prior["status"] != "completed" or len(ids) != 8 or len(set(ids)) != 8:
            raise ValueError("Require all eight completed frozen source-fitting cases")
        for key in ("model_name", "revision", "seed", "enable_thinking"):
            if config[key] != prior["inherited_config"][key]:
                raise ValueError("Dose model/initialization differs from frozen source targets")
        if config["generation_batch_size"] != 8 or config["learning_rate"] != 1e-4:
            raise ValueError("This bounded diagnostic fixes B8 and learning rate 1e-4")
        qualification = config["batch_qualification"]
        qualified = read_json(Path(qualification["path"]))
        if (
            sha(qualification["path"]) != qualification["sha256"]
            or qualified["status"] != "completed"
            or qualified["selected_batch_size"] != 8
            or not qualified["repeat_stability"]["8"]
            or any(qualified["config"][key] != config[key] for key in ("model_name", "revision"))
        ):
            raise ValueError("Require pinned same-model repeat-stable B8 qualification")
        frozen = {
            str(p): sha(p)
            for p in (
                data / "manifest.json",
                fidelity / "receipt.json",
                fidelity / "lessons.json",
                Path(__file__),
            )
        }

        def check():
            if time.monotonic() - started >= config["max_invocation_seconds"] or datetime.now(
                timezone.utc
            ) >= datetime.fromisoformat(config["absolute_deadline_utc"]):
                raise TimeoutError("Dose diagnostic deadline reached")

        configure_deterministic_execution(config["deterministic_execution"])
        if active is None:
            active = HFEngine(
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
        for ep_id in ids:
            check()
            ep = episodes[ep_id]
            if len(ep.validation) != 8:
                raise ValueError("Require V8 per source")
            source_path = fidelity / "sources" / f"{ep_id}.json"
            source = read_json(source_path)
            generation = Generation(**source["generation"])
            lesson = lessons[ep_id]
            if (
                sha(source_path) != lesson["source_sha256"]
                or source["task_id"] != ep.source.task_id
                or generation.prompt != ep.source.prompt
                or generation.prompt_ids != active.encode_prompt(generation.prompt)
            ):
                raise ValueError("Frozen lesson/source/native prompt identity mismatch")
            tensor_path = fidelity / "distributions" / f"{ep_id}.pt"
            artifact = read_json(tensor_path.with_suffix(".json"))["artifact"]
            if sha(tensor_path) != artifact["sha256"]:
                raise ValueError("025 full distribution checksum mismatch")
            saved = torch.load(tensor_path, weights_only=True, map_location="cpu")
            if (
                saved["response_ids"] != generation.response_ids
                or saved["prompt_ids"] != generation.prompt_ids
            ):
                raise ValueError("025 target native token identity mismatch")
            target = saved["target"]
            del saved
            for condition, steps in (
                ("baseline", 0),
                ("direct_pi", 0),
                ("steps_4", 4),
                ("steps_16", 16),
            ):
                adapter = None
                if steps:
                    active.reset_student(seed_for(config["seed"], "student_initialization"))
                    metrics = active.adapt(generation, target, steps, 1e-4)
                    post = active.score_rollout(generation, "", adapter="student")
                    kl = (target.exp() * (target - post)).sum(-1)
                    append_json(
                        output / "adaptations.jsonl",
                        dict(
                            episode_id=ep_id,
                            condition=condition,
                            metrics=metrics,
                            fresh_source_kl_mean=kl.mean().item(),
                            fresh_source_kl_positions=kl.tolist(),
                            source_sha256=sha(source_path),
                            target_sha256=artifact["sha256"],
                        ),
                    )
                    adapter = "student"
                prompts = [
                    with_pi(t.prompt, lesson["lesson"]) if condition == "direct_pi" else t.prompt
                    for t in ep.validation
                ]
                generated = active.generate(
                    prompts,
                    max_new_tokens=config["evaluation_tokens"],
                    sample=False,
                    seed=config["seed"],
                    adapter=adapter,
                )
                for task, value in zip(ep.validation, generated, strict=True):
                    append_json(
                        output / "observations.jsonl",
                        dict(
                            episode_id=ep_id,
                            task_id=task.task_id,
                            family=ep.family,
                            role="V",
                            condition=condition,
                            **grade(value.text, task),
                            generation=dataclasses.asdict(value),
                            generation_batch_size=8,
                        ),
                    )
            receipt["completed_episode_ids"].append(ep_id)
            receipt["work"] = dict(active.counters)
            write_json(output / "receipt.json", receipt)
        if active.counters != receipt["planned_work"]:
            raise ValueError("Dose actual work differs from fixed budget")
        if any(sha(p) != expected for p, expected in frozen.items()):
            raise ValueError("Dose frozen inputs changed")
        recipe = select_recipe(rows(output / "observations.jsonl"))
        write_json(output / "recipe.json", recipe)
        receipt.update(status="completed", selected_recipe=recipe, frozen_files=frozen)
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
    for name in ("config", "data", "fidelity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(run(read_json(args.config), args.data, args.fidelity, args.output)["status"])
