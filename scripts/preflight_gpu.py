"""Bounded E0 numerical qualification; writes a receipt, never provisions a GPU.

Run only on the user-approved GPU. The CLI refuses CPU execution before loading
weights. Unit tests inject an offline tiny HFEngine into run_checks instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from hindsight.data import Task, grade, load_corpus
from hindsight.model import HFEngine, configure_deterministic_execution, forward_kl

LIMITS = {"generation_sequences": 8, "scoring_forwards": 8, "student_optimizer_steps": 8}


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class Deadline:
    def __init__(self, seconds: float, absolute_utc: str):
        self.started = time.monotonic()
        self.seconds = seconds
        self.absolute = datetime.fromisoformat(absolute_utc)
        if self.absolute.tzinfo is None or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Preflight needs a positive timeout and timezone-aware deadline")

    def __call__(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("Preflight invocation timeout")
        if datetime.now(timezone.utc) >= self.absolute:
            raise TimeoutError("Absolute research deadline")


def parameter_hash(engine: HFEngine, which: str, check: Callable[[], None]) -> str:
    """Hash every selected parameter byte in bounded CPU chunks, including BF16."""
    digest = hashlib.sha256()
    for name, parameter in engine.model.named_parameters():
        selected = "lora_" not in name if which == "base" else f".{which}." in name
        if not selected:
            continue
        check()
        digest.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)}".encode())
        raw = parameter.detach().contiguous().view(torch.uint8).flatten()
        for offset in range(0, raw.numel(), 32 * 1024 * 1024):
            check()
            chunk = raw[offset:offset + 32 * 1024 * 1024].cpu().numpy()
            digest.update(memoryview(chunk))
    return digest.hexdigest()


def build_forecast(measurements: dict, generation_rows: dict, config: dict) -> dict:
    """Explicit scenarios, not a guarantee from one short microbenchmark."""
    counts = {"source_and_calibration": 128, "pi": 336, "greedy_evaluation": 8512}
    caps = {
        "source_and_calibration": config["source_tokens"],
        "pi": config["pi_tokens"], "greedy_evaluation": config["evaluation_tokens"],
    }
    rates = {}
    for kind, key in (("source_and_calibration", "source"), ("pi", "pi"),
                      ("greedy_evaluation", "greedy_batch")):
        tokens = sum(len(row["response_ids"]) for row in generation_rows[key])
        seconds = measurements[f"generate_{key}"]["seconds"]
        rates[kind] = tokens / max(seconds, 1e-9)
    generation_seconds = sum(counts[k] * caps[k] / rates[k] for k in counts)
    step_seconds = sum(
        measurements[key]["seconds"] for key in ("adapt_first", "adapt_repeat")
    ) / 4
    reset_seconds = statistics.mean(
        measurements[key]["seconds"] for key in ("reset_matched", "reset_neutral")
    )
    causal_forward = statistics.mean(
        measurements[key]["seconds"] for key in ("score_causal", "score_causal_edit")
    )
    bidir_forward = statistics.mean(
        measurements[key]["seconds"] for key in ("score_bidir", "score_bidir_edit")
    )
    teacher_ratio = max(1.0, bidir_forward / max(causal_forward, 1e-9))
    teacher_proxy = 64 * config["teacher_batch_size"] * step_seconds * teacher_ratio
    scoring_proxy = 544 * max(
        causal_forward, bidir_forward, measurements["score_pi"]["seconds"]
    )
    components = {
        "generation_at_observed_rates_and_planned_caps": generation_seconds,
        "2112_student_optimizer_steps": 2112 * step_seconds,
        "528_cached_clone_resets": 528 * reset_seconds,
        "64_teacher_steps_proxy_NOT_measured_fit": teacher_proxy,
        "544_teacher_scoring_forwards_upper_count_proxy": scoring_proxy,
    }
    return {
        "planned_generations": sum(counts.values()),
        "planned_output_token_ceiling": sum(counts[k] * caps[k] for k in counts),
        "planned_student_steps": 2112, "planned_teacher_steps": 64,
        "diagnostic_generation_batch_size": len(generation_rows["greedy_batch"]),
        "planned_generation_batch_size": config["generation_batch_size"],
        "observed_output_tokens_per_second": rates,
        "observed_non_neutral_student_seconds_per_step": step_seconds,
        "components_seconds": components,
        "projected_core_seconds": sum(components.values()),
        "projected_core_hours": sum(components.values()) / 3600,
        "limitations": [
            "One source and one short greedy batch; lengths and batch shapes vary in the core.",
            "Diagnostic batch is four, not the configured batch of eight when those differ.",
            "Teacher fit is a four-microexample proxy; no teacher optimizer step was measured.",
            "Raw-causal repetition and worst-case 4096-token backward are unmeasured.",
            "Projection excludes provisioning/downloads, result I/O, and additional investigation.",
            "This is a reforecast input, not completion or scientific-success evidence.",
        ],
    }


def run_checks(
    engine: HFEngine,
    calibration: list[Task],
    config: dict,
    receipt: dict,
    check: Callable[[], None],
    persist: Callable[[], None] = lambda: None,
) -> None:
    if len(calibration) < 6:
        raise ValueError("Preflight requires six distinct development calibration tasks")
    start_counts = engine.counters.copy()
    measurements = receipt.setdefault("measurements", {})
    checks = receipt.setdefault("checks", {})
    generation_rows = receipt.setdefault("generations", {})
    device = engine.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def counts():
        return {key: engine.counters[key] - start_counts[key] for key in engine.counters}

    def measure(name, function):
        check()
        sync()
        started = time.perf_counter()
        try:
            return function()
        finally:
            sync()
            measurements[name] = {"seconds": time.perf_counter() - started}
            receipt["actual_work"] = counts()
            if device.type == "cuda":
                receipt["memory_bytes"] = {
                    "peak_allocated": torch.cuda.max_memory_allocated(device),
                    "peak_reserved": torch.cuda.max_memory_reserved(device),
                    "currently_allocated": torch.cuda.memory_allocated(device),
                }
            persist()

    def require(name, condition, **evidence):
        evidence = {
            key: (str(value) if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in evidence.items()
        }
        checks[name] = {"passed": bool(condition), **evidence}
        persist()
        if not condition:
            raise AssertionError(f"Preflight gate failed: {name}")

    def reserve(kind, amount):
        check()
        if counts()[kind] + amount > LIMITS[kind]:
            raise RuntimeError(f"Diagnostic work limit would be exceeded: {kind}")

    def generate(key, prompts, cap, sample, adapter=None):
        reserve("generation_sequences", len(prompts))
        values = measure(f"generate_{key}", lambda: engine.generate(
            prompts, max_new_tokens=cap, sample=sample, seed=config["seed"],
            adapter=adapter,
            context_limit=(config["max_teacher_context_tokens"] if key == "pi"
                           else config["max_context_tokens"]),
        ))
        generation_rows[key] = [dataclasses.asdict(value) for value in values]
        persist()
        return values

    def score(key, generation, feedback, **kwargs):
        reserve("scoring_forwards", 1)
        value = measure(key, lambda: engine.score_rollout(generation, feedback, **kwargs))
        normal_error = float(value.logsumexp(-1).abs().max())
        require(
            f"{key}_normalized_detached", value.device.type == "cpu"
            and not value.requires_grad and bool(torch.isfinite(value).all())
            and normal_error < 1e-3, max_log_normalization_error=normal_error,
            action_positions=value.shape[0], vocabulary=value.shape[1],
        )
        return value

    def adapt(key, generation, target, steps):
        reserve("student_optimizer_steps", steps)
        values = measure(key, lambda: engine.adapt(
            generation, target, steps=steps, lr=config["learning_rate"]
        ))
        receipt.setdefault("adaptation_metrics", {})[key] = values
        persist()
        return values

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base_before = measure("hash_base_before", lambda: parameter_hash(engine, "base", check))
    teacher_before = parameter_hash(engine, "teacher", check)
    require("base_parameters_frozen", all(
        not parameter.requires_grad for name, parameter in engine.model.named_parameters()
        if "lora_" not in name
    ))
    source = generate("source", [calibration[0].prompt], config["source_tokens"], True)[0]
    result = grade(source.text, calibration[0])
    feedback = "The submitted answer was " + ("correct." if result["success"] else "incorrect.")
    receipt["source"] = {"task_id": calibration[0].task_id, "grade": result, "feedback": feedback}
    require("source_has_action_tokens", len(source.response_ids) >= 2)
    greedy = generate(
        "greedy_batch", [task.prompt for task in calibration[1:5]],
        config["evaluation_tokens"], False,
    )
    receipt["calibration_screen"] = [
        {"task_id": task.task_id, **grade(value.text, task), "cap_hit": value.cap_hit}
        for task, value in zip(calibration[1:5], greedy, strict=True)
    ]
    proposal_prompt = (
        "Review this completed attempt. Extract a short transferable lesson using only "
        "this experience. Do not write a new solution or copy its final answer. "
        "Give the lesson only.\n\n"
        f"TASK:\n{source.prompt}\n\nATTEMPT:\n{source.text}\n\n"
        f"OBSERVED FEEDBACK:\n{feedback}"
    )
    pi = generate("pi", [proposal_prompt], config["pi_tokens"], True)[0]
    causal = score("score_causal", source, feedback)
    replacement = (source.response_ids[0] + 17) % engine.model.config.vocab_size
    while replacement in (engine.eos_token_id, engine.pad_token_id, source.response_ids[0]):
        replacement = (replacement + 1) % engine.model.config.vocab_size
    edited = dataclasses.replace(source, response_ids=[replacement] + source.response_ids[1:])
    causal_edit = score("score_causal_edit", edited, feedback)
    require("causal_scored_token_invariance", torch.equal(causal[0], causal_edit[0]),
            max_abs_difference=float((causal[0] - causal_edit[0]).abs().max()))
    bidir = score("score_bidir", source, feedback, mode="bidirectional")
    bidir_edit = score("score_bidir_edit", edited, feedback, mode="bidirectional")
    require("bidirectional_scored_token_visible", not torch.equal(bidir[0], bidir_edit[0]),
            max_abs_difference=float((bidir[0] - bidir_edit[0]).abs().max()))
    target = score("score_pi", source, feedback, pi=pi.text)
    target_kl = float(forward_kl(causal, target))
    require("pi_target_is_non_neutral", math.isfinite(target_kl) and target_kl > 1e-7,
            source_action_mean_forward_kl=target_kl)

    measure("reset_initial", lambda: engine.reset_student(config["seed"]))
    initial_student = parameter_hash(engine, "student", check)
    require("fresh_optimizer_initial", not engine._student_optimizer.state)
    adapt("adapt_first", source, target, 2)
    first_hash = parameter_hash(engine, "student", check)
    require("non_neutral_update_changes_adapter", first_hash != initial_student)
    first_score = score("score_adapted_first", source, feedback, adapter="student")
    require("non_neutral_update_changes_causal_policy", not torch.equal(first_score, causal),
            max_abs_logprob_change=float((first_score - causal).abs().max()))
    generate(
        "adapted_first", [calibration[5].prompt], config["evaluation_tokens"], False, "student"
    )

    measure("reset_matched", lambda: engine.reset_student(config["seed"]))
    restored_hash = parameter_hash(engine, "student", check)
    require("matched_initial_adapter_reset", restored_hash == initial_student)
    require("matched_optimizer_reset", not engine._student_optimizer.state)
    adapt("adapt_repeat", source, target, 2)
    repeated_hash = parameter_hash(engine, "student", check)
    repeated_score = score("score_adapted_repeat", source, feedback, adapter="student")
    require("matched_update_reproducible", torch.allclose(
        first_score, repeated_score, atol=1e-4, rtol=0
    ), max_abs_logprob_difference=float((first_score - repeated_score).abs().max()),
        adapter_bytes_identical=first_hash == repeated_hash)
    generate(
        "adapted_repeat", [calibration[5].prompt], config["evaluation_tokens"], False, "student"
    )

    measure("reset_neutral", lambda: engine.reset_student(config["seed"]))
    neutral_metrics = adapt("adapt_neutral", source, causal, 4)
    neutral_score = score("score_neutral", source, feedback, adapter="student")
    require("neutral_minimum_is_exact_noop", parameter_hash(engine, "student", check)
            == initial_student and torch.equal(causal, neutral_score),
            losses=[row["loss"] for row in neutral_metrics],
            max_abs_logprob_change=float((causal - neutral_score).abs().max()))
    base_after = measure("hash_base_after", lambda: parameter_hash(engine, "base", check))
    teacher_after = parameter_hash(engine, "teacher", check)
    require("frozen_base_bytes_unchanged", base_before == base_after,
            before_sha256=base_before, after_sha256=base_after)
    require("teacher_adapter_untouched", teacher_before == teacher_after,
            before_sha256=teacher_before, after_sha256=teacher_after)
    require("no_base_or_teacher_gradients", all(
        parameter.grad is None for name, parameter in engine.model.named_parameters()
        if "lora_" not in name or ".teacher." in name
    ))
    receipt["actual_work"] = counts()
    require("diagnostic_allowance_respected", all(counts()[key] <= value
            for key, value in LIMITS.items()), limits=LIMITS, actual=counts())
    receipt["forecast"] = build_forecast(measurements, generation_rows, config)
    if "absolute_deadline_utc" in config:
        remaining = (
            datetime.fromisoformat(config["absolute_deadline_utc"])
            - datetime.now(timezone.utc)
        ).total_seconds()
        receipt["forecast"]["seconds_to_absolute_deadline"] = max(0.0, remaining)
        receipt["forecast"]["projection_fits_absolute_deadline"] = (
            receipt["forecast"]["projected_core_seconds"] <= remaining
        )
    receipt["status"] = "passed_numerical_checks"
    receipt["claim_boundary"] = (
        "GPU numerical path qualified on one development source. Not a learning-benefit "
        "result, full calibration pass, worst-case memory proof, or permission to scale."
    )
    persist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/initial.json"))
    parser.add_argument("--data", type=Path, default=Path("data/corpus-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Receipt already exists; use a new explicit output path, not an overwrite")
    started = time.monotonic()
    receipt = {"schema_version": 1, "kind": "gpu_preflight", "status": "started",
               "started_at": datetime.now(timezone.utc).isoformat(), "limits": LIMITS}
    engine = None

    def persist():
        write_receipt(args.output, receipt)

    try:
        config = json.loads(args.config.read_text())
        receipt["numerical_execution"] = configure_deterministic_execution(
            config.get("deterministic_execution", False)
        )
        deadline = Deadline(args.max_seconds, config["absolute_deadline_utc"])
        deadline()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU preflight requires CUDA; no checkpoint was loaded")
        _, calibration = load_corpus(args.data)
        properties = torch.cuda.get_device_properties(0)
        receipt["identity"] = {
            "model_name": config["model_name"], "revision": config["revision"],
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "corpus_manifest_sha256": hashlib.sha256(
                (args.data / "manifest.json").read_bytes()
            ).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "peft")},
            "torch_cuda": torch.version.cuda, "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
        }
        persist()
        load_start = time.perf_counter()
        engine = HFEngine(
            model_name=config["model_name"], revision=config["revision"], device="cuda",
            max_generation_context=config["max_context_tokens"],
            max_teacher_context=config["max_teacher_context_tokens"], budget_check=deadline,
            enable_thinking=config.get("enable_thinking"),
        )
        torch.cuda.synchronize()
        receipt["model_load_seconds"] = time.perf_counter() - load_start
        receipt["adapter_configuration"] = {
            "targets": engine.lora_targets,
            "module_coverage": engine.lora_module_coverage,
            "parameter_counts": {
                adapter: sum(p.numel() for p in engine._adapter_parameters(adapter).values())
                for adapter in ("student", "teacher")
            },
            "teacher_mask_keys": list(engine._teacher_mask_keys),
        }
        persist()
        run_checks(engine, calibration, config, receipt, deadline, persist)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if engine is not None:
            receipt["engine_counters"] = engine.counters.copy()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        persist()
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] == "passed_numerical_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
