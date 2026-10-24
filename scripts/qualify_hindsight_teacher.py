"""Bounded teacher-backward qualification, using an already scored source target.

No model loading, generation, provider operations, or standalone experiment here.
The following full teacher fit must reset its adapter and optimizer as usual.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.model import Generation, HFEngine

_spec = importlib.util.spec_from_file_location(
    "teacher_qualification_preflight", Path(__file__).with_name("preflight_gpu.py")
)
_preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preflight)
parameter_hash, write_receipt = _preflight.parameter_hash, _preflight.write_receipt


FIT_SEED = 7301
EXPECTED_WORK = {
    "generation_sequences": 0,
    "scoring_forwards": 5,
    "student_optimizer_steps": 0,
    "teacher_optimizer_steps": 2,
    "teacher_example_exposures": 2,
}


def qualify_teacher(
    engine: HFEngine, generation: Generation, feedback: str,
    target: torch.Tensor, output: Path,
) -> dict:
    """Fail closed and persist receipts for precisely five scores and two updates."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    initial_counts = engine.counters.copy()
    checks: dict = {}
    receipt = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "expected_work": EXPECTED_WORK.copy(), "checks": checks,
        "fit": {"epochs": 2, "batch_size": 1, "lr": 1e-4, "seed": FIT_SEED},
        "model_name": engine.model_name, "revision": engine.revision,
        "target_scoring_outside_this_allowance": True,
        "attention": {
            "description": (
                "Globally future-aware hybrid teacher: full-attention blocks are "
                "bidirectional; DeltaNet convolutions and recurrence remain causal. "
                "Later DeltaNet blocks can consume future-aware hidden states."
                if engine.model_type == "qwen3_5_text" else
                "Full-attention teacher uses noncausal unmasked attention."
            ),
            "layer_types": list(engine.model.config.layer_types),
            "mask_keys": list(engine._teacher_mask_keys),
            "use_cache": False,
        },
    }

    def persist():
        receipt["actual_work"] = {
            key: engine.counters[key] - initial_counts[key] for key in EXPECTED_WORK
        }
        receipt["elapsed_seconds"] = time.perf_counter() - started
        write_receipt(output / "qualification.json", receipt)

    def require(name, condition, **evidence):
        checks[name] = {"passed": bool(condition), **evidence}
        persist()
        if not condition:
            raise AssertionError(f"Teacher qualification failed: {name}")

    def parameter_digest(which):
        return parameter_hash(engine, which, engine._check_budget)

    def score(observed_feedback, **kwargs):
        result = engine.score_rollout(generation, observed_feedback, **kwargs)
        engine._validate_target(generation, result)
        require(f"score_{engine.counters['scoring_forwards'] - initial_counts['scoring_forwards']}"
                "_finite_normalized_detached", bool(torch.isfinite(result).all())
                and not result.requires_grad)
        return result

    persist()
    try:
        engine._check_budget()
        engine._validate_target(generation, target)
        before = {key: parameter_digest(key) for key in ("base", "student")}
        # Match fit_teacher's own reset: changing the random LoRA A initialization
        # alone must not be mistaken for evidence of a successful optimizer update.
        engine._reset_adapter("teacher", FIT_SEED)
        initial_teacher = parameter_digest("teacher")
        causal_before = score(feedback, adapter="student")
        metrics = engine.fit_teacher(
            [(generation, feedback, target)], epochs=2, batch_size=1,
            lr=1e-4, seed=FIT_SEED,
        )
        finite = (len(metrics["losses"]) == len(metrics["grad_norms"]) == 2
                  and all(math.isfinite(value) for value in
                          [*metrics["losses"], *metrics["grad_norms"],
                           metrics["elapsed_seconds"]]))
        require("finite_fit_metrics", finite)
        receipt["fit_metrics"] = metrics
        require("two_real_teacher_updates", metrics["optimizer_steps"] == 2
                and metrics["example_exposures"] == 2)
        fitted_teacher = parameter_digest("teacher")
        require("teacher_weights_changed_from_matched_initialization",
                fitted_teacher != initial_teacher, before_sha256=initial_teacher,
                after_sha256=fitted_teacher)
        fitted = score(feedback, mode="bidirectional", adapter="teacher")
        changed_feedback = ("The submitted answer was correct." if "incorrect" in feedback.lower()
                            else "The submitted answer was incorrect.")
        intervened = score(changed_feedback, mode="bidirectional", adapter="teacher")
        require("feedback_visible", not torch.equal(fitted, intervened),
                original_feedback=feedback, intervened_feedback=changed_feedback,
                max_abs_logprob_difference=float((fitted - intervened).abs().max()))
        checkpoint = output / "checkpoint.pt"
        engine.save_teacher(checkpoint)
        checksum = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            while chunk := stream.read(32 * 1024 * 1024):
                engine._check_budget()
                checksum.update(chunk)
        receipt["checkpoint"] = {
            "file": checkpoint.name, "sha256": checksum.hexdigest(),
            "bytes": checkpoint.stat().st_size, "parameter_sha256": fitted_teacher,
        }
        # Perturb the adapter first, so a silently ineffective reload cannot pass.
        engine._reset_adapter("teacher", FIT_SEED + 1)
        require("checkpoint_reload_exercises_restoration",
                parameter_digest("teacher") != fitted_teacher)
        engine.load_teacher(checkpoint)
        require("checkpoint_parameter_roundtrip",
                parameter_digest("teacher") == fitted_teacher)
        restored = score(feedback, mode="bidirectional", adapter="teacher")
        require("checkpoint_distribution_roundtrip", torch.equal(fitted, restored),
                max_abs_logprob_difference=float((fitted - restored).abs().max()))
        causal_after = score(feedback, adapter="student")
        require("causal_student_unchanged", torch.equal(causal_before, causal_after),
                max_abs_logprob_difference=float((causal_before - causal_after).abs().max()))
        for key, original_hash in before.items():
            current_hash = parameter_digest(key)
            require(f"{key}_parameters_unchanged", current_hash == original_hash,
                    before_sha256=original_hash, after_sha256=current_hash)
        persist()
        require("exact_work_allowance", receipt["actual_work"] == EXPECTED_WORK)
        receipt["status"] = "passed_teacher_qualification"
        return receipt
    except BaseException as error:
        receipt["status"] = "failed_teacher_qualification"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        persist()
