"""Matched future-aware/causal teacher fitting and fixed-actor transfer controls."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch

from hindsight.model import forward_kl
from hindsight.shared import (
    SharedRecipe,
    adapter_values,
    alignment,
    audit_forward,
    restore,
    tensor_digest,
)

ARMS = (
    "baseline",
    "adapted_pi",
    "adapted_fitted_bidirectional",
    "adapted_unfitted_bidirectional",
    "adapted_fitted_raw_causal",
    "adapted_raw_causal",
    "adapted_replay",
)


@dataclass(frozen=True)
class TeacherRecipe(SharedRecipe):
    teacher_learning_rate: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.teacher_learning_rate) or self.teacher_learning_rate <= 0:
            raise ValueError("Require a positive finite teacher learning rate")


def fit_teacher_mode(engine, examples, recipe, initial, mode, seed, evidence, on_step=None):
    """Match weights/order/targets/work while comparing declared information-access modes.

    Both modes see the completed response and feedback. Ordinary causal mode
    serializes them before repeated action prediction, so context length/token
    positions differ. This is an information-matched, not pure mask, ablation.
    """
    if mode not in ("raw_causal", "bidirectional") or not examples:
        raise ValueError("Teacher fit mode must be raw_causal or bidirectional on nonempty sources")
    if len(examples) % recipe.teacher_batch_size:
        raise ValueError("Require complete fixed-size teacher batches")
    prepared = []
    for item in examples:
        if (
            item.provenance.get("selection_recipe") != recipe.selection_recipe()
            or item.provenance.get("derived_from") != "post_update_V"
        ):
            raise ValueError("Teacher targets require actual-V selection under this student recipe")
        engine._validate_target(item.generation, item.target)
        prepared.append((alignment(engine, item.generation, item.feedback, mode), item.target))
    actor_before = tensor_digest(adapter_values(engine), engine._check_budget)
    buffers_before = tensor_digest(dict(engine.model.named_buffers()), engine._check_budget)
    restore(engine, initial, "teacher")
    before = tensor_digest(adapter_values(engine, "teacher"), engine._check_budget)
    if before != tensor_digest(initial, engine._check_budget):
        raise RuntimeError("Teacher initialization differs from declared exact weights")
    parameters = list(engine._adapter_parameters("teacher").values())
    optimizer = engine._optimizer(parameters, recipe.teacher_learning_rate)
    schedule, rng = [], random.Random(seed)
    while len(schedule) < recipe.teacher_steps:
        order = list(range(len(examples)))
        rng.shuffle(order)
        schedule.extend(
            order[start : start + recipe.teacher_batch_size]
            for start in range(0, len(order), recipe.teacher_batch_size)
        )
    metrics = []
    with (
        audit_forward(engine, "teacher", evidence),
        engine._using_adapter("teacher", trainable=True),
    ):
        if {id(p) for p in engine.model.parameters() if p.requires_grad} != {
            id(p) for p in parameters
        }:
            raise RuntimeError("Teacher fit may update only teacher parameters")
        for step, group in enumerate(schedule[: recipe.teacher_steps]):
            started = time.monotonic()
            engine._check_budget()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in group:
                engine._check_budget()
                source, target = prepared[index]
                prediction = engine._logprobs(
                    source["input_ids"],
                    source["predecessor_positions"],
                    source["future_aware_attention"],
                )
                loss = forward_kl(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite matched teacher KL")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
                engine.counters["teacher_example_exposures"] += 1
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            engine.counters["teacher_optimizer_steps"] += 1
            row = {
                "step": step + 1,
                "mode": mode,
                "loss": sum(losses) / len(losses),
                "grad_norm": float(norm),
                "source_ids": [examples[index].episode_id for index in group],
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics.append(row)
            if on_step is not None:
                on_step(row)
    engine.model.zero_grad(set_to_none=True)
    if actor_before != tensor_digest(adapter_values(engine), engine._check_budget):
        raise RuntimeError("Teacher fitting changed frozen actor parameters")
    if buffers_before != tensor_digest(dict(engine.model.named_buffers()), engine._check_budget):
        raise RuntimeError("Teacher fitting changed model buffers")
    return {
        "mode": mode,
        "before_sha256": before,
        "after_sha256": tensor_digest(adapter_values(engine, "teacher"), engine._check_budget),
        "metrics": metrics,
        "optimizer_steps": len(metrics),
        "example_exposures": sum(len(row["source_ids"]) for row in metrics),
    }


def replay_target(generation, vocabulary):
    if not generation.response_ids or any(
        type(token) is not int or not 0 <= token < vocabulary for token in generation.response_ids
    ):
        raise ValueError("Replay requires native response tokens inside the vocabulary")
    result = torch.full((len(generation.response_ids), vocabulary), -torch.inf, dtype=torch.float32)
    result.scatter_(1, torch.tensor(generation.response_ids).unsqueeze(1), 0.0)
    return result


def distribution_metrics(reference, prediction, observed_ids=None):
    """Full-distribution diagnostics, including explicit infinite KL for one-hot controls."""
    q, p = reference.detach().float().cpu(), prediction.detach().float().cpu()
    if q.shape != p.shape or q.ndim != 2 or not q.shape[0]:
        raise ValueError("Metrics require aligned nonempty action distributions")
    if any(
        torch.isnan(value).any()
        or torch.isposinf(value).any()
        or not torch.isfinite(value.logsumexp(-1)).all()
        or value.logsumexp(-1).abs().max() > 1e-3
        for value in (q, p)
    ):
        raise ValueError("Metrics require normalized log probabilities")
    probabilities = q.exp()
    positive = probabilities > 0
    per_position = (
        probabilities * (torch.where(positive, q, 0.0) - torch.where(positive, p, 0.0))
    ).sum(-1)
    per_position = torch.where((p == q).all(-1), 0.0, per_position)
    mean_kl = float(per_position.mean())
    result = {
        "forward_kl_reference_to_prediction": mean_kl if math.isfinite(mean_kl) else None,
        "forward_kl_is_infinite": math.isinf(mean_kl),
        "per_position_forward_kl": [
            float(value) if torch.isfinite(value) else None for value in per_position
        ],
        "mean_total_variation": float((q.exp() - p.exp()).abs().sum(-1).mean() / 2),
        "argmax_agreement": float((q.argmax(-1) == p.argmax(-1)).float().mean()),
        "action_positions": q.shape[0],
    }
    if observed_ids is not None:
        if len(observed_ids) != q.shape[0] or any(
            type(token) is not int or not 0 <= token < q.shape[1] for token in observed_ids
        ):
            raise ValueError("Observed native IDs must match the distribution positions/vocabulary")
        observed = torch.tensor(observed_ids)
        reference_argmax, prediction_argmax = q.argmax(-1), p.argmax(-1)
        mask = reference_argmax != observed
        positions = mask.nonzero().flatten().tolist()
        diagnostics = []
        for position in positions:
            token, preferred = observed_ids[position], int(reference_argmax[position])
            q_observed, q_preferred = float(q[position, token]), float(q[position, preferred])
            p_observed, p_preferred = float(p[position, token]), float(p[position, preferred])
            reference_observed = float(q[position, token].exp())
            predicted_observed = float(p[position, token].exp())
            reference_preferred = float(q[position, preferred].exp())
            predicted_preferred = float(p[position, preferred].exp())
            diagnostics.append(
                {
                    "response_position": position,
                    "observed_token_id": token,
                    "reference_argmax_token_id": preferred,
                    "prediction_argmax_token_id": int(prediction_argmax[position]),
                    "reference_observed_probability": reference_observed,
                    "prediction_observed_probability": predicted_observed,
                    "reference_argmax_probability": reference_preferred,
                    "prediction_reference_argmax_probability": predicted_preferred,
                    "reference_observed_logprob": q_observed if math.isfinite(q_observed) else None,
                    "prediction_observed_logprob": p_observed
                    if math.isfinite(p_observed)
                    else None,
                    "reference_argmax_logprob": q_preferred if math.isfinite(q_preferred) else None,
                    "prediction_reference_argmax_logprob": p_preferred
                    if math.isfinite(p_preferred)
                    else None,
                    "reference_strictly_prefers_argmax": q_preferred > q_observed,
                    "prediction_strictly_prefers_reference_argmax": p_preferred > p_observed,
                    "prediction_ties_reference_argmax_and_observed": p_preferred == p_observed,
                    "forward_kl_reference_to_prediction": result["per_position_forward_kl"][
                        position
                    ],
                    "forward_kl_is_infinite": bool(torch.isinf(per_position[position])),
                }
            )
        count = len(diagnostics)
        kl = float(per_position[mask].mean()) if count else None
        strict = [row for row in diagnostics if row["reference_strictly_prefers_argmax"]]
        result["reference_action_disagreement"] = {
            "positions": count,
            "reference_strict_preference_positions": len(strict),
            "reference_tied_probability_positions": count - len(strict),
            "prediction_argmax_retained_positions": sum(
                row["prediction_argmax_token_id"] == row["reference_argmax_token_id"]
                for row in diagnostics
            ),
            "prediction_argmax_copies_observed_positions": sum(
                row["prediction_argmax_token_id"] == row["observed_token_id"] for row in diagnostics
            ),
            "strict_direction_retention_fraction": sum(
                row["prediction_strictly_prefers_reference_argmax"] for row in strict
            )
            / len(strict)
            if strict
            else None,
            "mean_forward_kl": kl if kl is not None and math.isfinite(kl) else None,
            "forward_kl_is_infinite": kl is not None and math.isinf(kl),
            "mean_reference_observed_probability": sum(
                row["reference_observed_probability"] for row in diagnostics
            )
            / count
            if count
            else None,
            "mean_prediction_observed_probability": sum(
                row["prediction_observed_probability"] for row in diagnostics
            )
            / count
            if count
            else None,
            "per_position": diagnostics,
            "interpretation": "Reference argmax differs from the historical native action; "
            "this is policy disagreement, not a ground-truth error label. Probability ties "
            "use log probabilities to avoid exponential underflow; null logprob means exact "
            "zero probability (-inf), not missing. Positions are correlated within a source, "
            "not independent samples.",
        }
    return result


def planned_work(training_sources, evaluation_sources, query_tasks, seeds, recipe):
    if any(
        type(value) is not int or value < 1
        for value in (training_sources, evaluation_sources, query_tasks, seeds)
    ):
        raise ValueError("Require positive explicit source/query/seed counts")
    if training_sources % recipe.teacher_batch_size:
        raise ValueError("Require complete fixed-size teacher batches")
    return {
        "generation_sequences": seeds * len(ARMS) * query_tasks,
        "scoring_forwards": seeds * (4 * training_sources + 10 * evaluation_sources),
        "student_optimizer_steps": seeds
        * (len(ARMS) - 1)
        * evaluation_sources
        * recipe.student_steps,
        "teacher_optimizer_steps": seeds * 2 * recipe.teacher_steps,
        "teacher_example_exposures": seeds * 2 * recipe.teacher_steps * recipe.teacher_batch_size,
    }
