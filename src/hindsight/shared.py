"""Bounded shared-parameter hindsight fitting and causal learning.

The ``student`` LoRA is the single active trainable parameter set in the shared
arms. The existing ``teacher`` LoRA is used only by the named separate control.
Qwen3.5 remains a future-aware hybrid, not an all-layer bidirectional model.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import torch
from torch import Tensor

from hindsight.model import Generation, HFEngine, forward_kl

ARMS = ("baseline", "actor_only", "teacher_only", "shared", "separate")
UPDATED_ARMS = ("actor_only", "shared", "separate")


@dataclass(frozen=True)
class SharedRecipe:
    student_steps: int = 4
    learning_rate: float = 1e-4
    teacher_steps: int = 64
    teacher_batch_size: int = 4
    loss: str = "forward_kl"

    def __post_init__(self):
        if (
            type(self.student_steps) is not int
            or self.student_steps < 1
            or type(self.teacher_steps) is not int
            or self.teacher_steps < 1
            or type(self.teacher_batch_size) is not int
            or self.teacher_batch_size < 1
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.loss != "forward_kl"
        ):
            raise ValueError("Shared learning requires explicit positive counts/lr and forward KL")

    def selection_recipe(self):
        return {key: asdict(self)[key] for key in ("student_steps", "learning_rate", "loss")}


@dataclass(frozen=True)
class TeachingExample:
    episode_id: str
    generation: Generation
    feedback: str
    target: Tensor
    provenance: dict


def tensor_digest(values: dict[str, Tensor], check: Callable[[], None] = lambda: None) -> str:
    """Stable name/dtype/shape/byte digest with bounded CPU copies, including BF16."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        check()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0, raw.numel(), 32 * 1024 * 1024):
            check()
            digest.update(memoryview(raw[offset : offset + 32 * 1024 * 1024].cpu().numpy()))
    return digest.hexdigest()


def adapter_values(engine: HFEngine, adapter: str = "student") -> dict[str, Tensor]:
    values = engine._adapter_parameters(adapter)
    if not values:
        raise ValueError("Requested adapter has no parameters")
    return {name.replace(f".{adapter}.", ".__adapter__."): value for name, value in values.items()}


def snapshot(engine: HFEngine, adapter: str = "student") -> dict[str, Tensor]:
    engine._check_budget()
    return {
        name: value.detach().cpu().clone()
        for name, value in adapter_values(engine, adapter).items()
    }


def restore(engine: HFEngine, values: dict[str, Tensor], adapter: str = "student") -> None:
    current = adapter_values(engine, adapter)
    if current.keys() != values.keys():
        raise ValueError("Adapter snapshot parameter names differ")
    for name, value in values.items():
        if value.shape != current[name].shape or value.dtype != current[name].dtype:
            raise ValueError("Adapter snapshot shape or dtype differs")
        if not torch.isfinite(value).all():
            raise ValueError("Adapter snapshot is nonfinite")
    with torch.no_grad():
        for name, value in values.items():
            engine._check_budget()
            current[name].copy_(value)
    engine.model.zero_grad(set_to_none=True)
    if adapter == "student":
        engine._student_optimizer = engine._optimizer(list(current.values()), 1e-4)
        engine._student_ready = True


def storage_map(engine: HFEngine, adapter: str = "student") -> dict:
    return {
        name: {
            "object_id": id(value),
            "data_ptr": value.data_ptr(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(engine._adapter_parameters(adapter).items())
    }


@contextmanager
def audit_forward(engine: HFEngine, adapter: str, evidence: list[dict]) -> Iterator[None]:
    """Check actual model calls, not just a declared adapter name."""
    expected = storage_map(engine, adapter)
    bindings = [
        (engine.model.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1], value)
        for name, value in engine._adapter_parameters(adapter).items()
    ]
    pointers = [value.data_ptr() for _, _, value in bindings]
    calls = 0

    def before_forward(_module, _args):
        nonlocal calls
        active = engine.model.active_adapter
        active = [active] if isinstance(active, str) else list(active)
        # Avoid a complete base-model named_parameters traversal for every
        # decoded token, while still checking the actual parameter bindings.
        unchanged = all(
            module._parameters[name] is value and value.data_ptr() == pointer
            for (module, name, value), pointer in zip(bindings, pointers, strict=True)
        )
        if active != [adapter] or not unchanged:
            raise RuntimeError("Forward adapter or live parameter storage changed")
        calls += 1

    # PEFT.generate bypasses its wrapper's forward, while PEFT's scoring path
    # can call the LM's .forward directly. Both routes call the text backbone
    # through Module.__call__, so its hook observes acting and scoring alike.
    observed = engine.model.get_base_model().model
    hook = observed.register_forward_pre_hook(before_forward)
    try:
        yield
    finally:
        hook.remove()
        evidence.append(
            {
                "adapter": adapter,
                "forward_calls": calls,
                "same_parameter_storage": storage_map(engine, adapter) == expected,
            }
        )


def alignment(engine: HFEngine, generation: Generation, feedback: str, mode: str) -> dict:
    if generation.prompt_ids != engine.encode_prompt(generation.prompt):
        raise ValueError("Native source prompt IDs do not match the pinned tokenizer")
    tokens, positions, bidirectional = engine._scoring_input(generation, feedback, None, mode)
    if [tokens[index + 1] for index in positions] != generation.response_ids:
        raise ValueError("Scoring positions lost native response/predecessor alignment")
    return {
        "input_ids": tokens,
        "predecessor_positions": positions,
        "response_ids": generation.response_ids,
        "mode": mode,
        "future_aware_attention": bidirectional,
    }


def score(
    engine: HFEngine,
    generation: Generation,
    feedback: str,
    *,
    mode: str,
    adapter: str,
    evidence: list[dict],
) -> Tensor:
    alignment(engine, generation, feedback, mode)
    with audit_forward(engine, adapter, evidence):
        result = engine.score_rollout(generation, feedback, mode=mode, adapter=adapter)
    engine._validate_target(generation, result)
    if result.requires_grad or not torch.isfinite(result).all():
        raise ValueError("Scoring must produce finite detached full-vocabulary targets")
    return result


def fit_shared(
    engine: HFEngine,
    examples: list[TeachingExample],
    recipe: SharedRecipe,
    seed: int,
    evidence: list[dict],
) -> dict:
    """Fit hindsight mode on the actor's exact adapter; never reset it here."""
    if not examples or recipe.teacher_batch_size > len(examples):
        raise ValueError("Teacher batch must fit the nonempty training source set")
    prepared = []
    for example in examples:
        if example.provenance.get("selection_recipe") != recipe.selection_recipe():
            raise ValueError("Teacher labels were selected under another student recipe")
        if example.provenance.get("derived_from") != "post_update_V":
            raise ValueError("Teacher target selection must use actual post-update V only")
        engine._validate_target(example.generation, example.target)
        source = alignment(engine, example.generation, example.feedback, "bidirectional")
        prepared.append((source, example.target.detach().cpu()))
    untouched = tensor_digest(adapter_values(engine, "teacher"), engine._check_budget)
    before = tensor_digest(adapter_values(engine), engine._check_budget)
    parameters = list(engine._adapter_parameters("student").values())
    optimizer = engine._optimizer(parameters, recipe.learning_rate)
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < recipe.teacher_steps:
        order = list(range(len(examples)))
        rng.shuffle(order)
        schedule.extend(
            order[start : start + recipe.teacher_batch_size]
            for start in range(0, len(order), recipe.teacher_batch_size)
        )
    metrics = []
    with (
        audit_forward(engine, "student", evidence),
        engine._using_adapter("student", trainable=True),
    ):
        if {id(p) for p in engine.model.parameters() if p.requires_grad} != {
            id(p) for p in parameters
        }:
            raise RuntimeError("Hindsight fitting must update only the shared actor adapter")
        for step, group in enumerate(schedule[: recipe.teacher_steps]):
            engine._check_budget()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in group:
                source, target = prepared[index]
                prediction = engine._logprobs(
                    source["input_ids"], source["predecessor_positions"], True
                )
                loss = forward_kl(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite shared hindsight loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
                engine.counters["teacher_example_exposures"] += 1
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            engine.counters["teacher_optimizer_steps"] += 1
            metrics.append(
                {
                    "step": step + 1,
                    "loss": sum(losses) / len(losses),
                    "grad_norm": float(grad),
                    "source_ids": [examples[i].episode_id for i in group],
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
    engine.model.zero_grad(set_to_none=True)
    after = tensor_digest(adapter_values(engine), engine._check_budget)
    if untouched != tensor_digest(adapter_values(engine, "teacher"), engine._check_budget):
        raise RuntimeError("Shared fitting mutated the separate control adapter")
    if before == after:
        raise RuntimeError("Shared hindsight fitting made no parameter change")
    return {
        "before_sha256": before,
        "after_sha256": after,
        "metrics": metrics,
        "optimizer_steps": len(metrics),
        "example_exposures": sum(len(row["source_ids"]) for row in metrics),
    }


def adapt_shared(
    engine: HFEngine,
    generation: Generation,
    target: Tensor,
    recipe: SharedRecipe,
    evidence: list[dict],
) -> dict:
    """Fresh AdamW state, unchanged starting weights, fixed detached target."""
    alignment(engine, generation, "", "causal")
    target = target.detach().cpu().clone()
    target_sha = tensor_digest({"target": target}, engine._check_budget)
    parameters = list(engine._adapter_parameters("student").values())
    engine._student_optimizer = engine._optimizer(parameters, recipe.learning_rate)
    engine._student_ready = True
    initial = []
    before = tensor_digest(adapter_values(engine), engine._check_budget)
    with audit_forward(engine, "student", evidence):
        metrics = engine.adapt(
            generation,
            target,
            recipe.student_steps,
            recipe.learning_rate,
            initial_logprobs_callback=initial.append,
        )
    if len(initial) != 1 or target_sha != tensor_digest({"target": target}, engine._check_budget):
        raise RuntimeError("Initial distribution missing or detached target changed")
    return {
        "metrics": metrics,
        "initial": initial[0],
        "target_sha256": target_sha,
        "before_sha256": before,
        "after_sha256": tensor_digest(adapter_values(engine), engine._check_budget),
    }


def distances(reference: Tensor, prediction: Tensor) -> dict:
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Distribution comparison requires aligned matrices")
    q, p = reference.detach().float(), prediction.detach().float()
    if not torch.isfinite(q).all() or not torch.isfinite(p).all():
        raise ValueError("Nonfinite distribution comparison")
    return {
        "forward_kl_reference_to_prediction": float(forward_kl(p, q)),
        "mean_total_variation": float((q.exp() - p.exp()).abs().sum(-1).mean() / 2),
        "argmax_agreement": float((q.argmax(-1) == p.argmax(-1)).float().mean()),
        "action_positions": q.shape[0],
    }


def planned_work(
    training_sources: int,
    evaluation_sources: int,
    query_tasks: int,
    seeds: int,
    recipe: SharedRecipe,
) -> dict:
    if training_sources % recipe.teacher_batch_size:
        raise ValueError("Require complete fixed-size teacher batches for exact work accounting")
    return {
        "generation_sequences": seeds * len(ARMS) * (evaluation_sources + query_tasks),
        "scoring_forwards": seeds
        * (4 * training_sources + 3 * len(UPDATED_ARMS) * evaluation_sources),
        "student_optimizer_steps": seeds
        * len(UPDATED_ARMS)
        * evaluation_sources
        * recipe.student_steps,
        "teacher_optimizer_steps": seeds * recipe.teacher_steps,
        "teacher_example_exposures": seeds * recipe.teacher_steps * recipe.teacher_batch_size,
    }
