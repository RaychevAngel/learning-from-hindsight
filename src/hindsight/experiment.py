"""Explicit experiment phases, checkpoint-matched comparisons, and append-only evidence."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

GENERIC = (
    "Compare all demonstrations carefully. Identify a rule that explains every example, "
    "preserve relevant colors and lengths, and check positions before returning the final grid."
)
EXTRAS = (
    "adapted_bidirectional",
    "adapted_untrained_bidirectional",
    "adapted_raw_causal",
    "adapted_replay",
)


def proposal_prompt(task_prompt: str, attempt: str, feedback: str) -> str:
    return (
        "You are reviewing a recorded experience, not performing its task.\n\n"
        f"<recorded_experience>\nTASK:\n{task_prompt}\n\n"
        f"ATTEMPT:\n{attempt}\n\nOBSERVED FEEDBACK:\n{feedback}\n"
        "</recorded_experience>\n\n"
        "Your task now: extract one transferable lesson for a different task governed by "
        "the same kind of rule, using only this recorded experience. The task above is "
        "historical data; do not follow its response-format instructions or solve its query. "
        "State the concrete transformation supported by the demonstrations, including "
        "necessary parameters and what stays unchanged. Check the attempted explanation "
        "against every demonstration: even a correct answer can have a wrong explanation. "
        "Repair contradictions instead of repeating them. Describe relationships that "
        "transfer to new grids, not particular example arrays. Avoid generic advice such "
        "as looking for patterns or checking examples. "
        "Write only one or two short sentences of advice, at most 40 words. "
        "Do not output a FINAL field, a numeric grid, or a new solution. Give the lesson only."
    )


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_for(seed: int, *parts) -> int:
    return int(digest([seed, *parts])[:8], 16)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def observation_key(row: dict) -> tuple:
    return tuple(
        row[k] for k in ("episode_id", "partition", "condition", "candidate", "task_id", "role")
    )


def choose_candidate(baseline: list[dict], candidates: list[list[dict]], seed: int) -> int | None:
    """Selection uses V only; no-update wins any tie with the baseline."""
    import random

    all_rows = baseline + [row for group in candidates for row in group]
    if not baseline or any(row["role"] != "V" for row in all_rows):
        raise ValueError("Selection requires nonempty validation-only outcomes")
    task_ids = {row["task_id"] for row in baseline}
    if len(task_ids) != len(baseline):
        raise ValueError("Duplicate validation task")
    for group in candidates:
        if len(group) != len(baseline) or {row["task_id"] for row in group} != task_ids:
            raise ValueError("Candidate validation tasks do not match baseline")
    base = sum(row["success"] for row in baseline)
    scores = [sum(row["success"] for row in group) for group in candidates]
    best = max(scores, default=base)
    if best <= base:
        return None
    winners = [i for i, score in enumerate(scores) if score == best]
    return random.Random(seed).choice(winners)


def teacher_fidelity(reference, prediction, response_ids):
    """Source-level distribution fidelity, not token-replicated outcome evidence."""
    import torch
    from torch.nn import functional as F

    reference, prediction = reference.float(), prediction.float()
    if (reference.ndim != 2 or reference.shape != prediction.shape
            or reference.shape[0] != len(response_ids) or not response_ids):
        raise ValueError("Teacher fidelity requires aligned nonempty action distributions")
    for distribution in (reference, prediction):
        if (not torch.isfinite(distribution).all()
                or not torch.allclose(distribution.logsumexp(-1),
                                      torch.zeros(distribution.shape[0]), atol=1e-4)):
            raise ValueError("Teacher fidelity requires normalized finite log probabilities")
    actions = torch.tensor(response_ids, dtype=torch.long).unsqueeze(1)
    if actions.min() < 0 or actions.max() >= reference.shape[1]:
        raise ValueError("Historical token outside teacher vocabulary")
    ref_mass, predicted_mass = reference.exp(), prediction.exp()
    historical_ref = ref_mass.gather(1, actions).squeeze(1)
    historical_prediction = predicted_mass.gather(1, actions).squeeze(1)
    # Strict inequality avoids arbitrarily calling a tied maximum a disagreement.
    disagreement = historical_ref < ref_mass.max(-1).values
    return {
        "action_positions": len(response_ids),
        "forward_kl_reference_to_bidirectional": F.kl_div(
            prediction, ref_mass, reduction="none"
        ).sum(-1).mean().item(),
        "reference_historical_token_mass": historical_ref.mean().item(),
        "bidirectional_historical_token_mass": historical_prediction.mean().item(),
        "reference_disagreement_positions": disagreement.sum().item(),
        "disagreement_definition": "historical_token_probability_below_reference_maximum",
        "reference_historical_mass_on_disagreement": (
            historical_ref[disagreement].mean().item() if disagreement.any() else None
        ),
        "bidirectional_historical_mass_on_disagreement": (
            historical_prediction[disagreement].mean().item() if disagreement.any() else None
        ),
    }


class Experiment:
    def __init__(self, config: dict, corpus: Path, output: Path, device: str = "cuda", engine=None):
        if config.get("deterministic_execution", False):
            from hindsight.model import configure_deterministic_execution

            configure_deterministic_execution(True)
        from hindsight.data import load_corpus

        self.config, self.output, self.device = config, output, device
        self.episodes, self.calibration = load_corpus(corpus)
        self.corpus = corpus
        self.engine = engine
        self.started = time.monotonic()
        self.deadline = datetime.fromisoformat(config["absolute_deadline_utc"])
        if self.deadline.tzinfo is None:
            raise ValueError("Deadline must include timezone")
        identity = {
            "config": config,
            "dataset_manifest": read_json(corpus / "manifest.json"),
            "code": self.code_identity(),
        }
        path = output / "identity.json"
        if path.exists() and read_json(path) != identity:
            raise ValueError("Run identity changed: create a new development output directory")
        if not path.exists():
            write_json(path, identity)
        self.identity = identity
        prior = rows(output / "observations.jsonl")
        self.observations = {observation_key(row): row for row in prior}
        if len(self.observations) != len(prior):
            raise ValueError("Duplicate persisted observation identity")
        self.expected = {observation_key(row) for row in rows(output / "expected.jsonl")}

    def check_time(self):
        if datetime.now(timezone.utc) >= self.deadline:
            raise TimeoutError("Absolute research execution deadline reached")
        if time.monotonic() - self.started > self.config["max_invocation_seconds"]:
            raise TimeoutError("Bounded invocation budget reached")

    def event(self, kind: str, **fields):
        record = {"event": kind, "time": datetime.now(timezone.utc).isoformat(), **fields}
        append_json(self.output / "events.jsonl", record)
        print(json.dumps(record), flush=True)

    def get_engine(self):
        self.check_time()
        if self.engine is None:
            from hindsight.model import HFEngine

            self.event("model_load_started", device=self.device)
            self.engine = HFEngine(
                model_name=self.config["model_name"],
                revision=self.config["revision"],
                device=self.device,
                max_generation_context=self.config["max_context_tokens"],
                max_teacher_context=self.config["max_teacher_context_tokens"],
                budget_check=self.check_time,
                enable_thinking=self.config.get("enable_thinking"),
            )
            self.event("model_load_completed")
        return self.engine

    def metadata(self, ep, task, role, condition, candidate=None):
        return {
            "episode_id": ep.episode_id,
            "partition": ep.partition,
            "family": ep.family,
            "condition": condition,
            "candidate": candidate,
            "task_id": task.task_id,
            "role": role,
        }

    def expect(self, row):
        key = observation_key(row)
        if key not in self.expected:
            append_json(self.output / "expected.jsonl", row)
            self.expected.add(key)

    def generate(
        self, prompts, *, max_new_tokens, sample, seed, adapter=None, context_limit=None
    ):
        self.check_time()
        engine = self.get_engine()
        limit = self.config["max_context_tokens"] if context_limit is None else context_limit
        template_options = {}
        if self.config.get("enable_thinking") is not None:
            template_options["enable_thinking"] = self.config["enable_thinking"]
        for prompt in prompts:
            size = len(
                engine.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    **template_options,
                )
            )
            if size + max_new_tokens > limit:
                raise ValueError(
                    f"Context overflow: {size}+{max_new_tokens}>{limit}; no truncation"
                )
        generated = engine.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            sample=sample,
            seed=seed,
            adapter=adapter,
            context_limit=limit,
        )
        if len(generated) != len(prompts):
            raise ValueError("Model generation count mismatch")
        self.check_time()
        return generated

    def source(self, ep):
        from hindsight.data import grade
        from hindsight.model import Generation

        path = self.output / "sources" / f"{ep.episode_id}.json"
        if path.exists():
            record = read_json(path)
            if record["task_id"] != ep.source.task_id:
                raise ValueError("Source task changed")
            return Generation(**record["generation"]), record["feedback"]
        generation = self.generate(
            [ep.source.prompt],
            max_new_tokens=self.config["source_tokens"],
            sample=True,
            seed=seed_for(self.config["seed"], ep.episode_id, "source"),
        )[0]
        result = grade(generation.text, ep.source)
        feedback = "The submitted answer was " + ("correct." if result["success"] else "incorrect.")
        write_json(
            path,
            {
                "task_id": ep.source.task_id,
                "generation": dataclasses.asdict(generation),
                "grade": result,
                "feedback": feedback,
            },
        )
        return generation, feedback

    def proposals(self, ep, generation, feedback):
        proposals = []
        for candidate in range(self.config["candidate_count"]):
            path = self.output / "proposals" / f"{ep.episode_id}-{candidate}.json"
            if path.exists():
                proposals.append(read_json(path)["text"])
                continue
            prompt = proposal_prompt(ep.source.prompt, generation.text, feedback)
            value = self.generate(
                [prompt],
                max_new_tokens=self.config["pi_tokens"],
                sample=True,
                seed=seed_for(self.config["seed"], ep.episode_id, "pi", candidate),
                context_limit=self.config["max_teacher_context_tokens"],
            )[0]
            write_json(path, {"text": value.text, "cap_hit": value.cap_hit})
            proposals.append(value.text)
        return proposals

    def evaluate(self, ep, tasks, role, condition, candidate=None, pi=None, adapter=None):
        from hindsight.data import grade
        from hindsight.model import with_pi

        wanted = [self.metadata(ep, task, role, condition, candidate) for task in tasks]
        teacher_hash = self.validate_teacher() if condition == "adapted_bidirectional" else None
        if teacher_hash is not None:
            for meta in wanted:
                cached = self.observations.get(observation_key(meta))
                if cached is not None and cached.get("teacher_checkpoint_sha256") != teacher_hash:
                    raise ValueError("Cached outcome belongs to a different teacher checkpoint")
        missing = [
            (task, meta)
            for task, meta in zip(tasks, wanted)
            if observation_key(meta) not in self.observations
        ]
        batch_size = self.config["generation_batch_size"]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            before = time.monotonic()
            generated = self.generate(
                [with_pi(task.prompt, pi) if pi else task.prompt for task, _ in batch],
                max_new_tokens=self.config["evaluation_tokens"],
                sample=False,
                seed=self.config["seed"],
                adapter=adapter,
            )
            for (task, meta), value in zip(batch, generated):
                record = {
                    **meta,
                    **grade(value.text, task),
                    "cap_hit": value.cap_hit,
                    "text": value.text,
                    "generated_tokens": len(value.response_ids),
                    "batch_elapsed_seconds": time.monotonic() - before,
                }
                if teacher_hash is not None:
                    record["teacher_checkpoint_sha256"] = teacher_hash
                append_json(self.output / "observations.jsonl", record)
                self.observations[observation_key(meta)] = record
        return [self.observations[observation_key(meta)] for meta in wanted]

    def episode_expected(self, ep, phase):
        groups = []
        if phase != "development_hindsight":
            for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                if role == "Q" and phase == "meta_train":
                    continue
                groups.append((tasks, role, "baseline", None))
                if phase != "meta_train":
                    groups += [(tasks, role, c, None) for c in ("generic", "mismatched_pi")]
                for candidate in range(self.config["candidate_count"]):
                    groups += [(tasks, role, c, candidate) for c in ("direct_pi", "adapted_pi")]
        if phase in ("development_hindsight", "final"):
            groups += [(ep.queries, "Q", c, None) for c in EXTRAS]
        return [self.metadata(ep, task, role, condition, candidate)
                for tasks, role, condition, candidate in groups for task in tasks]

    def prepare_expected(self, episodes, phase):
        for ep in episodes:
            for expected in self.episode_expected(ep, phase):
                self.expect(expected)

    def score(self, generation, feedback, **kwargs):
        self.check_time()
        target = self.get_engine().score_rollout(generation, feedback, **kwargs)
        self.event(
            "teacher_scored",
            mode=kwargs.get("mode", "causal"),
            action_positions=len(generation.response_ids),
        )
        return target

    def adaptation(self, ep, generation, targets, condition, candidate=None,
                   initial_logprobs_callback=None):
        self.check_time()
        engine = self.get_engine()
        engine.reset_student(seed=seed_for(self.config["seed"], "student_initialization"))
        metrics = engine.adapt(
            generation,
            targets,
            steps=self.config["student_steps"],
            lr=self.config["learning_rate"],
            initial_logprobs_callback=initial_logprobs_callback,
        )
        self.event(
            "student_adapted",
            episode_id=ep.episode_id,
            condition=condition,
            candidate=candidate,
            metrics=metrics,
        )
        self.check_time()

    def selection(self, ep, baseline, candidates, condition):
        selected = choose_candidate(
            baseline, candidates, seed_for(self.config["seed"], ep.episode_id, condition)
        )
        selection = {
            "episode_id": ep.episode_id,
            "partition": ep.partition,
            "family": ep.family,
            "condition": condition,
            "selected_candidate": selected,
            "derived_from": "V",
            "baseline_successes": sum(row["success"] for row in baseline),
            "candidate_successes": [sum(row["success"] for row in x) for x in candidates],
        }
        prior = [
            row
            for row in rows(self.output / "selections.jsonl")
            if row["episode_id"] == ep.episode_id and row["condition"] == condition
        ]
        if prior and (len(prior) != 1 or prior[0] != selection):
            raise ValueError("Selection identity/outcome changed")
        if not prior:
            append_json(self.output / "selections.jsonl", selection)
        return selected

    def reference_identity(self, ep, generation, feedback, winner):
        return {
            "episode_id": ep.episode_id,
            "partition": ep.partition,
            "family": ep.family,
            "identity_digest": digest(self.identity),
            "source_digest": digest([dataclasses.asdict(generation), feedback]),
            "selected_candidate": winner,
            "derived_from": "post_update_V",
            "reference_kind": "frozen_causal_no_update" if winner is None else "PI_teacher",
        }

    def development_reference(self, ep, generation, feedback, selected=None, winner=None):
        """Only development crosses phases; final references remain in memory."""
        import torch

        selections = [row for row in rows(self.output / "selections.jsonl")
                      if row["episode_id"] == ep.episode_id and row["condition"] == "adapted_pi"]
        if len(selections) != 1 or selections[0]["derived_from"] != "V":
            raise ValueError("Fidelity reference requires one validation-derived selection")
        chosen = selections[0]["selected_candidate"]
        if selected is not None and winner != chosen:
            raise ValueError("Fidelity reference selection mismatch")
        identity = self.reference_identity(ep, generation, feedback, chosen)
        path = self.output / "fidelity-references" / f"{ep.episode_id}.pt"
        receipt_path = path.with_suffix(".json")
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["reference"] != identity:
                raise ValueError("Fidelity reference source or selection changed")
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["tensor_sha256"]:
                raise ValueError("Fidelity reference tensor changed")
            return torch.load(path, map_location="cpu", weights_only=True), identity
        if selected is None:
            raise ValueError("Development fidelity reference is incomplete")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        torch.save(selected.detach().cpu(), temporary)
        temporary.replace(path)
        write_json(receipt_path, {"reference": identity,
                                 "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return selected, identity

    def record_fidelity(self, ep, generation, reference, reference_identity, prediction):
        provenance = {**reference_identity, "teacher_checkpoint_sha256": self.validate_teacher()}
        path = self.output / "teacher-fidelity" / f"{ep.episode_id}.json"
        if path.exists():
            if read_json(path)["provenance"] != provenance:
                raise ValueError("Fidelity evidence source, selection, or teacher changed")
            return  # One source-level record, including across incomplete-phase resumes.
        write_json(path, {"provenance": provenance,
                          "metrics": teacher_fidelity(reference, prediction,
                                                      generation.response_ids)})

    def run_episode(self, ep, phase, generation, feedback, proposals, mismatch=None):
        import torch

        self.event("episode_started", episode_id=ep.episode_id, partition=ep.partition, phase=phase)
        reference = reference_identity = None
        if phase != "development_hindsight":
            baseline = self.evaluate(ep, ep.validation, "V", "baseline")
            if phase != "meta_train":
                self.evaluate(ep, ep.queries, "Q", "baseline")
                for role, tasks in (("V", ep.validation), ("Q", ep.queries)):
                    self.evaluate(ep, tasks, role, "generic", pi=GENERIC)
                    self.evaluate(ep, tasks, role, "mismatched_pi", pi=mismatch)
            direct, adapted, targets, initial = [], [], [], []
            for candidate, pi in enumerate(proposals):
                direct.append(self.evaluate(ep, ep.validation, "V", "direct_pi", candidate, pi=pi))
                if phase != "meta_train":
                    self.evaluate(ep, ep.queries, "Q", "direct_pi", candidate, pi=pi)
                target = self.score(generation, feedback, pi=pi)
                self.adaptation(
                    ep, generation, target, "adapted_pi", candidate,
                    initial_logprobs_callback=(
                        initial.append if phase != "meta_train" and candidate == 0 else None
                    ),
                )
                adapted.append(
                    self.evaluate(
                        ep, ep.validation, "V", "adapted_pi", candidate, adapter="student"
                    )
                )
                if phase != "meta_train":
                    self.evaluate(ep, ep.queries, "Q", "adapted_pi", candidate, adapter="student")
                targets.append(target)
            self.selection(ep, baseline, direct, "direct_pi")
            winner = self.selection(ep, baseline, adapted, "adapted_pi")
            if phase == "meta_train":
                selected = (
                    targets[winner] if winner is not None else self.score(generation, feedback)
                )
                path = self.output / "targets" / f"{ep.episode_id}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".partial")
                torch.save(selected, temporary)
                temporary.replace(path)
                write_json(
                    path.with_suffix(".json"),
                    {
                        "episode_id": ep.episode_id,
                        "selected_candidate": winner,
                        "identity_digest": digest(self.identity),
                        "derived_from": "post_update_V",
                        "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                )
            else:
                if len(initial) != 1:
                    raise ValueError("Missing initial causal distribution for fidelity reference")
                reference = targets[winner] if winner is not None else initial[0]
                reference_identity = self.reference_identity(ep, generation, feedback, winner)
                if phase == "development":
                    self.development_reference(ep, generation, feedback, reference, winner)
            del targets
        if phase in ("development_hindsight", "final"):
            if phase == "development_hindsight":
                reference, reference_identity = self.development_reference(ep, generation, feedback)
            for condition in EXTRAS:
                if condition == "adapted_bidirectional":
                    target = self.score(
                        generation, feedback, mode="bidirectional", adapter="teacher"
                    )
                    self.record_fidelity(ep, generation, reference, reference_identity, target)
                elif condition == "adapted_untrained_bidirectional":
                    target = self.score(generation, feedback, mode="bidirectional")
                elif condition == "adapted_raw_causal":
                    target = self.score(generation, feedback, mode="raw_causal")
                else:
                    vocabulary = self.get_engine().model.config.vocab_size
                    target = torch.full((len(generation.response_ids), vocabulary), -torch.inf)
                    target.scatter_(1, torch.tensor(generation.response_ids).unsqueeze(1), 0.0)
                self.adaptation(ep, generation, target, condition)
                self.evaluate(ep, ep.queries, "Q", condition, adapter="student")
        self.event("episode_completed", episode_id=ep.episode_id, phase=phase)

    def code_identity(self):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }

    def freeze(self):
        receipt = self.output / "teacher" / "FIT-COMPLETE.json"
        if not receipt.exists():
            raise ValueError("Cannot freeze without completed teacher fit")
        completed = {
            row["phase"]
            for row in rows(self.output / "events.jsonl")
            if row["event"] == "phase_completed"
        }
        if not {"development", "meta_train", "development_hindsight"} <= completed:
            raise ValueError("Required development phases have not completed")
        if self.expected - self.observations.keys():
            raise ValueError("Planned observations are incomplete")
        self.validate_teacher()
        protocol = {
            "identity_digest": digest(self.identity),
            "code": self.code_identity(),
            "fit_receipt": read_json(receipt),
        }
        path = self.output / "FINAL-PROTOCOL.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Cannot overwrite a frozen protocol")
        write_json(path, protocol)
        self.event("final_protocol_frozen", digest=digest(protocol))

    def verify_final(self):
        protocol = read_json(self.output / "FINAL-PROTOCOL.json")
        if protocol["identity_digest"] != digest(self.identity):
            raise ValueError("Final run identity mismatch")
        if protocol["code"] != self.code_identity():
            raise ValueError("Code changed after final protocol freeze")
        if protocol["fit_receipt"] != read_json(self.output / "teacher" / "FIT-COMPLETE.json"):
            raise ValueError("Teacher changed after protocol freeze")
        self.validate_teacher()

    def validate_teacher(self):
        receipt = read_json(self.output / "teacher" / "FIT-COMPLETE.json")
        if receipt["identity_digest"] != digest(self.identity):
            raise ValueError("Teacher run identity mismatch")
        if "checkpoint.pt" not in receipt["checkpoint_files"]:
            raise ValueError("Teacher receipt does not bind checkpoint.pt")
        for filename, expected in receipt["checkpoint_files"].items():
            path = self.output / "teacher" / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Teacher checkpoint bytes changed")
        return receipt["checkpoint_files"]["checkpoint.pt"]

    def validate_completed_episode(self, ep, phase):
        """A done event is not enough: check its planned outcomes and durable artifacts."""
        expected = self.episode_expected(ep, phase)
        if any(observation_key(row) not in self.observations for row in expected):
            raise ValueError("Completed episode has missing planned observations")
        files = [self.output / "sources" / f"{ep.episode_id}.json"] + [
            self.output / "proposals" / f"{ep.episode_id}-{candidate}.json"
            for candidate in range(self.config["candidate_count"])
        ]
        if not all(path.is_file() for path in files):
            raise ValueError("Completed episode has missing source/proposal artifacts")
        generation, feedback = self.source(ep)  # Verified cache only; never regenerates here.
        selections = [row for row in rows(self.output / "selections.jsonl")
                      if row["episode_id"] == ep.episode_id and row["condition"] == "adapted_pi"]
        if len(selections) != 1 or selections[0]["derived_from"] != "V":
            raise ValueError("Completed episode has missing or duplicate selection")
        winner = selections[0]["selected_candidate"]
        if phase == "meta_train":
            path = self.output / "targets" / f"{ep.episode_id}.pt"
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise ValueError("Completed episode has missing teacher target")
            receipt = read_json(path.with_suffix(".json"))
            target_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if (receipt["identity_digest"] != digest(self.identity)
                    or receipt["selected_candidate"] != winner
                    or receipt.get("tensor_sha256") != target_hash):
                raise ValueError("Completed episode teacher target identity changed")
        if phase in ("development", "development_hindsight"):
            self.development_reference(ep, generation, feedback)
        if phase in ("development_hindsight", "final"):
            path = self.output / "teacher-fidelity" / f"{ep.episode_id}.json"
            if not path.is_file():
                raise ValueError("Completed episode has missing fidelity evidence")
            teacher_hash = self.validate_teacher()
            provenance = {**self.reference_identity(ep, generation, feedback, winner),
                          "teacher_checkpoint_sha256": teacher_hash}
            if read_json(path)["provenance"] != provenance:
                raise ValueError("Completed episode fidelity identity changed")
            for row in expected:
                if row["condition"] == "adapted_bidirectional":
                    cached_hash = self.observations[observation_key(row)].get(
                        "teacher_checkpoint_sha256"
                    )
                    if cached_hash != teacher_hash:
                        raise ValueError("Completed episode outcome teacher changed")

    def run(self, phase: str):
        if phase != "final" and (self.output / "FINAL-PROTOCOL.json").exists():
            raise ValueError("Development cannot resume after final freeze in the same run")
        if phase == "final":
            self.verify_final()
        prior = rows(self.output / "events.jsonl")
        partitions = {
            "development": {"development"},
            "meta_train": {"meta_train"},
            "development_hindsight": {"development"},
            "final": {"final_known", "final_unseen"},
        }[phase]
        episodes = [ep for ep in self.episodes if ep.partition in partitions]
        self.prepare_expected(episodes, phase)
        completed = {row["episode_id"] for row in prior
                     if row["event"] == "episode_completed" and row["phase"] == phase}
        for ep in episodes:
            if ep.episode_id in completed:
                self.validate_completed_episode(ep, phase)
        if any(row["event"] == "phase_completed" and row["phase"] == phase for row in prior):
            if any(ep.episode_id not in completed for ep in episodes):
                raise ValueError("Completed phase has missing episode completion evidence")
            raise ValueError("Phase already completed; refusing repeated compute")
        self.event("phase_started", phase=phase)
        if phase in ("development_hindsight", "final") and any(
            ep.episode_id not in completed for ep in episodes
        ):
            self.validate_teacher()
            self.get_engine().load_teacher(self.output / "teacher" / "checkpoint.pt")
        prepared = {}
        for ep in episodes:
            generation, feedback = self.source(ep)
            prepared[ep.episode_id] = (
                generation,
                feedback,
                self.proposals(ep, generation, feedback),
            )
        for ep in episodes:
            if ep.episode_id in completed:
                self.event("completed_episode_reused", episode_id=ep.episode_id, phase=phase)
                continue
            if any(row["event"] == "episode_started" and row["episode_id"] == ep.episode_id
                   and row["phase"] == phase for row in prior):
                self.event("episode_resume_reconstruction", episode_id=ep.episode_id, phase=phase,
                           prior_student_updates=sum(len(row.get("metrics", [])) for row in prior
                               if row["event"] == "student_adapted"
                               and row["episode_id"] == ep.episode_id),
                           note=("Explicit phase resume; cached generations reused. Reconstructed "
                                 "teacher forwards and student updates are recorded again as "
                                 "actual compute."))
            others = [other for other in episodes if other.family != ep.family]
            if not others:
                raise ValueError("Mismatch control requires another source family")
            other = others[seed_for(self.config["seed"], ep.episode_id, "mismatch") % len(others)]
            self.run_episode(
                ep, phase, *prepared[ep.episode_id], mismatch=prepared[other.episode_id][2][0]
            )
        self.event("phase_completed", phase=phase, episodes=len(episodes))

    def fit(self):
        import torch

        if (self.output / "FINAL-PROTOCOL.json").exists():
            raise ValueError("No teacher fitting after final freeze")
        teacher_outcomes = [
            row for row in self.observations.values()
            if row["condition"] == "adapted_bidirectional"
        ]
        if (self.output / "teacher" / "FIT-COMPLETE.json").exists():
            teacher_hash = self.validate_teacher()
            if any(
                row.get("teacher_checkpoint_sha256") != teacher_hash for row in teacher_outcomes
            ):
                raise ValueError("Existing outcomes belong to a different teacher checkpoint")
            self.event("teacher_fit_already_complete")
            return
        if teacher_outcomes:
            raise ValueError("Cannot refit: teacher-dependent outcomes lack their checkpoint")
        examples = []
        for ep in self.episodes:
            if ep.partition != "meta_train":
                continue
            path = self.output / "targets" / f"{ep.episode_id}.pt"
            if read_json(path.with_suffix(".json"))["identity_digest"] != digest(self.identity):
                raise ValueError("Target identity mismatch")
            generation, feedback = self.source(ep)
            examples.append((generation, feedback, torch.load(path, weights_only=True)))
        self.check_time()
        metrics = self.get_engine().fit_teacher(
            examples,
            epochs=self.config["teacher_epochs"],
            batch_size=self.config["teacher_batch_size"],
            lr=self.config["learning_rate"],
            seed=seed_for(self.config["seed"], "teacher_fit"),
        )
        self.get_engine().save_teacher(self.output / "teacher" / "checkpoint.pt")
        hashes = {
            str(path.relative_to(self.output / "teacher")): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (self.output / "teacher").rglob("*")
            if path.is_file()
        }
        write_json(
            self.output / "teacher" / "FIT-COMPLETE.json",
            {
                "identity_digest": digest(self.identity),
                "metrics": metrics,
                "checkpoint_files": hashes,
            },
        )
        self.event("teacher_fit_completed", metrics=metrics)

    def calibrate(self):
        from hindsight.data import grade

        planned = [{"task_id": task.task_id, "family": task.family} for task in self.calibration]
        expected_path = self.output / "calibration-expected.json"
        if expected_path.exists() and read_json(expected_path) != planned:
            raise ValueError("Calibration plan changed")
        if len({row["task_id"] for row in planned}) != len(planned):
            raise ValueError("Duplicate planned calibration task")
        expected = {row["task_id"]: row for row in planned}
        prior = rows(self.output / "calibration.jsonl")
        outcomes = {row["task_id"]: row for row in prior}
        if len(outcomes) != len(prior):
            raise ValueError("Duplicate persisted calibration task")
        if any(
            task_id not in expected or row["family"] != expected[task_id]["family"]
            for task_id, row in outcomes.items()
        ):
            raise ValueError("Persisted calibration outcome is outside the frozen plan")
        if not expected_path.exists():
            write_json(expected_path, planned)
        new_generations = 0
        for task in self.calibration:
            if task.task_id in outcomes:
                continue
            value = self.generate(
                [task.prompt],
                max_new_tokens=self.config["source_tokens"],
                sample=True,
                seed=seed_for(self.config["seed"], task.task_id, "calibration"),
            )[0]
            outcome = {
                "task_id": task.task_id,
                "family": task.family,
                **grade(value.text, task),
                "text": value.text,
                "cap_hit": value.cap_hit,
                "generated_tokens": len(value.response_ids),
            }
            append_json(self.output / "calibration.jsonl", outcome)
            outcomes[task.task_id] = outcome
            new_generations += 1
        self.event(
            "calibration_completed",
            tasks=len(outcomes),
            new_generations=new_generations,
            successes=sum(row["success"] for row in outcomes.values()),
            malformed=sum(not row["format_ok"] for row in outcomes.values()),
            cap_hits=sum(row["cap_hit"] for row in outcomes.values()),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "run", "fit", "freeze", "summarize"])
    parser.add_argument("--config", type=Path, default=Path("configs/initial.json"))
    parser.add_argument("--data", type=Path, default=Path("data/corpus-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/initial-v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--phase", choices=["development", "meta_train", "development_hindsight", "final"]
    )
    args = parser.parse_args()
    experiment = Experiment(read_json(args.config), args.data, args.output, args.device)
    try:
        if args.command == "run":
            if args.phase is None:
                parser.error("run requires --phase")
            experiment.run(args.phase)
        elif args.command == "fit":
            experiment.fit()
        elif args.command == "freeze":
            experiment.freeze()
        elif args.command == "calibrate":
            experiment.calibrate()
        else:
            from hindsight.analysis import summarize

            summary = summarize(
                rows(args.output / "observations.jsonl"),
                expected=rows(args.output / "expected.jsonl"),
                selections=rows(args.output / "selections.jsonl"),
                protocol_frozen=(args.output / "FINAL-PROTOCOL.json").exists(),
            )
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
    except Exception as error:
        experiment.event(
            "phase_failed",
            command=args.command,
            phase=args.phase,
            exception=type(error).__name__,
            message=str(error),
        )
        raise


if __name__ == "__main__":
    main()
