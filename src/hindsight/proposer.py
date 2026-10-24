"""Isolated source-only proposer policy gradients on the existing causal engine."""

from __future__ import annotations

import copy
import hashlib
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import GenerationConfig

from hindsight.model import Generation, HFEngine

PROPOSER_COUNTERS = (
    "proposer_generation_sequences",
    "proposer_scoring_forwards",
    "proposer_backward_sequences",
    "proposer_optimizer_steps",
    "proposer_groups",
    "proposer_zero_advantage_groups",
)


@dataclass
class Proposal:
    generation: Generation
    temperature: float
    seed: int
    policy_version: int
    initial: bool
    behavior_token_logprobs: list[float]


def leave_one_out_advantages(rewards) -> torch.Tensor:
    """No learned critic and no token/length normalization of the sequence objective."""
    values = torch.as_tensor(rewards, dtype=torch.float32).detach().cpu()
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("Require at least two finite scalar rewards")
    return values - (values.sum() - values) / (len(values) - 1)


def usable_lesson(proposal: Proposal) -> bool:
    text = proposal.generation.text.strip()
    return bool(
        text
        and not proposal.generation.cap_hit
        and not re.search(r"\bFINAL\s*:|\[\s*\d|</?think>", text)
    )


class ProposerEngine(HFEngine):
    """Adds two LoRA slots; source teachers and student updates retain their core paths."""

    def initialize_proposer(self, *, seed: int, lr: float = 1e-5) -> None:
        self._check_budget()
        if hasattr(self, "proposer_optimizer"):
            raise ValueError("Proposer initialization is one-shot; use a fresh engine")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Proposer learning rate must be positive and finite")
        previous = self.model.active_adapter
        lora = copy.deepcopy(self.model.peft_config["student"])
        self.model.add_adapter("proposer", lora)
        self.model.add_adapter("proposer_initial", copy.deepcopy(lora))
        self._reset_adapter("proposer", seed)
        current = self._adapter_parameters("proposer")
        initial = self._adapter_parameters("proposer_initial")
        with torch.no_grad():
            for name, value in current.items():
                initial[name.replace(".proposer.", ".proposer_initial.")].copy_(value)
        self.model.set_adapter(previous)
        for name, value in self.model.named_parameters():
            if ".proposer." in name or ".proposer_initial." in name:
                value.requires_grad_(False)
        self.proposer_optimizer = self._optimizer(list(current.values()), lr)
        self.proposer_seed, self.proposer_lr = seed, lr
        self.proposer_version = 0
        self.counters.update(dict.fromkeys(PROPOSER_COUNTERS, 0))

    @contextmanager
    def _using_adapter(self, adapter, trainable=False):
        if adapter not in ("proposer", "proposer_initial"):
            with super()._using_adapter(adapter, trainable=trainable):
                yield
            return
        if adapter == "proposer_initial" and trainable:
            raise ValueError("The frozen proposer initialization cannot be trained")
        previous_adapter, previous_training = self.model.active_adapter, self.model.training
        parameters = dict(self.model.named_parameters())
        previous_grad = {name: value.requires_grad for name, value in parameters.items()}
        try:
            self.model.set_adapter(adapter)
            for name, value in parameters.items():
                value.requires_grad_(trainable and "lora_" in name and f".{adapter}." in name)
            self.model.train(trainable)
            yield
        finally:
            self.model.set_adapter(previous_adapter)
            for name, value in parameters.items():
                value.requires_grad_(previous_grad[name])
            self.model.train(previous_training)

    def proposer_state_sha256(self, *, initial=False):
        hasher = hashlib.sha256()
        adapter = "proposer_initial" if initial else "proposer"
        for name, value in sorted(self._adapter_parameters(adapter).items()):
            self._check_budget()
            tensor = value.detach().cpu().contiguous()
            hasher.update(name.replace(f".{adapter}.", ".proposer.").encode())
            hasher.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return hasher.hexdigest()

    def sample_proposal(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        initial: bool = False,
    ) -> Proposal:
        self._check_budget()
        if (
            not hasattr(self, "proposer_optimizer")
            or not isinstance(prompt, str)
            or not prompt
            or type(max_new_tokens) is not int
            or max_new_tokens < 1
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Require initialized proposer, prompt, positive cap and temperature")
        prefix = self.encode_prompt(prompt)
        self._check_length(len(prefix) + max_new_tokens, self.max_teacher_context, "Proposer")
        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        # A fresh config excludes inherited top-k/top-p, repetition penalties, forced
        # tokens and other transforms. Policy scoring uses this same temperature.
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            typical_p=1.0,
            repetition_penalty=1.0,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        adapter = "proposer_initial" if initial else "proposer"
        with self._using_adapter(adapter), self._seeded(seed), torch.no_grad():
            output = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=config,
            )
        response = output.sequences[0, len(prefix) :].tolist()
        ended = self.eos_token_id in response
        if ended:
            response = response[: response.index(self.eos_token_id) + 1]
        if not response or len(output.scores) < len(response):
            raise ValueError("Missing native proposer response or generation scores")
        behavior = [
            float(score[0].float().log_softmax(-1)[token])
            for score, token in zip(output.scores, response)
        ]
        if not all(math.isfinite(value) for value in behavior):
            raise FloatingPointError("Nonfinite sampled proposer log probabilities")
        value = Proposal(
            Generation(
                prompt,
                prefix,
                response,
                self.tokenizer.decode(response, skip_special_tokens=True),
                len(response) >= max_new_tokens and not ended,
            ),
            temperature,
            seed,
            self.proposer_version,
            initial,
            behavior,
        )
        self.counters["generation_sequences"] += 1
        self.counters["proposer_generation_sequences"] += 1
        self._check_budget()
        return value

    def proposal_token_logprobs(self, proposal: Proposal):
        generation = proposal.generation
        if (
            generation.prompt_ids != self.encode_prompt(generation.prompt)
            or not generation.response_ids
            or any(type(token) is not int or token < 0 for token in generation.response_ids)
        ):
            raise ValueError("Proposer likelihood requires exact native prompt/response IDs")
        tokens, positions, bidirectional = self._scoring_input(generation, "", None, "causal")
        if bidirectional:
            raise ValueError("Proposer likelihood must remain causal")
        # logsoftmax(logsoftmax(logits)/T) equals logsoftmax(logits/T).
        distribution = (
            self._logprobs(tokens, positions, False) / proposal.temperature
        ).log_softmax(-1)
        actions = torch.tensor(generation.response_ids, device=self.device).unsqueeze(1)
        result = distribution.gather(1, actions).squeeze(1)
        self.counters["proposer_scoring_forwards"] += 1
        return result

    def reinforce_group(self, proposals: list[Proposal], rewards) -> dict:
        self._check_budget()
        if len(proposals) != 4:
            raise ValueError("The declared proposer group has exactly four candidates")
        advantages = leave_one_out_advantages(rewards)
        if len(advantages) != 4:
            raise ValueError("Every candidate requires its own scalar reward")
        first = proposals[0]
        if (
            any(p.initial or p.policy_version != self.proposer_version for p in proposals)
            or any(
                p.generation.prompt != first.generation.prompt
                or p.generation.prompt_ids != first.generation.prompt_ids
                or p.temperature != first.temperature
                for p in proposals
            )
            or len({p.seed for p in proposals}) != 4
        ):
            raise ValueError("Require one fresh on-policy source group with independent seeds")
        values = torch.as_tensor(rewards, dtype=torch.float32).detach().cpu()
        metrics = dict(
            rewards=values.tolist(),
            advantages=advantages.tolist(),
            policy_version_before=self.proposer_version,
            optimizer_steps=0,
            sequence_lengths=[len(p.generation.response_ids) for p in proposals],
            objective="mean(-leave_one_out_advantage * sequence_logprob_sum)",
            temperature=first.temperature,
            loss=0.0,
            grad_norm=0.0,
            cached_vs_teacher_forced_max_logprob_delta=None,
        )
        self.model.zero_grad(set_to_none=True)
        if torch.count_nonzero(advantages) == 0:
            self.counters["proposer_zero_advantage_groups"] += 1
        else:
            losses, deltas = [], []
            with self._using_adapter("proposer", trainable=True):
                self.proposer_optimizer.zero_grad(set_to_none=True)
                for proposal, advantage in zip(proposals, advantages, strict=True):
                    self._check_budget()
                    logprobs = self.proposal_token_logprobs(proposal)
                    loss = -advantage.to(self.device) * logprobs.sum() / len(proposals)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite proposer policy-gradient loss")
                    loss.backward()
                    self.counters["proposer_backward_sequences"] += 1
                    losses.append(float(loss.detach()))
                    behavior = torch.tensor(proposal.behavior_token_logprobs, device=self.device)
                    if behavior.shape != logprobs.shape:
                        raise ValueError(
                            "Behavior log probabilities do not align with native tokens"
                        )
                    deltas.append(float((behavior - logprobs.detach()).abs().max()))
                norm = torch.nn.utils.clip_grad_norm_(
                    list(self._adapter_parameters("proposer").values()),
                    1.0,
                    error_if_nonfinite=True,
                )
                self.proposer_optimizer.step()
                self.counters["proposer_optimizer_steps"] += 1
            metrics.update(
                optimizer_steps=1,
                loss=sum(losses),
                grad_norm=float(norm),
                cached_vs_teacher_forced_max_logprob_delta=max(deltas),
            )
        self.proposer_version += 1
        self.counters["proposer_groups"] += 1
        metrics["policy_version_after"] = self.proposer_version
        self._check_budget()
        return metrics

    def save_proposer(self, path: Path, *, metadata=None):
        self._check_budget()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format_version=1,
            model_name=self.model_name,
            revision=self.revision,
            seed=self.proposer_seed,
            lr=self.proposer_lr,
            policy_version=self.proposer_version,
            counters=dict(self.counters),
            metadata=metadata or {},
            optimizer=self.proposer_optimizer.state_dict(),
        )
        for name in ("proposer", "proposer_initial"):
            payload[name] = {
                key: value.detach().cpu()
                for key, value in get_peft_model_state_dict(
                    self.model, adapter_name=name, save_embedding_layers=False
                ).items()
            }
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        self._check_budget()

    def load_proposer(self, path: Path):
        """Load a complete checkpoint into initialized slots; validate before mutation."""
        self._check_budget()
        if not hasattr(self, "proposer_optimizer"):
            raise ValueError("Initialize proposer slots before loading")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format_version") != 1
            or payload["model_name"] != self.model_name
            or payload["revision"] != self.revision
            or payload["seed"] != self.proposer_seed
            or payload["lr"] != self.proposer_lr
        ):
            raise ValueError("Proposer checkpoint identity differs from this engine")
        for adapter in ("proposer", "proposer_initial"):
            expected = get_peft_model_state_dict(
                self.model, adapter_name=adapter, save_embedding_layers=False
            )
            supplied = payload[adapter]
            if not isinstance(supplied, dict) or supplied.keys() != expected.keys():
                raise ValueError("Proposer checkpoint tensor names differ")
            if any(
                not isinstance(supplied[name], torch.Tensor)
                or supplied[name].shape != value.shape
                or supplied[name].dtype != value.dtype
                or not torch.isfinite(supplied[name]).all()
                for name, value in expected.items()
            ):
                raise ValueError("Proposer checkpoint tensor shape/dtype/value differs")
        if type(payload["policy_version"]) is not int or payload["policy_version"] < 0:
            raise ValueError("Proposer checkpoint policy version is invalid")
        validated_optimizer = copy.deepcopy(self.proposer_optimizer)
        validated_optimizer.load_state_dict(payload["optimizer"])
        for parameter, state in validated_optimizer.state.items():
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and (
                    not torch.isfinite(value).all()
                    or (name in ("exp_avg", "exp_avg_sq") and value.shape != parameter.shape)
                    or (name == "step" and value.numel() != 1)
                ):
                    raise ValueError("Proposer checkpoint optimizer tensor is invalid")
        for adapter in ("proposer", "proposer_initial"):
            result = set_peft_model_state_dict(self.model, payload[adapter], adapter_name=adapter)
            if result.unexpected_keys:
                raise ValueError("Unexpected proposer checkpoint keys")
        self.proposer_optimizer.load_state_dict(payload["optimizer"])
        self.proposer_version = payload["policy_version"]
        return payload
