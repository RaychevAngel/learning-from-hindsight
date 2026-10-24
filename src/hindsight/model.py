"""Small, explicit model operations for the hindsight experiments.

Production construction loads the pinned checkpoint. Passing both ``config`` and
``tokenizer`` constructs a random, offline model for numerical unit tests.
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
QWEN35_LORA_TARGETS = LORA_TARGETS + [
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"
]


def configure_deterministic_execution(enabled: bool = False) -> dict[str, Any]:
    """Opt in before CUDA initialization; otherwise leave process settings alone."""
    if enabled:
        workspace = ":4096:8"
        if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
            raise RuntimeError("Configure deterministic execution before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
    return {
        "requested": bool(enabled),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


@dataclass
class Generation:
    prompt: str
    prompt_ids: list[int]
    response_ids: list[int]
    text: str
    cap_hit: bool


def with_pi(prompt: str, pi: str) -> str:
    """The identical PI intervention for direct evaluation and prefix rescoring."""
    return (
        "Use the following hindsight advice when solving the task. "
        "The advice may be imperfect; follow the task's output format.\n"
        f"<hindsight_advice>\n{pi}\n</hindsight_advice>\n\n"
        f"{prompt}"
    )


def forward_kl(prediction_logprobs: Tensor, target_logprobs: Tensor) -> Tensor:
    """Action-mean, full-vocabulary KL(target || prediction), detached target.

    Probability-target KL handles exact one-hot replay targets (log 0 = -inf)
    without the 0 * -inf NaNs of a naive log-target implementation.
    """
    if prediction_logprobs.shape != target_logprobs.shape:
        raise ValueError("Prediction and target distributions must have the same shape")
    target = target_logprobs.detach().to(
        device=prediction_logprobs.device, dtype=torch.float32
    )
    if torch.equal(prediction_logprobs.detach(), target):
        # At the exact KL minimum the gradient is exactly zero. Floating-point
        # probability summation in the generic KL backward can otherwise leave
        # tiny residual gradients that Adam amplifies on a neutral target.
        return prediction_logprobs.sum() * 0.0
    target_probs = target.exp()
    return F.kl_div(
        prediction_logprobs.float(), target_probs, reduction="none"
    ).sum(dim=-1).mean()


class HFEngine:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        device: str = "cuda",
        *,
        config: Any | None = None,
        tokenizer: Any | None = None,
        dtype: torch.dtype | None = None,
        max_generation_context: int = 1536,
        max_teacher_context: int = 4096,
        budget_check: Callable[[], None] | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.budget_check = budget_check
        self.enable_thinking = enable_thinking
        self._check_budget()
        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(device)
        self.max_generation_context = max_generation_context
        self.max_teacher_context = max_teacher_context
        dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        if config is not None and tokenizer is None:
            raise ValueError("Offline config construction requires an explicit tokenizer")
        checkpoint_config = config if config is not None else AutoConfig.from_pretrained(
            model_name, revision=revision
        )
        text_config = checkpoint_config.get_text_config()
        self.model_type = text_config.model_type
        is_qwen35 = self.model_type == "qwen3_5_text"
        if is_qwen35:
            # Import lazily so the original Qwen3 environment remains supported.
            from transformers import Qwen3_5ForCausalLM

            model_class = Qwen3_5ForCausalLM
        else:
            model_class = Qwen3ForCausalLM
        if config is not None:
            if not is_qwen35 and self.model_type != "qwen3":
                raise ValueError("Offline construction supports Qwen3 and Qwen3.5 text configs")
            text_config._attn_implementation = "sdpa"
            base = model_class(text_config).to(device=self.device, dtype=dtype)
        elif is_qwen35:
            base = model_class.from_pretrained(
                model_name, revision=revision, config=text_config, dtype=dtype,
                attn_implementation="sdpa",
            ).to(self.device)
        else:
            base = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, torch_dtype=dtype, attn_implementation="sdpa"
            ).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name, revision=revision
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.eos_token_id is None or self.pad_token_id is None:
            raise ValueError("This engine requires explicit EOS and padding token IDs")
        self.lora_targets = list(QWEN35_LORA_TARGETS if is_qwen35 else LORA_TARGETS)
        self.lora_module_coverage = {
            target: [name for name, module in base.named_modules()
                     if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] == target]
            for target in self.lora_targets
        }
        if is_qwen35 and any(not names for names in self.lora_module_coverage.values()):
            raise ValueError("Qwen3.5 checkpoint does not expose the declared LoRA projections")
        self._teacher_mask_keys = (
            ("full_attention", "linear_attention") if is_qwen35 else ("full_attention",)
        )
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=self.lora_targets, task_type="CAUSAL_LM",
        )
        with self._seeded(0):
            self.model = get_peft_model(base, lora, adapter_name="student")
            self.model.add_adapter("teacher", lora)
        self.model.eval()
        self._student_snapshot_seed: int | None = None
        self._student_snapshot: dict[str, Tensor] = {}
        self._student_optimizer: torch.optim.Optimizer | None = None
        self._student_ready = False
        self.teacher_fit_metrics: dict[str, Any] = {}
        self.counters = {
            "generation_sequences": 0,
            "scoring_forwards": 0,
            "student_optimizer_steps": 0,
            "teacher_optimizer_steps": 0,
            "teacher_example_exposures": 0,
        }
        self.reset_student(0)
        self._check_budget()

    def _check_budget(self) -> None:
        if self.budget_check is not None:
            self.budget_check()

    @contextmanager
    def _seeded(self, seed: int) -> Iterator[None]:
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    def _adapter_parameters(self, adapter: str) -> dict[str, torch.nn.Parameter]:
        return {
            name: parameter for name, parameter in self.model.named_parameters()
            if "lora_" in name and f".{adapter}." in name
        }

    @contextmanager
    def _using_adapter(self, adapter: str | None, trainable: bool = False) -> Iterator[None]:
        if adapter not in (None, "student", "teacher"):
            raise ValueError(f"Unknown adapter: {adapter}")
        previous_adapter = self.model.active_adapter
        previous_training = self.model.training
        parameters = dict(self.model.named_parameters())
        previous_grad = {name: p.requires_grad for name, p in parameters.items()}
        try:
            if adapter is not None:
                self.model.set_adapter(adapter)
            for name, parameter in parameters.items():
                parameter.requires_grad_(
                    trainable and "lora_" in name and f".{adapter}." in name
                )
            self.model.train(trainable)
            if adapter is None:
                with self.model.disable_adapter():
                    yield
            else:
                yield
        finally:
            self.model.set_adapter(previous_adapter)
            for name, parameter in parameters.items():
                parameter.requires_grad_(previous_grad[name])
            self.model.train(previous_training)

    def _chat_ids(self, messages: list[dict[str, str]], generation_prompt: bool) -> list[int]:
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=generation_prompt,
            return_dict=False,
            **template_kwargs,
        )
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
            raise TypeError("Tokenizer chat template must return one list of token IDs")
        return ids

    def encode_prompt(self, prompt: str) -> list[int]:
        return self._chat_ids([{"role": "user", "content": prompt}], True)

    def _check_length(self, length: int, requested_limit: int, label: str) -> None:
        limit = min(requested_limit, self.model.config.max_position_embeddings)
        if length > limit:
            raise ValueError(
                f"{label} length {length} exceeds context limit {limit}; not truncated"
            )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        sample: bool,
        seed: int,
        adapter: str | None = None,
        *,
        context_limit: int | None = None,
    ) -> list[Generation]:
        self._check_budget()
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not prompts:
            return []
        prompt_ids = [self.encode_prompt(prompt) for prompt in prompts]
        limit = context_limit or self.max_generation_context
        for ids in prompt_ids:
            self._check_length(len(ids) + max_new_tokens, limit, "Generation")
        width = max(map(len, prompt_ids))
        ids = torch.full(
            (len(prompts), width), self.pad_token_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros_like(ids)
        for index, tokens in enumerate(prompt_ids):
            ids[index, -len(tokens):] = torch.tensor(tokens, device=self.device)
            mask[index, -len(tokens):] = 1
        kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, do_sample=sample,
            pad_token_id=self.pad_token_id, eos_token_id=self.eos_token_id, use_cache=True,
        )
        if sample:
            kwargs.update(temperature=0.7, top_p=0.8, top_k=20)
        else:
            # Clear any sampling defaults shipped with the checkpoint. Greedy
            # evaluation should neither sample nor emit misleading config warnings.
            kwargs.update(temperature=None, top_p=None, top_k=None)
        with self._using_adapter(adapter), self._seeded(seed), torch.no_grad():
            output = self.model.generate(input_ids=ids, attention_mask=mask, **kwargs)
        results = []
        for prompt, original_ids, row in zip(prompts, prompt_ids, output, strict=True):
            response = row[width:].tolist()
            ended = self.eos_token_id in response
            if ended:
                response = response[:response.index(self.eos_token_id) + 1]
            results.append(Generation(
                prompt=prompt, prompt_ids=original_ids, response_ids=response,
                text=self.tokenizer.decode(response, skip_special_tokens=True),
                cap_hit=len(response) >= max_new_tokens and not ended,
            ))
        self.counters["generation_sequences"] += len(results)
        self._check_budget()
        return results

    def _continuation_tail(
        self, generation: Generation, user_message: str, generation_prompt: bool
    ) -> list[int]:
        # Derive role delimiters from the pinned template using an EMPTY assistant
        # message. Never decode/re-tokenize the original response for alignment.
        prefix = generation.prompt_ids
        if prefix != self.encode_prompt(generation.prompt):
            raise ValueError("Generation prompt IDs do not match the current chat template")
        empty_completed = self._chat_ids([
            {"role": "user", "content": generation.prompt},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": user_message},
        ], generation_prompt)
        if empty_completed[:len(prefix)] == prefix:
            tail = empty_completed[len(prefix):]
        else:
            # Thinking-capable templates add a prefill to the generation prompt,
            # but may omit it when rendering an earlier assistant history turn.
            # Verify the original prompt, then take only the empty turn's closing
            # delimiter and subsequent user turn. Never reconstruct response IDs.
            user_prefix = self._chat_ids([{"role": "user", "content": generation.prompt}], False)
            if empty_completed[:len(user_prefix)] != user_prefix:
                raise ValueError("Chat template changed the original user-turn prefix")
            assistant_tail = empty_completed[len(user_prefix):]
            if self.eos_token_id not in assistant_tail:
                raise ValueError("Chat template has no empty-assistant closing token")
            tail = assistant_tail[assistant_tail.index(self.eos_token_id):]
        if generation.response_ids and generation.response_ids[-1] == self.eos_token_id:
            if not tail or tail[0] != self.eos_token_id:
                raise ValueError("Unexpected chat template assistant-closing token")
            tail = tail[1:]
        return tail

    def _scoring_input(
        self,
        generation: Generation,
        feedback: str,
        pi: str | None,
        mode: str,
    ) -> tuple[list[int], list[int], bool]:
        if not generation.prompt_ids or not generation.response_ids:
            raise ValueError("A rollout needs nonempty prompt and response token IDs")
        response = generation.response_ids
        if mode == "causal":
            prefix = generation.prompt_ids if pi is None else self.encode_prompt(
                with_pi(generation.prompt, pi)
            )
            tokens = prefix + response
        elif mode == "bidirectional":
            if pi is not None:
                raise ValueError("The bidirectional teacher must not receive PI")
            prefix = generation.prompt_ids
            tail = self._continuation_tail(
                generation, f"Observed feedback: {feedback}", False
            )
            tokens = prefix + response + tail
        elif mode == "raw_causal":
            if pi is not None:
                raise ValueError("The raw causal control must not receive PI")
            tail = self._continuation_tail(
                generation,
                f"Observed feedback: {feedback}\n\n"
                "Revisit the original task using this completed attempt as hindsight.\n\n"
                + generation.prompt,
                True,
            )
            # First response copy is privileged raw context; the second is the
            # original action sequence whose causal prefixes are being scored.
            prefix = generation.prompt_ids + response + tail
            tokens = prefix + response
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")
        positions = list(range(len(prefix) - 1, len(prefix) + len(response) - 1))
        self._check_length(len(tokens), self.max_teacher_context, "Teacher/student scoring")
        return tokens, positions, mode == "bidirectional"

    def _logprobs(self, tokens: list[int], positions: list[int], bidirectional: bool) -> Tensor:
        kwargs: dict[str, Any] = {}
        if bidirectional:
            kwargs.update(
                attention_mask=dict.fromkeys(self._teacher_mask_keys), is_causal=False
            )
        output = self.model(
            input_ids=torch.tensor([tokens], dtype=torch.long, device=self.device),
            logits_to_keep=torch.tensor(positions, dtype=torch.long, device=self.device),
            use_cache=False, past_key_values=None, **kwargs,
        )
        return output.logits[0].float().log_softmax(dim=-1)

    def score_rollout(
        self,
        generation: Generation,
        feedback: str,
        pi: str | None = None,
        mode: str = "causal",
        adapter: str | None = None,
    ) -> Tensor:
        self._check_budget()
        tokens, positions, bidirectional = self._scoring_input(generation, feedback, pi, mode)
        with self._using_adapter(adapter), torch.no_grad():
            result = self._logprobs(tokens, positions, bidirectional).detach().cpu()
        self.counters["scoring_forwards"] += 1
        self._check_budget()
        return result

    def _reset_adapter(self, adapter: str, seed: int) -> None:
        with self._seeded(seed), torch.no_grad():
            for module in self.model.modules():
                if hasattr(module, "reset_lora_parameters") and adapter in module.lora_A:
                    module.reset_lora_parameters(adapter, init_lora_weights=True)

    @staticmethod
    def _optimizer(parameters: Sequence[torch.nn.Parameter], lr: float) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

    def reset_student(self, seed: int) -> None:
        # Cache only the current seed's initial adapter, not one 132MB copy per episode.
        if seed != self._student_snapshot_seed:
            self._reset_adapter("student", seed)
            self._student_snapshot = {
                name: p.detach().cpu().clone()
                for name, p in self._adapter_parameters("student").items()
            }
            self._student_snapshot_seed = seed
        else:
            with torch.no_grad():
                for name, parameter in self._adapter_parameters("student").items():
                    parameter.copy_(self._student_snapshot[name])
        self.model.zero_grad(set_to_none=True)
        self._student_optimizer = self._optimizer(
            list(self._adapter_parameters("student").values()), 1e-4
        )
        self._student_ready = True

    def _validate_target(self, generation: Generation, target: Tensor) -> None:
        if not generation.response_ids:
            raise ValueError("A target requires at least one original response token")
        expected = (len(generation.response_ids), self.model.config.vocab_size)
        if tuple(target.shape) != expected:
            raise ValueError(f"Target shape {tuple(target.shape)} != {expected}")
        detached = target.detach().float()
        if torch.isnan(detached).any() or torch.isposinf(detached).any():
            raise ValueError("Target log probabilities contain NaN or positive infinity")
        normalization = detached.logsumexp(dim=-1)
        if not torch.isfinite(normalization).all() or normalization.abs().max() > 1e-3:
            raise ValueError("Target log probabilities are not normalized")

    def adapt(
        self,
        generation: Generation,
        target_logprobs: Tensor,
        steps: int,
        lr: float,
        *,
        initial_logprobs_callback: Callable[[Tensor], None] | None = None,
    ) -> list[dict[str, float | int]]:
        self._check_budget()
        if not self._student_ready or self._student_optimizer is None:
            raise RuntimeError("Call reset_student(seed) before every independent adaptation")
        if steps < 1 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Adaptation requires positive steps and finite positive learning rate")
        self._validate_target(generation, target_logprobs)
        tokens, positions, _ = self._scoring_input(generation, "", None, "causal")
        target = target_logprobs.detach().to(self.device, dtype=torch.float32)
        optimizer = self._student_optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = []
        self._student_ready = False
        with self._using_adapter("student", trainable=True):
            parameters = list(self._adapter_parameters("student").values())
            for step in range(steps):
                self._check_budget()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prediction = self._logprobs(tokens, positions, False)
                if step == 0 and initial_logprobs_callback is not None:
                    initial_logprobs_callback(prediction.detach().to(device="cpu", copy=True))
                loss = forward_kl(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite student KL")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                self.counters["student_optimizer_steps"] += 1
                metrics.append({
                    "step": step + 1, "optimizer_steps": 1, "example_exposures": 1,
                    "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                    "lr": lr, "elapsed_seconds": time.perf_counter() - start,
                })
                self._check_budget()
        return metrics

    def fit_teacher(
        self,
        examples: list[tuple[Generation, str, Tensor]],
        epochs: int = 4,
        batch_size: int = 4,
        lr: float = 1e-4,
        seed: int = 0,
    ) -> dict[str, Any]:
        self._check_budget()
        if not examples or epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Teacher fit requires examples and positive finite hyperparameters")
        # Validate all examples before mutating teacher weights.
        prepared = []
        for generation, feedback, target in examples:
            self._check_budget()
            self._validate_target(generation, target)
            tokens, positions, _ = self._scoring_input(generation, feedback, None, "bidirectional")
            prepared.append((tokens, positions, target.detach().cpu()))
        self._reset_adapter("teacher", seed)
        self.model.zero_grad(set_to_none=True)
        parameters = list(self._adapter_parameters("teacher").values())
        optimizer = self._optimizer(parameters, lr)
        rng = random.Random(seed)
        metrics: dict[str, Any] = {
            "optimizer_steps": 0, "example_exposures": 0, "epochs": epochs,
            "seed": seed, "batch_size": batch_size, "lr": lr,
            "losses": [], "grad_norms": [],
        }
        start = time.perf_counter()
        with self._using_adapter("teacher", trainable=True):
            for _epoch in range(epochs):
                order = list(range(len(prepared)))
                rng.shuffle(order)
                for offset in range(0, len(order), batch_size):
                    self._check_budget()
                    group = order[offset:offset + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    group_losses = []
                    for index in group:
                        self._check_budget()
                        tokens, positions, target = prepared[index]
                        loss = forward_kl(self._logprobs(tokens, positions, True), target)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite teacher KL")
                        (loss / len(group)).backward()
                        group_losses.append(float(loss.detach()))
                        metrics["example_exposures"] += 1
                        self.counters["teacher_example_exposures"] += 1
                        self._check_budget()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    metrics["optimizer_steps"] += 1
                    self.counters["teacher_optimizer_steps"] += 1
                    metrics["losses"].append(sum(group_losses) / len(group_losses))
                    metrics["grad_norms"].append(float(grad_norm))
                    self._check_budget()
        metrics["elapsed_seconds"] = time.perf_counter() - start
        self.teacher_fit_metrics = metrics
        return metrics

    def save_teacher(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(
            self.model, adapter_name="teacher", save_embedding_layers=False
        )
        torch.save({
            "format_version": 1, "model_name": self.model_name, "revision": self.revision,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in state.items()},
            "fit_metrics": self.teacher_fit_metrics,
        }, destination)

    def load_teacher(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["model_name"] != self.model_name or payload["revision"] != self.revision:
            raise ValueError("Teacher checkpoint does not match the frozen base identity")
        if payload["format_version"] != 1:
            raise ValueError("Unknown teacher checkpoint format")
        if not isinstance(payload["fit_metrics"], dict):
            raise ValueError("Invalid teacher fit metadata")
        expected = get_peft_model_state_dict(
            self.model, adapter_name="teacher", save_embedding_layers=False
        )
        supplied = payload["state_dict"]
        if not isinstance(supplied, dict) or supplied.keys() != expected.keys():
            raise ValueError("Teacher checkpoint tensor names do not match the full adapter")
        for name, reference in expected.items():
            tensor = supplied[name]
            if not isinstance(tensor, Tensor) or tensor.shape != reference.shape:
                raise ValueError(f"Teacher checkpoint tensor shape mismatch: {name}")
            if tensor.dtype != reference.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Teacher checkpoint tensor dtype/finite-value mismatch: {name}")
        result = set_peft_model_state_dict(self.model, supplied, adapter_name="teacher")
        if result.unexpected_keys:
            raise ValueError(f"Unexpected loaded teacher keys: {result.unexpected_keys}")
        self.teacher_fit_metrics = payload["fit_metrics"]
