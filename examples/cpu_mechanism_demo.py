"""Exercise model primitives on a tiny random CPU model, without downloads.

An executable API example, not a task-learning result or a miniature estimate of
the historical comparison. No oracle accuracy, learned proposer, candidate
selection or real-model performance claim is implied.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

from hindsight.model import Generation, HFEngine, forward_kl  # noqa: E402


class ByteTokenizer:
    """Explicit tokenizer fixture, not a pretrained tokenizer."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text):
        return [byte + 3 for byte in text.encode("utf-8")]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(token - 3 for token in tokens if 3 <= token < 259).decode(
            "utf-8", errors="replace"
        )

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=False
    ):
        if not tokenize or return_dict:
            raise ValueError("This fixture returns one native token-ID list")
        result = []
        for message in messages:
            if message["role"] == "user":
                result += self.encode("<u>" + message["content"] + "</u>")
            elif message["role"] == "assistant":
                result += self.encode("<a>" + message["content"]) + [1] + self.encode("\n")
            else:
                raise ValueError("Unsupported fixture message role")
        return result + (self.encode("<a>") if add_generation_prompt else [])


def execute():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = Qwen3Config(
        vocab_size=272,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    engine = HFEngine(
        model_name="random-tiny-qwen3",
        revision="offline-example",
        config=config,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )
    # Fixed numerical test vector, not a sampled scientific source task.
    source = Generation(
        "source task",
        engine.encode_prompt("source task"),
        engine.tokenizer.encode("abc") + [engine.eos_token_id],
        "abc",
        False,
    )
    feedback = "The submitted attempt was incorrect."
    target = engine.score_rollout(source, feedback, pi="Check the complete pattern")
    if target.requires_grad or target.shape != (4, 272):
        raise AssertionError("Expected detached full-vocabulary native-position targets")
    engine.reset_student(7)
    initial = engine.score_rollout(source, feedback, adapter="student")
    updates = engine.adapt(source, target, steps=2, lr=1e-3)
    adapted = engine.score_rollout(source, feedback, adapter="student")
    # Fit a distinct future-aware teacher from raw experience, without PI input.
    fit = engine.fit_teacher(
        [(source, feedback, target)] * 3, epochs=2, batch_size=2, lr=1e-3, seed=9
    )
    hindsight = engine.score_rollout(source, feedback, mode="bidirectional", adapter="teacher")
    engine.reset_student(7)
    second_updates = engine.adapt(source, hindsight, steps=2, lr=1e-3)
    # Next-task generation contains no lesson or historical response context.
    next_prompt = "different task"
    generated = engine.generate(
        [next_prompt], max_new_tokens=2, sample=False, seed=3, adapter="student"
    )[0]
    if generated.prompt != next_prompt:
        raise AssertionError("Next-task context was changed")
    return {
        "purpose": "Offline numerical/API demonstration, not useful task learning",
        "model": "Two-layer random Qwen3 with an explicit byte-tokenizer fixture",
        "device": str(engine.device),
        "pretrained_weights_downloaded": False,
        "native_target_shape": list(target.shape),
        "target_detached": not target.requires_grad,
        "explicit_target_initial_KL": float(forward_kl(initial, target)),
        "explicit_target_post_update_KL": float(forward_kl(adapted, target)),
        "explicit_student_steps": len(updates),
        "teacher_optimizer_steps": fit["optimizer_steps"],
        "teacher_example_exposures": fit["example_exposures"],
        "hindsight_student_steps": len(second_updates),
        "next_task_prompt": generated.prompt,
        "next_task_native_response_ids": generated.response_ids,
        "work": engine.counters,
        "performance_claim": None,
    }


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2, sort_keys=True, allow_nan=False))
