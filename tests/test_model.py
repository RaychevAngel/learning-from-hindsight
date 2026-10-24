"""Offline numerical tests: random tiny Qwen3, no Hub access or checkpoint downloads."""

from dataclasses import replace

import pytest
import torch
from transformers import Qwen3Config

from hindsight.model import Generation, HFEngine, forward_kl, with_pi


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text):
        return [byte + 3 for byte in text.encode("utf-8")]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(token - 3 for token in tokens if 3 <= token < 259).decode(
            "utf-8", errors="replace"
        )

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        assert tokenize
        assert return_dict is False
        result = []
        for message in messages:
            if message["role"] == "user":
                result += self.encode("<u>" + message["content"] + "</u>")
            elif message["role"] == "assistant":
                result += self.encode("<a>" + message["content"]) + [1] + self.encode("\n")
            else:
                raise AssertionError(message["role"])
        if add_generation_prompt:
            result += self.encode("<a>")
        return result


@pytest.fixture
def engine():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = Qwen3Config(
        vocab_size=272, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=1024,
        bos_token_id=2, eos_token_id=1, pad_token_id=0,
        attention_dropout=0.0, tie_word_embeddings=True,
    )
    return HFEngine(config=config, tokenizer=TinyTokenizer(), device="cpu")


def rollout(engine, response="abc", ended=True):
    response_ids = engine.tokenizer.encode(response) + ([1] if ended else [])
    return Generation("task", engine.encode_prompt("task"), response_ids, response, not ended)


def snapshot(engine, adapter=None):
    return {
        name: parameter.detach().clone()
        for name, parameter in engine.model.named_parameters()
        if (adapter is None and "lora_" not in name)
        or (adapter is not None and f".{adapter}." in name)
    }


def unchanged(engine, before):
    current = dict(engine.model.named_parameters())
    return all(torch.equal(value, current[name]) for name, value in before.items())


@pytest.mark.parametrize("constructor_kwargs, expected_extra", [
    ({}, {}),
    ({"enable_thinking": None}, {}),
    ({"enable_thinking": False}, {"enable_thinking": False}),
    ({"enable_thinking": True}, {"enable_thinking": True}),
])
def test_thinking_template_flag_only_forwarded_when_explicit(
    engine, monkeypatch, constructor_kwargs, expected_extra
):
    tokenizer = TinyTokenizer()
    original_template = tokenizer.apply_chat_template
    calls = []

    def recording_template(messages, **kwargs):
        calls.append(kwargs.copy())
        kwargs.pop("enable_thinking", None)
        return original_template(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", recording_template)
    configured = HFEngine(
        config=engine.model.config, tokenizer=tokenizer, device="cpu", **constructor_kwargs
    )
    messages = [{"role": "user", "content": "task"}]
    for generation_prompt in (False, True):
        assert configured._chat_ids(messages, generation_prompt) == original_template(
            messages, tokenize=True, add_generation_prompt=generation_prompt
        )
        assert calls[-1] == {
            "tokenize": True, "add_generation_prompt": generation_prompt,
            "return_dict": False, **expected_extra
        }


def test_same_token_and_future_visibility_and_no_mode_leak(engine):
    original = rollout(engine)
    edited = replace(
        original, response_ids=[original.response_ids[0] + 1] + original.response_ids[1:]
    )
    causal = engine.score_rollout(original, "wrong")
    changed_causal = engine.score_rollout(edited, "wrong")
    assert torch.equal(causal[0], changed_causal[0])
    bidirectional = engine.score_rollout(original, "wrong", mode="bidirectional")
    changed_bidir = engine.score_rollout(edited, "wrong", mode="bidirectional")
    assert not torch.allclose(bidirectional[0], changed_bidir[0], atol=1e-7, rtol=0)
    assert torch.equal(causal, engine.score_rollout(original, "wrong"))
    assert bidirectional.device.type == "cpu" and not bidirectional.requires_grad
    assert bidirectional.shape == (len(original.response_ids), engine.model.config.vocab_size)
    future_edit = replace(
        original, response_ids=original.response_ids[:2] + [120] + original.response_ids[3:]
    )
    assert torch.equal(causal[:2], engine.score_rollout(future_edit, "wrong")[:2])
    assert not torch.allclose(
        bidirectional[0],
        engine.score_rollout(future_edit, "wrong", mode="bidirectional")[0],
        atol=1e-7, rtol=0,
    )


def test_original_ids_offsets_and_single_role_closure(engine):
    generation = rollout(engine)
    tokens, positions, bidir = engine._scoring_input(generation, "wrong", None, "bidirectional")
    prefix_size = len(generation.prompt_ids)
    assert tokens[:prefix_size] == generation.prompt_ids
    assert tokens[prefix_size:prefix_size + len(generation.response_ids)] == generation.response_ids
    assert positions == list(range(prefix_size - 1, prefix_size + len(generation.response_ids) - 1))
    assert bidir
    # Original EOS is the one assistant close, not duplicated by the suffix.
    assert tokens[prefix_size + len(generation.response_ids)] != engine.eos_token_id
    for mode in ("causal", "raw_causal"):
        ids, indexes, _ = engine._scoring_input(
            generation, "wrong", "advice" if mode == "causal" else None, mode
        )
        assert ids[indexes[0] + 1:indexes[-1] + 2] == generation.response_ids
    pi_ids, pi_positions, _ = engine._scoring_input(generation, "wrong", "advice", "causal")
    assert pi_ids[:pi_positions[0] + 1] == engine.encode_prompt(with_pi("task", "advice"))
    with pytest.raises(ValueError, match="must not receive PI"):
        engine.score_rollout(generation, "wrong", pi="forbidden", mode="bidirectional")
    # Text is presentation metadata, never the source of action alignment.
    inaccurate_text = replace(generation, text="not the actual response")
    for mode in ("causal", "raw_causal", "bidirectional"):
        assert engine._scoring_input(generation, "wrong", None, mode) == engine._scoring_input(
            inaccurate_text, "wrong", None, mode
        )
    capped = rollout(engine, ended=False)
    capped_ids, _, _ = engine._scoring_input(capped, "wrong", None, "bidirectional")
    assert capped_ids[len(capped.prompt_ids) + len(capped.response_ids)] == engine.eos_token_id


def test_detached_full_kl_and_exact_replay_targets():
    logits = torch.randn(3, 7, requires_grad=True)
    reference = logits.log_softmax(-1).detach().requires_grad_(True)
    loss = forward_kl(logits.log_softmax(-1), reference)
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert reference.grad is None
    replay = torch.full((3, 7), -torch.inf)
    replay[:, 2] = 0
    actual = forward_kl(logits.log_softmax(-1), replay)
    expected = -logits.log_softmax(-1)[:, 2].mean()
    assert torch.isfinite(actual) and torch.allclose(actual, expected)


def test_adaptation_isolates_base_and_teacher_and_resets_optimizer(engine):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "wrong", pi="Consider all examples")
    base_before = snapshot(engine)
    teacher_before = snapshot(engine, "teacher")
    engine.reset_student(7)
    initial = snapshot(engine, "student")
    first = engine.adapt(generation, target, steps=2, lr=1e-3)
    trained = snapshot(engine, "student")
    assert len(first) == 2 and sum(item["optimizer_steps"] for item in first) == 2
    assert not unchanged(engine, initial)
    assert unchanged(engine, base_before) and unchanged(engine, teacher_before)
    assert engine._student_optimizer.state
    with pytest.raises(RuntimeError, match="reset_student"):
        engine.adapt(generation, target, steps=1, lr=1e-3)
    engine.reset_student(7)
    assert unchanged(engine, initial) and not engine._student_optimizer.state
    second = engine.adapt(generation, target, steps=2, lr=1e-3)
    assert unchanged(engine, trained)
    assert [x["loss"] for x in first] == [x["loss"] for x in second]
    assert all(parameter.grad is None for name, parameter in engine.model.named_parameters()
               if "lora_" not in name or ".teacher." in name)


def test_teacher_fit_accumulation_isolation_and_checkpoint(engine, tmp_path):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "wrong", pi="Check the pattern")
    base_before = snapshot(engine)
    student_before = snapshot(engine, "student")
    teacher_before = snapshot(engine, "teacher")
    metrics = engine.fit_teacher(
        [(generation, "wrong", target)] * 3, epochs=2, batch_size=2, lr=1e-3, seed=9
    )
    assert metrics["optimizer_steps"] == 4 and metrics["example_exposures"] == 6
    assert unchanged(engine, base_before) and unchanged(engine, student_before)
    assert not unchanged(engine, teacher_before)
    expected = engine.score_rollout(generation, "wrong", mode="bidirectional", adapter="teacher")
    path = tmp_path / "teacher.pt"
    engine.save_teacher(path)
    engine._reset_adapter("teacher", 10)
    engine.load_teacher(path)
    restored = engine.score_rollout(generation, "wrong", mode="bidirectional", adapter="teacher")
    assert torch.equal(expected, restored)
    assert engine.teacher_fit_metrics["optimizer_steps"] == 4
    good_teacher = snapshot(engine, "teacher")
    payload = torch.load(path, weights_only=True)
    payload["state_dict"].pop(next(iter(payload["state_dict"])))
    broken_path = tmp_path / "partial-teacher.pt"
    torch.save(payload, broken_path)
    with pytest.raises(ValueError, match="tensor names"):
        engine.load_teacher(broken_path)
    assert unchanged(engine, good_teacher)


def test_batched_generation_preserves_unpadded_prefixes_and_cap_flags(engine, monkeypatch):
    def fake_generate(input_ids, attention_mask, **kwargs):
        assert kwargs["use_cache"] and "is_causal" not in kwargs
        assert attention_mask[0, 0] == 0
        suffix = torch.tensor([[100, 1, 0], [101, 102, 103]])
        return torch.cat((input_ids, suffix), dim=1)

    monkeypatch.setattr(engine.model, "generate", fake_generate)
    results = engine.generate(["a", "longer"], max_new_tokens=3, sample=False, seed=1)
    assert results[0].prompt_ids == engine.encode_prompt("a")
    assert results[0].response_ids == [100, 1] and not results[0].cap_hit
    assert results[1].response_ids == [101, 102, 103] and results[1].cap_hit
    with pytest.raises(ValueError, match="not truncated"):
        engine.generate(["too long"], max_new_tokens=30, sample=False, seed=0, context_limit=10)


def test_real_tiny_generation_has_no_hub_dependency(engine):
    generated = engine.generate(["a", "bb"], max_new_tokens=2, sample=False, seed=3)
    assert len(generated) == 2
    assert all(1 <= len(item.response_ids) <= 2 for item in generated)


def test_config_requires_tokenizer_without_loading_hub():
    with pytest.raises(ValueError, match="explicit tokenizer"):
        HFEngine(config=Qwen3Config(), device="cpu")


def test_invalid_targets_fail_before_training_and_one_hot_adaptation_is_finite(engine):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "wrong")
    before = snapshot(engine, "student")
    with pytest.raises(ValueError, match="not normalized"):
        engine.adapt(generation, target + 3, steps=1, lr=1e-3)
    assert unchanged(engine, before)
    replay = torch.full_like(target, -torch.inf)
    replay[torch.arange(len(generation.response_ids)), generation.response_ids] = 0
    metrics = engine.adapt(generation, replay, steps=1, lr=1e-3)
    assert metrics[0]["loss"] > 0 and metrics[0]["grad_norm"] > 0


def test_identical_causal_target_remains_exact_noop_after_four_adam_steps(engine):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "")
    initial = snapshot(engine, "student")
    metrics = engine.adapt(generation, target, steps=4, lr=1e-4)
    assert all(item["loss"] == 0 and item["grad_norm"] == 0 for item in metrics)
    assert unchanged(engine, initial)
    assert torch.equal(target, engine.score_rollout(generation, "", adapter="student"))


def test_budget_callback_aborts_between_steps_and_restores_mode(engine):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "wrong", pi="advice")
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise TimeoutError("deadline")

    engine.budget_check = deadline
    with pytest.raises(TimeoutError, match="deadline"):
        engine.adapt(generation, target, steps=4, lr=1e-3)
    assert engine.counters["student_optimizer_steps"] == 1
    assert not engine.model.training
    assert all(not parameter.requires_grad for name, parameter in engine.model.named_parameters()
               if "lora_" not in name or ".teacher." in name)


def test_budget_check_after_generation_keeps_actual_sequence_count(engine, monkeypatch):
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("deadline after generation")

    def fake_generate(input_ids, **kwargs):
        return torch.cat((input_ids, torch.tensor([[1]])), dim=1)

    engine.budget_check = deadline
    monkeypatch.setattr(engine.model, "generate", fake_generate)
    with pytest.raises(TimeoutError, match="deadline after generation"):
        engine.generate(["a"], max_new_tokens=1, sample=False, seed=0)
    assert engine.counters["generation_sequences"] == 1


def test_teacher_budget_abort_inside_accumulation_reports_no_optimizer_step(engine):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "wrong", pi="advice")
    student = snapshot(engine, "student")
    base = snapshot(engine)

    def deadline():
        if engine.counters["teacher_example_exposures"] == 1:
            raise TimeoutError("deadline during accumulation")

    engine.budget_check = deadline
    with pytest.raises(TimeoutError, match="deadline during accumulation"):
        engine.fit_teacher([(generation, "wrong", target)] * 4, epochs=4, batch_size=4)
    assert engine.counters["teacher_optimizer_steps"] == 0
    assert engine.counters["teacher_example_exposures"] == 1
    assert unchanged(engine, student) and unchanged(engine, base)
    assert not engine.model.training


def test_initial_distribution_callback_adds_no_forward_or_numerical_change(engine, monkeypatch):
    generation = rollout(engine)
    target = engine.score_rollout(generation, "wrong", pi="advice")
    engine.reset_student(9)
    expected_initial = engine.score_rollout(generation, "", adapter="student")
    first = engine.adapt(generation, target, steps=2, lr=1e-3)
    expected_parameters = snapshot(engine, "student")
    engine.reset_student(9)
    captured = []
    forward_calls = 0
    original = engine._logprobs

    def counted(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original(*args, **kwargs)

    def callback(value):
        assert value.device.type == "cpu" and not value.requires_grad
        captured.append(value.clone())
        value.zero_()  # The callback receives its own copy, not training storage.

    monkeypatch.setattr(engine, "_logprobs", counted)
    second = engine.adapt(
        generation, target, steps=2, lr=1e-3, initial_logprobs_callback=callback
    )
    assert forward_calls == 2 and len(captured) == 1
    assert torch.equal(captured[0], expected_initial)
    assert unchanged(engine, expected_parameters)
    assert [item["loss"] for item in first] == [item["loss"] for item in second]
