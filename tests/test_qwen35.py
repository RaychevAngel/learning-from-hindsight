"""Tiny random Qwen3.5: no checkpoint/tokenizer downloads or GPU execution."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import transformers

from hindsight.model import QWEN35_LORA_TARGETS, Generation, HFEngine

if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("Qwen3.5 tests require the separately pinned environment", allow_module_level=True)


class ThinkingTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text):
        return [byte + 3 for byte in text.encode()]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(token - 3 for token in tokens if 3 <= token < 259).decode(errors="replace")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            enable_thinking=None, return_dict=False):
        assert tokenize
        assert return_dict is False
        result = []
        for message in messages:
            if message["role"] == "user":
                result += self.encode("<u>" + message["content"] + "</u>")
            elif message["role"] == "assistant":
                # Historical assistant turns omit the generation-only thinking prefill.
                result += self.encode("<a>" + message["content"]) + [1] + self.encode("\n")
            else:
                raise AssertionError(message["role"])
        if add_generation_prompt:
            result += self.encode("<a><think>\n" if enable_thinking is not False
                                  else "<a><think>\n\n</think>\n\n")
        return result


@pytest.fixture
def engine():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=272, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=1024,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        bos_token_id=2, eos_token_id=1, pad_token_id=0, attention_dropout=0.0,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                         "partial_rotary_factor": 1.0},
    )
    return HFEngine(config=config, tokenizer=ThinkingTokenizer(), device="cpu",
                    enable_thinking=False)


def rollout(engine, ended=True):
    response = engine.tokenizer.encode("abc") + ([1] if ended else [])
    return Generation("task", engine.encode_prompt("task"), response, "not the real token text",
                      not ended)


def snapshot(engine, adapter=None):
    return {name: value.detach().clone() for name, value in engine.model.named_parameters()
            if (adapter is None and "lora_" not in name)
            or (adapter is not None and f".{adapter}." in name)}


def unchanged(engine, before):
    parameters = dict(engine.model.named_parameters())
    return all(torch.equal(value, parameters[name]) for name, value in before.items())


def test_text_only_loading_and_explicit_lora_projection_coverage(engine):
    assert engine.model_type == "qwen3_5_text"
    assert engine.lora_targets == QWEN35_LORA_TARGETS
    counts = {target: len(names) for target, names in engine.lora_module_coverage.items()}
    assert counts == {"q_proj": 1, "k_proj": 1, "v_proj": 1, "o_proj": 1,
                      "gate_proj": 4, "up_proj": 4, "down_proj": 4,
                      "in_proj_qkv": 3, "in_proj_z": 3, "in_proj_b": 3,
                      "in_proj_a": 3, "out_proj": 3}
    assert not any("visual" in name for name, _ in engine.model.named_parameters())
    assert set(engine.model.peft_config) == {"student", "teacher"}


def test_multimodal_checkpoint_keys_load_into_text_only_model(engine, tmp_path):
    # Exercise production dispatch/key conversion using only a tiny local file.
    bare = transformers.Qwen3_5ForCausalLM(engine.model.config)
    expected = bare.state_dict()
    checkpoint = {
        (name.replace("model.", "model.language_model.", 1)
         if name.startswith("model.") else name): value
        for name, value in expected.items()
    }
    checkpoint["model.visual.unused.weight"] = torch.zeros(1)
    transformers.Qwen3_5Config(text_config=bare.config.to_dict()).save_pretrained(tmp_path)
    torch.save(checkpoint, tmp_path / "pytorch_model.bin")
    loaded = HFEngine(str(tmp_path), revision="local-test", tokenizer=ThinkingTokenizer(),
                      device="cpu", enable_thinking=False)
    actual = {
        name.replace(".base_layer.", "."): value
        for name, value in loaded.model.get_base_model().state_dict().items()
        if "lora_" not in name
    }
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())


def test_same_token_future_visibility_and_no_causal_mode_leak(engine):
    original = rollout(engine)
    edited = replace(original, response_ids=[105] + original.response_ids[1:])
    causal = engine.score_rollout(original, "incorrect")
    assert torch.equal(causal[0], engine.score_rollout(edited, "incorrect")[0])
    bidir = engine.score_rollout(original, "incorrect", mode="bidirectional")
    assert not torch.allclose(
        bidir[0], engine.score_rollout(edited, "incorrect", mode="bidirectional")[0],
        atol=1e-7, rtol=0,
    )
    future = replace(original, response_ids=original.response_ids[:2] + [109, 1])
    assert torch.equal(causal[:2], engine.score_rollout(future, "incorrect")[:2])
    assert not torch.allclose(
        bidir[0], engine.score_rollout(future, "incorrect", mode="bidirectional")[0],
        atol=1e-7, rtol=0,
    )
    assert torch.equal(causal, engine.score_rollout(original, "incorrect"))
    assert not bidir.requires_grad and bidir.device.type == "cpu"
    assert torch.allclose(bidir.logsumexp(-1), torch.zeros(len(original.response_ids)), atol=1e-6)


@pytest.mark.parametrize("ended", [False, True])
def test_thinking_prefill_continuation_preserves_original_ids(engine, monkeypatch, ended):
    original = rollout(engine, ended=ended)
    monkeypatch.setattr(
        engine.tokenizer, "decode",
        lambda *args, **kwargs: pytest.fail("Must not decode original response IDs"),
    )
    for mode in ("bidirectional", "raw_causal"):
        tokens, positions, _ = engine._scoring_input(original, "incorrect", None, mode)
        p = len(original.prompt_ids)
        assert tokens[:p] == original.prompt_ids
        assert tokens[p:p + len(original.response_ids)] == original.response_ids
        assert tokens[positions[0] + 1:positions[-1] + 2] == original.response_ids
        assert tokens.count(engine.eos_token_id) == (1 if mode == "bidirectional"
                                                   else 1 + int(ended))
        if mode == "bidirectional":
            assert positions[0] == p - 1
        # A shortened prefix is not accepted just because it matches history.
        broken = replace(original, prompt_ids=original.prompt_ids[:5])
        with pytest.raises(ValueError, match="prompt IDs"):
            engine._scoring_input(broken, "incorrect", None, mode)


def test_student_backward_reset_and_base_teacher_isolation(engine):
    original = rollout(engine)
    target = engine.score_rollout(original, "incorrect", pi="Check the transformation.")
    base, teacher = snapshot(engine), snapshot(engine, "teacher")
    engine.reset_student(7)
    initial = snapshot(engine, "student")
    first = engine.adapt(original, target, steps=2, lr=1e-3)
    trained = snapshot(engine, "student")
    assert len(first) == 2 and all(torch.isfinite(torch.tensor(row["loss"])) for row in first)
    assert not unchanged(engine, initial)
    assert unchanged(engine, base) and unchanged(engine, teacher)
    assert engine._student_optimizer.state
    engine.reset_student(7)
    assert unchanged(engine, initial) and not engine._student_optimizer.state
    second = engine.adapt(original, target, steps=2, lr=1e-3)
    assert unchanged(engine, trained)
    assert [row["loss"] for row in first] == [row["loss"] for row in second]
    assert all(value.grad is None for name, value in engine.model.named_parameters()
               if "lora_" not in name or ".teacher." in name)


def test_bidirectional_teacher_backward_isolated_and_reusable(engine):
    original = rollout(engine)
    target = engine.score_rollout(original, "incorrect", pi="Check the transformation.")
    base, student = snapshot(engine), snapshot(engine, "student")
    engine._reset_adapter("teacher", 7)
    teacher = snapshot(engine, "teacher")
    metrics = engine.fit_teacher([(original, "incorrect", target)], epochs=1, batch_size=1,
                                 lr=1e-3, seed=7)
    assert metrics["optimizer_steps"] == 1 and metrics["example_exposures"] == 1
    assert unchanged(engine, base) and unchanged(engine, student)
    assert not unchanged(engine, teacher)
    result = engine.score_rollout(original, "incorrect", mode="bidirectional", adapter="teacher")
    assert torch.isfinite(result).all()


def test_tiny_causal_generation_works_with_hybrid_cache(engine):
    generated = engine.generate(["a", "bb"], max_new_tokens=2, sample=False, seed=3)
    assert len(generated) == 2 and all(1 <= len(value.response_ids) <= 2 for value in generated)


def qualification_helper():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_hindsight_teacher.py"
    spec = importlib.util.spec_from_file_location("qualify_hindsight_teacher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_qualification_exact_counts_and_checkpoint(engine, tmp_path, monkeypatch):
    helper = qualification_helper()
    original = rollout(engine)
    target = engine.score_rollout(original, "incorrect", pi="Check the transformation.")
    monkeypatch.setattr(engine, "generate", lambda *args, **kwargs: pytest.fail("No generation"))
    output = tmp_path / "qualification"
    receipt = helper.qualify_teacher(engine, original, "incorrect", target, output)
    assert receipt["status"] == "passed_teacher_qualification"
    assert receipt["actual_work"] == helper.EXPECTED_WORK
    assert all(value["passed"] for value in receipt["checks"].values())
    assert receipt["attention"]["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert "DeltaNet" in receipt["attention"]["description"]
    checkpoint = output / receipt["checkpoint"]["file"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt["checkpoint"]["sha256"]
    assert json.loads((output / "qualification.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        helper.qualify_teacher(engine, original, "incorrect", target, output)


def test_teacher_qualification_cannot_confuse_reset_with_learning(engine, tmp_path, monkeypatch):
    helper = qualification_helper()
    original = rollout(engine)
    target = engine.score_rollout(original, "incorrect", pi="Check the transformation.")

    def no_learning(examples, epochs, batch_size, lr, seed):
        engine._reset_adapter("teacher", seed)
        engine.counters["teacher_optimizer_steps"] += 2
        engine.counters["teacher_example_exposures"] += 2
        return {"optimizer_steps": 2, "example_exposures": 2,
                "losses": [1.0, 1.0], "grad_norms": [0.0, 0.0], "elapsed_seconds": 0.1}

    monkeypatch.setattr(engine, "fit_teacher", no_learning)
    output = tmp_path / "qualification"
    with pytest.raises(AssertionError, match="weights_changed"):
        helper.qualify_teacher(engine, original, "incorrect", target, output)
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["status"] == "failed_teacher_qualification"
    assert receipt["actual_work"]["scoring_forwards"] == 1


def test_teacher_qualification_rejects_ignored_feedback(engine, tmp_path, monkeypatch):
    helper = qualification_helper()
    original = rollout(engine)
    target = engine.score_rollout(original, "incorrect", pi="Check the transformation.")
    score = engine.score_rollout
    monkeypatch.setattr(
        engine, "score_rollout",
        lambda generation, feedback, **kwargs: score(generation, "incorrect", **kwargs),
    )
    output = tmp_path / "qualification"
    with pytest.raises(AssertionError, match="feedback_visible"):
        helper.qualify_teacher(engine, original, "incorrect", target, output)
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["actual_work"]["scoring_forwards"] == 3
    assert receipt["status"] == "failed_teacher_qualification"


def test_teacher_qualification_rejects_noop_checkpoint_reload(engine, tmp_path, monkeypatch):
    helper = qualification_helper()
    original = rollout(engine)
    target = engine.score_rollout(original, "incorrect", pi="Check the transformation.")
    monkeypatch.setattr(engine, "load_teacher", lambda path: None)
    output = tmp_path / "qualification"
    with pytest.raises(AssertionError, match="checkpoint_parameter_roundtrip"):
        helper.qualify_teacher(engine, original, "incorrect", target, output)
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["actual_work"]["scoring_forwards"] == 3
    assert receipt["status"] == "failed_teacher_qualification"
    assert receipt["checkpoint"]["sha256"]
