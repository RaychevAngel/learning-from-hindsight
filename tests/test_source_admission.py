"""Text/native-action consistency at high-level reusable-source admission."""

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_model import TinyTokenizer

from hindsight.model import Generation

SPEC = importlib.util.spec_from_file_location(
    "source_admission", Path(__file__).parents[1] / "scripts/source_admission.py"
)
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)


@pytest.fixture
def source():
    tokenizer = TinyTokenizer()
    engine = SimpleNamespace(tokenizer=tokenizer)
    text = "FINAL: [1]"
    generation = Generation("task", [100], tokenizer.encode(text), text, True)
    return engine, generation


@pytest.mark.parametrize("ended", [False, True])
def test_exact_decode_admits_uncapped_or_eos_terminated_native_actions(source, ended):
    engine, generation = source
    if ended:
        generation = replace(
            generation,
            response_ids=generation.response_ids + [engine.tokenizer.eos_token_id],
            cap_hit=False,
        )
    assert ADMISSION.validate_source_text(engine, generation) is None


def test_text_is_not_normalized_into_a_different_attempt(source):
    engine, generation = source
    for text in ("FINAL: [0]", generation.text + " ", ""):
        with pytest.raises(ValueError, match="tokenizer-decoded native response IDs"):
            ADMISSION.validate_source_text(engine, replace(generation, text=text))


@pytest.mark.parametrize("tokens", [[], [False], [-1], [1.5], ["100"]])
def test_invalid_native_action_ids_fail_before_decode(source, tokens):
    engine, generation = source

    def forbidden_decode(*args, **kwargs):
        pytest.fail("Invalid native action IDs reached decoding")

    engine.tokenizer.decode = forbidden_decode
    with pytest.raises(ValueError, match="nonempty native response IDs"):
        ADMISSION.validate_source_text(engine, replace(generation, response_ids=tokens))
