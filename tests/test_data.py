import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from hindsight.data import Task, grade, load_corpus

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("corpus_builder", ROOT / "scripts/build_corpus.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_grade_accepts_exact_final_field():
    task = Task("test", "prompt", (0, 2, 2, 0), "family")
    assert grade("Brief explanation.\nFINAL: [0, 2, 2, 0]\n", task) == {
        "success": True,
        "format_ok": True,
        "prediction": [0, 2, 2, 0],
    }
    assert grade("FINAL: [0, 2, 0, 0]", task) == {
        "success": False,
        "format_ok": True,
        "prediction": [0, 2, 0, 0],
    }


@pytest.mark.parametrize(
    "response",
    [
        "[0, 2, 2, 0]",
        "FINAL: [0, 2, 2, 0]\nextra",
        "FINAL: [0, 2, 2]",
        "FINAL: [false, 2, 2, 0]",
        "FINAL: [0.0, 2, 2, 0]",
        "FINAL: [0, 12, 2, 0]",
        "FINAL: [0, 2, 2, 0]\nFINAL: [0, 2, 2, 0]",
        "FINAL: [[0,2,2,0]]",
        "FINAL: [0, 2, 2, 0] trailing",
        "FINAL: [0 2 2 0]",
        "",
        None,
    ],
)
def test_grade_rejects_ambiguous_or_invalid_format(response):
    assert grade(response, Task("t", "p", (0, 2, 2, 0), "f")) == {
        "success": False,
        "format_ok": False,
        "prediction": None,
    }


def test_independent_oracles_on_hand_checked_examples():
    examples = [
        ("shift_2", [0, 3, 4, 0, 0, 0], [0, 0, 0, 3, 4, 0]),
        ("block_and_noise_remove", [3, 0, 3, 3, 0], [0, 0, 3, 3, 0]),
        ("recolor_blocks_by_size", [3, 3, 0, 3, 3, 3], [2, 2, 0, 1, 1, 1]),
        ("gravity_right", [0, 3, 0, 4], [3, 4, 0, 0]),
        ("gravity_left", [0, 3, 0, 4], [0, 0, 3, 4]),
        ("two_points_and_fill", [0, 3, 0, 3, 0], [0, 3, 3, 3, 0]),
        ("two_points_and_fill_inv", [0, 3, 3, 3, 0], [0, 3, 0, 3, 0]),
        ("copy_block_to_dots_colors", [2, 2, 2, 0, 0, 7, 0, 0], [2, 2, 2, 0, 7, 7, 7, 0]),
        ("repeat_pattern_full", [2, 3, 2, 3, 0, 0, 0], [2, 3, 2, 3, 2, 3, 2]),
        ("block_touch_dot", [3, 3, 0, 0, 1, 0], [0, 0, 3, 3, 1, 0]),
        ("reflect_block_around_dot", [3, 0, 2, 0, 0], [0, 0, 2, 0, 3]),
        ("sort_blocks_by_size", [0, 3, 3, 0, 3, 0], [3, 0, 3, 3, 0, 0]),
        ("fill_until_collision", [5, 0, 2, 0, 3, 0], [5, 2, 2, 3, 3, 0]),
        ("fill_until_collision", [0, 2, 0, 3, 0, 5], [0, 2, 2, 3, 3, 5]),
        ("color_left_half_blocks", [2, 2, 2, 0, 2, 2], [8, 2, 2, 0, 8, 2]),
    ]
    for alias, row, expected in examples:
        assert builder.independent_answer(alias, row) == expected, alias


def test_pair_signature_catches_mirrored_palette_variants():
    pair = {"input": [0, 2, 2, 0, 0], "output": [0, 0, 2, 2, 0]}
    variant = {"input": [0, 0, 7, 7, 0], "output": [0, 7, 7, 0, 0]}
    assert builder.pair_signature(pair) == builder.pair_signature(variant)


def test_shipped_corpus_integrity_and_public_boundary():
    corpus = ROOT / "data/corpus-v1"
    if not corpus.exists():
        pytest.skip("Full corpus has not been generated yet")
    episodes, calibration = load_corpus(corpus)
    assert len(episodes) == 112 and len(calibration) == 16
    assert Counter(e.partition for e in episodes) == {
        "meta_train": 64,
        "development": 16,
        "final_known": 16,
        "final_unseen": 16,
    }
    assert all(len(e.validation) == 4 and len(e.queries) == 8 for e in episodes)
    assert (
        len(
            {t.task_id for e in episodes for t in (e.source, *e.validation, *e.queries)}
            | {t.task_id for t in calibration}
        )
        == 1472
    )
    public = [json.loads(line) for line in (corpus / "tasks.public.jsonl").read_text().splitlines()]
    assert all(set(row) == {"task_id", "prompt", "demonstrations", "query_input"} for row in public)
    private = [
        json.loads(line) for line in (corpus / "tasks.private.jsonl").read_text().splitlines()
    ]
    assert len({p for row in private for p in row["canonical_pair_sha256"]}) == 7360
    exclusions = json.loads(
        (ROOT / "third_party/reasoning_gym/preview_exclusions.json").read_text()
    )
    all_pairs = {p for row in private for p in row["canonical_pair_sha256"]}
    assert not all_pairs.intersection(exclusions["canonical_pair_sha256"])
    for row, secret in zip(public, private, strict=True):
        pairs = [*row["demonstrations"], {"input": row["query_input"], "output": secret["answer"]}]
        assert builder.witness_check(pairs) is not None
        for pair in pairs:
            actual = builder.independent_answer(secret["generator_alias"], pair["input"])
            assert actual == pair["output"]


def test_builder_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        builder.build(tmp_path)


def test_reproduces_shipped_data_and_detects_tampering(tmp_path):
    original = ROOT / "data/corpus-v1"
    if not original.exists():
        pytest.skip("Full corpus has not been generated yet")
    reproduced = tmp_path / "reproduced"
    manifest = builder.build(reproduced)
    for name in manifest["files"]:
        assert (original / name).read_bytes() == (reproduced / name).read_bytes()
    load_corpus(reproduced)
    tampered = tmp_path / "tampered"
    shutil.copytree(reproduced, tampered)
    with (tampered / "tasks.public.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="integrity failure"):
        load_corpus(tampered)
