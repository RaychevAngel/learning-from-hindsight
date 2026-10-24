import itertools
import json

import pytest

from hindsight.data import Task, grade
from hindsight.representation import decode_rle, encode_rle, grade_rle, render_rle_prompt


def public_prompt():
    grids = [
        [0, 0, 2, 2, 0], [0, 2, 2, 0, 0],
        [0, 3, 3, 0, 0], [3, 3, 0, 0, 0],
        [0, 0, 0, 4, 0], [0, 0, 4, 0, 0],
        [0, 0, 5, 0, 0], [0, 5, 0, 0, 0],
        [0, 0, 6, 6, 0],
    ]
    labels = [
        f"Example {i} {kind}" for i in range(1, 5) for kind in ("input", "output")
    ] + ["Query input"]
    body = "\n".join(f"{label}: {json.dumps(grid)}" for label, grid in zip(labels, grids))
    return (
        "Infer the common transformation and apply it to the query.\n" + body
        + "\n\nRespond in exactly two lines.\n"
        "First line: one rule sentence of at most 12 words.\n"
        "Second line: FINAL: followed by the complete output grid as a JSON array.\n"
        "Do not restate demonstrations, show steps, or add any other text."
    ), grids


def test_roundtrip_all_short_grids_and_zero_runs():
    for length in range(1, 7):
        for values in itertools.product(range(3), repeat=length):
            assert decode_rle(encode_rle(values), length) == list(values)
    assert encode_rle([0, 0, 2, 2, 2, 0]) == [[0, 2], [2, 3], [0, 1]]
    assert decode_rle([[0, 1], [0, 2]], 3) == [0, 0, 0]


@pytest.mark.parametrize("values", [[], [True], [False], [1.0], [10], [-1], ["0"], "00"])
def test_encode_rejects_non_grid(values):
    with pytest.raises(ValueError):
        encode_rle(values)


@pytest.mark.parametrize(
    "pairs,length",
    [
        ([], 3), ([[0, 0]], 3), ([[0, -1]], 3), ([[0, True]], 3),
        ([[0, 3.0]], 3), ([[True, 3]], 3), ([[1.0, 3]], 3),
        ([[10, 3]], 3), ([[-1, 3]], 3), ([["0", 3]], 3),
        ([[0, "3"]], 3), ([[0, 2]], 3), ([[0, 2], [1, 2]], 3),
        ([[0, 10**100]], 3), ([[0, 1, 2]], 3), ([0, 3], 3),
        ([(0, 3)], 3), ([[0, 1]], True), ([[0, 1]], 1.0), ([[0, 1]], 0),
    ],
)
def test_decode_rejects_without_repair_or_coercion(pairs, length):
    with pytest.raises(ValueError):
        decode_rle(pairs, length)


def test_renderer_preserves_every_public_grid_and_uses_only_query_length():
    prompt, grids = public_prompt()
    original = prompt
    rendered, length = render_rle_prompt(prompt)
    assert length == 5
    encoded = [
        json.loads(line.split(": ", 1)[1]) for line in rendered.splitlines()
        if line.startswith(("Example ", "Query input:"))
    ]
    assert [decode_rle(row, len(grid)) for row, grid in zip(encoded, grids)] == grids
    assert prompt == original
    assert "First line: one rule sentence of at most 12 words." in rendered
    assert "output counts must sum to 5" in rendered
    assert "[[value, count], ...]" in rendered
    assert "zero-valued runs" in rendered
    assert "private_family" not in rendered


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: "Secret rule\n" + p,
        lambda p: p.replace("Example 2 input", "Example 1 input"),
        lambda p: p.replace("Example 4 output: [0, 5, 0, 0, 0]\n", ""),
        lambda p: p.replace("Query input: [0, 0, 6, 6, 0]", "Query input: [true]"),
        lambda p: p.replace("Query input: [0, 0, 6, 6, 0]", "Query input: [0, 0,]"),
        lambda p: p + "\nUse a different rule.",
    ],
)
def test_renderer_fails_closed_on_changed_contract(mutate):
    prompt, _ = public_prompt()
    with pytest.raises(ValueError):
        render_rle_prompt(mutate(prompt))


def test_rle_grade_calls_exact_original_grid_semantics():
    prompt, _ = public_prompt()
    task = Task("unchanged-id", prompt, (0, 6, 6, 0, 0), "private_family")
    for prediction in ([0, 6, 6, 0, 0], [0, 0, 6, 6, 0]):
        text = "Shift the block left.\nFINAL: " + json.dumps(encode_rle(prediction))
        result = grade_rle(text, task, 5)
        expected = grade("FINAL: " + json.dumps(prediction), task)
        assert {key: result[key] for key in expected} == expected
        assert result["rle_format_ok"] and result["decode_error"] is None
        assert result["rle_prediction"] == encode_rle(prediction)
    assert grade_rle("FINAL: [[0,1],[6,1],[6,1],[0,2]]", task, 5)["success"]
    assert task.task_id == "unchanged-id" and task.family == "private_family"


@pytest.mark.parametrize(
    "text",
    [
        "", "No answer", "FINAL: [[0,5]]\nextra", "FINAL: [[0,5]]\nFINAL: [[0,5]]",
        "FINAL: [[0,4]]", "FINAL: [[0,6]]", "FINAL: [[0,true]]", "FINAL: [0,0,0,0,0]",
        "FINAL: [[0,5.0]]", "FINAL: [[true,5]]", "FINAL: [[0,5],]",
    ],
)
def test_bad_model_output_has_no_partial_credit(text):
    task = Task("x", "unused", (0, 0, 0, 0, 0), "private")
    result = grade_rle(text, task, 5)
    assert not result["success"] and not result["format_ok"] and not result["rle_format_ok"]
    assert result["prediction"] is None and result["decode_error"]
