"""Lossless run-length interface for the development-only representation check.

The renderer receives only public prompt text. Decoding never repairs an answer:
the existing exact grid grader still owns success, after bounded expansion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from hindsight.data import Task, grade

REPRESENTATION_VERSION = "rle-calibration-v1"
_INTRO = "Infer the common transformation and apply it to the query."
_OLD_RESPONSE = "\n".join(
    [
        "Respond in exactly two lines.",
        "First line: one rule sentence of at most 12 words.",
        "Second line: FINAL: followed by the complete output grid as a JSON array.",
        "Do not restate demonstrations, show steps, or add any other text.",
    ]
)


def encode_rle(values: Sequence[int]) -> list[list[int]]:
    """Encode every cell, including zeros, into maximal [value, count] runs."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Grid must be a nonempty list or tuple")
    if any(type(value) is not int or not 0 <= value <= 9 for value in values):
        raise ValueError("Grid values must be integers from 0 through 9")
    runs: list[list[int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def decode_rle(pairs: object, expected_length: int) -> list[int]:
    """Accept semantic RLE, including adjacent equal runs, without length repair."""
    if type(expected_length) is not int or expected_length <= 0:
        raise ValueError("Expected length must be a positive integer")
    if not isinstance(pairs, list) or not pairs or len(pairs) > expected_length:
        raise ValueError("Runs must be a nonempty bounded JSON array")
    total = 0
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each run must be [value, count]")
        value, count = pair
        if type(value) is not int or not 0 <= value <= 9:
            raise ValueError("Run value must be an integer from 0 through 9")
        if type(count) is not int or not 1 <= count <= expected_length:
            raise ValueError("Run count must be a positive bounded integer")
        total += count
        if total > expected_length:
            raise ValueError("Decoded length exceeds the query length")
    if total != expected_length:
        raise ValueError("Decoded length differs from the query length")
    # Validate the entire representation before allocating its expanded cells.
    return [value for value, count in pairs for _ in range(count)]


def render_rle_prompt(task_prompt: str) -> tuple[str, int]:
    """Render only the known public v2 grid contract; reject unrecognized text."""
    prefix, suffix = _INTRO + "\n", "\n\n" + _OLD_RESPONSE
    if not task_prompt.startswith(prefix) or not task_prompt.endswith(suffix):
        raise ValueError("Expected the unchanged corpus-v2 prompt contract")
    lines = task_prompt[len(prefix) : -len(suffix)].splitlines()
    labels = [
        f"Example {index} {kind}"
        for index in range(1, 5)
        for kind in ("input", "output")
    ] + ["Query input"]
    if len(lines) != len(labels):
        raise ValueError("Expected exactly four demonstration pairs and one query")
    rendered, query_length = [], 0
    for line, label in zip(lines, labels, strict=True):
        if not line.startswith(label + ": "):
            raise ValueError("Unexpected demonstration/query label or ordering")
        try:
            values = json.loads(line[len(label) + 2 :])
        except ValueError as error:
            raise ValueError("Malformed public grid JSON") from error
        runs = encode_rle(values)
        rendered.append(label + ": " + json.dumps(runs))
        if label == "Query input":
            query_length = len(values)
    prompt = "\n".join(
        [
            _INTRO,
            "Every grid uses run-length encoding: [[value, count], ...].",
            "Each pair means count consecutive cells of value; include zero-valued runs.",
            "Values are integers 0 through 9; counts are positive integers.",
            *rendered,
            "",
            "Respond in exactly two lines.",
            "First line: one rule sentence of at most 12 words.",
            "Second line: FINAL: followed by the complete output as [[value, count], ...].",
            f"The output counts must sum to {query_length}, the query's cell count.",
            "Do not restate demonstrations, show steps, or add any other text.",
        ]
    )
    return prompt, query_length


def grade_rle(text: str, task: Task, expected_length: int) -> dict:
    """Decode exactly one terminal FINAL field, then call the original grader."""
    invalid = {
        "success": False,
        "format_ok": False,
        "prediction": None,
        "rle_format_ok": False,
        "rle_prediction": None,
    }
    try:
        if not isinstance(text, str) or len(re.findall(r"\bFINAL\s*:", text)) != 1:
            raise ValueError("Expected exactly one FINAL field")
        lines = text.strip().splitlines()
        match = re.fullmatch(r"FINAL:\s*(\[[^\r\n]*\])", lines[-1].strip())
        if match is None:
            raise ValueError("FINAL must be the final nonblank line")
        pairs = json.loads(match.group(1))
        prediction = decode_rle(pairs, expected_length)
    except (ValueError, TypeError) as error:
        return {**invalid, "decode_error": str(error)}
    result = grade("FINAL: " + json.dumps(prediction), task)
    return {
        **result,
        "rle_format_ok": True,
        "rle_prediction": pairs,
        "decode_error": None,
    }
