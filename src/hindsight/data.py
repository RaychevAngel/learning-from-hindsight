"""Immutable task records, integrity-checked corpus loading, and exact grid grading.

Task contains private research metadata. Only ``Task.prompt`` is model-visible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    answer: tuple[int, ...]
    family: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    partition: str
    family: str
    source: Task
    validation: tuple[Task, ...]
    queries: tuple[Task, ...]


def grade(text: str, task: Task) -> dict:
    """Accept exactly one FINAL JSON-array field on the final nonblank line.

    Explanations can precede the field. No extraction from prose, substring
    matching, float coercion, partial credit, or silent length repair is used.
    """
    invalid = {"success": False, "format_ok": False, "prediction": None}
    if not isinstance(text, str):
        return invalid
    lines = text.strip().splitlines()
    if not lines or len(re.findall(r"\bFINAL\s*:", text)) != 1:
        return invalid
    match = re.fullmatch(r"FINAL:\s*(\[[^\r\n]*\])", lines[-1].strip())
    if match is None:
        return invalid
    try:
        prediction = json.loads(match.group(1))
    except (ValueError, TypeError):
        return invalid
    if not isinstance(prediction, list) or len(prediction) != len(task.answer):
        return invalid
    if any(type(value) is not int or not 0 <= value <= 9 for value in prediction):
        return invalid
    return {
        "success": tuple(prediction) == task.answer,
        "format_ok": True,
        "prediction": prediction,
    }


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_corpus(path: Path) -> tuple[list[Episode], list[Task]]:
    """Load all frozen episodes and the separate development calibration tasks.

    Fail closed on file tampering, duplicate IDs, missing/private extra records,
    role reuse, family disagreement, or partition/count drift.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    balanced = manifest.get("protocol_version") == "balanced-development-v1"
    role_counts = [1, 8, 8] if balanced else [1, 4, 8]
    balanced_families = {"translation", "denoising", "size_recoloring", "gravity"}
    if balanced and (
        manifest.get("role_counts") != {"source": 1, "validation": 8, "queries": 8}
        or manifest.get("partition_episodes") != {"development": 8}
        or manifest.get("tasks") != 136 or manifest.get("calibration_tasks") != 0
        or len(manifest.get("declared_families", [])) != 4
        or set(manifest.get("declared_families", [])) != balanced_families
    ):
        raise ValueError("Invalid bounded balanced-development manifest")
    files = ["tasks.public.jsonl", "tasks.private.jsonl", "episodes.jsonl"]
    if balanced:
        files.append("assignment.jsonl")
    for name in files:
        actual = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if actual != manifest["files"][name]:
            raise ValueError(f"Corpus integrity failure: {name}")
    public_rows = _rows(path / "tasks.public.jsonl")
    private_rows = _rows(path / "tasks.private.jsonl")
    public = {record["task_id"]: record for record in public_rows}
    private = {record["task_id"]: record for record in private_rows}
    if (
        len(public) != len(public_rows)
        or len(private) != len(private_rows)
        or public.keys() != private.keys()
    ):
        raise ValueError("Duplicate or unmatched public/private task IDs")
    tasks = {}
    for task_id, record in public.items():
        if set(record) != {"task_id", "prompt", "demonstrations", "query_input"}:
            raise ValueError(f"Unexpected public fields: {task_id}")
        secret = private[task_id]
        if _hash(record) != secret["public_sha256"]:
            raise ValueError(f"Public task hash mismatch: {task_id}")
        answer = secret["answer"]
        if (
            _hash(answer) != secret["oracle_sha256"]
            or not isinstance(answer, list)
            or len(answer) != len(record["query_input"])
            or any(type(x) is not int or not 0 <= x <= 9 for x in answer)
        ):
            raise ValueError(f"Invalid oracle: {task_id}")
        tasks[task_id] = Task(task_id, record["prompt"], tuple(answer), secret["family"])
    result, consumed, episode_ids = [], set(), set()
    allowed = {"development"} if balanced else {
        "meta_train", "development", "final_known", "final_unseen"
    }
    for record in _rows(path / "episodes.jsonl"):
        episode_id = record["episode_id"]
        if episode_id in episode_ids or record["partition"] not in allowed:
            raise ValueError("Duplicate episode ID or invalid partition")
        episode_ids.add(episode_id)
        roles = {
            "source": [record["source"]],
            "validation": record["validation"],
            "queries": record["queries"],
        }
        if [len(roles[k]) for k in roles] != role_counts:
            raise ValueError(f"Episode role counts must be {role_counts}")
        for role, ids in roles.items():
            for task_id in ids:
                secret = private.get(task_id, {})
                if (
                    task_id in consumed
                    or secret.get("episode_id") != episode_id
                    or secret.get("family") != record["family"]
                    or secret.get("partition") != record["partition"]
                    or secret.get("role") != role
                ):
                    raise ValueError("Task reuse or episode metadata mismatch")
                consumed.add(task_id)
        result.append(
            Episode(
                episode_id,
                record["partition"],
                record["family"],
                tasks[record["source"]],
                tuple(tasks[key] for key in record["validation"]),
                tuple(tasks[key] for key in record["queries"]),
            )
        )
    calibration = []
    for task_id, secret in private.items():
        if secret["role"] == "calibration":
            if task_id in consumed or secret["partition"] != "calibration":
                raise ValueError("Invalid calibration membership")
            calibration.append(tasks[task_id])
            consumed.add(task_id)
    counts = dict(Counter(episode.partition for episode in result))
    if (
        consumed != tasks.keys()
        or len(tasks) != manifest["tasks"]
        or len(calibration) != manifest["calibration_tasks"]
        or counts != manifest["partition_episodes"]
    ):
        raise ValueError("Corpus membership/count mismatch")
    known = {e.family for e in result if e.partition != "final_unseen"}
    unseen = {e.family for e in result if e.partition == "final_unseen"}
    if balanced:
        if Counter(e.family for e in result) != dict.fromkeys(balanced_families, 2):
            raise ValueError("Balanced development requires two episodes per declared family")
    elif known & unseen or len(known) != 8 or len(unseen) != 4:
        raise ValueError("Family split violation")
    return result, calibration
