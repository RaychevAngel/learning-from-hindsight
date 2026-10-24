"""Immutable partition assignment over already validated, disjoint corpus chunks."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from hindsight.data import load_corpus


def load_cohort(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("protocol_version") != "hindsight-cohort-v1":
        return load_corpus(path)
    result, episode_ids, task_ids, inputs = [], set(), set(), set()
    for chunk in manifest["chunks"]:
        directory = path / chunk["path"]
        if directory.resolve().parent != path.resolve():
            raise ValueError("Chunk must be a direct child of this collection")
        if (
            hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
            != chunk["sha256"]
        ):
            raise ValueError("Chunk manifest changed")
        episodes, calibration = load_corpus(directory)
        if calibration or chunk["partition"] not in {"meta_train", "development", "confirmation"}:
            raise ValueError("Invalid cohort partition or calibration")
        for ep in episodes:
            ids = [task.task_id for task in (ep.source, *ep.validation, *ep.queries)]
            if ep.episode_id in episode_ids or task_ids.intersection(ids):
                raise ValueError("Episode/task reused between cohorts")
            episode_ids.add(ep.episode_id)
            task_ids.update(ids)
            result.append(replace(ep, partition=chunk["partition"]))
        for line in (directory / "tasks.private.jsonl").read_text().splitlines():
            signatures = json.loads(line)["canonical_input_sha256"]
            if len(set(signatures)) != 5 or inputs.intersection(signatures):
                raise ValueError("Canonical input reused between cohorts")
            inputs.update(signatures)
    if (
        dict(Counter(ep.partition for ep in result)) != manifest["partition_episodes"]
        or len(result) != manifest["episodes"]
        or len(inputs) != manifest["unique_inputs"]
    ):
        raise ValueError("Collection membership changed")
    return result, []
