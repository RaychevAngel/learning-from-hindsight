import hashlib
import json
import shutil
from pathlib import Path

import pytest

from hindsight.cohorts import load_cohort

ROOT = Path(__file__).resolve().parents[1]


def collection(tmp_path):
    child = tmp_path / "chunk-00"
    shutil.copytree(ROOT / "data/balanced-transfer-v1", child)
    manifest = {
        "protocol_version": "hindsight-cohort-v1",
        "episodes": 8,
        "partition_episodes": {"confirmation": 8},
        "unique_inputs": 680,
        "chunks": [
            {
                "path": child.name,
                "partition": "confirmation",
                "sha256": hashlib.sha256((child / "manifest.json").read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_collection_assigns_partition_without_changing_task(tmp_path):
    collection(tmp_path)
    actual, calibration = load_cohort(tmp_path)
    original, _ = load_cohort(ROOT / "data/balanced-transfer-v1")
    assert not calibration and len(actual) == 8
    assert all(ep.partition == "confirmation" for ep in actual)
    assert [ep.source for ep in actual] == [ep.source for ep in original]


def test_changed_child_is_rejected(tmp_path):
    collection(tmp_path)
    (tmp_path / "chunk-00/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        load_cohort(tmp_path)


def test_duplicate_chunk_is_rejected(tmp_path):
    manifest = collection(tmp_path)
    manifest["chunks"] *= 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reused"):
        load_cohort(tmp_path)


def test_cohort_counts_are_authoritative(tmp_path):
    manifest = collection(tmp_path)
    manifest["partition_episodes"] = {"development": 8}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership changed"):
        load_cohort(tmp_path)
