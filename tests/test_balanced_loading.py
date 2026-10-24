"""Loader opt-in and independent feature checks on the builder's temporary corpus."""

import hashlib
import json
import shutil
from collections import Counter
from itertools import groupby

import pytest
import test_balanced_development as shared

from hindsight.data import load_corpus

builder, corpus, rows = shared.builder, shared.corpus, shared.rows


def copied(corpus, tmp_path):
    return shutil.copytree(corpus[0], tmp_path / "copy")


def save_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_actual_builder_fixture_loads_only_eight_balanced_development_episodes(corpus):
    episodes, calibration = load_corpus(corpus[0])
    assert not calibration and len(episodes) == 8
    assert Counter(ep.family for ep in episodes) == dict.fromkeys(builder.ALIASES, 2)
    assert all(ep.partition == "development" and len(ep.validation) == len(ep.queries) == 8
               for ep in episodes)


@pytest.mark.parametrize("protocol", [None, "balanced-development-unknown"])
def test_balanced_roles_require_exact_opt_in(corpus, tmp_path, protocol):
    path = copied(corpus, tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.pop("protocol_version")
    if protocol is not None:
        manifest["protocol_version"] = protocol
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="role counts"):
        load_corpus(path)


@pytest.mark.parametrize("change", [
    {"role_counts": {"source": 1, "validation": 4, "queries": 8}},
    {"tasks": 135}, {"calibration_tasks": 1}, {"partition_episodes": {"final_known": 8}},
    {"declared_families": list(builder.ALIASES) + ["translation"]},
])
def test_malformed_bounded_manifest_rejected(corpus, tmp_path, change):
    path = copied(corpus, tmp_path)
    save_manifest(path, {**corpus[1], **change})
    with pytest.raises(ValueError, match="balanced-development manifest"):
        load_corpus(path)


@pytest.mark.parametrize("defect", ["final_partition", "family_allocation", "role_counts"])
def test_rehashed_semantic_changes_still_rejected(corpus, tmp_path, defect):
    path = copied(corpus, tmp_path)
    episodes, private = rows(path, "episodes.jsonl"), rows(path, "tasks.private.jsonl")
    episode = episodes[0]
    if defect == "role_counts":
        task_id = episode["validation"].pop()
        episode["queries"].append(task_id)
        next(row for row in private if row["task_id"] == task_id)["role"] = "queries"
    else:
        field = "partition" if defect == "final_partition" else "family"
        episode[field] = "final_known" if defect == "final_partition" else "denoising"
        for row in private:
            if row["episode_id"] == episode["episode_id"]:
                row[field] = episode[field]
    manifest = json.loads((path / "manifest.json").read_text())
    for name, records in (("episodes.jsonl", episodes), ("tasks.private.jsonl", private)):
        raw = b"".join(builder.base.encoded(row) + b"\n" for row in records)
        (path / name).write_bytes(raw)
        manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    save_manifest(path, manifest)
    with pytest.raises(ValueError):
        load_corpus(path)


def test_assignment_file_hash_is_enforced(corpus, tmp_path):
    path = copied(corpus, tmp_path)
    (path / "assignment.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="assignment.jsonl"):
        load_corpus(path)


def test_public_query_features_independently_reproduce_balanced_strata(corpus):
    private = {row["task_id"]: row for row in rows(corpus[0], "tasks.private.jsonl")}
    counts = Counter()
    for public in rows(corpus[0], "tasks.public.jsonl"):
        secret, values = private[public["task_id"]], public["query_input"]
        blocks = [len(list(group)) for nonzero, group in groupby(values, key=bool) if nonzero]
        family = secret["family"]
        small = (blocks[0] > blocks[1] if family == "size_recoloring" else
                 max(blocks) <= 7 if family == "denoising" else
                 sum(value != 0 for value in values) <= (7 if family == "translation" else 10))
        stratum = ("short" if len(values) <= 19 else "long", "small" if small else "large")
        assert list(stratum) == secret["stratum"]
        if secret["role"] != "source":
            counts[secret["episode_id"], secret["role"], stratum] += 1
    assert len(counts) == 64 and set(counts.values()) == {2}
