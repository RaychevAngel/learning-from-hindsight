import copy
import importlib.util
import json
from pathlib import Path

import pytest

from hindsight.data import load_corpus

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prompt_revision", ROOT / "scripts/revise_prompt.py")
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)


def test_prompt_revision_preserves_all_semantics_and_parent_bytes(tmp_path):
    source = ROOT / "data/corpus-v1"
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    output = tmp_path / "v2"
    manifest = revision.revise(source, output)
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == before
    assert (source / "episodes.jsonl").read_bytes() == (output / "episodes.jsonl").read_bytes()
    old_public = revision.parse_rows(before["tasks.public.jsonl"])
    new_public = revision.parse_rows((output / "tasks.public.jsonl").read_bytes())
    for old, new in zip(old_public, new_public, strict=True):
        assert old["prompt"] != new["prompt"]
        assert new["prompt"].endswith(revision.RESPONSE_CONTRACT)
        assert {k: v for k, v in old.items() if k != "prompt"} == {
            k: v for k, v in new.items() if k != "prompt"
        }
        assert old["prompt"][len(revision.OLD_HEADER) + 1 :] in new["prompt"]
    old_private = revision.parse_rows(before["tasks.private.jsonl"])
    new_private = revision.parse_rows((output / "tasks.private.jsonl").read_bytes())
    for old, new in zip(old_private, new_private, strict=True):
        assert {k: v for k, v in old.items() if k != "public_sha256"} == {
            k: v for k, v in new.items() if k != "public_sha256"
        }
    old_episodes, old_calibration = load_corpus(source)
    new_episodes, new_calibration = load_corpus(output)
    assert len(new_episodes) == 112 and len(new_calibration) == 16
    assert [e.episode_id for e in old_episodes] == [e.episode_id for e in new_episodes]
    assert [t.task_id for t in old_calibration] == [t.task_id for t in new_calibration]
    assert manifest["prompt_revision"]["paired_tasks"] == 1472
    assert manifest["prompt_revision"]["regenerated_tasks"] == 0
    pairing = revision.parse_rows((output / "prompt-pairing.jsonl").read_bytes())
    for old, new, pair in zip(old_public, new_public, pairing, strict=True):
        assert old["task_id"] == new["task_id"] == pair["task_id"]
        assert pair["parent_public_sha256"] == revision.sha(revision.encoded(old))
        assert pair["revised_public_sha256"] == revision.sha(revision.encoded(new))


def test_revision_is_reproducible_and_refuses_overwrite(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    revision.revise(ROOT / "data/corpus-v1", a)
    revision.revise(ROOT / "data/corpus-v1", b)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }
    with pytest.raises(FileExistsError):
        revision.revise(ROOT / "data/corpus-v1", a)


def test_revision_fails_closed_on_unexpected_contract_or_body():
    source = ROOT / "data/corpus-v1/tasks.public.jsonl"
    row = json.loads(source.open().readline())
    changed = copy.deepcopy(row)
    changed["prompt"] = "Different instruction.\n" + row["prompt"]
    with pytest.raises(ValueError, match="Unexpected v1 prompt"):
        revision.revise_prompt(changed)
    changed = copy.deepcopy(row)
    changed["query_input"][0] = (changed["query_input"][0] + 1) % 10
    with pytest.raises(ValueError, match="body disagrees"):
        revision.revise_prompt(changed)
