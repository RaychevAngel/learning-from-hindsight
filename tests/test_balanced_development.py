import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "balanced_development", ROOT / "scripts/build_balanced_development.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("balanced") / "corpus"
    manifest = builder.build(path, ROOT / "data/corpus-v2")
    return path, manifest


def rows(path, name):
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


def test_exact_balanced_roles_and_aliases(corpus):
    path, manifest = corpus
    episodes = rows(path, "episodes.jsonl")
    private = rows(path, "tasks.private.jsonl")
    assert manifest["partition_episodes"] == {"development": 8}
    assert len(episodes) == 8 and len(private) == 136
    for episode in episodes:
        assert len(episode["validation"]) == len(episode["queries"]) == 8
        selected = [row for row in private if row["episode_id"] == episode["episode_id"]]
        assert {row["generator_alias"] for row in selected} == {builder.ALIASES[episode["family"]]}
        by_role = {role: Counter(tuple(row["stratum"]) for row in selected if row["role"] == role)
                   for role in ("validation", "queries")}
        assert by_role["validation"] == by_role["queries"] == dict.fromkeys(builder.BINS, 2)
    assert manifest["model_calls"] == 0


def test_no_old_or_internal_canonical_input_overlap(corpus):
    path, manifest = corpus
    old, _ = builder.old_public_exclusions(ROOT / "data/corpus-v2")
    inputs = [value for row in rows(path, "tasks.private.jsonl")
              for value in row["canonical_input_sha256"]]
    assert len(inputs) == len(set(inputs)) == manifest["new_unique_inputs"] == 680
    assert not old.intersection(inputs)
    pairs = [value for row in rows(path, "tasks.private.jsonl")
             for value in row["canonical_pair_sha256"]]
    assert len(pairs) == len(set(pairs)) == 680


def test_valid_oracle_witness_and_public_boundary(corpus):
    path, _ = corpus
    private = {row["task_id"]: row for row in rows(path, "tasks.private.jsonl")}
    for public in rows(path, "tasks.public.jsonl"):
        assert set(public) == {"task_id", "prompt", "demonstrations", "query_input"}
        secret = private[public["task_id"]]
        pairs = public["demonstrations"] + [{"input": public["query_input"],
                                             "output": secret["answer"]}]
        assert len(pairs) == 5
        assert builder.base.witness_check(pairs)
        assert all(builder.base.independent_answer(secret["generator_alias"], pair["input"])
                   == pair["output"] for pair in pairs)
        assert secret["public_sha256"] == builder.base.digest(public)
        assert "Respond in exactly two lines." in public["prompt"]


def test_assignment_is_reproducible_and_pool_roles_are_balanced(corpus):
    path, _ = corpus
    private = {row["task_id"]: row for row in rows(path, "tasks.private.jsonl")}
    for assignment in rows(path, "assignment.jsonl"):
        shuffled = assignment["shuffled_ids"]
        original = sorted(shuffled, key=lambda key: private[key]["pool_serial"])
        builder.Random(assignment["seed"]).shuffle(original)
        assert original == shuffled
        assert [private[key]["role"] for key in shuffled] == ["validation"] * 2 + ["queries"] * 2


def test_refuses_overwrite_and_old_public_tampering(corpus, tmp_path):
    path, _ = corpus
    with pytest.raises(FileExistsError):
        builder.build(path, ROOT / "data/corpus-v2")
    fake = tmp_path / "old"
    fake.mkdir()
    (fake / "manifest.json").write_text(json.dumps({"files": {"tasks.public.jsonl": "bad"}}))
    (fake / "tasks.public.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        builder.old_public_exclusions(fake)


def test_additional_corpus_exclusion_reads_only_public_inputs(corpus, tmp_path, monkeypatch):
    old = ROOT / "data/corpus-v2"
    additional, _ = corpus
    expected_old, parent_hash = builder.old_public_exclusions(old)
    expected_additional, additional_hash = builder.old_public_exclusions(additional)
    read_bytes = Path.read_bytes
    read_text = Path.read_text

    def public_only_read(original):
        def guarded(path, *args, **kwargs):
            if path.parent in (old, additional):
                assert path.name in {"manifest.json", "tasks.public.jsonl"}
            return original(path, *args, **kwargs)
        return guarded

    monkeypatch.setattr(Path, "read_bytes", public_only_read(read_bytes))
    monkeypatch.setattr(Path, "read_text", public_only_read(read_text))
    fresh = tmp_path / "fresh"
    manifest = builder.build(fresh, old, seed=2026090602, additional_corpora=(additional,))
    actual, _ = builder.old_public_exclusions(fresh)
    assert len(actual) == manifest["new_unique_inputs"] == 680
    assert not actual.intersection(expected_old | expected_additional)
    assert manifest["parent_manifest_sha256"] == parent_hash
    assert manifest["excluded_canonical_inputs"] == len(expected_old | expected_additional)
    assert manifest["additional_excluded_corpora"] == [{
        "manifest_sha256": additional_hash,
        "canonical_inputs": len(expected_additional),
        "newly_excluded_canonical_inputs": len(expected_additional - expected_old),
    }]


def test_additional_corpus_public_tampering_prevents_build(corpus, tmp_path):
    fake = tmp_path / "extra"
    fake.mkdir()
    original = b"{}\n"
    (fake / "manifest.json").write_text(json.dumps({
        "files": {"tasks.public.jsonl": hashlib.sha256(original).hexdigest()}}))
    (fake / "tasks.public.jsonl").write_bytes(original + b"\n")
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="integrity"):
        builder.build(output, corpus[0], additional_corpora=(fake,))
    assert not output.exists()


def test_repeatable_exclude_corpus_cli(monkeypatch, tmp_path):
    observed = {}

    def build(*args, **kwargs):
        observed.update(args=args, kwargs=kwargs)
        return {}

    monkeypatch.setattr(builder, "build", build)
    monkeypatch.setattr("sys.argv", ["builder", "--output", str(tmp_path / "fresh"),
                                    "--exclude-corpus", "one", "--exclude-corpus", "two"])
    builder.main()
    assert observed["kwargs"]["additional_corpora"] == [Path("one"), Path("two")]


@pytest.mark.parametrize("family,row,expected", [
    ("translation", [1] * 7 + [0] * 9, ("short", "small")),
    ("translation", [1] * 8 + [0] * 12, ("long", "large")),
    ("denoising", [1] * 8 + [0, 1] + [0] * 6, ("short", "large")),
    ("size_recoloring", [3] * 4 + [0] + [3] * 2 + [0] * 9, ("short", "small")),
    ("size_recoloring", [3] * 2 + [0] + [3] * 4 + [0] * 13, ("long", "large")),
    ("gravity", [1] * 10 + [0] * 6, ("short", "small")),
    ("gravity", [1] * 11 + [0] * 9, ("long", "large")),
])
def test_predeclared_strata(family, row, expected):
    assert builder.stratum(family, row) == expected
