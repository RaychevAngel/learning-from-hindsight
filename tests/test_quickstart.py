"""Self-contained helper tests, not scientific result fixtures."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight.experiment import read_json, write_json
from hindsight.shared import SharedRecipe
from hindsight.teacher_controls import TeacherRecipe

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


CONFIGS = module(ROOT / "examples/quickstart/prepare_configs.py")
CHECK = module(ROOT / "scripts/check_fresh_clone.py")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    path = tmp_path / "data"
    write_json(path / "manifest.json", {"test_fixture": True})
    episodes = [
        SimpleNamespace(
            episode_id=f"{partition}-{family}-{index}", partition=partition, family=family
        )
        for partition, count in (("meta_train", 8), ("development", 4), ("confirmation", 8))
        for family in ("a", "b", "c", "d")
        for index in range(count)
    ]
    monkeypatch.setattr(CONFIGS, "load_cohort", lambda _: (episodes, []))
    return path


def test_without_qualification_only_instrumentation_config_is_emitted(corpus, tmp_path):
    output = tmp_path / "config"
    result = CONFIGS.prepare(corpus, output, "2099-01-01T00:00:00+00:00", 60)
    assert result["stage"] == "qualification_config_only"
    assert [path.name for path in output.iterdir()] == ["throughput.json"]
    assert read_json(output / "throughput.json")["student_steps"] == 1
    with pytest.raises(FileExistsError):
        CONFIGS.prepare(corpus, output, "2099-01-01T00:00:00+00:00", 60)


def test_qualified_configs_match_runner_schemas_and_small_recipe(corpus, tmp_path):
    qualification = tmp_path / "qualification.json"
    write_json(
        qualification,
        {
            "status": "completed",
            "selected_batch_size": 8,
            "repeat_stability": {"4": True, "8": True},
            "dataset_sha256": CONFIGS.sha(corpus / "manifest.json"),
            "config": {"model_name": CONFIGS.MODEL, "revision": CONFIGS.REVISION},
        },
    )
    output = tmp_path / "config"
    result = CONFIGS.prepare(corpus, output, "2099-01-01T00:00:00+00:00", 60, qualification)
    assert result["train_sources"] == 4 and result["shared_eval_sources"] == 8
    proposer = read_json(output / "proposer.json")
    teacher = read_json(output / "teacher-controls.json")
    shared = read_json(output / "shared.json")
    assert set(proposer["train_episode_ids"]).isdisjoint(proposer["evaluation_episode_ids"])
    assert teacher["training_episode_ids"] == proposer["train_episode_ids"]
    assert len(teacher["evaluation_episode_ids"]) == 16 and len(shared["seeds"]) == 2
    assert TeacherRecipe(**teacher["recipe"]).student_steps == 1
    assert SharedRecipe(**shared["recipe"]).teacher_steps == 2
    assert read_json(output / "pi-comparison.json")["expected_sources"] == 16


@pytest.mark.parametrize(
    "deadline,seconds",
    [("2020-01-01T00:00:00+00:00", 60), ("2099-01-01", 60), ("2099-01-01T00:00:00+00:00", 7201)],
)
def test_config_preparation_refuses_expired_unzoned_or_unbounded_runs(
    corpus, tmp_path, deadline, seconds
):
    with pytest.raises(ValueError, match="deadline"):
        CONFIGS.prepare(corpus, tmp_path / "bad", deadline, seconds)


def test_fresh_clone_suite_is_explicit_and_excludes_known_artifact_dependencies():
    command = CHECK.command()
    assert "test_order_and_pi.py" not in CHECK.TESTS
    assert "test_source_fidelity.py" not in CHECK.TESTS
    assert "test_qwen35.py" in CHECK.TESTS and "test_proposer_training.py" in CHECK.TESTS
    assert command[-2:] == ["-k", CHECK.EXPRESSION]


def test_fresh_clone_replaces_inherited_source_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/editable/src")
    environment = CHECK.test_environment()
    assert environment["PYTHONPATH"] == str(ROOT / "src")
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_fresh_clone_child_rejects_out_of_checkout_import(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", CHECK.RUN_TESTS, str(tmp_path)],
        cwd=ROOT,
        env=CHECK.test_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing out-of-checkout hindsight import" in result.stderr
