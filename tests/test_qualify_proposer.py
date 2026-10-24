"""Source-derived diagnostic representations; no model construction or GPU calls."""

import ast
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_proposer.py"
SPEC = importlib.util.spec_from_file_location("qualify_proposer", SCRIPT)
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)

PUBLIC_PROMPT = (
    "Infer the common transformation.\n"
    "Example 1 input: [0, 0, 4, 4, 0]\n"
    "Example 1 output: [4, 4, 0, 0, 0]\n"
    "Query input: [0, 5, 5, 0, 0]\n"
    "Respond with a rule sentence and FINAL."
)


def restore(runs):
    restored = []
    for value, start, length in runs:
        assert start == len(restored)
        assert length > 0
        restored.extend([value] * length)
    return restored


def test_grid_runs_roundtrip_including_zeros_and_boundaries():
    rng = random.Random(7301)
    grids = [[0], [0] * 32, [9], [0, 0, 4, 4, 0], [1, 0, 1, 0]]
    grids += [[rng.randrange(10) for _ in range(rng.randrange(1, 50))] for _ in range(100)]
    for grid in grids:
        runs = QUALIFY.grid_runs(grid)
        assert restore(runs) == grid
        assert all(left[0] != right[0] for left, right in zip(runs, runs[1:]))
    assert QUALIFY.grid_runs([0, 0, 4, 4, 0]) == [(0, 0, 2), (4, 2, 2), (0, 4, 1)]


@pytest.mark.parametrize("grid", [[], [True], [1.0], [-1], [10], "[0]", [[0]]])
def test_grid_runs_rejects_non_grids(grid):
    with pytest.raises(ValueError, match="integer grid"):
        QUALIFY.grid_runs(grid)


def test_view_roundtrips_only_given_labels_and_never_creates_query_output():
    view = QUALIFY.indexed_grid_view(PUBLIC_PROMPT)
    reconstructed = {}
    for line in view.splitlines():
        if " runs: " in line:
            label, runs = line.split(" runs: ", 1)
            reconstructed[label] = restore(ast.literal_eval(runs))
    original = {}
    for line in PUBLIC_PROMPT.splitlines():
        if ": [" in line:
            label, grid = line.split(": ", 1)
            original[label] = json.loads(grid)
    assert reconstructed == original
    assert set(reconstructed) == {"Example 1 input", "Example 1 output", "Query input"}
    assert "Query output:" not in view
    assert QUALIFY.indexed_grid_view(PUBLIC_PROMPT + "\nFINAL: [9, 9, 9]") == view


@pytest.mark.parametrize("prompt", [
    PUBLIC_PROMPT + "\nQuery output: [9, 9, 9]",
    PUBLIC_PROMPT + "\nQuery input: [0]",
    "No labeled grids here.",
])
def test_view_rejects_query_answers_duplicate_labels_and_missing_grids(prompt):
    with pytest.raises(ValueError):
        QUALIFY.indexed_grid_view(prompt)


def test_auxiliary_view_preserves_experience_and_keeps_active_instruction_last():
    prompt = QUALIFY.proposal_prompt(PUBLIC_PROMPT, "actual attempted response", "incorrect")
    enriched = QUALIFY.add_indexed_grids(prompt, PUBLIC_PROMPT)
    block = QUALIFY.indexed_grid_view(PUBLIC_PROMPT) + "\n"
    assert enriched.replace(block, "", 1) == prompt
    assert enriched.index("</source_derived_indexed_grids>") < enriched.index(
        "</recorded_experience>"
    ) < enriched.index("Your task now:")
    assert enriched.endswith("Give the lesson only.")
    with pytest.raises(ValueError, match="boundary"):
        QUALIFY.add_indexed_grids("missing boundary", PUBLIC_PROMPT)


@pytest.mark.parametrize("indexed,deliberate", [(False, False), (True, False), (True, True)])
def test_diagnostic_flag_is_opt_in_and_uses_saved_public_prompt_only(
    monkeypatch, tmp_path, indexed, deliberate,
):
    sources = tmp_path / "saved"
    (sources / "sources").mkdir(parents=True)
    config = {"candidate_count": 3, "pi_tokens": 80, "seed": 7,
              "model_name": "unused-model", "revision": "unused-revision",
              "max_context_tokens": 1536, "max_teacher_context_tokens": 4096,
              "absolute_deadline_utc": "2099-01-01T00:00:00+00:00"}
    (sources / "identity.json").write_text(json.dumps({"config": config}))
    episodes = []
    for index in range(4):
        task = SimpleNamespace(task_id=f"task-{index}", prompt=PUBLIC_PROMPT,
                               answer="PRIVATE_ANSWER_NEVER_READ")
        episode = SimpleNamespace(episode_id=f"ep-{index}", partition="development",
                                  family=f"PRIVATE_FAMILY_{index}", source=task,
                                  validation="PRIVATE_V_NEVER_READ", queries="PRIVATE_Q_NEVER_READ")
        episodes.append(episode)
        (sources / "sources" / f"{episode.episode_id}.json").write_text(json.dumps({
            "task_id": task.task_id, "generation": {
                "prompt": PUBLIC_PROMPT, "text": "actual attempted response",
            }, "feedback": "incorrect", "grade": "PRIVATE_GRADE_NEVER_READ",
        }))
    monkeypatch.setattr(QUALIFY, "load_corpus", lambda path: (episodes, []))
    captured = []

    @dataclass
    class Output:
        text: str = "LESSON: Preserve block lengths." if deliberate else "Preserve block lengths."
        cap_hit: bool = False

    class Engine:
        def __init__(self, **kwargs):
            self.counters = {}

        def generate(self, prompts, **kwargs):
            captured.extend(prompts)
            assert kwargs["max_new_tokens"] == 123
            return [Output()]

    monkeypatch.setattr(QUALIFY, "HFEngine", Engine)
    output = tmp_path / "receipt.json"
    argv = ["qualify_proposer.py", "--sources", str(sources), "--data", "unused",
            "--output", str(output), "--token-limit", "123",
            "--model-name", "override-model", "--revision", "pinned-override"]
    if indexed:
        argv.append("--indexed-grids")
    if deliberate:
        argv.append("--deliberate")
    monkeypatch.setattr("sys.argv", argv)
    assert QUALIFY.main() == 0
    assert len(captured) == 12
    for prompt in captured:
        assert PUBLIC_PROMPT in prompt
        assert "actual attempted response" in prompt
        assert "PRIVATE_" not in prompt
        assert ("<source_derived_indexed_grids>" in prompt) is indexed
        if not indexed:
            assert prompt == QUALIFY.proposal_prompt(
                PUBLIC_PROMPT, "actual attempted response", "incorrect"
            )
        if deliberate:
            assert prompt.index("</recorded_experience>") < prompt.index("First use at most")
            assert prompt.endswith("Do not solve the query or output a FINAL field.")
    receipt = json.loads(output.read_text())
    assert receipt["indexed_grids"] is indexed
    assert receipt["model_name"] == "override-model"
    assert receipt["revision"] == "pinned-override"
