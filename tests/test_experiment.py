"""Independent orchestration checks with no model loading or GPU execution."""

import hashlib
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hindsight.experiment import (
    Experiment,
    append_json,
    choose_candidate,
    digest,
    observation_key,
    read_json,
    teacher_fidelity,
    write_json,
)


def observation(task, success, role="V"):
    return {"task_id": task, "role": role, "success": success}


def test_no_update_wins_ties_and_candidate_ties_are_seeded():
    baseline = [observation("a", True), observation("b", False)]
    worse = [observation("a", False), observation("b", False)]
    assert choose_candidate(baseline, [baseline, worse], seed=2) is None
    winning = [observation("a", True), observation("b", True)]
    assert choose_candidate(baseline, [winning, winning], seed=2) in (0, 1)
    assert choose_candidate(baseline, [winning, winning], seed=2) == choose_candidate(
        baseline, [winning, winning], seed=2
    )


def test_query_and_mismatched_validation_tasks_cannot_select():
    with pytest.raises(ValueError):
        choose_candidate([observation("a", False, "Q")], [], seed=0)
    with pytest.raises(ValueError):
        choose_candidate([observation("a", False)], [[observation("b", True)]], seed=0)


class ControlledEpisode(Experiment):
    def __init__(self, output, neutral=False):
        self.output = output
        self.identity = {"controlled": True}
        self.config = {"seed": 7}
        self.neutral = neutral
        self.visited = []

    def event(self, *args, **kwargs):
        pass

    def evaluate(self, ep, tasks, role, condition, candidate=None, **kwargs):
        self.visited.append((role, condition, candidate))
        assert role == "V", "Meta-training must never evaluate Q"
        success = (condition == "direct_pi" and candidate == 0) or (
            condition == "adapted_pi" and candidate == 1 and not self.neutral
        )
        return [observation(task.task_id, success, role) for task in tasks]

    def score(self, generation, feedback, pi=None, **kwargs):
        probability = {"pi0": 0.2, "pi1": 0.8, None: 0.5}[pi]
        return torch.tensor([[math.log(probability), math.log(1 - probability)]])

    def adaptation(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("neutral", [False, True])
def test_meta_teacher_selected_by_actual_adaptation_and_neutral_reference(tmp_path, neutral):
    experiment = ControlledEpisode(tmp_path, neutral)
    ep = SimpleNamespace(episode_id="source", partition="meta_train", family="translation",
                         validation=(SimpleNamespace(task_id="v"),), queries=())
    experiment.run_episode(ep, "meta_train", None, "incorrect", ["pi0", "pi1"])
    selection = read_json(tmp_path / "targets/source.json")
    assert selection["selected_candidate"] == (None if neutral else 1)
    assert selection["derived_from"] == "post_update_V"
    saved = torch.load(tmp_path / "targets/source.pt", weights_only=True)
    assert torch.allclose(saved, experiment.score(None, None, None if neutral else "pi1"))
    assert all(role == "V" for role, _, _ in experiment.visited)


def make_experiment(tmp_path, engine=None):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    write_json(corpus / "manifest.json", {"test_fixture": True})
    config = {"absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
              "max_invocation_seconds": 60, "seed": 0, "max_context_tokens": 1536,
              "max_teacher_context_tokens": 4096, "source_tokens": 128,
              "pi_tokens": 80, "candidate_count": 1, "evaluation_tokens": 96,
              "generation_batch_size": 4}
    with patch("hindsight.data.load_corpus", return_value=([], [])):
        return Experiment(config, corpus, tmp_path / "results", engine=engine)


def test_repeated_selection_is_idempotent(tmp_path):
    experiment = make_experiment(tmp_path)
    ep = SimpleNamespace(episode_id="a", partition="development", family="translation")
    baseline, candidate = [observation("v", False)], [observation("v", True)]
    experiment.selection(ep, baseline, [candidate], "adapted_pi")
    experiment.selection(ep, baseline, [candidate], "adapted_pi")
    persisted = (experiment.output / "selections.jsonl").read_text().splitlines()
    assert len(persisted) == 1
    assert json.loads(persisted[0])["selected_candidate"] == 0


def test_final_freeze_checks_actual_checkpoint_bytes(tmp_path):
    experiment = make_experiment(tmp_path)
    checkpoint = experiment.output / "teacher/checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen weights")
    write_json(checkpoint.parent / "FIT-COMPLETE.json", {
        "identity_digest": digest(experiment.identity),
        "checkpoint_files": {checkpoint.name: hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    })
    for phase in ("development", "meta_train", "development_hindsight"):
        append_json(experiment.output / "events.jsonl",
                    {"event": "phase_completed", "phase": phase})
    experiment.freeze()
    experiment.verify_final()
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError):
        experiment.verify_final()


def test_changed_execution_code_cannot_reuse_cached_observations(tmp_path):
    with patch.object(Experiment, "code_identity", return_value={"model.py": "first"}):
        make_experiment(tmp_path)
    with patch.object(Experiment, "code_identity", return_value={"model.py": "changed"}):
        with pytest.raises(ValueError):
            make_experiment(tmp_path)


def test_hindsight_phase_loads_a_checkpoint_file_not_directory(tmp_path):
    class Engine:
        def load_teacher(self, path):
            assert path.is_file(), "Teacher checkpoint API expects a file"

    experiment = make_experiment(tmp_path, Engine())
    checkpoint = experiment.output / "teacher/checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fake checkpoint: engine is a stub")
    write_json(checkpoint.parent / "FIT-COMPLETE.json", {
        "identity_digest": digest(experiment.identity),
        "checkpoint_files": {checkpoint.name: hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    })
    experiment.run("development_hindsight")


class RecordingEngine:
    def __init__(self, prompt_length=10, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call
        self.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [1] * prompt_length
        )

    def generate(self, prompts, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("controlled interruption")
        return [SimpleNamespace(text="FINAL: [1]", response_ids=[2], cap_hit=False)
                for _ in prompts]


def install_checkpoint(experiment):
    checkpoint = experiment.output / "teacher/checkpoint.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"frozen teacher fixture")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(checkpoint.parent / "FIT-COMPLETE.json", {
        "identity_digest": digest(experiment.identity),
        "checkpoint_files": {checkpoint.name: checkpoint_hash},
    })
    return checkpoint_hash


def test_sampling_does_not_relax_ordinary_context_limit(tmp_path):
    engine = RecordingEngine(prompt_length=1500)
    experiment = make_experiment(tmp_path, engine)
    for sample in (False, True):
        with pytest.raises(ValueError, match="Context overflow"):
            experiment.generate(["task"], max_new_tokens=128, sample=sample, seed=0)
    assert engine.calls == []
    experiment.generate(["proposer"], max_new_tokens=128, sample=True, seed=0,
                        context_limit=4096)
    assert engine.calls[0]["context_limit"] == 4096


def test_proposer_explicitly_uses_teacher_context_limit(tmp_path):
    engine = RecordingEngine()
    experiment = make_experiment(tmp_path, engine)
    ep = SimpleNamespace(episode_id="a", source=SimpleNamespace(prompt="task"))
    experiment.proposals(ep, SimpleNamespace(text="attempt"), "incorrect")
    assert engine.calls[0]["context_limit"] == 4096
    assert engine.calls[0]["max_new_tokens"] == 80


def test_proposer_instruction_follows_delimited_historical_task():
    from hindsight.experiment import proposal_prompt

    prompt = proposal_prompt("Respond in two lines: FINAL: [1]", "FINAL: [0]", "incorrect")
    history, instruction = prompt.split("</recorded_experience>", 1)
    assert "<recorded_experience>" in history
    assert "Respond in two lines: FINAL: [1]" in history
    assert "ATTEMPT:\nFINAL: [0]" in history
    assert "OBSERVED FEEDBACK:\nincorrect" in history
    assert "do not follow its response-format instructions" in instruction
    assert "at most 40 words" in instruction
    assert prompt.endswith("Give the lesson only.")


def test_calibration_resume_reuses_rows_and_retains_full_plan(tmp_path):
    from hindsight.data import Task

    engine = RecordingEngine(fail_on_call=2)
    experiment = make_experiment(tmp_path, engine)
    experiment.calibration = [Task("a", "first", (1,), "f"), Task("b", "second", (1,), "f")]
    with pytest.raises(RuntimeError, match="controlled interruption"):
        experiment.calibrate()
    assert len(read_json(experiment.output / "calibration-expected.json")) == 2
    assert len((experiment.output / "calibration.jsonl").read_text().splitlines()) == 1
    engine.fail_on_call = None
    experiment.calibrate()
    experiment.calibrate()
    assert len(engine.calls) == 3  # first success, failed second, successful retry only
    assert all(call["context_limit"] == 1536 for call in engine.calls)
    assert len((experiment.output / "calibration.jsonl").read_text().splitlines()) == 2
    experiment.calibration = experiment.calibration[:1]
    with pytest.raises(ValueError, match="Calibration plan changed"):
        experiment.calibrate()


def test_refit_cannot_replace_teacher_behind_cached_outcomes(tmp_path):
    experiment = make_experiment(tmp_path)
    experiment.observations[("fixture",)] = {
        "condition": "adapted_bidirectional", "teacher_checkpoint_sha256": "old"
    }
    with pytest.raises(ValueError, match="Cannot refit"):
        experiment.fit()
    checkpoint_hash = install_checkpoint(experiment)
    with pytest.raises(ValueError, match="different teacher checkpoint"):
        experiment.fit()
    experiment.observations[("fixture",)]["teacher_checkpoint_sha256"] = checkpoint_hash
    experiment.fit()  # completed fit is reused without ever loading an engine
    assert experiment.engine is None


def test_teacher_dependent_outcomes_bind_checkpoint_before_cache_reuse(tmp_path):
    from hindsight.data import Task

    engine = RecordingEngine()
    experiment = make_experiment(tmp_path, engine)
    checkpoint_hash = install_checkpoint(experiment)
    ep = SimpleNamespace(episode_id="a", partition="development", family="f")
    task = Task("q", "query", (1,), "f")
    result = experiment.evaluate(ep, [task], "Q", "adapted_bidirectional", adapter="student")
    assert result[0]["teacher_checkpoint_sha256"] == checkpoint_hash
    assert result[0]["success"]
    experiment.evaluate(ep, [task], "Q", "adapted_bidirectional", adapter="student")
    assert len(engine.calls) == 1
    experiment.observations[observation_key(result[0])]["teacher_checkpoint_sha256"] = "old"
    with pytest.raises(ValueError, match="different teacher checkpoint"):
        experiment.evaluate(ep, [task], "Q", "adapted_bidirectional", adapter="student")
    assert len(engine.calls) == 1


def test_fidelity_uses_full_distribution_and_detects_copying_on_disagreement():
    reference = torch.tensor([[0.1, 0.9], [0.2, 0.8]]).log()
    faithful = teacher_fidelity(reference, reference, [0, 1])
    assert faithful["forward_kl_reference_to_bidirectional"] == pytest.approx(0, abs=1e-7)
    assert faithful["action_positions"] == 2
    assert faithful["reference_disagreement_positions"] == 1
    assert faithful["bidirectional_historical_mass_on_disagreement"] == pytest.approx(0.1)
    copying = teacher_fidelity(reference, torch.tensor([[0.99, 0.01], [0.01, 0.99]]).log(), [0, 1])
    assert copying["forward_kl_reference_to_bidirectional"] > 1
    assert copying["bidirectional_historical_mass_on_disagreement"] == pytest.approx(0.99)
    no_disagreement = teacher_fidelity(reference, reference, [1, 1])
    assert no_disagreement["reference_disagreement_positions"] == 0
    assert no_disagreement["bidirectional_historical_mass_on_disagreement"] is None
    with pytest.raises(ValueError, match="aligned"):
        teacher_fidelity(reference, reference[:1], [0, 1])


class HeldoutEpisode(ControlledEpisode):
    def __init__(self, output, neutral=False):
        super().__init__(output, neutral)
        self.scored = []
        self.engine = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size=2)))

    def get_engine(self):
        return self.engine

    def validate_teacher(self):
        return "frozen-teacher-checkpoint"

    def evaluate(self, ep, tasks, role, condition, candidate=None, **kwargs):
        success = (condition == "direct_pi" and candidate == 0) or (
            condition == "adapted_pi" and candidate == 1 and not self.neutral
        )
        return [observation(task.task_id, success, role) for task in tasks]

    def score(self, generation, feedback, pi=None, **kwargs):
        self.scored.append((pi, kwargs.get("mode"), kwargs.get("adapter")))
        return super().score(generation, feedback, pi)

    def adaptation(self, *args, initial_logprobs_callback=None, **kwargs):
        if initial_logprobs_callback is not None:
            initial_logprobs_callback(torch.tensor([[0.5, 0.5]]).log())


@pytest.mark.parametrize("neutral", [False, True])
def test_heldout_fidelity_uses_actual_postupdate_winner_without_extra_forward(tmp_path, neutral):
    from hindsight.model import Generation

    experiment = HeldoutEpisode(tmp_path, neutral)
    ep = SimpleNamespace(episode_id="source", partition="development", family="translation",
                         validation=(SimpleNamespace(task_id="v"),),
                         queries=(SimpleNamespace(task_id="q"),))
    generation = Generation("task", [1], [0], "answer", False)
    experiment.run_episode(ep, "development", generation, "incorrect", ["pi0", "pi1"])
    assert len(experiment.scored) == 2  # Existing candidate PI forwards only, even for no-update.
    receipt = read_json(tmp_path / "fidelity-references/source.json")
    assert receipt["reference"]["selected_candidate"] == (None if neutral else 1)
    saved = torch.load(tmp_path / "fidelity-references/source.pt", weights_only=True)
    assert saved.exp()[0, 0].item() == pytest.approx(0.5 if neutral else 0.8)
    experiment.run_episode(ep, "development_hindsight", generation, "incorrect", [])
    assert len(experiment.scored) == 5  # Existing trained, untrained, raw forwards only.
    fidelity_path = tmp_path / "teacher-fidelity/source.json"
    first = fidelity_path.read_bytes()
    metrics = read_json(fidelity_path)
    assert metrics["provenance"]["derived_from"] == "post_update_V"
    assert metrics["provenance"]["teacher_checkpoint_sha256"] == "frozen-teacher-checkpoint"
    experiment.run_episode(ep, "development_hindsight", generation, "incorrect", [])
    assert fidelity_path.read_bytes() == first  # One record across an incomplete-phase resume.
    assert len(list((tmp_path / "teacher-fidelity").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="source or selection changed"):
        experiment.development_reference(ep, generation, "changed feedback")
    (tmp_path / "fidelity-references/source.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="tensor changed"):
        experiment.development_reference(ep, generation, "incorrect")


def test_final_fidelity_reference_is_not_persisted_and_teacher_cannot_be_swapped(tmp_path):
    from hindsight.model import Generation

    experiment = HeldoutEpisode(tmp_path)
    ep = SimpleNamespace(episode_id="source", partition="final_unseen", family="translation",
                         validation=(SimpleNamespace(task_id="v"),),
                         queries=(SimpleNamespace(task_id="q"),))
    generation = Generation("task", [1], [0], "answer", False)
    experiment.run_episode(ep, "final", generation, "incorrect", ["pi0", "pi1"])
    assert len(experiment.scored) == 5
    assert not (tmp_path / "fidelity-references").exists()
    assert (tmp_path / "teacher-fidelity/source.json").is_file()
    experiment.validate_teacher = lambda: "changed teacher"
    with pytest.raises(ValueError, match="teacher changed"):
        experiment.run_episode(ep, "final", generation, "incorrect", ["pi0", "pi1"])


def cached_meta_episode(experiment, episode_id, *, completed):
    ep = SimpleNamespace(episode_id=episode_id, partition="meta_train", family=episode_id,
                         source=SimpleNamespace(task_id=f"s-{episode_id}", prompt="task"),
                         validation=(SimpleNamespace(task_id=f"v-{episode_id}"),), queries=())
    write_json(experiment.output / "sources" / f"{episode_id}.json", {
        "task_id": ep.source.task_id, "feedback": "incorrect",
        "generation": {"prompt": "task", "prompt_ids": [1], "response_ids": [0],
                       "text": "answer", "cap_hit": False},
    })
    write_json(experiment.output / "proposals" / f"{episode_id}-0.json", {"text": "lesson"})
    if not completed:
        return ep
    for expected in experiment.episode_expected(ep, "meta_train"):
        experiment.observations[observation_key(expected)] = {**expected, "success": False}
    append_json(experiment.output / "selections.jsonl", {
        "episode_id": episode_id, "condition": "adapted_pi", "derived_from": "V",
        "selected_candidate": None,
    })
    target = experiment.output / "targets" / f"{episode_id}.pt"
    target.parent.mkdir(exist_ok=True)
    torch.save(torch.tensor([[0.5, 0.5]]).log(), target)
    write_json(target.with_suffix(".json"), {
        "identity_digest": digest(experiment.identity), "selected_candidate": None,
        "tensor_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    })
    append_json(experiment.output / "events.jsonl", {
        "event": "episode_completed", "episode_id": episode_id, "phase": "meta_train",
    })
    return ep


def test_completed_phase_rejected_before_compute_and_done_event_cannot_mask_missing_data(tmp_path):
    experiment = make_experiment(tmp_path)
    ep = cached_meta_episode(experiment, "a", completed=True)
    experiment.episodes = [ep]
    append_json(experiment.output / "events.jsonl", {
        "event": "phase_completed", "phase": "meta_train",
    })
    with pytest.raises(ValueError, match="already completed"):
        experiment.run("meta_train")
    assert experiment.engine is None
    one_key = next(iter(experiment.observations))
    record = experiment.observations.pop(one_key)
    with pytest.raises(ValueError, match="missing planned observations"):
        experiment.run("meta_train")
    experiment.observations[one_key] = record
    (experiment.output / "targets/a.pt").unlink()
    with pytest.raises(ValueError, match="missing teacher target"):
        experiment.run("meta_train")
    assert experiment.engine is None


def test_partial_phase_skips_completed_episode_and_records_explicit_reconstruction(tmp_path):
    experiment = make_experiment(tmp_path)
    done = cached_meta_episode(experiment, "a", completed=True)
    partial = cached_meta_episode(experiment, "b", completed=False)
    experiment.episodes = [done, partial]
    append_json(experiment.output / "events.jsonl", {
        "event": "episode_started", "episode_id": "b", "phase": "meta_train",
    })
    append_json(experiment.output / "events.jsonl", {
        "event": "student_adapted", "episode_id": "b", "metrics": [{}, {}, {}, {}],
    })
    executed = []
    experiment.run_episode = lambda ep, *args, **kwargs: executed.append(ep.episode_id)
    experiment.run("meta_train")
    assert executed == ["b"]
    assert experiment.engine is None
    events = [json.loads(line) for line in
              (experiment.output / "events.jsonl").read_text().splitlines()]
    reconstructed = [row for row in events if row["event"] == "episode_resume_reconstruction"]
    assert len(reconstructed) == 1
    assert reconstructed[0]["prior_student_updates"] == 4


def test_completed_phase_without_episode_evidence_is_rejected(tmp_path):
    experiment = make_experiment(tmp_path)
    experiment.episodes = [cached_meta_episode(experiment, "a", completed=False)]
    append_json(experiment.output / "events.jsonl", {
        "event": "phase_completed", "phase": "meta_train",
    })
    with pytest.raises(ValueError, match="missing episode completion evidence"):
        experiment.run("meta_train")
    assert experiment.engine is None


def test_completed_final_episode_requires_fidelity_bound_to_teacher_and_source(tmp_path):
    experiment = make_experiment(tmp_path)
    ep = cached_meta_episode(experiment, "a", completed=False)
    ep.partition = "final_known"
    ep.queries = (SimpleNamespace(task_id="q"),)
    teacher_hash = install_checkpoint(experiment)
    append_json(experiment.output / "selections.jsonl", {
        "episode_id": "a", "condition": "adapted_pi", "derived_from": "V",
        "selected_candidate": None,
    })
    for expected in experiment.episode_expected(ep, "final"):
        experiment.observations[observation_key(expected)] = {
            **expected, "success": False, "teacher_checkpoint_sha256": teacher_hash,
        }
    generation, feedback = experiment.source(ep)
    reference = torch.tensor([[0.5, 0.5]]).log()
    experiment.record_fidelity(ep, generation, reference,
                               experiment.reference_identity(ep, generation, feedback, None),
                               reference)
    experiment.validate_completed_episode(ep, "final")
    path = experiment.output / "teacher-fidelity/a.json"
    path.unlink()
    with pytest.raises(ValueError, match="missing fidelity evidence"):
        experiment.validate_completed_episode(ep, "final")
    assert experiment.engine is None
