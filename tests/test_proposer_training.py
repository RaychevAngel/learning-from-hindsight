"""Offline policy-gradient math, native generation, adapter isolation and E4 accounting."""

import copy
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_model import TinyTokenizer
from transformers import Qwen3Config

from hindsight.data import Episode, Task
from hindsight.experiment import read_json, rows, seed_for, write_json
from hindsight.model import Generation
from hindsight.proposer import PROPOSER_COUNTERS, Proposal, ProposerEngine, leave_one_out_advantages

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_proposer", ROOT / "scripts/train_proposer.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture
def engine():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = Qwen3Config(
        vocab_size=272,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2048,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    value = ProposerEngine(config=config, tokenizer=TinyTokenizer(), device="cpu")
    value.initialize_proposer(seed=17, lr=1e-3)
    return value


def fixed_proposals(engine):
    result = []
    for index, text in enumerate(("ab", "cd", "ef", "gh")):
        generation = Generation(
            "source-only",
            engine.encode_prompt("source-only"),
            engine.tokenizer.encode(text) + [1],
            "display text is not alignment",
            False,
        )
        proposal = Proposal(generation, 0.7, index + 10, engine.proposer_version, False, [])
        with engine._using_adapter("proposer"), torch.no_grad():
            proposal.behavior_token_logprobs = engine.proposal_token_logprobs(proposal).tolist()
        result.append(proposal)
    return result


def test_leave_one_out_reward_detachment_and_non_length_normalized_sequence_objective():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    advantages = leave_one_out_advantages(rewards)
    assert torch.allclose(advantages, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))
    assert not advantages.requires_grad
    logprobs = torch.tensor([-2.0, -3.0], requires_grad=True)
    loss = -advantages[0] * logprobs.sum() / 4
    loss.backward()
    assert logprobs.grad.tolist() == [-0.25, -0.25]
    assert rewards.grad is None


def test_real_policy_update_changes_only_proposer_and_improves_rewarded_sequence(engine):
    proposals = fixed_proposals(engine)
    protected = {
        name: value.detach().clone()
        for name, value in engine.model.named_parameters()
        if ".proposer." not in name
    }
    old = engine.proposer_state_sha256()
    initial = engine.proposer_state_sha256(initial=True)
    before = dict(engine.counters)
    rewarded_before = sum(proposals[0].behavior_token_logprobs)
    metrics = engine.reinforce_group(
        proposals, torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    )
    assert metrics["optimizer_steps"] == 1 and metrics["grad_norm"] > 0
    assert engine.proposer_state_sha256() != old
    assert engine.proposer_state_sha256(initial=True) == initial == old
    assert all(
        torch.equal(value, dict(engine.model.named_parameters())[name])
        for name, value in protected.items()
    )
    assert all(engine.counters[key] == before[key] for key in D.CORE_COUNTERS)
    assert engine.counters["proposer_optimizer_steps"] == 1
    assert engine.counters["proposer_backward_sequences"] == 4
    assert engine.counters["proposer_scoring_forwards"] - before["proposer_scoring_forwards"] == 4
    with engine._using_adapter("proposer"), torch.no_grad():
        assert float(engine.proposal_token_logprobs(proposals[0]).sum()) > rewarded_before
    with pytest.raises(ValueError, match="fresh on-policy"):
        engine.reinforce_group(proposals, [1, 0, 0, 0])


def test_equal_rewards_are_truthful_zero_update_without_scoring_or_adam_momentum(engine):
    proposals = fixed_proposals(engine)
    before = dict(engine.counters)
    state = engine.proposer_state_sha256()
    metrics = engine.reinforce_group(proposals, [0.5] * 4)
    assert metrics["optimizer_steps"] == 0 and metrics["advantages"] == [0.0] * 4
    assert engine.proposer_state_sha256() == state and not engine.proposer_optimizer.state
    assert engine.counters["proposer_scoring_forwards"] == before["proposer_scoring_forwards"]
    assert engine.counters["proposer_zero_advantage_groups"] == 1
    assert engine.counters["proposer_groups"] == 1


def test_full_categorical_sampling_and_behavior_probabilities_match_temperature(
    engine, monkeypatch
):
    raw = torch.arange(272, dtype=torch.float32) / 272
    seen = {}

    def generate(input_ids, attention_mask, generation_config):
        seen["config"] = generation_config
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        return SimpleNamespace(
            sequences=torch.cat((input_ids, torch.tensor([[100, 1]])), 1),
            scores=(raw.unsqueeze(0) / 0.7, raw.unsqueeze(0) / 0.7),
        )

    monkeypatch.setattr(engine.model, "generate", generate)
    value = engine.sample_proposal("public source", max_new_tokens=3, temperature=0.7, seed=19)
    assert seen["config"].top_p == 1 and seen["config"].top_k == 0
    assert seen["config"].repetition_penalty == 1 and seen["config"].do_sample
    assert value.generation.response_ids == [100, 1] and not value.generation.cap_hit
    assert value.generation.prompt_ids == engine.encode_prompt("public source")
    expected = (raw / 0.7).log_softmax(-1)
    assert value.behavior_token_logprobs == [float(expected[100]), float(expected[1])]
    assert (
        engine.counters["proposer_generation_sequences"]
        == engine.counters["generation_sequences"]
        == 1
    )


def test_real_tiny_generation_and_temperature_adjusted_native_predecessor_gradient(engine):
    sample = engine.sample_proposal("a", max_new_tokens=2, temperature=0.7, seed=21)
    assert 1 <= len(sample.generation.response_ids) <= 2
    proposal = fixed_proposals(engine)[0]
    gen = proposal.generation
    with engine._using_adapter("proposer", trainable=True):
        actual = engine.proposal_token_logprobs(proposal)
        positions = list(
            range(len(gen.prompt_ids) - 1, len(gen.prompt_ids) + len(gen.response_ids) - 1)
        )
        raw = (
            engine.model(
                input_ids=torch.tensor([gen.prompt_ids + gen.response_ids]),
                logits_to_keep=torch.tensor(positions),
                use_cache=False,
            )
            .logits[0]
            .float()
        )
        expected = (
            (raw / proposal.temperature)
            .log_softmax(-1)
            .gather(1, torch.tensor(gen.response_ids).unsqueeze(1))
            .squeeze(1)
        )
        assert actual.requires_grad and torch.allclose(actual, expected, atol=1e-6)
        edited = dataclasses.replace(
            proposal, generation=dataclasses.replace(gen, text="changed display")
        )
        assert torch.equal(actual, engine.proposal_token_logprobs(edited))


def test_complete_checkpoint_roundtrip_and_corrupt_tensor_fails_before_mutation(engine, tmp_path):
    engine.reinforce_group(fixed_proposals(engine), [1, 0, 0, 0])
    path = tmp_path / "proposer.pt"
    engine.save_proposer(path, metadata={"proof": "fixture"})
    trained, initial = engine.proposer_state_sha256(), engine.proposer_state_sha256(initial=True)
    engine._reset_adapter("proposer", 99)
    assert engine.proposer_state_sha256() != trained
    engine.load_proposer(path)
    assert engine.proposer_state_sha256() == trained
    assert engine.proposer_state_sha256(initial=True) == initial
    payload = torch.load(path, weights_only=True)
    payload["proposer"].pop(next(iter(payload["proposer"])))
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="tensor names"):
        engine.load_proposer(invalid)
    assert engine.proposer_state_sha256() == trained


@pytest.mark.parametrize("defect", ["parameter_group", "moment_shape"])
def test_corrupt_optimizer_fails_before_adapter_mutation(engine, tmp_path, defect):
    engine.reinforce_group(fixed_proposals(engine), [1, 0, 0, 0])
    path = tmp_path / "proposer.pt"
    engine.save_proposer(path)
    payload = torch.load(path, weights_only=True)
    if defect == "parameter_group":
        payload["optimizer"]["param_groups"][0]["params"].pop()
    else:
        next(iter(payload["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
    torch.save(payload, path)
    engine._reset_adapter("proposer", 99)
    before = engine.proposer_state_sha256()
    with pytest.raises(ValueError):
        engine.load_proposer(path)
    assert engine.proposer_state_sha256() == before


@pytest.fixture
def cohort(tmp_path):
    episodes = []
    for index, partition in enumerate(("meta_train", "meta_train", "confirmation")):

        def task(role, number):
            name = f"e{index}-{role}{number}"
            return Task(name, "PUBLIC " + name, (1,), "PRIVATE FAMILY")

        episodes.append(
            Episode(
                f"e{index}",
                partition,
                "PRIVATE FAMILY",
                task("S", 0),
                tuple(task("V", i) for i in range(8)),
                tuple(task("Q", i) for i in range(8)),
            )
        )
    data = tmp_path / "data"
    write_json(data / "manifest.json", {"fixture": "synthetic cohort"})
    config = dict(
        model_name="offline",
        revision="pinned",
        seed=7301,
        source_tokens=128,
        pi_tokens=80,
        evaluation_tokens=160,
        student_steps=4,
        learning_rate=1e-4,
        generation_batch_size=4,
        max_context_tokens=1536,
        max_teacher_context_tokens=4096,
        deterministic_execution=True,
        max_invocation_seconds=60,
        absolute_deadline_utc="2099-01-01T00:00:00+00:00",
        dataset_manifest_sha256=D.sha(data / "manifest.json"),
    )
    qualification = tmp_path / "qualification.json"
    write_json(
        qualification,
        dict(
            status="completed",
            selected_batch_size=4,
            repeat_stability={"4": True},
            config={key: config[key] for key in ("model_name", "revision")},
            dataset_sha256=config["dataset_manifest_sha256"],
        ),
    )
    config["batch_qualification"] = dict(path=str(qualification), sha256=D.sha(qualification))
    return SimpleNamespace(
        data=data,
        output=tmp_path / "result",
        config=config,
        episodes=episodes,
        loader=lambda path: (episodes, []),
    )


class FakeEngine:
    def __init__(self, cohort, *, invalid=False, fail_generation=None, ties=False):
        self.cohort, self.invalid = cohort, invalid
        self.fail_generation, self.ties = fail_generation, ties
        self.counters = dict.fromkeys(D.CORE_COUNTERS, 0)
        self.calls, self.proposals, self.scores, self.resets, self.updates = [], [], [], [], []
        self.student_choice, self.weight, self.proposer_version = None, 0, 0
        self.budget_check = lambda: None

    def initialize_proposer(self, **kwargs):
        self.init_kwargs = kwargs
        self.counters.update(dict.fromkeys(PROPOSER_COUNTERS, 0))

    def encode_prompt(self, prompt):
        return [ord(c) for c in prompt]

    def generate(self, prompts, *, adapter, **kwargs):
        self.budget_check()
        self.calls.append((prompts, adapter, kwargs))
        if len(self.calls) == self.fail_generation:
            raise RuntimeError("controlled generation failure")
        assert all("hindsight_advice" not in prompt for prompt in prompts)
        success = adapter == "student" and self.student_choice == 1 and not self.ties
        values = [
            Generation(
                prompt,
                self.encode_prompt(prompt),
                [10, 1],
                "FINAL: [1]" if success or "-S0" in prompt else "FINAL: [0]",
                False,
            )
            for prompt in prompts
        ]
        self.counters["generation_sequences"] += len(values)
        return values

    def sample_proposal(self, prompt, *, max_new_tokens, temperature, seed, initial=False):
        self.budget_check()
        assert "PRIVATE FAMILY" not in prompt and "-V" not in prompt and "-Q" not in prompt
        choice = len(self.proposals) % 4
        text = "" if self.invalid else f"Lesson {choice}"
        result = Proposal(
            Generation(prompt, self.encode_prompt(prompt), [choice + 10, 1], text, False),
            temperature,
            seed,
            self.proposer_version,
            initial,
            [-1.0, -1.0],
        )
        self.proposals.append(result)
        self.counters["proposer_generation_sequences"] += 1
        self.counters["generation_sequences"] += 1
        return result

    def score_rollout(self, generation, feedback, *, pi):
        self.scores.append((generation, feedback, pi))
        self.counters["scoring_forwards"] += 1
        return torch.tensor([float(pi.split()[-1])])

    def reset_student(self, seed):
        self.resets.append(seed)

    def adapt(self, generation, target, *, steps, lr):
        self.student_choice = int(target.item())
        self.counters["student_optimizer_steps"] += steps
        return [{"loss": 0.1, "optimizer_steps": 1}] * steps

    def proposer_state_sha256(self, *, initial=False):
        return str(0 if initial else self.weight)

    def reinforce_group(self, proposals, rewards):
        advantage = leave_one_out_advantages(rewards)
        changed = bool(torch.count_nonzero(advantage))
        self.weight += int(changed)
        self.proposer_version += 1
        self.updates.append((proposals, rewards))
        self.counters["proposer_groups"] += 1
        self.counters["proposer_optimizer_steps"] += int(changed)
        self.counters["proposer_zero_advantage_groups"] += int(not changed)
        self.counters["proposer_scoring_forwards"] += 4 * int(changed)
        self.counters["proposer_backward_sequences"] += 4 * int(changed)
        return dict(optimizer_steps=int(changed), rewards=rewards, advantages=advantage.tolist())

    def save_proposer(self, path, metadata=None):
        write_json(path, {"weight": self.weight, "metadata": metadata})

    def load_proposer(self, path):
        value = read_json(path)
        self.weight = value["weight"]
        return value


def invoke(cohort, engine, **kwargs):
    return D.run(
        cohort.config, cohort.data, cohort.output, engine=engine, loader=cohort.loader, **kwargs
    )


def test_training_actual_post_update_stratified_v_rewards_exact_work_and_no_query_access(cohort):
    engine = FakeEngine(cohort)
    assert invoke(cohort, engine) == 0
    receipt = read_json(cohort.output / "receipt.json")
    assert receipt["status"] == "completed" and receipt["actual_sources"] == 2
    assert receipt["work"]["generation_sequences"] == 50
    assert receipt["work"]["scoring_forwards"] == 8
    assert receipt["work"]["student_optimizer_steps"] == 32
    assert receipt["work"]["proposer_optimizer_steps"] == 2
    assert len(engine.proposals) == len(engine.scores) == 8
    assert engine.resets == [seed_for(7301, "student_initialization")] * 8
    assert all(group[1] == [0.0, 1.0, 0.0, 0.0] for group in engine.updates)
    observations = rows(cohort.output / "observations.jsonl")
    assert len(observations) == 40 and {row["role"] for row in observations} == {"V"}
    assert {int(row["task_id"][-1]) for row in observations} == {0, 2, 4, 6}
    assert all(len(prompts) in (1, 4) for prompts, *_ in engine.calls)
    assert all(row["selected_candidate"] == 1 for row in rows(cohort.output / "groups.jsonl"))
    assert read_json(cohort.output / "checkpoints/final.pt")["weight"] == 2
    saved = (cohort.output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(cohort, engine)
    assert (cohort.output / "receipt.json").read_bytes() == saved


def test_invalid_proposals_reuse_measured_baseline_and_zero_advantages_without_fake_generations(
    cohort,
):
    engine = FakeEngine(cohort, invalid=True)
    assert invoke(cohort, engine) == 0
    work = read_json(cohort.output / "receipt.json")["work"]
    assert work["generation_sequences"] == 18 and work["scoring_forwards"] == 0
    assert work["student_optimizer_steps"] == work["proposer_optimizer_steps"] == 0
    assert work["proposer_zero_advantage_groups"] == 2
    observations = rows(cohort.output / "observations.jsonl")
    assert sum(row["reused_baseline"] for row in observations) == 32
    assert all(row["selected_candidate"] is None for row in rows(cohort.output / "groups.jsonl"))


def test_frozen_init_and_trained_equal_budget_eval_is_disjoint_and_has_no_rl_updates(
    cohort, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pt"
    write_json(
        checkpoint,
        {
            "weight": 9,
            "metadata": dict(
                dataset_manifest_sha256=D.sha(cohort.data / "manifest.json"),
                train_episode_ids=["e0", "e1"],
                student_recipe={
                    key: cohort.config[key] for key in ("student_steps", "learning_rate", "seed")
                },
                proposer_temperature=0.7,
                proposer_module_sha256=D.sha(ROOT / "src/hindsight/proposer.py"),
            ),
        },
    )
    engine = FakeEngine(cohort)
    assert invoke(cohort, engine, phase="evaluate", checkpoint=checkpoint) == 0
    receipt = read_json(cohort.output / "receipt.json")
    assert receipt["work"]["generation_sequences"] == 51
    assert receipt["work"]["student_optimizer_steps"] == 8
    assert receipt["work"]["proposer_optimizer_steps"] == 0
    assert (
        len(engine.proposals) == 2
        and engine.proposals[0].initial
        and not engine.proposals[1].initial
    )
    assert engine.proposals[0].seed == engine.proposals[1].seed
    assert not engine.updates and engine.weight == 9
    observations = rows(cohort.output / "observations.jsonl")
    assert {row["episode_id"] for row in observations} == {"e2"}
    assert {row["condition"] for row in observations} == {
        "baseline",
        "initial_proposer",
        "trained_proposer",
    }


def test_controlled_failure_preserves_partial_counts_without_retry(cohort):
    engine = FakeEngine(cohort, fail_generation=3)
    assert invoke(cohort, engine) == 1
    receipt = read_json(cohort.output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["actual_sources"] == 0
    assert receipt["work"]["generation_sequences"] == 9
    assert receipt["work"]["student_optimizer_steps"] == 4
    assert len(engine.calls) == 3 and len(engine.proposals) == 4
    assert len(list((cohort.output / "proposals").glob("*.json"))) == 4


@pytest.mark.parametrize("defect", ["overlap", "wrong_partition", "deadline", "bad_k"])
def test_plan_defects_fail_before_model_work(cohort, defect):
    config = copy.deepcopy(cohort.config)
    if defect == "overlap":
        config.update(train_episode_ids=["e0"], evaluation_episode_ids=["e0"])
    elif defect == "wrong_partition":
        config.update(train_episode_ids=["e2"], evaluation_episode_ids=["e0"])
    elif defect == "deadline":
        config["absolute_deadline_utc"] = "2000-01-01T00:00:00+00:00"
    else:
        config["proposer_k"] = 3
    cohort.config = config
    engine = FakeEngine(cohort)
    assert invoke(cohort, engine) == 1
    assert not any(engine.counters.values())


@pytest.mark.parametrize(
    "defect",
    [
        "dataset",
        "receipt_sha",
        "status",
        "selected_batch_size",
        "repeat_stability",
        "model",
        "receipt_dataset",
    ],
)
def test_batch_and_dataset_qualification_fail_before_model_work(cohort, defect):
    if defect == "dataset":
        cohort.config["dataset_manifest_sha256"] = "wrong"
    elif defect == "receipt_sha":
        cohort.config["batch_qualification"]["sha256"] = "wrong"
    else:
        path = Path(cohort.config["batch_qualification"]["path"])
        receipt = read_json(path)
        if defect == "status":
            receipt["status"] = "failed"
        elif defect == "selected_batch_size":
            receipt["selected_batch_size"] = 8
        elif defect == "repeat_stability":
            receipt["repeat_stability"]["4"] = False
        elif defect == "model":
            receipt["config"]["revision"] = "different"
        else:
            receipt["dataset_sha256"] = "different"
        write_json(path, receipt)
        cohort.config["batch_qualification"]["sha256"] = D.sha(path)
    engine = FakeEngine(cohort)
    assert invoke(cohort, engine) == 1
    assert not any(engine.counters.values())
    assert read_json(cohort.output / "receipt.json")["status"] == "failed"
