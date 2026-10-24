"""No CUDA or model downloads: E0 helpers and a complete tiny injected-engine run."""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hindsight.data import Task

MODEL_TESTS = Path(__file__).with_name("test_model.py")
SPEC = importlib.util.spec_from_file_location("offline_model_helpers", MODEL_TESTS)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_gpu.py"
SPEC = importlib.util.spec_from_file_location("preflight_gpu", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_forecast_tracks_current_160_token_plan():
    measurements = {key: {"seconds": 1.0} for key in (
        "generate_source", "generate_pi", "generate_greedy_batch", "adapt_first",
        "adapt_repeat", "reset_matched", "reset_neutral", "score_causal",
        "score_causal_edit", "score_bidir", "score_bidir_edit", "score_pi",
    )}
    rows = {"source": [{"response_ids": [1, 2]}], "pi": [{"response_ids": [1, 2]}],
            "greedy_batch": [{"response_ids": [1, 2]}] * 4}
    config = {"source_tokens": 128, "pi_tokens": 80, "evaluation_tokens": 160,
              "teacher_batch_size": 4, "generation_batch_size": 8}
    forecast = PREFLIGHT.build_forecast(measurements, rows, config)
    assert forecast["planned_generations"] == 8976
    assert forecast["planned_output_token_ceiling"] == 1_405_184
    assert forecast["planned_student_steps"] == 2112
    assert forecast["planned_teacher_steps"] == 64
    assert forecast["projected_core_seconds"] > 0


def test_deadline_rejects_naive_and_expired_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        PREFLIGHT.Deadline(1, "2020-01-01T00:00:00")
    deadline = PREFLIGHT.Deadline(1, "2020-01-01T00:00:00+00:00")
    with pytest.raises(TimeoutError, match="Absolute"):
        deadline()


def test_full_tiny_cpu_e0_has_exact_diagnostic_counts(monkeypatch):
    engine = HELPERS.engine.__wrapped__()
    config = {"seed": 7, "source_tokens": 3, "pi_tokens": 3, "evaluation_tokens": 3,
              "generation_batch_size": 8, "teacher_batch_size": 4,
              "max_context_tokens": 1024, "max_teacher_context_tokens": 1024,
              "learning_rate": 1e-3}
    # The real tiny model still scores and trains. Fixed tiny generations avoid
    # interpreting random-EOS luck as a numerical qualification failure.
    def fixed_generate(prompts, max_new_tokens, sample, seed, adapter=None, **kwargs):
        values = [HELPERS.Generation(prompt, engine.encode_prompt(prompt), [100, 101, 1],
                                     "ab", False) for prompt in prompts]
        engine.counters["generation_sequences"] += len(values)
        return values

    monkeypatch.setattr(engine, "generate", fixed_generate)
    tasks = [Task(str(i), "task " + str(i), (1, 2), "development") for i in range(6)]
    receipt = {}
    deadline = PREFLIGHT.Deadline(
        60, (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    )
    PREFLIGHT.run_checks(engine, tasks, config, receipt, deadline)
    assert receipt["status"] == "passed_numerical_checks"
    for key, limit in PREFLIGHT.LIMITS.items():
        assert receipt["actual_work"][key] == limit
    assert receipt["actual_work"]["teacher_optimizer_steps"] == 0
    assert all(check["passed"] for check in receipt["checks"].values())


def test_no_cuda_writes_failure_receipt_without_loading_weights(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    output = tmp_path / "receipt.json"
    config.write_text(json.dumps({
        "absolute_deadline_utc": (
            datetime.now(timezone.utc) + timedelta(minutes=1)
        ).isoformat()
    }))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Weights must never be loaded without CUDA")

    monkeypatch.setattr(PREFLIGHT.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(PREFLIGHT, "HFEngine", forbidden_load)
    monkeypatch.setattr("sys.argv", ["preflight_gpu.py", "--config", str(config),
                                    "--output", str(output)])
    assert PREFLIGHT.main() == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert "no checkpoint was loaded" in receipt["error"]["message"]
