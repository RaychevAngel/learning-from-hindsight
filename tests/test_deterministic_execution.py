"""Opt-in process configuration, tested without initializing CUDA or loading weights."""

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from hindsight.experiment import Experiment
from hindsight.model import configure_deterministic_execution

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def restore_numeric_settings(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    # Track the environment value so the helper cannot affect subsequent tests.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark


@pytest.mark.parametrize("enabled", [False, None])
def test_default_does_not_mutate_existing_settings(monkeypatch, enabled):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Default path must not configure or initialize CUDA")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    benchmark = torch.backends.cudnn.benchmark
    result = (configure_deterministic_execution() if enabled is None
              else configure_deterministic_execution(enabled))
    assert result == {
        "requested": False,
        "cublas_workspace_config": ":16:8",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert torch.backends.cudnn.benchmark == benchmark


def test_opt_in_sets_workspace_before_strict_algorithms(monkeypatch, restore_numeric_settings):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    original = torch.use_deterministic_algorithms
    calls = []

    def record(enabled, *, warn_only):
        calls.append((enabled, warn_only, os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
        original(enabled, warn_only=warn_only)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", record)
    result = configure_deterministic_execution(True)
    assert calls == [(True, False, ":4096:8")]
    assert result["requested"] is True
    assert result["cublas_workspace_config"] == ":4096:8"
    assert result["deterministic_algorithms"] is True
    assert result["deterministic_warn_only"] is False
    assert result["cudnn_benchmark"] is False


def test_late_cuda_init_rejected_only_when_workspace_not_configured(
    monkeypatch, restore_numeric_settings
):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        configure_deterministic_execution(True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert configure_deterministic_execution(True)["deterministic_algorithms"] is True


@pytest.mark.parametrize("opt_in", [False, True])
def test_experiment_configures_before_corpus_and_lazy_engine(monkeypatch, tmp_path, opt_in):
    events = []
    config = {
        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
        "max_invocation_seconds": 60,
        "model_name": "offline", "revision": "test",
        "max_context_tokens": 1536, "max_teacher_context_tokens": 4096,
    }
    if opt_in:
        config["deterministic_execution"] = True
    monkeypatch.setattr("hindsight.model.configure_deterministic_execution",
                        lambda enabled: events.append(("configure", enabled)))
    monkeypatch.setattr("hindsight.data.load_corpus",
                        lambda path: events.append("corpus") or ([], []))
    monkeypatch.setattr(Experiment, "code_identity", lambda self: {})
    monkeypatch.setattr(Experiment, "event", lambda *a, **kw: None)
    monkeypatch.setattr("hindsight.model.HFEngine",
                        lambda **kw: events.append("engine") or object())
    (tmp_path / "manifest.json").write_text("{}")
    experiment = Experiment(config, tmp_path, tmp_path / "output", device="cpu")
    experiment.get_engine()
    assert events == ([("configure", True)] if opt_in else []) + ["corpus", "engine"]


@pytest.mark.parametrize("opt_in", [False, True])
def test_preflight_records_flags_before_any_cuda_probe(monkeypatch, tmp_path, opt_in):
    spec = importlib.util.spec_from_file_location("deterministic_preflight",
                                                ROOT / "scripts" / "preflight_gpu.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    events = []
    snapshot = {"requested": opt_in, "deterministic_algorithms": opt_in,
                "deterministic_warn_only": False, "cudnn_benchmark": False,
                "cudnn_deterministic": False,
                "cublas_workspace_config": ":4096:8" if opt_in else None}

    def configure(enabled):
        events.append(("configure", enabled))
        return snapshot

    def unavailable():
        events.append("cuda_probe")
        return False

    monkeypatch.setattr(preflight, "configure_deterministic_execution", configure)
    monkeypatch.setattr(torch.cuda, "is_available", unavailable)
    monkeypatch.setattr(preflight, "HFEngine",
                        lambda **kw: pytest.fail("No model load permitted"))
    config = {"absolute_deadline_utc": (
        datetime.now(timezone.utc) + timedelta(minutes=1)
    ).isoformat()}
    if opt_in:
        config["deterministic_execution"] = True
    config_path, output = tmp_path / "config.json", tmp_path / "receipt.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", ["preflight_gpu.py", "--config", str(config_path),
                                    "--output", str(output)])
    assert preflight.main() == 1
    assert events == [("configure", opt_in), "cuda_probe"]
    receipt = json.loads(output.read_text())
    assert receipt["numerical_execution"] == snapshot
    assert "no checkpoint was loaded" in receipt["error"]["message"]


def test_only_qwen35_config_opts_in():
    assert json.loads((ROOT / "configs" / "qwen35-4b.json").read_text())[
        "deterministic_execution"
    ] is True
    assert "deterministic_execution" not in json.loads(
        (ROOT / "configs" / "initial.json").read_text()
    )
