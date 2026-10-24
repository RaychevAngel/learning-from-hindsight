"""Qualification entry-point budget probes without a model, CUDA call or sleep."""

import importlib.util
import json
from datetime import datetime as RealDatetime
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "throughput_budget", ROOT / "scripts/qualify_overnight_throughput.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
NOW = RealDatetime(2026, 10, 24, 10, tzinfo=timezone.utc)


class StopAfterProbe(Exception):
    pass


def invoke(
    tmp_path,
    monkeypatch,
    *,
    seconds=1,
    elapsed=2,
    omit_seconds=False,
    deadline=None,
    wall_time=None,
):
    clock = SimpleNamespace(elapsed=0, wall=NOW)

    class Datetime:
        fromisoformat = RealDatetime.fromisoformat

        @classmethod
        def now(cls, tz):
            return clock.wall

    monkeypatch.setattr(D, "datetime", Datetime)
    monkeypatch.setattr(D.time, "monotonic", lambda: clock.elapsed)
    monkeypatch.setattr(D, "configure_deterministic_execution", lambda *_: None)
    monkeypatch.setattr(
        D,
        "load_cohort",
        lambda _: (
            [
                SimpleNamespace(
                    partition="development",
                    validation=[SimpleNamespace(task_id=f"task-{index}", prompt="task")],
                )
                for index in range(16)
            ],
            [],
        ),
    )
    constructed, admitted = [], []

    def engine(**kwargs):
        constructed.append(True)
        clock.elapsed = elapsed
        if wall_time is not None:
            clock.wall = wall_time
        kwargs["budget_check"]()
        admitted.append(True)
        raise StopAfterProbe("No model is constructed by this deterministic probe")

    monkeypatch.setattr(D, "HFEngine", engine)
    config = {
        "model_name": "unused",
        "revision": "unused",
        "absolute_deadline_utc": (NOW + timedelta(days=1)).isoformat()
        if deadline is None
        else deadline,
    }
    if not omit_seconds:
        config["max_invocation_seconds"] = seconds
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    output = tmp_path / "output"
    completed = D.run(path, data, output)
    receipt = json.loads((output / "receipt.json").read_text())
    return SimpleNamespace(
        completed=completed,
        receipt=receipt,
        constructed=constructed,
        admitted=admitted,
        output=output,
    )


@pytest.mark.parametrize(
    "seconds,elapsed,passes",
    [
        (1, 0.999, True),
        (1, 1, False),
        (1, 2, False),
        (7200, 601, True),
        (7200, 7199.99, True),
        (7200, 7200, False),
        (0.5, 0.49, True),
        (0.5, 0.5, False),
    ],
)
def test_actual_constructor_callback_honors_configured_limit(
    tmp_path,
    monkeypatch,
    seconds,
    elapsed,
    passes,
):
    result = invoke(tmp_path, monkeypatch, seconds=seconds, elapsed=elapsed)
    assert result.completed is False and result.receipt["status"] == "failed"
    assert result.constructed == [True] and bool(result.admitted) is passes
    assert result.receipt["max_invocation_seconds"] == seconds
    assert result.receipt["error"]["type"] == ("StopAfterProbe" if passes else "TimeoutError")
    assert result.receipt["work"] == {}


@pytest.mark.parametrize("elapsed,passes", [(599.99, True), (600, False), (601, False)])
def test_missing_limit_retains_legacy_600_second_default(tmp_path, monkeypatch, elapsed, passes):
    result = invoke(tmp_path, monkeypatch, omit_seconds=True, elapsed=elapsed)
    assert bool(result.admitted) is passes
    assert result.receipt["max_invocation_seconds"] == 600
    assert result.receipt["error"]["type"] == ("StopAfterProbe" if passes else "TimeoutError")


@pytest.mark.parametrize(
    "seconds", [0, -1, None, True, False, "1", float("nan"), float("inf"), float("-inf")]
)
def test_malformed_limit_is_rejected_before_output_or_model(tmp_path, monkeypatch, seconds):
    with pytest.raises(ValueError, match="positive finite"):
        invoke(tmp_path, monkeypatch, seconds=seconds)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("deadline", ["not a date", "2026-10-25T00:00:00", 1, ""])
def test_malformed_or_unzoned_deadline_is_rejected_before_output_or_model(
    tmp_path,
    monkeypatch,
    deadline,
):
    with pytest.raises(ValueError, match="timezone-aware"):
        invoke(tmp_path, monkeypatch, deadline=deadline)
    assert not (tmp_path / "output").exists()


def test_absolute_deadline_independently_wins_over_long_invocation_allowance(tmp_path, monkeypatch):
    deadline = NOW + timedelta(seconds=5)
    result = invoke(
        tmp_path,
        monkeypatch,
        seconds=7200,
        elapsed=0.25,
        deadline=deadline.isoformat(),
        wall_time=deadline,
    )
    assert result.constructed == [True] and result.admitted == []
    assert result.receipt["status"] == "failed"
    assert result.receipt["error"]["type"] == "TimeoutError"


def test_already_expired_absolute_deadline_stops_before_engine_constructor(tmp_path, monkeypatch):
    result = invoke(tmp_path, monkeypatch, seconds=7200, deadline=NOW.isoformat())
    assert result.constructed == [] and result.admitted == []
    assert result.receipt["status"] == "failed"
    assert result.receipt["error"]["type"] == "TimeoutError"
