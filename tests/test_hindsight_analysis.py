"""Post-run analysis rejects incomplete or tampered evidence without a model."""

import importlib.util
from pathlib import Path

import pytest
from test_hindsight_teacher import Engine, inputs, invoke

from hindsight.experiment import append_json, read_json, rows, write_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_hindsight_teacher.py"
SPEC = importlib.util.spec_from_file_location("analyze_hindsight_teacher", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
__all__ = ["inputs"]


@pytest.fixture
def completed(inputs, monkeypatch):
    assert invoke(inputs, Engine(inputs.output)) == 0
    monkeypatch.setattr(A, "load_corpus", lambda path: (inputs.corpus[path], []))
    qualification = inputs.output / "qualification/qualification.json"
    write_json(qualification, dict(status="passed_teacher_qualification",
                                   actual_work=A.QUALIFICATION))
    fit = inputs.output / "teacher/FIT-COMPLETE.json"
    value = read_json(fit)
    value["metrics"].update(losses=[0.1] * 64, grad_norms=[0.2] * 64)
    write_json(fit, value)
    return inputs


def test_independent_regrade_counts_clusters_fidelity_and_metrics(completed):
    report = A.analyze(completed.output, completed.data)
    assert report["regraded_observations"] == 512
    assert report["work"] == A.WORK
    assert report["fit_target_kinds"] == {"PI": 6, "neutral": 2}
    baseline = report["fitted_q_contrasts"]["baseline"]
    assert baseline["gains"] == 64 and baseline["losses"] == 0
    assert baseline["episodes"] == 8 and baseline["mean_difference"] == 1
    assert baseline["ci95"] == [1, 1] and baseline["bootstrap_samples"] == 10000
    assert report["fitted_q_contrasts"]["adapted_pi"]["mean_difference"] == 0
    assert report["fitted_q_contrasts"]["adapted_untrained_bidirectional"]["ci95"] == [0, 0]
    assert set(report["fitted_q_contrasts"]) == {
        "baseline", "adapted_untrained_bidirectional", "adapted_pi",
        "adapted_raw_causal", "adapted_replay",
    }
    for control in ("adapted_raw_causal", "adapted_replay"):
        contrast = report["fitted_q_contrasts"][control]
        assert contrast["mean_difference"] == 0 and contrast["ci95"] == [0, 0]
        assert contrast["episodes"] == 8 and contrast["bootstrap_samples"] == 10000
        assert contrast["seed"] == 7301 and contrast["gains"] == contrast["losses"] == 0
    assert len(report["fidelity"]["fit-fidelity-before"]["per_source"]) == 8
    assert len(report["student_adaptations"]) == 40 and report["failures"] == []
    assert len(report["per_source"]) == 8


@pytest.mark.parametrize("defect", ["duplicate", "regrade", "work", "checkpoint", "fit", "qual"])
def test_rejects_incomplete_or_tampered_results(completed, defect):
    result = completed.output
    if defect == "duplicate":
        append_json(result / "observations.jsonl", rows(result / "observations.jsonl")[0])
    elif defect == "regrade":
        observed = rows(result / "observations.jsonl")
        observed[0]["text"] = "FINAL: [1]"
        (result / "observations.jsonl").unlink()
        for row in observed:
            append_json(result / "observations.jsonl", row)
    elif defect == "checkpoint":
        (result / "teacher/checkpoint.pt").write_bytes(b"changed")
    else:
        path = result / {"work": "receipt.json", "fit": "teacher/FIT-COMPLETE.json",
                         "qual": "qualification/qualification.json"}[defect]
        value = read_json(path)
        if defect == "work":
            value["work"]["scoring_forwards"] -= 1
        elif defect == "fit":
            value["metrics"]["losses"][0] = float("nan")
        else:
            value["status"] = "failed"
        write_json(path, value)
    with pytest.raises(ValueError):
        A.analyze(result, completed.data)
