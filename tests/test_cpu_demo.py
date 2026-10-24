"""The public API demonstration remains entirely offline and bounded."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_demo", ROOT / "examples/cpu_mechanism_demo.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_tiny_api_sequence_has_real_optimizer_steps_but_no_performance_claim():
    result = DEMO.execute()
    assert result["device"] == "cpu"
    assert result["pretrained_weights_downloaded"] is False
    assert result["native_target_shape"] == [4, 272] and result["target_detached"]
    assert result["explicit_student_steps"] == result["hindsight_student_steps"] == 2
    assert result["teacher_optimizer_steps"] == 4 and result["teacher_example_exposures"] == 6
    assert result["work"]["student_optimizer_steps"] == 4
    assert result["next_task_prompt"] == "different task"
    assert result["performance_claim"] is None
