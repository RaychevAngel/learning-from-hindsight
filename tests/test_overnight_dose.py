import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dose", Path(__file__).parents[1] / "scripts/check_overnight_dose.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def observations():
    return [
        dict(episode_id="e", task_id="t", role="V", condition=c, success=False)
        for c in ("baseline", "direct_pi", "steps_4", "steps_16")
    ]


def test_recipe_uses_actual_v_and_cheaper_tie():
    records = observations()
    assert D.select_recipe(records)["student_steps"] == 4
    records[-1]["success"] = True
    assert D.select_recipe(records)["student_steps"] == 16


def test_recipe_rejects_q_and_missing_or_duplicate_identities():
    records = observations()
    with pytest.raises(ValueError):
        D.select_recipe(records[:-1])
    with pytest.raises(ValueError):
        D.select_recipe(records + [records[-1]])
    records[-1]["role"] = "Q"
    with pytest.raises(ValueError):
        D.select_recipe(records)
