"""Run the explicit, data-free CPU test subset; never download model weights."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "test_analysis.py",
    "test_cpu_demo.py",
    "test_deterministic_execution.py",
    "test_engine_config.py",
    "test_experiment.py",
    "test_export_proposer_targets.py",
    "test_external_lessons.py",
    "test_hindsight_analysis.py",
    "test_hindsight_teacher.py",
    "test_model.py",
    "test_overnight_dose.py",
    "test_pi_comparison.py",
    "test_preflight_gpu.py",
    "test_prepare_model_sources.py",
    "test_proposer_training.py",
    "test_qualify_proposer.py",
    "test_qwen35.py",
    "test_quickstart.py",
    "test_representation.py",
    "test_representation_diagnostic.py",
    "test_shared_learning.py",
    "test_teacher_controls.py",
    "test_update_dose.py",
)
EXPRESSION = "not actual_cohort_admits and not manifest_cohort_admission"
RUN_TESTS = """
import pathlib
import sys
import hindsight

root = pathlib.Path(sys.argv[1]).resolve()
package = pathlib.Path(hindsight.__file__).resolve()
assert package.is_relative_to(root / "src"), (
    f"Refusing out-of-checkout hindsight import: {package}; expected {root / 'src'}"
)
print(f"Verified clone-local hindsight import: {package}", flush=True)
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
"""


def command():
    paths = [ROOT / "tests" / name for name in TESTS]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Incomplete source checkout: a declared CPU test file is missing")
    return [
        sys.executable,
        "-c",
        RUN_TESTS,
        str(ROOT),
        "-q",
        *map(str, paths),
        "-k",
        EXPRESSION,
    ]


def test_environment():
    # Never inherit an out-of-tree source prefix or trust an editable install.
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="Print the exact suite without running it"
    )
    args = parser.parse_args()
    invocation = command()
    if args.list:
        print("\n".join(invocation))
        return 0
    # An accidental pretrained construction must fail instead of accessing Hub.
    environment = test_environment()
    print(
        "Data-free CPU subset; corpus-dependent and historical replay tests are excluded.",
        flush=True,
    )
    return subprocess.run(invocation, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
