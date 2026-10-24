"""Measure matched batch scheduling without assuming singleton output equivalence.

Configured execution bounds are cooperative callback checks, not a hard process
interrupt or an independent GPU/provider billing shutdown mechanism.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hindsight.cohorts import load_cohort
from hindsight.data import grade
from hindsight.experiment import append_json, write_json
from hindsight.model import HFEngine, configure_deterministic_execution


def run(config_path, data, output):
    config = json.loads(config_path.read_text())
    seconds = config.get("max_invocation_seconds", 600)
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("max_invocation_seconds must be a positive finite number")
    try:
        deadline = datetime.fromisoformat(config["absolute_deadline_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Require a timezone-aware absolute_deadline_utc") from error
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("Require a timezone-aware absolute_deadline_utc")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def check():
        if datetime.now(timezone.utc) >= deadline or time.monotonic() - started >= seconds:
            raise TimeoutError("Throughput qualification deadline")

    configure_deterministic_execution(True)
    episodes, _ = load_cohort(data)
    tasks = [ep.validation[0] for ep in episodes if ep.partition == "development"]
    if len(tasks) != 16:
        raise ValueError("Require16 outcome-blind development tasks")
    receipt = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "planned_generations": 80,
        "max_invocation_seconds": seconds,
        "execution_bound": "Cooperative callback; not independent provider billing shutdown",
        "config": config,
        "dataset_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
        "task_ids": [t.task_id for t in tasks],
        "runs": [],
    }
    write_json(output / "receipt.json", receipt)
    engine = None
    try:
        check()
        engine = HFEngine(
            model_name=config["model_name"],
            revision=config["revision"],
            device="cuda",
            enable_thinking=False,
            budget_check=check,
        )
        layouts = {}
        for batch_size, repeat in [(1, 0), (4, 0), (4, 1), (8, 0), (8, 1)]:
            check()
            records = []
            torch.cuda.synchronize()
            start = time.monotonic()
            for offset in range(0, len(tasks), batch_size):
                check()
                group = tasks[offset : offset + batch_size]
                generations = engine.generate([t.prompt for t in group], 160, False, 7301)
                for task, generation in zip(group, generations, strict=True):
                    row = {
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "task_id": task.task_id,
                        "generation": dataclasses.asdict(generation),
                        "grade": grade(generation.text, task),
                    }
                    append_json(output / "observations.jsonl", row)
                    records.append(row)
            torch.cuda.synchronize()
            check()
            elapsed = time.monotonic() - start
            layouts[(batch_size, repeat)] = records
            receipt["runs"].append(
                {
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "sequences_per_second": 16 / elapsed,
                    "correct": sum(r["grade"]["success"] for r in records),
                    "cap_hits": sum(r["generation"]["cap_hit"] for r in records),
                }
            )
            receipt["work"] = dict(engine.counters)
            write_json(output / "receipt.json", receipt)
        stability = {}
        drift = {}
        for size in (4, 8):

            def key(row):
                return row["generation"]["response_ids"]

            stability[size] = all(
                key(a) == key(b)
                for a, b in zip(layouts[(size, 0)], layouts[(size, 1)], strict=True)
            )
            drift[size] = sum(
                key(a) != key(b) for a, b in zip(layouts[(1, 0)], layouts[(size, 0)], strict=True)
            )
        stable = [size for size in (4, 8) if stability[size]]
        if not stable:
            raise ValueError("Neither batched schedule reproduces its own token outputs")
        rates = {
            size: sum(r["seconds"] for r in receipt["runs"] if r["batch_size"] == size) / 2
            for size in stable
        }
        selected = min(rates, key=rates.get)
        receipt.update(
            status="completed",
            repeat_stability=stability,
            singleton_token_differences=drift,
            selected_batch_size=selected,
            selection="Fastest repeat-stable schedule; not selected by task accuracy",
            interpretation="New matched schedule, not equivalence to historical singleton runs",
        )
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        receipt.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            work=dict(engine.counters) if engine else {},
        )
        write_json(output / "receipt.json", receipt)
    print(receipt["status"])
    return receipt["status"] == "completed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.config, args.data, args.output) else 1)
