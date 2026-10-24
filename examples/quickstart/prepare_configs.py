"""Emit bounded integration-demo configs; does not run a model or change a dataset."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from hindsight.cohorts import load_cohort
from hindsight.experiment import write_json

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced(episodes, partition, per_family):
    selected = []
    for family in sorted({ep.family for ep in episodes if ep.partition == partition}):
        group = [ep for ep in episodes if ep.partition == partition and ep.family == family]
        if len(group) < per_family:
            raise ValueError("Insufficient predetermined sources in a family")
        selected.extend(ep.episode_id for ep in group[:per_family])
    if len(selected) != 4 * per_family:
        raise ValueError("The demo requires the four-family overnight cohort")
    return selected


def prepare(data, output, deadline, max_seconds, qualification=None):
    cutoff = datetime.fromisoformat(deadline)
    if cutoff.tzinfo is None or cutoff <= datetime.now(timezone.utc) or not 0 < max_seconds <= 7200:
        raise ValueError(
            "Set a future timezone-aware deadline and a 1..7200 second invocation bound"
        )
    episodes, _ = load_cohort(data)
    train = balanced(episodes, "meta_train", 1)
    evaluation = balanced(episodes, "development", 2)
    all_dev = [ep.episode_id for ep in episodes if ep.partition == "development"]
    if len(all_dev) != 16:
        raise ValueError("PI comparison requires the complete 16-source development partition")
    base = {
        "protocol_version": "quickstart-integration-only-v1",
        "model_name": MODEL,
        "revision": REVISION,
        "enable_thinking": False,
        "deterministic_execution": True,
        "seed": 7301,
        "source_tokens": 128,
        "pi_tokens": 160,
        "evaluation_tokens": 160,
        "student_steps": 1,
        "learning_rate": 1e-4,
        "generation_batch_size": 1,
        "max_context_tokens": 1536,
        "max_teacher_context_tokens": 4096,
        "absolute_deadline_utc": deadline,
        "max_invocation_seconds": max_seconds,
        "dataset_manifest_sha256": sha(data / "manifest.json"),
        "purpose": "Integration demonstration, not the 16-step/64-teacher-step research matrix",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "throughput.json", base)
    if qualification is None:
        return {"stage": "qualification_config_only", "output": str(output)}
    receipt = json.loads(qualification.read_text())
    size = receipt["selected_batch_size"]
    if (
        receipt["status"] != "completed"
        or size not in (4, 8)
        or not receipt["repeat_stability"].get(str(size), False)
        or not receipt["repeat_stability"].get("4", False)
        or receipt["dataset_sha256"] != base["dataset_manifest_sha256"]
        or any(receipt["config"][key] != base[key] for key in ("model_name", "revision"))
    ):
        raise ValueError("Qualification does not match the pinned model/corpus or stable batch")
    base.update(
        generation_batch_size=size,
        batch_qualification={"path": str(qualification.resolve()), "sha256": sha(qualification)},
    )
    write_json(
        output / "proposer.json",
        {
            **base,
            "train_episode_ids": train,
            "evaluation_episode_ids": evaluation,
            "evaluation_partition": "development",
            "evaluation_roles": ["V", "Q"],
            "proposer_k": 4,
            "reward_tasks": 4,
            "proposer_temperature": 0.7,
            "proposer_learning_rate": 1e-5,
            "checkpoint_every": 4,
        },
    )
    write_json(
        output / "pi-comparison.json",
        {**base, "partition": "development", "expected_sources": 16, "candidate_count": 4},
    )
    recipe = {
        "student_steps": 1,
        "learning_rate": 1e-4,
        "teacher_steps": 2,
        "teacher_batch_size": 4,
        "loss": "forward_kl",
    }
    write_json(
        output / "teacher-controls.json",
        {
            **base,
            "seeds": [7301],
            "training_episode_ids": train,
            "evaluation_episode_ids": all_dev,
            "evaluation_partition": "development",
            "recipe": {**recipe, "teacher_learning_rate": 1e-4},
        },
    )
    write_json(
        output / "shared.json",
        {**base, "seeds": [7301, 7302], "evaluation_episode_ids": evaluation, "recipe": recipe},
    )
    return {
        "stage": "integration_configs",
        "train_sources": len(train),
        "proposer_eval_sources": len(evaluation),
        "pi_teacher_eval_sources": len(all_dev),
        "shared_eval_sources": len(evaluation),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.data, args.output, args.deadline_utc, args.max_seconds, args.qualification)
        )
    )


if __name__ == "__main__":
    main()
