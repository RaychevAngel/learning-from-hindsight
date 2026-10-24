"""Compare saved raw-grid calibration with lossless RLE on the same sixteen tasks."""

import argparse
import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from hindsight import representation
from hindsight.data import grade, load_corpus
from hindsight.experiment import digest, read_json, rows, seed_for, write_json
from hindsight.model import MODEL_NAME, MODEL_REVISION, HFEngine

WORK = {"generation_sequences": 16, "scoring_forwards": 0,
        "student_optimizer_steps": 0, "teacher_optimizer_steps": 0,
        "teacher_example_exposures": 0}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(baseline: Path, data: Path, output: Path, *, engine=None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=False)
    receipt = {
        "protocol": "rle-calibration-v1", "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(), "planned_work": WORK,
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "baseline_directory": str(baseline.resolve()), "data_directory": str(data.resolve()),
        "baseline_generation_ids": "not persisted by original calibration; unavailable",
        "selection": "none; all original sixteen calibration tasks in original order",
        "expected": [], "rows": [],
    }
    write_json(output, receipt)
    started = time.monotonic()
    try:
        identity = read_json(baseline / "identity.json")
        config = identity["config"]
        if (config["model_name"] != MODEL_NAME or config["revision"] != MODEL_REVISION
                or config["source_tokens"] != 128):
            raise ValueError("Require the pinned original 4B model and 128-token calibration cap")
        if identity["dataset_manifest"] != read_json(data / "manifest.json"):
            raise ValueError("Calibration corpus differs from the original baseline")
        _, calibration = load_corpus(data)
        planned = [{"task_id": task.task_id, "family": task.family} for task in calibration]
        original = rows(baseline / "calibration.jsonl")
        if (len(planned) != 16 or len({row["task_id"] for row in planned}) != 16
                or read_json(baseline / "calibration-expected.json") != planned
                or len(original) != 16 or [row["task_id"] for row in original]
                != [row["task_id"] for row in planned]):
            raise ValueError("Require all sixteen original calibration IDs, in original order")
        receipt.update(
            config=config, expected=planned,
            sampling={"sample": True, "max_new_tokens": 128, "adapter": None,
                      "batch_size": 1, "seed_rule": "seed_for(seed, task_id, 'calibration')",
                      "other_generation_settings": "unchanged HFEngine defaults"},
            baseline_hashes={name: sha256((baseline / name).read_bytes()) for name in
                             ("identity.json", "calibration-expected.json", "calibration.jsonl")},
            manifest_sha256=sha256((data / "manifest.json").read_bytes()),
            core_source_hashes={path.name: sha256(path.read_bytes()) for path in
                                sorted(Path(representation.__file__).parent.glob("*.py"))},
            representation_sha256=sha256(Path(representation.__file__).read_bytes()),
            representation_version=representation.REPRESENTATION_VERSION,
        )
        prepared = []
        for task, raw in zip(calibration, original, strict=True):
            raw_grade = grade(raw["text"], task)
            if raw["family"] != task.family or any(raw[key] != value
                                                   for key, value in raw_grade.items()):
                raise ValueError("Saved raw-grid grade or family does not match the fixed oracle")
            prompt, length = representation.render_rle_prompt(task.prompt)
            if length != len(task.answer):
                raise ValueError("Public query length differs from the unchanged oracle length")
            prepared.append((task, prompt, length, raw, raw_grade))
        write_json(output, receipt)
        absolute = datetime.fromisoformat(config["absolute_deadline_utc"])
        if absolute.tzinfo is None:
            raise ValueError("Deadline must include timezone")

        def check():
            if (datetime.now(timezone.utc) >= absolute
                    or time.monotonic() - started >= config["max_invocation_seconds"]):
                raise TimeoutError("Representation calibration execution deadline")

        check()
        if engine is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"], revision=config["revision"], local_files_only=True,
            )
        else:
            tokenizer = engine.tokenizer
        # Oracle tokens are used only for this calibration capacity check, never model context.
        receipt["token_preflight"] = []
        for task, prompt, _, _, _ in prepared:
            prompt_tokens = len(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
            ))
            encoded = representation.encode_rle(list(task.answer))
            lengths = [len(tokenizer.encode(
                "Rule.\nFINAL: " + json.dumps(encoded, separators=separators),
                add_special_tokens=False,
            )) + 1 for separators in ((",", ":"), (", ", ": "))]
            receipt["token_preflight"].append({
                "task_id": task.task_id, "prompt_tokens": prompt_tokens,
                "reference_output_tokens_with_eos": lengths,
                "fits_context": prompt_tokens + 128 <= config["max_context_tokens"],
                "canonical_reference_fits_cap": min(lengths) <= 128,
            })
        receipt["required_global_reference_token_floor"] = max(
            min(row["reference_output_tokens_with_eos"]) for row in receipt["token_preflight"]
        )
        receipt["minimum_global_context_for_current_cap"] = max(
            row["prompt_tokens"] + 128 for row in receipt["token_preflight"]
        )
        receipt["reference_token_floor_definition"] = (
            "Shortest of compact/spaced canonical RLE, prefixed by Rule. and FINAL:, plus EOS; "
            "not a guarantee for longer explanations or a proof over all possible encodings."
        )
        write_json(output, receipt)
        if any(not row["fits_context"] or not row["canonical_reference_fits_cap"]
               for row in receipt["token_preflight"]):
            raise ValueError("RLE context/reference exceeds fixed token cap; no generation")
        check()
        if engine is None:
            engine = HFEngine(
                model_name=config["model_name"], revision=config["revision"],
                max_generation_context=config["max_context_tokens"],
                max_teacher_context=config["max_teacher_context_tokens"], budget_check=check,
                tokenizer=tokenizer,
            )
        for task, prompt, length, raw, raw_grade in prepared:
            check()
            seed = seed_for(config["seed"], task.task_id, "calibration")
            generated = engine.generate([prompt], max_new_tokens=128, sample=True,
                                        seed=seed, adapter=None)
            if len(generated) != 1:
                raise ValueError("Calibration must return exactly one generation per task")
            value = generated[0]
            result = representation.grade_rle(value.text, task, length)
            receipt["rows"].append({
                "task_id": task.task_id, "family": task.family, "seed": seed,
                "original_prompt": task.prompt, "rle_prompt": prompt,
                "original_prompt_sha256": sha256(task.prompt.encode()),
                "rle_prompt_sha256": sha256(prompt.encode()),
                "rle_prompt_ids_sha256": digest(value.prompt_ids),
                "unchanged_oracle_sha256": digest(task.answer), "query_length": length,
                "baseline": raw, "baseline_grade": raw_grade,
                "generation": dataclasses.asdict(value), "rle_grade": result,
                "decoded_answer": result["prediction"],
                "cap_hit": value.cap_hit, "generated_tokens": len(value.response_ids),
            })
            receipt["work"] = dict(engine.counters)
            write_json(output, receipt)
            check()
        if engine.counters != WORK or len(receipt["rows"]) != 16:
            raise ValueError("Actual work differs from sixteen generations and zero updates")
        receipt["summary"] = {
            "raw_successes": sum(row["baseline_grade"]["success"] for row in receipt["rows"]),
            "rle_successes": sum(row["rle_grade"]["success"] for row in receipt["rows"]),
            "raw_format_failures": sum(not row["baseline_grade"]["format_ok"]
                                       for row in receipt["rows"]),
            "rle_format_failures": sum(not row["rle_grade"]["format_ok"]
                                       for row in receipt["rows"]),
            "raw_cap_hits": sum(row["baseline"]["cap_hit"] for row in receipt["rows"]),
            "rle_cap_hits": sum(row["cap_hit"] for row in receipt["rows"]),
        }
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["work"] = dict(engine.counters) if engine is not None else dict.fromkeys(WORK, 0)
        receipt["actual_observations"] = len(receipt["rows"])
        receipt["missing_observations"] = 16 - len(receipt["rows"])
        write_json(output, receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    return run(args.baseline, args.data, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
