"""Bounded, source-only check of proposer framing before restarting development."""

import argparse
import dataclasses
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from hindsight.data import load_corpus
from hindsight.experiment import proposal_prompt, read_json, seed_for, write_json
from hindsight.model import HFEngine


def grid_runs(grid: list[int]) -> list[tuple[int, int, int]]:
    """Lossless (value, zero-based start, length), including every zero run."""
    if (not isinstance(grid, list) or not grid
            or any(type(value) is not int or not 0 <= value <= 9 for value in grid)):
        raise ValueError("Expected a nonempty integer grid with values 0 through 9")
    runs = []
    for start, value in enumerate(grid):
        if runs and runs[-1][0] == value:
            previous, offset, length = runs[-1]
            runs[-1] = (previous, offset, length + 1)
        else:
            runs.append((value, start, 1))
    return runs


def indexed_grid_view(saved_task_prompt: str) -> str:
    """Read only labeled public grids; never infer or supply a query output."""
    grids = {}
    for line in saved_task_prompt.splitlines():
        if line.startswith("Query output:"):
            raise ValueError("A public source prompt must not contain Query output")
        match = re.fullmatch(r"(Example [1-9]\d* (?:input|output)|Query input):\s*(.+)", line)
        if match:
            label, encoded = match.groups()
            if label in grids:
                raise ValueError(f"Duplicate public grid label: {label}")
            grids[label] = grid_runs(json.loads(encoded))
    if "Query input" not in grids or not any(label.startswith("Example ") for label in grids):
        raise ValueError("Expected labeled demonstration grids and Query input")
    return "\n".join([
        "<source_derived_indexed_grids>",
        "Auxiliary data copied losslessly from the public TASK grids above.",
        "Each run is (value, zero-based start, length); zero runs are included.",
        "This view supplies no inferred rule or query output.",
        *(f"{label} runs: {runs}" for label, runs in grids.items()),
        "</source_derived_indexed_grids>",
    ])


def add_indexed_grids(prompt: str, saved_task_prompt: str) -> str:
    """Keep auxiliary data inside experience and the current lesson instruction last."""
    closing = "</recorded_experience>"
    if prompt.count(closing) != 1:
        raise ValueError("Expected exactly one recorded-experience boundary")
    return prompt.replace(closing, indexed_grid_view(saved_task_prompt) + "\n" + closing, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deliberate", action="store_true",
        help="Source-only diagnostic: allow 512 tokens of analysis and a LESSON line",
    )
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--token-limit", type=int)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--final-attempt-only", action="store_true")
    parser.add_argument(
        "--indexed-grids", action="store_true",
        help="Add lossless indexed runs of saved public source grids to diagnostic context",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostic receipts; use a new output path")
    identity = read_json(args.sources / "identity.json")
    config = identity["config"]
    if bool(args.model_name) != bool(args.revision):
        raise ValueError("A diagnostic proposer override needs both model name and pinned revision")
    episodes, _ = load_corpus(args.data)
    chosen, families = [], set()
    for episode in episodes:
        if episode.partition == "development" and episode.family not in families:
            chosen.append(episode)
            families.add(episode.family)
        if len(chosen) == 4:
            break
    if len(chosen) != 4 or config["candidate_count"] != 3:
        raise ValueError("Diagnostic is exactly four distinct development families, K=3")
    receipt = {
        "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
        "selection": "first development episode from each of first four distinct families",
        "episode_ids": [ep.episode_id for ep in chosen], "generation_limit": 12,
        "deliberate": args.deliberate,
        "indexed_grids": args.indexed_grids,
        "enable_thinking": False if args.no_thinking else None,
        "final_attempt_only": args.final_attempt_only,
        "generation_token_limit": (
            args.token_limit or (512 if args.deliberate else config["pi_tokens"])
        ),
        "model_name": args.model_name or config["model_name"],
        "revision": args.revision or config["revision"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_identity_sha256": hashlib.sha256(
            (args.sources / "identity.json").read_bytes()
        ).hexdigest(),
        "rows": [],
    }
    write_json(args.output, receipt)

    def deadline():
        if datetime.now(timezone.utc) >= datetime.fromisoformat(config["absolute_deadline_utc"]):
            raise TimeoutError("Absolute research deadline")

    try:
        engine = HFEngine(
            model_name=receipt["model_name"], revision=receipt["revision"],
            max_generation_context=config["max_context_tokens"],
            max_teacher_context=config["max_teacher_context_tokens"], budget_check=deadline,
            enable_thinking=receipt["enable_thinking"],
        )
        for ep in chosen:
            source_path = args.sources / "sources" / f"{ep.episode_id}.json"
            source = read_json(source_path)
            if (source["task_id"] != ep.source.task_id
                    or source["generation"]["prompt"] != ep.source.prompt):
                raise ValueError("Saved development source does not match the admitted task")
            attempt = source["generation"]["text"]
            if args.final_attempt_only:
                finals = [line.strip() for line in attempt.splitlines()
                          if line.strip().startswith("FINAL:")]
                if len(finals) != 1:
                    raise ValueError("Final-only diagnostic needs one original FINAL line")
                attempt = "Original explanation omitted for this diagnostic.\n" + finals[0]
            prompt = proposal_prompt(ep.source.prompt, attempt, source["feedback"])
            if args.deliberate:
                prompt = prompt.split("Write only one or two short sentences", 1)[0] + (
                    "First use at most 200 words to compare the input/output demonstrations "
                    "and test the attempted explanation against them. Then end with exactly "
                    "one line beginning LESSON: containing the concrete transferable rule "
                    "in at most 40 words. Do not solve the query or output a FINAL field."
                )
            if args.indexed_grids:
                prompt = add_indexed_grids(prompt, source["generation"]["prompt"])
            for candidate in range(3):
                value = engine.generate(
                    [prompt], max_new_tokens=receipt["generation_token_limit"], sample=True,
                    seed=seed_for(config["seed"], ep.episode_id, "pi", candidate),
                    context_limit=config["max_teacher_context_tokens"],
                )[0]
                lesson = value.text.strip()
                if args.deliberate:
                    answer = value.text.rsplit("</think>", 1)[-1]
                    matches = re.findall(r"(?m)^LESSON:\s*(.+)$", answer)
                    lesson = matches[0].strip() if len(matches) == 1 else ""
                gate = (bool(lesson) and not value.cap_hit
                        and not re.search(r"\bFINAL\s*:|\[\s*\d", lesson))
                receipt["rows"].append({
                    "episode_id": ep.episode_id, "candidate": candidate,
                    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "generation": dataclasses.asdict(value), "lesson_shape_pass": bool(gate),
                    "lesson": lesson,
                })
                write_json(args.output, receipt)
        receipt["status"] = (
            "passed_framing" if all(row["lesson_shape_pass"] for row in receipt["rows"])
            else "failed_framing"
        )
        receipt["work"] = engine.counters
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "passed_framing" else 1


if __name__ == "__main__":
    raise SystemExit(main())
