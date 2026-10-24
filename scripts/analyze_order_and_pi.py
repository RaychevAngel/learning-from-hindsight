"""Offline terminal regrade and paired comparison for diagnostic 024."""

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from hindsight.analysis import paired_interval
from hindsight.data import grade, load_corpus
from hindsight.model import with_pi


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    result = root / "results/order-and-pi-qwen35-024"
    reference = root / "results/hindsight-teacher-qwen35-023"
    receipt = json.loads((result / "receipt.json").read_text())
    new, old = rows(result / "generations.jsonl"), rows(reference / "observations.jsonl")
    episodes, _ = load_corpus(root / "data/balanced-transfer-v1")
    expected_work = dict(generation_sequences=224, scoring_forwards=2,
                         student_optimizer_steps=8, teacher_optimizer_steps=0,
                         teacher_example_exposures=0)
    assert receipt["status"] == "completed" and receipt["actual_records"] == len(new) == 224
    assert receipt["missing_records"] == 0 and receipt["work"] == expected_work
    assert all(receipt[key] for key in ("order_gate", "current_order_invariance",
                                      "history_reproduction", "frozen_inputs_unchanged"))
    for remote, value in receipt["frozen_files"].items():
        assert sha(root / Path(remote).relative_to("/opt/scalable-hindsight-024")) == value
    lessons = json.loads((result / "lessons.json").read_text())
    # Same lesson structure used by the frozen driver admission.
    import check_order_and_pi as driver
    _, prepared, _, _ = driver.admit(
        root / "results/sources-balanced-qwen35-022", reference,
        root / "results/frontier-lessons-balanced-qwen35-023.json",
        root / "data/balanced-transfer-v1")
    lesson_by_id = {ep.episode_id: lesson for ep, _, _, lesson in prepared}
    assert lessons == json.loads((reference / "lessons.json").read_text())
    tasks = {(ep.episode_id, role, task.task_id): task for ep in episodes
             for role, members in (("V", ep.validation), ("Q", ep.queries)) for task in members}

    def key(row):
        return row["episode_id"], row["role"], row["task_id"]

    planned = []
    for episode_id in ("208831d7e726ae9462d2bb3a", "eb2e08ce9a30ca1da12fc3f8"):
        ep = next(ep for ep in episodes if ep.episode_id == episode_id)
        v = [(ep.episode_id, "V", t.task_id) for t in ep.validation]
        q = [(ep.episode_id, "Q", t.task_id) for t in ep.queries]
        for layout, order in (("original", v + q), ("reverse", (v + q)[::-1]),
                              ("interleaved", [k for pair in zip(v, q) for k in pair])):
            planned.extend((layout, "adapted_pi", *k) for k in order)
    planned.extend(("direct_pi", "direct_pi", *k) for k in tasks)
    assert [(r["layout"], r["condition"], *key(r)) for r in new] == planned
    by_layout = defaultdict(dict)
    for i, row in enumerate(new):
        assert row["sequence"] == i and key(row) not in by_layout[row["layout"]]
        task, generation = tasks[key(row)], row["generation"]
        assert grade(generation["text"], task) == row["grade"]
        assert generation["prompt"] == (with_pi(task.prompt, lesson_by_id[row["episode_id"]])
                                         if row["layout"] == "direct_pi" else task.prompt)
        assert len(generation["response_ids"]) == row["generated_tokens"] <= 160
        assert all(isinstance(x, int) and x >= 0 for field in ("prompt_ids", "response_ids")
                   for x in generation[field])
        by_layout[row["layout"]][key(row)] = row
    assert len(by_layout["direct_pi"]) == 128
    for a, b in itertools.combinations(("original", "reverse", "interleaved"), 2):
        assert len(by_layout[a]) == len(by_layout[b]) == 32
        for k, row in by_layout[a].items():
            assert row["generation"] == by_layout[b][k]["generation"]
            assert row["grade"] == by_layout[b][k]["grade"]
    old_by = {(r["condition"], *key(r)): r for r in old}
    for r in old:
        assert all(r[k] == v for k, v in grade(r["text"], tasks[key(r)]).items())
    for layout in ("original", "reverse", "interleaved"):
        for k, row in by_layout[layout].items():
            history = old_by[("adapted_pi", *k)]
            assert all(history[field] == row["generation"][field] for field in ("text", "cap_hit"))
            assert all(history[field] == value for field, value in row["grade"].items())
            assert history["generated_tokens"] == row["generated_tokens"]
    states = rows(result / "student-state.jsonl")
    assert len(states) == 6 and all(s["unchanged"] for s in states)
    for ep_id in receipt["order_source_ids"]:
        values = [s for s in states if s["episode_id"] == ep_id]
        assert len({v[f] for v in values for f in ("before_sha256", "after_sha256")}) == 1
    assert receipt["base_before_order_sha256"] == receipt["base_after_order_sha256"]
    events = rows(result / "events.jsonl")
    updates = [e for e in events if e["event"] == "student_adapted"]
    assert len(updates) == 2 and sum(e["event"] == "teacher_scored" for e in events) == 2
    old_updates = {e["episode_id"]: e for e in rows(reference / "events.jsonl")
                   if e["event"] == "student_adapted" and e["condition"] == "adapted_pi"}
    for update in updates:
        assert len(update["metrics"]) == 4
        for actual, previous in zip(update["metrics"], old_updates[update["episode_id"]]["metrics"],
                                    strict=True):
            assert all(math.isfinite(v) for v in actual.values())
            assert all(actual[k] == previous[k] for k in actual if k != "elapsed_seconds")
    counts, family_counts, paired = [], defaultdict(lambda: defaultdict(int)), {}
    for ep in episodes:
        values = dict(episode_id=ep.episode_id, family=ep.family)
        for role in ("V", "Q"):
            for condition in ("baseline", "adapted_pi", "direct_pi"):
                score = sum((by_layout["direct_pi"][k]["grade"]["success"]
                             if condition == "direct_pi" else old_by[(condition, *k)]["success"])
                            for k in tasks if k[:2] == (ep.episode_id, role))
                values[f"{role}/{condition}"] = score
                family_counts[ep.family][f"{role}/{condition}"] += score
        counts.append(values)
    for role in ("V", "Q"):
        for control in ("baseline", "adapted_pi"):
            contrasts = []
            for ep in episodes:
                pairs = [(row["grade"]["success"], old_by[(control, *k)]["success"])
                         for k, row in by_layout["direct_pi"].items()
                         if k[:2] == (ep.episode_id, role)]
                gains = sum(a and not b for a, b in pairs)
                losses = sum(b and not a for a, b in pairs)
                contrasts.append(dict(episode_id=ep.episode_id, gains=gains, losses=losses,
                                      difference=(gains - losses) / 8))
            paired[f"{role}/direct_pi_minus_{control}"] = dict(
                gains=sum(x["gains"] for x in contrasts),
                losses=sum(x["losses"] for x in contrasts),
                per_source=contrasts, **paired_interval([x["difference"] for x in contrasts],
                                                       bootstrap_samples=10000, seed=7301))
    return dict(
        status="passed_offline_terminal_audit", work=expected_work,
        per_source=counts, per_family=dict(family_counts), paired=paired,
        totals={k: sum(r[k] for r in counts) for k in counts[0] if "/" in k},
        malformed=sum(not r["grade"]["format_ok"] for r in new),
        direct_pi_malformed=sum(not r["grade"]["format_ok"] for r in new
                                if r["layout"] == "direct_pi"),
        cap_hits=sum(r["generation"]["cap_hit"] for r in new),
        hashes={name: sha(result / name) for name in (
            "receipt.json", "generations.jsonl", "student-state.jsonl", "mismatch-matrix.json")},
        limitations=["New direct-PI outcomes; baseline/adapted controls reused from023",
                     "Current native IDs compared; historical evaluation IDs not saved",
                     "Recorded state digests checked; model tensors not independently rehashed",
                     "Eight development sources; no final holdout or confirmatory claim"])


if __name__ == "__main__":
    print(json.dumps(analyze(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
