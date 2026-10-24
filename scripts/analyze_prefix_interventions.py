"""CPU-only gold audit of already frozen025/029 source-prefix distributions."""

import argparse
import dataclasses
import importlib.util
import json
import re
from pathlib import Path
from statistics import mean

import torch

from hindsight.data import grade, load_corpus
from hindsight.experiment import digest, read_json, write_json


def helper(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFIX = helper(Path(__file__).with_name("score_prefix_interventions.py"))
AUDIT025 = helper(Path(__file__).resolve().parents[1] / "paper/audit_source_fidelity025.py")
sha, require = PREFIX.sha, PREFIX.require
ARMS = ("old_PI", "B", "C")


def native_values(displays, ids, prediction, gold):
    """Map numeric character spans to saved native tokens, never re-tokenize a value."""
    require(len(displays) == len(ids), "Missing saved native token displays")
    pieces = [display.replace("Ġ", " ").replace("Ċ", "\n") for display in displays]
    text = "".join(pieces)
    matches = list(re.finditer(r"FINAL:\s*(\[[^\]\r\n]*\])", text))
    require(len(matches) == 1, "Cannot locate one native FINAL array")
    match = matches[0]
    require(
        json.loads(match.group(1)) == prediction and len(prediction) == len(gold),
        "Saved native value displays disagree with the regraded original answer",
    )
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    final_start = next(i for i, (_, end) in enumerate(offsets) if end > match.start())
    spans = []
    for index, number in enumerate(re.finditer(r"[0-9]+", match.group(1))):
        start, end = match.start(1) + number.start(), match.start(1) + number.end()
        positions = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
        observed = int(number.group())
        require(observed == prediction[index], "Native value indexing differs")
        single = (
            len(positions) == 1
            and len(number.group()) == 1
            and pieces[positions[0]].strip() == number.group()
        )
        spans.append(
            dict(
                array_index=index,
                observed_value=observed,
                correct_value_offline=gold[index],
                native_positions=positions,
                single_scalar_token=single,
                alignment_kind="single_scalar_token"
                if single
                else "multi_token_value"
                if len(positions) > 1
                else "compound_or_multidigit_token",
            )
        )
    require(len(spans) == len(prediction), "Native array value count differs")
    identities = {}
    for row in spans:
        if row["single_scalar_token"]:
            position = row["native_positions"][0]
            pattern = pieces[position].replace(str(row["observed_value"]), "{value}")
            key = (pattern, row["observed_value"])
            require(
                key not in identities or identities[key] == ids[position],
                "Ambiguous native scalar token identity",
            )
            identities[key] = ids[position]
    wrong = [
        row["array_index"] for row in spans if row["observed_value"] != row["correct_value_offline"]
    ]
    for row in spans:
        position = row["native_positions"][0]
        pattern = pieces[position].replace(str(row["observed_value"]), "{value}")
        correct_id = (
            identities.get((pattern, row["correct_value_offline"]))
            if row["single_scalar_token"]
            else None
        )
        row.update(
            correct_token_id=correct_id,
            scalar_probability_identified=correct_id is not None,
            first_wrong_array_value=bool(wrong) and row["array_index"] == wrong[0],
            earlier_array_values_correct=not any(index < row["array_index"] for index in wrong),
            native_displays=[displays[i] for i in row["native_positions"]],
            regions=[AUDIT025.region(displays[i], i, final_start) for i in row["native_positions"]],
        )
    return spans


def policy_metrics(policies, ids, spans):
    """Full-vocabulary float64 distances; scalar probability only on identified scalar tokens."""
    results = []
    for span in spans:
        entries = []
        for position in span["native_positions"]:
            old = policies["old_PI"][position].double()
            q = old.exp()
            arms = {}
            for arm in ARMS:
                target = policies[arm][position].double()
                p = target.exp()
                argmax = int(target.argmax())
                correct_id = span["correct_token_id"]
                arms[arm] = dict(
                    observed_token_probability=float(p[ids[position]]),
                    correct_scalar_probability=float(p[correct_id])
                    if correct_id is not None
                    else None,
                    argmax_token_id=argmax,
                    argmax_is_observed=argmax == ids[position],
                    argmax_is_correct_scalar=argmax == correct_id
                    if correct_id is not None
                    else None,
                    forward_KL_old_PI_to_arm=float((q * (old - target)).sum()),
                    reverse_KL_arm_to_old_PI=float((p * (target - old)).sum()),
                    total_variation_to_old_PI=float((p - q).abs().sum() / 2),
                )
            entries.append(
                dict(response_position=position, observed_token_id=ids[position], arms=arms)
            )
        results.append(dict(span, positions=entries))
    return results


def summarize(values):
    positions = [position for value in values for position in value["positions"]]
    result = dict(
        array_values=len(values),
        native_positions=len(positions),
        identified_scalar_values=sum(value["scalar_probability_identified"] for value in values),
        arms={},
    )
    for arm in ARMS:
        rows = [position["arms"][arm] for position in positions]
        scalars = [row for row in rows if row["correct_scalar_probability"] is not None]
        result["arms"][arm] = dict(
            observed_probability_mean=mean(row["observed_token_probability"] for row in rows)
            if rows
            else None,
            correct_scalar_probability_mean=mean(
                row["correct_scalar_probability"] for row in scalars
            )
            if scalars
            else None,
            correct_scalar_probability_denominator=len(scalars),
            argmax_correct_scalar=sum(row["argmax_is_correct_scalar"] for row in scalars),
            argmax_observed=sum(row["argmax_is_observed"] for row in rows),
            forward_KL_old_PI_to_arm_mean=mean(row["forward_KL_old_PI_to_arm"] for row in rows)
            if rows
            else None,
            total_variation_to_old_PI_mean=mean(row["total_variation_to_old_PI"] for row in rows)
            if rows
            else None,
        )
    return result


def check_alignment(alignment, ids):
    tokens, positions = alignment["input_ids"], alignment["predecessor_positions"]
    require(
        positions == list(range(len(tokens) - len(ids) - 1, len(tokens) - 1))
        and [tokens[position + 1] for position in positions] == ids
        and alignment["input_ids_sha256"] == digest(tokens),
        "Native action/predecessor alignment or token hash differs",
    )


def analyze(fidelity, framing, data, *, loader=load_corpus):
    torch.set_num_threads(2)
    # Admit teacher inputs before opening any source gold in this offline analysis.
    inherited, prepared, frozen = PREFIX.admit(fidelity, data)
    receipt, plan = read_json(framing / "receipt.json"), read_json(framing / "plan.json")
    ids = [item["episode_id"] for item in prepared]
    require(
        receipt["status"] == "completed"
        and receipt["frozen_inputs_unchanged"]
        and receipt["work"] == PREFIX.WORK
        and receipt["completed_episode_ids"] == ids
        and receipt["inherited_config"] == inherited
        and receipt["plan_sha256"] == sha(framing / "plan.json")
        and set(frozen.values()) <= set(receipt["frozen_files"].values())
        and plan["generic_frames"] == dict(B=PREFIX.FRAME_B, C=PREFIX.FRAME_C)
        and [row["episode_id"] for row in plan["sources"]] == ids,
        "Require exact completed029 from the pinned025 sources/config/frames",
    )
    paths = [framing / name for name in ("receipt.json", "plan.json")]
    paths += [Path(__file__), Path(AUDIT025.__file__)]
    frozen.update({str(path.resolve()): sha(path) for path in paths})
    contexts = []
    for item, entry in zip(prepared, plan["sources"], strict=True):
        expected = PREFIX.contexts(item["lesson"], item["generation"], item["feedback"])
        require(
            entry["task_id"] == item["task_id"]
            and entry["source_sha256"] == item["source_sha256"]
            and entry["old_tensor_sha256"] == item["old_tensor_sha256"]
            and entry["lesson"] == item["lesson"]
            and entry["lesson_sha256"] == digest(item["lesson"])
            and entry["contexts"] == expected,
            "Source/lesson/context plan lineage differs",
        )
        contexts.append(expected)
    episodes, _ = loader(data)
    by_id = {ep.episode_id: ep for ep in episodes}
    require(set(by_id) == set(ids), "Offline gold source membership differs")
    sources, vocabulary = [], None
    for item, expected in zip(prepared, contexts, strict=True):
        ep_id, generation = item["episode_id"], item["generation"]
        task = by_id[ep_id].source
        require(
            task.task_id == item["task_id"] and task.prompt == generation.prompt,
            "Offline source gold/public prompt join differs",
        )
        original = read_json(fidelity / "sources" / f"{ep_id}.json")
        regraded = grade(generation.text, task)
        require(
            regraded == original["grade"]
            and regraded["format_ok"]
            and item["feedback"]
            == f"The submitted answer was {'correct' if regraded['success'] else 'incorrect'}.",
            "Original source regrade/feedback differs",
        )
        path = framing / "distributions" / f"{ep_id}.pt"
        sidecar_path = path.with_suffix(".json")
        sidecar = read_json(sidecar_path)
        require(
            sidecar["episode_id"] == ep_id
            and sidecar["artifact"]["sha256"] == sha(path)
            and sidecar["artifact"]["bytes"] == path.stat().st_size,
            "029 target artifact checksum/size differs",
        )
        frozen.update({str(value.resolve()): sha(value) for value in (path, sidecar_path)})
        old = torch.load(item["old_tensor_path"], map_location="cpu", weights_only=True)
        new = torch.load(path, map_location="cpu", weights_only=True)
        require(
            old["prompt_ids"] == generation.prompt_ids
            and old["response_ids"] == generation.response_ids
            and new["format"] == "source-prefix-framing-v1"
            and new["episode_id"] == ep_id
            and new["generation"] == dataclasses.asdict(generation)
            and new["response_ids"] == generation.response_ids
            and new["old_tensor_sha256"] == item["old_tensor_sha256"]
            and new["contexts"] == expected
            and new["unchanged_lesson"] == item["lesson"]
            and new["feedback"] == item["feedback"],
            "Cached source/target identities differ",
        )
        for mode in ("teacher", "student"):
            check_alignment(old["alignment"][mode], generation.response_ids)
        for arm in ("B", "C"):
            check_alignment(new["alignment"][arm], generation.response_ids)
            require(
                new["alignment"][arm]["context_sha256"] == digest(expected[arm]),
                "Context hash differs",
            )
        policies = dict(old_PI=old["target"], B=new["B"], C=new["C"])
        vocabulary = vocabulary or old["target"].shape[1]
        for value in policies.values():
            require(
                value.dtype == torch.float32
                and value.shape == (len(generation.response_ids), vocabulary),
                "FP32 full-vocabulary target shape differs",
            )
            PREFIX.distribution(value, generation.response_ids)
        displays = old["alignment"]["response_token_displays"]
        require(new["response_token_displays"] == displays, "029 native displays differ from025")
        spans = native_values(
            displays, generation.response_ids, regraded["prediction"], list(task.answer)
        )
        values = policy_metrics(policies, generation.response_ids, spans)
        first = [value for value in values if value["first_wrong_array_value"]]
        require(
            len(first) == int(not regraded["success"]), "First-wrong source classification differs"
        )
        sources.append(
            dict(
                episode_id=ep_id,
                family=by_id[ep_id].family,
                source_grade=regraded,
                source_sha256=item["source_sha256"],
                old_tensor_sha256=item["old_tensor_sha256"],
                framing_tensor_sha256=sha(path),
                values=values,
                summary=summarize(values),
                first_wrong_array_value=first[0] if first else None,
            )
        )
        del old, new, policies
    require(all(sha(path) == checksum for path, checksum in frozen.items()), "Audit inputs changed")
    first = [
        source["first_wrong_array_value"] for source in sources if source["first_wrong_array_value"]
    ]
    require(
        len(sources) == 8 and len(first) == 4, "Expected all eight originals and four wrong sources"
    )
    source_equal = {}
    for arm in ARMS:
        source_equal[arm] = {
            key: mean(source["summary"]["arms"][arm][key] for source in sources)
            for key in (
                "observed_probability_mean",
                "correct_scalar_probability_mean",
                "forward_KL_old_PI_to_arm_mean",
                "total_variation_to_old_PI_mean",
            )
            if all(source["summary"]["arms"][arm][key] is not None for source in sources)
        }
    return dict(
        status="completed_offline_saved_target_audit",
        sources=sources,
        first_wrong_summary=summarize(first),
        source_equal_array_means=source_equal,
        pooled_array_positions_descriptive=summarize(
            [value for source in sources for value in source["values"]]
        ),
        source_count=8,
        incorrect_source_count=4,
        frozen_files=frozen,
        new_model_work=dict(generation_sequences=0, scoring_forwards=0, optimizer_steps=0),
        limitations=[
            "Gold opened only offline after teacher outputs were frozen; never passed to a model.",
            "P(correct) is a verified native scalar-token probability, not an answer probability.",
            "Multi-token or compound values have no fabricated scalar or corrected-prefix score.",
            "First wrong value has correct earlier array entries; its original rule may be wrong.",
            "Later array positions are conditioned on the original possibly erroneous prefix.",
            "Source positions are correlated; pooled counts imply no token-level inference.",
            "B changes framing; C also reveals the complete recorded source and binary feedback.",
            "No downstream-transfer, causal-mechanism or arm-selection claim follows.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fidelity", "framing", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = analyze(args.fidelity, args.framing, args.data)
        write_json(args.output / "analysis.json", result)
        write_json(
            args.output / "receipt.json",
            dict(status="completed", analysis_sha256=sha(args.output / "analysis.json")),
        )
        print(
            json.dumps(
                dict(
                    source_equal_array_means=result["source_equal_array_means"],
                    first_wrong_summary=result["first_wrong_summary"],
                ),
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        write_json(
            args.output / "receipt.json",
            dict(status="failed", error=dict(type=type(error).__name__, message=str(error))),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
