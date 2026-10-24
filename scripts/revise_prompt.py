"""Create prompt-only corpus v2 from verified v1 without regenerating any task."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

OLD_HEADER = "\n".join(
    [
        "Infer the common transformation from the four demonstrations and apply it to the query.",
        "Give a brief explanation, then exactly one final line: FINAL: [0, 1, ...].",
        "The FINAL field must be a JSON array of integers with the same length as the query.",
    ]
)
INTRO = "Infer the common transformation and apply it to the query."
RESPONSE_CONTRACT = "\n".join(
    [
        "Respond in exactly two lines.",
        "First line: one rule sentence of at most 12 words.",
        "Second line: FINAL: followed by the complete output grid as a JSON array.",
        "Do not restate demonstrations, show steps, or add any other text.",
    ]
)
DATA_FILES = ("tasks.public.jsonl", "tasks.private.jsonl", "episodes.jsonl")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def revise_prompt(record):
    old_prompt = record["prompt"]
    prefix = OLD_HEADER + "\n"
    if not old_prompt.startswith(prefix):
        raise ValueError("Unexpected v1 prompt contract; refusing a broad text replacement")
    body = old_prompt[len(prefix) :]
    expected = []
    for i, pair in enumerate(record["demonstrations"], 1):
        expected.extend(
            [
                f"Example {i} input: {json.dumps(pair['input'])}",
                f"Example {i} output: {json.dumps(pair['output'])}",
            ]
        )
    expected.append(f"Query input: {json.dumps(record['query_input'])}")
    if body != "\n".join(expected):
        raise ValueError("Public prompt body disagrees with its demonstration/query records")
    return INTRO + "\n" + body + "\n\n" + RESPONSE_CONTRACT


def revise(source: Path, output: Path):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    parent_raw = (source / "manifest.json").read_bytes()
    parent = json.loads(parent_raw)
    if parent["version"] != "arc1d-hindsight-v1":
        raise ValueError("This revision accepts only the original corpus v1")
    raw = {name: (source / name).read_bytes() for name in DATA_FILES}
    if any(sha(raw[name]) != parent["files"][name] for name in DATA_FILES):
        raise ValueError("Source corpus integrity failure")
    public = parse_rows(raw["tasks.public.jsonl"])
    private = parse_rows(raw["tasks.private.jsonl"])
    secrets = {row["task_id"]: row for row in private}
    if (
        len(public) != parent["tasks"]
        or len(private) != parent["tasks"]
        or len(secrets) != len(private)
        or {row["task_id"] for row in public} != secrets.keys()
    ):
        raise ValueError("Source task identities/counts disagree")
    revised_public, revised_private, pairing, semantics = [], copy.deepcopy(private), [], []
    revised_secrets = {row["task_id"]: row for row in revised_private}
    for row in public:
        task_id = row["task_id"]
        before_hash = sha(encoded(row))
        if before_hash != secrets[task_id]["public_sha256"]:
            raise ValueError("Source public/private hash disagreement")
        updated = {**row, "prompt": revise_prompt(row)}
        after_hash = sha(encoded(updated))
        revised_public.append(updated)
        revised_secrets[task_id]["public_sha256"] = after_hash
        semantics.append(
            {
                "public_without_prompt": {k: v for k, v in row.items() if k != "prompt"},
                "private_without_public_hash": {
                    k: v for k, v in secrets[task_id].items() if k != "public_sha256"
                },
            }
        )
        pairing.append(
            {
                "task_id": task_id,
                "parent_public_sha256": before_hash,
                "revised_public_sha256": after_hash,
            }
        )
    revised_raw = {
        "tasks.public.jsonl": b"".join(encoded(row) + b"\n" for row in revised_public),
        "tasks.private.jsonl": b"".join(encoded(row) + b"\n" for row in revised_private),
        "episodes.jsonl": raw["episodes.jsonl"],
        "prompt-pairing.jsonl": b"".join(encoded(row) + b"\n" for row in pairing),
    }
    manifest = copy.deepcopy(parent)
    manifest.update(
        {
            "version": "arc1d-hindsight-v2-prompt-only",
            "status": "data_qualified_prompt_revision_model_unqualified",
            "files": {name: sha(value) for name, value in revised_raw.items()},
            "prompt_revision": {
                "revision": "two-lines-rule-at-most-12-words",
                "parent_version": parent["version"],
                "parent_manifest_sha256": sha(parent_raw),
                "parent_files": {name: sha(raw[name]) for name in DATA_FILES},
                "old_header": OLD_HEADER,
                "new_intro": INTRO,
                "new_trailing_contract": RESPONSE_CONTRACT,
                "changed_fields": ["tasks.public.prompt", "tasks.private.public_sha256"],
                "semantic_identity_sha256": sha(encoded(semantics)),
                "paired_tasks": len(pairing),
                "episodes_byte_identical": True,
                "regenerated_tasks": 0,
                "model_calls": 0,
                "reason": (
                    "Development E0 observed truncation under the loose brief-explanation "
                    "instruction. This revision constrains response length without changing "
                    "any task or split."
                ),
                "script_sha256": sha(Path(__file__).read_bytes()),
            },
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, value in revised_raw.items():
        (output / name).write_bytes(value)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if (source / "manifest.json").read_bytes() != parent_raw or any(
        (source / name).read_bytes() != raw[name] for name in DATA_FILES
    ):
        raise RuntimeError("Parent corpus changed during prompt revision")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = revise(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tasks": manifest["tasks"],
                "episodes": manifest["episodes"],
                "files": manifest["files"],
                "parent_manifest_sha256": manifest["prompt_revision"]["parent_manifest_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
