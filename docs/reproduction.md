# Reproduction: integration path and scientific matrix

Start with [QUICKSTART.md](../QUICKSTART.md). Run all commands from the repository
root using the separate Qwen3.5 environment. Treat each section as a review gate,
not an unattended shell script. This guide creates **new experiments**, not historical
result artifacts, and never provisions a GPU or publishes anything.

## Model, hardware and invariants

The real-model commands pin `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with thinking disabled.
The qualified research hardware was one H100 80 GB. Smaller GPUs, CPU execution of
the 4B model, and other kernels are not established by the tiny-model CPU tests.
Allow tens of GB of disk for weights, FP32 full-vocabulary targets and checkpoints;
multiple target collections can occupy several GB. Measure memory and throughput
before scaling. Batch qualification does not itself certify training-memory capacity.

Qwen3.5 has eight full-attention and 24 DeltaNet layers. The hindsight path makes
full-attention layers unrestricted; DeltaNet recurrence/convolution remains causal.
Call it a **future-aware hybrid**, not a fully bidirectional model.

The information boundary is source prompt + observed source response + binary
feedback. Validation scores may select an actual post-update candidate, with no-update
winning ties. Query outcomes never choose candidates, hyperparameters or checkpoints.
All actions retain original response token IDs and predecessor-logit alignment;
targets are detached full-vocabulary log probabilities. Keep every failure, invalid
proposal, output-cap hit and zero-advantage group.

## 1. Prepare bounded configurations

First generate `data/quickstart` with the CPU helper. Choose a real future,
timezone-aware deadline; the placeholder below intentionally cannot launch a run.

```sh
RUN_DEADLINE='REPLACE_WITH_FUTURE_YYYY-MM-DDTHH:MM:SS+00:00'
.venv-qwen35/bin/python examples/quickstart/prepare_configs.py \
  --data data/quickstart --output results/quickstart-config-before-qualification \
  --deadline-utc "$RUN_DEADLINE" --max-seconds 1800
```

This only creates `throughput.json`. The bound applies per invocation, while the
absolute deadline is common to all commands. Checks are cooperative; an in-flight
model operation is not forcibly interrupted. Qualification honors this configured
positive finite allowance, using 600 seconds only if it is absent.
It is not a dollar-budget controller:
you still own billing limits, process supervision, checkpoint backup and GPU shutdown.

## 2. Qualify a new generation schedule — GPU

```sh
.venv-qwen35/bin/python scripts/qualify_overnight_throughput.py \
  --config results/quickstart-config-before-qualification/throughput.json \
  --data data/quickstart --output results/quickstart-throughput
```

This measures 80 generated responses on 16 development validation tasks. It selects
the fastest repeat-stable batch of four/eight, not the highest-scoring schedule.
Inspect `receipt.json`: require `status: completed`, repeat stability and a realistic
completion forecast. Batched tokens may differ from singleton tokens; all subsequent
arms use the same newly qualified schedule. Do not copy a historical qualification
receipt onto a different corpus/model/configuration.

```sh
.venv-qwen35/bin/python examples/quickstart/prepare_configs.py \
  --data data/quickstart --output results/quickstart-config \
  --deadline-utc "$RUN_DEADLINE" --max-seconds 1800 \
  --qualification results/quickstart-throughput/receipt.json
```

The helper pins the actual dataset and qualification SHA256 values. It chooses source
IDs before outcomes, with four balanced meta-training sources and eight balanced
development sources for E4/E7. The E1/E3 CLI requires the complete 16-source development
partition, and E5/E6 subsequently uses that complete comparison manifest.
E4's four-task reward groups physically use batch four even when the maximum batch
is eight; the example helper therefore also requires repeat stability at four.

## 3. E4: train, inspect, evaluate and export targets — GPU

```sh
.venv-qwen35/bin/python scripts/train_proposer.py \
  --config results/quickstart-config/proposer.json --data data/quickstart \
  --output results/quickstart-proposer-train --phase train
```

Each source draws four independent proposals from the current proposer. Rewards are
actual task-only success on validation indices `[0,2,4,6]` after matched student
updates. Unusable proposals map to the measured no-update baseline. The objective is
leave-one-out REINFORCE over summed native token log likelihood, not a learned critic
or a length-normalized/PPO objective. Every group remains in the receipts.

Before continuing, inspect `receipt.json`, `groups.jsonl`, `adaptations.jsonl` and
`checkpoints/final.pt`. Four completed source groups do not imply four proposer updates:
zero-contrast groups perform no optimizer step. Changed proposer weights establish
optimization, not benefit; evaluate them against their frozen initialization:

```sh
.venv-qwen35/bin/python scripts/train_proposer.py \
  --config results/quickstart-config/proposer.json --data data/quickstart \
  --output results/quickstart-proposer-evaluate --phase evaluate \
  --checkpoint results/quickstart-proposer-train/checkpoints/final.pt

.venv-qwen35/bin/python scripts/export_proposer_targets.py \
  --training-output results/quickstart-proposer-train --data data/quickstart \
  --output results/quickstart-teaching --deadline-utc "$RUN_DEADLINE" --max-seconds 300
```

Export is a real scoring pass, not CPU-only file conversion: one deterministic source
score per training source, no generation or student update. It verifies completed E4
V4 evidence and emits `manifest.json` plus native sources and selected FP32 targets.
Its neutral selection exports the causal no-update distribution. You may reuse native
sources for a matched run using `train_proposer.py --sources PRIOR_OUTPUT`; do not
regenerate or relabel them while claiming identical-source comparisons.
The coordinator rejects saved text that differs from the active tokenizer's
decoding of those native response IDs; it does not silently retokenize or normalize
a contradictory attempt into a different one.

## 4. E1/E3: compare the candidate pool — GPU

```sh
.venv-qwen35/bin/python scripts/compare_pi.py \
  --config results/quickstart-config/pi-comparison.json --data data/quickstart \
  --output results/quickstart-pi-comparison
```

This command uses the initialized proposer, not the E4 checkpoint, unless you explicitly
supply `--checkpoint` **and** configure its pinned `proposer_checkpoint_sha256` and
matching training identity. The default above keeps the two comparisons distinct.
It generates and evaluates the full K4 pool on all 16 development sources, with direct
context, adapted-student, neutral and other-family-mismatch controls. It produces
`teaching.json`, selecting targets with actual post-update V8 evidence. Inspect the
receipt rather than interpreting process exit alone: this CLI prints its returned
status, and a failed result must not be treated as complete downstream input.

## 5. E5/E6: fitted hindsight and causal teachers — GPU

```sh
.venv-qwen35/bin/python scripts/check_teacher_controls.py \
  --config results/quickstart-config/teacher-controls.json --data data/quickstart \
  --training-manifest results/quickstart-teaching/manifest.json \
  --evaluation-manifest results/quickstart-pi-comparison/teaching.json \
  --output results/quickstart-teacher-controls
```

The two fitted teachers share initial tensors, source ordering, selected targets and
optimizer work. One uses future-aware attention; the other receives the completed
response and feedback through ordinary causal serialization. This is an information-
matched control, **not** identical token positions/context length or a pure mask ablation.

All seven arms generate fresh task-only Q answers: baseline, selected PI targets,
fitted future-aware teacher, unfitted future-aware teacher, fitted raw-causal teacher,
unfitted raw-causal teacher and exact source-token replay. They share the frozen source
and actor initialization. Inspect `fit-fidelity.jsonl`, `cases.jsonl`, native
`observations.jsonl`, checkpoints and terminal counters. Replay can yield infinite
PI-to-one-hot KL; the diagnostic explicitly records this rather than inventing a finite value.

## 6. E7: same active parameters in both modes — GPU

```sh
.venv-qwen35/bin/python scripts/check_shared_learning.py \
  --config results/quickstart-config/shared.json --data data/quickstart \
  --teacher-manifest results/quickstart-teaching/manifest.json \
  --output results/quickstart-shared
```

This initial CLI cut requires eight development sources, two seeds and five arms.
The same `student` adapter is genuinely fitted in hindsight mode and updated/used in
causal mode; runtime storage audits observe actual forwards. A separate adapter is used
only for the explicit separate-teacher control. Each actor draws its own source, so
different starting actors may yield different source trajectories: this is a full-method
contrast, not a claim that parameter sharing alone explains a Q-score difference.

Review `fit-interference.jsonl`, `cases.jsonl`, `generations.jsonl`, storage/forward
audits and terminal `receipt.json`. The current CLI is not the original legacy
`hindsight freeze`/final lifecycle and does not silently expand into confirmation.

## 7. Independently audit completed outputs — CPU

```sh
.venv-qwen35/bin/python scripts/analyze_overnight.py --kind proposer \
  --run results/quickstart-proposer-evaluate --data data/quickstart \
  --output results/quickstart-proposer-audit.json
.venv-qwen35/bin/python scripts/analyze_overnight.py --kind comparison \
  --run results/quickstart-pi-comparison --data data/quickstart \
  --output results/quickstart-pi-audit.json
.venv-qwen35/bin/python scripts/analyze_overnight.py --kind teachers \
  --run results/quickstart-teacher-controls --data data/quickstart \
  --output results/quickstart-teacher-audit.json
.venv-qwen35/bin/python scripts/analyze_overnight.py --kind shared \
  --run results/quickstart-shared --data data/quickstart \
  --output results/quickstart-shared-audit.json
```

These regrade stored responses against the corpus and validate complete outcome/work
identities. They report paired source-level differences and bootstrap intervals per
seed, not token/task-independent significance. They do not regenerate model outputs,
reconstruct a missing run, or certify useful improvement merely from audit completion.

## Integration examples versus scientific execution

| Setting | Generated integration examples | Overnight scientific recipe |
|---|---|---|
| Student adaptation | 1 step, LR 1e-4 | 16 steps, LR 1e-4 |
| Teacher fitting | 2 updates, batch 4 | 64 updates, batch 4 |
| Teacher training sources | 4, one per family | 32 meta-training sources |
| Development/confirmation | Demo development only | Separate frozen 16-development/32-confirmation cohorts |
| E7 initial cut | 8 sources, 2 seeds, 5 arms | Same initial cut; expansion requires an explicit frozen plan |

Small update counts are for integration, not adequate optimization or scientific
confirmation. Even this optional path contains thousands of generations across all
sections; run only the pieces you need, measure throughput, and stop at your deadline.
Do not change the student recipe without redoing actual-V target selection/export.
Teacher/student work counters and example exposures are different quantities.

The full scientific datasets, immutable source/result receipts and frozen per-run
configurations are required to reproduce historical numbers. Neither a successful
fresh-clone test nor regenerating task IDs reconstructs a historical model rollout.
The original `corpus-v1/v2` contains 112 episodes across meta-train, development,
final-known and final-unseen partitions. The later four-family 80-episode cohort has
32/16/32 meta-train/development/confirmation partitions; those are different experiments.

## Included original corpora and fixture-dependent tests

The curated public release includes the historical synthetic corpus schemas used
by these fixtures. The following commands are **optional regeneration instructions**,
not steps to run over the included directories. Builders intentionally refuse to
overwrite them. To compare regenerated bytes, use a fresh temporary directory and
explicitly point downstream commands at those new paths.

To generate the original data schemas without any model calls, in a checkout where
the destination paths do not already exist:

```sh
.venv-qwen35/bin/python scripts/build_corpus.py --output data/corpus-v1
.venv-qwen35/bin/python scripts/revise_prompt.py --source data/corpus-v1 --output data/corpus-v2
.venv-qwen35/bin/python scripts/build_balanced_development.py \
  --output data/balanced-development-v1 --old-corpus data/corpus-v2 --seed 2026090601
.venv-qwen35/bin/python scripts/build_balanced_development.py \
  --output data/balanced-transfer-v1 --old-corpus data/corpus-v2 --seed 2026090602 \
  --exclude-corpus data/balanced-development-v1
.venv-qwen35/bin/python scripts/build_overnight_cohort.py --output data/overnight-v2
```

These commands need no historical result files. Manifest script hashes can change
when generator source changes; compare hashes before claiming byte-identical data.
The standalone demo helper avoids these fixed historical path dependencies entirely.

With the included folders, corpus tests and data-dependent E7/E5 admission tests
can run. The release also includes the compact historical metadata needed by:

- `tests/test_order_and_pi.py`: 022 sources and 023 comparison records.
- `tests/test_source_fidelity.py`: the same records plus 023 per-step events.

Those fixtures remain exact original bytes, pinned in the public evidence manifest;
they are not synthetic stand-ins for missing historical records. Keep old artifacts
read-only and run historical reconstruction separately from the
data-free CPU proof. Numerical behavior on a new dependency stack/hardware is not
automatically identical to the original checkpointed experiment.
