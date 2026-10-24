# What can be reproduced from the release?

The release separates three levels of reproduction.

## 1. Historical evidence reanalysis — included, no scientific environment needed

`python scripts/reproduce_evidence.py` uses only the Python standard library.
It checks every packaged input against `provenance/evidence-manifest.json` and:

- independently grades 1,728 initial-comparison answers, 384 earlier balanced
  transfer answers, 512 main-teacher answers and 224 follow-up diagnostic answers;
- rejects duplicates, incomplete identity matrices and stored grades that disagree
  with the released grading function and synthetic oracles;
- recomputes V/Q totals by condition and source, retaining malformed/capped answers;
- recomputes the main fitted-teacher paired source-cluster intervals;
- verifies direct-context versus adapted-student query overlap;
- recomputes 025 source-equal and pooled aggregates from all 576 saved native-position
  metric records, without treating positions as independent statistical samples;
- checks the arithmetic in completed 028/029 independent audit summaries.

The 2,848 total is a count of **saved evaluation records**, not distinct tasks,
independent training runs or independent samples. Conditions share tasks, and 96
of the 224 follow-up records are deliberate three-order replays on two sources.
The main inferential grouping remains the source episode, not the record count.

The original 023 evaluation texts did not retain native evaluation token IDs.
The 024 diagnostic did retain them; its three orderings agree and reproduce the
historical adapted-student texts on the two checked sources. This is not a claim
that original evaluation token IDs were recovered for every answer.

The fixed external lesson strings are released verbatim with source hashes and
their source-only instruction/provenance. The proposer sessions did not record an
immutable model alias/revision. Reusing those exact historical strings reproduces
the intervention text; asking a current model to generate new strings is a new
intervention, not an exact reconstruction. The later actual-reward proposer
optimization is separate from the earlier fitted-teacher comparison.

`paper/build_evidence.py` rebuilds 023/024 counts and logged fresh-source teacher
fidelity. The latter fidelity records are logged numerical measurements; they
are not recomputed teacher predictions because the fitted teacher's evaluation
prediction tensors were not retained. The independent 025 diagnostics concern
the explicit-lesson teacher, not that fitted hindsight teacher.

`paper/plot_source_fidelity025.py` and `paper/plot_print_figures.py` regenerate
figures from included compact data. Their outputs are descriptive analyses of
completed development work, not new generated answers or model scoring calls.

## 2. Mechanism correctness — included, needs the pinned scientific environment

`python scripts/check_fresh_clone.py` runs a deliberately explicit CPU subset.
It uses tiny randomly initialized models and temporary fake-coordinator fixtures.
The helper disables Hub access and CUDA visibility and verifies imports belong
to this checkout. It does not download a checkpoint or prove a large model learns
usefully. The complete suite additionally uses the released historical synthetic
corpora/metadata, but still creates no real-model results.

## 3. Scientific rerun — implementation supplied, not executed by this release

The runners and pinned model/environment identities are available in
[reproduction.md](reproduction.md). Running them requires model access, a suitable
GPU, storage and an explicitly bounded new configuration. A freshly generated
quickstart corpus is an integration demo, not byte-identical historical training.

The following are not in this Git package:

| Artifact | Why it matters | Release status |
|---|---|---|
| 025 full-vocabulary target, initial and post-update tensors | Recompute per-position KL/TV/probabilities directly from tensors | Roughly 1.6 GB historical run directory; only compact metrics, native alignment references and tensor hashes are included |
| Trained 023 teacher adapter | Independently re-forward the trained teacher | Checkpoint hash retained; weights are not distributed |
| 028 proposer checkpoints and full training outputs | Reaudit optimization likelihoods/parameters from checkpoints | Independent completed audit and hashes included, not tensors |
| Partial later confirmation/shared-mode backups | Establish a completed expanded comparison | Excluded from reported outcomes; no completed terminal evidence recovered |

`provenance/large-artifacts.json` records exact known hashes and sizes, not invented
public download links. No public hosting location for these optional tensors has
been established. Do not describe their audit as independently rerunnable from a
fresh clone alone.

The original source-tensor audit implementation is retained at
`paper/audit_source_fidelity025.py`. It intentionally fails if its exact historical
tensor inputs are absent or different. Publication does not weaken its checks to
pretend those tensors were shipped.

## Scope of inference

All reported Q outcomes are development evidence. Source clusters, familiar
families, repeated development choices, one model/fit seed and batching changes
limit generalization. Source-cluster intervals do not cover task-family sampling,
model initialization variation or all researcher choices. Equal totals are not
evidence of equivalence. Actual proposer updates are not evidence of improved
held-out proposer usefulness. Implemented shared-active parameters are not proof
of continual self-improvement.
