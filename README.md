# Learning from hindsight

**Select teaching signals by the learning they cause, then learn to predict those
signals directly from completed experience.**

This research implementation studies a concrete route from an agent's past
attempt to a better parameter update. A source-only proposer supplies candidate
lessons. Each lesson defines a full-vocabulary teacher distribution on the
student's original response tokens. Matched student updates are evaluated on
different validation tasks; the selected distributions become targets for a
future-aware hindsight teacher. The updated student answers new tasks causally,
without receiving the lesson or completed experience in its evaluation prompt.

By **Angel Raychev**. The implementation and historical experiments were developed
in September 2026; this is a curated public research release.

[Overview](https://angelraychev.com/scalable-meta-learning-from-hindsight/)
· [PDF](report/learning-from-hindsight.pdf)

The report uses the shared manuscript style as a 24 October 2026 presentation
revision. Its scientific content, original results, figures and date are
unchanged; no new experiments were performed. [Resources](https://angelraychev.com/scalable-meta-learning-from-hindsight/#resources)
collects the standalone LaTeX package, code release and version record. Start with
the [Quickstart](QUICKSTART.md) to exercise the implementation or the
[evidence guide](docs/evidence.md) to inspect the historical findings.

## The method in one view

```text
completed source attempt + observed feedback
                  │
          source-only lesson proposals
                  │
     full-vocabulary teacher distributions
                  │
       matched causal student updates
                  │
    validation on other tasks, without lessons
                  │
    select useful distributions (or no update)
                  │
      fit a future-aware hindsight teacher
                  │
 raw experience → distributions → causal student update
```

The teacher learns **distributions over the original response positions**, not
textual lessons or replacement solutions. The outer loop can use validation
performance; the inner hindsight scoring pass need not autoregressively generate
a new lesson. That removes a particular generation step, not all teacher,
optimization, or evaluation work.

## Three ways to use this repository

| Goal | Start here | Model / GPU needed? |
|---|---|---|
| Check the historical result arithmetic and grades | `python scripts/reproduce_evidence.py` | No; Python standard library |
| Exercise the learning mechanisms | [CPU quickstart](QUICKSTART.md) | No downloads or GPU during tests; scientific dependencies must be installed |
| Run a new real-model study | [Experiment guide](docs/reproduction.md) | Yes; explicit operator budget and supervision |

These are different claims. Rebuilding a table from saved outputs is not rerunning
training. Random-model correctness tests are not evidence of improved real-model
performance. The new-run guide does not provision or start a GPU automatically.

## What is implemented

- Native response-token and predecessor-logit alignment; detached, full-vocabulary
  targets rather than top-k approximations or decoded/re-tokenized text.
- Source-only lesson proposals, matched adapter/optimizer resets, actual
  post-update validation selection and an explicit no-update alternative.
- Causal student adaptation and task-only evaluation on different tasks.
- Future-aware teacher fitting with no lesson supplied in that teacher's input.
- Lesson-proposer REINFORCE using actual post-update validation rewards.
- Matched causal-teacher controls, source replay and shared-active-parameter
  machinery, with storage/forward-mode tests.
- Synthetic ARC-1D task construction with independent rule checks, integrity
  manifests, episode separation and exact output grading.

For the qualified Qwen3.5-4B model, "future-aware" has a precise scope: eight full-attention layers receive
unrestricted attention, while 24 DeltaNet layers remain causal. It is not a fully
bidirectional network. The completed main teacher comparison used separate LoRA
adapters on a common frozen base; implemented parameter sharing is a separate
experimental branch, not an established continual-learning result.

## Historical findings

The study separates **useful contextual information**, **distribution fitting**
and **causal student improvement** instead of treating any one as proof of all
three. All counts below are descriptive development evidence, not independent
confirmation or a broad benchmark claim.

| Matched development comparison | No update | Lesson in context | Lesson-derived update | Fitted hindsight update |
|---|---:|---:|---:|---:|
| Earlier source-only transfer study: query successes /64 | 23 | 43 | 36 | — |
| Main teacher study: query successes /64 | 27 | 42 | 30 | 30 |

The first row and second row use different episodes/lessons and must not be pooled
as repetitions. In the second row, lesson-in-context results were obtained in a
subsequent matched diagnostic; the original baseline and updated-student outputs
were retained. Each /64 comprises eight query tasks per source, with eight source
clusters from four familiar task families.

The main teacher experiment used one fixed, externally proposed source-only lesson
per episode plus the neutral alternative; its fitted teacher was not trained from
the subsequently optimized proposer. Those lessons and source identities are
included. The external proposer was a Codex subagent whose immutable model revision
was not recorded, so verbatim lesson replay is possible but identical fresh lesson
generation is not claimed. The candidate-pool and learned-proposer runners are
separate implemented paths, with their own evidence and controls.

In the main teacher study, fitted hindsight distributions improved fidelity on
all eight fresh source experiences: source-equal forward KL to selected reference
distributions fell from **0.27905 to 0.05826**. But fitted-student query contrasts
had source-cluster intervals spanning zero. **Reliable end-to-end improvement from
the learned hindsight teacher remains unestablished.**

A source-level diagnostic helps explain what needs closer investigation. In the
explicit-lesson update, **98.54% of initial imitation KL lay in the verbal rule**.
The teacher and initial student agreed on their top choice at all 150 inspected
array-value positions, including incorrect choices. Strong aggregate source
fitting can therefore coexist with a weak corrective signal in the answer. This
is a measured diagnostic, not a complete causal explanation of transfer.

The release also retains the earlier complete comparison, where the
validation-selected adaptation did not improve query performance, and the
completed proposer-training audit. The latter establishes actual optimization
(14 updates across 32 groups), not held-out proposer usefulness. Negative,
inconclusive and invalid outputs remain in the released evidence.

## Reproduce the evidence without inference

```sh
git clone https://github.com/RaychevAngel/learning-from-hindsight.git
cd learning-from-hindsight
python3 scripts/reproduce_evidence.py
python3 -m unittest discover -s tests -p 'test_public_evidence.py'
```

The first command verifies packaged input hashes, independently regrades saved
answers against the synthetic oracles, checks expected observation identities,
recomputes source/condition counts and paired comparisons, and checks the
source-fidelity aggregates against the released native-position metrics. It makes
**zero model, tokenizer, GPU, provider or network calls**. Use `--output PATH` to
save its report; the default only prints it.

After installing the optional plotting dependency:

```sh
python3 -m pip install 'matplotlib==3.11.1'
python3 paper/build_evidence.py --plot
python3 paper/plot_source_fidelity025.py
python3 paper/plot_print_figures.py
python3 paper/build_closeout_evidence.py
```

These rebuild the compact historical tables and figures from included evidence.
The 025 figures use audited per-position metrics; recomputing those metrics from
the full probability tensors is a different, larger audit whose tensor inputs are
not included in Git. See [evidence scope](docs/evidence.md) and
[provenance](provenance/README.md).

## Repository map

| Directory | Purpose |
|---|---|
| `src/hindsight/` | Model operations, task/episode integrity, adaptation, proposer, controls and shared-mode implementation |
| `scripts/` | Dataset builders, experiment runners, diagnostics and analysis |
| `examples/quickstart/` | Self-contained new demo data and bounded run-configuration preparation |
| `tests/` | Tiny random-model, fake-coordinator and integrity tests |
| `data/` | Selected historical synthetic corpora needed by released evidence/tests |
| `results/` | Compact completed observations, native generations and independent audit summaries |
| `paper/` | Evidence/figure regeneration; the publication is linked separately |
| `report/` | Canonical research report, LaTeX sources and self-contained build instructions |
| `provenance/` | Source identity, export manifest, large-artifact boundaries and release verification |

The historical file label `tasks.private.jsonl` means **oracle information hidden
from model prompts**, not personal/private customer data. The released oracles are
synthetic and are necessary for independent regrading. These public historical
tasks are not a newly sealed evaluation set.

## Requirements and scope

The original work used Python 3.11, Torch 2.9.1, PEFT 0.20.0 and a pinned
Transformers source revision for Qwen3.5. The separate lock is authoritative;
see [QUICKSTART.md](QUICKSTART.md). The qualified real-model hardware was one
H100 80 GB; smaller GPUs have not been qualified here. Full-vocabulary tensors
and checkpoints require substantial storage even when generation is short.

No model weights, trained adapters, multi-GB target tensors, billing/account
records or historical provider automation are in this release. The numerical
model path and experiment controls are supplied; a fresh scientific rerun is not
claimed by publication of this package.

## License and attribution

The original project code is released under [Apache-2.0](LICENSE). Retained
ReasoningGym source and notices are under `third_party/reasoning_gym/`; see
[NOTICE](NOTICE) for the exact scope, including the original ARC-1D MIT notice.
The research report is an authored manuscript; the software license does not
relicense cited papers, external model weights or third-party material.
Downloaded model weights and external
packages remain governed by their own terms. This license does not grant rights
to external model weights or imply endorsement by their authors.
