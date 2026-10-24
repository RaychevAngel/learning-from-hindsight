# Start here

This repository has two entirely offline entry points. **Neither downloads model
weights, starts a GPU or calls a provider.** Network access is needed only if you
choose to install missing software dependencies.

## 1. Reanalyse released evidence with the Python standard library

From the repository root:

```sh
python3 scripts/reproduce_evidence.py
python3 -m unittest discover -s tests -p 'test_public_evidence.py'
```

This verifies input hashes, regrades saved answers, checks source/task pairing and
recomputes the reported tables and compact diagnostic aggregates. It does not rerun
historical training. The default writes nothing; optional `--output PATH` saves a
report only to a new destination. See [the evidence guide](docs/evidence.md).

## 2. Install the pinned scientific environment

Python 3.11 is the qualified version. The lock also records Python 3.12 dependency
markers, but that interpreter was not used for the release verification.

```sh
python3.11 -m venv .bootstrap
.bootstrap/bin/python -m pip install 'uv==0.11.28'
.bootstrap/bin/uv venv --python 3.11 .venv-qwen35
.bootstrap/bin/uv pip install --python .venv-qwen35/bin/python -r requirements-qwen35.lock
.bootstrap/bin/uv pip install --python .venv-qwen35/bin/python --no-deps -e .
```

The lock pins Torch 2.9.1, PEFT 0.20.0, NumPy 2.4.6 on Python 3.11, and
Transformers source revision `c93057d4835cd31752bb56f59989dd27696eb45b`
(reported version `5.17.0.dev0`). Public project metadata is aligned with this
Qwen3.5 environment. The private archive's older Qwen3 `uv.lock` is deliberately
not copied as a misleading default. Use the explicit interpreter below.

## 3. Run the self-contained CPU mechanism proof

```sh
.venv-qwen35/bin/python scripts/check_fresh_clone.py
.venv-qwen35/bin/python examples/cpu_mechanism_demo.py
```

It runs an explicit test subset using tiny randomly initialized Qwen3/Qwen3.5
models and temporary fake-coordinator fixtures. The child process disables CUDA
visibility and Hub access, replaces inherited `PYTHONPATH`, and verifies the
imported package belongs to this checkout. `--list` prints the exact scope.

The tests cover native token/logit alignment, causal and future-aware modes,
detached full-vocabulary targets, real tiny-model student/teacher optimization,
proposer REINFORCE, shared-active-parameter storage, matched teacher controls,
grading and evidence accounting. Passing them establishes implementation
properties—not useful learning on Qwen3.5-4B.

GitHub CI runs the dependency-free evidence checks on every push. Its optional
manual `cpu_microtests` job installs the pinned packages and runs this same CPU
proof on macOS; it still cannot download model weights. A green dependency-free
job alone does not establish that the optional scientific package install passed.

The second command is a short standalone API example: construct a tiny random
model, score detached full-vocabulary targets, make actual student updates, fit
a separate future-aware teacher, update a reset student from its predictions,
and generate on a different task-only prompt. It has no task-accuracy claim and
does not pretend to demonstrate useful validation-selected meta-learning.

The complete historical-fixture suite can also be run offline:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=src .venv-qwen35/bin/python -m pytest -q
```

The included synthetic corpora supply the file-integrity/admission fixtures that
were absent from the private archive's original fresh-clone path. All test model
constructions remain tiny/random or injected fake engines. No real-model result
is inferred from this suite.

## 4. Generate an independent demo corpus, still CPU-only

```sh
PYTHONPATH=src .venv-qwen35/bin/python examples/quickstart/prepare_data.py \
  --output data/quickstart
```

The helper creates its own support corpora and 80 new episodes: 32 meta-training,
16 development and 32 confirmation, with one source, eight validation tasks and
eight queries per episode. It does not require historical `data/` or `results/`.
It refuses to overwrite existing output. Repeated `--exclude-corpus PATH` flags
can exclude verified leaf public corpora from its generated inputs.

This is a new integration-demo dataset, not the historical study and not
unseen-family generalization: it retains four familiar operator families. The
current model-facing task interface and grader are **arrays**, not competition
mathematics or a general task adapter.

## 5. Optional real-model work

Continue with [the experiment guide](docs/reproduction.md) only after setting your
own GPU, runtime and storage limits. It provides explicit commands for proposer
training/evaluation/target export, candidate-pool comparison, fitted teacher
controls and shared-active-parameter experiments. Configuration preparation is
CPU-only; subsequent scoring, generation and optimization require the model.

The example uses one student step and two teacher updates to check integration.
It is not the historical scientific recipe and is not a seconds-long smoke test.
An E4 `completed` status can still contain zero optimizer steps when every reward
group has zero contrast. Inspect work counters and parameter differences; actual
updates do not establish held-out benefit.

**Operational boundary:** application timeouts do not end GPU billing. This
includes qualification's configured per-invocation limit and shared absolute
deadline: callbacks enforce them at execution boundaries, not by interrupting an
arbitrary in-flight model operation. The default qualification allowance is 600
seconds only when no explicit limit is supplied.
This
release contains no provider credentials or provider-control daemon. Arrange a
shutdown bound independent of the client machine before unattended work, and
store outputs on verified persistent storage. Do not use a local network-dependent
watchdog as a hard spending guarantee.
