# Evidence and figure sources

The current article and canonical research report are linked from the
[project page](https://angelraychev.com/scalable-meta-learning-from-hindsight/).
This directory contains their empirical support, not a second competing paper.

## Rebuild the empirical material

No model calls are made by these commands:

```sh
python scripts/reproduce_evidence.py
python paper/build_evidence.py --plot
python paper/plot_source_fidelity025.py
python paper/plot_print_figures.py
python paper/build_closeout_evidence.py
```

Only the plotting commands require Matplotlib (`3.11.1` produced the released
figures). The evidence verifier is standard-library-only. The historical
`build_evidence.py` independently checks the exact original observation hashes
before producing its tables; the public verifier additionally regrades the saved
response text against the released synthetic oracles.

| Script | Inputs and interpretation |
|---|---|
| `build_evidence.py` | Completed 023/024 observations and logged fresh-source fidelity; source-equal count/KL panels |
| `plot_source_fidelity025.py` | Independent completed audit's 576 native-position metric records; signal localization |
| `plot_print_figures.py` | Same inputs in narrower print layouts, no changed data |
| `build_closeout_evidence.py` | Arithmetic checks on completed 028 proposer-training and 029 scoring audits |
| `audit_source_fidelity025.py` | Optional original full-tensor re-audit; requires unshipped exact multi-GB inputs |

The retained audit data include native token IDs/displays, regional labels,
correctness references used offline only, full-tensor hashes and exact metric
lineage. Their presence does not mean the complete tensors or historical fitted
models are downloadable. [Evidence scope](../docs/evidence.md) and
[large-artifact manifest](../provenance/large-artifacts.json) define that boundary.

The context/teacher study and subsequent localization study are different
interventions: the latter audits the explicit-lesson target and student update,
not a new inference pass through the fitted hindsight teacher. Keep those labels
intact when reusing a figure.
