# Build the report

This source package builds the October 2026 technical report **Scalable Meta-Learning from Hindsight**, by Angel Raychev. It contains the report's LaTeX, bibliography and exact figure assets. It does not launch experiments, download models or require a GPU.

## Requirements

- A recent TeX distribution with `latexmk`, pdfLaTeX and BibTeX.
- The packages named in the source preamble, including Latin Modern, AMS math, microtype, TikZ, natbib, caption, titlesec, enumitem, fancyhdr and placeins.
- Python 3.10 or later only if using the convenience builder.

From this directory:

```sh
python3 build-publication.py learning-from-hindsight.tex --output build
```

Or compile directly:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error learning-from-hindsight.tex
```

The convenience builder writes the PDF, build log and input/output hash manifest into `build/`. It checks the final LaTeX log for unresolved references, missing glyphs and layout warnings. Rendering and visually inspecting the pages remains a separate acceptance step.

The pinned `angel-publication.sty` supplies the common typography. The
24 October 2026 presentation revision preserves the report date and scientific
content. See [presentation notes](README-PRESENTATION.md) and
[input hashes](presentation-manifest.json). The original `build.py` remains as
historical build support; the command above includes the style in its manifest.

## What the figures represent

- `figures/context-fidelity-transfer.png` uses saved results from the completed fitted-teacher comparison and direct-context diagnostic. The three panels show distinct quantities: contextual utility, logged teacher-reference KL and causal student accuracy. Their denominators and source-cluster limitations are described in the report.
- `figures/source-fidelity-localization.png` uses an independent audit of saved full-vocabulary source distributions. It localizes imitation discrepancy and selected first-error probabilities; it is not a new answer-generation experiment.
- `figures/learning-loop.svg` is the website's vector method diagram. The report contains an equivalent native TikZ diagram.

Recreating historical figures and verifying compact result records is documented in the companion repository. A fresh GPU execution is a different reproduction task. The source package does not contain model weights, full-vocabulary tensors or adapter checkpoints.

## Historical and publication identity

The report consolidates the September 2026 development component study. It was prepared for public presentation in October 2026 without new experiments. Original private research archive: commit `0116d67f869453c2ba4086b4be9db963284da3f8`.

- Article: https://angelraychev.com/scalable-meta-learning-from-hindsight/
- Repository: https://github.com/RaychevAngel/learning-from-hindsight
- Canonical PDF: https://angelraychev.com/research/learning-from-hindsight/learning-from-hindsight.pdf

Scientific figures retain their original verified values; typography and exposition have been revised. Operational logs, provider records and unrelated private material are not part of this source archive.
