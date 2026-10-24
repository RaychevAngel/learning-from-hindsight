# Presentation revision - 24 October 2026

This is a local presentation revision of the current public report, using
`angel-publication` version 1.0. It is not an arXiv or journal revision and does
not report a new experiment. The report's original preparation date and author
attribution remain in the title block. Scientific claims, values, equations,
captions, cited works and exact figure assets are preserved.

The original preamble and scientific macros are retained. A shared layout
package is loaded last to set one-inch margins, Latin Modern typography,
headings, captions, links and page behavior. Wide diagrams have an explicit
shrink-to-width wrapper. The search report additionally has one explicit line
break among long source-path links. Front-matter links use the shared labels
Overview and Code, in that order, and point to the main report page and project
repository. The delivery's external validation receipt records the exact diff
and original/new hashes.

Build with a local TeX distribution providing pdfLaTeX, BibTeX and latexmk:

```sh
python3 build-publication.py REPORT.tex --output build
```

Replace `REPORT.tex` with this package's main report source. The Python helper
performs no model calls, experiments or network operations. It checks for
unresolved/multiply-defined references and citations, missing characters and
overfull boxes, then writes the PDF and an input/output hash manifest. Rendering
and page-by-page visual review remain separate acceptance steps. The original
build instructions are retained as provenance; use the new helper so the pinned
shared style is included in its manifest.

This package contains its exact style copy and builds independently. Do not
point a historical or submitted paper at a moving shared style. Future content
edits, repository adoption and publication require their own review.
