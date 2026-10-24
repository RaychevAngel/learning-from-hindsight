# Release provenance

The public package is curated from the private historical
`RaychevAngel/scalable-hindsight` archive at commit
`0116d67f869453c2ba4086b4be9db963284da3f8`. It is a new repository, not a
visibility change or history rewrite of that archive.

- `source-manifest.json` compares each exported file with its historical source.
  All ten original `src/hindsight/` scientific modules remain byte-identical.
  Five coordinator scripts have the release-only safety repairs described below;
  additions provide documentation, dependency alignment, reanalysis, tests and CI.
- `evidence-manifest.json` pins the selected compact data/result inputs by SHA256
  and size. Some were intentionally ignored by the original Git repository, so
  the source commit alone does not pin those historical artifacts.
- `large-artifacts.json` records known hashes/sizes for optional, unshipped
  full-vocabulary tensors and adapter checkpoints. It is not a download catalog:
  **there is no public retrieval URL for these artifacts in this release**.

The package preserves completed observations, including malformed and incorrect
answers. It does not include provider credentials, account IDs, operational
transcripts, caches, model weights, multi-GB tensors or partial runs presented as
completed science. Synthetic oracle files are released deliberately for regrading;
they must not be put into a model's input by an experiment runner.

The original per-run configurations contain expired September deadlines and
historical dataset/code hashes. They are evidence of those runs—not ready-to-launch
commands. The quickstart prepares new, bounded configurations with new identities.
Do not remove historical admission/hash checks to make a differently conditioned
run look like a reproduction.

## Release-only changes

- Clear top-level navigation among offline reanalysis, CPU correctness and new
  real-model experimentation.
- Project metadata aligned with the already pinned Qwen3.5 environment, instead
  of advertising the private archive's older default Qwen3 dependency stack.
- A standard-library evidence verifier/regrader and adversarial integrity tests.
- Apache-2.0 licensing/attribution, selected synthetic fixtures and offline CI.
- Public-facing reproduction instructions no longer depend on private GitHub
  access or excluded provider-control scripts.
- A compact 7.2 × 5.4 inch print layout for the 025 localization figure matches
  the canonical report; the historical numeric inputs and screen layout stay
  unchanged.
- Throughput qualification now honors a positive finite configured per-invocation
  limit (600 seconds when absent), alongside its timezone-aware absolute deadline.
  These are cooperative callback checks, not an external process or billing kill.
- Reused source attempts must exactly match the active tokenizer's decoding of
  their native response IDs before text-derived feedback or teaching is admitted.
  The check is shared by the proposer/candidate comparison and teacher/shared
  coordinators. Structural CLI preflight remains before model construction;
  exact decoding and feedback admission occur once the tokenizer exists, before
  fitting. Newly generated shared-mode sources are checked immediately afterward.
  Low-level token-defined scoring/alignment APIs are unchanged.
- Deterministic timeout/admission regressions and coherent fake tokenizers extend
  the original tests. These repairs prevent inconsistent new runs; they do not
  imply that historical model-generated attempts had contradictory text/tokens.

No saved model outputs, stored scores, scientific numerical modules or historical
evidence texts were edited to improve a result. The source manifest distinguishes
each unchanged historical file, release edit and addition. Publication did not
rerun the experiments.
