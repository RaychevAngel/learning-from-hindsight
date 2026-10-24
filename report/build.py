"""Build the public report locally without model calls or network operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latexmk", default=shutil.which("latexmk"))
    parser.add_argument("--output", type=Path, default=Path("build"))
    args = parser.parse_args()
    if not args.latexmk:
        parser.error("latexmk is not on PATH; install a TeX distribution or pass --latexmk")
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = [root / "learning-from-hindsight.tex", root / "references.bib"]
    inputs += sorted((root / "figures").glob("*.png"))
    inputs += sorted((root / "figures").glob("*.tex"))
    result = subprocess.run(
        [args.latexmk, "-pdf", "-quiet", "-interaction=nonstopmode", "-halt-on-error",
         f"-outdir={output}", "learning-from-hindsight.tex"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    (output / "build-console.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit(f"Compilation failed; inspect {output / 'build-console.log'}")
    log = (output / "learning-from-hindsight.log").read_text()
    problems = re.findall(
        r"^.*(?:Overfull|Underfull|Missing character|undefined references|undefined citations|"
        r"LaTeX(?: Font)? Warning|Package \w+ Warning).*$", log, flags=re.MULTILINE,
    )
    if problems:
        raise SystemExit("The final LaTeX log needs review:\n" + "\n".join(problems))
    pdf = output / "learning-from-hindsight.pdf"
    manifest = {
        "title": "Scalable Meta-Learning from Hindsight",
        "author": "Angel Raychev",
        "publication_date": "2026-10-24",
        "historical_archive_commit": "0116d67f869453c2ba4086b4be9db963284da3f8",
        "input_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs},
        "output_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "output_bytes": pdf.stat().st_size,
        "new_model_calls": 0,
        "acceptance": "Compilation/log checks passed; rendered-page visual QA is separate",
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
