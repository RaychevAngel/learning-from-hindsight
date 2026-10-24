"""Build a local presentation artifact and check references; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build"))
    args = parser.parse_args()
    source = args.source.resolve()
    root = source.parent
    output = args.output.resolve()
    executable = shutil.which("latexmk")
    if not executable:
        parser.error("latexmk is unavailable; use a local TeX distribution")
    output.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", f"-outdir={output}", source.name],
        cwd=root, text=True, capture_output=True, check=False,
    )
    (output / "build-console.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise SystemExit(f"Compilation failed; inspect {output / 'build-console.log'}")
    log = (output / f"{source.stem}.log").read_text(errors="replace")
    failures = re.findall(
        r"^.*(?:undefined references|undefined citations|Citation .+ undefined|"
        r"Reference .+ undefined|Missing character|Overfull|"
        r"multiply[- ]defined|destination with the same identifier).*$",
        log, flags=re.MULTILINE | re.IGNORECASE,
    )
    if failures:
        raise SystemExit("Build acceptance failed:\n" + "\n".join(failures))
    inputs = sorted(path for path in root.rglob("*")
                    if path.is_file() and output not in path.parents
                    and path.suffix in {".tex", ".sty", ".bib", ".png", ".svg"})
    pdf = output / f"{source.stem}.pdf"
    manifest = {
        "source": source.name,
        "style_version": "angel-publication 1.0 (2026-10-24)",
        "input_sha256": {str(path.relative_to(root)):
                         hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "acceptance": "Compilation and reference checks passed; visual QA is separate",
        "new_experiments": 0,
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
