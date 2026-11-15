#!/usr/bin/env python
# [NEW] Written for this project (CS512 Option 2); not part of upstream DINO-WM.
"""Build the Canvas submission zip.

usage: python scripts/make_submission.py [--extra submission_extra] [--out cs512_project.zip]

Contents (zip root = repo root):
  code       every file git tracks or would track (git ls-files --cached --others
             --exclude-standard), minus logs, patches, caches and plan_targets.pkl
  outputs/   the main.py outputs (gitignored, so added explicitly); run `python main.py all`
             (and `demo`) first
  --extra    a folder copied to the zip root as is: put the report PDF, the presentation
             and papers/ there
Not included: datasets, the checkpoint (links in data.txt / model.txt), plan_outputs/.
"""
import argparse
import fnmatch
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = ["p[0-9]*.log", "*.patch", "*.pyc", "*/__pycache__/*", "*plan_targets.pkl", "*.zip",
           ".vscode/*", "plan_outputs/*"]
REQUIRED = ["README.md", "main.py", "requirements.txt", "data.txt", "model.txt"]


def excluded(path):
    return any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="submission_extra",
                    help="folder with report.pdf, presentation, papers/ (copied to the zip root)")
    ap.add_argument("--out", default="cs512_project.zip")
    a = ap.parse_args()

    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    extra = Path(a.extra)
    extra_rel = str(extra.resolve().relative_to(ROOT)) if extra.resolve().is_relative_to(ROOT) else None
    files = [f for f in files if f and (ROOT / f).is_file() and not excluded(f)
             and not (extra_rel and f.startswith(extra_rel + "/"))]
    outputs = ROOT / "outputs"
    if not (outputs / "key_metrics.md").is_file():
        sys.exit("outputs/ is missing or incomplete: run `python main.py all` first")
    files += [str(p.relative_to(ROOT)) for p in sorted(outputs.rglob("*"))
              if p.is_file() and not excluded(str(p.relative_to(ROOT)))]
    missing = [f for f in REQUIRED if f not in files]
    if missing:
        sys.exit(f"missing required files: {missing}")

    entries = [(ROOT / f, f) for f in sorted(set(files))]
    if extra.is_dir():
        entries += [(p, str(p.relative_to(extra))) for p in sorted(extra.rglob("*")) if p.is_file()]
    else:
        print(f"note: no {extra}/ folder; report, presentation and papers are NOT in the zip")
    names = [n for _, n in entries]
    for want in ("*.pdf",):
        if not any(fnmatch.fnmatch(n, want) and "/" not in n for n in names):
            print("note: no PDF at the zip root (report.pdf expected there)")
    if not (outputs / "demo" / "demo_summary.md").is_file():
        print("note: outputs/demo/ is missing (run `python main.py demo` on a GPU machine)")

    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in entries:
            z.write(src, name)
    big = sorted(((src.stat().st_size, n) for src, n in entries), reverse=True)[:5]
    print(f"wrote {out}: {len(entries)} files, {out.stat().st_size / 2**20:.1f} MB compressed")
    print("largest files:", ", ".join(f"{n} ({s / 2**20:.1f} MB)" for s, n in big))


if __name__ == "__main__":
    main()
