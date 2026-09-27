#!/usr/bin/env python3
"""
Build the final submission zip in the layout the challenge requires:

  <team>_submission.zip
  ├── output/matching_results.tsv, output/candidate_pairs.tsv     (from --submission-dir)
  ├── code/business_entity_resolution/{src/, tests/, config/, README.md, requirements.txt}
  └── Documentation_template.md                                    (your filled-in document)

requirements.txt is regenerated with the EXACT installed versions (pinned) of the libraries used.
The two TSVs are checked again (every README rule) before zipping.

Usage (from the repo root):
  python src/make_package.py --team MyTeam --submission-dir output_v2 \
      --doc Documentation_template.md
"""

import argparse
import importlib.metadata as md
import shutil
import sys
import zipfile
from pathlib import Path

LIBS = ["pandas", "numpy", "pyarrow", "rapidfuzz", "scipy", "scikit-learn", "lightgbm", "xgboost"]


def pinned_requirements() -> str:
    lines = []
    for lib in LIBS:
        try:
            lines.append(f"{lib}=={md.version(lib)}")
        except md.PackageNotFoundError:
            lines.append(f"# {lib}: not installed in this environment")
    lines.append(f"# python {sys.version.split()[0]}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--submission-dir", default="output_v2",
                    help="folder holding the chosen matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--doc", default="Documentation_template.md", help="filled-in methodology document")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()

    sub = Path(a.submission_dir)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not (sub / f).exists():
            sys.exit(f"missing {sub / f}")
    if not Path(a.doc).exists():
        sys.exit(f"missing documentation file {a.doc}")
    if not a.skip_check:
        sys.path.insert(0, "src")
        import phase8_match as P8
        from common import read_table
        ids = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id"])["entity_id"]
        res = P8.check_submission(sub, a.block_dir, ids)
        print("submission checks:", res)
        if not res["all_ok"]:
            sys.exit("submission checks failed - not packaging")

    stage = Path("package_build")
    if stage.exists():
        shutil.rmtree(stage)
    code = stage / "code" / "business_entity_resolution"
    (stage / "output").mkdir(parents=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copyfile(sub / f, stage / "output" / f)
    for d in ("src", "tests", "config"):
        if Path(d).exists():
            shutil.copytree(d, code / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("README.md",):
        if Path(f).exists():
            shutil.copyfile(f, code / f)
    (code / "requirements.txt").write_text(pinned_requirements(), encoding="utf-8")
    shutil.copyfile(a.doc, stage / Path(a.doc).name)

    zpath = Path(f"{a.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    print((code / "requirements.txt").read_text())
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    print("\n".join(n for n in names if n.count("/") <= 2)[:3000])


if __name__ == "__main__":
    main()
