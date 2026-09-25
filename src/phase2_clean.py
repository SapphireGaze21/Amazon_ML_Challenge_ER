#!/usr/bin/env python3
"""
Phase 2 - Cleaning. Applies src/clean.py to all six sources (train + test identically).
Raw columns are preserved untouched; cleaned columns are added next to them.

Output per source  ->  {out-dir}/{split}_source{k}.parquet  with columns:
  entity_id, business_name, business_address, country        raw, unchanged
  name_orig, name_fold                                        None if name missing
  address_orig, address_fold, address_numcompounds            None if address missing
  country_norm                                                trim + casefold
  name_missing, address_missing, country_missing              bool

Plus {out-dir}/phase2_report.json and phase2_summary.md with an audit:
  missing counts (raw placeholder vs. empty-after-cleaning), most frequent full values
  (reveals placeholders we don't know about), which punctuation/symbol characters get
  removed and how often, how many records each rule touched, numeric-compound stats,
  and before/after examples.

Usage:
  python tests/test_clean.py              # unit tests first
  python src/phase2_clean.py --parquet-dir data_parquet --out-dir data_clean
"""

import argparse
import json
import os
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from clean import (_DOTTED_ABBR, _SPECIAL_LATIN, clean_address_chunk, clean_country,
                   clean_name_chunk)
from common import load_source

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def parallel_map(func, values: list, pool, chunk: int = 50_000) -> list:
    chunks = [values[i:i + chunk] for i in range(0, len(values), chunk)]
    out = []
    for part in pool.imap(func, chunks):  # imap keeps order
        out.extend(part)
    return out


def clean_source(df: pd.DataFrame, pool) -> pd.DataFrame:
    names = parallel_map(clean_name_chunk, df["business_name"].tolist(), pool)
    addrs = parallel_map(clean_address_chunk, df["business_address"].tolist(), pool)
    ctry = [clean_country(x) for x in df["country"].tolist()]
    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["name_orig"] = [n[0] for n in names]
    out["name_fold"] = [n[1] for n in names]
    out["name_missing"] = [n[2] for n in names]
    out["address_orig"] = [a[0] for a in addrs]
    out["address_fold"] = [a[1] for a in addrs]
    out["address_numcompounds"] = [a[2] for a in addrs]
    out["address_missing"] = [a[3] for a in addrs]
    out["country_norm"] = [c[0] for c in ctry]
    out["country_missing"] = [c[1] for c in ctry]
    return out


def write_table(df: pd.DataFrame, path_no_ext: Path) -> Path:
    try:
        import pyarrow  # noqa: F401
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    except ImportError:  # fallback so the script still runs without pyarrow
        p = path_no_ext.with_suffix(".tsv.gz")
        df.to_csv(p, sep="\t", index=False)
    return p


# ----------------------------------------------------------------------------- audit

def audit(raw: pd.DataFrame, cl: pd.DataFrame, sample_n: int, seed: int) -> dict:
    rep = {"rows": int(len(cl))}
    for field, raw_col in (("name", "business_name"), ("address", "business_address")):
        miss = cl[f"{field}_missing"]
        raw_blank = raw[raw_col].map(lambda x: x.strip() == "")
        rep[f"{field}_missing"] = {
            "total": int(miss.sum()),
            "raw_blank": int((miss & raw_blank).sum()),
            "placeholder_or_only_punctuation": int((miss & ~raw_blank).sum()),
            "placeholder_raw_values": {k: int(v) for k, v in
                                       raw.loc[miss & ~raw_blank, raw_col].value_counts()
                                       .head(20).items()},
        }
        # most frequent full values among NON-missing records: unknown placeholders show up here
        rep[f"{field}_top_values"] = [[k, int(v)] for k, v in
                                      cl.loc[~miss, f"{field}_fold"].value_counts()
                                      .head(30).items()]
    rep["country_norm_counts"] = {str(k): int(v) for k, v in
                                  cl["country_norm"].value_counts(dropna=False).items()}

    # rule-level stats on a sample
    s = raw.sample(n=min(sample_n, len(raw)), random_state=seed)
    removed, touched = Counter(), Counter()
    for col in ("business_name", "business_address"):
        for x in s[col]:
            if not x:
                continue
            nk = unicodedata.normalize("NFKC", x)
            for ch in nk:
                cat = unicodedata.category(ch)
                if cat[0] in "PS":
                    removed[ch] += 1
            if "&" in nk:
                touched[f"{col}|ampersand"] += 1
            if any(c in nk for c in "'’‘ʼ`´"):
                touched[f"{col}|apostrophe"] += 1
            if nk != x:
                touched[f"{col}|nfkc_changed"] += 1
            if not x.isascii():
                touched[f"{col}|non_ascii"] += 1
                if any(unicodedata.category(c) == "Nd" and not c.isascii() for c in nk):
                    touched[f"{col}|non_ascii_digits"] += 1
            if not x.isascii():
                low = nk.casefold()
                if any(c in _SPECIAL_LATIN for c in low) or any(
                        0x0300 <= ord(c) <= 0x036F for c in unicodedata.normalize("NFD", low)):
                    touched[f"{col}|latin_accent_stripped"] += 1
            if _DOTTED_ABBR.search(nk.casefold()):
                touched[f"{col}|dotted_abbreviation_joined"] += 1
    rep["sample_size"] = int(len(s))
    rep["punct_symbol_chars_top"] = [[f"{c} (U+{ord(c):04X} {unicodedata.name(c, '?')})", n]
                                     for c, n in removed.most_common(40)]
    rep["rule_touch_counts_sample"] = dict(sorted(touched.items()))

    comp = cl["address_numcompounds"].dropna()
    rep["numeric_compounds"] = {
        "addresses_with_compound": int(len(comp)),
        "share_of_nonmissing_addresses": round(len(comp) / max(int((~cl.address_missing).sum()), 1), 4),
        "top_by_country": {
            str(c): [[k, int(v)] for k, v in
                     g["address_numcompounds"].str.split(" ").explode().value_counts().head(15).items()]
            for c, g in cl[cl.address_numcompounds.notna()].groupby("country_norm")},
    }

    # before/after examples: records where cleaning changed something
    changed = cl[(cl["name_fold"].fillna("") != cl["business_name"].str.lower()) |
                 (cl["address_fold"].fillna("") != cl["business_address"].str.lower())]
    ex = changed.sample(n=min(25, len(changed)), random_state=seed)
    rep["examples_changed"] = [
        {"raw_name": r.business_name, "name_fold": r.name_fold,
         "raw_address": r.business_address, "address_fold": r.address_fold,
         "numcompounds": r.address_numcompounds if isinstance(r.address_numcompounds, str) else None}
        for r in ex.itertuples()]
    return rep


def write_summary(report: dict, path: Path):
    L = ["# Phase 2 summary\n"]
    for name, r in report["sources"].items():
        L.append(f"## {name}  ({r['rows']:,} rows)")
        for f in ("name", "address"):
            m = r[f"{f}_missing"]
            L.append(f"- {f} missing: {m['total']:,} (blank {m['raw_blank']:,}, placeholder/"
                     f"punctuation-only {m['placeholder_or_only_punctuation']:,}) "
                     f"placeholders seen: {m['placeholder_raw_values']}")
            L.append(f"- top {f} values: " + ", ".join(f"{v} ({n})" for v, n in r[f'{f}_top_values'][:12]))
        L.append(f"- countries: {r['country_norm_counts']}")
        L.append(f"- rule touches (sample of {r['sample_size']:,}): {r['rule_touch_counts_sample']}")
        L.append("- punctuation/symbols removed: " +
                 ", ".join(f"{c} ×{n}" for c, n in r["punct_symbol_chars_top"][:20]))
        nc = r["numeric_compounds"]
        L.append(f"- numeric compounds: {nc['addresses_with_compound']:,} addresses "
                 f"({nc['share_of_nonmissing_addresses']}); top: {nc['top_by_country']}")
        L.append("- examples:")
        for e in r["examples_changed"][:8]:
            L.append(f"    - `{e['raw_name']}` -> `{e['name_fold']}` | `{e['raw_address']}` -> "
                     f"`{e['address_fold']}` | nums: {e['numcompounds']}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--parquet-dir", default="data_parquet")
    ap.add_argument("--out-dir", default="data_clean")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--audit-sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "sources": {}}
    log(f"using {args.workers} worker processes")

    with Pool(args.workers) as pool:
        for split in args.splits.split(","):
            for k in (1, 2, 3):
                name = f"{split}_source{k}"
                raw = load_source(Path(args.data_dir), split, k, Path(args.parquet_dir))
                log(f"{name}: loaded {len(raw):,}")
                cl = clean_source(raw, pool)
                p = write_table(cl, out / name)
                log(f"{name}: cleaned -> {p.name}")
                report["sources"][name] = audit(raw, cl, args.audit_sample, args.seed)
                n, a = report["sources"][name]["name_missing"], report["sources"][name]["address_missing"]
                log(f"{name}: name missing {n['total']:,}, address missing {a['total']:,}")
                del raw, cl

    with open(out / "phase2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    write_summary(report, out / "phase2_summary.md")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
