#!/usr/bin/env python3
"""
Phase 0 - Setup and split for the Business Entity Resolution challenge.

What it does
  1. Loads train/test source files (tab-separated, everything as raw strings,
     no automatic NaN conversion) and verifies the parse (row counts vs line counts).
  2. Audits each source: row counts, duplicate entity_ids, wrong ID prefixes,
     duplicate content (same name+address+country under different IDs),
     placeholder/empty counts, per-country counts.
  3. Parses train_ground_truth.tsv into a long table (s1_id, matched_id, matched_source)
     and runs integrity checks.
  4. Makes a reproducible 80/20 split of S1 IDs, stratified by country x match-count bucket.
     S2/S3 are NOT split.
  5. Computes the country-mismatch statistic on the TRAIN split only.

Outputs (in --out-dir)
  phase0_report.json       every number computed below, machine-readable
  gt_long.tsv              one row per (s1_id, matched_id) true pair (all S1s)
  s1_split.tsv             s1_id, split (train/val), country_norm, n_matches, match_bucket
  country_mismatch_pairs.tsv  train-split true pairs whose countries differ (for inspection)

Usage
  python src/phase0_setup_split.py --data-dir dataset --out-dir artifacts/phase0
  (expects dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv,
   dataset/test/test_source{1,2,3}.tsv)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
# Strings that mean "no value" once lowercased and stripped. Phase 0 only COUNTS them;
# the actual null standardisation happens in Phase 2.
PLACEHOLDERS = {"", "null", "none", "nan", "na", "n/a", "-", "--", "?"}


# ----------------------------------------------------------------------------- loading

def count_data_lines(path: Path) -> int:
    """Number of non-empty lines after the header, read as raw bytes."""
    n = 0
    with open(path, "rb") as f:
        next(f, None)  # header
        for line in f:
            if line.strip():
                n += 1
    return n


def read_tsv(path: Path, expected_cols) -> tuple[pd.DataFrame, dict]:
    """
    Read a TSV with every value as a raw string. Tries default quoting first; if the
    row count doesn't match the file's line count (a stray quote character can swallow
    lines), retries with quoting disabled and keeps whichever matches.
    """
    info = {"path": str(path)}
    n_lines = count_data_lines(path)
    info["data_lines_in_file"] = n_lines

    common = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                  encoding="utf-8")
    try:
        df = pd.read_csv(path, **common)
        info["quoting"] = "default"
    except pd.errors.ParserError as e:  # e.g. unclosed quote at the start of a field
        df = None
        info["default_quoting_error"] = str(e)[:200]
    if df is None or len(df) != n_lines:
        df_nq = pd.read_csv(path, quoting=csv.QUOTE_NONE, **common)
        info["rows_default_quoting"] = None if df is None else len(df)
        info["rows_no_quoting"] = len(df_nq)
        if len(df_nq) == n_lines or df is None:
            df = df_nq
            info["quoting"] = "QUOTE_NONE"
        if len(df) != n_lines:
            info["warning"] = "row count matches file line count under neither quoting mode"

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    extra = [c for c in df.columns if c not in expected_cols]
    if extra:
        info["extra_columns"] = extra
    # Plain Python-string object columns: behaves the same on pandas 2.x and 3.x.
    df = df[expected_cols].astype(object)
    info["rows_loaded"] = len(df)
    return df, info


# ----------------------------------------------------------------------------- helpers

def norm_country(s: pd.Series) -> pd.Series:
    """Minimal, value-agnostic normalisation: trim + casefold. No value mapping."""
    return s.map(lambda x: x.strip().casefold())


def is_placeholder(s: pd.Series) -> pd.Series:
    return s.map(lambda x: x.strip().lower() in PLACEHOLDERS)


def audit_source(df: pd.DataFrame, prefix: str) -> dict:
    ids = df["entity_id"].map(str.strip)
    out = {
        "rows": int(len(df)),
        "duplicate_entity_ids": int(ids.duplicated().sum()),
        "ids_with_wrong_prefix": int((~ids.str.startswith(prefix)).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        # same content under different IDs - relevant for multi-match S1s later
        "duplicate_content_rows": int(
            df[["business_name", "business_address", "country"]].duplicated().sum()),
        "placeholder_or_empty": {c: int(is_placeholder(df[c]).sum())
                                 for c in ["business_name", "business_address", "country"]},
        "literal_null_string": {c: int((df[c].map(lambda x: x.strip().lower()) == "null").sum())
                                for c in ["business_name", "business_address", "country"]},
    }
    raw_counts = df["country"].value_counts(dropna=False)
    out["country_counts_raw"] = {str(k): int(v) for k, v in raw_counts.items()}
    cn = norm_country(df["country"])
    out["country_counts_norm"] = {str(k): int(v) for k, v in cn.value_counts().items()}
    # raw spellings that collapse to one normalised value (e.g. "India", "india ")
    variants = (pd.DataFrame({"raw": df["country"], "norm": cn})
                .drop_duplicates().groupby("norm")["raw"].apply(list))
    out["country_raw_variants"] = {k: v for k, v in variants.items() if len(v) > 1}
    return out


def match_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


# ----------------------------------------------------------------------------- ground truth

def parse_ground_truth(gt: pd.DataFrame, s1_ids: set, s2_ids: set, s3_ids: set):
    checks = {}
    gt = gt.copy()
    gt["source1_entity_id"] = gt["source1_entity_id"].map(str.strip)

    checks["gt_rows"] = int(len(gt))
    checks["gt_duplicate_s1_rows"] = int(gt["source1_entity_id"].duplicated().sum())
    gt_s1 = set(gt["source1_entity_id"])
    checks["gt_s1_not_in_source1"] = sorted(gt_s1 - s1_ids)[:50]
    checks["gt_s1_not_in_source1_count"] = len(gt_s1 - s1_ids)
    checks["source1_ids_missing_from_gt_count"] = len(s1_ids - gt_s1)

    rows, within_list_dupes = [], 0
    for s1, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        toks = [t.strip() for t in lst.split(",") if t.strip()]
        within_list_dupes += len(toks) - len(set(toks))
        for t in dict.fromkeys(toks):  # de-dup, keep order
            rows.append((s1, t))
    checks["within_list_duplicate_ids"] = within_list_dupes

    long = pd.DataFrame(rows, columns=["s1_id", "matched_id"])
    long["matched_source"] = long["matched_id"].str[:3].str.rstrip("-")
    checks["true_pairs"] = int(len(long))
    checks["true_pairs_by_source"] = {k: int(v) for k, v in
                                      long["matched_source"].value_counts().items()}

    bad_prefix = ~long["matched_id"].str.startswith(("S2-", "S3-"))
    checks["matched_ids_bad_prefix"] = int(bad_prefix.sum())
    unknown = ~long["matched_id"].isin(s2_ids | s3_ids)
    checks["matched_ids_not_in_s2_s3"] = int(unknown.sum())
    checks["matched_ids_not_in_s2_s3_examples"] = long.loc[unknown, "matched_id"].head(20).tolist()

    # Is any S2/S3 record claimed by more than one S1? (S1 is supposedly deduplicated.)
    multi = long.groupby("matched_id")["s1_id"].nunique()
    multi = multi[multi > 1]
    checks["matched_ids_under_multiple_s1"] = int(len(multi))
    checks["matched_ids_under_multiple_s1_examples"] = {
        k: long.loc[long["matched_id"] == k, "s1_id"].tolist() for k in multi.index[:10]}

    # Share of S2/S3 records that are matched to some S1 at all
    checks["share_s2_matched"] = round(long.loc[long.matched_source == "S2", "matched_id"]
                                       .nunique() / max(len(s2_ids), 1), 4)
    checks["share_s3_matched"] = round(long.loc[long.matched_source == "S3", "matched_id"]
                                       .nunique() / max(len(s3_ids), 1), 4)
    return long, checks


# ----------------------------------------------------------------------------- split

def stratified_split(s1_meta: pd.DataFrame, val_frac: float, seed: int) -> pd.Series:
    """
    Deterministic stratified split by (country_norm, match_bucket).
    IDs are sorted before shuffling so the result does not depend on file row order.
    In each stratum, round(n * val_frac) IDs go to validation; for tiny strata where that
    rounds to 0, the stratum's IDs are assigned by a seeded coin flip with p = val_frac.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=s1_meta.index, dtype=object)
    strata = s1_meta.groupby(["country_norm", "match_bucket"], sort=True)
    for _, grp in strata:
        ids = np.array(sorted(grp.index))
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_frac))
        if n_val == 0:
            val_ids = ids[rng.random(len(ids)) < val_frac]
        else:
            val_ids = ids[:n_val]
        split.loc[val_ids] = "val"
    return split


# ----------------------------------------------------------------------------- country check

def country_mismatch(long_train: pd.DataFrame, s1: pd.DataFrame,
                     s23: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    s1c = s1.set_index("entity_id")["country"]
    s23c = s23.set_index("entity_id")["country"]
    df = long_train.copy()
    df["s1_country_raw"] = df["s1_id"].map(s1c)
    df["matched_country_raw"] = df["matched_id"].map(s23c)
    df = df.dropna(subset=["s1_country_raw", "matched_country_raw"])  # unknown IDs excluded
    df["s1_country"] = norm_country(df["s1_country_raw"])
    df["matched_country"] = norm_country(df["matched_country_raw"])

    either_missing = is_placeholder(df["s1_country_raw"]) | is_placeholder(df["matched_country_raw"])
    raw_diff = df["s1_country_raw"] != df["matched_country_raw"]
    norm_diff = df["s1_country"] != df["matched_country"]
    n = len(df)

    res = {
        "train_true_pairs_checked": int(n),
        "mismatch_raw_string": int(raw_diff.sum()),
        "mismatch_after_trim_casefold": int(norm_diff.sum()),
        "mismatch_rate_after_trim_casefold": round(float(norm_diff.mean()) if n else 0.0, 6),
        "pairs_with_country_missing_on_either_side": int(either_missing.sum()),
        "mismatch_excluding_missing": int((norm_diff & ~either_missing).sum()),
        "by_s1_country": {},
        "by_matched_source": {},
        "mismatch_country_pairs": {},
    }
    for c, g in df.groupby("s1_country"):
        d = g["s1_country"] != g["matched_country"]
        res["by_s1_country"][c] = {"pairs": int(len(g)), "mismatches": int(d.sum())}
    for src, g in df.groupby("matched_source"):
        d = g["s1_country"] != g["matched_country"]
        res["by_matched_source"][src] = {"pairs": int(len(g)), "mismatches": int(d.sum())}
    mm = df[norm_diff]
    combo = (mm["s1_country"] + " -> " + mm["matched_country"]).value_counts()
    res["mismatch_country_pairs"] = {k: int(v) for k, v in combo.items()}

    # Also count S1 entities affected (a hard filter would hurt these entities' recall)
    res["train_s1_entities_with_any_mismatch"] = int(mm["s1_id"].nunique())
    res["train_s1_entities_with_matches"] = int(df["s1_id"].nunique())
    return res, mm


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="artifacts/phase0")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-test", action="store_true", help="don't load/audit test sources")
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "pandas": pd.__version__, "numpy": np.__version__}

    # ---- 1-2. load + audit
    src, load_info, audits = {}, {}, {}
    splits = ["train"] + ([] if args.skip_test else ["test"])
    for split in splits:
        for k in (1, 2, 3):
            name = f"{split}_source{k}"
            df, info = read_tsv(data / split / f"{name}.tsv", EXPECTED_COLS)
            df["entity_id"] = df["entity_id"].map(str.strip)
            src[name], load_info[name] = df, info
            audits[name] = audit_source(df, f"S{k}-")
            print(f"[load] {name}: {len(df):,} rows (quoting={info['quoting']})"
                  + (f"  WARNING: {info['warning']}" if "warning" in info else ""))
    report["load"] = load_info
    report["audit"] = audits

    # ---- 3. ground truth
    gt, gt_info = read_tsv(data / "train" / "train_ground_truth.tsv", GT_COLS)
    report["load"]["train_ground_truth"] = gt_info
    s1, s2, s3 = src["train_source1"], src["train_source2"], src["train_source3"]
    long, gt_checks = parse_ground_truth(gt, set(s1.entity_id), set(s2.entity_id),
                                         set(s3.entity_id))
    report["ground_truth"] = gt_checks
    long.to_csv(out / "gt_long.tsv", sep="\t", index=False)
    print(f"[gt] {gt_checks['true_pairs']:,} true pairs; "
          f"{gt_checks['matched_ids_under_multiple_s1']} S2/S3 IDs claimed by >1 S1; "
          f"{gt_checks['matched_ids_not_in_s2_s3']} matched IDs not found in S2/S3")

    # ---- 4. split S1 (every S1 in source1, including any missing from the GT file -> 0 matches)
    n_matches = long.groupby("s1_id").size()
    s1_meta = pd.DataFrame(index=pd.Index(s1["entity_id"], name="s1_id"))
    s1_meta = s1_meta[~s1_meta.index.duplicated()]
    s1_meta["country_norm"] = norm_country(s1.drop_duplicates("entity_id")
                                           .set_index("entity_id")["country"]).reindex(s1_meta.index)
    s1_meta["n_matches"] = n_matches.reindex(s1_meta.index).fillna(0).astype(int)
    s1_meta["match_bucket"] = s1_meta["n_matches"].map(match_bucket)
    s1_meta["split"] = stratified_split(s1_meta, args.val_frac, args.seed)
    s1_meta.reset_index()[["s1_id", "split", "country_norm", "n_matches", "match_bucket"]] \
        .to_csv(out / "s1_split.tsv", sep="\t", index=False)

    tab = (s1_meta.groupby(["country_norm", "match_bucket", "split"]).size()
           .unstack("split", fill_value=0))
    report["split"] = {
        "train_s1": int((s1_meta.split == "train").sum()),
        "val_s1": int((s1_meta.split == "val").sum()),
        "val_share": round(float((s1_meta.split == "val").mean()), 4),
        "strata": {f"{c}|{b}": {k: int(v) for k, v in row.items()}
                   for (c, b), row in tab.iterrows()},
        "train_true_pairs": int(long.s1_id.isin(s1_meta.index[s1_meta.split == "train"]).sum()),
        "val_true_pairs": int(long.s1_id.isin(s1_meta.index[s1_meta.split == "val"]).sum()),
    }
    print(f"[split] train={report['split']['train_s1']:,} val={report['split']['val_s1']:,} "
          f"(val share {report['split']['val_share']})")
    print(tab.to_string())

    # ---- 5. country mismatch, TRAIN split only
    train_ids = set(s1_meta.index[s1_meta.split == "train"])
    long_train = long[long.s1_id.isin(train_ids)]
    s23 = pd.concat([s2, s3], ignore_index=True)
    cm, mm = country_mismatch(long_train, s1, s23)
    report["country_mismatch_train"] = cm
    mm.to_csv(out / "country_mismatch_pairs.tsv", sep="\t", index=False)
    print(f"[country] {cm['mismatch_after_trim_casefold']:,} / {cm['train_true_pairs_checked']:,} "
          f"train true pairs differ in country (rate {cm['mismatch_rate_after_trim_casefold']}); "
          f"{cm['pairs_with_country_missing_on_either_side']} have a missing country")

    with open(out / "phase0_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out}/phase0_report.json, gt_long.tsv, s1_split.tsv, "
          f"country_mismatch_pairs.tsv")


if __name__ == "__main__":
    sys.exit(main())
