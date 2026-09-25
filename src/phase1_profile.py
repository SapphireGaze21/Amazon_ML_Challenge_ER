#!/usr/bin/env python3
"""
Phase 1 - Profiling (TRAIN split only; validation S1s are never looked at here).

Answers, with numbers, the questions that decide Phase 2 (preprocessing) and blocking:
  A. What do records look like per source/country?  (lengths, scripts, empty fields, digits)
  B. Is missingness informative?  (address-missing rate: matched vs unmatched S2/S3 records)
  C. How are an S1's matches spread across S2/S3?  (within-source duplicates)
  D. Do identical S2/S3 records belong to the same S1?
  E. How similar are true pairs vs random same-country pairs?  (name/address/numeric)
  F. What share of true pairs would simple blocking keys catch, at what cost?  (early recall
     ceiling for Tier 1 keys + rough candidates-per-S1 estimate)
  G. Which token substitutions occur in true pairs?  (abbreviations / typos, data-driven)
  H. Which leading/trailing name tokens are most common per country?  (legal suffixes)

Usage:
  python src/phase1_profile.py --data-dir dataset --parquet-dir data_parquet \
      --phase0-dir artifacts/phase0 --out-dir artifacts/phase1 --with-test
Main knobs: --n-s1 (sampled train S1 entities, default 50000), --neg-per-pos (default 2),
            --record-sample (records per source for section A/H, default 300000).
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import char_ngrams, fold, load_source, numeric_tokens, script_of, tokens

try:
    from rapidfuzz.fuzz import ratio as _rf_ratio

    def sim_ratio(a: str, b: str) -> float:
        return _rf_ratio(a, b) / 100.0
except ImportError:  # stdlib fallback (slower, very similar values)
    from difflib import SequenceMatcher

    def sim_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

CAPS = [10, 100, 1000, 10000, 100000, float("inf")]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def is_empty(x: str) -> bool:
    return x.strip().lower() in {"", "null", "none", "nan", "na", "n/a", "-", "--", "?"}


def jacc(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def qsummary(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(s)), "mean": round(float(s.mean()), 4),
            "p05": round(float(q[0.05]), 4), "p25": round(float(q[0.25]), 4),
            "p50": round(float(q[0.5]), 4), "p75": round(float(q[0.75]), 4),
            "p95": round(float(q[0.95]), 4),
            "share_eq_0": round(float((s == 0).mean()), 4),
            "share_eq_1": round(float((s == 1).mean()), 4)}


# ============================================================================ A. records

def profile_records(df: pd.DataFrame, n: int, seed: int) -> dict:
    samp = df.sample(n=min(n, len(df)), random_state=seed)
    out = {}
    for country, g in samp.groupby("country"):
        names = g["business_name"].map(fold)
        addrs = g["business_address"].map(fold)
        name_empty = g["business_name"].map(is_empty)
        addr_empty = g["business_address"].map(is_empty)
        ntok = names.map(lambda s: len(tokens(s)))
        atok = addrs.map(lambda s: len(tokens(s)))
        nums = addrs.map(numeric_tokens)
        out[country] = {
            "records_sampled": int(len(g)),
            "name_empty_rate": round(float(name_empty.mean()), 5),
            "address_empty_rate": round(float(addr_empty.mean()), 5),
            "name_chars": qsummary(names.str.len()),
            "name_tokens": qsummary(ntok),
            "address_chars": qsummary(addrs[~addr_empty].str.len()),
            "address_tokens": qsummary(atok[~addr_empty]),
            "address_has_digit_rate": round(float(nums[~addr_empty].map(bool).mean()), 4),
            "address_numeric_tokens": qsummary(nums[~addr_empty].map(len)),
            "name_script": {k: int(v) for k, v in names.map(script_of).value_counts().items()},
            "address_script": {k: int(v) for k, v in
                               addrs[~addr_empty].map(script_of).value_counts().items()},
        }
    return out


# ============================================================================ B/C/D. full-data

def missingness_vs_matched(s2, s3, matched_ids: set) -> dict:
    out = {}
    for tag, df in (("S2", s2), ("S3", s3)):
        m = df["entity_id"].isin(matched_ids)
        e = df["business_address"].map(is_empty)
        for country in sorted(df["country"].unique()):
            c = df["country"] == country
            out[f"{tag}|{country}"] = {
                "addr_empty_rate_matched": round(float(e[c & m].mean()), 5),
                "addr_empty_rate_unmatched": round(float(e[c & ~m].mean()), 5),
                "n_matched": int((c & m).sum()), "n_unmatched": int((c & ~m).sum()),
            }
    return out


def match_composition(gt_train: pd.DataFrame, train_s1: pd.Index) -> dict:
    per = gt_train.groupby(["s1_id", "matched_source"]).size().unstack(fill_value=0)
    per = per.reindex(train_s1, fill_value=0)
    for col in ("S2", "S3"):
        if col not in per:
            per[col] = 0

    def dist(s):
        b = s.clip(upper=5).map(lambda v: "5+" if v == 5 else str(v))
        return {k: int(v) for k, v in b.value_counts().sort_index().items()}

    has2, has3 = per["S2"] > 0, per["S3"] > 0
    return {
        "s1_entities": int(len(per)),
        "only_S2": int((has2 & ~has3).sum()), "only_S3": int((~has2 & has3).sum()),
        "both": int((has2 & has3).sum()), "none": int((~has2 & ~has3).sum()),
        "n_S2_matches_dist": dist(per["S2"]), "n_S3_matches_dist": dist(per["S3"]),
        "s1_with_2plus_from_same_source": int(((per["S2"] >= 2) | (per["S3"] >= 2)).sum()),
    }


def duplicate_content(s2, s3, owner: dict) -> dict:
    """Groups of identical (name, address, country) records: do they share the same S1?"""
    both = pd.concat([s2.assign(src="S2"), s3.assign(src="S3")], ignore_index=True)
    key = both["business_name"] + "\t" + both["business_address"] + "\t" + both["country"]
    dup = key.duplicated(keep=False)
    d = both.loc[dup, ["entity_id", "src"]].assign(key=key[dup].values)
    d["owner"] = d["entity_id"].map(owner)
    res = Counter()
    for _, g in d.groupby("key"):
        owners = g["owner"].dropna().unique()
        n_un = g["owner"].isna().sum()
        srcs = "S2+S3" if g["src"].nunique() > 1 else g["src"].iloc[0]
        if len(owners) == 0:
            kind = "all_unmatched"
        elif len(owners) == 1 and n_un == 0:
            kind = "same_s1"
        elif len(owners) == 1:
            kind = "same_s1_plus_unmatched"
        else:
            kind = "different_s1s"
        res[f"{srcs}|{kind}"] += 1
        res[f"ALL|{kind}"] += 1
    return {"records_in_dup_groups": int(dup.sum()),
            "groups": {k: int(v) for k, v in sorted(res.items())}}


# ============================================================================ DF tables

def build_df_tables(s23: pd.DataFrame):
    """Per country: number of S2+S3 records containing each name token / address token /
    numeric token. Used for 'rare token' probes and later blocking caps."""
    name_df, addr_df, num_df = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    pool = Counter()
    for i, (n, a, c) in enumerate(zip(s23["business_name"], s23["business_address"],
                                      s23["country"])):
        pool[c] += 1
        name_df[c].update(set(tokens(fold(n))))
        fa = fold(a)
        addr_df[c].update(set(tokens(fa)))
        num_df[c].update(set(numeric_tokens(fa)))
        if i and i % 2_000_000 == 0:
            log(f"  DF tables: {i:,} records")
    return name_df, addr_df, num_df, pool


# ============================================================================ E/F. pairs

def pair_features(r1, r2, c, name_df, addr_df, num_df) -> dict:
    n1, n2 = fold(r1[0]), fold(r2[0])
    a1, a2 = fold(r1[1]), fold(r2[1])
    t1, t2 = set(tokens(n1)), set(tokens(n2))
    u1, u2 = set(tokens(a1)), set(tokens(a2))
    m1, m2 = set(numeric_tokens(a1)), set(numeric_tokens(a2))
    a2_empty = is_empty(r2[1])
    sn, sa, sm = t1 & t2, u1 & u2, m1 & m2
    sm3 = {x for x in sm if len(x) >= 3}
    s1n, s2n = script_of(n1), script_of(n2)
    f = {
        "name_exact": n1.split() == n2.split(),
        "name_sorted_equal": sorted(t1) == sorted(t2) and bool(t1),
        "name_tok_jacc": jacc(t1, t2),
        "name_c3_jacc": jacc(char_ngrams(n1), char_ngrams(n2)),
        "name_ratio": sim_ratio(" ".join(n1.split()), " ".join(n2.split())),
        "name_sort_ratio": sim_ratio(" ".join(sorted(t1)), " ".join(sorted(t2))),
        "name_zero_tok": len(sn) == 0,
        "m_addr_empty": a2_empty,
        "addr_tok_jacc": np.nan if a2_empty else jacc(u1, u2),
        "addr_c3_jacc": np.nan if a2_empty else jacc(char_ngrams(a1), char_ngrams(a2)),
        "addr_zero_tok": (not a2_empty) and len(sa) == 0,
        "num_any_shared": bool(sm),
        "num3_shared": bool(sm3),
        "s1_has_num": bool(m1),
        "name_min_df": min((name_df[c][t] for t in sn), default=np.inf),
        "addr_min_df": min((addr_df[c][t] for t in sa), default=np.inf),
        "num3_min_df": min((num_df[c][t] for t in sm3), default=np.inf),
        "cross_field": bool((t1 & u2) | (u1 & t2)),
        "s1_name_script": s1n, "m_name_script": s2n,
        "cross_script": s1n != s2n and "none" not in (s1n, s2n),
    }
    return f


def coverage_table(df: pd.DataFrame) -> dict:
    out = {}
    for cap in CAPS:
        k = "inf" if cap == float("inf") else str(int(cap))
        name = df["name_min_df"] <= cap
        addr = df["addr_min_df"] <= cap
        num3 = df["num3_min_df"] <= cap
        union = df["name_exact"] | name | addr | num3
        out[k] = {"name_token": round(float(name.mean()), 5),
                  "addr_token": round(float(addr.mean()), 5),
                  "num3_token": round(float(num3.mean()), 5),
                  "union": round(float(union.mean()), 5)}
    out["exact_name"] = round(float(df["name_exact"].mean()), 5)
    return out


def est_candidates(neg: pd.DataFrame, pool: Counter) -> dict:
    """Random same-country negative hit rate x pool size ~ expected candidates per S1."""
    out = {}
    for cap in CAPS:
        k = "inf" if cap == float("inf") else str(int(cap))
        row = {}
        for key in ("name_token", "addr_token", "num3_token", "union"):
            tot, n = 0.0, 0
            for c, g in neg.groupby("country"):
                if key == "union":
                    hit = (g["name_exact"] | (g["name_min_df"] <= cap) |
                           (g["addr_min_df"] <= cap) | (g["num3_min_df"] <= cap))
                else:
                    col = {"name_token": "name_min_df", "addr_token": "addr_min_df",
                           "num3_token": "num3_min_df"}[key]
                    hit = g[col] <= cap
                tot += float(hit.mean()) * pool[c] * len(g)
                n += len(g)
            row[key] = round(tot / max(n, 1), 1)
        out[k] = row
    return out


# ============================================================================ G/H. tokens

def substitutions(pairs, field_idx: int, top: int = 300) -> dict:
    by_c = defaultdict(Counter)
    for r1, r2, c in pairs:
        a, b = set(tokens(fold(r1[field_idx]))), set(tokens(fold(r2[field_idx])))
        x, y = a - b, b - a
        if len(x) == 1 and len(y) == 1:
            by_c[c][(next(iter(x)), next(iter(y)))] += 1
    return {c: [[f"{k[0]} -> {k[1]}", v] for k, v in cnt.most_common(top)]
            for c, cnt in by_c.items()}


def edge_tokens(dfs: dict, n: int, seed: int, top: int = 60) -> dict:
    out = {}
    for label, df in dfs.items():
        samp = df.sample(n=min(n, len(df)), random_state=seed)
        for c, g in samp.groupby("country"):
            first, last = Counter(), Counter()
            for s in g["business_name"]:
                t = tokens(fold(s))
                if len(t) >= 2:
                    first[t[0]] += 1
                    last[t[-1]] += 1
            out[f"{label}|{c}"] = {"records": int(len(g)),
                                   "last": last.most_common(top),
                                   "first": first.most_common(top)}
    return out


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--parquet-dir", default="data_parquet")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--out-dir", default="artifacts/phase1")
    ap.add_argument("--n-s1", type=int, default=50000)
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--record-sample", type=int, default=300000)
    ap.add_argument("--with-test", action="store_true",
                    help="also profile unlabeled test sources (sections A and H only)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rep = {"args": vars(args)}

    # ---- load
    src = {}
    for k in (1, 2, 3):
        src[f"train_source{k}"] = load_source(Path(args.data_dir), "train", k, Path(args.parquet_dir))
        log(f"loaded train_source{k}: {len(src[f'train_source{k}']):,}")
    if args.with_test:
        for k in (1, 2, 3):
            src[f"test_source{k}"] = load_source(Path(args.data_dir), "test", k, Path(args.parquet_dir))
            log(f"loaded test_source{k}: {len(src[f'test_source{k}']):,}")
    s1, s2, s3 = src["train_source1"], src["train_source2"], src["train_source3"]
    gt = pd.read_csv(Path(args.phase0_dir) / "gt_long.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    split = pd.read_csv(Path(args.phase0_dir) / "s1_split.tsv", sep="\t", dtype=str,
                        keep_default_na=False)
    train_s1 = pd.Index(split.loc[split["split"] == "train", "s1_id"])
    gt_train = gt[gt["s1_id"].isin(set(train_s1))]
    owner = dict(zip(gt["matched_id"], gt["s1_id"]))  # each S2/S3 has <=1 owner (Phase 0)
    log(f"train S1: {len(train_s1):,}; train true pairs: {len(gt_train):,}")

    # ---- A
    rep["A_records"] = {name: profile_records(df, args.record_sample, args.seed)
                        for name, df in src.items()}
    log("A. record profiles done")

    # ---- B, C, D (full data). Missingness/duplicates use all S2/S3 labels: these are
    # properties of the pool, and S2/S3 are shared by train and val anyway.
    rep["B_missingness_vs_matched"] = missingness_vs_matched(s2, s3, set(gt["matched_id"]))
    rep["C_match_composition_train"] = match_composition(gt_train, train_s1)
    rep["D_duplicate_content"] = duplicate_content(s2, s3, owner)
    log("B/C/D done")

    # ---- DF tables over full train S2+S3
    s23 = pd.concat([s2, s3], ignore_index=True)
    name_df, addr_df, num_df, pool = build_df_tables(s23)
    rep["pool_size_by_country"] = dict(pool)
    log("DF tables done")

    # ---- sample pairs
    s1_rec = dict(zip(s1["entity_id"], zip(s1["business_name"], s1["business_address"],
                                            s1["country"])))
    s23_rec = dict(zip(s23["entity_id"], zip(s23["business_name"], s23["business_address"],
                                              s23["country"])))
    ids_by_c = {c: g["entity_id"].to_numpy() for c, g in s23.groupby("country")}
    sample_s1 = sorted(train_s1)
    rng.shuffle(sample_s1)
    sample_s1 = sample_s1[:args.n_s1]
    truth = gt_train[gt_train["s1_id"].isin(set(sample_s1))].groupby("s1_id")["matched_id"] \
        .apply(set).to_dict()
    nprng = np.random.default_rng(args.seed)

    rows, pos_pairs = [], []
    for sid in sample_s1:
        r1 = s1_rec[sid]
        c = r1[2]
        true_set = truth.get(sid, set())
        for mid in true_set:
            r2 = s23_rec[mid]
            f = pair_features(r1, r2, c, name_df, addr_df, num_df)
            f.update(label=1, s1_id=sid, other_id=mid, country=c, source=mid[:2])
            rows.append(f)
            pos_pairs.append((r1, r2, c))
        n_neg = max(1, len(true_set)) * args.neg_per_pos
        cand = ids_by_c[c][nprng.integers(0, len(ids_by_c[c]), size=n_neg * 2)]
        for mid in [m for m in cand if m not in true_set][:n_neg]:
            f = pair_features(r1, s23_rec[mid], c, name_df, addr_df, num_df)
            f.update(label=0, s1_id=sid, other_id=mid, country=c, source=mid[:2])
            rows.append(f)
    P = pd.DataFrame(rows)
    pos, neg = P[P.label == 1], P[P.label == 0]
    log(f"pairs: {len(pos):,} positive, {len(neg):,} negative")

    # ---- E. similarity distributions
    sims = ["name_tok_jacc", "name_c3_jacc", "name_ratio", "name_sort_ratio",
            "addr_tok_jacc", "addr_c3_jacc"]
    flags = ["name_exact", "name_sorted_equal", "name_zero_tok", "m_addr_empty",
             "addr_zero_tok", "num_any_shared", "num3_shared", "s1_has_num",
             "cross_field", "cross_script"]
    E = {}
    for lab, g in (("positive", pos), ("negative", neg)):
        E[lab] = {"similarity": {s: qsummary(g[s]) for s in sims},
                  "flags": {f: round(float(g[f].mean()), 5) for f in flags}}
        E[lab]["both_name_and_addr_zero_overlap"] = round(float(
            (g["name_zero_tok"] & (g["addr_zero_tok"] | g["m_addr_empty"])).mean()), 5)
    E["positive_by_country_source"] = {
        f"{c}|{s}": {"pairs": int(len(g)),
                     **{f: round(float(g[f].mean()), 4) for f in flags},
                     "name_ratio_p50": round(float(g["name_ratio"].median()), 4),
                     "addr_tok_jacc_p50": round(float(g["addr_tok_jacc"].median()), 4)}
        for (c, s), g in pos.groupby(["country", "source"])}
    E["positive_cross_script_combos"] = {
        k: int(v) for k, v in (pos.loc[pos.cross_script, "s1_name_script"] + " -> " +
                               pos.loc[pos.cross_script, "m_name_script"]).value_counts()
        .head(30).items()}
    rep["E_pair_similarity"] = E

    # ---- F. blocking probes
    F = {"positive_pair_coverage": coverage_table(pos),
         "positive_coverage_by_country": {c: coverage_table(g) for c, g in pos.groupby("country")},
         "positive_coverage_cross_script": coverage_table(pos[pos.cross_script])
         if pos.cross_script.any() else {},
         "positive_coverage_addr_missing": coverage_table(pos[pos.m_addr_empty])
         if pos.m_addr_empty.any() else {},
         "est_candidates_per_s1_single_key": est_candidates(neg, pool)}
    # entity-level: share of sampled S1s (with >=1 match) whose matches are ALL covered
    ent = {}
    for cap in (1000, 10000, 100000):
        cov = (pos["name_exact"] | (pos["name_min_df"] <= cap) | (pos["addr_min_df"] <= cap) |
               (pos["num3_min_df"] <= cap))
        per = cov.groupby(pos["s1_id"]).mean()
        ent[str(cap)] = {"s1_all_matches_covered": round(float((per == 1).mean()), 4),
                         "mean_per_s1_recall": round(float(per.mean()), 4)}
    F["entity_level_union"] = ent
    rep["F_blocking_probes"] = F
    log("E/F done")

    # ---- G, H
    rep["G_substitutions_name"] = substitutions(pos_pairs, 0)
    rep["G_substitutions_address"] = substitutions(pos_pairs, 1)
    edge_src = {k: v for k, v in src.items()}
    rep["H_name_edge_tokens"] = edge_tokens(edge_src, args.record_sample, args.seed)
    log("G/H done")

    # ---- dumps for manual reading
    def attach(df):
        d = df.copy()
        d["s1_name"] = d.s1_id.map(lambda i: s1_rec[i][0])
        d["s1_address"] = d.s1_id.map(lambda i: s1_rec[i][1])
        d["other_name"] = d.other_id.map(lambda i: s23_rec[i][0])
        d["other_address"] = d.other_id.map(lambda i: s23_rec[i][1])
        return d

    cols = ["s1_id", "other_id", "country", "source", "s1_name", "other_name", "s1_address",
            "other_address", "name_ratio", "addr_tok_jacc", "name_min_df", "addr_min_df",
            "num3_min_df", "cross_script", "cross_field"]
    unc = pos[~(pos["name_exact"] | (pos["name_min_df"] <= 10000) |
                (pos["addr_min_df"] <= 10000) | (pos["num3_min_df"] <= 10000))]
    attach(unc.sample(n=min(500, len(unc)), random_state=1))[cols] \
        .to_csv(out / "uncovered_true_pairs_cap10000.tsv", sep="\t", index=False)
    z = pos[pos.name_zero_tok]
    attach(z.sample(n=min(300, len(z)), random_state=1))[cols] \
        .to_csv(out / "zero_name_overlap_true_pairs.tsv", sep="\t", index=False)
    x = pos[pos.cross_script]
    attach(x.sample(n=min(300, len(x)), random_state=1))[cols] \
        .to_csv(out / "cross_script_true_pairs.tsv", sep="\t", index=False)
    if _has_pyarrow():
        P.to_parquet(out / "pair_features.parquet", index=False)
    else:
        P.to_csv(out / "pair_features.tsv.gz", sep="\t", index=False)

    with open(out / "phase1_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=_json_default)
    write_summary(rep, out / "phase1_summary.md")
    log(f"done -> {out}")


def _has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_summary(rep: dict, path: Path):
    """Compact human-readable summary - this is the file to paste back into the chat."""
    L = ["# Phase 1 summary\n"]
    L.append("## Pool size (train S2+S3) by country\n" + json.dumps(rep["pool_size_by_country"]))
    E = rep["E_pair_similarity"]
    L.append("\n## True pairs vs random same-country pairs (flags = share of pairs)")
    for lab in ("positive", "negative"):
        L.append(f"\n**{lab}**: " + json.dumps(E[lab]["flags"]))
        L.append(f"both name & address zero overlap: {E[lab]['both_name_and_addr_zero_overlap']}")
        for s, q in E[lab]["similarity"].items():
            L.append(f"- {s}: p05={q.get('p05')} p25={q.get('p25')} p50={q.get('p50')} "
                     f"p75={q.get('p75')} p95={q.get('p95')} zero={q.get('share_eq_0')}")
    L.append("\n## Positives by country|source\n```\n" +
             json.dumps(E["positive_by_country_source"], indent=1) + "\n```")
    L.append("\n## Cross-script combos (positives)\n" + json.dumps(E["positive_cross_script_combos"]))
    F = rep["F_blocking_probes"]
    L.append("\n## Blocking probe coverage of true pairs (key: max doc-freq cap)\n```\n" +
             json.dumps(F["positive_pair_coverage"], indent=1) + "\n```")
    L.append("\n## Est. candidates per S1 (single key, from random negatives)\n```\n" +
             json.dumps(F["est_candidates_per_s1_single_key"], indent=1) + "\n```")
    L.append("\n## Entity-level union coverage\n" + json.dumps(F["entity_level_union"]))
    L.append("\n## Coverage by country\n```\n" +
             json.dumps(F["positive_coverage_by_country"], indent=1) + "\n```")
    if F.get("positive_coverage_cross_script"):
        L.append("\n## Coverage, cross-script pairs\n" + json.dumps(F["positive_coverage_cross_script"]))
    if F.get("positive_coverage_addr_missing"):
        L.append("\n## Coverage, matched address missing\n" + json.dumps(F["positive_coverage_addr_missing"]))
    L.append("\n## Missingness vs matched\n```\n" +
             json.dumps(rep["B_missingness_vs_matched"], indent=1) + "\n```")
    L.append("\n## Match composition (train S1)\n" + json.dumps(rep["C_match_composition_train"]))
    L.append("\n## Duplicate-content groups\n" + json.dumps(rep["D_duplicate_content"]))
    L.append("\n## Top name substitutions (first 25 per country)")
    for c, lst in rep["G_substitutions_name"].items():
        L.append(f"- {c}: " + ", ".join(f"{a} ({n})" for a, n in lst[:25]))
    L.append("\n## Top address substitutions (first 25 per country)")
    for c, lst in rep["G_substitutions_address"].items():
        L.append(f"- {c}: " + ", ".join(f"{a} ({n})" for a, n in lst[:25]))
    L.append("\n## Most common LAST name tokens (first 20)")
    for k, v in rep["H_name_edge_tokens"].items():
        L.append(f"- {k}: " + ", ".join(f"{t} ({n})" for t, n in v["last"][:20]))
    L.append("\n## Record profiles: scripts\n")
    for sname, per_c in rep["A_records"].items():
        for c, d in per_c.items():
            L.append(f"- {sname}|{c}: name={d['name_script']} addr_empty={d['address_empty_rate']} "
                     f"addr_has_digit={d['address_has_digit_rate']}")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
