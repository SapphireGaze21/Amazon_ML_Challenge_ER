#!/usr/bin/env python3
"""
Phase 5 - Blocking (candidate generation).

Passes, unioned per S1 (country-partitioned):
  exact_sorted / exact_concat / exact_alias   hash joins on the Phase 3 exact keys
  sparse_comb   top-K by  w_name*cos(name features) + w_addr*cos(address features)
  sparse_name   top-K by the name score alone      (matches whose address is missing/different)
  sparse_addr   top-K by the address score alone   (matches whose name is different: DBA etc.)
  char_name     top-K by char n-gram TF-IDF cosine on the glued name (typos, squashed names)
Candidates are merged, ranked by priority (see blocking.merge_candidates) and trimmed to a
per-S1 budget.

Splits
  --split val    S1 = validation split, pool = train S2+S3  -> the numbers we report
  --split train  S1 = train split,      pool = train S2+S3  -> candidates to train the matcher
  --split test   S1 = test S1,          pool = test S2+S3   -> writes output candidate_pairs.tsv
Use --s1-sample for quick experiments on a subset of S1.

Outputs
  {out-dir}/{run}/report.json, summary.md, missed_true_pairs.tsv      (val/train)
  {cand-dir}/{run}/{country}_{chunk}.parquet   candidates with passes, scores, priority, rank
  {out-dir}/{run}/candidate_pairs.tsv          (test) README format, final budget applied

Usage
  python src/phase5_blocking.py --split val --s1-sample 50000      # quick experiment
  python src/phase5_blocking.py --split val                        # full validation
  python src/phase5_blocking.py --split test                       # submission candidates
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import blocking as B
from common import read_table, write_table

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def build_features(df: pd.DataFrame, workers: int, chunk: int = 50_000):
    rows = list(zip(df.name_core, df.name_phon, df.address_tokens_norm, df.address_nums,
                    df.address_numcompounds))
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    nl, ni, al, ai = [], [], [], []
    with Pool(workers) as p:
        for a, b, c, d in p.imap(B.feature_chunk, parts):
            nl.append(a); ni.append(b); al.append(c); ai.append(d)  # noqa: E702
    return B.csr_from_parts(nl, ni), B.csr_from_parts(al, ai)


def f05(r):
    return 1.25 * r / (0.25 + r) if r > 0 else 0.0


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train", "test"], default="val")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--out-dir", default="artifacts/phase5")
    ap.add_argument("--cand-dir", default="data_cand")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--s1-sample", type=int, default=0, help="0 = all S1 of the split")
    ap.add_argument("--countries", default=None, help="comma list, default all")
    ap.add_argument("--max-df", type=int, default=5000, help="drop hashed features more common than this")
    ap.add_argument("--k-comb", type=int, default=100)
    ap.add_argument("--k-name", type=int, default=50)
    ap.add_argument("--k-addr", type=int, default=50)
    ap.add_argument("--k-char", type=int, default=50)
    ap.add_argument("--no-char", action="store_true")
    ap.add_argument("--char-ngram", type=int, default=3)
    ap.add_argument("--char-max-df", type=int, default=5000)
    ap.add_argument("--w-name", type=float, default=1.0)
    ap.add_argument("--w-addr", type=float, default=1.0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--exact-max-block", type=int, default=5000)
    ap.add_argument("--budget-max", type=int, default=200, help="candidates kept per S1 on disk")
    ap.add_argument("--budgets", default="10,25,50,100,200,300")
    ap.add_argument("--final-budget", type=int, default=200, help="budget for candidate_pairs.tsv / misses")
    ap.add_argument("--chunk", type=int, default=50_000, help="S1 rows merged/written at a time")
    ap.add_argument("--sub", type=int, default=2_000, help="S1 rows per worker task")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()

    run = a.run_name or f"{a.split}" + (f"_s{a.s1_sample}" if a.s1_sample else "_full")
    out_dir, cand_dir = Path(a.out_dir) / run, Path(a.cand_dir) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in a.budgets.split(",")]
    labeled = a.split != "test"

    # ---- load
    src = "test" if a.split == "test" else "train"
    s1 = read_table(Path(a.block_dir) / f"{src}_source1")
    if a.split in ("val", "train"):
        s1 = s1[s1["split"] == a.split]
    if a.s1_sample and a.s1_sample < len(s1):
        s1 = s1.sample(n=a.s1_sample, random_state=a.seed)
    pool = pd.concat([read_table(Path(a.block_dir) / f"{src}_source2"),
                      read_table(Path(a.block_dir) / f"{src}_source3")], ignore_index=True)
    log(f"S1: {len(s1):,}  pool: {len(pool):,}")
    truth = {}
    if labeled:
        gt = pd.read_csv(Path(a.phase0_dir) / "gt_long.tsv", sep="\t", dtype=str, keep_default_na=False)
        gt = gt[gt.s1_id.isin(set(s1.entity_id))]
        truth = gt.groupby("s1_id")["matched_id"].apply(list).to_dict()

    countries = sorted(s1.country_norm.dropna().unique())
    if a.countries:
        countries = [c for c in countries if c in a.countries.split(",")]
    tsv = None
    if a.split == "test":
        tsv = open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8")
        tsv.write("source1_entity_id\tcandidate_entity_ids\n")

    true_rows, s1_rows, build_info = [], [], {}
    for c in countries:
        s1c = s1[s1.country_norm == c].reset_index(drop=True)
        pc = pool[pool.country_norm == c].reset_index(drop=True)
        log(f"== {c}: S1 {len(s1c):,}  pool {len(pc):,}")

        # ---- features
        B_n, B_a = build_features(pc, a.workers)
        A_n, A_a = build_features(s1c, a.workers)
        info = {"name": B.idf_weight(A_n, B_n, a.max_df), "addr": B.idf_weight(A_a, B_a, a.max_df)}
        BT_n, BT_a = B_n.T.tocsr(), B_a.T.tocsr()
        del B_n, B_a
        A_ch = BT_ch = None
        if not a.no_char:
            A_ch, B_ch, ci = B.char_vectors(pc.name_concat.tolist(), s1c.name_concat.tolist(),
                                            a.char_ngram, a.char_max_df)
            BT_ch = B_ch.T.tocsr()
            del B_ch
            info["char"] = ci
        build_info[c] = info
        log(f"   features built {info}")

        # ---- exact passes
        ex = {}
        ex["exact_sorted"] = B.exact_pass(s1c.name_sorted_key.to_numpy(), pc.name_sorted_key.to_numpy(), a.exact_max_block)
        ex["exact_concat"] = B.exact_pass(s1c.name_concat.to_numpy(), pc.name_concat.to_numpy(), a.exact_max_block)
        r1, c1 = B.exact_pass(s1c.name_sorted_key.to_numpy(), pc.name_alias.to_numpy(), a.exact_max_block)
        r2, c2 = B.exact_pass(s1c.name_alias.to_numpy(), pc.name_sorted_key.to_numpy(), a.exact_max_block)
        ex["exact_alias"] = (np.concatenate([r1, r2]), np.concatenate([c1, c2]))
        log("   exact passes done " + str({k: len(v[0]) for k, v in ex.items()}))

        # ---- sparse + char passes, chunked
        B.G.clear()
        B.G.update(A_n=A_n, A_a=A_a, BT_n=BT_n, BT_a=BT_a, A_ch=A_ch, BT_ch=BT_ch,
                   w_name=a.w_name, w_addr=a.w_addr, min_score=a.min_score,
                   k_comb=a.k_comb, k_name=a.k_name, k_addr=a.k_addr, k_char=a.k_char)
        pos_of = {e: i for i, e in enumerate(pc.entity_id)}
        with Pool(a.workers) as wp:
            for ci_, lo in enumerate(range(0, len(s1c), a.chunk)):
                hi = min(lo + a.chunk, len(s1c))
                bounds = [(x, min(x + a.sub, hi)) for x in range(lo, hi, a.sub)]
                acc = {}
                for res in wp.imap(B.sparse_worker, bounds):
                    for k, v in res.items():
                        acc.setdefault(k, []).append(v)
                parts = [(k, np.concatenate([v[0] for v in vs]), np.concatenate([v[1] for v in vs]),
                          np.concatenate([v[2] for v in vs])) for k, vs in acc.items()]
                for k, (r, cc) in ex.items():
                    m = (r >= lo) & (r < hi)
                    parts.append((k, r[m], cc[m], np.ones(int(m.sum()), np.float32)))
                cand = B.merge_candidates(parts, len(pc))       # full union, ranked
                n_union = cand.groupby("s1").size()

                # ---- write shard (trimmed to --budget-max per S1)
                shard = cand[cand["rank"] < a.budget_max].copy()
                shard.insert(0, "s1_id", s1c.entity_id.to_numpy()[shard.s1.to_numpy()])
                shard.insert(1, "cand_id", pc.entity_id.to_numpy()[shard.cand.to_numpy()])
                write_table(shard.drop(columns=["s1", "cand"]), cand_dir / f"{c}_{ci_:03d}")

                # ---- test: candidate_pairs.tsv for this chunk (every S1 gets a line)
                if tsv is not None:
                    keep = shard[shard["rank"] < a.final_budget]
                    lists = keep.groupby("s1_id")["cand_id"].apply(",".join).to_dict()
                    for sid in s1c.entity_id.to_numpy()[lo:hi]:
                        tsv.write(f"{sid}\t{lists.get(sid, '')}\n")

                # ---- evaluation bookkeeping
                ids = s1c.entity_id.to_numpy()[lo:hi]
                s1_rows.append(pd.DataFrame({
                    "s1": np.arange(lo, hi), "s1_id": ids, "country": c,
                    "n_true": [len(truth.get(x, [])) for x in ids],
                    "n_cand": n_union.reindex(np.arange(lo, hi), fill_value=0).to_numpy(),
                    "pool_size": len(pc)}))
                if labeled:
                    tr = [(i, pos_of.get(m, -1), m) for i, x in zip(range(lo, hi), ids)
                          for m in truth.get(x, [])]
                    if tr:
                        T = pd.DataFrame(tr, columns=["s1", "cand", "other_id"])
                        T = T.merge(cand[["s1", "cand", "passes", "rank"]], on=["s1", "cand"], how="left")
                        ok = T.cand >= 0
                        T["country"] = c
                        T["source"] = T.other_id.str[:2]
                        T["s1_id"] = s1c.entity_id.to_numpy()[T.s1.to_numpy()]
                        T["cross_script"] = False
                        T.loc[ok, "cross_script"] = (s1c.name_script.to_numpy()[T.s1[ok].to_numpy()] !=
                                                     pc.name_script.to_numpy()[T.cand[ok].to_numpy()])
                        T["other_addr_missing"] = False
                        T.loc[ok, "other_addr_missing"] = pc.address_missing.to_numpy()[T.cand[ok].to_numpy()].astype(bool)
                        T["in_pool"] = ok
                        true_rows.append(T)
                log(f"   chunk {ci_}: S1 {lo:,}-{hi:,}  union {len(cand):,}  kept {len(shard):,}")
        B.G.clear()
        del A_n, A_a, BT_n, BT_a, A_ch, BT_ch

    if tsv is not None:
        tsv.close()
        log(f"wrote {out_dir / 'candidate_pairs.tsv'}")

    S = pd.concat(s1_rows, ignore_index=True)
    report = {"args": vars(a), "run": run, "features": build_info, "s1": int(len(S)),
              "note": "budget 'all' = full union of all passes (not trimmed)",
              "candidates_per_s1_union": {"mean": round(float(S.n_cand.mean()), 1),
                                          "p50": float(S.n_cand.median()),
                                          "p99": float(S.n_cand.quantile(0.99)),
                                          "max": int(S.n_cand.max())}}
    if labeled and true_rows:
        report.update(evaluate(S, pd.concat(true_rows, ignore_index=True), budgets, a.final_budget,
                               out_dir, s1, pool))
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    write_summary(report, out_dir / "summary.md")
    log(f"done -> {out_dir}")


# ============================================================================ evaluation

def evaluate(S, T, budgets, final_budget, out_dir, s1, pool):
    rep = {"true_pairs": int(len(T)), "true_pairs_not_in_country_pool": int((~T.in_pool).sum()),
           "by_budget": {}}
    T["rank"] = T["rank"].fillna(np.inf)
    T["passes"] = T["passes"].fillna(0).astype(int)
    total_pool = float(S.pool_size.sum())
    for b in budgets + [10 ** 9]:
        found = T["rank"] < b
        per = T.assign(f=found).groupby("s1_id")["f"].agg(["sum", "count"])
        r = (per["sum"] / per["count"]).reindex(S.s1_id).fillna(-1).to_numpy()   # -1 = singleton
        f = np.where(r < 0, 1.0, np.vectorize(f05)(np.clip(r, 0, 1)))
        ncand = np.minimum(S.n_cand.to_numpy(), b)
        key = "all" if b == 10 ** 9 else str(b)
        rep["by_budget"][key] = {
            "pair_recall": round(float(found.mean()), 5),
            "macro_s1_recall": round(float(r[r >= 0].mean()), 5),
            "s1_all_matches_found": round(float((r[r >= 0] == 1).mean()), 5),
            "oracle_macro_f05": round(float(f.mean()), 5),
            "cand_per_s1_mean": round(float(ncand.mean()), 1),
            "cand_per_s1_p99": float(np.quantile(ncand, 0.99)),
            "reduction_ratio": round(1 - float(ncand.sum()) / total_pool, 7)}
    # passes (on everything kept on disk)
    kept = T["rank"] < np.inf
    rep["passes"] = {}
    for p, bit in B.PASS_BIT.items():
        has = (T.passes & bit) > 0
        rep["passes"][p] = {"recall_alone": round(float(has.mean()), 5),
                            "unique_recall": round(float(((T.passes == bit) & kept).mean()), 5)}
    # segments at final budget
    T["found"] = T["rank"] < final_budget
    T["bucket"] = T.s1_id.map(S.set_index("s1_id").n_true).clip(upper=4)
    rep["segments_at_final_budget"] = {
        seg: {str(k): {"pairs": int(len(g)), "recall": round(float(g.found.mean()), 5)}
              for k, g in T.groupby(seg)}
        for seg in ("country", "source", "cross_script", "other_addr_missing", "bucket")}
    # misses dump
    miss = T[~T.found]
    miss = miss.sample(n=min(300, len(miss)), random_state=1) if len(miss) else miss
    s1i = s1.set_index("entity_id")
    pli = pool.set_index("entity_id")
    cols = ["name_core", "name_phon", "address_tokens_norm"]
    dump = miss[["s1_id", "other_id", "country", "rank", "passes", "cross_script", "other_addr_missing"]].copy()
    for col in cols:
        dump[f"s1_{col}"] = dump.s1_id.map(s1i[col])
        dump[f"other_{col}"] = dump.other_id.map(pli[col])
    dump.to_csv(out_dir / "missed_true_pairs.tsv", sep="\t", index=False)
    return rep


def write_summary(r, path):
    L = [f"# Blocking run `{r['run']}`\n", f"S1 evaluated: {r['s1']:,}",
         f"Candidates per S1 before budget: {r['candidates_per_s1_union']}"]
    if "by_budget" in r:
        L.append("\n## Recall and oracle F0.5 by per-S1 budget\n")
        L.append("| budget | pair recall | macro S1 recall | S1 all found | oracle F0.5 | cand/S1 mean | p99 | reduction |")
        L.append("|---|---|---|---|---|---|---|---|")
        for b, v in r["by_budget"].items():
            L.append(f"| {b} | {v['pair_recall']} | {v['macro_s1_recall']} | {v['s1_all_matches_found']} | "
                     f"{v['oracle_macro_f05']} | {v['cand_per_s1_mean']} | {v['cand_per_s1_p99']} | {v['reduction_ratio']} |")
        L.append("\n## Passes (recall alone / unique recall)\n")
        for p, v in r["passes"].items():
            L.append(f"- {p}: alone {v['recall_alone']}, unique {v['unique_recall']}")
        L.append("\n## Recall at final budget by segment\n```")
        for seg, d in r["segments_at_final_budget"].items():
            L.append(f"{seg}: " + ", ".join(f"{k}={v['recall']} ({v['pairs']:,})" for k, v in d.items()))
        L.append("```")
    L.append("\n## Feature build info\n```\n" + json.dumps(r["features"], indent=1) + "\n```")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
