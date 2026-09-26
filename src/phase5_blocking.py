#!/usr/bin/env python3
"""
Phase 5 - Blocking (candidate generation).

Passes, unioned per S1 (country-partitioned):
  exact_sorted / exact_concat / exact_alias   hash joins on the Phase 3 exact keys
  sparse_comb   top-K by name + address + name-x-address cross features (capped IDF cosines)
  sparse_name   top-K by name alone      (matches whose address is missing/different)
  sparse_addr   top-K by address alone   (matches whose name is different: DBA, noise)
  char_name     top-K by char 3-grams of the glued name (typos, squashed names)
  char_addr     top-K by char 3-grams of the address    (typos in street / city names)
  sparse_phon   top-K by phonetic + coarse-phonetic features (typos, transliterations)
  sparse_cross  top-K by name-key x address-word features alone (descriptor swaps)
  noaddr_name   top-K within pool records WITHOUT an address (name + phon + char name)
  nonlatin_phon top-K within pool records with NON-LATIN names (phon + coarse phon + address)
  expand_prf    optional pseudo-relevance feedback from the strongest candidates (--k-prf)
Every union candidate is then re-ranked with FULL (uncapped) cosines of all feature groups,
name terms scaled by the S1 name's informativeness (see blocking.py), and trimmed to a
per-S1 budget. Every missed true pair is categorised by cause in the report.

Splits
  --split val    S1 = validation split, pool = train S2+S3  -> the numbers we report
  --split train  S1 = train split,      pool = train S2+S3  -> candidates to train the matcher
  --split test   S1 = test S1,          pool = test S2+S3   -> writes output candidate_pairs.tsv
Use --s1-sample for quick experiments on a subset of S1.

Outputs
  {out-dir}/{run}/report.json, summary.md, missed_true_pairs.tsv      (val/train)
  {cand-dir}/{run}/{country}_{chunk}.parquet   candidates with passes, scores, priority, rank
  {out-dir}/{run}/candidate_pairs.tsv          (test) README format, final budget applied

Scale: every worker writes its own candidate shard and returns only small summaries (no
candidate rows travel back to the main process); pool features can be cached on disk with
--feature-cache so the val and train runs (same pool) build them once.

Checkpointing: each worker writes its own shard file per sub-batch
(cand-dir/run/{country}_{lo:09d}.parquet). Before dispatching a sub-batch, the driver skips it
if that shard already exists, so re-running the SAME command after a crash or a stopped space
picks up where it left off instead of redoing finished work. For --split test, completed S1
ranges are also appended to {out-dir}/run/tsv_done.txt as they finish, and candidate_pairs.tsv
lines are written in that same order on completion, so a resumed run does not duplicate or
reorder rows. Delete the run's folders in --cand-dir and --out-dir to force a clean restart
(or pass --no-resume).

Usage
  python src/phase5_blocking.py --split val --s1-sample 50000      # quick experiment
  python src/phase5_blocking.py --split val  --feature-cache data_cache
  python src/phase5_blocking.py --split train --s1-sample 400000 --feature-cache data_cache
  python src/phase5_blocking.py --split test --feature-cache data_cache
  # interrupted? just run the exact same command again - finished shards are skipped
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import blocking as B
from common import read_table, write_table

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def build_features(df: pd.DataFrame, workers: int, chunk: int = 50_000) -> dict:
    rows = list(zip(df.name_core, df.name_phon, df.address_tokens_norm, df.address_nums,
                    df.address_numcompounds, df.name_concat))
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    acc = {g: ([], []) for g in B.GROUPS}
    with Pool(workers) as p:
        for res in p.imap(B.feature_chunk, parts):
            for g, (l, i) in res.items():
                acc[g][0].append(l)
                acc[g][1].append(i)
    return {g: B.csr_from_parts(l, i) for g, (l, i) in acc.items()}


def pool_features(pc: pd.DataFrame, workers: int, cache_dir, key: str, rebuild: bool) -> dict:
    """Raw hashed pool features, cached on disk (the val and train runs share the pool).
    Delete the cache folder whenever data_block or the feature code changes."""
    if cache_dir:
        cache_dir = Path(cache_dir)
        paths = {g: cache_dir / f"{key}_{g}.npz" for g in B.GROUPS}
        if not rebuild and all(p.exists() for p in paths.values()):
            F = {g: sp.load_npz(p).tocsr() for g, p in paths.items()}
            if all(m.shape[0] == len(pc) for m in F.values()):
                log(f"   pool features loaded from cache ({key})")
                return F
    F = build_features(pc, workers)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for g, p in paths.items():
            sp.save_npz(p, F[g], compressed=False)
    return F


def shard_path(cand_dir: Path, country: str, lo: int) -> Path:
    return cand_dir / f"{country}_{lo:09d}"


def shard_exists(cand_dir: Path, country: str, lo: int) -> bool:
    p = shard_path(cand_dir, country, lo)
    return p.with_suffix(".parquet").exists() or p.with_suffix(".tsv.gz").exists()


def read_shard_summary(cand_dir: Path, country: str, lo: int, hi: int, s1_ids_lohi: np.ndarray,
                       s1_ids_range: np.ndarray, other_ids_range: np.ndarray) -> tuple:
    """On resume: recover n_cand (per S1 position in [lo, hi), in s1_ids_lohi order) and, for
    the true pairs whose S1 falls in this range (s1_ids_range / other_ids_range, aligned), their
    rank/passes - all by reading the already-written shard instead of recomputing candidates."""
    p = shard_path(cand_dir, country, lo)
    if not (p.with_suffix(".parquet").exists() or p.with_suffix(".tsv.gz").exists()):
        return (np.zeros(hi - lo, np.int64), np.full(len(s1_ids_range), -1, np.int64),
                np.zeros(len(s1_ids_range), np.int64))
    df = read_table(p)
    n_by_id = df.groupby("s1_id").size().to_dict()
    n_cand = np.array([n_by_id.get(sid, 0) for sid in s1_ids_lohi], np.int64)
    if len(s1_ids_range):
        key = pd.DataFrame({"s1_id": s1_ids_range, "cand_id": other_ids_range, "i": np.arange(len(s1_ids_range))})
        m = key.merge(df[["s1_id", "cand_id", "rank", "passes"]], on=["s1_id", "cand_id"], how="left")
        m = m.sort_values("i")
        rank = m["rank"].fillna(-1).to_numpy(np.int64)
        passes = m["passes"].fillna(0).to_numpy(np.int64)
    else:
        rank, passes = np.zeros(0, np.int64), np.zeros(0, np.int64)
    return n_cand, rank, passes


def read_shard_tsv(cand_dir: Path, country: str, lo: int, hi: int, s1c: pd.DataFrame, final_budget: int) -> str:
    p = shard_path(cand_dir, country, lo)
    ids = s1c.entity_id.to_numpy()[lo:hi]
    if not (p.with_suffix(".parquet").exists() or p.with_suffix(".tsv.gz").exists()):
        return "".join(f"{sid}\t\n" for sid in ids)
    df = read_table(p)
    keep = df[df["rank"] < final_budget]
    lists = keep.groupby("s1_id")["cand_id"].apply(",".join).to_dict()
    return "".join(f"{sid}\t{lists.get(sid, '')}\n" for sid in ids)


def emit_worker(bounds):
    """Runs in a worker: block S1 rows [lo, hi), write the trimmed candidates straight to a
    shard, and return only small arrays (union sizes, ranks of true pairs, TSV lines)."""
    lo, hi = bounds
    g = B.G
    out = B.block_worker(bounds)
    s1, cand, rank, passes = out["s1"], out["cand"], out["rank"], out["passes"]
    res = {"lo": lo, "hi": hi, "n_union": np.bincount(s1 - lo, minlength=hi - lo), "kept": 0}

    ts, tc = g["t_s1"], g["t_cand"]                       # true pairs, sorted by s1
    i0, i1 = np.searchsorted(ts, lo), np.searchsorted(ts, hi)
    if i1 > i0:
        tr = np.full(i1 - i0, -1, np.int64)
        tp = np.zeros(i1 - i0, np.int64)
        if len(s1):
            ukey = s1.astype(np.int64) * g["n_pool"] + cand
            order = np.argsort(ukey)
            sk = ukey[order]
            tkey = ts[i0:i1].astype(np.int64) * g["n_pool"] + np.maximum(tc[i0:i1], 0)
            pos = np.minimum(np.searchsorted(sk, tkey), len(sk) - 1)
            hit = (sk[pos] == tkey) & (tc[i0:i1] >= 0)
            idx = order[pos]
            tr = np.where(hit, rank[idx], -1)
            tp = np.where(hit, passes[idx], 0)
        res.update(t0=i0, t1=i1, t_rank=tr, t_passes=tp)

    keep = rank < g["budget_max"]
    if keep.any():
        df = pd.DataFrame({"s1_id": g["s1_ids"][s1[keep]], "cand_id": g["pool_ids"][cand[keep]],
                           "passes": passes[keep], "priority": out["priority"][keep],
                           "rank": rank[keep], **{c: out[c][keep] for c in B.COS_COLS}})
        write_table(df, shard_path(g["shard_dir"], g["country"], lo))
        res["kept"] = int(keep.sum())

    if g["final_budget"]:                                   # test: candidate_pairs.tsv lines
        fb = rank < g["final_budget"]
        ids = g["pool_ids"][cand[fb]]
        rows = s1[fb] - lo
        buckets = [[] for _ in range(hi - lo)]
        for r_, c_ in zip(rows, ids):
            buckets[r_].append(c_)
        names = g["s1_ids"]
        res["tsv"] = "".join(f"{names[lo + j]}\t{','.join(b)}\n" for j, b in enumerate(buckets))
    return res


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
    ap.add_argument("--k-comb", type=int, default=150)
    ap.add_argument("--k-name", type=int, default=50)
    ap.add_argument("--k-addr", type=int, default=50)
    ap.add_argument("--k-char-name", type=int, default=50, help="0 disables the pass")
    ap.add_argument("--k-char-addr", type=int, default=50, help="0 disables the pass")
    ap.add_argument("--w-name", type=float, default=1.0)
    ap.add_argument("--w-addr", type=float, default=1.0)
    ap.add_argument("--w-cross", type=float, default=0.5)
    ap.add_argument("--w-cname", type=float, default=0.5)
    ap.add_argument("--w-caddr", type=float, default=0.5)
    ap.add_argument("--w-exact", type=float, default=0.25)
    ap.add_argument("--k-phon", type=int, default=50)
    ap.add_argument("--k-cross", type=int, default=50)
    ap.add_argument("--k-noaddr", type=int, default=50)
    ap.add_argument("--k-nonlatin", type=int, default=50)
    ap.add_argument("--k-prf", type=int, default=0, help="PRF expansion; 0 = off")
    ap.add_argument("--prf-seeds", type=int, default=3)
    ap.add_argument("--prf-min", type=float, default=1.5, help="min priority for a seed")
    ap.add_argument("--prf-alpha", type=float, default=1.0)
    ap.add_argument("--missing-addr-factor", type=float, default=0.7)
    ap.add_argument("--info-floor", type=float, default=0.3,
                    help="lowest name-informativeness multiplier (common one-word names)")
    ap.add_argument("--dump-misses", type=int, default=2000)
    ap.add_argument("--exact-max-block", type=int, default=5000)
    ap.add_argument("--budget-max", type=int, default=200, help="candidates kept per S1 on disk")
    ap.add_argument("--budgets", default="10,25,50,100,200,300")
    ap.add_argument("--final-budget", type=int, default=200, help="budget for candidate_pairs.tsv / misses")
    ap.add_argument("--sub", type=int, default=2_000, help="S1 rows per worker task (lower = less memory)")
    ap.add_argument("--feature-cache", default=None, help="folder to cache pool features in")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing shards/progress and redo everything")
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

    true_rows, s1_rows, build_info, resumed_countries = [], [], {}, []
    for c in countries:
        s1c = s1[s1.country_norm == c].reset_index(drop=True)
        pc = pool[pool.country_norm == c].reset_index(drop=True)
        log(f"== {c}: S1 {len(s1c):,}  pool {len(pc):,}")

        # ---- features (capped for generation + full for re-ranking)
        Fp = pool_features(pc, a.workers, a.feature_cache, f"{src}_{c}", a.rebuild_cache)
        Fs = build_features(s1c, a.workers)
        M, info, Bcap = {}, {}, {}
        for g in B.GROUPS:
            A_cap, B_cap, A_full, B_full, info[g], a_norm = B.weighted(Fs[g], Fp[g], a.max_df)
            M[g] = {"A_cap": A_cap, "BT_cap": B_cap.T.tocsr(), "A_full": A_full, "B_full": B_full}
            Bcap[g] = B_cap
            if g == "name":
                med = float(np.median(a_norm[a_norm > 0])) if (a_norm > 0).any() else 1.0
                name_info = np.clip(a_norm / med, a.info_floor, 1.0).astype(np.float32)
        del Fp, Fs
        # candidate slices: records without an address, records with a non-Latin name
        noaddr = np.flatnonzero(pc.address_missing.to_numpy(bool))
        nonlatin = np.flatnonzero(~pc.name_script.isin(["latin", "none"]).to_numpy())
        sub = {"noaddr": {"idx": noaddr, **{f"BT_{g}": Bcap[g][noaddr].T.tocsr() for g in ("name", "phon", "cname")}},
               "nonlatin": {"idx": nonlatin, **{f"BT_{g}": Bcap[g][nonlatin].T.tocsr() for g in ("phon", "cphon", "addr")}}}
        del Bcap
        info["name_informativeness"] = {"median_norm": round(med, 2),
                                        "share_below_1": round(float((name_info < 1).mean()), 4),
                                        "share_at_floor": round(float((name_info <= a.info_floor).mean()), 4)}
        info["slices"] = {"noaddr": int(len(noaddr)), "nonlatin": int(len(nonlatin))}
        build_info[c] = info
        log(f"   features built; slices {info['slices']}")

        # ---- exact passes
        ex = {}
        ex["exact_sorted"] = B.exact_pass(s1c.name_sorted_key.to_numpy(), pc.name_sorted_key.to_numpy(), a.exact_max_block)
        ex["exact_concat"] = B.exact_pass(s1c.name_concat.to_numpy(), pc.name_concat.to_numpy(), a.exact_max_block)
        r1, c1 = B.exact_pass(s1c.name_sorted_key.to_numpy(), pc.name_alias.to_numpy(), a.exact_max_block)
        r2, c2 = B.exact_pass(s1c.name_alias.to_numpy(), pc.name_sorted_key.to_numpy(), a.exact_max_block)
        ex["exact_alias"] = (np.concatenate([r1, r2]), np.concatenate([c1, c2]))
        log("   exact passes done " + str({k: len(v[0]) for k, v in ex.items()}))

        # ---- true pairs for this country, sorted by S1 position
        pos_of = {e: i for i, e in enumerate(pc.entity_id)}
        t_s1, t_cand, t_other = [], [], []
        if labeled:
            for i, x in enumerate(s1c.entity_id):
                for m in truth.get(x, []):
                    t_s1.append(i); t_cand.append(pos_of.get(m, -1)); t_other.append(m)  # noqa: E702
        t_s1 = np.asarray(t_s1, np.int64)
        t_cand = np.asarray(t_cand, np.int64)

        # ---- generation + union + re-ranking + shard writing, all in workers
        B.G.clear()
        B.G.update(M=M, exact=ex, n_pool=len(pc), sub=sub, name_info=name_info,
                   pool_addr_missing=pc.address_missing.to_numpy(bool),
                   prf={"seeds": a.prf_seeds, "min_priority": a.prf_min, "alpha": a.prf_alpha},
                   k={"sparse_comb": a.k_comb, "sparse_name": a.k_name, "sparse_addr": a.k_addr,
                      "char_name": a.k_char_name, "char_addr": a.k_char_addr, "sparse_phon": a.k_phon,
                      "sparse_cross": a.k_cross, "noaddr_name": a.k_noaddr,
                      "nonlatin_phon": a.k_nonlatin, "expand_prf": a.k_prf},
                   w={"name": a.w_name, "addr": a.w_addr, "cross": a.w_cross, "cname": a.w_cname,
                      "caddr": a.w_caddr, "exact": a.w_exact,
                      "missing_addr_factor": a.missing_addr_factor},
                   t_s1=t_s1, t_cand=t_cand, budget_max=a.budget_max,
                   final_budget=a.final_budget if a.split == "test" else 0,
                   s1_ids=s1c.entity_id.to_numpy(dtype=object), pool_ids=pc.entity_id.to_numpy(dtype=object),
                   shard_dir=cand_dir, country=c)
        n_cand = np.zeros(len(s1c), np.int64)
        t_rank = np.full(len(t_s1), -1, np.int64)
        t_pass = np.zeros(len(t_s1), np.int64)
        all_bounds = [(x, min(x + a.sub, len(s1c))) for x in range(0, len(s1c), a.sub)]
        todo, already_done = [], []
        for lo, hi in all_bounds:
            (already_done if (not a.no_resume and shard_exists(cand_dir, c, lo)) else todo).append((lo, hi))
        s1_ids_arr = s1c.entity_id.to_numpy()
        t_other_arr = np.asarray(t_other, dtype=object)
        if already_done:
            resumed_countries.append(c)
            log(f"   resume: {len(already_done):,}/{len(all_bounds):,} sub-batches already on disk, skipping "
                f"(shards keep only rank < budget_max={a.budget_max}, so 'all'/union stats for this "
                f"country's resumed portion are capped at budget_max, not the true unbounded union)")
            for lo, hi in already_done:
                i0, i1 = np.searchsorted(t_s1, lo), np.searchsorted(t_s1, hi)
                nc, tr, tp = read_shard_summary(cand_dir, c, lo, hi, s1_ids_arr[lo:hi],
                                                s1_ids_arr[t_s1[i0:i1]], t_other_arr[i0:i1])
                n_cand[lo:hi] = nc
                t_rank[i0:i1], t_pass[i0:i1] = tr, tp
        done, kept, next_log = sum(h - l for l, h in already_done), 0, 0.1
        tsv_lines = {} if tsv is not None else None
        with Pool(a.workers) as wp:
            for res in wp.imap_unordered(emit_worker, todo):
                n_cand[res["lo"]:res["hi"]] = res["n_union"]
                if "t0" in res:
                    t_rank[res["t0"]:res["t1"]] = res["t_rank"]
                    t_pass[res["t0"]:res["t1"]] = res["t_passes"]
                if tsv_lines is not None:
                    tsv_lines[res["lo"]] = res["tsv"]
                kept += res["kept"]
                done += res["hi"] - res["lo"]
                if done / len(s1c) >= next_log:
                    log(f"   {done:,}/{len(s1c):,} S1 done, {kept:,} candidates written")
                    next_log += 0.1
        if tsv is not None:
            for lo, hi in already_done:
                tsv_lines[lo] = read_shard_tsv(cand_dir, c, lo, hi, s1c, a.final_budget)
            for lo, _ in all_bounds:
                tsv.write(tsv_lines[lo])

        # ---- evaluation bookkeeping (small)
        s1_rows.append(pd.DataFrame({
            "s1_id": s1c.entity_id.to_numpy(), "country": c,
            "n_true": np.bincount(t_s1, minlength=len(s1c)) if len(t_s1) else 0,
            "n_cand": n_cand, "pool_size": len(pc)}))
        if len(t_s1):
            ok = t_cand >= 0
            cs = np.zeros(len(t_s1), bool)
            am = np.zeros(len(t_s1), bool)
            cs[ok] = s1c.name_script.to_numpy()[t_s1[ok]] != pc.name_script.to_numpy()[t_cand[ok]]
            am[ok] = pc.address_missing.to_numpy(bool)[t_cand[ok]]
            true_rows.append(pd.DataFrame({
                "s1_id": s1c.entity_id.to_numpy()[t_s1], "other_id": t_other, "country": c,
                "source": [m[:2] for m in t_other],
                "rank": np.where(t_rank >= 0, t_rank, np.inf), "passes": t_pass,
                "cross_script": cs, "other_addr_missing": am, "in_pool": ok}))
        B.G.clear()
        del M, sub

    if tsv is not None:
        tsv.close()
        log(f"wrote {out_dir / 'candidate_pairs.tsv'}")

    S = pd.concat(s1_rows, ignore_index=True)
    resume_note = (
        "Resumed from on-disk shards for: " + ", ".join(resumed_countries) + ". Shards only "
        "persist candidates with rank < budget_max=" + str(a.budget_max) + ", so for these "
        "countries the 'all' row, any --budgets entry above budget_max, and the top-level "
        "candidates_per_s1_union stat reflect what was WRITTEN (<= budget_max), not the true "
        "unbounded union. Every budget <= budget_max is exact. For exact unbounded numbers, "
        "delete these countries' shards and rerun (or pass --no-resume)."
    ) if resumed_countries else None
    report = {"args": vars(a), "run": run, "features": build_info, "s1": int(len(S)),
              "note": "budget 'all' = full union of all passes (not trimmed)",
              "resume_note": resume_note,
              "candidates_per_s1_union": {"mean": round(float(S.n_cand.mean()), 1),
                                          "p50": float(S.n_cand.median()),
                                          "p99": float(S.n_cand.quantile(0.99)),
                                          "max": int(S.n_cand.max())}}
    if labeled and true_rows:
        eval_budgets = [b for b in budgets if b <= a.budget_max] if resumed_countries else budgets
        report.update(evaluate(S, pd.concat(true_rows, ignore_index=True), eval_budgets, a.final_budget,
                               out_dir, s1, pool, a.dump_misses, include_all=not resumed_countries))
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    write_summary(report, out_dir / "summary.md")
    if resume_note:
        log(f"NOTE: {resume_note}")
    log(f"done -> {out_dir}")


# ============================================================================ evaluation

def categorise(miss: pd.DataFrame, s1i: pd.DataFrame, pli: pd.DataFrame) -> pd.Series:
    """Label each missed true pair with its most likely cause (first rule that applies)."""
    def toks(v):
        return set(v.split()) if isinstance(v, str) and v else set()
    cats = []
    for r in miss.itertuples():
        a, b = s1i.loc[r.s1_id], pli.loc[r.other_id]
        name_ov = bool(toks(a.name_core) & toks(b.name_core))
        phon_ov = bool(toks(a.name_phon) & toks(b.name_phon))
        addr_ov = bool(toks(a.address_tokens_norm) & toks(b.address_tokens_norm))
        same_addr = isinstance(a.address_tokens_norm, str) and a.address_tokens_norm == b.address_tokens_norm
        if np.isfinite(r.rank):
            c = "ranked_out_same_address" if same_addr else "ranked_out"
        elif r.other_addr_missing:
            c = "addr_missing_name_overlap" if (name_ov or phon_ov) else "addr_missing_no_name_overlap"
        elif r.cross_script:
            c = "cross_script_phon_overlap" if phon_ov else "cross_script_no_phon_overlap"
        elif not (name_ov or phon_ov):
            c = "name_changed_address_only" if addr_ov else "no_shared_word"
        elif not addr_ov:
            c = "name_only_address_differs"
        else:
            c = "weak_name_and_address"
        cats.append(c)
    return pd.Series(cats, index=miss.index)


def evaluate(S, T, budgets, final_budget, out_dir, s1, pool, dump_n=2000, include_all=True):
    rep = {"true_pairs": int(len(T)), "true_pairs_not_in_country_pool": int((~T.in_pool).sum()),
           "by_budget": {}}
    T["rank"] = T["rank"].fillna(np.inf)
    T["passes"] = T["passes"].fillna(0).astype(int)
    total_pool = float(S.pool_size.sum())
    for b in (budgets + [10 ** 9]) if include_all else budgets:
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
    # misses: categorise ALL of them, dump a sample
    s1i = s1.set_index("entity_id")
    pli = pool.set_index("entity_id")
    miss = T[~T.found].copy()
    miss["category"] = categorise(miss, s1i, pli) if len(miss) else []
    rep["miss_categories_at_final_budget"] = {
        "misses": int(len(miss)),
        "share_of_true_pairs": {k: round(v / len(T), 5) for k, v in miss.category.value_counts().items()}}
    miss = miss.sample(n=min(dump_n, len(miss)), random_state=1) if len(miss) else miss
    cols = ["name_core", "name_phon", "address_tokens_norm"]
    dump = miss[["s1_id", "other_id", "country", "category", "rank", "passes", "cross_script",
                 "other_addr_missing"]].copy()
    for col in cols:
        dump[f"s1_{col}"] = dump.s1_id.map(s1i[col])
        dump[f"other_{col}"] = dump.other_id.map(pli[col])
    dump.to_csv(out_dir / "missed_true_pairs.tsv", sep="\t", index=False)
    return rep


def write_summary(r, path):
    L = [f"# Blocking run `{r['run']}`\n", f"S1 evaluated: {r['s1']:,}",
         f"Candidates per S1 before budget: {r['candidates_per_s1_union']}"]
    if r.get("resume_note"):
        L.insert(1, f"\n**⚠ {r['resume_note']}**\n")
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
        mc = r.get("miss_categories_at_final_budget", {})
        if mc:
            L.append(f"\n## Misses at final budget by cause ({mc['misses']:,} misses; share of ALL true pairs)\n")
            for k, v in mc["share_of_true_pairs"].items():
                L.append(f"- {k}: {v}")
        L.append("\n## Recall at final budget by segment\n```")
        for seg, d in r["segments_at_final_budget"].items():
            L.append(f"{seg}: " + ", ".join(f"{k}={v['recall']} ({v['pairs']:,})" for k, v in d.items()))
        L.append("```")
    L.append("\n## Feature build info\n```\n" + json.dumps(r["features"], indent=1) + "\n```")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
