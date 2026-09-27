#!/usr/bin/env python3
"""
Phase 7 - Stage-2 pairwise features for Stage-1's top-N candidates.

Reads   data_stage1/{split}/<shard>   (s1_id, cand_id, stage1_score, stage1_rank, passes, priority,
                                        rank, 7 blocking cosines) from phase6_stage1.py
joins   data_repr/{split}_source{1,2,3} (Phase 3 representations)
writes  data_stage2/{split}/<shard>   one row per (S1, candidate): the Stage-1 columns plus
                                        ~55 features. No labels (the Stage-2 trainer joins them).

Features (no country column or country-derived feature - France has no labels):
  string similarity (rapidfuzz, 0-1)   name_core: ratio, token_sort, token_set, partial, Jaro-Winkler
                                        name_concat: ratio, Jaro-Winkler;  name_roman: token_set
                                        address_norm: ratio, token_sort, token_set, partial
                                        address_roman: ratio
  token overlap                         Jaccard + overlap coefficient for name words, phonetic keys,
                                        coarse phonetic keys, address words; IDF-weighted overlap
                                        for name and address words
  numbers                               house-number agree / conflict, postal agree / conflict /
                                        present (postal = longest 5+ digit number: ZIP, PIN, French CP),
                                        any-number agree / conflict
  name extras                           alias match, initials match, legal suffix agree / conflict,
                                        name informativeness (S1, candidate, min)
  flags                                 address missing (S1 / candidate / either), cross-script
  neighbourhood (within the S1's list)  gap to the S1's best Stage-1 score, similarity to the S1's
                                        best candidate (name, address), identical copy present
Empty text is absence of evidence: all similarities are 0 when either side is empty.

Why this version is fast (the first draft needed ~2 s per S1 at 12.5M records):
  - ID lookup: np.searchsorted on one sorted numpy array of IDs built once in the parent
    (the draft ran Series.isin(source.index) per S1, which rebuilds a hash table over all
    12.5M IDs on every call);
  - text columns are held as Arrow arrays (or numpy in the fallback): shared with forked
    workers without copy-on-write blow-up; each shard takes only the rows it needs;
  - each record's tokens / keys / numbers are parsed ONCE per shard, not once per pair;
  - string similarities are computed in batches with rapidfuzz.process.cpdist (C++);
  - the pair loop fills numpy arrays (no per-pair dicts, no pandas row objects);
  - neighbourhood features are vectorized with groupby;
  - workers write their own output shards; the parent logs progress and an ETA.
Restarting the same command skips shards that are already written (resume).

Usage
  python src/phase7_stage2_features.py --split val --limit-shards 16    # timing check first
  python src/phase7_stage2_features.py --split train
  python src/phase7_stage2_features.py --split val
  python src/phase7_stage2_features.py --split test
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from blocking import coarse_key
from common import read_table, write_table

try:
    import pyarrow as pa
except ImportError:          # fallback: numpy object arrays (fine for small tests)
    pa = None

T0 = time.time()
TEXT_COLS = ["name_script", "name_roman", "name_core", "name_suffix", "name_sorted_key",
             "name_alias", "name_concat", "name_initials", "name_phon", "address_roman",
             "address_tokens_norm", "address_nums"]
FLAG_COLS = ["address_missing"]
STAGE1_COLS = ["s1_id", "cand_id", "stage1_score", "stage1_rank", "passes", "priority", "rank"]
COS_COLS = ["cos_name", "cos_addr", "cos_cross", "cos_cname", "cos_caddr", "cos_phon", "cos_cphon"]
# (output name, text column, scorer, divide by 100?)
STRING_FEATURES = [
    ("name_core_ratio", "name_core", fuzz.ratio, True),
    ("name_core_token_sort", "name_core", fuzz.token_sort_ratio, True),
    ("name_core_token_set", "name_core", fuzz.token_set_ratio, True),
    ("name_core_partial", "name_core", fuzz.partial_ratio, True),
    ("name_core_jaro_winkler", "name_core", JaroWinkler.similarity, False),
    ("name_concat_ratio", "name_concat", fuzz.ratio, True),
    ("name_concat_jaro_winkler", "name_concat", JaroWinkler.similarity, False),
    ("name_roman_token_set", "name_roman", fuzz.token_set_ratio, True),
    ("address_norm_ratio", "address_tokens_norm", fuzz.ratio, True),
    ("address_norm_token_sort", "address_tokens_norm", fuzz.token_sort_ratio, True),
    ("address_norm_token_set", "address_tokens_norm", fuzz.token_set_ratio, True),
    ("address_norm_partial", "address_tokens_norm", fuzz.partial_ratio, True),
    ("address_roman_ratio", "address_roman", fuzz.ratio, True),
]
LOOP_FEATURES = ["name_token_jaccard", "name_token_overlap", "phon_jaccard", "phon_overlap",
                 "coarse_phon_jaccard", "coarse_phon_overlap", "address_token_jaccard",
                 "address_token_overlap", "name_idf_overlap", "address_idf_overlap",
                 "alias_match", "initials_match", "legal_suffix_agree", "legal_suffix_conflict",
                 "house_num_agree", "house_num_conflict", "postal_present_both", "postal_agree",
                 "postal_conflict", "number_agree", "number_conflict",
                 "name_info_s1", "name_info_cand", "name_info_min"]
OTHER_FEATURES = ["address_missing_s1", "address_missing_cand", "address_missing_either",
                  "cross_script", "stage1_score_gap", "to_s1_best_name_ratio",
                  "to_s1_best_addr_ratio", "identical_copy_present"]
FEATURES = COS_COLS + [f[0] for f in STRING_FEATURES] + LOOP_FEATURES + OTHER_FEATURES
HAS_CPDIST = hasattr(process, "cpdist")


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def shards(d: Path) -> list[str]:
    out = []
    for p in sorted(Path(d).iterdir()):
        for ext in (".parquet", ".tsv.gz"):
            if p.name.endswith(ext):
                out.append(str(p)[:-len(ext)])
    return out


def shard_done(target: Path) -> bool:
    return Path(str(target) + ".parquet").exists() or Path(str(target) + ".tsv.gz").exists()


# ============================================================================ record store

class Store:
    """All records of a split (S1 + S2 + S3), built once in the parent and shared with forked
    workers. IDs live in one sorted numpy unicode array (lookup = searchsorted, vectorized,
    no Python objects); text columns are Arrow arrays (no per-element refcounts, so reading
    them in forked workers does not copy memory)."""

    def __init__(self, df: pd.DataFrame):
        ids = np.asarray(df["entity_id"].to_numpy(dtype=object), dtype=str)
        self.order = np.argsort(ids, kind="stable")
        self.sorted_ids = ids[self.order]
        dup = self.sorted_ids[1:] == self.sorted_ids[:-1]
        if dup.any():
            raise ValueError(f"duplicate entity_id: {self.sorted_ids[1:][dup][0]!r}")
        self.n = len(ids)
        self.text = {c: self._arr(df[c]) for c in TEXT_COLS}
        self.flags = {c: df[c].fillna(False).to_numpy(dtype=bool) for c in FLAG_COLS}

    @staticmethod
    def _arr(s: pd.Series):
        if pa is not None:
            return pa.array(s.to_numpy(dtype=object), type=pa.large_string(), from_pandas=True)
        return s.to_numpy(dtype=object)

    def lookup(self, ids) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(ids, dtype=str)
        i = np.searchsorted(self.sorted_ids, q)
        i_c = np.minimum(i, self.n - 1)
        ok = (i < self.n) & (self.sorted_ids[i_c] == q)
        return self.order[i_c], ok

    def take(self, col: str, pos: np.ndarray) -> list:
        a = self.text[col]
        if pa is not None:
            return a.take(pa.array(pos, type=pa.int64())).to_pylist()
        return a[pos].tolist()


def load_store(repr_dir: Path, split: str) -> Store:
    cols = ["entity_id"] + TEXT_COLS + FLAG_COLS
    frames = [read_table(repr_dir / f"{split}_source{k}", columns=cols) for k in (1, 2, 3)]
    return Store(pd.concat(frames, ignore_index=True))


def load_idf(repr_dir: Path, split: str, pool_size: int) -> tuple[dict, dict]:
    """Global (all countries together) IDF over the S2+S3 pool, from Phase 3's per-country
    document-frequency table: df summed across countries."""
    t = read_table(repr_dir / f"df_{split}")
    t = t[t["field"].isin(["name_core", "address_tokens_norm"])]
    t["df"] = pd.to_numeric(t["df"])
    g = t.groupby(["field", "token"], sort=False)["df"].sum()
    out = {}
    for field in ("name_core", "address_tokens_norm"):
        s = g.loc[field]
        out[field] = dict(zip(s.index, (np.log((pool_size + 1) / (s.to_numpy() + 1)) + 1.0).tolist()))
    return out["name_core"], out["address_tokens_norm"]


# ============================================================================ feature helpers

def batch_scores(a: list, b: list, scorer, div100: bool) -> np.ndarray:
    """Element-wise similarity of two equal-length string lists; 0 where either is empty."""
    a = [x or "" for x in a]
    b = [x or "" for x in b]
    if HAS_CPDIST:
        s = process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=1)
    else:
        s = np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))
    s = np.asarray(s, dtype=np.float32)
    if div100:
        s = s / 100.0
    empty = np.fromiter((not x or not y for x, y in zip(a, b)), dtype=bool, count=len(a))
    s[empty] = 0.0
    return s


def longest_postal(nums: set):
    x = [n for n in nums if len(n) >= 5]
    return max(x, key=lambda n: (len(n), n)) if x else None


def parse_record(r: dict, idf_name: dict, idf_addr: dict) -> tuple:
    """Everything the pair loop needs from one record, computed once per shard."""
    nt = set((r["name_core"] or "").split())
    ph = set((r["name_phon"] or "").split())
    at = set((r["address_tokens_norm"] or "").split())
    nums = set((r["address_nums"] or "").split())
    postal = longest_postal(nums)
    return (nt, ph, {coarse_key(k) for k in ph}, at,
            set((r["name_suffix"] or "").split()), nums, postal,
            nums - ({postal} if postal else set()),
            r["name_sorted_key"] or "", r["name_alias"] or "", r["name_initials"] or "",
            sum(idf_name.get(t, 1.0) for t in nt), sum(idf_addr.get(t, 1.0) for t in at))


def jac_ov(a: set, b: set) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    n = len(a & b)
    return n / (len(a) + len(b) - n), n / min(len(a), len(b))


# ============================================================================ worker

G = {}


def init_worker(state: dict):
    if "store" not in G:                     # spawn fallback (non-Linux): build once per worker
        G.update(state)
        G["store"] = load_store(Path(state["repr_dir"]), state["split"])
        G["idf_name"], G["idf_addr"] = load_idf(Path(state["repr_dir"]), state["split"],
                                                G["pool_size"])


def build_shard(base: str) -> tuple[int, float]:
    t0 = time.time()
    store, idf_n, idf_a = G["store"], G["idf_name"], G["idf_addr"]
    p = read_table(Path(base))
    for c in STAGE1_COLS[2:] + COS_COLS:
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    px, okx = store.lookup(p["s1_id"].to_numpy(dtype=object))
    py, oky = store.lookup(p["cand_id"].to_numpy(dtype=object))
    keep = okx & oky
    p, px, py = p[keep].reset_index(drop=True), px[keep], py[keep]
    n = len(p)
    if n == 0:
        write_table(p.assign(**{f: [] for f in FEATURES if f not in p}), Path(G["out_dir"]) / Path(base).name)
        return 0, time.time() - t0

    # --- fetch each distinct record once
    ux, ix = np.unique(px, return_inverse=True)
    uy, iy = np.unique(py, return_inverse=True)
    X = {c: store.take(c, ux) for c in TEXT_COLS}
    Y = {c: store.take(c, uy) for c in TEXT_COLS}
    out = {c: p[c].to_numpy() for c in STAGE1_COLS + COS_COLS if c in p}
    for c in COS_COLS:
        out[c] = np.nan_to_num(np.asarray(out.get(c, np.zeros(n)), dtype=np.float32))

    # --- string similarities, batched
    for name, col, scorer, div in STRING_FEATURES:
        xs, ys = X[col], Y[col]
        out[name] = batch_scores([xs[i] for i in ix], [ys[j] for j in iy], scorer, div)

    # --- token / number features: parse each record once, then one tight loop over pairs
    PX = [parse_record({c: X[c][k] for c in TEXT_COLS}, idf_n, idf_a) for k in range(len(ux))]
    PY = [parse_record({c: Y[c][k] for c in TEXT_COLS}, idf_n, idf_a) for k in range(len(uy))]
    F = {f: np.zeros(n, np.float32) for f in LOOP_FEATURES}
    cols = [F[f] for f in LOOP_FEATURES]
    for k in range(n):
        a, b = PX[ix[k]], PY[iy[k]]
        v = [0.0] * len(LOOP_FEATURES)
        v[0], v[1] = jac_ov(a[0], b[0])
        v[2], v[3] = jac_ov(a[1], b[1])
        v[4], v[5] = jac_ov(a[2], b[2])
        v[6], v[7] = jac_ov(a[3], b[3])
        if a[0] and b[0]:
            inter = sum(idf_n.get(t, 1.0) for t in a[0] & b[0])
            v[8] = inter / (a[11] + b[11] - inter) if (a[11] + b[11] - inter) > 0 else 0.0
        if a[3] and b[3]:
            inter = sum(idf_a.get(t, 1.0) for t in a[3] & b[3])
            v[9] = inter / (a[12] + b[12] - inter) if (a[12] + b[12] - inter) > 0 else 0.0
        v[10] = float(bool(a[9]) and (a[9] == b[8] or a[9] == b[9]) or bool(b[9]) and b[9] == a[8])
        v[11] = float(bool(a[10]) and a[10] == b[10])
        v[12] = float(bool(a[4] & b[4]))
        v[13] = float(bool(a[4]) and bool(b[4]) and not (a[4] & b[4]))
        ha, hb = a[7], b[7]
        v[14] = float(bool(ha & hb))
        v[15] = float(bool(ha) and bool(hb) and not (ha & hb))
        pa_, pb_ = a[6], b[6]
        v[16] = float(pa_ is not None and pb_ is not None)
        v[17] = float(pa_ is not None and pa_ == pb_)
        v[18] = float(pa_ is not None and pb_ is not None and pa_ != pb_)
        v[19] = float(bool(a[5] & b[5]))
        v[20] = float(bool(a[5]) and bool(b[5]) and not (a[5] & b[5]))
        v[21], v[22] = a[11], b[11]
        v[23] = min(a[11], b[11])
        for c_, val in zip(cols, v):
            c_[k] = val
    out.update(F)

    # --- flags (vectorized)
    am = store.flags["address_missing"]
    out["address_missing_s1"] = am[px].astype(np.float32)
    out["address_missing_cand"] = am[py].astype(np.float32)
    out["address_missing_either"] = np.maximum(out["address_missing_s1"], out["address_missing_cand"])
    sx = np.asarray([X["name_script"][i] or "" for i in ix], dtype=object)
    sy = np.asarray([Y["name_script"][j] or "" for j in iy], dtype=object)
    out["cross_script"] = ((sx != "") & (sy != "") & (sx != sy)).astype(np.float32)

    # --- neighbourhood (within each S1's candidate list, vectorized)
    s1 = p["s1_id"].to_numpy(dtype=object)
    score = out["stage1_score"].astype(np.float64)
    out["stage1_score_gap"] = (pd.Series(score).groupby(s1).transform("max").to_numpy() - score).astype(np.float32)
    rank = out["stage1_rank"].astype(np.int64)
    first = (pd.DataFrame({"s1": s1, "rank": rank, "row": np.arange(n)})
             .sort_values(["s1", "rank"], kind="stable").drop_duplicates("s1").set_index("s1")["row"])
    best_row = pd.Series(s1).map(first).to_numpy()
    y_name = [Y["name_core"][j] for j in iy]
    y_addr = [Y["address_tokens_norm"][j] for j in iy]
    out["to_s1_best_name_ratio"] = batch_scores(y_name, [y_name[r] for r in best_row], fuzz.ratio, True)
    out["to_s1_best_addr_ratio"] = batch_scores(y_addr, [y_addr[r] for r in best_row], fuzz.ratio, True)
    key = pd.Series([f"{Y['name_roman'][j] or ''}\x1f{Y['address_roman'][j] or ''}" for j in iy])
    valid = np.array([bool(Y["name_roman"][j]) and bool(Y["address_roman"][j]) for j in iy])
    cnt = key.groupby([s1, key.to_numpy()]).transform("size").to_numpy()
    out["identical_copy_present"] = ((cnt > 1) & valid).astype(np.float32)

    df = pd.DataFrame(out)
    write_table(df, Path(G["out_dir"]) / Path(base).name)
    return n, time.time() - t0


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--stage1-dir", default="data_stage1")
    ap.add_argument("--repr-dir", default="data_repr")
    ap.add_argument("--out-dir", default="data_stage2")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit-shards", type=int, default=0, help="only this many shards (timing test)")
    ap.add_argument("--start-method", choices=("auto", "fork", "spawn"), default="auto")
    a = ap.parse_args()

    inp, out = Path(a.stage1_dir) / a.split, Path(a.out_dir) / a.split
    out.mkdir(parents=True, exist_ok=True)
    src_split = "test" if a.split == "test" else "train"        # val S1s live in train_source1
    todo = [b for b in shards(inp) if not shard_done(out / Path(b).name)]
    n_all = len(shards(inp))
    if a.limit_shards:
        todo = todo[:a.limit_shards]
    n_done = sum(shard_done(out / Path(b).name) for b in shards(inp))
    log(f"{a.split}: {n_all} Stage-1 shards, {n_done} already done, {len(todo)} to do now; "
        f"rapidfuzz cpdist: {'yes' if HAS_CPDIST else 'NO (slower per-pair fallback)'}")
    if not todo:
        log("nothing to do")
        return

    method = a.start_method
    if method == "auto":
        method = "fork" if sys.platform.startswith("linux") else "spawn"
    state = {"repr_dir": str(a.repr_dir), "split": src_split, "out_dir": str(out)}
    if method == "fork":
        log("loading records once (shared with forked workers)")
        G.update(state)
        G["store"] = load_store(Path(a.repr_dir), src_split)
        pool_size = int(G["store"].n - len(read_table(Path(a.repr_dir) / f"{src_split}_source1", columns=["entity_id"])))
        G["pool_size"] = pool_size
        G["idf_name"], G["idf_addr"] = load_idf(Path(a.repr_dir), src_split, pool_size)
        log(f"records: {G['store'].n:,} (pool {pool_size:,}); IDF tokens: name {len(G['idf_name']):,}, "
            f"address {len(G['idf_addr']):,}")
    else:
        pool_size = sum(len(read_table(Path(a.repr_dir) / f"{src_split}_source{k}", columns=["entity_id"]))
                        for k in (2, 3))
        state["pool_size"] = pool_size

    ctx = mp.get_context(method)
    done, rows, busy, t_start, next_log = 0, 0, 0.0, time.time(), 0.0
    with ctx.Pool(a.workers, initializer=init_worker, initargs=(state,)) as pool:
        for n, sec in pool.imap_unordered(build_shard, todo):
            done += 1
            rows += n
            busy += sec
            frac = done / len(todo)
            if frac >= next_log or done == len(todo):
                el = time.time() - t_start
                eta = el / frac - el
                log(f"   {done}/{len(todo)} shards, {rows:,} pairs, "
                    f"{rows / max(el, 1e-9):,.0f} pairs/s, {busy / max(rows, 1) * 1e6:.0f} us/pair/core, "
                    f"ETA {eta / 60:.1f} min")
                next_log += 0.05
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
