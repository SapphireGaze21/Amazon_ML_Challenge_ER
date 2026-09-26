"""
Blocking library (used by phase5_blocking.py).

Why this design (numbers from Phase 1/4):
  - Exact keys (sorted core name, glued name, alias) catch 64% of true pairs with tiny blocks
    (median 4-6 records)  -> cheap hash-join pass.
  - Rare single tokens catch 94-99% of true pairs, but a random record also shares a rare
    token with ~0.2% of the pool = ~10,000 candidates per S1 -> single-token blocks are
    far too big to use unranked.
  - The misses are mostly GENERIC names ("eastern projects", "good trading", "modern
    industries") whose words are each common, but whose word PAIRS are rare.
So the workhorse is a weighted retrieval: every record becomes a sparse vector of hashed
features (name words, phonetic keys, word pairs, phonetic pairs, address words, adjacent
address word pairs, numbers, numeric compounds), weighted by IDF over the pool, with very
common features dropped (max_df). The dot product S1 x pool scores every candidate that
shares at least one feature; we keep the top K per S1. Common words still contribute
through the pair features, and a candidate sharing many weak features outranks one
sharing a single strong feature.

A character n-gram TF-IDF pass on the glued name adds robustness to typos inside words
("renbewable", "sttrabteic") and squashed names ("earnosethroat").

Everything is computed per country (Phase 0: 0 of 6.1M true pairs cross countries).
"""

import zlib
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DIM_BITS = 24
DIM = 1 << DIM_BITS  # 16.7M hash buckets; collisions only add a few spurious candidates
_MASK = DIM - 1

PASSES = ["exact_sorted", "exact_concat", "exact_alias", "sparse_comb", "sparse_name",
          "sparse_addr", "char_name"]
PASS_BIT = {p: 1 << i for i, p in enumerate(PASSES)}


# ============================================================================ features

def _s(v):
    return v if isinstance(v, str) and v else ""


def name_features(core, phon, max_pair_tokens: int = 6) -> list[str]:
    ct = list(dict.fromkeys(_s(core).split()))
    pk = list(dict.fromkeys(_s(phon).split()))
    f = ["n:" + t for t in ct] + ["p:" + k for k in pk]
    u = sorted(ct)[:max_pair_tokens]
    f += ["nb:" + a + "|" + b for a, b in combinations(u, 2)]
    up = sorted(pk)[:max_pair_tokens]
    f += ["pb:" + a + "|" + b for a, b in combinations(up, 2)]
    return f


def address_features(norm, nums, comps) -> list[str]:
    at = _s(norm).split()
    f = ["a:" + t for t in dict.fromkeys(at)]
    f += ["ab:" + at[i] + "|" + at[i + 1] for i in range(len(at) - 1)]
    f += ["#:" + x for x in dict.fromkeys(_s(nums).split())]
    f += ["c:" + x for x in dict.fromkeys(_s(comps).split())]
    return f


def _hash(f: str) -> int:
    return zlib.crc32(f.encode("utf-8")) & _MASK


def feature_chunk(rows):
    """rows: list of (core, phon, addr_norm, nums, comps) -> hashed name/address rows.
    Returns (name_lengths, name_indices, addr_lengths, addr_indices) as numpy arrays."""
    nl, ni, al, ai = [], [], [], []
    for core, phon, norm, nums, comps in rows:
        h = sorted({_hash(x) for x in name_features(core, phon)})
        nl.append(len(h))
        ni.extend(h)
        h = sorted({_hash(x) for x in address_features(norm, nums, comps)})
        al.append(len(h))
        ai.extend(h)
    return (np.asarray(nl, np.int32), np.asarray(ni, np.int32),
            np.asarray(al, np.int32), np.asarray(ai, np.int32))


def csr_from_parts(lengths_list, indices_list) -> sp.csr_matrix:
    lengths = np.concatenate(lengths_list) if lengths_list else np.zeros(0, np.int32)
    indices = np.concatenate(indices_list) if indices_list else np.zeros(0, np.int32)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(lengths), DIM))


def idf_weight(A: sp.csr_matrix, B: sp.csr_matrix, max_df: int) -> dict:
    """IDF from the pool B; features with df > max_df (too common to be useful and too
    expensive to multiply) or absent from the pool are dropped. Rows are L2-normalised,
    so each group's score is a cosine in [0, 1]. Modifies A and B in place."""
    df = np.bincount(B.indices, minlength=B.shape[1])
    n = B.shape[0]
    keep = (df >= 1) & (df <= max_df)
    w = np.where(keep, np.log((n + 1) / (df + 1)) + 1.0, 0.0).astype(np.float32)
    for M in (A, B):
        M.data = w[M.indices]
        M.eliminate_zeros()
        normalize(M, norm="l2", copy=False)
    return {"features_in_pool": int((df > 0).sum()), "features_dropped_common": int((df > max_df).sum())}


def char_vectors(pool_text, s1_text, ngram: int, max_df: int):
    """Char n-gram TF-IDF on the glued name. max_df (absolute) drops very common n-grams,
    which keeps the product sparse."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(ngram, ngram), min_df=2, max_df=max_df,
                          dtype=np.float32, sublinear_tf=True)
    pad = lambda xs: [f" {x} " if isinstance(x, str) and x else "" for x in xs]  # noqa: E731
    B = vec.fit_transform(pad(pool_text)).tocsr()
    A = vec.transform(pad(s1_text)).tocsr()
    return A, B, {"char_vocab": int(len(vec.vocabulary_))}


# ============================================================================ top-k

def topk_rows(C: sp.csr_matrix, k: int, min_score: float, row_offset: int):
    """Top-k columns per row of a CSR score matrix -> (rows, cols, scores)."""
    C.sum_duplicates()
    indptr, idx, dat = C.indptr, C.indices, C.data
    R, Cc, S = [], [], []
    for i in range(C.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if a == b:
            continue
        s = dat[a:b]
        if b - a > k:
            sel = np.argpartition(-s, k - 1)[:k]
        else:
            sel = np.arange(b - a)
        if min_score > 0:
            sel = sel[s[sel] >= min_score]
        if len(sel):
            R.append(np.full(len(sel), i + row_offset, dtype=np.int32))
            Cc.append(idx[a:b][sel].astype(np.int32))
            S.append(s[sel].astype(np.float32))
    if not R:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    return np.concatenate(R), np.concatenate(Cc), np.concatenate(S)


# Worker globals (set in the parent before the Pool is created; inherited by fork)
G = {}


def sparse_worker(bounds):
    lo, hi = bounds
    out = {}
    Cn = G["A_n"][lo:hi] @ G["BT_n"]
    Ca = G["A_a"][lo:hi] @ G["BT_a"]
    Cc = Cn * G["w_name"] + Ca * G["w_addr"]
    out["sparse_comb"] = topk_rows(Cc, G["k_comb"], G["min_score"], lo)
    out["sparse_name"] = topk_rows(Cn, G["k_name"], G["min_score"], lo)
    out["sparse_addr"] = topk_rows(Ca, G["k_addr"], G["min_score"], lo)
    if G.get("A_ch") is not None:
        Ch = G["A_ch"][lo:hi] @ G["BT_ch"]
        out["char_name"] = topk_rows(Ch, G["k_char"], G["min_score"], lo)
    return out


# ============================================================================ exact keys

def exact_pass(s1_keys: np.ndarray, pool_keys: np.ndarray, max_block: int):
    """Hash join on equal non-empty keys; blocks larger than max_block are skipped.
    Returns (s1_positions, pool_positions)."""
    import pandas as pd
    a = pd.DataFrame({"k": s1_keys, "s1": np.arange(len(s1_keys), dtype=np.int32)})
    b = pd.DataFrame({"k": pool_keys, "c": np.arange(len(pool_keys), dtype=np.int32)})
    a = a[a.k.map(lambda v: isinstance(v, str) and v != "")]
    b = b[b.k.map(lambda v: isinstance(v, str) and v != "")]
    size = b.groupby("k").size()
    b = b[b.k.map(size) <= max_block]
    m = a.merge(b, on="k")
    return m.s1.to_numpy(np.int32), m.c.to_numpy(np.int32)


# ============================================================================ merging

def merge_candidates(parts: list, n_pool: int):
    """parts: list of (pass_name, s1_pos, cand_pos, score). Returns a DataFrame with one row
    per (s1, cand) - the full union, not yet trimmed: passes bitmask, best score per pass,
    priority and rank within S1 (0 = best).
    priority = sum over passes of (score / best score of that pass for this S1): a candidate
    found by several passes, near the top of each, ranks first. Pure numpy (fast on 10M+ rows)."""
    import pandas as pd
    cols = ["s1", "cand", "passes", "priority", "rank"] + [f"score_{p}" for p in PASSES]
    parts = [x for x in parts if len(x[1])]
    if not parts:
        return pd.DataFrame(columns=cols)
    s1 = np.concatenate([x[1] for x in parts]).astype(np.int64)
    cand = np.concatenate([x[2] for x in parts]).astype(np.int64)
    ps = np.concatenate([np.full(len(x[1]), PASSES.index(x[0]), np.int64) for x in parts])
    sc = np.concatenate([x[3] for x in parts]).astype(np.float32)
    key = s1 * n_pool + cand

    # 1. one row per (pair, pass), keeping the best score
    kp = key * 8 + ps
    o = np.lexsort((-sc, kp))
    first = np.ones(len(o), bool)
    first[1:] = kp[o][1:] != kp[o][:-1]
    o = o[first]
    s1, cand, ps, sc, key = s1[o], cand[o], ps[o], sc[o], key[o]

    # 2. normalise each score by the best score of that pass for that S1
    g = s1 * 8 + ps
    og = np.argsort(g, kind="stable")
    gs = g[og]
    starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    mx = np.maximum.reduceat(sc[og], starts)
    best = np.empty_like(sc)
    best[og] = np.repeat(mx, np.diff(np.r_[starts, len(og)]))
    norm = np.where(best > 0, sc / np.where(best > 0, best, 1), 1.0).astype(np.float32)

    # 3. aggregate per pair
    uk, kinv = np.unique(key, return_inverse=True)
    passes = np.bincount(kinv, weights=np.left_shift(1, ps), minlength=len(uk)).astype(np.int32)
    priority = np.bincount(kinv, weights=norm, minlength=len(uk)).astype(np.float32)
    out = {"s1": (uk // n_pool).astype(np.int32), "cand": (uk % n_pool).astype(np.int32),
           "passes": passes, "priority": priority}
    for i, p in enumerate(PASSES):
        col = np.full(len(uk), np.nan, np.float32)
        m = ps == i
        col[kinv[m]] = sc[m]
        out[f"score_{p}"] = col

    # 4. rank within S1 by priority
    o = np.lexsort((-priority, out["s1"]))
    s1o = out["s1"][o]
    starts = np.r_[0, np.flatnonzero(s1o[1:] != s1o[:-1]) + 1]
    rank = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    df = pd.DataFrame({k: v[o] for k, v in out.items()})
    df["rank"] = rank.astype(np.int32)
    return df[cols]
