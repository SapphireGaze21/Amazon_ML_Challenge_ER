"""
Blocking library (used by phase5_blocking.py).

Design, driven by Phase 4 and the first blocking run (val_s50000):

  GENERATION (cheap, capped): each record becomes five sparse hashed feature vectors
    name   : core words, phonetic keys, word pairs, phonetic pairs
    addr   : address words, adjacent address word pairs, numbers, numeric compounds
    cross  : name key x address word ("fortune|ghaziabad", "baba|106") - rare even when both
             parts are common; fixes generic/duplicate names whose city was capped away
    cname  : character 3-grams of the glued name   (typos inside words, squashed names)
    caddr  : character 3-grams of the address      (typos in street / city names)
  IDF-weighted over the pool; features more common than max_df are dropped FOR GENERATION
  ONLY (they make the sparse products slow). Per S1 we take the top-K of several scores and
  the exact-key joins, and union them.

  RE-RANKING (per candidate, uncapped): run 1 ranked true pairs 202-961 because the old
  priority counted passes, so hundreds of same-name records in a big exact block outranked a
  match with an IDENTICAL address. Now every union candidate is scored with the full,
  uncapped cosine of each group, and ranked by
      priority = w_name*cos_name + w_addr*cos_addr + w_cross*cos_cross
               + w_cname*cos_cname + w_caddr*cos_caddr + w_exact*[any exact key]
  The five cosines are stored with the candidates (useful matcher features later).

  v3 additions (from the val_s50000 misses: cross-script 86% and address-missing 86% recall,
  vs 97% elsewhere; half of all misses were ranked out, not unfound):
    phon group + sparse_phon pass   phonetic keys and phonetic pairs alone (typos, scripts)
    slice passes                    top-K restricted to pool records whose address is missing
                                    (noaddr_name) and to non-Latin names (nonlatin_phon). In the
                                    full pool these compete with Latin/addressed records that
                                    share exact words, so they lose their top-K slots; within
                                    their slice they do not. This is the per-segment budget:
                                    the segments are properties of the CANDIDATE, so the extra
                                    budget is given to candidate slices, not to S1s.
    fairer re-ranking               name evidence = max(cos_name, cos_phon); a candidate with no
                                    address gets its address terms imputed from its name
                                    evidence (x missing_addr_factor) instead of zero
    expand_prf pass                 pseudo-relevance feedback / transitive expansion: the query
                                    of each S1 is extended with its strongest candidates
                                    (seeds) and retrieval is repeated, which finds records that
                                    resemble a sibling match more than the S1 itself.

  v3 changes driven by the val_s50000_v2 misses:
    name informativeness   cosine gives a one-word name 1.0 for an exact match whether the word
                           is rare or appears 20,000 times, so hundreds of "ventures" outranked
                           a true match with an IDENTICAL address. Name terms are now scaled per
                           S1 by clip(name IDF norm / country median, floor, 1).
    sparse_cross pass      descriptor swaps ("black technology" vs "black services") with a
                           truncated address share only first word + number/city: the cross
                           features alone, not diluted inside the combined score.
    coarse phonetics       transliterated English ("imphrastrakchar", "haitek") differs from the
                           Latin key by 1-2 letters: coarse keys merge c/k/q, m/n, b/v, j/g and a
                           final -ng; the cphon group adds char 3-grams over the coarse keys.
    PRF expansion          available (--k-prf) but off by default: recall is flat across match
                           counts, so misses are not sibling-lookalikes.

Everything runs per country (Phase 0: 0 of 6.1M true pairs cross countries). Generation,
union and re-ranking all happen inside the worker processes.
"""

import zlib
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from represent import phonetic_key

DIM_BITS = 24
DIM = 1 << DIM_BITS  # 16.7M hash buckets; collisions only add a few spurious candidates
_MASK = DIM - 1

GROUPS = ["name", "addr", "cross", "cname", "caddr", "phon", "cphon"]
PASSES = ["exact_sorted", "exact_concat", "exact_alias", "sparse_comb", "sparse_name",
          "sparse_addr", "char_name", "char_addr", "sparse_phon", "noaddr_name",
          "nonlatin_phon", "expand_prf", "sparse_cross"]
NAME_GROUPS = ("name", "phon", "cphon")
PASS_BIT = {p: 1 << i for i, p in enumerate(PASSES)}
EXACT_MASK = PASS_BIT["exact_sorted"] | PASS_BIT["exact_concat"] | PASS_BIT["exact_alias"]
COS_COLS = [f"cos_{g}" for g in GROUPS]


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


_COARSE = str.maketrans({"c": "k", "q": "k", "m": "n", "b": "v", "j": "g"})


def coarse_key(k: str) -> str:
    """Coarser phonetic key for transliterated English: c/k/q, m/n (anusvara), b/v (Bengali),
    j/g merged, repeats collapsed, a final -ng reduced to -n ("injiniyarin" ~ "engineering")."""
    t = k.translate(_COARSE)
    out = [t[0]] if t else []
    for ch in t[1:]:
        if ch != out[-1]:
            out.append(ch)
    t = "".join(out)
    return t[:-1] if t.endswith("ng") and len(t) > 2 else t


def phon_features(phon, max_pair_tokens: int = 6) -> list[str]:
    pk = list(dict.fromkeys(_s(phon).split()))
    ck = list(dict.fromkeys(coarse_key(k) for k in pk))
    up, uc = sorted(pk)[:max_pair_tokens], sorted(ck)[:max_pair_tokens]
    return (["q:" + k for k in pk] + ["qb:" + a + "|" + b for a, b in combinations(up, 2)] +
            ["qc:" + k for k in ck] + ["qcb:" + a + "|" + b for a, b in combinations(uc, 2)])


def cphon_text(phon) -> str:
    return " ".join(coarse_key(k) for k in _s(phon).split())


def cross_features(core, norm, max_keys: int = 4, max_addr: int = 12) -> list[str]:
    """Name key x address word. Name key = phonetic key (robust to spelling/script), or the
    word itself when it has no key (short words like 'om', 'sai', numbers)."""
    keys = []
    for t in dict.fromkeys(_s(core).split()):
        pk = phonetic_key(t)
        k = coarse_key(pk) if pk else (t if len(t) >= 2 else None)
        if k and k not in keys:
            keys.append(k)
    addr = [t for t in dict.fromkeys(_s(norm).split()) if len(t) >= 2][:max_addr]
    return ["x:" + k + "|" + t for k in keys[:max_keys] for t in addr]


def char_features(text, prefix: str, n: int = 3) -> list[str]:
    t = _s(text)
    if not t:
        return []
    t = f" {t} "
    return [prefix + t[i:i + n] for i in range(len(t) - n + 1)]


def _hash(f: str) -> int:
    return zlib.crc32(f.encode("utf-8")) & _MASK


def feature_chunk(rows):
    """rows: list of (core, phon, addr_norm, nums, comps, concat).
    Returns {group: (lengths, indices)} of hashed, de-duplicated features."""
    acc = {g: ([], []) for g in GROUPS}
    for core, phon, norm, nums, comps, concat in rows:
        concat = _s(concat)
        if len(concat) >= 10 and concat.endswith("com"):      # "superagrocom" -> "superagro"
            concat = concat[:-3]
        feats = {"name": name_features(core, phon),
                 "addr": address_features(norm, nums, comps),
                 "cross": cross_features(core, norm),
                 "cname": char_features(concat, "cn:"),
                 "caddr": char_features(norm, "ca:"),
                 "phon": phon_features(phon),
                 "cphon": char_features(cphon_text(phon), "cq:")}
        for g, fl in feats.items():
            h = sorted({_hash(x) for x in fl})
            acc[g][0].append(len(h))
            acc[g][1].extend(h)
    return {g: (np.asarray(l, np.int32), np.asarray(i, np.int32)) for g, (l, i) in acc.items()}


def csr_from_parts(lengths_list, indices_list) -> sp.csr_matrix:
    lengths = np.concatenate(lengths_list) if lengths_list else np.zeros(0, np.int32)
    indices = np.concatenate(indices_list) if indices_list else np.zeros(0, np.int32)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    return sp.csr_matrix((np.ones(len(indices), np.float32), indices, indptr), shape=(len(lengths), DIM))


def weighted(A_raw: sp.csr_matrix, B_raw: sp.csr_matrix, max_df: int):
    """IDF from the pool B. Returns (A_cap, B_cap, A_full, B_full, info):
    *_full keeps every feature seen in the pool (for re-ranking);
    *_cap also drops features with df > max_df (for fast candidate generation).
    All rows are L2-normalised, so every score is a cosine in [0, 1]."""
    df = np.bincount(B_raw.indices, minlength=B_raw.shape[1])
    n = B_raw.shape[0]
    idf = np.where(df >= 1, np.log((n + 1) / (df + 1)) + 1.0, 0.0).astype(np.float32)
    cap = np.where(df <= max_df, idf, 0.0).astype(np.float32)
    out = []
    for w in (cap, idf):
        for M in (A_raw, B_raw):
            X = sp.csr_matrix((w[M.indices], M.indices.copy(), M.indptr.copy()), shape=M.shape)
            X.eliminate_zeros()
            normalize(X, norm="l2", copy=False)
            out.append(X)
    A_cap, B_cap, A_full, B_full = out
    sq = sp.csr_matrix((idf[A_raw.indices] ** 2, A_raw.indices, A_raw.indptr), shape=A_raw.shape)
    a_norm = np.sqrt(np.asarray(sq.sum(axis=1)).ravel()).astype(np.float32)
    info = {"features_in_pool": int((df > 0).sum()), "dropped_for_generation": int((df > max_df).sum())}
    return A_cap, B_cap, A_full, B_full, info, a_norm


# ============================================================================ worker

def topk_rows(C: sp.csr_matrix, k: int, row_offset: int):
    """Top-k columns per row of a CSR score matrix -> (rows, cols)."""
    C.sum_duplicates()
    indptr, idx, dat = C.indptr, C.indices, C.data
    R, Cc = [], []
    for i in range(C.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if a == b:
            continue
        sel = np.argpartition(-dat[a:b], k - 1)[:k] if b - a > k else np.arange(b - a)
        R.append(np.full(len(sel), i + row_offset, dtype=np.int64))
        Cc.append(idx[a:b][sel].astype(np.int64))
    if not R:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(R), np.concatenate(Cc)


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, r: np.ndarray, c: np.ndarray, batch: int = 400_000):
    """cos(A[r_i], B[c_i]) for each pair (rows are L2-normalised)."""
    out = np.zeros(len(r), np.float32)
    for i in range(0, len(r), batch):
        X, Y = A[r[i:i + batch]], B[c[i:i + batch]]
        out[i:i + batch] = np.asarray(X.multiply(Y).sum(axis=1)).ravel()
    return out


def union_pairs(parts, n_pool: int):
    """parts: list of (pass_name, s1_rows, cand_rows) -> unique (s1, cand, passes bitmask)."""
    parts = [p for p in parts if len(p[1])]
    if not parts:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.int32)
    key = np.concatenate([p[1] * n_pool + p[2] for p in parts])
    bit = np.concatenate([np.full(len(p[1]), PASS_BIT[p[0]], np.int64) for p in parts])
    uk, inv = np.unique(key, return_inverse=True)
    passes = np.zeros(len(uk), np.int64)
    np.bitwise_or.at(passes, inv, bit)
    return uk // n_pool, uk % n_pool, passes.astype(np.int32)


def rank_within(s1: np.ndarray, priority: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order by (s1, priority desc); returns (order, rank)."""
    o = np.lexsort((-priority, s1))
    s = s1[o]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    rank = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    return o, rank.astype(np.int32)


def priority_of(cos: dict, passes: np.ndarray, cand_addr_missing: np.ndarray, w: dict,
                info: np.ndarray) -> np.ndarray:
    """Re-ranking score.
    - name evidence = max(name, phonetic, coarse-phonetic) cosine (transliterations are not
      punished for spelling), scaled by the S1 name's informativeness (a common one-word name
      matching exactly is weak evidence);
    - when the CANDIDATE has no address, the address terms are imputed from the name evidence
      (x missing_addr_factor) instead of counting as 0."""
    name_ev = np.maximum(np.maximum(cos["name"], cos["phon"]), cos["cphon"]) * info
    addr_terms = w["addr"] * cos["addr"] + w["cross"] * cos["cross"] + w["caddr"] * cos["caddr"]
    imputed = w["missing_addr_factor"] * (w["addr"] + w["cross"] + w["caddr"]) * name_ev
    addr_terms = np.where(cand_addr_missing, imputed, addr_terms)
    return (w["name"] * name_ev + addr_terms + w["cname"] * cos["cname"] * info
            + w["exact"] * info * ((passes & EXACT_MASK) > 0)).astype(np.float32)


# Worker globals: set in the parent before the Pool is created, inherited by fork.
G = {}


def _slice_topk(C: sp.csr_matrix, k: int, lo: int, idx: np.ndarray):
    r, c = topk_rows(C, k, lo)
    return r, idx[c]


def _score_union(parts, lo, hi):
    s1, cand, passes = union_pairs(parts, G["n_pool"])
    M = G["M"]
    cos = {g: rowdot(M[g]["A_full"], M[g]["B_full"], s1, cand) for g in GROUPS}
    pr = priority_of(cos, passes, G["pool_addr_missing"][cand], G["w"], G["name_info"][s1])
    return s1, cand, passes, cos, pr


def block_worker(bounds):
    """Generate candidates for S1 rows [lo, hi), union them, re-rank with full cosines,
    expand from the strongest candidates (PRF), re-rank again. Returns the ranked union."""
    lo, hi = bounds
    k, w, M, SUB = G["k"], G["w"], G["M"], G["sub"]
    parts = []

    # ---- stage 1: generation
    A = {g: M[g]["A_cap"][lo:hi] for g in GROUPS}
    C = {g: A[g] @ M[g]["BT_cap"] for g in GROUPS}
    info = sp.diags(G["name_info"][lo:hi])
    comb = (info @ C["name"]) * w["name"] + C["addr"] * w["addr"] + C["cross"] * w["cross"]
    phon = C["phon"] + C["cphon"]
    for name, mat in (("sparse_comb", comb), ("sparse_name", C["name"]), ("sparse_addr", C["addr"]),
                      ("char_name", C["cname"]), ("char_addr", C["caddr"]), ("sparse_phon", phon),
                      ("sparse_cross", C["cross"])):
        if k.get(name, 0) > 0:
            parts.append((name,) + topk_rows(mat.tocsr(), k[name], lo))
    del C, comb, phon
    if k.get("noaddr_name", 0) > 0 and len(SUB["noaddr"]["idx"]):
        T = SUB["noaddr"]
        Cs = A["name"] @ T["BT_name"] + A["phon"] @ T["BT_phon"] + A["cname"] @ T["BT_cname"]
        parts.append(("noaddr_name",) + _slice_topk(Cs, k["noaddr_name"], lo, T["idx"]))
    if k.get("nonlatin_phon", 0) > 0 and len(SUB["nonlatin"]["idx"]):
        T = SUB["nonlatin"]
        Cs = A["phon"] @ T["BT_phon"] + A["cphon"] @ T["BT_cphon"] + A["addr"] @ T["BT_addr"]
        parts.append(("nonlatin_phon",) + _slice_topk(Cs, k["nonlatin_phon"], lo, T["idx"]))
    for name, (r, c) in G["exact"].items():            # r is sorted
        i0, i1 = np.searchsorted(r, lo), np.searchsorted(r, hi)
        parts.append((name, r[i0:i1].astype(np.int64), c[i0:i1].astype(np.int64)))

    s1, cand, passes, cos, pr = _score_union(parts, lo, hi)

    # ---- stage 2: pseudo-relevance feedback from the strongest candidates
    if k.get("expand_prf", 0) > 0 and len(s1):
        o, rank = rank_within(s1, pr)
        seed = o[(rank < G["prf"]["seeds"]) & (pr[o] >= G["prf"]["min_priority"])]
        if len(seed):
            rows = (s1[seed] - lo).astype(np.int64)
            cnt = np.bincount(rows, minlength=hi - lo).astype(np.float32)
            S = sp.csr_matrix((1.0 / cnt[rows], (rows, cand[seed])), shape=(hi - lo, G["n_pool"]))
            has = cnt > 0
            prf = None
            for g, wg in (("name", w["name"]), ("addr", w["addr"]), ("cross", w["cross"])):
                Q = A[g] + G["prf"]["alpha"] * (S @ M[g]["B_full"])
                part = (Q @ M[g]["BT_cap"]) * wg
                prf = part if prf is None else prf + part
            prf = sp.diags(has.astype(np.float32)) @ prf     # only S1s that had seeds
            parts.append(("expand_prf",) + topk_rows(prf.tocsr(), k["expand_prf"], lo))
            s1, cand, passes, cos, pr = _score_union(parts, lo, hi)

    o, rank = rank_within(s1, pr)
    out = {"s1": s1[o].astype(np.int32), "cand": cand[o].astype(np.int32),
           "passes": passes[o], "priority": pr[o], "rank": rank}
    for g in GROUPS:
        out[f"cos_{g}"] = cos[g][o]
    return out


# ============================================================================ exact keys

def exact_pass(s1_keys: np.ndarray, pool_keys: np.ndarray, max_block: int):
    """Hash join on equal non-empty keys; blocks larger than max_block are skipped.
    Returns (s1_positions, pool_positions) sorted by s1 position."""
    import pandas as pd
    a = pd.DataFrame({"k": s1_keys, "s1": np.arange(len(s1_keys), dtype=np.int64)})
    b = pd.DataFrame({"k": pool_keys, "c": np.arange(len(pool_keys), dtype=np.int64)})
    a = a[a.k.map(lambda v: isinstance(v, str) and v != "")]
    b = b[b.k.map(lambda v: isinstance(v, str) and v != "")]
    size = b.groupby("k").size()
    b = b[b.k.map(size) <= max_block]
    m = a.merge(b, on="k").sort_values("s1", kind="stable")
    return m.s1.to_numpy(np.int64), m.c.to_numpy(np.int64)
