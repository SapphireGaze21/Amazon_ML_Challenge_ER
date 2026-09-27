#!/usr/bin/env python3
"""
Phase 9 - improve the Stage-2 matcher. One script, checkpointed stages, re-runnable.

Starts from Phase 8's cached tables (artifacts/phase8/cache/{train,val}, val_pred, decision.json)
and data_stage2/test. Writes a NEW submission to output_v2/ (the Phase 8 baseline is untouched).

Stages
  errors      Step 1 - every validation error is categorised (not just the 300 most confident),
              category shares are reported, and readable examples (names + addresses) are
              written to errors_report.md.
  skew        Step 2 - measures the Stage-1 score distribution on train (out-of-fold) vs
              validation (final model) and trains three fast single-seed variants:
                all        all features (the fair baseline for every experiment)
                no_stage1  without stage1_score / stage1_score_gap / stage1_rank
                rel_stage1 Stage-1 score replaced by within-S1 relative versions (ratio to the
                           S1's best, z-score within the S1) - robust to a shift in scale
  stack       Step 4 - stacked second pass on the best feature set from `skew`: out-of-fold
              first-pass probabilities on train (K folds by S1), full-model probabilities on
              validation, then cross-candidate features per candidate (what the S1's OTHER
              candidates say about it: best probability among candidates sharing its address /
              house number / postal code / name, similarity to the S1's confident candidates,
              own rank...), and a second model on base + cross features.
  xgb         Step 5 - XGBoost (Apache 2.0) on the best feature set so far, and a LightGBM+XGBoost
              blend. Skipped with a warning if xgboost is not installed.
  select      Every variant is tuned on half A with Step 3's decision layer (monotone
              calibration fitted PER GROUP - candidates with / without an address - then either
              per-group thresholds or expected-F0.5 selection, with / without one-to-one) and
              scored on half B. The best variant on A is chosen; B is the honest estimate.
  final       the chosen variant retrained properly (--final-seeds seeds, --final-lr), decision
              re-tuned on A / scored on B / refit on all validation for test.
  test        Step 7 - test prediction IN-PROCESS (LightGBM/XGBoost use all cores themselves; the
              Phase 8 per-shard single-thread workers took 83 min for 17M pairs), cross features
              for test if stacking was chosen, decision, output_v2/ files, README checks.

Usage
  python src/phase9_improve.py                        # everything (resumes)
  python src/phase9_improve.py --from-stage select    # redo selection and later stages
  python src/phase9_improve.py --stop-after select    # experiments only, no final/test yet
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import phase8_match as P8
from common import read_table
from rapidfuzz.distance import Levenshtein

T0 = time.time()
STAGES = ["errors", "skew", "stack", "xgb", "select", "final", "test"]
STAGE1_FEATS = ["stage1_score", "stage1_score_gap", "stage1_rank"]
CROSS = ["p1", "p1_rank", "p1_ratio", "p1_max_other", "p1_sum_other", "n_conf_other",
         "sib_addr_max", "sib_addr_same_max", "n_same_addr", "sib_num_max", "sib_postal_max",
         "sib_name_max", "sib_phon_max", "conf_sim_addr", "conf_sim_name",
         # round 2 (from the error analysis):
         # decoys = near-identical address, different house number, next to a confident sibling
         "sib_num_conflict_max", "sib_same_name_num_conflict_max",
         # what KIND of number difference: truncation (noise) vs one-digit substitution
         "num_prefix_match", "num_edit1", "num_best_sim", "n_house_s1", "n_house_cand",
         # name ambiguity: how many businesses share this exact name (address-missing misses)
         "exact_sorted_name", "s1_name_n_s1", "cand_name_n_s1", "cand_name_n_pool",
         # global address ambiguity: an exact address is much stronger when it is rare; missing
         # addresses are deliberately no evidence, never a shared pseudo-address.
         "exact_norm_address", "exact_addr_unique_pool", "s1_addr_n_s1",
         "cand_addr_n_s1", "cand_addr_n_pool"]


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


class State:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def done(self, s):
        return (self.root / f"{s}.done.json").exists()

    def mark(self, s, info=None):
        P8.atomic_json(self.root / f"{s}.done.json", {"stage": s, **(info or {})})

    def reset_from(self, s):
        for x in STAGES[STAGES.index(s):]:
            p = self.root / f"{x}.done.json"
            if p.exists():
                p.unlink()


# ============================================================================ feature sets

def add_relative_stage1(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("s1_id", sort=False)["stage1_score"]
    mx, mean, sd = g.transform("max"), g.transform("mean"), g.transform("std").fillna(0)
    df["s1rel_ratio"] = (df["stage1_score"] / (mx + 1e-9)).astype(np.float32)
    df["s1rel_z"] = ((df["stage1_score"] - mean) / (sd + 1e-6)).astype(np.float32)
    return df


def feature_set(name: str, base: list) -> list:
    if name == "all":
        return list(base)
    if name == "no_stage1":
        return [f for f in base if f not in STAGE1_FEATS]
    if name == "rel_stage1":
        return [f for f in base if f not in ("stage1_score", "stage1_score_gap")] + ["s1rel_ratio", "s1rel_z"]
    raise ValueError(name)


# ============================================================================ models

def train_model(kind, X, y, Xv, yv, lr, seed, threads, max_rounds, early_stop, num_leaves, min_leaf,
                rounds=None):
    """kind: lightgbm | xgboost | sklearn (testing only). Returns (model, best_iter)."""
    if kind == "lightgbm":
        import lightgbm as lgb
        prm = {"objective": "binary", "metric": "binary_logloss", "learning_rate": lr,
               "num_leaves": num_leaves, "min_data_in_leaf": min_leaf, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "num_threads": threads, "verbose": -1, "seed": seed}
        dtr = lgb.Dataset(X, label=y, free_raw_data=True)
        if rounds:
            m = lgb.train(prm, dtr, num_boost_round=rounds)
            return m, rounds
        dv = lgb.Dataset(Xv, label=yv, reference=dtr)
        m = lgb.train(prm, dtr, num_boost_round=max_rounds, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(500)])
        return m, int(m.best_iteration or max_rounds)
    if kind == "xgboost":
        import xgboost as xgb
        prm = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": lr,
               "tree_method": "hist", "grow_policy": "lossguide", "max_depth": 0,
               "max_leaves": num_leaves, "min_child_weight": max(1, min_leaf // 10),
               "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "nthread": threads,
               "seed": seed}
        dtr = xgb.DMatrix(X, label=y, nthread=threads)
        if rounds:
            return xgb.train(prm, dtr, num_boost_round=rounds), rounds
        dv = xgb.DMatrix(Xv, label=yv, nthread=threads)
        m = xgb.train(prm, dtr, num_boost_round=max_rounds, evals=[(dv, "holdout")],
                      early_stopping_rounds=early_stop, verbose_eval=500)
        return m, int(m.best_iteration + 1)
    from sklearn.ensemble import HistGradientBoostingClassifier       # tests only (BSD license)
    m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=rounds or min(max_rounds, 150),
                                       max_leaf_nodes=num_leaves, random_state=seed, early_stopping=False)
    m.fit(X, y)
    return m, m.n_iter_


def predict(kind, m, best, X, threads):
    if kind == "lightgbm":
        return m.predict(X, num_iteration=best, num_threads=threads).astype(np.float32)
    if kind == "xgboost":
        import xgboost as xgb
        return m.predict(xgb.DMatrix(X, nthread=threads), iteration_range=(0, best)).astype(np.float32)
    return m.predict_proba(X)[:, 1].astype(np.float32)


def save_model(kind, m, best, base: Path):
    if kind == "lightgbm":
        m.save_model(str(base) + ".txt", num_iteration=best)
    elif kind == "xgboost":
        m.save_model(str(base) + ".json")
    else:
        import pickle
        Path(str(base) + ".pkl").write_bytes(pickle.dumps(m))
    nf = getattr(m, "num_feature", None)
    nf = nf() if callable(nf) else getattr(m, "n_features_in_", None)
    if nf is None and kind == "xgboost":
        nf = int(m.num_features())
    P8.atomic_json(Path(str(base) + ".meta.json"), {"kind": kind, "best": best, "n_features": nf})


def load_model(base: Path):
    meta = json.loads(Path(str(base) + ".meta.json").read_text())
    kind, best = meta["kind"], meta["best"]
    if kind == "lightgbm":
        import lightgbm as lgb
        m = lgb.Booster(model_file=str(base) + ".txt")
    elif kind == "xgboost":
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(str(base) + ".json")
    else:
        import pickle
        m = pickle.loads(Path(str(base) + ".pkl").read_bytes())
    return kind, m, best


def model_exists(base: Path, n_features=None):
    """True if a saved model exists AND (when n_features is given) was trained on that many
    features - a model trained on an older feature set is never reused."""
    p = Path(str(base) + ".meta.json")
    if not p.exists():
        return False
    if n_features is None:
        return True
    return json.loads(p.read_text()).get("n_features") in (None, n_features)


# ============================================================================ decision layer (Step 3)

def tune_decision(p, lab, s1, cand, grp, n_true_A, idx_A):
    """Fit calibration per group (0 = candidate has an address, 1 = it has none) on half A,
    then pick the best of: per-group thresholds (coordinate search) or expected-F0.5, each with /
    without one-to-one. Returns (rule, calibration, F0.5 on A)."""
    a = idx_A
    cal = {}
    pc = np.zeros(len(p), np.float32)
    for gv in (0, 1):
        m = a & (grp == gv)
        if m.sum() >= 200 and 0 < lab[m].sum() < m.sum():
            cal[str(gv)] = P8.fit_calibration(p[m], lab[m])
        else:
            cal[str(gv)] = P8.fit_calibration(p[a], lab[a])
    for gv in (0, 1):
        pc[grp == gv] = P8.apply_calibration(p[grp == gv], cal[str(gv)])
    sA, cA, lA, gA, pA = s1[a], cand[a], lab[a], grp[a], pc[a]
    o2o = {False: pA, True: P8.one_to_one(sA, cA, pA)}

    def score_thr(t0, t1, oo):
        q = o2o[oo]
        pred = np.where(gA == 1, q >= t1, q >= t0)
        return P8.macro_f05(sA, lA, pred, n_true_A)["macro_f05"]

    best = (-1, None)
    grid = np.round(np.arange(0.05, 0.975, 0.025), 3)
    for oo in (False, True):
        t0 = max(grid, key=lambda t: score_thr(t, t, oo))
        t1 = max(grid, key=lambda t: score_thr(t0, t, oo))
        t0 = max(grid, key=lambda t: score_thr(t, t1, oo))
        f = score_thr(t0, t1, oo)
        if f > best[0]:
            best = (f, {"method": "threshold", "t": float(t0), "t_noaddr": float(t1), "one_to_one": oo})
        mu = float((n_true_A.sum() - lA.sum()) / max(len(n_true_A), 1))
        for m_ in (0.0, mu):
            pred = P8.decide_expected_f(sA, o2o[oo], m_)
            f = P8.macro_f05(sA, lA, pred, n_true_A)["macro_f05"]
            if f > best[0]:
                best = (f, {"method": "expected_f", "mu": m_, "one_to_one": oo})
    return best[1], cal, best[0]


def apply_decision(rule, cal, p, s1, cand, grp, gamma=1.0):
    """gamma > 1 makes the decision more conservative (calibrated p -> p ** gamma); it can be a
    scalar or one value per row (used to adapt unseen countries on test)."""
    pc = np.zeros(len(p), np.float32)
    for gv in (0, 1):
        pc[grp == gv] = P8.apply_calibration(p[grp == gv], cal[str(gv)])
    pc = np.power(pc, gamma).astype(np.float32)
    q = P8.one_to_one(s1, cand, pc) if rule["one_to_one"] else pc
    if rule["method"] == "threshold":
        return np.where(grp == 1, q >= rule["t_noaddr"], q >= rule["t"])
    return P8.decide_expected_f(s1, q, rule["mu"])


def evaluate_variant(name, p, V):
    rule, cal, fA = tune_decision(p, V["lab"], V["s1"], V["cand"], V["grp"], V["nA"], V["A"])
    B = V["B"]
    fB = P8.macro_f05(V["s1"][B], V["lab"][B],
                      apply_decision(rule, cal, p[B], V["s1"][B], V["cand"][B], V["grp"][B]), V["nB"])
    log(f"   {name:<22} A {fA:.5f}   B {fB['macro_f05']:.5f}   (P {fB['macro_precision']}, "
        f"R {fB['macro_recall']})  rule {rule}")
    return {"A": fA, "B": fB["macro_f05"], "B_precision": fB["macro_precision"],
            "B_recall": fB["macro_recall"], "rule": rule, "calibration": cal}


# ============================================================================ cross-candidate features (Step 4)

G = {}


EMPTY_REC = (frozenset(), frozenset(), frozenset(), set(), None, "", "")


def _parse(store, pos):
    cols = {c: store.take(c, pos) for c in ("name_core", "name_phon", "address_tokens_norm", "address_nums",
                                             "name_sorted_key")}
    out = []
    for k in range(len(pos)):
        nums = set((cols["address_nums"][k] or "").split())
        postal = P8_postal(nums)
        at = frozenset((cols["address_tokens_norm"][k] or "").split())
        out.append((frozenset((cols["name_core"][k] or "").split()), frozenset((cols["name_phon"][k] or "").split()),
                    at, nums - ({postal} if postal else set()), postal, cols["name_sorted_key"][k] or "",
                    # Preserve the normalized address string, rather than a token set, for exact
                    # global frequency.  Empty remains an explicit no-evidence value.
                    cols["address_tokens_norm"][k] or ""))
    return out


def name_frequencies(store):
    """Exact-name (sorted core key) counts among S1 records and among the S2+S3 pool of a split.
    Both tables are complete for train and for test, so the feature means the same in both."""
    keys = pd.Series(store.take("name_sorted_key", np.arange(store.n)), dtype=object)
    is_s1 = np.empty(store.n, bool)
    is_s1[store.order] = np.char.startswith(store.sorted_ids, "S1-")
    ok = keys.notna() & (keys != "")
    return (keys[ok & is_s1].value_counts().to_dict(), keys[ok & ~is_s1].value_counts().to_dict())


def address_frequencies(store):
    """Exact normalized-address counts, separately for S1 and the candidate pool.

    Address absence is not a key: it is excluded before counting so all missing-address records
    never become artificial siblings.  These are global unsupervised statistics and therefore
    have the same definition on train, validation, and test (including France).
    """
    keys = pd.Series(store.take("address_tokens_norm", np.arange(store.n)), dtype=object)
    is_s1 = np.empty(store.n, bool)
    is_s1[store.order] = np.char.startswith(store.sorted_ids, "S1-")
    ok = keys.notna() & (keys != "")
    return (keys[ok & is_s1].value_counts().to_dict(), keys[ok & ~is_s1].value_counts().to_dict())


def number_pair_features(hs: set, hc: set):
    """(prefix/truncation match, one-digit substitution, best normalised similarity) over the
    house numbers of the S1 and the candidate (postal codes excluded)."""
    if not hs or not hc:
        return 0.0, 0.0, 0.0
    pref = edit1 = 0.0
    best = 0.0
    for x in hs:
        for y in hc:
            if x == y:
                best = 1.0
                continue
            if x.startswith(y) or y.startswith(x):
                pref = 1.0
            d = Levenshtein.distance(x, y)
            if d == 1 and len(x) == len(y):
                edit1 = 1.0
            best = max(best, 1.0 - d / max(len(x), len(y)))
    return pref, edit1, best


def P8_postal(nums):
    x = [n for n in nums if len(n) >= 5]
    return max(x, key=lambda n: (len(n), n)) if x else None


def _jac(a, b):
    if not a or not b:
        return 0.0
    n = len(a & b)
    return n / (len(a) + len(b) - n)


def cross_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """df: rows (s1_id, cand_id, p1) for complete S1 groups -> cross-candidate features."""
    store, nf_s1, nf_pool = G["store"], G["nf_s1"], G["nf_pool"]
    af_s1, af_pool = G["af_s1"], G["af_pool"]
    pos, ok = store.lookup(df["cand_id"].to_numpy(dtype=object))
    up, inv = np.unique(np.where(ok, pos, 0), return_inverse=True)
    parsed = _parse(store, up)
    rec = [parsed[i] if o else EMPTY_REC for i, o in zip(inv, ok)]
    spos, sok = store.lookup(df["s1_id"].to_numpy(dtype=object))
    sup, sinv = np.unique(np.where(sok, spos, 0), return_inverse=True)
    sparsed = _parse(store, sup)
    srec = [sparsed[i] if o else EMPTY_REC for i, o in zip(sinv, sok)]
    p1 = df["p1"].to_numpy(np.float64)
    s1 = df["s1_id"].to_numpy(dtype=object)
    n = len(df)
    F = {c: np.zeros(n, np.float32) for c in CROSS}
    F["p1"] = p1.astype(np.float32)
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1, n]
    for gi in range(len(starts) - 1):
        lo, hi = starts[gi], starts[gi + 1]
        idx = range(lo, hi)
        pg = p1[lo:hi]
        mx = pg.max() if hi > lo else 0.0
        order = np.argsort(-pg, kind="stable")
        rk = np.empty(hi - lo, np.int64)
        rk[order] = np.arange(hi - lo)
        tot = pg.sum()
        s_rec = srec[lo]
        for ii, i in enumerate(idx):
            ni, pi_, ai, hi_, poi, ki, aki = rec[i]
            best_addr = best_same = best_num = best_post = best_name = best_phon = 0.0
            num_conf = name_num_conf = 0.0
            n_same = n_conf = 0
            sa = sn = 0.0
            pmax_o = 0.0
            for jj, j in enumerate(idx):
                if j == i:
                    continue
                pj = pg[jj]
                nj, pj_ph, aj, hj, poj, kj, akj = rec[j]
                pmax_o = max(pmax_o, pj)
                ja = _jac(ai, aj)
                best_addr = max(best_addr, pj * ja)
                if ai and ai == aj:
                    best_same = max(best_same, pj)
                    n_same += 1
                if hi_ and (hi_ & hj):
                    best_num = max(best_num, pj)
                if poi is not None and poi == poj:
                    best_post = max(best_post, pj)
                jn = _jac(ni, nj)
                best_name = max(best_name, pj * jn)
                best_phon = max(best_phon, pj * _jac(pi_, pj_ph))
                if pj > 0.5:
                    n_conf += 1
                    sa += ja
                    sn += jn
                conflict = bool(hi_) and bool(hj) and not (hi_ & hj)
                if conflict and ja >= 0.5:
                    num_conf = max(num_conf, pj)
                if conflict and ni and ni == nj:
                    name_num_conf = max(name_num_conf, pj)
            F["p1_rank"][i] = rk[ii]
            F["p1_ratio"][i] = pg[ii] / (mx + 1e-9)
            F["p1_max_other"][i] = pmax_o
            F["p1_sum_other"][i] = tot - pg[ii]
            F["n_conf_other"][i] = n_conf
            F["sib_addr_max"][i] = best_addr
            F["sib_addr_same_max"][i] = best_same
            F["n_same_addr"][i] = n_same
            F["sib_num_max"][i] = best_num
            F["sib_postal_max"][i] = best_post
            F["sib_name_max"][i] = best_name
            F["sib_phon_max"][i] = best_phon
            F["conf_sim_addr"][i] = sa / n_conf if n_conf else 0.0
            F["conf_sim_name"][i] = sn / n_conf if n_conf else 0.0
            F["sib_num_conflict_max"][i] = num_conf
            F["sib_same_name_num_conflict_max"][i] = name_num_conf
            pref, e1, bs = number_pair_features(s_rec[3], hi_)
            F["num_prefix_match"][i], F["num_edit1"][i], F["num_best_sim"][i] = pref, e1, bs
            F["n_house_s1"][i], F["n_house_cand"][i] = len(s_rec[3]), len(hi_)
            sk = s_rec[5]
            F["exact_sorted_name"][i] = float(bool(sk) and sk == ki)
            F["s1_name_n_s1"][i] = np.log1p(nf_s1.get(sk, 0)) if sk else 0.0
            F["cand_name_n_s1"][i] = np.log1p(nf_s1.get(ki, 0)) if ki else 0.0
            F["cand_name_n_pool"][i] = np.log1p(nf_pool.get(ki, 0)) if ki else 0.0
            sak = s_rec[6]
            exact_addr = bool(sak) and sak == aki
            F["exact_norm_address"][i] = float(exact_addr)
            F["exact_addr_unique_pool"][i] = float(exact_addr and af_pool.get(aki, 0) == 1)
            F["s1_addr_n_s1"][i] = np.log1p(af_s1.get(sak, 0)) if sak else 0.0
            F["cand_addr_n_s1"][i] = np.log1p(af_s1.get(aki, 0)) if aki else 0.0
            F["cand_addr_n_pool"][i] = np.log1p(af_pool.get(aki, 0)) if aki else 0.0
    out = pd.DataFrame(F)
    out.insert(0, "cand_id", df["cand_id"].to_numpy())
    out.insert(0, "s1_id", df["s1_id"].to_numpy())
    return out


def cross_features(df: pd.DataFrame, store, workers: int, chunk_s1: int = 20000) -> pd.DataFrame:
    """Parallel over chunks of complete S1 groups (fork: the record store is shared)."""
    d = df[["s1_id", "cand_id", "p1"]].reset_index(drop=True)
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["s1_id", "_o"], kind="stable")
    ids = d["s1_id"].to_numpy(dtype=object)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    cuts = list(starts[::chunk_s1]) + [len(d)]
    chunks = [d.iloc[cuts[k]:cuts[k + 1]].drop(columns="_o") for k in range(len(cuts) - 1)]
    order = d["_o"].to_numpy()
    if G.get("store") is not store or "nf_s1" not in G:
        G["nf_s1"], G["nf_pool"] = name_frequencies(store)
        G["af_s1"], G["af_pool"] = address_frequencies(store)
    G["store"] = store
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    if ctx.get_start_method() == "fork" and workers > 1:
        with ctx.Pool(workers) as pool:
            parts = pool.map(cross_chunk, chunks)
    else:
        parts = [cross_chunk(c) for c in chunks]
    out = pd.concat(parts, ignore_index=True)
    res = pd.DataFrame(index=np.arange(len(d)))
    for c in CROSS:
        arr = np.zeros(len(d), np.float32)
        arr[order] = out[c].to_numpy()
        res[c] = arr
    return res


def write_lists(path: Path, header: str, lists: dict, s1_ids):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{lists.get(sid, '')}\n")
    os.replace(tmp, path)


def country_stats(s1, pred, p, rule, cal, grp, country_of: pd.Series, all_ids) -> dict:
    """Per country: predicted matches per S1, share predicted empty, and the share of candidates
    in the uncertain band (calibrated p in 0.2-0.8) - comparable between validation and test."""
    pc = np.zeros(len(p), np.float32)
    for gv in (0, 1):
        pc[grp == gv] = P8.apply_calibration(p[grp == gv], cal[str(gv)])
    ids = pd.Index(all_ids)
    npred = pd.Series(pred.astype(int), index=s1).groupby(level=0).sum().reindex(ids, fill_value=0)
    unc = pd.Series(((pc > 0.2) & (pc < 0.8)).astype(float), index=s1).groupby(level=0).mean().reindex(ids)
    c = pd.Series(ids.map(country_of), index=ids)
    return {str(k): {"s1": int((c == k).sum()), "mean_pred_matches": round(float(npred[c == k].mean()), 3),
                     "share_empty": round(float((npred[c == k] == 0).mean()), 4),
                     "uncertain_share": round(float(unc[c == k].mean()), 4)}
            for k in sorted(c.dropna().unique())}


# ============================================================================ error analysis (Step 1)

def error_report(a, out: Path):
    c8 = Path(a.phase8_dir) / "cache"
    vp = read_table(c8 / "val_pred")
    va = read_table(c8 / "val")
    dec = json.loads((Path(a.phase8_dir) / "decision.json").read_text())
    for c in ("label", "p"):
        vp[c] = pd.to_numeric(vp[c])
    s1, cand = vp["s1_id"].to_numpy(dtype=object), vp["cand_id"].to_numpy(dtype=object)
    pc = P8.apply_calibration(vp["p"].to_numpy(), dec["calibration"])
    pred = P8.apply_rule(dec["rule"], s1, cand, pc)
    df = pd.DataFrame({"s1_id": s1, "cand_id": cand, "label": vp["label"].to_numpy(), "p_cal": pc, "pred": pred})
    need = ["name_token_jaccard", "phon_jaccard", "address_token_overlap", "number_conflict",
            "address_missing_cand", "cross_script", "name_core_ratio", "address_norm_ratio", "stage1_rank"]
    f = va[["s1_id", "cand_id"] + need].copy()
    for c in need:
        f[c] = pd.to_numeric(f[c])
    df = df.merge(f, on=["s1_id", "cand_id"], how="left")
    split = pd.read_csv(Path(a.phase0_dir) / "s1_split.tsv", sep="\t", dtype={"s1_id": str})
    nt = split.set_index("s1_id")["n_matches"].astype(int)
    df["s1_true"] = df["s1_id"].map(nt)

    def cat_fn(r):
        name_link = r.name_token_jaccard > 0 or r.phon_jaccard > 0 or r.name_core_ratio >= 0.8
        if r.address_missing_cand:
            return "addr_missing_" + ("name_similar" if name_link else "name_different")
        if r.cross_script:
            return "cross_script"
        if not name_link and r.address_token_overlap > 0:
            return "name_replaced_address_links"
        if r.name_core_ratio >= 0.9 and r.address_norm_ratio < 0.5:
            return "same_name_other_address"
        if r.number_conflict:
            return "number_conflict"
        return "similar_name_and_address"

    fn = df[(df.label == 1) & ~df.pred].copy()
    fp = df[(df.label == 0) & df.pred].copy()
    fn["category"] = [cat_fn(r) for r in fn.itertuples()]
    fp["category"] = [cat_fn(r) for r in fp.itertuples()]
    fp["singleton_s1"] = fp["s1_true"] == 0
    # text for examples (data_block has the slim name/address columns)
    ex = pd.concat([fn.groupby("category").head(15), fp.groupby("category").head(15)])
    ids = set(ex.s1_id) | set(ex.cand_id)
    txt = pd.concat([read_table(Path(a.block_dir) / f"train_source{k}",
                                columns=["entity_id", "name_core", "address_tokens_norm"]) for k in (1, 2, 3)])
    txt = txt[txt.entity_id.isin(ids)].set_index("entity_id")
    L = ["# Validation error analysis (Phase 8 baseline)\n",
         f"False negatives: {len(fn):,}  |  False positives: {len(fp):,}  "
         f"(FP on singleton S1s: {int(fp.singleton_s1.sum()):,})\n",
         "## False negatives by category (share of all FNs)"]
    L += [f"- {k}: {v:,} ({v / max(len(fn), 1):.1%})" for k, v in fn.category.value_counts().items()]
    L += ["\n## False positives by category (share of all FPs)"]
    L += [f"- {k}: {v:,} ({v / max(len(fp), 1):.1%})" for k, v in fp.category.value_counts().items()]
    L += [f"\nFalse positives on singleton S1s by category: "
          f"{fp[fp.singleton_s1].category.value_counts().to_dict()}"]
    L += ["\nFN stage-1 rank distribution: " + str(fn.stage1_rank.value_counts().sort_index().to_dict())]
    for kind, d in (("FALSE NEGATIVES", fn), ("FALSE POSITIVES", fp)):
        L.append(f"\n## Examples - {kind}")
        for cat, g in d.groupby("category"):
            L.append(f"\n### {cat}")
            for r in g.head(15).itertuples():
                a1 = txt.loc[r.s1_id] if r.s1_id in txt.index else None
                a2 = txt.loc[r.cand_id] if r.cand_id in txt.index else None
                L.append(f"- p={r.p_cal:.2f} rank={int(r.stage1_rank)} | S1 `{getattr(a1, 'name_core', '?')}` @ "
                         f"`{getattr(a1, 'address_tokens_norm', '?')}`  ||  cand `{getattr(a2, 'name_core', '?')}` @ "
                         f"`{getattr(a2, 'address_tokens_norm', '?')}`")
    (out / "errors_report.md").write_text("\n".join(L), encoding="utf-8")
    return {"fn": int(len(fn)), "fp": int(len(fp)), "fn_categories": fn.category.value_counts().to_dict(),
            "fp_categories": fp.category.value_counts().to_dict(),
            "fp_on_singletons": int(fp.singleton_s1.sum())}


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8-dir", default="artifacts/phase8")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--stage2-dir", default="data_stage2")
    ap.add_argument("--repr-dir", default="data_repr")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--out-dir", default="artifacts/phase9")
    ap.add_argument("--submit-dir", default="output_v2")
    ap.add_argument("--from-stage", choices=STAGES)
    ap.add_argument("--stop-after", choices=STAGES)
    ap.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm",
                    help="sklearn only for testing without LightGBM (BSD - not for the final model)")
    ap.add_argument("--exp-lr", type=float, default=0.1)
    ap.add_argument("--final-lr", type=float, default=0.05)
    ap.add_argument("--final-seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--holdout-pct", type=int, default=10)
    ap.add_argument("--final-mode", choices=["reuse", "retrain"], default="reuse",
                    help="reuse = submit the experiment models exactly as scored on half B (fast); "
                         "retrain = retrain the chosen variant with --final-seeds / --final-lr (slow)")
    ap.add_argument("--test-gammas", type=float, nargs="*", default=[1.15, 1.35],
                    help="extra, more conservative test variants written to output_v2/variants/")
    ap.add_argument("--min-gain", type=float, default=0.0003,
                    help="a more complex variant must beat the plain one by this much on half A")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    out = Path(a.out_dir)
    mdir, cdir = out / "models", out / "cache"
    for d in (mdir, cdir):
        d.mkdir(parents=True, exist_ok=True)
    st = State(out / "state")
    if a.from_stage:
        st.reset_from(a.from_stage)
    rp = out / "report.json"
    R = json.loads(rp.read_text()) if rp.exists() else {}
    R["args"] = vars(a)

    def save():
        P8.atomic_json(rp, R)

    def stop(s):
        if a.stop_after == s:
            save()
            log(f"stopping after '{s}' as requested")
            sys.exit(0)

    LGB = a.backend
    kw = dict(threads=a.workers, max_rounds=a.max_rounds, early_stop=a.early_stop,
              num_leaves=a.num_leaves, min_leaf=a.min_leaf)

    # ---- Step 1
    if not st.done("errors"):
        try:
            R["errors"] = error_report(a, out)
            log(f"errors: {R['errors']['fn']:,} FN, {R['errors']['fp']:,} FP -> {out / 'errors_report.md'}")
        except Exception as e:                     # analysis must never block the improvements
            R["errors"] = {"failed": repr(e)}
            log(f"WARNING error analysis failed: {e!r}")
        save(); st.mark("errors")  # noqa: E702
    stop("errors")

    # ---- shared data
    c8 = Path(a.phase8_dir) / "cache"
    base_feats = json.loads((c8 / "features.json").read_text())
    TR = add_relative_stage1(P8.prepare(read_table(c8 / "train"), base_feats)[0])
    VA = add_relative_stage1(P8.prepare(read_table(c8 / "val"), base_feats)[0])
    TR["label"] = pd.to_numeric(TR["label"]).astype(np.int8)
    VA["label"] = pd.to_numeric(VA["label"]).astype(np.int8)
    hold = P8.bucket_of(TR["s1_id"].to_numpy()) < a.holdout_pct
    split = pd.read_csv(Path(a.phase0_dir) / "s1_split.tsv", sep="\t", dtype={"s1_id": str})
    n_true = split[split["split"] == "val"].set_index("s1_id")["n_matches"].astype(int)
    hs = P8.half_of(n_true.index.to_numpy())
    V = {"s1": VA["s1_id"].to_numpy(dtype=object), "cand": VA["cand_id"].to_numpy(dtype=object),
         "lab": VA["label"].to_numpy(), "grp": VA["address_missing_cand"].to_numpy().astype(np.int8),
         "nA": n_true[hs == 0], "nB": n_true[hs == 1]}
    h = P8.half_of(V["s1"])
    V["A"], V["B"] = h == 0, h == 1
    log(f"train {len(TR):,} rows, validation {len(VA):,} rows")
    res_path = cdir / "variants.json"
    variants = json.loads(res_path.read_text()) if res_path.exists() else {}

    def record(name, p, extra=None):
        # Keep IDs with probabilities.  Downstream stacking must never assume that two cached
        # tables have identical row order after a merge, shard resume, or feature rebuild.
        P8.atomic_table(pd.DataFrame({"s1_id": V["s1"], "cand_id": V["cand"], "p": p}),
                        cdir / f"valp_{name}")
        variants[name] = {**evaluate_variant(name, p, V), **(extra or {})}
        P8.atomic_json(res_path, variants)

    def valp(name):
        return pd.to_numeric(read_table(cdir / f"valp_{name}")["p"]).to_numpy(np.float32)

    # ---- Step 2
    if not st.done("skew"):
        log("Step 2: Stage-1 score distribution, train (out-of-fold) vs validation (final model)")
        sk = {}
        for lab in (0, 1):
            q = [0.05, 0.25, 0.5, 0.75, 0.95]
            t = np.quantile(TR.loc[TR.label == lab, "stage1_score"], q)
            v = np.quantile(VA.loc[VA.label == lab, "stage1_score"], q)
            from scipy.stats import ks_2samp
            ks = ks_2samp(TR.loc[TR.label == lab, "stage1_score"].sample(min(200000, int((TR.label == lab).sum())), random_state=0),
                          VA.loc[VA.label == lab, "stage1_score"].sample(min(200000, int((VA.label == lab).sum())), random_state=0))
            sk[f"label={lab}"] = {"train_quantiles": np.round(t, 4).tolist(), "val_quantiles": np.round(v, 4).tolist(),
                                  "ks_statistic": round(float(ks.statistic), 4)}
            log(f"   label {lab}: train q {np.round(t, 3).tolist()} | val q {np.round(v, 3).tolist()} | KS {ks.statistic:.4f}")
        R["stage1_skew"] = sk
        if "phase8_baseline" not in variants:
            p8 = read_table(c8 / "val_pred")
            p8 = VA[["s1_id", "cand_id"]].merge(p8.assign(p=pd.to_numeric(p8["p"])), on=["s1_id", "cand_id"], how="left")
            record("phase8_baseline", p8["p"].fillna(0).to_numpy(np.float32), {"note": "3-seed ensemble, lr 0.05"})
        for fs in ("all", "no_stage1", "rel_stage1"):
            name = f"fs_{fs}"
            if name in variants:
                continue
            feats = feature_set(fs, base_feats)
            m, best = train_model(LGB, TR.loc[~hold, feats].to_numpy(np.float32), TR.label[~hold].to_numpy(),
                                  TR.loc[hold, feats].to_numpy(np.float32), TR.label[hold].to_numpy(),
                                  a.exp_lr, 11, **kw)
            save_model(LGB, m, best, mdir / name)
            record(name, predict(LGB, m, best, VA[feats].to_numpy(np.float32), a.workers),
                   {"features": fs, "rounds": best})
        R["variants"] = variants
        save(); st.mark("skew")  # noqa: E702
    stop("skew")
    best_fs = max(("all", "no_stage1", "rel_stage1"), key=lambda f: variants[f"fs_{f}"]["A"])
    if best_fs != "all" and variants[f"fs_{best_fs}"]["A"] - variants["fs_all"]["A"] < a.min_gain:
        best_fs = "all"
    log(f"best feature set (on A): {best_fs}")

    # ---- Step 4
    from phase7_stage2_features import load_store
    if not st.done("stack"):
        try:
            feats = feature_set(best_fs, base_feats)
            kind0, m0, best0 = load_model(mdir / f"fs_{best_fs}")
            fold = P8.bucket_of(TR["s1_id"].to_numpy()) % a.folds
            p1_tr = np.zeros(len(TR), np.float32)
            for k in range(a.folds):
                base = mdir / f"stack_fold{k}"
                sel = fold != k
                if not model_exists(base):
                    mk, _ = train_model(LGB, TR.loc[sel, feats].to_numpy(np.float32), TR.label[sel].to_numpy(),
                                        None, None, a.exp_lr, 100 + k, rounds=best0, **kw)
                    save_model(LGB, mk, best0, base)
                kk, mk, bk = load_model(base)
                p1_tr[~sel] = predict(kk, mk, bk, TR.loc[~sel, feats].to_numpy(np.float32), a.workers)
                log(f"   stack: fold {k + 1}/{a.folds} out-of-fold probabilities")
            p1_va = predict(kind0, m0, best0, VA[feats].to_numpy(np.float32), a.workers)
            store = load_store(Path(a.repr_dir), "train")
            log("   stack: cross-candidate features (train)")
            Xc_tr = cross_features(TR.assign(p1=p1_tr), store, a.workers)
            log("   stack: cross-candidate features (validation)")
            Xc_va = cross_features(VA.assign(p1=p1_va), store, a.workers)
            del store
            P8.atomic_table(Xc_tr, cdir / "cross_train")
            P8.atomic_table(Xc_va, cdir / "cross_val")
            f2 = feats + CROSS
            Xtr = np.hstack([TR[feats].to_numpy(np.float32), Xc_tr[CROSS].to_numpy(np.float32)])
            Xva = np.hstack([VA[feats].to_numpy(np.float32), Xc_va[CROSS].to_numpy(np.float32)])
            m2, b2 = train_model(LGB, Xtr[~hold], TR.label[~hold].to_numpy(), Xtr[hold], TR.label[hold].to_numpy(),
                                 a.exp_lr, 21, **kw)
            save_model(LGB, m2, b2, mdir / "stack_second")
            record("stack", predict(LGB, m2, b2, Xva, a.workers), {"features": best_fs, "stacked": True, "rounds": b2})
            R["stack_features"] = f2
        except Exception as e:
            log(f"WARNING stacking failed, continuing without it: {e!r}")
            R["stack_error"] = repr(e)
        R["variants"] = variants
        save(); st.mark("stack")  # noqa: E702
    stop("stack")

    # ---- Step 5
    if not st.done("xgb"):
        try:
            import xgboost  # noqa: F401
            use_stack = "stack" in variants and variants["stack"]["A"] > variants[f"fs_{best_fs}"]["A"]
            feats = feature_set(best_fs, base_feats)
            Xtr, Xva = TR[feats].to_numpy(np.float32), VA[feats].to_numpy(np.float32)
            if use_stack:
                Xc_tr, Xc_va = read_table(cdir / "cross_train"), read_table(cdir / "cross_val")
                Xtr = np.hstack([Xtr, Xc_tr[CROSS].to_numpy(np.float32)])
                Xva = np.hstack([Xva, Xc_va[CROSS].to_numpy(np.float32)])
            mx, bx = train_model("xgboost", Xtr[~hold], TR.label[~hold].to_numpy(), Xtr[hold],
                                 TR.label[hold].to_numpy(), a.exp_lr, 31, **kw)
            save_model("xgboost", mx, bx, mdir / "xgb")
            px = predict("xgboost", mx, bx, Xva, a.workers)
            record("xgb", px, {"features": best_fs, "stacked": use_stack, "rounds": bx})
            pl = valp("stack" if use_stack else f"fs_{best_fs}")
            record("blend", (px + pl) / 2, {"features": best_fs, "stacked": use_stack})
        except ImportError:
            log("WARNING xgboost not installed - skipping Step 5 (pip install xgboost)")
        except Exception as e:
            log(f"WARNING XGBoost step failed, continuing: {e!r}")
            R["xgb_error"] = repr(e)
        R["variants"] = variants
        save(); st.mark("xgb")  # noqa: E702
    stop("xgb")

    # ---- select
    if not st.done("select"):
        cand_names = [k for k in variants if k not in ("phase8_baseline", "final")]
        best = max(cand_names, key=lambda k: variants[k]["A"])
        # noise guard: with ~220k S1 per half, F0.5 differences below ~0.0003 are not reliable,
        # so a more complex variant must beat the plain all-features model by --min-gain
        if best != "fs_all" and variants[best]["A"] - variants["fs_all"]["A"] < a.min_gain:
            log(f"   {best} beats fs_all by less than {a.min_gain} on A -> keeping the simpler fs_all")
            best = "fs_all"
        v = variants[best]
        R["selected"] = {"variant": best, "A": v["A"], "B": v["B"], "rule": v["rule"],
                         "features": v.get("features", best_fs), "stacked": bool(v.get("stacked", False)),
                         "xgb": best in ("xgb", "blend"), "blend": best == "blend"}
        R["leaderboard_B"] = sorted([[k, variants[k]["B"], variants[k]["A"]] for k in variants], key=lambda x: -x[1])
        log(f"selected: {best} (A {v['A']:.5f}, B {v['B']:.5f}); Phase 8 baseline B "
            f"{variants['phase8_baseline']['B']:.5f}")
        save(); st.mark("select")  # noqa: E702
    stop("select")

    # ---- final: retrain the chosen variant properly
    S = R["selected"]
    feats = feature_set(S["features"], base_feats)
    if not st.done("final") and a.final_mode == "reuse":
        name = S["variant"]
        lgb_base = mdir / ("stack_second" if S["stacked"] else f"fs_{S['features']}")
        bases = {"xgb": [mdir / "xgb"], "blend": [lgb_base, mdir / "xgb"],
                 "stack": [mdir / "stack_second"]}.get(name, [mdir / name])
        p_final = valp(name)
        rule, cal, _ = tune_decision(p_final, V["lab"], V["s1"], V["cand"], V["grp"], n_true,
                                     np.ones(len(p_final), bool))
        P8.atomic_json(out / "decision.json", {"rule": rule, "calibration": cal, "variant": S,
                                               "models": [str(b) for b in bases]})
        pv = apply_decision(rule, cal, p_final, V["s1"], V["cand"], V["grp"])
        R["final"] = {"A": variants[name]["A"], "B_honest": variants[name]["B"], "rule_for_test": rule,
                      "mode": "reuse", "models": [Path(b).name for b in bases]}
        R["val_by_country"] = country_stats(V["s1"], pv, p_final, rule, cal, V["grp"],
                                            split.set_index("s1_id")["country_norm"], n_true.index)
        log(f"final (reuse): {name} with models {[Path(b).name for b in bases]}; "
            f"B (honest) {variants[name]['B']:.5f}; rule for test {rule}")
        save(); st.mark("final")  # noqa: E702
    if not st.done("final"):
        Xtr, Xva = TR[feats].to_numpy(np.float32), VA[feats].to_numpy(np.float32)
        if S["stacked"]:
            Xtr = np.hstack([Xtr, read_table(cdir / "cross_train")[CROSS].to_numpy(np.float32)])
            Xva = np.hstack([Xva, read_table(cdir / "cross_val")[CROSS].to_numpy(np.float32)])
        preds = []
        kinds = (["xgboost"] if S["xgb"] and not S["blend"] else [LGB]) + (["xgboost"] if S["blend"] else [])
        for kind in kinds:
            n_seed = a.final_seeds if kind == LGB else 1
            for i in range(n_seed):
                base = mdir / f"final_{kind}_{i}"
                if not model_exists(base, Xtr.shape[1]):
                    m, b = train_model(kind, Xtr[~hold], TR.label[~hold].to_numpy(), Xtr[hold],
                                       TR.label[hold].to_numpy(), a.final_lr, 500 + i, **kw)
                    save_model(kind, m, b, base)
                    log(f"   final {kind} model {i + 1}/{n_seed}: {b} rounds")
                k_, m_, b_ = load_model(base)
                preds.append((kind, predict(k_, m_, b_, Xva, a.workers)))
        by_kind = {}
        for kind, p in preds:
            by_kind.setdefault(kind, []).append(p)
        p_final = np.mean([np.mean(v, axis=0) for v in by_kind.values()], axis=0).astype(np.float32)
        record("final", p_final, {"final": True})
        rule, cal, _ = tune_decision(p_final, V["lab"], V["s1"], V["cand"], V["grp"], n_true,
                                     np.ones(len(p_final), bool))
        P8.atomic_json(out / "decision.json", {"rule": rule, "calibration": cal, "variant": S,
                                               "models": [str(mdir / f"final_{k}_{i}")
                                                          for k in kinds for i in range(a.final_seeds if k == LGB else 1)]})
        R["final"] = {"A": variants["final"]["A"], "B_honest": variants["final"]["B"], "rule_for_test": rule}
        pv = apply_decision(rule, cal, p_final, V["s1"], V["cand"], V["grp"])
        R["val_by_country"] = country_stats(V["s1"], pv, p_final, rule, cal, V["grp"],
                                            split.set_index("s1_id")["country_norm"], n_true.index)
        log(f"final model: A {variants['final']['A']:.5f}, B (honest) {variants['final']['B']:.5f}")
        save(); st.mark("final")  # noqa: E702
    stop("final")

    # ---- Step 7: test, in-process
    if not st.done("test"):
        dec = json.loads((out / "decision.json").read_text())
        shards = P8.list_shards(Path(a.stage2_dir) / "test")
        log(f"test: reading {len(shards)} shards")
        with ThreadPoolExecutor(max_workers=min(16, a.workers)) as ex:
            parts = list(ex.map(lambda b: read_table(Path(b)), shards))
        TE = add_relative_stage1(P8.prepare(pd.concat(parts, ignore_index=True), base_feats)[0])
        del parts
        Xte = TE[feats].to_numpy(np.float32)
        if S["stacked"]:
            k0, m0, b0 = load_model(mdir / f"fs_{S['features']}")
            p1 = predict(k0, m0, b0, Xte, a.workers)
            store = load_store(Path(a.repr_dir), "test")
            log("   test: cross-candidate features")
            Xc = cross_features(TE.assign(p1=p1), store, a.workers)
            del store
            Xte = np.hstack([Xte, Xc[CROSS].to_numpy(np.float32)])
        by_kind = {}
        for base in dec["models"]:
            k_, m_, b_ = load_model(Path(base))
            by_kind.setdefault(k_, []).append(predict(k_, m_, b_, Xte, a.workers))
            log(f"   scored with {Path(base).name}")
        p = np.mean([np.mean(v, axis=0) for v in by_kind.values()], axis=0).astype(np.float32)
        s1, cand = TE["s1_id"].to_numpy(dtype=object), TE["cand_id"].to_numpy(dtype=object)
        grp = TE["address_missing_cand"].to_numpy().astype(np.int8)
        # Raw, pre-calibration scores are a checkpoint for later stacked models.  Candidate-list
        # submissions cannot be inverted to scores, and row order alone is not a safe join key.
        # Keeping explicit IDs makes any future layer verify one-to-one alignment before reuse.
        P8.atomic_table(pd.DataFrame({"s1_id": s1, "cand_id": cand, "p": p}), cdir / "test_pred")
        test_s1 = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id", "country_norm"])
        country_of = test_s1.set_index("entity_id")["country_norm"]
        row_country = pd.Series(s1).map(country_of).to_numpy(dtype=object)
        seen = set(split.loc[split["split"] == "val", "country_norm"].unique())
        unseen = np.array([c not in seen for c in row_country])
        sub = Path(a.submit_dir)
        sub.mkdir(parents=True, exist_ok=True)
        tp = pd.DataFrame({"s1_id": s1, "cand_id": cand})
        cl = tp.groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict()
        write_lists(sub / "candidate_pairs.tsv", "candidate_entity_ids", cl, test_s1["entity_id"])

        def decide(gamma_seen, align):
            g = np.full(len(p), gamma_seen, np.float32)
            pred = apply_decision(dec["rule"], dec["calibration"], p, s1, cand, grp, g)
            info = {"gamma": gamma_seen}
            if align and unseen.any():
                # unseen countries (no labels): choose their gamma so that their predicted matches
                # per S1 equal the seen countries' average - a label-free prior-matching assumption
                pm = pd.Series(pred.astype(int), index=s1).groupby(level=0).sum()
                c_ = pd.Series(pm.index.map(country_of), index=pm.index)
                seen_rows = c_.isin(list(seen))
                if not seen_rows.any():                  # nothing to align to
                    return pred, {**info, "unseen_gamma": None, "note": "no seen-country S1 in test"}
                target = float(pm[seen_rows].mean())
                sel = unseen
                lo_, hi_ = 0.3, 4.0
                for _ in range(14):
                    mid = (lo_ + hi_) / 2
                    pu = apply_decision(dec["rule"], dec["calibration"], p[sel], s1[sel], cand[sel], grp[sel], mid)
                    rate = pu.sum() / max(len(pd.unique(pd.Series(s1[sel]))), 1)
                    lo_, hi_ = (mid, hi_) if rate > target else (lo_, mid)
                g[sel] = (lo_ + hi_) / 2
                pred = apply_decision(dec["rule"], dec["calibration"], p, s1, cand, grp, g)
                info.update(unseen_gamma=round(float((lo_ + hi_) / 2), 3), target_rate=round(target, 3))
            return pred, info

        variants_out = {}
        specs = [("main", 1.0, False)] + [(f"gamma{gm}", gm, False) for gm in a.test_gammas] + \
                ([("gamma1.0_align", 1.0, True)] if unseen.any() else [])
        for name, gm, al in specs:
            pred, info = decide(gm, al)
            d = sub if name == "main" else sub / "variants" / name
            d.mkdir(parents=True, exist_ok=True)
            ml = tp[pred].groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict()
            write_lists(d / "matching_results.tsv", "matched_entity_ids", ml, test_s1["entity_id"])
            if name != "main":
                import shutil
                shutil.copyfile(sub / "candidate_pairs.tsv", d / "candidate_pairs.tsv")
            info["by_country"] = country_stats(s1, pred, p, dec["rule"], dec["calibration"], grp, country_of,
                                               test_s1["entity_id"])
            variants_out[name] = info
            log(f"   test variant {name}: {json.dumps({k: v for k, v in info.items() if k != 'by_country'})} "
                f"| matches/S1 by country {({c: v['mean_pred_matches'] for c, v in info['by_country'].items()})}")
        R["test_variants"] = variants_out
        R["submission_checks"] = P8.check_submission(sub, a.block_dir, test_s1["entity_id"])
        save()
        if not R["submission_checks"]["all_ok"]:
            raise RuntimeError(f"submission checks failed: {R['submission_checks']}")
        st.mark("test")
        log(f"wrote {sub}/matching_results.tsv and candidate_pairs.tsv (checks OK)")
    write_summary(R, out / "summary.md")
    log(f"done -> {out / 'summary.md'}")


def write_summary(R, path):
    L = ["# Phase 9 - matcher improvements\n"]
    if "errors" in R and "fn" in R["errors"]:
        e = R["errors"]
        L += ["## Step 1 - errors (Phase 8 baseline, all validation)",
              f"- FN {e['fn']:,}: {e['fn_categories']}", f"- FP {e['fp']:,}: {e['fp_categories']}",
              f"- FP on singleton S1s: {e['fp_on_singletons']:,}", "- examples: errors_report.md"]
    if "stage1_skew" in R:
        L += ["\n## Step 2 - Stage-1 score, train (out-of-fold) vs validation", "```",
              json.dumps(R["stage1_skew"], indent=1), "```"]
    if "leaderboard_B" in R:
        L += ["\n## All variants (tuned on A, honest score on B)", "| variant | B | A |", "|---|---|---|"]
        L += [f"| {k} | {b:.5f} | {a:.5f} |" for k, b, a in R["leaderboard_B"]]
    if "selected" in R:
        L.append(f"\n**Selected: {R['selected']['variant']}** - decision rule {R['selected']['rule']}")
    if "final" in R:
        L.append(f"\n**Final retrained model: B (honest) {R['final']['B_honest']:.5f}**; "
                 f"rule for test {R['final']['rule_for_test']}")
    if "val_by_country" in R:
        L += ["\n## Validation by country (final model)", "```", json.dumps(R["val_by_country"], indent=1), "```"]
    if "test_variants" in R:
        L += ["\n## Test variants (output_v2/ = main; others in output_v2/variants/)",
              "| variant | settings | matches per S1 by country | uncertain share by country |", "|---|---|---|---|"]
        for k, v in R["test_variants"].items():
            bc = v["by_country"]
            L.append(f"| {k} | { {x: y for x, y in v.items() if x != 'by_country'} } | "
                     f"{ {c: d['mean_pred_matches'] for c, d in bc.items()} } | "
                     f"{ {c: d['uncertain_share'] for c, d in bc.items()} } |")
    if "submission_checks" in R:
        L += ["\n## Submission checks (output_v2/)", "```", json.dumps(R["submission_checks"], indent=1), "```"]
    for k in ("stack_error", "xgb_error"):
        if k in R:
            L.append(f"\nWARNING {k}: {R[k]}")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
