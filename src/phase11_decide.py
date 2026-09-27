#!/usr/bin/env python3
"""
Phase 11 - where the remaining F0.5 goes, and a sharper decision layer. NO retraining needed:
it works from Phase 10's cached scores (minutes, not hours).

Reads
  artifacts/phase8/cache/val                 s1_id, cand_id, label, address_missing_cand
  artifacts/phase10/decision.json            chosen variant, rule + calibration used for test
  artifacts/phase10/cache/valp_<chosen>      validation probabilities (row order = phase8 val)
  artifacts/phase10/cache/test_scores        test probabilities with IDs
  artifacts/phase0/{s1_split.tsv, gt_long.tsv}
  data_stage2/test                           only address_missing_cand (calibration group)

Stages (all fast, no checkpoints needed)
  diagnose  Additive split of the validation loss (1 - macro F0.5) per S1 into
              loss_block  true matches never reached Stage 2 (blocking / Stage-1 top-N)
              loss_fn     true matches in the candidate list but not predicted
              loss_fp     false positives (on singleton S1s a single FP costs the whole S1)
            by country and by true match count, plus WHO owns the false-positive records
            (another validation S1, a train-split S1, nobody) and how often one-to-one fires on
            validation vs test. Validation only sees ~20% of the S1s competing for each record,
            test sees all of them, so this tells how pessimistic the validation estimate is.
  decide    The Phase 9/10 rule family (baseline, reproduces Phase 10's number) against:
              exact_f   expected F0.5 computed EXACTLY per S1 (Poisson-binomial over the top-K
                        candidates + Poisson tail), instead of the ratio-of-expectations
                        approximation; it matters most at the "predict empty?" boundary,
                        which decides every singleton S1
              soft o2o  a record belongs to at most one S1, so its claimants' probabilities are
                        renormalised (p_i -> odds_i / (1 + sum_j odds_j)) instead of zeroing
                        every claim but the best one
            Tuned on half A, scored on half B (honest). A new rule is used only if it beats the
            baseline by --min-gain on A; then output_v4/ is written from the cached test scores.
  curve     (opt-in, --learning-curve; retrains the level-1 model) validation F0.5 when the
            level-1 model is trained on a fraction of the training S1s. If halving the data
            costs a lot, running Phases 5-7 on MORE training S1s (Phase 5 used --s1-sample
            400000) is the biggest remaining lever; if it costs nothing, it is not.

Usage
  python src/phase11_decide.py                         # diagnose + decide (+ output_v4 if better)
  python src/phase11_decide.py --learning-curve        # also the data-size experiment
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import phase8_match as P8
import phase9_improve as P9
from common import read_table

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ============================================================================ decision rules

def soft_one_to_one(s1: np.ndarray, cand: np.ndarray, p: np.ndarray) -> np.ndarray:
    """At most one S1 owns a record: P(owner = i) = odds_i / (1 + sum_j odds_j) over the record's
    claimants (independent marginals conditioned on 'at most one'). One claimant: unchanged."""
    q = np.clip(np.asarray(p, np.float64), 0.0, 1 - 1e-6)
    odds = q / (1 - q)
    tot = pd.Series(odds).groupby(pd.Series(cand).to_numpy()).transform("sum").to_numpy()
    return (odds / (1 + tot)).astype(np.float32)


def apply_o2o(mode, s1, cand, p):
    if mode is True or mode == "hard":
        return P8.one_to_one(s1, cand, p)
    if mode == "soft":
        return soft_one_to_one(s1, cand, p)
    return np.asarray(p, np.float32)


def expected_f_table(s1: np.ndarray, p: np.ndarray, mu: float, K: int = 10, M: int = 12, chunk: int = 100_000):
    """Exact E[F0.5] of predicting the top-k candidates, k = 0..K, for every S1.

    Candidates are independent Bernoulli(p); the top K are handled exactly (Poisson-binomial),
    everything below rank K plus the expected matches missing from the list (mu) as one Poisson
    count. F(k) = 1.25 TP / (0.25 (TP + O) + k) for k >= 1; F(0) = 1 iff there is no match at all.
    Returns (d, E): d = rows sorted by (S1, -p) with columns s, p, i, code, pos; E = (n_S1, K + 1),
    -inf where k exceeds the S1's number of candidates.
    """
    d = pd.DataFrame({"s": s1, "p": np.clip(np.asarray(p, np.float64), 0.0, 1 - 1e-9), "i": np.arange(len(p))})
    d = d.sort_values(["s", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    code = pd.factorize(d["s"], sort=False)[0]
    pos = d.groupby("s", sort=False).cumcount().to_numpy()
    d["code"], d["pos"] = code, pos
    S = int(code.max()) + 1 if len(code) else 0
    P = np.zeros((S, K))
    top = pos < K
    P[code[top], pos[top]] = d["p"].to_numpy()[top]
    n_c = np.bincount(code, minlength=S)
    lam = np.bincount(code[~top], weights=d["p"].to_numpy()[~top], minlength=S) + float(mu)
    E = np.vstack([_expected_f_rows(P[i:i + chunk], lam[i:i + chunk], n_c[i:i + chunk], K, M)
                   for i in range(0, S, chunk)]) if S else np.zeros((0, K + 1))
    return d, E


def _expected_f_rows(P, lam, n_c, K, M):
    S = len(P)
    L = K + M + 1
    j = np.arange(L)
    loglam = np.log(np.maximum(lam, 1e-300))
    pois = np.exp(j[None, :] * loglam[:, None] - lam[:, None] - gammaln(j + 1)[None, :])
    pois[lam == 0] = 0.0
    pois[lam == 0, 0] = 1.0
    # suffix[k] = distribution of true matches outside the top k (ranks k..K-1 + Poisson tail)
    suffix = [None] * (K + 1)
    suffix[K] = pois
    for i in range(K - 1, -1, -1):
        q, prev = P[:, i:i + 1], suffix[i + 1]
        new = prev * (1 - q)
        new[:, 1:] += prev[:, :-1] * q
        suffix[i] = new
    # prefix[k] = distribution of true matches among the top k
    pre = np.zeros((S, K + 1))
    pre[:, 0] = 1.0
    t = np.arange(K + 1)[:, None]
    o = np.arange(L)[None, :]
    E = np.full((S, K + 1), -np.inf)
    E[:, 0] = suffix[0][:, 0]
    for k in range(1, K + 1):
        q = P[:, k - 1:k]
        nxt = pre * (1 - q)
        nxt[:, 1:] += pre[:, :-1] * q
        pre = nxt
        W = 1.25 * t / (0.25 * (t + o) + k)
        E[:, k] = ((pre @ W) * suffix[k]).sum(axis=1)
        E[n_c < k, k] = -np.inf
    return E


def decide_exact_f(s1: np.ndarray, p: np.ndarray, mu: float, K: int = 10) -> np.ndarray:
    d, E = expected_f_table(s1, p, mu, K)
    out = np.zeros(len(p), bool)
    if not len(d):
        return out
    kstar = np.argmax(E, axis=1)                      # first maximum -> fewest predictions on ties
    take = d["pos"].to_numpy() < kstar[d["code"].to_numpy()]
    out[d["i"].to_numpy()] = take
    return out


def calibrate(cal, p, grp):
    pc = np.zeros(len(p), np.float32)
    for gv in (0, 1):
        pc[grp == gv] = P8.apply_calibration(p[grp == gv], cal[str(gv)])
    return pc


def decide(rule, cal, p, s1, cand, grp):
    """Phase 9/10 rules (method threshold / expected_f, one_to_one bool) plus exact_f and
    one_to_one = 'soft'."""
    if rule["method"] in ("threshold", "expected_f") and rule["one_to_one"] in (True, False):
        return P9.apply_decision(rule, cal, p, s1, cand, grp)
    q = apply_o2o(rule["one_to_one"], s1, cand, calibrate(cal, p, grp))
    if rule["method"] == "exact_f":
        return decide_exact_f(s1, q, rule["mu"], rule.get("K", 10))
    if rule["method"] == "expected_f":
        return P8.decide_expected_f(s1, q, rule["mu"])
    return np.where(grp == 1, q >= rule["t_noaddr"], q >= rule["t"])


def candidate_rules(mu: float, K: int) -> list:
    rules = [{"method": "exact_f", "mu": m_, "one_to_one": oo, "K": K}
             for oo in (False, True, "soft") for m_ in (0.0, mu)]
    rules += [{"method": "expected_f", "mu": m_, "one_to_one": "soft"} for m_ in (0.0, mu)]
    return rules


# ============================================================================ diagnostics

def loss_parts(s1, lab, pred, n_true: pd.Series) -> pd.DataFrame:
    """Per S1: F and an additive split of 1 - F into loss_block + loss_fn + loss_fp."""
    d = pd.DataFrame({"s": s1, "tp": (lab == 1) & pred, "np": pred, "found": lab == 1})
    agg = d.groupby("s")[["tp", "np", "found"]].sum().reindex(n_true.index, fill_value=0)
    tp, npred, found = (agg[c].to_numpy(float) for c in ("tp", "np", "found"))
    nt = n_true.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(npred > 0, 1.25 * tp / (0.25 * nt + npred), 0.0)
        f_nofp = np.where(tp > 0, 1.25 * tp / (0.25 * nt + tp), 0.0)
        f_ceil = np.where(found > 0, 1.25 * found / (0.25 * nt + found), 0.0)
    single = nt == 0
    f = np.where(single, (npred == 0).astype(float), f)
    out = pd.DataFrame({
        "F": f,
        "loss_block": np.where(single, 0.0, 1 - f_ceil),
        "loss_fn": np.where(single, 0.0, f_ceil - f_nofp),
        "loss_fp": np.where(single, 1 - f, f_nofp - f),
        "wrong_pick": (~single) & (tp == 0) & (npred > 0),
        "n_true": nt.astype(int)}, index=n_true.index)
    return out


def summarize_loss(lp: pd.DataFrame, seg: pd.Series | None = None) -> dict:
    cols = ["loss_block", "loss_fn", "loss_fp"]

    def one(g):
        return {"s1": int(len(g)), "macro_f05": round(float(g.F.mean()), 5),
                **{c: round(float(g[c].mean()), 5) for c in cols},
                "wrong_pick_s1": int(g.wrong_pick.sum())}
    if seg is None:
        return one(lp)
    return {str(k): one(g) for k, g in lp.groupby(seg.reindex(lp.index).to_numpy())}


def fp_ownership(s1, cand, lab, pred, owner: pd.Series, split_of: pd.Series, scored_train: set,
                 n_true: pd.Series) -> dict:
    """Who owns the records we wrongly predicted? One-to-one on test can only fix an FP whose
    owner S1 is scored and claims the record more strongly - on validation only validation S1s
    compete, on test every test S1 does."""
    fp = pd.DataFrame({"s1": s1, "cand": cand})[(lab == 0) & pred]
    own = fp["cand"].map(owner)
    osplit = own.map(split_of)
    cat = np.select([own.isna(), osplit == "val", own.isin(scored_train)],
                    ["no_owner", "owner_is_val_s1", "owner_is_scored_train_s1"], "owner_is_unscored_train_s1")
    single = fp["s1"].map(n_true).fillna(1).to_numpy() == 0
    r = {"fp_pairs": int(len(fp))}
    for name, m in (("all_fp", np.ones(len(fp), bool)), ("fp_on_singleton_s1", single)):
        vc = pd.Series(cat[m]).value_counts()
        r[name] = {k: [int(v), round(float(v / max(m.sum(), 1)), 4)] for k, v in vc.items()}
    return r


def o2o_rate(s1, cand, pc, country=None) -> dict:
    """Share of confident claims (calibrated p >= 0.5) that hard one-to-one removes."""
    conf = pc >= 0.5
    killed = conf & (P8.one_to_one(s1, cand, pc) == 0)
    if country is None:
        return {"confident_claims": int(conf.sum()), "removed_share": round(float(killed.sum() / max(conf.sum(), 1)), 4)}
    out = {}
    for c in pd.unique(country):
        m = country == c
        out[str(c)] = {"confident_claims": int(conf[m].sum()),
                       "removed_share": round(float(killed[m].sum() / max(conf[m].sum(), 1)), 4)}
    return out


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--phase8-dir", default="artifacts/phase8")
    ap.add_argument("--phase9-dir", default="artifacts/phase9")
    ap.add_argument("--phase10-dir", default="artifacts/phase10")
    ap.add_argument("--stage2-dir", default="data_stage2")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--out-dir", default="artifacts/phase11")
    ap.add_argument("--submit-dir", default="output_v4")
    ap.add_argument("--K", type=int, default=10, help="candidates per S1 handled exactly by exact_f")
    ap.add_argument("--min-gain", type=float, default=0.0003)
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--curve-fracs", default="0.25,0.5,1.0")
    ap.add_argument("--curve-lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R = {"args": vars(a)}
    p10 = Path(a.phase10_dir)
    dec10 = json.loads((p10 / "decision.json").read_text())
    chosen = dec10["chosen"]

    # ---- validation, aligned exactly like Phase 9/10 (phase8 cache row order)
    c8 = Path(a.phase8_dir) / "cache"
    VA = read_table(c8 / "val", columns=["s1_id", "cand_id", "label", "address_missing_cand"])
    s1, cand = VA["s1_id"].to_numpy(dtype=object), VA["cand_id"].to_numpy(dtype=object)
    lab = pd.to_numeric(VA["label"]).to_numpy().astype(np.int8)
    grp = pd.to_numeric(VA["address_missing_cand"]).fillna(0).to_numpy().astype(np.int8)
    p = pd.to_numeric(read_table(p10 / "cache" / f"valp_{chosen}")["p"]).to_numpy(np.float32)
    if len(p) != len(VA):
        raise RuntimeError(f"valp_{chosen} has {len(p)} rows, phase8 val has {len(VA)} - caches out of sync")
    split = pd.read_csv(Path(a.phase0_dir) / "s1_split.tsv", sep="\t", dtype={"s1_id": str})
    split["n_matches"] = split["n_matches"].astype(int)
    n_true = split[split["split"] == "val"].set_index("s1_id")["n_matches"]
    country_of = split.set_index("s1_id")["country_norm"]
    hs = P8.half_of(n_true.index.to_numpy())
    nA, nB = n_true[hs == 0], n_true[hs == 1]
    h = P8.half_of(s1)
    A, B = h == 0, h == 1
    log(f"Phase 10 chosen variant: {chosen}; validation {len(VA):,} rows, {len(n_true):,} S1")

    # ================================================================ diagnose
    pred10 = decide(dec10["rule"], dec10["calibration"], p, s1, cand, grp)
    lp = loss_parts(s1, lab, pred10, n_true)
    nt_bucket = n_true.clip(upper=4).map(lambda v: f"{v}{'+' if v == 4 else ''}")
    R["loss_all_val"] = summarize_loss(lp)
    R["loss_by_country"] = summarize_loss(lp, country_of)
    R["loss_by_true_count"] = summarize_loss(lp, nt_bucket)
    L_ = R["loss_all_val"]
    log(f"validation (Phase 10 rule, refit on all val): F0.5 {L_['macro_f05']} = 1 - "
        f"{L_['loss_block']} (never reached Stage 2) - {L_['loss_fn']} (missed in list) - {L_['loss_fp']} (FP)")
    gt = pd.read_csv(Path(a.phase0_dir) / "gt_long.tsv", sep="\t", dtype=str, keep_default_na=False)
    owner = gt.drop_duplicates("matched_id").set_index("matched_id")["s1_id"]
    scored_train = set(read_table(c8 / "train", columns=["s1_id"])["s1_id"].unique())
    R["fp_ownership"] = fp_ownership(s1, cand, lab, pred10, owner, split.set_index("s1_id")["split"],
                                     scored_train, n_true)
    R["train_s1_scored_share"] = round(len(scored_train) / max(int((split["split"] == "train").sum()), 1), 4)
    pc_val = calibrate(dec10["calibration"], p, grp)
    R["o2o_removal_val"] = o2o_rate(s1, cand, pc_val)
    log(f"FP ownership: {R['fp_ownership']}")

    # test scores (for the o2o comparison and output_v4)
    TS = read_table(p10 / "cache" / "test_scores")
    TS["p"] = pd.to_numeric(TS["p"])
    shards = P8.list_shards(Path(a.stage2_dir) / "test")
    with ThreadPoolExecutor(max_workers=min(16, a.workers)) as ex:
        parts = list(ex.map(lambda b: read_table(Path(b), columns=["s1_id", "cand_id", "address_missing_cand"]),
                            shards))
    tg = pd.concat(parts, ignore_index=True)
    tg["address_missing_cand"] = pd.to_numeric(tg["address_missing_cand"]).fillna(0).astype(np.int8)
    TS = TS.merge(tg, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    del parts, tg
    if TS["address_missing_cand"].isna().any():
        raise RuntimeError("test_scores rows without a data_stage2/test row - caches out of sync")
    ts1, tcand = TS["s1_id"].to_numpy(dtype=object), TS["cand_id"].to_numpy(dtype=object)
    tgrp, tp_ = TS["address_missing_cand"].to_numpy().astype(np.int8), TS["p"].to_numpy(np.float32)
    test_s1 = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id", "country_norm"])
    tcountry = pd.Series(ts1).map(test_s1.set_index("entity_id")["country_norm"]).to_numpy(dtype=object)
    R["o2o_removal_test_by_country"] = o2o_rate(ts1, tcand, calibrate(dec10["calibration"], tp_, tgrp), tcountry)
    log(f"one-to-one removes {R['o2o_removal_val']['removed_share']:.2%} of confident claims on validation, "
        f"on test by country: { {k: v['removed_share'] for k, v in R['o2o_removal_test_by_country'].items()} }")

    # ================================================================ decide
    rule0, calA, fA0 = P9.tune_decision(p, lab, s1, cand, grp, nA, A)
    fB0 = P8.macro_f05(s1[B], lab[B], decide(rule0, calA, p[B], s1[B], cand[B], grp[B]), nB)["macro_f05"]
    log(f"baseline (Phase 10 rule family): A {fA0:.5f}  B {fB0:.5f}  {rule0}")
    muA = float((nA.sum() - lab[A].sum()) / max(len(nA), 1))
    res = [{"rule": rule0, "A": fA0, "B": fB0, "baseline": True}]
    for r in candidate_rules(muA, a.K):
        fA = P8.macro_f05(s1[A], lab[A], decide(r, calA, p[A], s1[A], cand[A], grp[A]), nA)["macro_f05"]
        fB = P8.macro_f05(s1[B], lab[B], decide(r, calA, p[B], s1[B], cand[B], grp[B]), nB)["macro_f05"]
        res.append({"rule": r, "A": fA, "B": fB})
        log(f"   {json.dumps(r):<75} A {fA:.5f}  B {fB:.5f}")
    best = max(res[1:], key=lambda x: x["A"])
    adopt = best["A"] - fA0 >= a.min_gain
    R["decision"] = {"results": res, "best_new": best, "adopted": adopt,
                     "gain_A": round(best["A"] - fA0, 5), "gain_B": round(best["B"] - fB0, 5)}
    log(f"best new rule {best['rule']}: A {best['A'] - fA0:+.5f}, B {best['B'] - fB0:+.5f} vs baseline -> "
        f"{'ADOPTED' if adopt else 'not adopted (gain on A below --min-gain)'}")

    if adopt:
        _, cal_all, _ = P9.tune_decision(p, lab, s1, cand, grp, n_true, np.ones(len(p), bool))
        rule = dict(best["rule"])
        if rule["mu"]:
            rule["mu"] = float((n_true.sum() - lab.sum()) / len(n_true))
        pred = decide(rule, cal_all, tp_, ts1, tcand, tgrp)
        sub = Path(a.submit_dir)
        sub.mkdir(parents=True, exist_ok=True)
        tp = pd.DataFrame({"s1_id": ts1, "cand_id": tcand})
        P9.write_lists(sub / "candidate_pairs.tsv", "candidate_entity_ids",
                       tp.groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict(), test_s1["entity_id"])
        P9.write_lists(sub / "matching_results.tsv", "matched_entity_ids",
                       tp[pred].groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict(), test_s1["entity_id"])
        pm = pd.Series(pred.astype(int), index=ts1).groupby(level=0).sum().reindex(test_s1["entity_id"], fill_value=0)
        R["test_matches_per_s1"] = {str(c): round(float(g.mean()), 3)
                                    for c, g in pm.groupby(test_s1.set_index("entity_id")["country_norm"]
                                                           .reindex(pm.index).to_numpy())}
        R["submission_checks"] = P8.check_submission(sub, a.block_dir, test_s1["entity_id"])
        P8.atomic_json(out / "decision.json", {"rule": rule, "calibration": cal_all, "phase10_variant": chosen})
        log(f"wrote {sub}/ - checks {R['submission_checks']}; matches/S1 {R['test_matches_per_s1']}")
        if not R["submission_checks"]["all_ok"]:
            raise RuntimeError(f"submission checks failed: {R['submission_checks']}")

    # ================================================================ curve (opt-in)
    if a.learning_curve:
        R["learning_curve"] = learning_curve(a, V={"s1": s1, "cand": cand, "lab": lab, "grp": grp,
                                                   "nA": nA, "nB": nB, "A": A, "B": B})
    P8.atomic_json(out / "report.json", R)
    write_summary(R, out / "summary.md")
    log(f"done -> {out / 'summary.md'}")


def learning_curve(a, V) -> dict:
    """Level-1 model (Phase 9 base + cross features) trained on growing fractions of the training
    S1s, same holdout and hyper-parameters, each tuned on A and scored on B."""
    c8, c9 = Path(a.phase8_dir) / "cache", Path(a.phase9_dir) / "cache"
    base_feats = json.loads((c8 / "features.json").read_text())
    variants9 = json.loads((c9 / "variants.json").read_text())
    best_fs = max(("all", "no_stage1", "rel_stage1"), key=lambda f: variants9.get(f"fs_{f}", {"A": -1})["A"])
    if best_fs != "all" and variants9[f"fs_{best_fs}"]["A"] - variants9["fs_all"]["A"] < a.min_gain:
        best_fs = "all"
    feats = P9.feature_set(best_fs, base_feats)
    TR = P9.add_relative_stage1(P8.prepare(read_table(c8 / "train"), base_feats)[0])
    VA = P9.add_relative_stage1(P8.prepare(read_table(c8 / "val"), base_feats)[0])
    X_tr = np.hstack([TR[feats].to_numpy(np.float32),
                      read_table(c9 / "cross_train")[P9.CROSS].apply(pd.to_numeric).to_numpy(np.float32)])
    X_va = np.hstack([VA[feats].to_numpy(np.float32),
                      read_table(c9 / "cross_val")[P9.CROSS].apply(pd.to_numeric).to_numpy(np.float32)])
    y = pd.to_numeric(TR["label"]).to_numpy().astype(np.int8)
    bucket = P8.bucket_of(TR["s1_id"].to_numpy())
    hold_pct = 10
    hold = bucket < hold_pct
    kw = dict(threads=a.workers, max_rounds=5000, early_stop=100, num_leaves=127, min_leaf=200)
    out = {}
    for frac in (float(x) for x in a.curve_fracs.split(",")):
        use = (~hold) & (bucket < hold_pct + frac * (100 - hold_pct))
        m, b = P9.train_model("lightgbm", X_tr[use], y[use], X_tr[hold], y[hold], a.curve_lr, 77, **kw)
        pv = P9.predict("lightgbm", m, b, X_va, a.workers)
        ev = P9.evaluate_variant(f"curve_{frac}", pv, V)
        out[str(frac)] = {"train_s1": int(pd.Series(TR["s1_id"][use]).nunique()), "rounds": b,
                          "A": round(ev["A"], 5), "B": round(ev["B"], 5)}
        log(f"learning curve: {frac:.2f} of training S1s -> {out[str(frac)]}")
    return out


def write_summary(R, path):
    L = ["# Phase 11 - loss breakdown and decision layer\n",
         "## Where the validation F0.5 goes (Phase 10 rule, all validation S1)",
         "loss_block = true match never reached Stage 2; loss_fn = in the list but not predicted; "
         "loss_fp = false positives. The three add up to 1 - F0.5.\n",
         "| segment | S1 | F0.5 | loss_block | loss_fn | loss_fp | S1 with a wrong pick |", "|---|---|---|---|---|---|---|"]

    def row(name, d):
        return (f"| {name} | {d['s1']:,} | {d['macro_f05']} | {d['loss_block']} | {d['loss_fn']} | "
                f"{d['loss_fp']} | {d['wrong_pick_s1']:,} |")
    L.append(row("all", R["loss_all_val"]))
    for k, d in R["loss_by_country"].items():
        L.append(row(f"country {k}", d))
    for k, d in R["loss_by_true_count"].items():
        L.append(row(f"true matches {k}", d))
    fo = R["fp_ownership"]
    L += ["\n## Who owns the false-positive records",
          "One-to-one can only remove an FP whose owner S1 is scored and claims it more strongly: on validation "
          "only validation S1s compete, on test every test S1 does.\n",
          f"| owner of the FP record | all FP ({fo['fp_pairs']:,}) | FP on singleton S1s |", "|---|---|---|"]
    for k in ("owner_is_val_s1", "owner_is_scored_train_s1", "owner_is_unscored_train_s1", "no_owner"):
        a_, s_ = fo["all_fp"].get(k, [0, 0.0]), fo["fp_on_singleton_s1"].get(k, [0, 0.0])
        L.append(f"| {k} | {a_[0]:,} ({a_[1]:.1%}) | {s_[0]:,} ({s_[1]:.1%}) |")
    L += [f"\nTraining S1s that have scored candidate lists: {R['train_s1_scored_share']:.1%}",
          "\n## One-to-one: share of confident claims removed", "```",
          json.dumps({"validation": R["o2o_removal_val"], "test": R["o2o_removal_test_by_country"]}, indent=1), "```",
          "\n## Decision rules (tuned on A, honest score on B)", "| rule | A | B |", "|---|---|---|"]
    for r in R["decision"]["results"]:
        L.append(f"| {'**baseline** ' if r.get('baseline') else ''}{json.dumps(r['rule'])} | {r['A']:.5f} | {r['B']:.5f} |")
    d = R["decision"]
    L.append(f"\nBest new rule vs baseline: A {d['gain_A']:+.5f}, B {d['gain_B']:+.5f} -> "
             f"**{'adopted, output_v4/ written' if d['adopted'] else 'not adopted'}**")
    if "submission_checks" in R:
        L += ["```", json.dumps({"matches_per_s1": R["test_matches_per_s1"], "checks": R["submission_checks"]},
                                indent=1), "```"]
    if "learning_curve" in R:
        L += ["\n## Learning curve (level-1 model, fraction of training S1s)", "```",
              json.dumps(R["learning_curve"], indent=1), "```"]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
