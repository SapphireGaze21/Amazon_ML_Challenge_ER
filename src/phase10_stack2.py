#!/usr/bin/env python3
"""
Phase 10 - second iteration of cross-candidate evidence (level-2 stacking), on top of Phase 9.

Standalone: it imports YOUR src/phase9_improve.py (whatever cross features it contains, e.g. the
address-ambiguity features) and reuses its saved artifacts - nothing has to be merged:
  artifacts/phase8/cache/{train,val}           base features + labels
  artifacts/phase9/cache/{cross_train,cross_val, variants.json, valp_*}
  artifacts/phase9/models/{fs_<set>, stack_second, xgb}

Idea: Phase 9's cross-candidate features were computed from the FIRST model's probabilities
(~0.963 quality). The level-1 stack is far better (~0.978), so recomputing the probability-
dependent cross features from level-1 probabilities gives sharper sibling / decoy evidence.
  1 oof     out-of-fold level-1 probabilities on train (K folds by S1, same rounds as stack_second)
  2 cross2  probability-dependent cross features recomputed from those probabilities
     + token-disagreement features (improvement plan B3/B1): which name / address tokens exist on
       only one side, whether the address differs ONLY in numbers (the "142b vs 145b" decoys),
       whether the names differ by a single token swap. A comparison with a missing side is
       NaN - an explicit "missing" state, never a fake 0 (Fellegi-Sunter style)
  3 models  level-2 LightGBM + XGBoost + blend, plus a LightGBM variant with hard negatives
            upweighted (plan B4); all tuned on half A, scored on half B
  4 select  best of {level-2 variants, Phase 9's best level-1 variant}; level 2 must beat
            level 1 by --min-gain on half A, otherwise the level-1 variant is kept
  5 test    ONE test pass (level-0 -> cross -> level-1 [-> cross2 -> level-2]),
            output_v3/ (main) + output_v3/variants/gamma<g>/ ; every README rule checked

Every stage is checkpointed (artifacts/phase10/state); rerun the same command to resume.
Usage:  python src/phase10_stack2.py            [--stop-after select]
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import phase8_match as P8
import phase9_improve as P9
from common import read_table
from phase7_stage2_features import load_store

T0 = time.time()
STAGES = ["oof", "cross2", "models", "select", "test"]
P_DEP_ALL = ["p1", "p1_rank", "p1_ratio", "p1_max_other", "p1_sum_other", "n_conf_other", "sib_addr_max",
             "sib_addr_same_max", "sib_num_max", "sib_postal_max", "sib_name_max", "sib_phon_max",
             "conf_sim_addr", "conf_sim_name", "sib_num_conflict_max", "sib_same_name_num_conflict_max"]


DIFF = ["name_only_s1", "name_only_cand", "name_diff_frac", "name_single_swap",
        "addr_only_s1", "addr_only_cand", "addr_diff_frac", "addr_diff_only_numbers",
        "addr_diff_numeric_count", "addr_same_except_numbers"]
DG = {}


def _has_digit(t):
    return any(ch.isdigit() for ch in t)


def diff_chunk(df: pd.DataFrame) -> np.ndarray:
    """Token-disagreement features for (s1_id, cand_id) rows. NaN = comparison impossible."""
    store = DG["store"]
    out = np.full((len(df), len(DIFF)), np.nan, np.float32)
    cache = {}

    def toks(ids):
        pos, ok = store.lookup(ids)
        up, inv = np.unique(np.where(ok, pos, 0), return_inverse=True)
        names = store.take("name_core", up)
        addrs = store.take("address_tokens_norm", up)
        nt = [frozenset(x.split()) if isinstance(x, str) else frozenset() for x in names]
        at = [frozenset(x.split()) if isinstance(x, str) else frozenset() for x in addrs]
        return [(nt[i], at[i]) if o else (frozenset(), frozenset()) for i, o in zip(inv, ok)]

    A = toks(df["s1_id"].to_numpy(dtype=object))
    B = toks(df["cand_id"].to_numpy(dtype=object))
    for r, ((na, aa), (nb, ab)) in enumerate(zip(A, B)):
        if na and nb:
            oa, ob = len(na - nb), len(nb - na)
            out[r, 0], out[r, 1] = oa, ob
            out[r, 2] = (oa + ob) / len(na | nb)
            out[r, 3] = float(oa == 1 and ob == 1)
        if aa and ab:
            da, db = aa - ab, ab - aa
            sym = da | db
            out[r, 4], out[r, 5] = len(da), len(db)
            out[r, 6] = len(sym) / len(aa | ab)
            nnum = sum(_has_digit(t) for t in sym)
            out[r, 7] = float(bool(sym) and nnum == len(sym))
            out[r, 8] = nnum
            out[r, 9] = float({t for t in aa if not _has_digit(t)} == {t for t in ab if not _has_digit(t)})
    return out


def diff_features(df: pd.DataFrame, store, workers: int, chunk: int = 200_000) -> np.ndarray:
    import multiprocessing as mp
    DG["store"] = store
    parts = [df.iloc[i:i + chunk][["s1_id", "cand_id"]] for i in range(0, len(df), chunk)]
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    if ctx.get_start_method() == "fork" and workers > 1:
        with ctx.Pool(workers) as pool:
            res = pool.map(diff_chunk, parts)
    else:
        res = [diff_chunk(x) for x in parts]
    return np.vstack(res) if res else np.zeros((0, len(DIFF)), np.float32)


def train_weighted_lgb(backend, X, y, w, Xv, yv, lr, seed, a):
    """LightGBM with row weights (hard-negative upweighting); sklearn fallback for tests."""
    if backend == "lightgbm":
        import lightgbm as lgb
        prm = {"objective": "binary", "metric": "binary_logloss", "learning_rate": lr,
               "num_leaves": a.num_leaves, "min_data_in_leaf": a.min_leaf, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "num_threads": a.workers, "verbose": -1, "seed": seed}
        dtr = lgb.Dataset(X, label=y, weight=w, free_raw_data=True)
        dv = lgb.Dataset(Xv, label=yv, reference=dtr)
        m = lgb.train(prm, dtr, num_boost_round=a.max_rounds, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(a.early_stop, verbose=False), lgb.log_evaluation(500)])
        return m, int(m.best_iteration or a.max_rounds)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=150, max_leaf_nodes=a.num_leaves,
                                       random_state=seed, early_stopping=False)
    m.fit(X, y, sample_weight=w)
    return m, m.n_iter_


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8-dir", default="artifacts/phase8")
    ap.add_argument("--phase9-dir", default="artifacts/phase9")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--stage2-dir", default="data_stage2")
    ap.add_argument("--repr-dir", default="data_repr")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--out-dir", default="artifacts/phase10")
    ap.add_argument("--submit-dir", default="output_v3")
    ap.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--holdout-pct", type=int, default=10)
    ap.add_argument("--min-gain", type=float, default=0.0003)
    ap.add_argument("--test-gammas", type=float, nargs="*", default=[1.15, 1.35])
    ap.add_argument("--no-diff", action="store_true", help="skip the token-disagreement features")
    ap.add_argument("--hard-neg-weight", type=float, default=3.0,
                    help="weight of hard negatives (level-1 OOF p > --hard-neg-p) in the _hw variant; 0 = skip")
    ap.add_argument("--hard-neg-p", type=float, default=0.3)
    ap.add_argument("--stop-after", choices=STAGES)
    ap.add_argument("--from-stage", choices=STAGES)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    out, p9 = Path(a.out_dir), Path(a.phase9_dir)
    mdir, cdir, m9, c9 = out / "models", out / "cache", p9 / "models", p9 / "cache"
    for d in (mdir, cdir):
        d.mkdir(parents=True, exist_ok=True)
    st = P9.State(out / "state")
    P9.STAGES[:] = STAGES                       # State.reset_from uses the module's stage list
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
            log(f"stopping after '{s}'")
            sys.exit(0)

    kw = dict(threads=a.workers, max_rounds=a.max_rounds, early_stop=a.early_stop,
              num_leaves=a.num_leaves, min_leaf=a.min_leaf)
    LGB = a.backend
    CROSS = list(P9.CROSS)
    P_DEP = [c for c in P_DEP_ALL if c in CROSS]
    I2 = [f"i2_{c}" for c in P_DEP]

    # ---- data (same construction as Phase 9)
    c8 = Path(a.phase8_dir) / "cache"
    base_feats = json.loads((c8 / "features.json").read_text())
    p9rep = json.loads((p9 / "report.json").read_text())
    variants9 = json.loads((c9 / "variants.json").read_text())
    best_fs = max(("all", "no_stage1", "rel_stage1"), key=lambda f: variants9.get(f"fs_{f}", {"A": -1})["A"])
    if best_fs != "all" and variants9[f"fs_{best_fs}"]["A"] - variants9["fs_all"]["A"] < a.min_gain:
        best_fs = "all"
    feats = P9.feature_set(best_fs, base_feats)
    TR = P9.add_relative_stage1(P8.prepare(read_table(c8 / "train"), base_feats)[0])
    VA = P9.add_relative_stage1(P8.prepare(read_table(c8 / "val"), base_feats)[0])
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
    Xc_tr = read_table(c9 / "cross_train")[CROSS].apply(pd.to_numeric).to_numpy(np.float32)
    Xc_va = read_table(c9 / "cross_val")[CROSS].apply(pd.to_numeric).to_numpy(np.float32)
    X1_tr = np.hstack([TR[feats].to_numpy(np.float32), Xc_tr])
    X1_va = np.hstack([VA[feats].to_numpy(np.float32), Xc_va])
    log(f"feature set '{best_fs}': {len(feats)} base + {len(CROSS)} cross features; "
        f"{len(P_DEP)} probability-dependent recomputed at level 2")
    k1, m1, b1 = P9.load_model(m9 / "stack_second")
    res_path = cdir / "variants.json"
    variants = json.loads(res_path.read_text()) if res_path.exists() else {}

    def record(name, p, extra=None):
        P8.atomic_table(pd.DataFrame({"p": p}), cdir / f"valp_{name}")
        variants[name] = {**P9.evaluate_variant(name, p, V), **(extra or {})}
        P8.atomic_json(res_path, variants)

    # ---- 1 out-of-fold level-1 probabilities
    if not st.done("oof"):
        fold = P8.bucket_of(TR["s1_id"].to_numpy()) % a.folds
        p2_tr = np.zeros(len(TR), np.float32)
        for k in range(a.folds):
            base = mdir / f"l1_fold{k}"
            sel = fold != k
            if not P9.model_exists(base, X1_tr.shape[1]):
                mk, _ = P9.train_model(LGB, X1_tr[sel], TR.label[sel].to_numpy(), None, None,
                                       a.lr, 300 + k, rounds=b1, **kw)
                P9.save_model(LGB, mk, b1, base)
            kk, mk, bk = P9.load_model(base)
            p2_tr[~sel] = P9.predict(kk, mk, bk, X1_tr[~sel], a.workers)
            log(f"oof: fold {k + 1}/{a.folds}")
        p2_va = P9.predict(k1, m1, b1, X1_va, a.workers)
        P8.atomic_table(pd.DataFrame({"p": p2_tr}), cdir / "p2_train")
        P8.atomic_table(pd.DataFrame({"p": p2_va}), cdir / "p2_val")
        st.mark("oof")
    stop("oof")

    # ---- 2 level-2 cross features
    if not st.done("cross2"):
        p2_tr = pd.to_numeric(read_table(cdir / "p2_train")["p"]).to_numpy(np.float32)
        p2_va = pd.to_numeric(read_table(cdir / "p2_val")["p"]).to_numpy(np.float32)
        store = load_store(Path(a.repr_dir), "train")
        log("cross2: train")
        C2_tr = P9.cross_features(TR.assign(p1=p2_tr), store, a.workers)[P_DEP]
        log("cross2: validation")
        C2_va = P9.cross_features(VA.assign(p1=p2_va), store, a.workers)[P_DEP]
        del store
        C2_tr.columns = C2_va.columns = I2
        if not a.no_diff:
            log("cross2: token-disagreement features (train, validation)")
            store = load_store(Path(a.repr_dir), "train")
            D_tr = diff_features(TR, store, a.workers)
            D_va = diff_features(VA, store, a.workers)
            del store
            for j, c in enumerate(DIFF):
                C2_tr[c], C2_va[c] = D_tr[:, j], D_va[:, j]
        P8.atomic_table(C2_tr, cdir / "cross2_train")
        P8.atomic_table(C2_va, cdir / "cross2_val")
        st.mark("cross2")
    stop("cross2")

    # ---- 3 level-2 models
    t2_tr, t2_va = read_table(cdir / "cross2_train"), read_table(cdir / "cross2_val")
    L2COLS = I2 + [c for c in DIFF if c in t2_tr.columns]
    C2_tr = t2_tr[L2COLS].apply(pd.to_numeric).to_numpy(np.float32)
    C2_va = t2_va[L2COLS].apply(pd.to_numeric).to_numpy(np.float32)
    del t2_tr, t2_va
    X2_tr, X2_va = np.hstack([X1_tr, C2_tr]), np.hstack([X1_va, C2_va])
    if not st.done("models"):
        if "l1_best" not in variants:        # re-evaluate Phase 9's best level-1 variant on this data
            best9 = max((k for k in variants9 if k not in ("phase8_baseline", "final")), key=lambda k: variants9[k]["A"])
            p = pd.to_numeric(read_table(c9 / f"valp_{best9}")["p"]).to_numpy(np.float32)
            record("l1_best", p, {"phase9_variant": best9})
        base = mdir / "l2_lgb"
        if not P9.model_exists(base, X2_tr.shape[1]):
            m, b = P9.train_model(LGB, X2_tr[~hold], TR.label[~hold].to_numpy(), X2_tr[hold],
                                  TR.label[hold].to_numpy(), a.lr, 401, **kw)
            P9.save_model(LGB, m, b, base)
        kk, mm, bb = P9.load_model(base)
        pl = P9.predict(kk, mm, bb, X2_va, a.workers)
        record("l2_lgb", pl)
        try:
            import xgboost  # noqa: F401
            base = mdir / "l2_xgb"
            if not P9.model_exists(base, X2_tr.shape[1]):
                m, b = P9.train_model("xgboost", X2_tr[~hold], TR.label[~hold].to_numpy(), X2_tr[hold],
                                      TR.label[hold].to_numpy(), a.lr, 402, **kw)
                P9.save_model("xgboost", m, b, base)
            kk, mm, bb = P9.load_model(base)
            px = P9.predict(kk, mm, bb, X2_va, a.workers)
            record("l2_xgb", px)
            record("l2_blend", (pl + px) / 2)
        except ImportError:
            log("WARNING xgboost not installed - level-2 XGBoost skipped")
        if a.hard_neg_weight > 0:
            base = mdir / "l2_lgb_hw"
            if not P9.model_exists(base, X2_tr.shape[1]):
                p2_tr = pd.to_numeric(read_table(cdir / "p2_train")["p"]).to_numpy(np.float32)
                y = TR.label.to_numpy()
                w = np.where((y == 0) & (p2_tr > a.hard_neg_p), a.hard_neg_weight, 1.0).astype(np.float32)
                log(f"hard negatives upweighted: {int(((y == 0) & (p2_tr > a.hard_neg_p)).sum()):,} rows x {a.hard_neg_weight}")
                m, b = train_weighted_lgb(LGB, X2_tr[~hold], y[~hold], w[~hold], X2_tr[hold], y[hold], a.lr, 403, a)
                P9.save_model(LGB, m, b, base)
            kk, mm, bb = P9.load_model(base)
            ph = P9.predict(kk, mm, bb, X2_va, a.workers)
            record("l2_lgb_hw", ph)
            if "l2_xgb" in variants:
                px = pd.to_numeric(read_table(cdir / "valp_l2_xgb")["p"]).to_numpy(np.float32)
                record("l2_blend_hw", (ph + px) / 2)
        R["variants"] = variants
        save(); st.mark("models")  # noqa: E702
    stop("models")

    # ---- 4 select
    if not st.done("select"):
        l2 = [k for k in variants if k.startswith("l2_")]
        best2 = max(l2, key=lambda k: variants[k]["A"]) if l2 else None
        gain = variants[best2]["A"] - variants["l1_best"]["A"] if best2 else -1
        chosen = best2 if best2 and gain >= a.min_gain else "l1_best"
        log(f"select: best level-2 {best2} (A gain {gain:+.5f}) -> using {chosen} "
            f"(A {variants[chosen]['A']:.5f}, B {variants[chosen]['B']:.5f})")
        p_final = pd.to_numeric(read_table(cdir / f"valp_{chosen}")["p"]).to_numpy(np.float32)
        rule, cal, _ = P9.tune_decision(p_final, V["lab"], V["s1"], V["cand"], V["grp"], n_true,
                                        np.ones(len(p_final), bool))
        if chosen == "l1_best":
            v9 = variants["l1_best"]["phase9_variant"]
            models = {"xgb": [m9 / "xgb"], "stack": [m9 / "stack_second"],
                      "blend": [m9 / "stack_second", m9 / "xgb"]}.get(v9, [m9 / v9])
            level = 1
        else:
            models = {"l2_lgb": [mdir / "l2_lgb"], "l2_xgb": [mdir / "l2_xgb"],
                      "l2_blend": [mdir / "l2_lgb", mdir / "l2_xgb"], "l2_lgb_hw": [mdir / "l2_lgb_hw"],
                      "l2_blend_hw": [mdir / "l2_lgb_hw", mdir / "l2_xgb"]}[chosen]
            level = 2
        P8.atomic_json(out / "decision.json", {"rule": rule, "calibration": cal, "level": level,
                                               "models": [str(x) for x in models], "chosen": chosen})
        R["selected"] = {"variant": chosen, "level": level, "A": variants[chosen]["A"], "B": variants[chosen]["B"],
                         "rule_for_test": rule, "leaderboard_B": sorted([[k, v["B"], v["A"]] for k, v in variants.items()],
                                                                        key=lambda x: -x[1])}
        save(); st.mark("select")  # noqa: E702
    stop("select")

    # ---- 5 test: one pass
    if not st.done("test"):
        dec = json.loads((out / "decision.json").read_text())
        shards = P8.list_shards(Path(a.stage2_dir) / "test")
        log(f"test: reading {len(shards)} shards")
        with ThreadPoolExecutor(max_workers=min(16, a.workers)) as ex:
            parts = list(ex.map(lambda b: read_table(Path(b)), shards))
        TE = P9.add_relative_stage1(P8.prepare(pd.concat(parts, ignore_index=True), base_feats)[0])
        del parts
        X0 = TE[feats].to_numpy(np.float32)
        k0, m0, b0 = P9.load_model(m9 / f"fs_{best_fs}")
        p1 = P9.predict(k0, m0, b0, X0, a.workers)
        store = load_store(Path(a.repr_dir), "test")
        log("test: level-1 cross features")
        C1 = P9.cross_features(TE.assign(p1=p1), store, a.workers)[CROSS].to_numpy(np.float32)
        X1 = np.hstack([X0, C1])
        if dec["level"] == 2:
            p2 = P9.predict(k1, m1, b1, X1, a.workers)
            log("test: level-2 cross features")
            C2 = P9.cross_features(TE.assign(p1=p2), store, a.workers)[P_DEP].to_numpy(np.float32)
            if len(L2COLS) > len(I2):
                log("test: token-disagreement features")
                C2 = np.hstack([C2, diff_features(TE, store, a.workers)])
            Xf = np.hstack([X1, C2])
        else:
            Xf = X1
        del store
        by_kind = {}
        for base in dec["models"]:
            kk, mm, bb = P9.load_model(Path(base))
            by_kind.setdefault(kk, []).append(P9.predict(kk, mm, bb, Xf, a.workers))
            log(f"test: scored with {Path(base).name}")
        p = np.mean([np.mean(v, axis=0) for v in by_kind.values()], axis=0).astype(np.float32)
        s1, cand = TE["s1_id"].to_numpy(dtype=object), TE["cand_id"].to_numpy(dtype=object)
        grp = TE["address_missing_cand"].to_numpy().astype(np.int8)
        P8.atomic_table(pd.DataFrame({"s1_id": s1, "cand_id": cand, "p": p}), cdir / "test_scores")
        test_s1 = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id", "country_norm"])
        country_of = test_s1.set_index("entity_id")["country_norm"]
        sub = Path(a.submit_dir)
        sub.mkdir(parents=True, exist_ok=True)
        tp = pd.DataFrame({"s1_id": s1, "cand_id": cand})
        P9.write_lists(sub / "candidate_pairs.tsv", "candidate_entity_ids",
                       tp.groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict(), test_s1["entity_id"])
        R["test_variants"] = {}
        for name, gm in [("main", 1.0)] + [(f"gamma{g}", g) for g in a.test_gammas]:
            pred = P9.apply_decision(dec["rule"], dec["calibration"], p, s1, cand, grp, gm)
            d = sub if name == "main" else sub / "variants" / name
            d.mkdir(parents=True, exist_ok=True)
            P9.write_lists(d / "matching_results.tsv", "matched_entity_ids",
                           tp[pred].groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict(),
                           test_s1["entity_id"])
            if name != "main":
                shutil.copyfile(sub / "candidate_pairs.tsv", d / "candidate_pairs.tsv")
            bc = P9.country_stats(s1, pred, p, dec["rule"], dec["calibration"], grp, country_of, test_s1["entity_id"])
            R["test_variants"][name] = {c: v["mean_pred_matches"] for c, v in bc.items()}
            log(f"test variant {name}: matches/S1 {R['test_variants'][name]}")
        R["submission_checks"] = P8.check_submission(sub, a.block_dir, test_s1["entity_id"])
        save()
        if not R["submission_checks"]["all_ok"]:
            raise RuntimeError(f"submission checks failed: {R['submission_checks']}")
        st.mark("test")
        log(f"wrote {sub}/ (checks OK)")
    L = ["# Phase 10 - level-2 stacking\n", "| variant | B | A |", "|---|---|---|"]
    L += [f"| {k} | {b:.5f} | {a_:.5f} |" for k, b, a_ in R.get("selected", {}).get("leaderboard_B", [])]
    if "selected" in R:
        L.append(f"\n**Selected: {R['selected']['variant']} (level {R['selected']['level']}), "
                 f"B {R['selected']['B']:.5f}**; rule {R['selected']['rule_for_test']}")
    if "test_variants" in R:
        L += ["\n## Test variants (matches per S1)", "```", json.dumps(R["test_variants"], indent=1), "```",
              "```", json.dumps(R.get("submission_checks"), indent=1), "```"]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    log(f"done -> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
