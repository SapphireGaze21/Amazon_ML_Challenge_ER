#!/usr/bin/env python3
"""
Phase 8 - Stage-2 matcher, decision layer, evaluation, test inference and submission.
One script, seven checkpointed stages. Re-running the same command skips finished stages;
test prediction additionally resumes shard by shard.

  1 data          load data_stage2/{train,val} in parallel, keep val rows with stage1_rank < N
                  (N = the Stage-1 N used for test, so val is scored exactly like test), attach
                  ground-truth labels, cache as tables.
  2 model         LightGBM (MIT) binary classifiers, an ensemble of --n-models seeds; early
                  stopping on a hash-based HOLDOUT of train S1s (validation stays untouched).
  3 val_predict   score every validation pair (averaged ensemble probability).
  4 decision      split validation S1s into halves A / B (hash). On A: fit a monotone (PAV)
                  probability calibration, then compare decision rules
                    - global threshold (grid),
                    - per-S1 expected-F0.5 subset selection (handles singletons: predicts
                      empty when "no match" has the highest expected score),
                  each with / without the one-to-one rule (every S2/S3 record goes to at most
                  its best S1 - true in 100% of the ground truth). The best rule on A is
                  scored on B (the honest estimate), then refit on all of validation for test.
  5 evaluate      macro F0.5 / precision / recall on B and on all validation, by country, match
                  count, singletons; pair-level results for cross-script / missing address;
                  ceiling check (oracle at N must equal Phase 6); error examples.
  6 test_predict  score every test pair (parallel, resumable per shard).
  7 submit        apply calibration + one-to-one + decision rule to test, write
                  output/matching_results.tsv and output/candidate_pairs.tsv (exactly the pairs
                  the model scored), check every README rule, run utils/validate_submission.py
                  if present, compare predicted-match statistics by country (France check).

Macro F0.5 is computed over ALL S1 of the evaluated set: S1s without candidates are predicted
empty; matches lost before Stage 2 count as misses (n_true comes from s1_split.tsv).

Usage
  python src/phase8_match.py                    # run / resume everything
  python src/phase8_match.py --from-stage decision   # redo from a stage onward
  python src/phase8_match.py --n-models 1       # quicker, slightly weaker
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_table, write_table

T0 = time.time()
STAGES = ["data", "model", "val_predict", "decision", "evaluate", "test_predict", "submit"]
PASSES = ["exact_sorted", "exact_concat", "exact_alias", "sparse_comb", "sparse_name",
          "sparse_addr", "char_name", "char_addr", "sparse_phon", "noaddr_name",
          "nonlatin_phon", "expand_prf", "sparse_cross"]
ID_COLS = ["s1_id", "cand_id"]


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ============================================================================ checkpoint helpers

class State:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def done(self, stage):
        return (self.root / f"{stage}.done.json").exists()

    def mark(self, stage, info=None):
        atomic_json(self.root / f"{stage}.done.json", {"stage": stage, "time": time.time(), **(info or {})})

    def reset_from(self, stage):
        for s in STAGES[STAGES.index(stage):]:
            p = self.root / f"{s}.done.json"
            if p.exists():
                p.unlink()


def atomic_json(path: Path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_table(df: pd.DataFrame, base: Path) -> Path:
    """Write to a temporary name, then rename: a crash never leaves a half-written table."""
    tmp_base = base.parent / (base.name + "__tmp")          # no dot: write_table adds the extension
    p = write_table(df, tmp_base)
    final = base.parent / (base.name + p.name[len(tmp_base.name):])
    os.replace(p, final)
    return final


def table_exists(base: Path) -> bool:
    return Path(str(base) + ".parquet").exists() or Path(str(base) + ".tsv.gz").exists()


def list_shards(d: Path) -> list[str]:
    out = []
    for p in sorted(Path(d).iterdir()):
        for ext in (".parquet", ".tsv.gz"):
            if p.name.endswith(ext) and "__tmp" not in p.name:
                out.append(str(p)[:-len(ext)])
    return out


def half_of(ids) -> np.ndarray:
    """Deterministic A/B assignment of S1s (0 = A, 1 = B)."""
    u = pd.unique(pd.Series(ids))
    m = {x: (zlib.crc32(("half:" + x).encode()) & 1) for x in u}
    return pd.Series(ids).map(m).to_numpy(np.int8)


def bucket_of(ids) -> np.ndarray:
    u = pd.unique(pd.Series(ids))
    m = {x: zlib.crc32(x.encode()) % 100 for x in u}
    return pd.Series(ids).map(m).to_numpy(np.int16)


# ============================================================================ features

def prepare(df: pd.DataFrame, feats: list | None = None) -> tuple[pd.DataFrame, list]:
    """Numeric conversion + pass bits. Returns (df, feature list)."""
    for c in df.columns:
        if c not in ID_COLS and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "passes" in df:
        pv = df["passes"].fillna(0).astype(np.int64).to_numpy()
        for i, p in enumerate(PASSES):
            if p != "expand_prf":
                df[f"pass_{p}"] = ((pv >> i) & 1).astype(np.float32)
    if feats is None:
        feats = [c for c in df.columns if c not in ID_COLS + ["passes", "label"]]
    for c in feats:
        if c not in df:
            df[c] = 0.0
    df[feats] = df[feats].astype(np.float32).fillna(0.0)
    return df, feats


def load_shard_task(args):
    base, n_keep = args
    df = read_table(Path(base))
    if n_keep:
        df["stage1_rank"] = pd.to_numeric(df["stage1_rank"], errors="coerce")
        df = df[df["stage1_rank"] < n_keep]
    return df


def load_split(d: Path, n_keep: int, workers: int) -> pd.DataFrame:
    shards = list_shards(d)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        parts = p.map(load_shard_task, [(s, n_keep) for s in shards], chunksize=4)
    return pd.concat(parts, ignore_index=True)


def attach_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(gt, on=ID_COLS, how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    return df


# ============================================================================ model

class Model:
    """LightGBM binary classifier (default; MIT license). backend 'sklearn' exists ONLY for
    testing without LightGBM - scikit-learn is BSD-3, not allowed for the final model."""

    def __init__(self, backend, params):
        self.backend, self.params, self.m, self.best_iter = backend, params, None, None

    def fit(self, X, y, Xv, yv, seed):
        P = self.params
        if self.backend == "lightgbm":
            import lightgbm as lgb
            prm = {"objective": "binary", "metric": "binary_logloss", "learning_rate": P["lr"],
                   "num_leaves": P["num_leaves"], "min_data_in_leaf": P["min_leaf"],
                   "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                   "lambda_l2": 1.0, "num_threads": P["threads"], "verbose": -1, "seed": seed}
            dtr = lgb.Dataset(X, label=y, free_raw_data=True)
            dv = lgb.Dataset(Xv, label=yv, reference=dtr)
            self.m = lgb.train(prm, dtr, num_boost_round=P["max_rounds"], valid_sets=[dv],
                               callbacks=[lgb.early_stopping(P["early_stop"], verbose=False),
                                          lgb.log_evaluation(200)])
            self.best_iter = int(self.m.best_iteration or P["max_rounds"])
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.m = HistGradientBoostingClassifier(learning_rate=P["lr"], max_iter=min(P["max_rounds"], 200),
                                                    max_leaf_nodes=P["num_leaves"], random_state=seed,
                                                    early_stopping=False)
            self.m.fit(X, y)
            self.best_iter = self.m.n_iter_
        return self.best_iter

    def predict(self, X, threads=None):
        if self.backend == "lightgbm":
            kw = {"num_threads": threads} if threads else {}
            return self.m.predict(X, num_iteration=self.best_iter, **kw)
        return self.m.predict_proba(X)[:, 1]

    def save(self, base: Path):
        if self.backend == "lightgbm":
            self.m.save_model(str(base) + ".txt", num_iteration=self.best_iter)
        else:
            Path(str(base) + ".pkl").write_bytes(pickle.dumps(self.m))
        atomic_json(Path(str(base) + ".meta.json"), {"backend": self.backend, "best_iter": self.best_iter})

    @staticmethod
    def load(base: Path):
        meta = json.loads(Path(str(base) + ".meta.json").read_text())
        m = Model(meta["backend"], {})
        m.best_iter = meta["best_iter"]
        if meta["backend"] == "lightgbm":
            import lightgbm as lgb
            m.m = lgb.Booster(model_file=str(base) + ".txt")
        else:
            m.m = pickle.loads(Path(str(base) + ".pkl").read_bytes())
        return m

    def importance(self, feats):
        if self.backend == "lightgbm":
            return dict(zip(feats, map(float, self.m.feature_importance(importance_type="gain"))))
        return {}


def ensemble_predict(models, X, threads=None):
    return np.mean([m.predict(X, threads) for m in models], axis=0).astype(np.float32)


# ============================================================================ calibration / decisions

def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing fit of y (weights w)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)  # noqa: E702
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v  # noqa: E702
            vals.pop(); wts.pop(); cnt.pop()  # noqa: E702
    return np.repeat(vals, cnt)


def fit_calibration(p, y, n_bins=200) -> dict:
    """Monotone calibration: quantile bins of p, mean label per bin, PAV-smoothed."""
    q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    cnt = np.bincount(idx, minlength=len(q) - 1).astype(float)
    ok = cnt > 0
    mp_ = np.bincount(idx, weights=p, minlength=len(q) - 1)[ok] / cnt[ok]
    my = np.bincount(idx, weights=y, minlength=len(q) - 1)[ok] / cnt[ok]
    fit = pav(my, cnt[ok])
    return {"x": mp_.tolist(), "y": np.clip(fit, 1e-6, 1 - 1e-6).tolist()}


def apply_calibration(p, cal) -> np.ndarray:
    return np.interp(p, cal["x"], cal["y"]).astype(np.float32)


def one_to_one(s1: np.ndarray, cand: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Zero every claim on a record except its highest-probability S1."""
    d = pd.DataFrame({"c": cand, "p": p, "i": np.arange(len(p))}).sort_values(["c", "p"], ascending=[True, False])
    keep = np.zeros(len(p), bool)
    keep[d.drop_duplicates("c")["i"].to_numpy()] = True
    return np.where(keep, p, 0.0).astype(np.float32)


def decide_threshold(p, t) -> np.ndarray:
    return p >= t


def decide_expected_f(s1: np.ndarray, p: np.ndarray, mu: float) -> np.ndarray:
    """Per S1, choose the top-k set maximizing (approximately) expected F0.5:
         E[F(k)] ~ 1.25 * sum_{i<=k} p_i / (0.25 * (sum_i p_i + mu) + k),   k >= 1
         E[F(0)] = P(no true match) ~ prod_i (1 - p_i) * exp(-mu)
       mu = expected number of true matches NOT in the candidate list (estimated on validation).
    """
    d = pd.DataFrame({"s": s1, "p": np.clip(p, 0.0, 1 - 1e-6), "i": np.arange(len(p))})
    d = d.sort_values(["s", "p"], ascending=[True, False], kind="stable")
    g = d.groupby("s", sort=False)
    k = g.cumcount().to_numpy() + 1
    cum = g["p"].cumsum().to_numpy()
    tot = g["p"].transform("sum").to_numpy()
    score = 1.25 * cum / (0.25 * (tot + mu) + k)
    d["lg"] = np.log1p(-d["p"].to_numpy())
    none = np.exp(d.groupby("s", sort=False)["lg"].transform("sum").to_numpy() - mu)
    d["score"] = score
    best = d.groupby("s", sort=False)["score"].transform("max").to_numpy()
    # k* = first position reaching the maximum
    first_best = (d.assign(hit=score >= best - 1e-12, k=k).query("hit")
                  .groupby("s", sort=False)["k"].first())
    kstar = d["s"].map(first_best).to_numpy()
    take = (k <= kstar) & (best > none)
    out = np.zeros(len(p), bool)
    out[d["i"].to_numpy()] = take
    return out


def macro_f05(s1: np.ndarray, label: np.ndarray, pred: np.ndarray, n_true: pd.Series) -> dict:
    """Macro F0.5 / P / R over ALL S1 in n_true (index = S1 id, value = true match count)."""
    d = pd.DataFrame({"s": s1, "tp": (label == 1) & pred, "np": pred})
    agg = d.groupby("s")[["tp", "np"]].sum().reindex(n_true.index, fill_value=0)
    tp, npred, nt = agg["tp"].to_numpy(float), agg["np"].to_numpy(float), n_true.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(nt == 0, (npred == 0).astype(float),
                     np.where(npred == 0, 0.0, 1.25 * tp / (0.25 * nt + npred)))
    prec = np.where(npred > 0, tp / np.maximum(npred, 1), np.where(nt == 0, 1.0, 0.0))
    rec = np.where(nt > 0, tp / np.maximum(nt, 1), 1.0)
    return {"macro_f05": round(float(f.mean()), 5), "macro_precision": round(float(prec.mean()), 5),
            "macro_recall": round(float(rec.mean()), 5), "s1": int(len(nt)),
            "pred_pairs": int(npred.sum()), "per_s1_f": f}


def apply_rule(rule: dict, s1, cand, p_cal, p_o2o=None) -> np.ndarray:
    """p_o2o: optional precomputed one_to_one(s1, cand, p_cal), to avoid recomputing it."""
    if rule["one_to_one"]:
        p = p_o2o if p_o2o is not None else one_to_one(s1, cand, p_cal)
    else:
        p = p_cal
    if rule["method"] == "threshold":
        return decide_threshold(p, rule["t"])
    return decide_expected_f(s1, p, rule["mu"])


# ============================================================================ test prediction worker

WK = {}


def init_predictor(model_bases, feats):
    WK["models"] = [Model.load(Path(b)) for b in model_bases]
    WK["feats"] = feats


def predict_shard_task(args):
    base, out_base = args
    if table_exists(Path(out_base)):
        return 0
    df, _ = prepare(read_table(Path(base)), WK["feats"])
    p = ensemble_predict(WK["models"], df[WK["feats"]].to_numpy(np.float32), threads=1)
    atomic_table(pd.DataFrame({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(), "p": p}),
                 Path(out_base))
    return len(df)


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", default="data_stage2")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--phase6-dir", default="artifacts/phase6")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--out-dir", default="artifacts/phase8")
    ap.add_argument("--submit-dir", default="output")
    ap.add_argument("--from-stage", choices=STAGES, default=None, help="redo this stage and all later ones")
    ap.add_argument("--top-n", type=int, default=0, help="Stage-1 N (0 = read from phase6 report / test)")
    ap.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    ap.add_argument("--n-models", type=int, default=3, help="ensemble size (different seeds)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--holdout-pct", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    out = Path(a.out_dir)
    st = State(out / "state")
    cache, models_dir, pred_dir = out / "cache", out / "models", out / "test_pred"
    for d in (cache, models_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    if a.from_stage:
        st.reset_from(a.from_stage)
        log(f"reset stages from '{a.from_stage}'")
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["args"] = vars(a)

    def save_report():
        atomic_json(report_path, report)

    split = pd.read_csv(Path(a.phase0_dir) / "s1_split.tsv", sep="\t", dtype={"s1_id": str, "split": str,
                                                                             "country_norm": str})
    split["n_matches"] = split["n_matches"].astype(int)
    val_split = split[split["split"] == "val"].set_index("s1_id")

    # ---- Stage-1 N
    N = a.top_n or report.get("N")
    if not N:
        p6 = Path(a.phase6_dir) / "stage1_report.json"
        if p6.exists():
            N = int(json.loads(p6.read_text()).get("chosen_N", 0))
    if not N:
        t0 = read_table(Path(list_shards(Path(a.stage2_dir) / "test")[0]))
        N = int(pd.to_numeric(t0["stage1_rank"]).max()) + 1
    report["N"] = int(N)
    log(f"Stage-1 N = {N} (validation is cut to stage1_rank < N to match test)")

    # ================================================================ 1 data
    if not st.done("data"):
        gt = pd.read_csv(Path(a.phase0_dir) / "gt_long.tsv", sep="\t", dtype=str, keep_default_na=False)
        gt = gt.rename(columns={"matched_id": "cand_id"})[ID_COLS].assign(label=np.int8(1))
        tr = attach_labels(load_split(Path(a.stage2_dir) / "train", 0, a.workers), gt)
        tr, feats = prepare(tr)
        va = attach_labels(load_split(Path(a.stage2_dir) / "val", N, a.workers), gt)
        va, _ = prepare(va, feats)
        atomic_table(tr[ID_COLS + ["label"] + feats], cache / "train")
        atomic_table(va[ID_COLS + ["label"] + feats], cache / "val")
        atomic_json(cache / "features.json", feats)
        report["data"] = {"train_rows": int(len(tr)), "train_pos": int(tr.label.sum()),
                          "train_s1": int(tr.s1_id.nunique()), "val_rows": int(len(va)),
                          "val_pos": int(va.label.sum()), "val_s1_with_candidates": int(va.s1_id.nunique()),
                          "features": len(feats)}
        save_report(); st.mark("data")  # noqa: E702
        log(f"data: {report['data']}")
        del tr, va
    feats = json.loads((cache / "features.json").read_text())

    # ================================================================ 2 model
    model_bases = [models_dir / f"stage2_seed{i}" for i in range(a.n_models)]
    if not st.done("model"):
        tr = read_table(cache / "train")
        tr, _ = prepare(tr, feats)
        hold = bucket_of(tr["s1_id"].to_numpy()) < a.holdout_pct
        X, y = tr[feats].to_numpy(np.float32), tr["label"].to_numpy()
        params = {"lr": a.lr, "num_leaves": a.num_leaves, "min_leaf": a.min_leaf,
                  "max_rounds": a.max_rounds, "early_stop": a.early_stop, "threads": a.workers}
        iters, imp = [], {}
        for i, base in enumerate(model_bases):
            if Path(str(base) + ".meta.json").exists():            # resume inside the stage
                log(f"model {i}: already trained")
                continue
            m = Model(a.backend, params)
            it = m.fit(X[~hold], y[~hold], X[hold], y[hold], seed=1000 + i)
            m.save(base)
            iters.append(it)
            imp = m.importance(feats) or imp
            log(f"model {i + 1}/{a.n_models}: {it} rounds")
        report["model"] = {"backend": a.backend, "n_models": a.n_models, "rounds": iters,
                           "top_features": sorted(imp.items(), key=lambda kv: -kv[1])[:25]}
        save_report(); st.mark("model")  # noqa: E702
        del tr, X, y

    # ================================================================ 3 val_predict
    if not st.done("val_predict"):
        va = read_table(cache / "val")
        va, _ = prepare(va, feats)
        models = [Model.load(b) for b in model_bases]
        va["p"] = ensemble_predict(models, va[feats].to_numpy(np.float32), threads=a.workers)
        atomic_table(va[ID_COLS + ["label", "p", "stage1_rank"]], cache / "val_pred")
        st.mark("val_predict")
        log("validation scored")
        del va

    # ================================================================ 4 decision
    if not st.done("decision"):
        vp = read_table(cache / "val_pred")
        for c in ("label", "p", "stage1_rank"):
            vp[c] = pd.to_numeric(vp[c])
        s1, cand = vp["s1_id"].to_numpy(dtype=object), vp["cand_id"].to_numpy(dtype=object)
        lab, p = vp["label"].to_numpy(), vp["p"].to_numpy()
        h = half_of(s1)
        n_true = val_split["n_matches"]
        hs = half_of(n_true.index.to_numpy())
        nA, nB = n_true[hs == 0], n_true[hs == 1]
        A, Bm = h == 0, h == 1
        cal_A = fit_calibration(p[A], lab[A])
        pc = apply_calibration(p, cal_A)
        # expected true matches missing from the candidate lists, per S1 (lost before Stage 2)
        mu = float((nA.sum() - lab[A].sum()) / len(nA))
        rules = [{"method": "threshold", "t": float(t), "one_to_one": o}
                 for t in np.round(np.arange(0.20, 0.96, 0.025), 3) for o in (False, True)]
        rules += [{"method": "expected_f", "mu": float(m_), "one_to_one": o}
                  for m_ in (0.0, mu, 2 * mu) for o in (False, True)]
        o2oA, o2oB = one_to_one(s1[A], cand[A], pc[A]), one_to_one(s1[Bm], cand[Bm], pc[Bm])
        res = []
        for r in rules:
            sc = macro_f05(s1[A], lab[A], apply_rule(r, s1[A], cand[A], pc[A], o2oA), nA)
            res.append((sc["macro_f05"], r))
        res.sort(key=lambda x: -x[0])
        best = res[0][1]
        best_thr = max((x for x in res if x[1]["method"] == "threshold"), key=lambda x: x[0])
        best_ef = max((x for x in res if x[1]["method"] == "expected_f"), key=lambda x: x[0])
        scB = macro_f05(s1[Bm], lab[Bm], apply_rule(best, s1[Bm], cand[Bm], pc[Bm], o2oB), nB)
        scB_thr = macro_f05(s1[Bm], lab[Bm], apply_rule(best_thr[1], s1[Bm], cand[Bm], pc[Bm], o2oB), nB)
        scB_ef = macro_f05(s1[Bm], lab[Bm], apply_rule(best_ef[1], s1[Bm], cand[Bm], pc[Bm], o2oB), nB)
        log(f"decision on A: best {best} F0.5 {res[0][0]}")
        log(f"   on B (honest): chosen {scB['macro_f05']} | best threshold {scB_thr['macro_f05']} "
            f"| expected-F {scB_ef['macro_f05']}")
        # refit calibration on ALL validation for test; re-tune the threshold if that rule won
        cal_all = fit_calibration(p, lab)
        final = dict(best)
        if final["method"] == "threshold":
            pca = apply_calibration(p, cal_all)
            o2o_all = one_to_one(s1, cand, pca) if best["one_to_one"] else None
            grid = np.round(np.arange(max(0.05, best["t"] - 0.1), min(0.99, best["t"] + 0.1), 0.01), 3)
            final["t"] = float(max(grid, key=lambda t: macro_f05(
                s1, lab, apply_rule({**best, "t": float(t)}, s1, cand, pca, o2o_all), n_true)["macro_f05"]))
        else:
            final["mu"] = float((n_true.sum() - lab.sum()) / len(n_true)) * (best["mu"] / mu if mu else 0.0)
        report["decision"] = {"rule_tuned_on_A": best, "F05_on_A": res[0][0],
                              "F05_on_B_honest": scB["macro_f05"],
                              "F05_on_B_best_threshold": [best_thr[1], scB_thr["macro_f05"]],
                              "F05_on_B_best_expected_f": [best_ef[1], scB_ef["macro_f05"]],
                              "top_rules_on_A": [[f, r] for f, r in res[:8]], "final_rule_for_test": final}
        atomic_json(out / "decision.json", {"rule": final, "calibration": cal_all})
        save_report(); st.mark("decision")  # noqa: E702

    # ================================================================ 5 evaluate
    if not st.done("evaluate"):
        vp = read_table(cache / "val_pred")
        for c in ("label", "p", "stage1_rank"):
            vp[c] = pd.to_numeric(vp[c])
        dec = json.loads((out / "decision.json").read_text())
        s1, cand = vp["s1_id"].to_numpy(dtype=object), vp["cand_id"].to_numpy(dtype=object)
        lab = vp["label"].to_numpy()
        pc = apply_calibration(vp["p"].to_numpy(), dec["calibration"])
        pred = apply_rule(dec["rule"], s1, cand, pc)
        n_true = val_split["n_matches"]
        full = macro_f05(s1, lab, pred, n_true)
        per_f = pd.Series(full.pop("per_s1_f"), index=n_true.index)
        # ceiling at N (a perfect matcher on these candidates) - must equal Phase 6's number
        found = pd.Series(lab, index=s1).groupby(level=0).sum().reindex(n_true.index, fill_value=0).to_numpy()
        nt = n_true.to_numpy(float)
        r = np.where(nt > 0, found / np.maximum(nt, 1), 1.0)
        ceiling = float(np.where(nt == 0, 1.0, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)).mean())
        seg = {}
        for name, key in (("country", val_split["country_norm"]),
                          ("match_count", val_split["n_matches"].clip(upper=4).map(lambda v: f"{v}{'+' if v == 4 else ''}"))):
            seg[name] = {str(k): round(float(per_f[key == k].mean()), 5) for k in sorted(key.unique())}
        seg["singletons_correct_empty"] = round(float(per_f[n_true == 0].mean()), 5)
        # pair-level views for pair properties
        te = read_table(cache / "val", columns=ID_COLS + ["cross_script", "address_missing_cand"])
        te = te.merge(pd.DataFrame({"s1_id": s1, "cand_id": cand, "pred": pred, "label": lab}), on=ID_COLS)
        for fcol in ("cross_script", "address_missing_cand"):
            te[fcol] = pd.to_numeric(te[fcol])
            for v in (0.0, 1.0):
                sub = te[te[fcol] == v]
                tp = int((sub.pred & (sub.label == 1)).sum())
                seg[f"pairs_{fcol}={int(v)}"] = {
                    "precision": round(tp / max(int(sub.pred.sum()), 1), 5),
                    "recall_within_candidates": round(tp / max(int(sub.label.sum()), 1), 5)}
        report["evaluation"] = {"validation_all": full, "ceiling_at_N": round(ceiling, 5),
                                "gap_to_ceiling": round(ceiling - full["macro_f05"], 5), "segments": seg}
        # error examples
        errs = pd.DataFrame({"s1_id": s1, "cand_id": cand, "p_cal": pc, "pred": pred, "label": lab})
        fp = errs[errs.pred & (errs.label == 0)].sort_values("p_cal", ascending=False).head(300)
        fn = errs[~errs.pred & (errs.label == 1)].sort_values("p_cal").head(300)
        atomic_table(pd.concat([fp.assign(kind="false_positive"), fn.assign(kind="false_negative")]),
                     out / "val_errors")
        save_report(); st.mark("evaluate")  # noqa: E702
        log(f"validation (all {full['s1']:,} S1): macro F0.5 {full['macro_f05']} "
            f"(P {full['macro_precision']}, R {full['macro_recall']}); ceiling at N={N}: {ceiling:.5f}")

    # ================================================================ 6 test_predict
    if not st.done("test_predict"):
        shards = list_shards(Path(a.stage2_dir) / "test")
        tasks = [(b, str(pred_dir / Path(b).name)) for b in shards]
        todo = [t for t in tasks if not table_exists(Path(t[1]))]
        log(f"test: {len(shards)} shards, {len(shards) - len(todo)} already scored, {len(todo)} to do")
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=init_predictor,
                      initargs=([str(b) for b in model_bases], feats)) as pool:
            done_n = 0
            for i, n in enumerate(pool.imap_unordered(predict_shard_task, todo, chunksize=2)):
                done_n += n
                if (i + 1) % max(1, len(todo) // 10) == 0:
                    log(f"   {i + 1}/{len(todo)} shards, {done_n:,} pairs")
        missing = [t for t in tasks if not table_exists(Path(t[1]))]
        if missing:
            raise RuntimeError(f"{len(missing)} test shards were not scored; rerun to resume")
        st.mark("test_predict")

    # ================================================================ 7 submit
    if not st.done("submit"):
        dec = json.loads((out / "decision.json").read_text())
        tp = pd.concat([read_table(Path(b)) for b in list_shards(pred_dir)], ignore_index=True)
        tp["p"] = pd.to_numeric(tp["p"])
        s1, cand = tp["s1_id"].to_numpy(dtype=object), tp["cand_id"].to_numpy(dtype=object)
        pc = apply_calibration(tp["p"].to_numpy(), dec["calibration"])
        pred = apply_rule(dec["rule"], s1, cand, pc)
        test_s1 = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id", "country_norm"])
        sub = Path(a.submit_dir)
        sub.mkdir(parents=True, exist_ok=True)
        cand_lists = tp.groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict()
        match_lists = tp[pred].groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict()
        for fname, header, lists in (("candidate_pairs.tsv", "candidate_entity_ids", cand_lists),
                                     ("matching_results.tsv", "matched_entity_ids", match_lists)):
            tmp = sub / (fname + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(f"source1_entity_id\t{header}\n")
                for sid in test_s1["entity_id"]:
                    f.write(f"{sid}\t{lists.get(sid, '')}\n")
            os.replace(tmp, sub / fname)
        log(f"wrote {sub / 'matching_results.tsv'} and {sub / 'candidate_pairs.tsv'}")
        report["submission_checks"] = check_submission(sub, a.block_dir, test_s1["entity_id"])
        # France check: predicted-match statistics by country vs validation
        pm = pd.Series(pred.astype(int), index=s1).groupby(level=0).sum()
        per = test_s1.set_index("entity_id").assign(n_pred=pm).fillna({"n_pred": 0})
        report["test_by_country"] = {
            c: {"s1": int(len(g)), "mean_pred_matches": round(float(g.n_pred.mean()), 3),
                "share_empty": round(float((g.n_pred == 0).mean()), 4)}
            for c, g in per.groupby("country_norm")}
        vstats = val_split.assign(n=val_split["n_matches"])
        report["val_truth_by_country"] = {
            c: {"mean_true_matches": round(float(g.n.mean()), 3), "share_singleton": round(float((g.n == 0).mean()), 4)}
            for c, g in vstats.groupby("country_norm")}
        ext = Path("utils/validate_submission.py")
        if ext.exists():
            try:
                r = subprocess.run([sys.executable, str(ext), "--matching", str(sub / "matching_results.tsv"),
                                    "--candidate", str(sub / "candidate_pairs.tsv"), "--test-dir", "dataset/test"],
                                   capture_output=True, text=True, timeout=1800)
                report["official_validator"] = {"returncode": r.returncode, "stdout_tail": r.stdout[-2000:]}
                log(f"official validator: return code {r.returncode}\n{r.stdout[-800:]}")
            except Exception as e:  # never lose the submission because the validator failed
                report["official_validator"] = {"error": repr(e)}
        save_report()
        if not report["submission_checks"]["all_ok"]:
            raise RuntimeError(f"submission checks failed: {report['submission_checks']}")
        st.mark("submit")
    write_summary(report, out / "summary.md")
    log(f"done. Summary: {out / 'summary.md'}")


def check_submission(sub: Path, block_dir: str, test_ids: pd.Series) -> dict:
    """Every README rule, checked locally."""
    pool = set()
    for k in (2, 3):
        pool.update(read_table(Path(block_dir) / f"test_source{k}", columns=["entity_id"])["entity_id"])
    need = set(test_ids)
    res = {}
    lists = {}
    for fname, header in (("matching_results.tsv", "matched_entity_ids"),
                          ("candidate_pairs.tsv", "candidate_entity_ids")):
        problems = []
        with open(sub / fname, encoding="utf-8") as f:
            h = f.readline().rstrip("\n").split("\t")
            if h != ["source1_entity_id", header]:
                problems.append(f"bad header {h}")
            seen, d = set(), {}
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    problems.append(f"bad line {line[:80]!r}")
                    continue
                sid, ids = parts
                if sid in seen:
                    problems.append(f"duplicate row {sid}")
                seen.add(sid)
                ids = [x for x in ids.split(",") if x] if ids else []
                if len(ids) != len(set(ids)):
                    problems.append(f"duplicate ids in {sid}")
                bad = [x for x in ids if x not in pool]
                if bad:
                    problems.append(f"{sid}: unknown or non-S2/S3 ids {bad[:3]}")
                d[sid] = set(ids)
        if seen != need:
            problems.append(f"S1 rows mismatch: missing {len(need - seen)}, extra {len(seen - need)}")
        lists[fname] = d
        res[fname] = problems[:20] or "OK"
    not_subset = sum(1 for s, m in lists["matching_results.tsv"].items()
                     if not m <= lists["candidate_pairs.tsv"].get(s, set()))
    res["matches_subset_of_candidates"] = "OK" if not_subset == 0 else f"{not_subset} S1 violate"
    res["all_ok"] = all(v == "OK" for v in res.values())
    return res


def write_summary(r: dict, path: Path):
    L = ["# Phase 8 - Stage-2 matcher\n", f"Stage-1 N: {r.get('N')}"]
    if "data" in r:
        L.append(f"Data: {r['data']}")
    if "model" in r:
        L.append(f"Model: {r['model']['backend']}, {r['model']['n_models']} seeds, rounds {r['model']['rounds']}")
    if "decision" in r:
        d = r["decision"]
        L += ["\n## Decision rule", f"- tuned on half A: {d['rule_tuned_on_A']} -> F0.5 {d['F05_on_A']}",
              f"- **honest estimate on half B: {d['F05_on_B_honest']}**",
              f"- best threshold on B: {d['F05_on_B_best_threshold']}",
              f"- best expected-F0.5 rule on B: {d['F05_on_B_best_expected_f']}",
              f"- final rule used for test: {d['final_rule_for_test']}"]
    if "evaluation" in r:
        e = r["evaluation"]
        L += ["\n## Validation (all validation S1)",
              f"- macro F0.5 {e['validation_all']['macro_f05']} (precision {e['validation_all']['macro_precision']}, "
              f"recall {e['validation_all']['macro_recall']})",
              f"- ceiling at N (perfect matcher on these candidates): {e['ceiling_at_N']}; gap {e['gap_to_ceiling']}",
              "\n### Segments\n```", json.dumps(e["segments"], indent=1), "```"]
    if "submission_checks" in r:
        L += ["\n## Submission checks", "```", json.dumps(r["submission_checks"], indent=1), "```",
              "\n## Test predictions by country (France check)", "```", json.dumps(r.get("test_by_country"), indent=1),
              "```", "Validation truth by country:", "```", json.dumps(r.get("val_truth_by_country"), indent=1), "```"]
    if "official_validator" in r:
        L += ["\n## Official validator", "```", json.dumps(r["official_validator"], indent=1)[:2500], "```"]
    if "model" in r and r["model"].get("top_features"):
        L += ["\n## Top features (gain)"] + [f"- {k}: {v:,.0f}" for k, v in r["model"]["top_features"][:15]]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
