#!/usr/bin/env python3
"""
Phase 6 - Stage 1: learned re-ranker over the blocking candidates.

Why: Stage-2 features (fuzzy string comparisons, number conflicts ...) are expensive. Blocking
keeps up to 200 candidates per S1; a cheap model on the scores blocking already stored can
re-rank them so that a much smaller top-N (e.g. 20-30) keeps almost every true match.

Inputs  (from Phase 5):  data_cand/{train_s400000, val_full, test_full}/<country>_<lo>.parquet
         columns: s1_id, cand_id, passes, priority, rank, cos_name, cos_addr, cos_cross,
                  cos_cname, cos_caddr, cos_phon, cos_cphon
Outputs: artifacts/phase6/stage1_model.*            final model (trained on train minus holdout)
         artifacts/phase6/stage1_report.json, summary.md
         data_stage1/val/*      top-max(budgets) per val S1, with stage1_score / stage1_rank
         data_stage1/train/*    top-N per train S1, OUT-OF-FOLD scores (for training Stage 2)
         data_stage1/test/*     top-N per test S1
         artifacts/phase6/candidate_pairs.tsv       test top-N in README format (every test S1)

Method
  1. Features per candidate: priority, blocking rank, 7 cosines, pass bits, and per-S1 context
     (number of candidates, gap/ratio to the S1's best priority, number of strong candidates,
     gap of each cosine to the S1's best value of that cosine). A shard holds complete S1s,
     so per-S1 features are computed correctly inside each worker.
  2. Train sample, built per shard in parallel: all positives, the top --hard-neg negatives
     by blocking rank, and --rand-neg random negatives from the rest WITH weight
     (#rest negatives / #sampled), so the deep-ranked negatives keep their true frequency.
  3. Model: LightGBM binary classifier with row weights (default; MIT license) or lambdarank.
     Early stopping on a HOLDOUT of train S1s (hash-based), never on validation.
  4. Evaluation on ALL validation S1s: n_true comes from s1_split.tsv, so matches lost by
     blocking and S1s without candidates count correctly. The same function applied to
     blocking's own order is the baseline and must reproduce the Phase 5 numbers.
  5. N = smallest budget whose validation oracle F0.5 >= --target (or --top-n).
  6. Train top-N from K-fold (by S1) out-of-fold models, so Stage 2 trains on realistic lists.
  7. Test top-N -> candidate_pairs.tsv (this is exactly what Stage 2 will score).

Parallelism: shards are processed by a 'spawn' process pool (safe with OpenMP libraries such as
LightGBM); workers load the saved model from disk and predict single-threaded.

Usage
  python src/phase6_stage1.py                       # everything (train, val, OOF train, test)
  python src/phase6_stage1.py --skip-oof --skip-test   # quick: train + validation numbers only
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_table, write_table

T0 = time.time()
PASSES = ["exact_sorted", "exact_concat", "exact_alias", "sparse_comb", "sparse_name",
          "sparse_addr", "char_name", "char_addr", "sparse_phon", "noaddr_name",
          "nonlatin_phon", "expand_prf", "sparse_cross"]         # same order as blocking.PASSES
COS_COLS = ["cos_name", "cos_addr", "cos_cross", "cos_cname", "cos_caddr", "cos_phon", "cos_cphon"]
NUM_COLS = ["passes", "priority", "rank"] + COS_COLS
FEATURES = (["priority", "rank", "n_cands", "gap_to_best", "priority_ratio", "n_strong"]
            + COS_COLS + [f"{c}_gap" for c in COS_COLS]
            + [f"pass_{p}" for p in PASSES if p != "expand_prf"])


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ============================================================================ shards / features

def list_shards(d: Path) -> list[str]:
    """Shard base paths (without extension); works for .parquet and the .tsv.gz fallback."""
    out = []
    for p in sorted(Path(d).iterdir()):
        if p.name.endswith(".parquet"):
            out.append(str(p)[:-len(".parquet")])
        elif p.name.endswith(".tsv.gz"):
            out.append(str(p)[:-len(".tsv.gz")])
    return out


def bucket_of(ids: pd.Series) -> np.ndarray:
    """Deterministic 0..99 bucket per S1 (holdout and folds are defined on it)."""
    u = pd.unique(ids)
    b = {x: zlib.crc32(x.encode()) % 100 for x in u}
    return ids.map(b).to_numpy(np.int16)


def load_shard(base: str) -> pd.DataFrame:
    df = read_table(Path(base))
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[COS_COLS] = df[COS_COLS].fillna(0.0)
    df["passes"] = df["passes"].fillna(0).astype(np.int64)
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("s1_id", sort=False)
    df["n_cands"] = g["cand_id"].transform("size").astype(np.float32)
    best = g["priority"].transform("max")
    df["gap_to_best"] = (best - df["priority"]).astype(np.float32)
    df["priority_ratio"] = (df["priority"] / (best + 1e-6)).astype(np.float32)
    df["n_strong"] = (df["priority_ratio"] > 0.8).groupby(df["s1_id"], sort=False).transform("sum").astype(np.float32)
    for c in COS_COLS:
        df[f"{c}_gap"] = (g[c].transform("max") - df[c]).astype(np.float32)
    for i, p in enumerate(PASSES):
        if p != "expand_prf":
            df[f"pass_{p}"] = ((df["passes"].to_numpy() >> i) & 1).astype(np.float32)
    return df


def attach_labels(df: pd.DataFrame, gt: pd.DataFrame | None) -> pd.DataFrame:
    if gt is None:
        df["label"] = np.int8(0)
        return df
    df = df.merge(gt, on=["s1_id", "cand_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    return df


def rank_within(s1: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each row inside its S1 by descending score (rows in any order)."""
    codes = pd.factorize(s1)[0]
    o = np.lexsort((-score, codes))
    cs = codes[o]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
    r = np.empty(len(o), np.int32)
    r[o] = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    return r


# ============================================================================ model wrapper

class Model:
    """LightGBM (default; MIT license) or scikit-learn HistGradientBoosting (for testing only -
    scikit-learn is BSD-3, which does not satisfy the challenge's MIT/Apache rule)."""

    def __init__(self, backend: str, objective: str, params: dict):
        self.backend, self.objective, self.params = backend, objective, params
        self.m, self.best_iter = None, None

    def fit(self, X, y, w, groups, Xv=None, yv=None, wv=None, groups_v=None, rounds=None):
        if self.backend == "lightgbm":
            import lightgbm as lgb
            p = {"objective": self.objective, "learning_rate": self.params["lr"],
                 "num_leaves": self.params["num_leaves"], "min_data_in_leaf": 100,
                 "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
                 "lambda_l2": 1.0, "num_threads": self.params["threads"], "verbose": -1,
                 "seed": self.params["seed"]}
            if self.objective == "lambdarank":
                p.update(metric="ndcg", eval_at=[10, 30])
                w, wv = None, None
            else:
                p["metric"] = "binary_logloss"
            dtr = lgb.Dataset(X, label=y, weight=w, group=groups, free_raw_data=True)
            if rounds is None:
                dv = lgb.Dataset(Xv, label=yv, weight=wv, group=groups_v, reference=dtr)
                self.m = lgb.train(p, dtr, num_boost_round=self.params["max_rounds"], valid_sets=[dv],
                                   callbacks=[lgb.early_stopping(self.params["early_stop"], verbose=False),
                                              lgb.log_evaluation(100)])
                self.best_iter = int(self.m.best_iteration or self.params["max_rounds"])
            else:
                self.m = lgb.train(p, dtr, num_boost_round=rounds)
                self.best_iter = int(rounds)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            it = rounds or self.params["max_rounds"]
            self.m = HistGradientBoostingClassifier(learning_rate=self.params["lr"], max_iter=it,
                                                    max_leaf_nodes=self.params["num_leaves"],
                                                    early_stopping=False, random_state=self.params["seed"])
            self.m.fit(X, y, sample_weight=w)
            self.best_iter = it
        return self.best_iter

    def predict(self, X, threads: int = 1):
        if self.backend == "lightgbm":
            return self.m.predict(X, num_iteration=self.best_iter, num_threads=threads)
        return self.m.predict_proba(X)[:, 1]

    def importance(self) -> dict:
        if self.backend == "lightgbm":
            return dict(zip(FEATURES, map(float, self.m.feature_importance(importance_type="gain"))))
        return {}

    def save(self, path: Path) -> Path:
        if self.backend == "lightgbm":
            p = Path(str(path) + ".txt")
            self.m.save_model(str(p), num_iteration=self.best_iter)
        else:
            p = Path(str(path) + ".pkl")
            p.write_bytes(pickle.dumps(self.m))
        (Path(str(path) + ".meta.json")).write_text(json.dumps(
            {"backend": self.backend, "objective": self.objective, "best_iter": self.best_iter,
             "features": FEATURES}))
        return p

    @staticmethod
    def load(path: Path) -> "Model":
        meta = json.loads(Path(str(path) + ".meta.json").read_text())
        m = Model(meta["backend"], meta["objective"], {})
        m.best_iter = meta["best_iter"]
        if meta["backend"] == "lightgbm":
            import lightgbm as lgb
            m.m = lgb.Booster(model_file=str(path) + ".txt")
        else:
            m.m = pickle.loads(Path(str(path) + ".pkl").read_bytes())
        return m


# ============================================================================ workers

W = {}  # per-worker state (set by the pool initializer)


def init_worker(state: dict):
    W.clear()
    W.update(state)
    if state.get("gt_path"):
        W["gt"] = pd.read_parquet(state["gt_path"]) if state["gt_path"].endswith(".parquet") else \
            pd.read_csv(state["gt_path"], sep="\t", dtype=str, keep_default_na=False).assign(label=np.int8(1))
    else:
        W["gt"] = None
    if state.get("model_paths"):
        W["models"] = [Model.load(Path(p)) for p in state["model_paths"]]


def prep_train_shard(base: str) -> pd.DataFrame:
    """Features + labels + weighted negative sampling for one training shard."""
    df = attach_labels(featurize(load_shard(base)), W["gt"])
    df["bucket"] = bucket_of(df["s1_id"])
    hard_n, rand_n = W["hard_neg"], W["rand_neg"]
    pos = df["label"] == 1
    hard = (~pos) & (df["rank"] < hard_n)
    rest = df[(~pos) & (df["rank"] >= hard_n)]
    rng = np.random.default_rng(W["seed"] + zlib.crc32(base.encode()))
    rest = rest.assign(_r=rng.random(len(rest))).sort_values(["s1_id", "_r"])
    n_rest = rest.groupby("s1_id").size()
    rand = rest.groupby("s1_id").head(rand_n)
    k = rand.groupby("s1_id").size()
    rand = rand.assign(weight=(rand["s1_id"].map(n_rest) / rand["s1_id"].map(k)).astype(np.float32))
    out = pd.concat([df[pos].assign(weight=np.float32(1.0)), df[hard].assign(weight=np.float32(1.0)),
                     rand.drop(columns="_r")], ignore_index=True)
    out = out.sort_values(["s1_id", "rank"], kind="stable")
    cols = ["s1_id", "bucket", "label", "weight", "rank_raw"] + FEATURES
    out["rank_raw"] = out["rank"]
    return out[cols].reset_index(drop=True)


def score_shard(base: str) -> dict:
    """Score one shard, write its top-`keep` rows, return per-S1 recall stats (and TSV lines)."""
    df = attach_labels(featurize(load_shard(base)), W["gt"])
    X = df[FEATURES].to_numpy(np.float32)
    models = W["models"]
    if len(models) == 1:
        pred = models[0].predict(X)
    else:                                            # out-of-fold: model of the S1's fold
        fold = bucket_of(df["s1_id"]) % len(models)
        pred = np.zeros(len(df))
        for k, m in enumerate(models):
            sel = fold == k
            if sel.any():
                pred[sel] = m.predict(X[sel])
    df["stage1_score"] = pred.astype(np.float32)
    df["stage1_rank"] = rank_within(df["s1_id"].to_numpy(), pred)
    res = {}
    if W["gt"] is not None:
        stats = {"s1_id": df["s1_id"].unique()}
        lab = df["label"].to_numpy()
        for b in W["budgets"]:
            for name, r in (("model", df["stage1_rank"].to_numpy()), ("base", df["rank"].to_numpy())):
                hit = pd.Series(lab * (r < b), index=df["s1_id"]).groupby(level=0, sort=False).sum()
                stats[f"{name}_{b}"] = hit.reindex(stats["s1_id"]).to_numpy()
        res["stats"] = pd.DataFrame(stats)
    keep = df[df["stage1_rank"] < W["keep"]]
    out_cols = ["s1_id", "cand_id", "stage1_score", "stage1_rank", "passes", "priority", "rank"] + COS_COLS
    write_table(keep[out_cols].reset_index(drop=True), Path(W["out_dir"]) / Path(base).name)
    if W.get("tsv"):
        k2 = keep.sort_values(["s1_id", "stage1_rank"])
        res["tsv"] = k2.groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict()
    return res


def run_pool(fn, shards, state, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=init_worker, initargs=(state,)) as p:
        return list(p.imap_unordered(fn, shards))


# ============================================================================ evaluation

def oracle_table(stats: pd.DataFrame, n_true: pd.Series, budgets: list) -> dict:
    """Oracle macro F0.5 over ALL S1 in n_true (index = S1 ids, value = true match count;
    0 = singleton). S1s absent from `stats` (no candidates) count as found = 0."""
    st = stats.set_index("s1_id").reindex(n_true.index, fill_value=0)
    has = (n_true > 0).to_numpy()
    nt = n_true.to_numpy().astype(float)
    out = {}
    for name in ("model", "base"):
        rows = {}
        for b in budgets:
            found = st[f"{name}_{b}"].to_numpy().astype(float)
            r = found[has] / nt[has]
            f = np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)
            rows[str(b)] = {"oracle_macro_f05": round(float((f.sum() + (~has).sum()) / len(nt)), 5),
                            "pair_recall": round(float(found.sum() / nt.sum()), 5),
                            "s1_all_matches_found": round(float((r >= 1).mean()), 5)}
        out[name] = rows
    return out


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cand", default="data_cand/train_s400000")
    ap.add_argument("--val-cand", default="data_cand/val_full")
    ap.add_argument("--test-cand", default="data_cand/test_full")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--block-dir", default="data_block", help="for the list of ALL test S1 ids")
    ap.add_argument("--out-dir", default="artifacts/phase6")
    ap.add_argument("--stage1-dir", default="data_stage1")
    ap.add_argument("--backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    ap.add_argument("--objective", choices=["binary", "lambdarank"], default="binary")
    ap.add_argument("--hard-neg", type=int, default=15)
    ap.add_argument("--rand-neg", type=int, default=30)
    ap.add_argument("--holdout-pct", type=int, default=10, help="%% of train S1s for early stopping")
    ap.add_argument("--folds", type=int, default=3, help="folds for out-of-fold train top-N")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--max-rounds", type=int, default=1000)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--budgets", default="5,10,15,20,30,50,75,100")
    ap.add_argument("--target", type=float, default=0.995)
    ap.add_argument("--top-n", type=int, default=0, help="force N (0 = choose from validation)")
    ap.add_argument("--skip-oof", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out_dir, st_dir = Path(a.out_dir), Path(a.stage1_dir)
    tmp = out_dir / "tmp"
    for d in (out_dir, tmp):
        d.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in a.budgets.split(",")]
    report = {"args": vars(a), "features": FEATURES}

    # ---- ground truth, per split (written once so each worker can load its own copy)
    split = pd.read_csv(Path(a.phase0_dir) / "s1_split.tsv", sep="\t", dtype={"s1_id": str, "split": str})
    split["n_matches"] = split["n_matches"].astype(int)
    gt = pd.read_csv(Path(a.phase0_dir) / "gt_long.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt.rename(columns={"matched_id": "cand_id"})[["s1_id", "cand_id"]]
    gt_paths = {}
    for sp in ("train", "val"):
        ids = set(split.loc[split["split"] == sp, "s1_id"])
        p = tmp / f"gt_{sp}.tsv"
        gt[gt.s1_id.isin(ids)].to_csv(p, sep="\t", index=False)
        gt_paths[sp] = str(p)
    n_true_val = split.loc[split["split"] == "val"].set_index("s1_id")["n_matches"]
    del gt
    base_state = {"hard_neg": a.hard_neg, "rand_neg": a.rand_neg, "seed": a.seed, "budgets": budgets}

    # ---- 1. training sample (parallel, per shard)
    tr_shards = list_shards(Path(a.train_cand))
    log(f"train: {len(tr_shards)} shards -> features + weighted negative sampling")
    parts = run_pool(prep_train_shard, tr_shards, {**base_state, "gt_path": gt_paths["train"]}, a.workers)
    T = pd.concat(parts, ignore_index=True)
    del parts
    T = T.sort_values(["s1_id", "rank_raw"], kind="stable").reset_index(drop=True)
    log(f"train sample: {len(T):,} rows, {int(T.label.sum()):,} positives, "
        f"{T.s1_id.nunique():,} S1; sum of weights {T.weight.sum():,.0f}")
    report["train_sample"] = {"rows": int(len(T)), "positives": int(T.label.sum()),
                              "s1": int(T.s1_id.nunique())}

    def groups_of(df):
        return df.groupby("s1_id", sort=False).size().to_numpy()

    params = {"lr": a.lr, "num_leaves": a.num_leaves, "max_rounds": a.max_rounds,
              "early_stop": a.early_stop, "threads": a.workers, "seed": a.seed}
    hold = (T["bucket"] < a.holdout_pct).to_numpy()
    Xtr, Xho = T.loc[~hold, FEATURES].to_numpy(np.float32), T.loc[hold, FEATURES].to_numpy(np.float32)

    # ---- 2. final model: train on train minus holdout, early stopping on the holdout
    model = Model(a.backend, a.objective, params)
    best = model.fit(Xtr, T.label[~hold].to_numpy(), T.weight[~hold].to_numpy(), groups_of(T[~hold]),
                     Xho, T.label[hold].to_numpy(), T.weight[hold].to_numpy(), groups_of(T[hold]))
    model_base = out_dir / "stage1_model"
    model.save(model_base)
    report["model"] = {"backend": a.backend, "objective": a.objective, "best_iteration": best,
                       "importance_gain": model.importance()}
    log(f"final model trained: {best} rounds -> {model_base}")
    del Xtr, Xho

    # ---- 3. validation: score all shards, evaluate against ALL validation S1s
    va_shards = list_shards(Path(a.val_cand))
    state = {**base_state, "gt_path": gt_paths["val"], "model_paths": [str(model_base)],
             "keep": max(budgets), "out_dir": str(st_dir / "val")}
    (st_dir / "val").mkdir(parents=True, exist_ok=True)
    res = run_pool(score_shard, va_shards, state, a.workers)
    stats = pd.concat([r["stats"] for r in res if "stats" in r], ignore_index=True)
    table = oracle_table(stats, n_true_val, budgets)
    report["validation"] = table
    log("validation oracle F0.5 (all validation S1)   stage1 | blocking order")
    for b in budgets:
        m, bb = table["model"][str(b)], table["base"][str(b)]
        log(f"   top {b:>3}: {m['oracle_macro_f05']:.5f} | {bb['oracle_macro_f05']:.5f}   "
            f"(pair recall {m['pair_recall']:.4f} | {bb['pair_recall']:.4f})")

    if a.top_n:
        N = a.top_n
    else:
        ok = [b for b in budgets if table["model"][str(b)]["oracle_macro_f05"] >= a.target]
        N = ok[0] if ok else max(budgets)
        if not ok:
            log(f"WARNING: target {a.target} not reached at any budget; using N = {N}")
    report["chosen_N"] = N
    log(f"chosen N = {N}")

    # ---- 4. out-of-fold scores for train top-N (Stage 2 must train on realistic lists)
    if not a.skip_oof:
        fold_paths = []
        fold_of = T["bucket"].to_numpy() % a.folds
        for k in range(a.folds):
            m = Model(a.backend, a.objective, params)
            sel = fold_of != k                      # train on the other folds, fixed rounds
            m.fit(T.loc[sel, FEATURES].to_numpy(np.float32), T.label[sel].to_numpy(),
                  T.weight[sel].to_numpy(), groups_of(T[sel]), rounds=best)
            m.save(out_dir / f"stage1_fold{k}")
            fold_paths.append(str(out_dir / f"stage1_fold{k}"))
            log(f"fold model {k + 1}/{a.folds} trained")
        (st_dir / "train").mkdir(parents=True, exist_ok=True)
        state = {**base_state, "gt_path": gt_paths["train"], "model_paths": fold_paths,
                 "keep": N, "out_dir": str(st_dir / "train")}
        res = run_pool(score_shard, tr_shards, state, a.workers)
        st_tr = pd.concat([r["stats"] for r in res if "stats" in r], ignore_index=True)
        # the exact list of sampled train S1s is not stored, so this uses the S1s that have
        # candidates (S1s with none are excluded) - an approximation, for sanity only
        nt = split.set_index("s1_id")["n_matches"].reindex(st_tr["s1_id"])
        report["train_oof_approx"] = oracle_table(st_tr, nt, [N])["model"]
        log(f"train out-of-fold top-{N} written; approx oracle {report['train_oof_approx']}")
    del T

    # ---- 5. test top-N and candidate_pairs.tsv (every test S1, README format)
    if not a.skip_test and Path(a.test_cand).exists():
        (st_dir / "test").mkdir(parents=True, exist_ok=True)
        state = {**base_state, "gt_path": None, "model_paths": [str(model_base)], "keep": N,
                 "out_dir": str(st_dir / "test"), "tsv": True}
        res = run_pool(score_shard, list_shards(Path(a.test_cand)), state, a.workers)
        lists = {}
        for r in res:
            lists.update(r.get("tsv", {}))
        test_ids = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id"])["entity_id"]
        with open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for sid in test_ids:
                f.write(f"{sid}\t{lists.get(sid, '')}\n")
        report["test"] = {"s1": int(len(test_ids)), "s1_with_candidates": int(len(lists)), "N": N}
        log(f"wrote {out_dir / 'candidate_pairs.tsv'} ({len(test_ids):,} test S1, "
            f"{len(lists):,} with candidates)")

    with open(out_dir / "stage1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    write_summary(report, out_dir / "summary.md", budgets)
    log(f"done -> {out_dir}")


def write_summary(r, path, budgets):
    L = ["# Stage 1 re-ranker\n",
         f"Train sample: {r['train_sample']}", f"Model: {r['model']['backend']} / "
         f"{r['model']['objective']}, {r['model']['best_iteration']} rounds",
         "\n## Validation oracle F0.5 (ALL validation S1; blocking losses included)\n",
         "| top-N | stage 1 | blocking order | stage 1 pair recall | blocking pair recall |",
         "|---|---|---|---|---|"]
    for b in budgets:
        m, bb = r["validation"]["model"][str(b)], r["validation"]["base"][str(b)]
        L.append(f"| {b} | {m['oracle_macro_f05']} | {bb['oracle_macro_f05']} | "
                 f"{m['pair_recall']} | {bb['pair_recall']} |")
    L.append(f"\n**Chosen N = {r['chosen_N']}**")
    if "train_oof_approx" in r:
        L.append(f"\nTrain out-of-fold (approx.): {r['train_oof_approx']}")
    if "test" in r:
        L.append(f"\nTest: {r['test']}")
    imp = r["model"].get("importance_gain") or {}
    if imp:
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:15]
        L.append("\n## Top features by gain\n" + "\n".join(f"- {k}: {v:,.0f}" for k, v in top))
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
