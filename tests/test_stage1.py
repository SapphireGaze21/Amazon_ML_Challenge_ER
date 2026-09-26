"""
Unit tests for src/phase6_stage1.py.   Run:  python tests/test_stage1.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase6_stage1 as S  # noqa: E402


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    # --- oracle: the three cases the first draft got wrong
    # A: 2 true matches, both found        -> 1.0
    # B: 1 true match lost by blocking      -> 0.0   (draft scored it 1.0, as a "singleton")
    # C: true singleton, has candidates     -> 1.0
    # D: 2 true matches, NO candidates      -> 0.0   (draft dropped it from the average)
    # E: true singleton, no candidates      -> 1.0
    stats = pd.DataFrame({"s1_id": ["A", "B", "C"], "model_10": [2, 0, 0], "base_10": [1, 0, 0]})
    n_true = pd.Series({"A": 2, "B": 1, "C": 0, "D": 2, "E": 0})
    t = S.oracle_table(stats, n_true, [10])
    check("oracle counts lost matches and candidate-less S1s", t["model"]["10"]["oracle_macro_f05"],
          round((1 + 0 + 1 + 0 + 1) / 5, 5))
    f_half = 1.25 * 0.5 / (0.25 + 0.5)
    check("baseline half recall", t["base"]["10"]["oracle_macro_f05"], round((f_half + 0 + 1 + 0 + 1) / 5, 5))
    check("pair recall uses ground-truth totals", t["model"]["10"]["pair_recall"], round(2 / 5, 5))

    # --- rank within S1
    r = S.rank_within(np.array(["x", "y", "x", "x", "y"]), np.array([0.1, 0.9, 0.8, 0.5, 0.2]))
    check("rank_within", r.tolist(), [2, 0, 0, 1, 1])

    # --- features: per-S1 context
    df = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["1", "2", "3"], "passes": [1 | 8, 8, 2],
                       "priority": [2.0, 1.0, 3.0], "rank": [0, 1, 0],
                       **{c: [0.5, 0.2, 0.9] for c in S.COS_COLS}})
    f = S.featurize(df)
    check("n_cands", f["n_cands"].tolist(), [2.0, 2.0, 1.0])
    check("gap_to_best", f["gap_to_best"].tolist(), [0.0, 1.0, 0.0])
    check("cosine gap", [round(x, 3) for x in f["cos_name_gap"].tolist()], [0.0, 0.3, 0.0])
    check("pass bits", (f["pass_exact_sorted"].tolist(), f["pass_sparse_comb"].tolist()),
          ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]))
    check("feature list excludes the always-off PRF pass", "pass_expand_prf" in S.FEATURES, False)

    # --- labels by merge
    gt = pd.DataFrame({"s1_id": ["a"], "cand_id": ["2"], "label": [np.int8(1)]})
    check("labels", S.attach_labels(df.copy(), gt)["label"].tolist(), [0, 1, 0])

    # --- weighted negative sampling keeps the true negative mass
    S.W.clear()
    S.W.update(hard_neg=2, rand_neg=3, seed=1,
               gt=pd.DataFrame({"s1_id": ["q"], "cand_id": ["c7"], "label": [np.int8(1)]}))
    shard = pd.DataFrame({"s1_id": ["q"] * 20, "cand_id": [f"c{i}" for i in range(20)],
                          "passes": 8, "priority": np.linspace(3, 1, 20), "rank": range(20),
                          **{c: 0.5 for c in S.COS_COLS}})
    S.load_shard = lambda base: shard.copy()            # bypass disk for the test
    out = S.prep_train_shard("dummy")
    neg = out[out.label == 0]
    check("all positives kept", int(out.label.sum()), 1)
    check("hard negatives kept", int((neg.rank_raw < 2).sum()), 2)
    check("random negatives sampled", int((neg.rank_raw >= 2).sum()), 3)
    check("weights restore the full negative count (17 rest negatives)",
          round(float(neg.loc[neg.rank_raw >= 2, "weight"].sum()), 4), 17.0)

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
