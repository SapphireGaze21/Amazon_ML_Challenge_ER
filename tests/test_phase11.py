"""Unit tests for src/phase11_decide.py.   Run: python tests/test_phase11.py"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase11_decide as P  # noqa: E402


def brute_expected_f(p, k):
    """E[F0.5] of predicting the top-k of p (sorted descending), by enumerating every outcome."""
    p = sorted(p, reverse=True)
    e = 0.0
    for outcome in itertools.product((0, 1), repeat=len(p)):
        pr = np.prod([q if o else 1 - q for q, o in zip(p, outcome)])
        nt, tp = sum(outcome), sum(outcome[:k])
        f = float(nt == 0) if k == 0 else 1.25 * tp / (0.25 * nt + k)
        e += pr * f
    return e


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    # --- exact expected F0.5 equals brute-force enumeration
    groups = {"a": [0.9, 0.6, 0.3, 0.05], "b": [0.45], "c": [0.8, 0.8, 0.7, 0.2, 0.1, 0.01]}
    s1 = np.array([g for g, ps in groups.items() for _ in ps], dtype=object)
    p = np.array([x for ps in groups.values() for x in ps])
    d, E = P.expected_f_table(s1, p, mu=0.0, K=6)
    codes = dict(zip(d["s"], d["code"]))
    worst = max(abs(E[codes[g], k] - brute_expected_f(ps, k)) for g, ps in groups.items() for k in range(len(ps) + 1))
    check("exact expected F0.5 == enumeration", worst < 1e-9, True)
    check("k beyond the list is impossible", bool(np.isneginf(E[codes["b"], 2])), True)
    _, E3 = P.expected_f_table(s1, p, mu=0.0, K=3)             # ranks >= 3 folded into a Poisson tail
    check("Poisson tail beyond K stays close", float(abs(E3[codes["c"], 2] - E[codes["c"], 2])) < 0.01, True)

    # --- the exact rule's empty/one boundary is p = 0.5 (the ratio approximation's is ~0.472)
    one = np.array(["x"], dtype=object)
    check("p=0.49 alone -> predict empty", P.decide_exact_f(one, np.array([0.49]), 0.0).tolist(), [False])
    check("p=0.51 alone -> predict it", P.decide_exact_f(one, np.array([0.51]), 0.0).tolist(), [True])
    check("approximate rule predicts p=0.49 (the difference exact_f fixes)",
          P.P8.decide_expected_f(one, np.array([0.49]), 0.0).tolist(), [True])
    check("missing-match mass (mu) makes an empty prediction less attractive",
          P.decide_exact_f(one, np.array([0.45]), 0.5).tolist(), [True])
    s = np.array(["m", "m", "m"], dtype=object)
    check("two strong + one weak -> the two strong, in input order",
          P.decide_exact_f(s, np.array([0.1, 0.95, 0.9]), 0.0).tolist(), [False, True, True])

    # --- soft one-to-one
    q = P.soft_one_to_one(np.array(["s1", "s2", "s3"], dtype=object), np.array(["r", "r", "u"], dtype=object),
                          np.array([0.9, 0.9, 0.7]))
    check("two equal claimants share the record", [round(float(x), 4) for x in q[:2]], [0.4737, 0.4737])
    check("a single claimant is unchanged", round(float(q[2]), 4), 0.7)

    # --- loss decomposition adds up to 1 - F and matches phase8's macro F0.5
    n_true = pd.Series({"A": 2, "B": 0, "D": 1, "E": 3, "F": 2})
    s1 = np.array(["A", "A", "B", "D", "E", "E", "F"], dtype=object)
    lab = np.array([1, 1, 0, 0, 1, 0, 1])
    pred = np.array([True, True, True, True, True, True, False])
    lp = P.loss_parts(s1, lab, pred, n_true)
    check("components add up", bool(np.allclose(lp.loss_block + lp.loss_fn + lp.loss_fp, 1 - lp.F)), True)
    check("same F as phase8", round(float(lp.F.mean()), 5), P.P8.macro_f05(s1, lab, pred, n_true)["macro_f05"])
    check("singleton FP costs the whole S1", float(lp.loc["B", "loss_fp"]), 1.0)
    check("D: true match never reached Stage 2", float(lp.loc["D", "loss_block"]), 1.0)
    check("F: match in the list but not predicted", float(lp.loc["F", "loss_fn"]) > 0, True)

    # --- FP ownership
    own = P.fp_ownership(np.array(["v1", "v1", "v2"], dtype=object), np.array(["r1", "r2", "r3"], dtype=object),
                         np.array([0, 0, 0]), np.array([True, True, True]),
                         owner=pd.Series({"r1": "v9", "r2": "t1"}), split_of=pd.Series({"v9": "val", "t1": "train"}),
                         scored_train={"t1"}, n_true=pd.Series({"v1": 0, "v2": 1}))
    check("FP owners categorised", own["all_fp"],
          {"owner_is_val_s1": [1, 0.3333], "owner_is_scored_train_s1": [1, 0.3333], "no_owner": [1, 0.3333]})
    check("singleton FPs counted separately", sum(v[0] for v in own["fp_on_singleton_s1"].values()), 2)

    # --- decide(): Phase 9 rules are delegated unchanged, new rules run
    cal = {"0": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "1": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
    args = (np.array([0.49, 0.9], np.float32), np.array(["a", "b"], dtype=object),
            np.array(["x", "y"], dtype=object), np.array([0, 0], np.int8))
    check("exact_f via decide()", P.decide({"method": "exact_f", "mu": 0.0, "one_to_one": "soft"}, cal, *args).tolist(),
          [False, True])
    check("phase 9 threshold rule via decide()",
          P.decide({"method": "threshold", "t": 0.5, "t_noaddr": 0.5, "one_to_one": True}, cal, *args).tolist(),
          [False, True])

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
