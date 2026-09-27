"""
Unit tests for src/phase8_match.py decision logic.   Run:  python tests/test_phase8.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase8_match as P  # noqa: E402


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    # --- macro F0.5 follows the README rules
    n_true = pd.Series({"A": 2, "B": 0, "C": 0, "D": 1, "E": 3})
    s1 = np.array(["A", "A", "B", "D", "E", "E"])
    lab = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([True, True, False, True, True, True])
    r = P.macro_f05(s1, lab, pred, n_true)
    # A: 2/2 -> 1; B: singleton, empty -> 1; C: singleton, no rows -> 1; D: 1 FP, 0 TP -> 0;
    # E: tp=1, pred=2, true=3 -> 1.25*1/(0.25*3+2)
    exp = (1 + 1 + 1 + 0 + 1.25 / (0.75 + 2)) / 5
    check("macro F0.5 (singletons, misses, FPs)", r["macro_f05"], round(exp, 5))
    check("predicting anything for a singleton scores 0",
          P.macro_f05(np.array(["B"]), np.array([0]), np.array([True]), pd.Series({"B": 0}))["macro_f05"], 0.0)

    # --- PAV / calibration are monotone
    fit = P.pav(np.array([0.1, 0.5, 0.3, 0.9]), np.ones(4))
    check("PAV non-decreasing", bool(np.all(np.diff(fit) >= 0)), True)
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p ** 2).astype(float)            # true prob = p^2 (over-confident scores)
    cal = P.fit_calibration(p, y)
    c = P.apply_calibration(np.array([0.3, 0.9]), cal)
    check("calibration pulls over-confident scores down", bool(c[0] < 0.3 and c[1] < 0.9), True)

    # --- one-to-one keeps only the best S1 per record
    out = P.one_to_one(np.array(["s1", "s2", "s3"]), np.array(["x", "x", "y"]), np.array([0.6, 0.9, 0.4]))
    check("one-to-one", out.tolist(), [0.0, np.float32(0.9), np.float32(0.4)])

    # --- expected-F0.5 decisions
    s = np.array(["weak", "weak", "strong", "strong", "strong", "multi", "multi", "multi"])
    p = np.array([0.15, 0.10, 0.97, 0.05, 0.02, 0.95, 0.90, 0.10])
    d = P.decide_expected_f(s, p, mu=0.0)
    check("weak evidence -> predict empty (singleton-safe)", d[:2].tolist(), [False, False])
    check("strong single match -> only that one", d[2:5].tolist(), [True, False, False])
    check("two strong matches -> both, not the weak third", d[5:].tolist(), [True, True, False])

    # --- threshold rule and the combined rule
    check("threshold", P.decide_threshold(np.array([0.2, 0.7]), 0.5).tolist(), [False, True])
    rule = {"method": "threshold", "t": 0.5, "one_to_one": True}
    check("rule with one-to-one", P.apply_rule(rule, np.array(["a", "b"]), np.array(["x", "x"]),
                                                np.array([0.8, 0.7], dtype=np.float32)).tolist(), [True, False])

    # --- atomic table write leaves no temp file
    tmp = Path("/tmp/p8atomic"); tmp.mkdir(exist_ok=True)
    for f in tmp.iterdir():
        f.unlink()
    P.atomic_table(pd.DataFrame({"a": [1]}), tmp / "t")
    names = sorted(f.name for f in tmp.iterdir())
    check("atomic write: one final file, no temp", (len(names), any("__tmp" in n for n in names)), (1, False))
    check("table_exists", P.table_exists(tmp / "t"), True)

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
