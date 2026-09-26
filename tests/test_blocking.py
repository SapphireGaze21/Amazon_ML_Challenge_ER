"""
Unit tests for src/blocking.py.   Run:  python tests/test_blocking.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blocking as B  # noqa: E402


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    # features
    f = B.name_features("eastern projects", "astrn prjkts")
    check("name word pair present", "nb:eastern|projects" in f, True)
    check("phonetic pair present", "pb:astrn|prjkts" in f, True)
    check("pair is order-free", "nb:eastern|projects" in B.name_features("projects eastern", ""), True)
    fa = B.address_features("618 pitampura delhi", "618", "8-2-293")
    check("address adjacent pair", "ab:618|pitampura" in fa, True)
    check("number feature", "#:618" in fa, True)
    check("compound feature", "c:8-2-293" in fa, True)
    check("missing values give no features", B.name_features(None, float("nan")), [])
    check("hash is deterministic", B._hash("n:sharma"), B._hash("n:sharma"))

    # a generic name whose words are common still matches through its word pair
    rows_pool = [("eastern projects", "astrn prjkts", "no 6 ambattur tn", "6", None)] + \
                [("eastern trading", "astrn trdng", f"x{i} tn", "", None) for i in range(50)] + \
                [("modern projects", "mdrn prjkts", f"y{i} tn", "", None) for i in range(50)]
    rows_s1 = [("eastern projects", "astrn prjkts", "vivekananda nagar 6 tn", "6", None)]
    a, b = B.feature_chunk(rows_pool), B.feature_chunk(rows_s1)
    Bn, An = B.csr_from_parts([a[0]], [a[1]]), B.csr_from_parts([b[0]], [b[1]])
    B.idf_weight(An, Bn, max_df=10)            # "eastern" and "projects" (df ~51) are dropped
    scores = (An @ Bn.T).toarray()[0]
    check("pair feature finds the generic-name match first", int(np.argmax(scores)), 0)

    # top-k
    C = sp.csr_matrix(np.array([[0.1, 0.9, 0.5, 0.0], [0, 0, 0, 0]], dtype=np.float32))
    r, c, s = B.topk_rows(C, 2, 0.0, row_offset=10)
    check("topk rows", sorted(zip(r.tolist(), c.tolist())), [(10, 1), (10, 2)])

    # exact pass
    r, c = B.exact_pass(np.array(["a b", None, "x"], dtype=object),
                        np.array(["a b", "a b", "y", None], dtype=object), max_block=5)
    check("exact join", sorted(zip(r.tolist(), c.tolist())), [(0, 0), (0, 1)])
    r, c = B.exact_pass(np.array(["a b"], dtype=object), np.array(["a b", "a b"], dtype=object), max_block=1)
    check("exact join skips big blocks", len(r), 0)

    # merge: candidate found by 2 passes ranks first; bitmask correct
    parts = [("sparse_comb", np.array([0, 0]), np.array([5, 7]), np.array([0.8, 0.4], np.float32)),
             ("exact_sorted", np.array([0]), np.array([7]), np.array([1.0], np.float32)),
             ("sparse_name", np.array([1]), np.array([5]), np.array([0.3], np.float32))]
    m = B.merge_candidates(parts, n_pool=10)
    top = m[m.s1 == 0].sort_values("rank").iloc[0]
    check("multi-pass candidate ranks first", int(top.cand), 7)
    check("bitmask", int(top.passes), B.PASS_BIT["sparse_comb"] | B.PASS_BIT["exact_sorted"])
    check("ranks restart per S1", m[m.s1 == 1]["rank"].tolist(), [0])
    check("one row per pair", len(m), 3)

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
