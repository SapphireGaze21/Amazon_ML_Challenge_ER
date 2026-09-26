"""
Unit tests for src/blocking.py.   Run:  python tests/test_blocking.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blocking as B  # noqa: E402


def build(rows_pool, rows_s1, max_df):
    fp, fs = B.feature_chunk(rows_pool), B.feature_chunk(rows_s1)
    M = {}
    for g in B.GROUPS:
        Ac, Bc, Af, Bf, _ = B.weighted(B.csr_from_parts([fs[g][0]], [fs[g][1]]),
                                       B.csr_from_parts([fp[g][0]], [fp[g][1]]), max_df)
        M[g] = {"A_cap": Ac, "BT_cap": Bc.T.tocsr(), "A_full": Af, "B_full": Bf}
    return M


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    # --- features
    f = B.name_features("eastern projects", "astrn prjkts")
    check("name word pair", "nb:eastern|projects" in f, True)
    check("phonetic pair", "pb:astrn|prjkts" in f, True)
    fa = B.address_features("618 pitampura delhi", "618", "8-2-293")
    check("address adjacent pair", "ab:618|pitampura" in fa, True)
    check("number + compound", ("#:618" in fa, "c:8-2-293" in fa), (True, True))
    x = B.cross_features("baba industries", "106 shakarpur delhi")
    check("cross feature: word itself when no phonetic key", "x:baba|shakarpur" in x and "x:baba|106" in x, True)
    check("cross feature uses phonetic key", "x:andstrs|shakarpur" in x, True)
    check("cross feature for keyless short word", "x:om|jaipur" in B.cross_features("om", "jaipur"), True)
    check("char features", B.char_features("ab", "cn:"), ["cn: ab", "cn:ab "])
    check("missing values give no features",
          (B.name_features(None, float("nan")), B.cross_features(None, "x y"), B.char_features(None, "c")),
          ([], [], []))
    check("hash deterministic", B._hash("n:sharma"), B._hash("n:sharma"))

    # --- weighting: capped drops common features, full keeps them
    A = sp.csr_matrix(np.array([[1, 1, 0]], np.float32))
    Bp = sp.csr_matrix(np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], np.float32))
    Ac, Bc, Af, Bf, info = B.weighted(A, Bp, max_df=2)
    check("capped drops df>max_df", Ac.toarray()[0, 0], 0.0)
    check("full keeps it", Af.toarray()[0, 0] > 0, True)

    # --- generic duplicate name: the right city wins (cross features + full re-rank)
    pool = [("fortune shree developers", "frtn sr dvlprs", f"plot {i} city{i} up", str(i), None,
             "fortuneshreedevelopers") for i in range(60)]
    pool.append(("fortune shri devalapars", "frtn sr dvlprs", "10 32 ghaziabad up", "10 32", None,
                 "fortuneshridevalapars"))
    s1 = [("fortune shree developers", "frtn sr dvlprs", "1032 sewa nagar ghaziabad up", "1032", None,
           "fortuneshreedevelopers")]
    M = build(pool, s1, max_df=10)
    B.G.clear()
    er, ec = B.exact_pass(np.array(["developers fortune shree"], dtype=object),
                          np.array(["developers fortune shree"] * 60 + ["devalapars fortune shri"], dtype=object), 5000)
    B.G.update(M=M, exact={"exact_sorted": (er, ec)}, n_pool=len(pool),
               k={"sparse_comb": 5, "sparse_name": 5, "sparse_addr": 5, "char_name": 5, "char_addr": 5},
               w={"name": 1, "addr": 1, "cross": .5, "cname": .5, "caddr": .5, "exact": .25})
    out = B.block_worker((0, 1))
    check("true match found although K=5 and 60 same-name records compete", 60 in out["cand"].tolist(), True)
    found_by = int(out["passes"][out["cand"] == 60][0])
    check("found by the combined pass (cross features)", bool(found_by & B.PASS_BIT["sparse_comb"]), True)
    true_rank = int(out["rank"][out["cand"] == 60][0])
    check("true match stays inside a normal budget", true_rank < 100, True)
    check("ranks restart per S1 and are 0..n-1", sorted(out["rank"].tolist()) == list(range(len(out["rank"]))), True)
    check("one row per pair", len(set(zip(out["s1"], out["cand"]))), len(out["s1"]))

    # --- union + bitmask
    s, c, p = B.union_pairs([("sparse_comb", np.array([0, 0]), np.array([5, 7])),
                             ("exact_sorted", np.array([0]), np.array([7]))], n_pool=10)
    got = dict(zip(zip(s.tolist(), c.tolist()), p.tolist()))
    check("bitmask OR", got[(0, 7)], B.PASS_BIT["sparse_comb"] | B.PASS_BIT["exact_sorted"])

    # --- exact join
    r, c = B.exact_pass(np.array(["a b", None, "x"], dtype=object),
                        np.array(["a b", "a b", "y", None], dtype=object), max_block=5)
    check("exact join", sorted(zip(r.tolist(), c.tolist())), [(0, 0), (0, 1)])
    r, c = B.exact_pass(np.array(["a b"], dtype=object), np.array(["a b", "a b"], dtype=object), max_block=1)
    check("exact join skips big blocks", len(r), 0)

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
