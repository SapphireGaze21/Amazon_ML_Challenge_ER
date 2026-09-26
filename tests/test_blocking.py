"""
Unit tests for src/blocking.py.   Run:  python tests/test_blocking.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blocking as B  # noqa: E402


def build(rows_pool, rows_s1, max_df, addr_missing=None, info=None):
    """Set up blocking.G exactly like the driver does."""
    fp, fs = B.feature_chunk(rows_pool), B.feature_chunk(rows_s1)
    M, Bcap = {}, {}
    for g in B.GROUPS:
        Ac, Bc, Af, Bf, _, a_norm = B.weighted(B.csr_from_parts([fs[g][0]], [fs[g][1]]),
                                               B.csr_from_parts([fp[g][0]], [fp[g][1]]), max_df)
        M[g] = {"A_cap": Ac, "BT_cap": Bc.T.tocsr(), "A_full": Af, "B_full": Bf}
        Bcap[g] = Bc
    am = np.zeros(len(rows_pool), bool) if addr_missing is None else np.asarray(addr_missing)
    noaddr, nonlatin = np.flatnonzero(am), np.zeros(0, np.int64)
    sub = {"noaddr": {"idx": noaddr, **{f"BT_{g}": Bcap[g][noaddr].T.tocsr() for g in ("name", "phon", "cname")}},
           "nonlatin": {"idx": nonlatin, **{f"BT_{g}": Bcap[g][nonlatin].T.tocsr() for g in ("phon", "cphon", "addr")}}}
    B.G.clear()
    B.G.update(M=M, exact={}, n_pool=len(rows_pool), sub=sub, pool_addr_missing=am,
               name_info=np.ones(len(rows_s1), np.float32) if info is None else np.asarray(info, np.float32),
               prf={"seeds": 3, "min_priority": 1.5, "alpha": 1.0},
               k={p: 5 for p in B.PASSES},
               w={"name": 1, "addr": 1, "cross": .5, "cname": .5, "caddr": .5, "exact": .25,
                  "missing_addr_factor": .7})
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
    Ac, Bc, Af, Bf, info, a_norm = B.weighted(A, Bp, max_df=2)
    check("capped drops df>max_df", Ac.toarray()[0, 0], 0.0)
    check("full keeps it", Af.toarray()[0, 0] > 0, True)

    # --- generic duplicate name: the right city wins (cross features + full re-rank)
    pool = [("fortune shree developers", "frtn sr dvlprs", f"plot {i} city{i} up", str(i), None,
             "fortuneshreedevelopers") for i in range(60)]
    pool.append(("fortune shri devalapars", "frtn sr dvlprs", "10 32 ghaziabad up", "10 32", None,
                 "fortuneshridevalapars"))
    s1 = [("fortune shree developers", "frtn sr dvlprs", "1032 sewa nagar ghaziabad up", "1032", None,
           "fortuneshreedevelopers")]
    build(pool, s1, max_df=10)
    er, ec = B.exact_pass(np.array(["developers fortune shree"], dtype=object),
                          np.array(["developers fortune shree"] * 60 + ["devalapars fortune shri"], dtype=object), 5000)
    B.G["exact"] = {"exact_sorted": (er, ec)}
    out = B.block_worker((0, 1))
    check("true match found although K=5 and 60 same-name records compete", 60 in out["cand"].tolist(), True)
    found_by = int(out["passes"][out["cand"] == 60][0])
    check("found by the combined pass (cross features)", bool(found_by & B.PASS_BIT["sparse_comb"]), True)
    true_rank = int(out["rank"][out["cand"] == 60][0])
    check("true match stays inside a normal budget", true_rank < 100, True)
    check("ranks restart per S1 and are 0..n-1", sorted(out["rank"].tolist()) == list(range(len(out["rank"]))), True)
    check("one row per pair", len(set(zip(out["s1"], out["cand"]))), len(out["s1"]))

    # --- coarse phonetics (v2 misses: haitek/hitech, injiniyarin/engineering, sebhen/seven)
    check("coarse c/k", B.coarse_key("htc"), B.coarse_key("htk"))
    check("coarse -ng and j/g", B.coarse_key("angnrng"), B.coarse_key("anjnrn"))
    check("coarse b/v", B.coarse_key("sbn"), B.coarse_key("svn"))
    check("squashed .com stripped for char name",
          B.feature_chunk([(None, None, None, None, None, "superagrocom")])["cname"][1].tolist(),
          B.feature_chunk([(None, None, None, None, None, "superagro")])["cname"][1].tolist())

    # --- informativeness: identical address beats 40 exact copies of a common one-word name
    pool = [("ventures", "vntrs", f"{i} other road city{i} ap", str(i), None, "ventures") for i in range(40)]
    pool.append(("venchars", "vncrs", "y mohana rao 14 2 46 srinivasa nagar palasa ap", "14 2 46", None, "venchars"))
    s1 = [("ventures", "vntrs", "y mohana rao 14 2 46 srinivasa nagar palasa ap", "14 2 46", None, "ventures")]
    build(pool, s1, max_df=100, info=[0.3])
    out = B.block_worker((0, 1))
    check("identical address ranks first for an uninformative name",
          int(out["cand"][out["rank"] == 0][0]), 40)

    # --- address-missing slice: typo'd name, no address, 30 addressed look-alikes
    pool = [("trinity chapel", "trnt cpl", f"{i} main street town{i}", str(i), None, "trinitychapel") for i in range(30)]
    pool.append(("trinity chepel", "trnt cpl", None, None, None, "trinitychepel"))
    s1 = [("trinity chapel", "trnt cpl", "21 williams road berlin vermont", "21", None, "trinitychapel")]
    build(pool, s1, max_df=100, addr_missing=[False] * 30 + [True])
    B.G["k"] = {p: (2 if p != "noaddr_name" else 5) for p in B.PASSES}
    out = B.block_worker((0, 1))
    check("address-missing match found by its slice pass",
          bool(out["passes"][out["cand"] == 30][0] & B.PASS_BIT["noaddr_name"]) if 30 in out["cand"] else False, True)

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
