"""
Unit tests for src/phase9_improve.py.   Run:  python tests/test_phase9.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase9_improve as P  # noqa: E402
from phase7_stage2_features import Store  # noqa: E402


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    # --- cross-candidate features: a weird name at the same address as two confident matches
    recs = pd.DataFrame([
        {"entity_id": "c1", "name_core": "acme tools", "address_tokens_norm": "12 main street 60601", "address_nums": "12 60601"},
        {"entity_id": "c2", "name_core": "acme tool", "address_tokens_norm": "12 main street 60601", "address_nums": "12 60601"},
        {"entity_id": "c3", "name_core": "lyraonyxevo", "address_tokens_norm": "12 main street 60601", "address_nums": "12 60601"},
        {"entity_id": "c4", "name_core": "zeta bakery", "address_tokens_norm": "9 oak avenue 10001", "address_nums": "9 10001"},
    ])
    for c in ("name_script", "name_roman", "name_suffix", "name_sorted_key", "name_alias", "name_concat",
              "name_initials", "address_roman"):
        recs[c] = None
    recs["name_phon"] = recs["name_core"]
    recs["address_missing"] = False
    P.G["store"] = Store(recs)
    df = pd.DataFrame({"s1_id": ["s"] * 4, "cand_id": ["c1", "c2", "c3", "c4"], "p1": [0.95, 0.9, 0.2, 0.1]})
    f = P.cross_chunk(df).set_index("cand_id")
    check("weird name, same address as confident matches -> high sibling address evidence",
          round(float(f.loc["c3", "sib_addr_same_max"]), 3), 0.95)
    check("... shares their house number", round(float(f.loc["c3", "sib_num_max"]), 3), 0.95)
    check("... and their postal code", round(float(f.loc["c3", "sib_postal_max"]), 3), 0.95)
    check("different address -> no sibling address evidence", float(f.loc["c4", "sib_addr_same_max"]), 0.0)
    check("confident others counted", (int(f.loc["c3", "n_conf_other"]), int(f.loc["c1", "n_conf_other"])), (2, 1))
    check("own rank within the S1", [int(f.loc[c, "p1_rank"]) for c in ("c1", "c2", "c3", "c4")], [0, 1, 2, 3])
    check("identical addresses counted", int(f.loc["c3", "n_same_addr"]), 2)

    # --- cross_features keeps the original row order across chunks
    df2 = pd.concat([df.assign(s1_id="t"), df], ignore_index=True)
    res = P.cross_features(df2, P.G["store"], workers=1, chunk_s1=1)
    check("row order preserved", [round(float(x), 2) for x in res["p1"]], [0.95, 0.9, 0.2, 0.1] * 2)

    # --- relative Stage-1 features
    r = P.add_relative_stage1(pd.DataFrame({"s1_id": ["a", "a", "b"], "stage1_score": [0.8, 0.4, 0.5]}))
    check("ratio to S1 best", [round(float(x), 3) for x in r["s1rel_ratio"]], [1.0, 0.5, 1.0])
    check("feature sets", (("stage1_score" in P.feature_set("no_stage1", ["stage1_score", "x"])),
                           ("s1rel_ratio" in P.feature_set("rel_stage1", ["stage1_score", "x"]))), (False, True))

    # --- per-group decision: no-address candidates get their own threshold
    rule = {"method": "threshold", "t": 0.7, "t_noaddr": 0.3, "one_to_one": False}
    cal = {"0": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "1": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
    pred = P.apply_decision(rule, cal, np.array([0.5, 0.5], np.float32), np.array(["a", "b"], dtype=object),
                            np.array(["x", "y"], dtype=object), np.array([0, 1], np.int8))
    check("same probability, different group -> different decision", pred.tolist(), [False, True])

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
