"""
Unit tests for src/phase7_stage2_features.py.   Run:  python tests/test_stage2_features.py
(Needs rapidfuzz; on SageMaker it is installed via requirements.txt.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase7_stage2_features as P  # noqa: E402


def rec(eid, name, addr, nums="", suffix="", alias=None, sorted_key=None, initials="", script="latin",
        missing=False, phon=None):
    return {"entity_id": eid, "name_script": script, "name_roman": name, "name_core": name,
            "name_suffix": suffix, "name_sorted_key": sorted_key or " ".join(sorted(name.split())),
            "name_alias": alias, "name_concat": name.replace(" ", ""), "name_initials": initials,
            "name_phon": phon if phon is not None else name, "address_roman": addr,
            "address_tokens_norm": addr, "address_nums": nums, "address_missing": missing}


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    recs = pd.DataFrame([
        rec("S1-1", "acme tools", "12 main street 60601", "12 60601", "llc"),
        rec("S2-1", "acme tools", "12 main street 60601", "12 60601", "inc"),     # same place
        rec("S2-2", "acme tools", "45 main street 60601", "45 60601", "llc"),     # house conflict
        rec("S3-1", "acme tools", "12 main street 60601", "12 60601", "inc"),     # identical copy of S2-1
        rec("S2-3", "riva", "", "", alias=None, missing=True),                    # no address
        rec("S1-2", "ciramira", "5 hill road", "5", alias="riva"),                 # alias -> S2-3
    ])
    store = P.Store(recs)
    pos, ok = store.lookup(["S2-2", "S9-9", "S1-1"])
    check("lookup found/missing", ok.tolist(), [True, False, True])
    check("lookup position", recs.entity_id.iloc[pos[0]], "S2-2")

    check("empty text scores 0", P.batch_scores(["abc", ""], ["abc", "abc"], P.fuzz.ratio, True).tolist(),
          [1.0, 0.0])
    check("jaccard/overlap empty", P.jac_ov(set(), {"a"}), (0.0, 0.0))
    check("postal = longest 5+ digit number", P.longest_postal({"12", "60601", "7"}), "60601")

    P.G.clear()
    P.G.update(store=store, idf_name={}, idf_addr={}, out_dir="/tmp/p7test")
    Path("/tmp/p7test").mkdir(exist_ok=True)
    pairs = pd.DataFrame({"s1_id": ["S1-1"] * 3 + ["S1-2"], "cand_id": ["S2-1", "S2-2", "S3-1", "S2-3"],
                          "stage1_score": [0.9, 0.5, 0.8, 0.7], "stage1_rank": [0, 2, 1, 0],
                          "passes": 8, "priority": 1.0, "rank": [0, 1, 2, 0], **{c: 0.3 for c in P.COS_COLS}})
    written = {}
    P.read_table = lambda p: pairs
    P.write_table = lambda df, p: written.update(df=df)
    n, _ = P.build_shard("x")
    d = written["df"].set_index("cand_id")
    check("all pairs featurized", n, 4)
    check("house number agree", d.loc["S2-1", "house_num_agree"], 1.0)
    check("house number conflict", d.loc["S2-2", "house_num_conflict"], 1.0)
    check("postal agree despite house conflict", d.loc["S2-2", "postal_agree"], 1.0)
    check("legal suffix conflict (llc vs inc)", d.loc["S2-1", "legal_suffix_conflict"], 1.0)
    check("legal suffix agree (llc vs llc)", d.loc["S2-2", "legal_suffix_agree"], 1.0)
    check("identical copy present", (d.loc["S2-1", "identical_copy_present"], d.loc["S2-2", "identical_copy_present"]),
          (1.0, 0.0))
    check("alias match", d.loc["S2-3", "alias_match"], 1.0)
    check("no alias -> 0 (not a crash on empty strings)", d.loc["S2-1", "alias_match"], 0.0)
    check("address missing flags", (d.loc["S2-3", "address_missing_cand"], d.loc["S2-3", "address_missing_either"]),
          (1.0, 1.0))
    check("empty address similarity is 0, not 1", d.loc["S2-3", "address_norm_ratio"], 0.0)
    check("best candidate compared with itself", d.loc["S2-1", "to_s1_best_addr_ratio"], 1.0)
    check("score gap to S1 best", round(float(d.loc["S2-2", "stage1_score_gap"]), 4), 0.4)
    check("blocking cosines carried through", d.loc["S2-1", "cos_cross"], 0.3)
    check("no NaN", int(written["df"][P.FEATURES].isna().sum().sum()), 0)
    check("no country feature", any("country" in f for f in P.FEATURES), False)

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
