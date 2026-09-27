"""Unit tests for the Phase 10 token-disagreement features.   Run: python tests/test_phase10.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase10_stack2 as P  # noqa: E402
from phase7_stage2_features import Store  # noqa: E402


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    recs = pd.DataFrame([
        {"entity_id": "s", "name_core": "tsg chits", "address_tokens_norm": "142b main road pune"},
        {"entity_id": "decoy", "name_core": "tsg services", "address_tokens_norm": "145b main road pune"},
        {"entity_id": "moved", "name_core": "tsg chits", "address_tokens_norm": "9 hill street mumbai"},
        {"entity_id": "noaddr", "name_core": "tsg chits", "address_tokens_norm": None},
    ])
    for c in ("name_script", "name_roman", "name_suffix", "name_sorted_key", "name_alias", "name_concat",
              "name_initials", "name_phon", "address_roman", "address_nums"):
        recs[c] = None
    recs["address_missing"] = recs["address_tokens_norm"].isna()
    P.DG["store"] = Store(recs)
    df = pd.DataFrame({"s1_id": ["s"] * 3, "cand_id": ["decoy", "moved", "noaddr"]})
    f = pd.DataFrame(P.diff_chunk(df), columns=P.DIFF, index=df.cand_id)
    check("decoy: address differs ONLY in numbers", f.loc["decoy", "addr_diff_only_numbers"], 1.0)
    check("decoy: same address apart from numbers", f.loc["decoy", "addr_same_except_numbers"], 1.0)
    check("decoy: names differ by a single token swap", f.loc["decoy", "name_single_swap"], 1.0)
    check("moved: address differs in words too", f.loc["moved", "addr_diff_only_numbers"], 0.0)
    check("identical names: nothing only on one side", (f.loc["moved", "name_only_s1"], f.loc["moved", "name_only_cand"]),
          (0.0, 0.0))
    check("missing address -> explicit missing state (NaN), not 0",
          bool(np.isnan(f.loc["noaddr", "addr_diff_frac"])), True)
    check("name comparison still present when the address is missing", f.loc["noaddr", "name_diff_frac"], 0.0)
    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
