#!/usr/bin/env python3
"""
Re-decide the Phase 10 test submission with other gammas, from the CACHED test scores (no model,
no feature recomputation - a couple of minutes). Same candidates, same scores, same rule; only
calibrated p -> p ** gamma before the decision, optionally a different gamma per country.

  python src/regamma.py --gammas 1.35 1.6 1.8                       # uniform gammas
  python src/regamma.py --gammas 1.6 --country-gamma france=1.9     # France stricter than the rest

Writes output_v3/variants/<name>/{matching_results.tsv, candidate_pairs.tsv} and prints predicted
matches per S1 by country for each variant (gamma 1.0 and 1.35 reproduce the Phase 10 files).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import phase8_match as P8
import phase9_improve as P9
from common import read_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase10-dir", default="artifacts/phase10")
    ap.add_argument("--stage2-dir", default="data_stage2")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--submit-dir", default="output_v3")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 1.35, 1.6, 1.8])
    ap.add_argument("--country-gamma", nargs="*", default=[], help="country=gamma, overrides --gammas there")
    a = ap.parse_args()

    p10 = Path(a.phase10_dir)
    dec = json.loads((p10 / "decision.json").read_text())
    ts = read_table(p10 / "cache" / "test_scores")
    grp = pd.concat([read_table(Path(b), columns=["s1_id", "cand_id", "address_missing_cand"])
                     for b in P8.list_shards(Path(a.stage2_dir) / "test")], ignore_index=True)
    ts = ts.merge(grp, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    s1, cand = ts["s1_id"].to_numpy(dtype=object), ts["cand_id"].to_numpy(dtype=object)
    p = pd.to_numeric(ts["p"]).to_numpy(np.float32)
    g = pd.to_numeric(ts["address_missing_cand"]).fillna(0).to_numpy().astype(np.int8)
    test_s1 = read_table(Path(a.block_dir) / "test_source1", columns=["entity_id", "country_norm"])
    country_of = test_s1.set_index("entity_id")["country_norm"]
    row_country = pd.Series(s1).map(country_of).to_numpy(dtype=object)
    over = {k: float(v) for k, v in (x.split("=") for x in a.country_gamma)}
    sub = Path(a.submit_dir)
    tp = pd.DataFrame({"s1_id": s1, "cand_id": cand})
    for gm in a.gammas:
        gamma = np.full(len(p), gm, np.float32)
        for c, v in over.items():
            gamma[row_country == c] = v
        pred = P9.apply_decision(dec["rule"], dec["calibration"], p, s1, cand, g, gamma)
        name = f"gamma{gm}" + "".join(f"_{c}{v}" for c, v in over.items())
        d = sub / "variants" / name
        d.mkdir(parents=True, exist_ok=True)
        P9.write_lists(d / "matching_results.tsv", "matched_entity_ids",
                       tp[pred].groupby("s1_id", sort=False)["cand_id"].apply(",".join).to_dict(), test_s1["entity_id"])
        shutil.copyfile(sub / "candidate_pairs.tsv", d / "candidate_pairs.tsv")
        pm = pd.Series(pred.astype(int), index=s1).groupby(level=0).sum().reindex(test_s1["entity_id"], fill_value=0)
        by_c = pm.groupby(country_of.reindex(pm.index).to_numpy())
        checks = P8.check_submission(d, a.block_dir, test_s1["entity_id"])
        print(f"{name:<28} pairs {int(pred.sum()):>9,} | matches/S1 "
              f"{ {c: round(float(x.mean()), 3) for c, x in by_c} } | empty share "
              f"{ {c: round(float((x == 0).mean()), 4) for c, x in by_c} } | checks {'OK' if checks['all_ok'] else checks}")


if __name__ == "__main__":
    main()
