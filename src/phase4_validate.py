#!/usr/bin/env python3
"""
Phase 4 - Validate the processing and produce the blocking inputs.

  A. INTEGRITY (all six sources, FAIL gates): row counts equal Phase 0, unique IDs with the
     right prefix, missing flags consistent with the cleaned/represented columns, no leftover
     non-Latin script, no empty core names.
  B. CONFIG (FAIL/WARN gates): abbreviation maps have no cycles or chains; every discovered
     legal suffix is abbreviation-length or a variant of a known legal word.
  C. SIGNAL COVERAGE on TRAIN-split true pairs vs random same-country pairs: for each blocking
     signal we plan to use (exact keys, rare core/phonetic/address tokens, numbers), the share
     of true pairs that carry it, by country / source / script / missing address, plus the
     entity-level view and example pairs with no usable signal. Validation S1s are NOT used.
  D. UNLABELED SANITY per country on the test pool (France has no labels): exact-key hit rates
     and rare-token availability should look similar to US/India.
  E. BLOCK-SIZE PREVIEW: how big exact-key blocks and rarest-token blocks are per S1.
  F. BLOCKING INPUTS: slim per-source tables in --block-dir (only the columns blocking needs;
     train_source1 also gets the train/val split column) + manifest.json.

Outputs: {out-dir}/phase4_report.json, phase4_summary.md, no_signal_true_pairs.tsv,
         no_rare_signal_true_pairs.tsv;  {block-dir}/{split}_source{k}.parquet, manifest.json
Exit code 1 if any FAIL gate trips.

Usage:  python src/phase4_validate.py
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import nn, read_table, write_table
from represent import LEGAL_SUFFIX_HAND, suffix_like
from romanize import has_unsupported_script

T0 = time.time()
SOURCES = [f"{s}_source{k}" for s in ("train", "test") for k in (1, 2, 3)]
BLOCK_COLS = ["entity_id", "country_norm", "name_missing", "address_missing", "name_script",
              "name_core", "name_sorted_key", "name_alias", "name_concat", "name_initials",
              "name_phon", "address_tokens_norm", "address_nums", "address_numcompounds"]
CAPS = (1000, 10000)
GATES = []  # (level, check, detail)


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def gate(level, check, ok, detail=""):
    GATES.append({"level": "PASS" if ok else level, "check": check, "detail": detail})
    if not ok:
        log(f"{level}: {check} {detail}")


def toks(v):
    v = nn(v)
    return set(v.split(" ")) if v else set()


# ============================================================================ A. integrity

def integrity(repr_dir: Path, phase0_report: dict) -> dict:
    out = {}
    cols = ["entity_id", "name_missing", "address_missing", "name_fold", "address_fold",
            "name_core", "name_roman", "address_roman", "address_tokens_norm"]
    for name in SOURCES:
        df = read_table(repr_dir / name, columns=cols)
        exp = phase0_report["load"][name]["rows_loaded"]
        prefix = f"S{name[-1]}-"
        nm, am = df.name_missing.astype(bool), df.address_missing.astype(bool)
        r = {
            "rows": int(len(df)), "expected_rows": int(exp),
            "duplicate_ids": int(df.entity_id.duplicated().sum()),
            "bad_prefix": int((~df.entity_id.str.startswith(prefix)).sum()),
            "name_missing_vs_fold_mismatch": int((nm != df.name_fold.isna()).sum()),
            "name_missing_vs_core_mismatch": int((nm != df.name_core.isna()).sum()),
            "address_missing_vs_fold_mismatch": int((am != df.address_fold.isna()).sum()),
            "empty_core": int((df.name_core.fillna("x").str.len() == 0).sum()),
            "empty_address_norm": int(((~am) & (df.address_tokens_norm.fillna("").str.len() == 0)).sum()),
            "unsupported_script_left": int(df.name_roman.map(has_unsupported_script).sum() +
                                           df.address_roman.map(has_unsupported_script).sum()),
            "name_missing": int(nm.sum()), "address_missing": int(am.sum()),
        }
        out[name] = r
        gate("FAIL", f"{name} row count", r["rows"] == exp, f"{r['rows']} vs {exp}")
        gate("FAIL", f"{name} unique ids", r["duplicate_ids"] == 0 and r["bad_prefix"] == 0)
        gate("FAIL", f"{name} missing flags consistent",
             r["name_missing_vs_fold_mismatch"] + r["name_missing_vs_core_mismatch"] +
             r["address_missing_vs_fold_mismatch"] == 0)
        gate("FAIL", f"{name} no empty core names", r["empty_core"] == 0, str(r["empty_core"]))
        gate("FAIL", f"{name} no unsupported script left", r["unsupported_script_left"] == 0)
        gate("WARN", f"{name} address normalised to nothing", r["empty_address_norm"] < 1000,
             str(r["empty_address_norm"]))
        log(f"A. {name} checked")
        del df
    return out


# ============================================================================ B. config

def config_checks(cfg_path: Path) -> dict:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = {"abbr_chains": {}, "suspicious_suffixes": {}}
    for kind in ("abbr_name", "abbr_address"):
        for c, m in cfg.get(kind, {}).items():
            chains = [(k, v) for k, v in m.items() if v in m]
            out["abbr_chains"][f"{kind}|{c}"] = chains[:20]
            gate("FAIL", f"{kind}|{c} has no chains/cycles", not chains, str(chains[:5]))
    for c, lst in cfg.get("suffix", {}).items():
        sus = [t for t in lst if t not in LEGAL_SUFFIX_HAND and not suffix_like(t, LEGAL_SUFFIX_HAND)]
        out["suspicious_suffixes"][c] = sus
        gate("WARN", f"suffix list {c} has only legal-like discoveries", not sus, str(sus))
    out["discovered_suffixes"] = {c: sorted(set(lst) - LEGAL_SUFFIX_HAND)
                                  for c, lst in cfg.get("suffix", {}).items()}
    return out


# ============================================================================ F. blocking inputs

def write_block_inputs(repr_dir: Path, block_dir: Path, phase0_dir: Path) -> dict:
    block_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(phase0_dir / "s1_split.tsv", sep="\t", dtype=str, keep_default_na=False)
    manifest = {"columns": BLOCK_COLS, "files": {}}
    for name in SOURCES:
        df = read_table(repr_dir / name, columns=BLOCK_COLS)
        if name == "train_source1":
            df = df.merge(split[["s1_id", "split"]], left_on="entity_id", right_on="s1_id",
                          how="left").drop(columns="s1_id")
            gate("FAIL", "every train S1 has a split label", df["split"].notna().all())
        p = write_table(df, block_dir / name)
        manifest["files"][name] = {"path": str(p), "rows": int(len(df))}
        log(f"F. {name} -> {p}")
    (block_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_block(block_dir: Path, name: str) -> pd.DataFrame:
    return read_table(block_dir / name)


def load_df_tables(repr_dir: Path, split: str) -> dict:
    t = read_table(repr_dir / f"df_{split}")
    out = {}
    for (c, f), g in t.groupby(["country", "field"]):
        out[(c, f)] = dict(zip(g["token"], g["df"].astype(int)))
    return out


# ============================================================================ C. signal coverage

def min_df(shared: set, table: dict) -> float:
    return min((table.get(t, 0) for t in shared), default=np.inf)


def pair_signals(x: dict, y: dict, c: str, dft: dict) -> dict:
    sk_x, sk_y = nn(x["name_sorted_key"]), nn(y["name_sorted_key"])
    al_x, al_y = nn(x["name_alias"]), nn(y["name_alias"])
    cc_x, cc_y = nn(x["name_concat"]), nn(y["name_concat"])
    in_x, in_y = nn(x["name_initials"]), nn(y["name_initials"])
    core_x, core_y = toks(x["name_core"]), toks(y["name_core"])
    ph = toks(x["name_phon"]) & toks(y["name_phon"])
    ad = toks(x["address_tokens_norm"]) & toks(y["address_tokens_norm"])
    nu = toks(x["address_nums"]) & toks(y["address_nums"])
    nc = toks(x["address_numcompounds"]) & toks(y["address_numcompounds"])
    sc = core_x & core_y
    return {
        "sorted_key_eq": bool(sk_x) and sk_x == sk_y,
        "concat_eq": bool(cc_x) and cc_x == cc_y,
        "alias_hit": bool((al_y and sk_x == al_y) or (al_x and sk_y == al_x)),
        "initials_hit": bool((in_x and len(core_y) == 1 and in_x in core_y) or
                             (in_y and len(core_x) == 1 and in_y in core_x)),
        "core_any": bool(sc), "phon_any": bool(ph), "addr_any": bool(ad), "num_any": bool(nu),
        "numcompound": bool(nc),
        "core_min_df": min_df(sc, dft.get((c, "name_core"), {})),
        "phon_min_df": min_df(ph, dft.get((c, "name_phon"), {})),
        "addr_min_df": min_df(ad, dft.get((c, "address_tokens_norm"), {})),
        "num3_min_df": min_df({t for t in nu if len(t) >= 3}, dft.get((c, "address_nums"), {})),
        "cross_script": (nn(x["name_script"]) or "none") != (nn(y["name_script"]) or "none"),
        "other_addr_missing": bool(y["address_missing"]),
    }


def derive(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    exact = df.sorted_key_eq | df.concat_eq | df.alias_hit
    return pd.DataFrame({
        "exact_key": exact,
        f"core_rare": df.core_min_df <= cap, f"phon_rare": df.phon_min_df <= cap,
        f"addr_rare": df.addr_min_df <= cap, f"num3_rare": df.num3_min_df <= cap,
        "numcompound": df.numcompound, "initials_hit": df.initials_hit,
        "any_rare_signal": exact | (df.core_min_df <= cap) | (df.phon_min_df <= cap) |
                           (df.addr_min_df <= cap) | (df.num3_min_df <= cap) | df.numcompound |
                           df.initials_hit,
        "any_signal": exact | df.core_any | df.phon_any | df.addr_any | df.num_any |
                      df.numcompound | df.initials_hit,
    })


def signal_coverage(block_dir, phase0_dir, repr_dir, out_dir, n_s1, seed):
    gt = pd.read_csv(phase0_dir / "gt_long.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = load_block(block_dir, "train_source1")
    s23 = pd.concat([load_block(block_dir, "train_source2"), load_block(block_dir, "train_source3")],
                    ignore_index=True)
    train_ids = s1.loc[s1["split"] == "train", "entity_id"].tolist()
    rng = random.Random(seed)
    sample = set(rng.sample(sorted(train_ids), min(n_s1, len(train_ids))))
    pairs = gt[gt.s1_id.isin(sample)]
    n_match = pairs.groupby("s1_id").size()
    truth = pairs.groupby("s1_id")["matched_id"].apply(set).to_dict()
    s1_country = dict(zip(s1.entity_id, s1.country_norm))
    ids_by_c = {c: g.entity_id.to_numpy() for c, g in s23.groupby("country_norm")}
    nprng = np.random.default_rng(seed)
    # draw negatives up front so only the records actually used are turned into dicts
    negs = {}
    for sid, mids in truth.items():
        pool = ids_by_c[s1_country[sid]]
        negs[sid] = [m for m in pool[nprng.integers(0, len(pool), size=len(mids) + 2)]
                     if m not in mids][:len(mids)]
    need23 = set(pairs.matched_id) | {m for v in negs.values() for m in v}
    s1r = s1[s1.entity_id.isin(truth.keys())].set_index("entity_id").to_dict("index")
    s23r = s23[s23.entity_id.isin(need23)].set_index("entity_id").to_dict("index")
    del s23
    dft = load_df_tables(repr_dir, "train")
    log(f"C. {len(pairs):,} true pairs from {len(sample):,} train S1")

    rows = []
    for sid, mids in truth.items():
        x = s1r[sid]
        c = x["country_norm"]
        for mid in mids:
            f = pair_signals(x, s23r[mid], c, dft)
            f.update(label=1, s1_id=sid, other_id=mid, country=c, source=mid[:2],
                     bucket=min(int(n_match[sid]), 4))
            rows.append(f)
        for mid in negs[sid]:
            if mid in mids:
                continue
            f = pair_signals(x, s23r[mid], c, dft)
            f.update(label=0, s1_id=sid, other_id=mid, country=c, source=mid[:2], bucket=-1)
            rows.append(f)
    P = pd.DataFrame(rows)
    pos, neg = P[P.label == 1], P[P.label == 0]

    rep = {"true_pairs": int(len(pos)), "random_pairs": int(len(neg)), "by_cap": {}}
    for cap in CAPS:
        dp, dn = derive(pos, cap), derive(neg, cap)
        r = {"true_pairs_share": dp.mean().round(4).to_dict(),
             "random_pairs_share": dn.mean().round(4).to_dict(), "segments": {}}
        seg = pos.assign(**{k: dp[k] for k in dp.columns})
        for segname, key in (("country", ["country"]), ("country_source", ["country", "source"]),
                             ("cross_script", ["cross_script"]),
                             ("other_address_missing", ["other_addr_missing"]),
                             ("s1_match_count", ["bucket"])):
            r["segments"][segname] = {
                "|".join(map(str, k if isinstance(k, tuple) else (k,))):
                    {"pairs": int(len(g)), **g[list(dp.columns)].mean().round(4).to_dict()}
                for k, g in seg.groupby(key)}
        ent = seg.groupby("s1_id")["any_rare_signal"].mean()
        r["entity_level"] = {"s1_all_matches_have_rare_signal": round(float((ent == 1).mean()), 4),
                             "mean_per_s1_share": round(float(ent.mean()), 4)}
        rep["by_cap"][str(cap)] = r
    # dumps (cap 1000)
    d = derive(pos, 1000)

    def dump(mask, fn):
        sub = pos[mask].sample(n=min(300, int(mask.sum())), random_state=1) if mask.any() else pos.head(0)
        rows_ = []
        for r in sub.itertuples():
            x, y = s1r[r.s1_id], s23r[r.other_id]
            rows_.append({"s1_id": r.s1_id, "other_id": r.other_id, "country": r.country,
                          "s1_core": x["name_core"], "other_core": y["name_core"],
                          "s1_phon": x["name_phon"], "other_phon": y["name_phon"],
                          "s1_address": x["address_tokens_norm"], "other_address": y["address_tokens_norm"],
                          "cross_script": r.cross_script, "other_addr_missing": r.other_addr_missing})
        pd.DataFrame(rows_).to_csv(out_dir / fn, sep="\t", index=False)
    dump(~d.any_signal.values, "no_signal_true_pairs.tsv")
    dump(~d.any_rare_signal.values, "no_rare_signal_true_pairs.tsv")
    no_rare = 1 - rep["by_cap"]["10000"]["true_pairs_share"]["any_rare_signal"]
    gate("WARN", "true pairs without any signal at cap 10000 < 3%", no_rare < 0.03, f"{no_rare:.4f}")
    return rep


# ============================================================================ D + E. unlabeled / blocks

def key_stats(split: str, block_dir: Path, repr_dir: Path) -> dict:
    s1 = load_block(block_dir, f"{split}_source1")
    s23 = pd.concat([load_block(block_dir, f"{split}_source2"),
                     load_block(block_dir, f"{split}_source3")], ignore_index=True)
    dft = load_df_tables(repr_dir, split)
    out = {}
    for key in ("name_sorted_key", "name_concat"):
        cnt = (s23.dropna(subset=[key]).groupby(["country_norm", key]).size()
               .rename("block").reset_index())
        m = s1[["entity_id", "country_norm", key]].merge(cnt, on=["country_norm", key], how="left")
        m["block"] = m["block"].fillna(0)
        for c, g in m.groupby("country_norm"):
            hit = g.block > 0
            b = g.loc[hit, "block"]
            out.setdefault(c, {})[key] = {
                "s1": int(len(g)), "hit_rate": round(float(hit.mean()), 4),
                "block_p50": float(b.median()) if len(b) else 0, "block_p99": float(b.quantile(0.99)) if len(b) else 0,
                "block_max": int(b.max()) if len(b) else 0, "share_block_gt_1000": round(float((g.block > 1000).mean()), 5)}
    # rarest-token availability per S1
    for field, col in (("name_core", "name_core"), ("name_phon", "name_phon"),
                       ("address_tokens_norm", "address_tokens_norm")):
        for c, g in s1.groupby("country_norm"):
            table = dft.get((c, field), {})
            mins = g[col].map(lambda v: min((table.get(t, 0) for t in toks(v)), default=np.inf))
            present = np.isfinite(mins)
            out.setdefault(c, {})[f"rarest_{field}_df"] = {
                "p50": float(np.median(mins[present])) if present.any() else None,
                "p90": float(np.quantile(mins[present], 0.9)) if present.any() else None,
                "share_with_token_df_le_1000": round(float((mins <= 1000).mean()), 4),
                "share_with_token_df_le_10000": round(float((mins <= 10000).mean()), 4),
                "share_no_token_in_pool": round(float((mins == 0).mean()), 4)}
    return out


def unlabeled_gate(test_stats: dict):
    rates = {c: v["name_sorted_key"]["hit_rate"] for c, v in test_stats.items()}
    known = [r for c, r in rates.items() if c in ("us", "india")]
    for c, r in rates.items():
        if c in ("us", "india") or not known:
            continue
        gate("WARN", f"test {c} exact-key hit rate comparable to us/india", r >= 0.5 * min(known),
             f"{c}={r} vs us/india={known}")


# ============================================================================ summary

def write_summary(rep: dict, path: Path):
    L = ["# Phase 4 summary\n", f"**READY FOR BLOCKING: {rep['ready_for_blocking']}**\n", "## Gates"]
    for g in rep["gates"]:
        if g["level"] != "PASS":
            L.append(f"- **{g['level']}** {g['check']} {g['detail']}")
    L.append(f"- {sum(g['level'] == 'PASS' for g in rep['gates'])} checks passed")
    L.append("\n## Discovered legal suffixes (after the Phase 4 rule)\n" +
             json.dumps(rep["config"]["discovered_suffixes"]))
    for cap, r in rep["signal_coverage"]["by_cap"].items():
        L.append(f"\n## Signal coverage, rare = doc-freq <= {cap}")
        L.append("true pairs:   " + json.dumps(r["true_pairs_share"]))
        L.append("random pairs: " + json.dumps(r["random_pairs_share"]))
        L.append("entity level: " + json.dumps(r["entity_level"]))
        for seg, d in r["segments"].items():
            L.append(f"\n### by {seg}\n```")
            for k, v in d.items():
                L.append(f"{k}: pairs={v['pairs']} exact={v['exact_key']} core={v['core_rare']} "
                         f"phon={v['phon_rare']} addr={v['addr_rare']} num3={v['num3_rare']} "
                         f"numcomp={v['numcompound']} any_rare={v['any_rare_signal']} any={v['any_signal']}")
            L.append("```")
    for split in ("train", "test"):
        L.append(f"\n## Key and block statistics ({split} pool)\n```\n" +
                 json.dumps(rep["key_stats"][split], indent=1) + "\n```")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr-dir", default="data_repr")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--phase3-dir", default="artifacts/phase3")
    ap.add_argument("--out-dir", default="artifacts/phase4")
    ap.add_argument("--block-dir", default="data_block")
    ap.add_argument("--n-s1", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-integrity", action="store_true", help="skip section A (slowest part)")
    a = ap.parse_args()
    repr_dir, phase0_dir, out_dir, block_dir = (Path(a.repr_dir), Path(a.phase0_dir),
                                                Path(a.out_dir), Path(a.block_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    p0 = json.loads((phase0_dir / "phase0_report.json").read_text())

    rep = {"args": vars(a)}
    if not a.skip_integrity:
        rep["integrity"] = integrity(repr_dir, p0)
    rep["config"] = config_checks(Path(a.phase3_dir) / "config" / "phase3_config.json")
    log("B. config checked")
    rep["blocking_inputs"] = write_block_inputs(repr_dir, block_dir, phase0_dir)
    rep["signal_coverage"] = signal_coverage(block_dir, phase0_dir, repr_dir, out_dir, a.n_s1, a.seed)
    log("C. signal coverage done")
    rep["key_stats"] = {s: key_stats(s, block_dir, repr_dir) for s in ("train", "test")}
    unlabeled_gate(rep["key_stats"]["test"])
    log("D/E. key statistics done")

    rep["gates"] = GATES
    rep["ready_for_blocking"] = not any(g["level"] == "FAIL" for g in GATES)
    with open(out_dir / "phase4_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
    write_summary(rep, out_dir / "phase4_summary.md")
    log(f"done -> {out_dir}  READY FOR BLOCKING: {rep['ready_for_blocking']}")
    sys.exit(0 if rep["ready_for_blocking"] else 1)


if __name__ == "__main__":
    main()
