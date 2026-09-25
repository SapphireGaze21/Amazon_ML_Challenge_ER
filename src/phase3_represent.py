#!/usr/bin/env python3
"""
Phase 3 - Representations.

Steps
  1. MINE from TRAIN-split true pairs (validation S1s are never used):
       - abbreviation maps for names and addresses (single-token substitutions like rd -> road)
       - "droppability" of name tokens: how often a token appears on one side of a true pair
         but not the other (legal suffixes get dropped a lot; real name words rarely)
  2. POSITION stats from ALL sources, train + test, unlabeled: how often each name token sits
     in the last two positions. This is how legal suffixes are found for France, which has
     no labels.
  3. DECIDE per country: legal-suffix set = hand list + discovered tokens; abbreviation map =
     hand map + mined map. Optional manual overrides from config/phase3_overrides.json.
     Everything is written to TSV so you can review and correct it.
  4. TRANSFORM all six sources with represent.build_repr (parallel) -> {out}/{split}_source{k}
  5. DOCUMENT FREQUENCIES per country over S2+S3, separately for the train and test pools.
  6. SANITY metrics on train true pairs vs random pairs, incl. the effect of romanization on
     cross-script pairs.

Usage
  python tests/test_represent.py
  python src/phase3_represent.py --clean-dir data_clean --phase0-dir artifacts/phase0 \
      --out-dir data_repr --report-dir artifacts/phase3
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from common import char_ngrams, nn, read_table, write_table
from represent import (ABBR_ADDRESS_HAND, ABBR_NAME_HAND, LEGAL_SUFFIX_HAND, REPR_COLUMNS,
                       SubstitutionMiner, address_token_list, build_repr, build_repr_chunk,
                       init_config, name_mining_tokens)
from romanize import has_unsupported_script

T0 = time.time()
KEY_COLS = ["entity_id", "name_fold", "address_fold", "country_norm"]
DF_FIELDS = {"name_core": "name_core", "name_phon": "name_phon",
             "address_tokens_norm": "address_tokens_norm", "address_nums": "address_nums",
             "address_numcompounds": "address_numcompounds"}


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ============================================================================ 1-2. mining

def mine_from_pairs(clean_dir, phase0_dir, n_pairs, seed):
    gt = pd.read_csv(Path(phase0_dir) / "gt_long.tsv", sep="\t", dtype=str, keep_default_na=False)
    split = pd.read_csv(Path(phase0_dir) / "s1_split.tsv", sep="\t", dtype=str, keep_default_na=False)
    train_ids = set(split.loc[split.split == "train", "s1_id"])
    pairs = gt[gt.s1_id.isin(train_ids)]
    if len(pairs) > n_pairs:
        pairs = pairs.sample(n=n_pairs, random_state=seed)
    need1, need23 = set(pairs.s1_id), set(pairs.matched_id)
    rec = {}
    for k in (1, 2, 3):
        df = read_table(Path(clean_dir) / f"train_source{k}", columns=KEY_COLS)
        need = need1 if k == 1 else need23
        df = df[df.entity_id.isin(need)]
        rec.update((i, (nn(n), nn(ad), nn(c))) for i, n, ad, c in
                   zip(df.entity_id, df.name_fold, df.address_fold, df.country_norm))
    log(f"mining on {len(pairs):,} train true pairs")

    name_m, addr_m = SubstitutionMiner(), SubstitutionMiner()
    present, dropped, n_pairs_c = defaultdict(Counter), defaultdict(Counter), Counter()
    kept_pairs = []
    for s1, m in zip(pairs.s1_id, pairs.matched_id):
        r1, r2 = rec[s1], rec[m]
        c = r1[2]
        a, b = name_mining_tokens(r1[0]), name_mining_tokens(r2[0])
        name_m.add(c, a, b)
        a1, a2 = address_token_list(r1[1]), address_token_list(r2[1])
        addr_m.add(c, a1, a2)
        addr_m.add_phrase(c, a1, a2)
        sa, sb = set(a), set(b)
        for t in sa | sb:
            present[c][t] += 1
        for t in sa ^ sb:
            dropped[c][t] += 1
        n_pairs_c[c] += 1
        kept_pairs.append((r1, r2, c))
    return name_m, addr_m, present, dropped, n_pairs_c, kept_pairs


def positional_stats(clean_dir, splits, per_source, seed):
    n_names, present, tail = Counter(), defaultdict(Counter), defaultdict(Counter)
    for split in splits:
        for k in (1, 2, 3):
            p = Path(clean_dir) / f"{split}_source{k}"
            df = read_table(p, columns=["name_fold", "country_norm"])
            if len(df) > per_source:
                df = df.sample(n=per_source, random_state=seed)
            for nf, c in zip(df.name_fold, df.country_norm):
                toks = name_mining_tokens(nn(nf))
                if len(toks) < 2:
                    continue
                n_names[c] += 1
                for t in set(toks):
                    present[c][t] += 1
                for t in set(toks[-2:]):
                    tail[c][t] += 1
            log(f"  position stats: {split}_source{k}")
    return n_names, present, tail


# ============================================================================ 3. decisions

def decide_suffixes(n_names, pos_present, tail, pair_present, dropped, n_pairs_c, a):
    decided, rows = {}, []
    for c in sorted(n_names):
        labeled = n_pairs_c.get(c, 0) >= 1000
        accepted = set(LEGAL_SUFFIX_HAND)
        cands = {t for t, n in tail[c].items() if n / n_names[c] >= a.suffix_report_share}
        for t in sorted(cands | LEGAL_SUFFIX_HAND):
            share_tail = tail[c][t] / n_names[c]
            p_tail = tail[c][t] / pos_present[c][t] if pos_present[c][t] else 0.0
            pp = pair_present[c][t] if labeled else 0
            drop = dropped[c][t] / pp if pp >= 50 else None
            if t in LEGAL_SUFFIX_HAND:
                ok, why = True, "hand"
            elif not t.isalpha():
                ok, why = False, "not alphabetic"
            elif labeled:
                ok = (len(t) <= 6 and share_tail >= a.suffix_min_share and p_tail >= 0.7
                      and drop is not None and drop >= a.suffix_min_drop)
                why = "discovered (position + droppability)" if ok else "rejected"
            else:
                ok = len(t) <= 5 and share_tail >= a.suffix_min_share_unlabeled and p_tail >= 0.85
                why = "discovered (position only, unlabeled country)" if ok else "rejected"
            if ok:
                accepted.add(t)
            if t in cands or (ok and tail[c][t] > 0):
                rows.append((c, t, why, ok, round(share_tail, 5), round(p_tail, 3),
                             None if drop is None else round(drop, 3), int(tail[c][t])))
        decided[c] = accepted
    return decided, rows


def apply_overrides(cfg, path):
    """config/phase3_overrides.json, all keys optional; country "*" means every country:
    {"suffix_add": {"france": ["sarlu"]}, "suffix_remove": {"*": ["co"]},
     "abbr_name_add": {"*": {"tech": "technologies"}}, "abbr_name_remove": {"us": ["st"]},
     "abbr_address_add": {...}, "abbr_address_remove": {...}}"""
    p = Path(path)
    if not p.exists():
        return {}
    ov = json.loads(p.read_text(encoding="utf-8"))
    countries = set(cfg["suffix"]) | set(cfg["abbr_name"]) | set(cfg["abbr_address"])

    def targets(c):
        return countries if c == "*" else {c}

    for c, toks in ov.get("suffix_add", {}).items():
        for cc in targets(c):
            cfg["suffix"].setdefault(cc, set(LEGAL_SUFFIX_HAND)).update(toks)
    for c, toks in ov.get("suffix_remove", {}).items():
        for cc in targets(c):
            cfg["suffix"].setdefault(cc, set(LEGAL_SUFFIX_HAND)).difference_update(toks)
    for kind, hand in (("name", ABBR_NAME_HAND), ("address", ABBR_ADDRESS_HAND)):
        for c, m in ov.get(f"abbr_{kind}_add", {}).items():
            for cc in targets(c):
                cfg[f"abbr_{kind}"].setdefault(cc, dict(hand)).update(m)
        for c, toks in ov.get(f"abbr_{kind}_remove", {}).items():
            for cc in targets(c):
                for t in toks:
                    cfg[f"abbr_{kind}"].setdefault(cc, dict(hand)).pop(t, None)
    return ov


# ============================================================================ 4-5. transform

def transform_all(clean_dir, out_dir, splits, cfg, workers, chunk=50_000):
    df_counts = {s: defaultdict(Counter) for s in splits}  # split -> (country, field) -> Counter
    script_stats, unsupported = {}, Counter()
    with Pool(workers, initializer=init_config, initargs=(cfg,)) as pool:
        for split in splits:
            for k in (1, 2, 3):
                name = f"{split}_source{k}"
                df = read_table(Path(clean_dir) / name)
                rows = [(nn(n), nn(ad), nn(c)) for n, ad, c in
                        zip(df.name_fold, df.address_fold, df.country_norm)]
                parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
                res = []
                for p in pool.imap(build_repr_chunk, parts):
                    res.extend(p)
                rep = pd.DataFrame(res, columns=REPR_COLUMNS, index=df.index)
                out = pd.concat([df, rep], axis=1)
                path = write_table(out, Path(out_dir) / name)
                script_stats[name] = {
                    "name": {str(k2): int(v) for k2, v in out.name_script.value_counts().items()},
                    "address": {str(k2): int(v) for k2, v in out.address_script.value_counts().items()}}
                unsupported[name] = int(out.name_roman.map(has_unsupported_script).sum() +
                                        out.address_roman.map(has_unsupported_script).sum())
                if k in (2, 3):
                    cnt = df_counts[split]
                    for field, col in DF_FIELDS.items():
                        for c, v in zip(out.country_norm, out[col]):
                            if isinstance(v, str) and v:
                                cnt[(c, field)].update(set(v.split(" ")))
                log(f"{name}: {len(out):,} rows -> {path.name}")
                del df, rows, res, rep, out
    return df_counts, script_stats, dict(unsupported)


def write_df_tables(df_counts, out_dir):
    summary = {}
    for split, cnt in df_counts.items():
        frames = []
        for (c, field), counter in cnt.items():
            if not counter:
                continue
            s = pd.Series(counter)
            frames.append(pd.DataFrame({"country": c, "field": field, "token": s.index,
                                        "df": s.values}))
            q = s.quantile([0.5, 0.9, 0.99, 0.999])
            summary[f"{split}|{c}|{field}"] = {
                "vocab": int(len(s)), "p50": float(q[0.5]), "p90": float(q[0.9]),
                "p99": float(q[0.99]), "p999": float(q[0.999]), "max": int(s.max()),
                "tokens_df_gt_1k": int((s > 1_000).sum()), "tokens_df_gt_10k": int((s > 10_000).sum()),
                "tokens_df_gt_100k": int((s > 100_000).sum()),
                "top25": [[t, int(n)] for t, n in s.nlargest(25).items()]}
        if frames:
            write_table(pd.concat(frames, ignore_index=True), Path(out_dir) / f"df_{split}")
    return summary


# ============================================================================ 6. sanity

def sanity(kept_pairs, cfg, n, seed):
    init_config(cfg)
    rng = random.Random(seed)
    sample = kept_pairs if len(kept_pairs) <= n else rng.sample(kept_pairs, n)
    by_c = defaultdict(list)
    for r1, r2, c in kept_pairs:
        by_c[c].append(r2)
    idx = {k: i for i, k in enumerate(REPR_COLUMNS)}

    def sets(r, col):
        v = r[idx[col]]
        return set(v.split(" ")) if v else set()

    def stats(pairs):
        agg = Counter()
        for x, y in pairs:
            agg["n"] += 1
            agg["sorted_key_equal"] += x[idx["name_sorted_key"]] == y[idx["name_sorted_key"]]
            agg["core_token_overlap"] += bool(sets(x, "name_core") & sets(y, "name_core"))
            agg["phon_overlap"] += bool(sets(x, "name_phon") & sets(y, "name_phon"))
            agg["nums_overlap"] += bool(sets(x, "address_nums") & sets(y, "address_nums"))
            at, bt = sets(x, "address_tokens"), sets(y, "address_tokens")
            an, bn = sets(x, "address_tokens_norm"), sets(y, "address_tokens_norm")
            if at and bt:
                agg["addr_pairs"] += 1
                agg["addr_jacc_raw_sum"] += len(at & bt) / len(at | bt)
                agg["addr_jacc_norm_sum"] += len(an & bn) / len(an | bn) if (an | bn) else 0.0
        n = max(agg["n"], 1)
        out = {k: round(agg[k] / n, 4) for k in
               ("sorted_key_equal", "core_token_overlap", "phon_overlap", "nums_overlap")}
        ap = max(agg["addr_pairs"], 1)
        out["addr_token_jaccard_raw_mean"] = round(agg["addr_jacc_raw_sum"] / ap, 4)
        out["addr_token_jaccard_norm_mean"] = round(agg["addr_jacc_norm_sum"] / ap, 4)
        out["pairs"] = agg["n"]
        return out

    pos, neg, cross = [], [], {"n": 0, "fold_c3": 0.0, "roman_c3": 0.0, "phon": 0}
    for r1, r2, c in sample:
        x, y = build_repr(*r1), build_repr(*r2)
        pos.append((x, y))
        z = build_repr(*rng.choice(by_c[c]))
        neg.append((x, z))
        sx, sy = x[idx["name_script"]], y[idx["name_script"]]
        if sx and sy and sx != sy and "none" not in (sx, sy):
            cross["n"] += 1
            cross["fold_c3"] += _jacc(char_ngrams(r1[0]), char_ngrams(r2[0]))
            cross["roman_c3"] += _jacc(char_ngrams(x[idx["name_roman"]]), char_ngrams(y[idx["name_roman"]]))
            cross["phon"] += bool(sets(x, "name_phon") & sets(y, "name_phon"))
    cn = max(cross["n"], 1)
    return {"true_pairs": stats(pos), "random_pairs": stats(neg),
            "cross_script_true_pairs": {
                "pairs": cross["n"],
                "name_char3_jaccard_before_romanization": round(cross["fold_c3"] / cn, 4),
                "name_char3_jaccard_after_romanization": round(cross["roman_c3"] / cn, 4),
                "phon_overlap": round(cross["phon"] / cn, 4)}}


def _jacc(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data_clean")
    ap.add_argument("--phase0-dir", default="artifacts/phase0")
    ap.add_argument("--out-dir", default="data_repr")
    ap.add_argument("--report-dir", default="artifacts/phase3")
    ap.add_argument("--overrides", default="config/phase3_overrides.json")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--mine-pairs", type=int, default=1_000_000)
    ap.add_argument("--suffix-sample", type=int, default=1_000_000, help="names per source")
    ap.add_argument("--min-abbr-count", type=int, default=30)
    ap.add_argument("--min-abbr-share", type=float, default=0.6)
    ap.add_argument("--suffix-min-share", type=float, default=0.001)
    ap.add_argument("--suffix-min-share-unlabeled", type=float, default=0.003)
    ap.add_argument("--suffix-min-drop", type=float, default=0.25)
    ap.add_argument("--suffix-report-share", type=float, default=0.0005)
    ap.add_argument("--sanity-pairs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    out, rep_dir = Path(a.out_dir), Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (rep_dir / "config").mkdir(parents=True, exist_ok=True)
    splits = a.splits.split(",")
    report = {"args": vars(a)}

    # 1-2
    name_m, addr_m, pair_present, dropped, n_pairs_c, kept = mine_from_pairs(
        a.clean_dir, a.phase0_dir, a.mine_pairs, a.seed)
    n_names, pos_present, tail = positional_stats(a.clean_dir, splits, a.suffix_sample, a.seed)
    log("mining done")

    # 3
    suffix, suffix_rows = decide_suffixes(n_names, pos_present, tail, pair_present, dropped,
                                          n_pairs_c, a)
    abbr_name_m, name_rows = name_m.result(a.min_abbr_count, a.min_abbr_share)
    abbr_addr_m, addr_rows = addr_m.result(a.min_abbr_count, a.min_abbr_share)
    phrase_m, phrase_rows = addr_m.phrase_result(a.min_abbr_count)
    countries = set(n_names)
    cfg = {"suffix": suffix,
           "phrase_address": {c: phrase_m.get(c, {}) for c in countries},
           "abbr_name": {c: {**ABBR_NAME_HAND, **abbr_name_m.get(c, {})} for c in countries},
           "abbr_address": {c: {**ABBR_ADDRESS_HAND, **abbr_addr_m.get(c, {})} for c in countries}}
    # a mined abbreviation must not map a legal suffix away before suffix removal sees it
    for c in countries:
        cfg["abbr_name"][c] = {s: l for s, l in cfg["abbr_name"][c].items() if s not in cfg["suffix"][c]}
    report["overrides_applied"] = apply_overrides(cfg, a.overrides)

    cdir = rep_dir / "config"
    pd.DataFrame(suffix_rows, columns=["country", "token", "decision", "accepted", "share_in_last2",
                                       "p_last2_given_present", "droppability", "n_last2"]) \
        .sort_values(["country", "accepted", "share_in_last2"], ascending=[True, False, False]) \
        .to_csv(cdir / "legal_suffixes.tsv", sep="\t", index=False)
    for rows, fn in ((name_rows, "abbr_name_mined.tsv"), (addr_rows, "abbr_address_mined.tsv")):
        pd.DataFrame(rows, columns=["country", "short", "long", "count", "short_total", "share",
                                    "accepted"]).sort_values(["country", "count"], ascending=[True, False]) \
            .to_csv(cdir / fn, sep="\t", index=False)
    pd.DataFrame(phrase_rows, columns=["country", "phrase", "initials", "count", "accepted"]) \
        .sort_values(["country", "count"], ascending=[True, False]) \
        .to_csv(cdir / "phrase_address_mined.tsv", sep="\t", index=False)

    def _jsonable(v):
        if isinstance(v, set):
            return sorted(v)
        if isinstance(v, dict):
            return {" ".join(k) if isinstance(k, tuple) else k: val for k, val in v.items()}
        return v
    with open(cdir / "phase3_config.json", "w", encoding="utf-8") as f:
        json.dump({k: {c: _jsonable(v) for c, v in d.items()} for k, d in cfg.items()},
                  f, indent=1, ensure_ascii=False)
    report["suffixes_discovered"] = {c: sorted(s - LEGAL_SUFFIX_HAND) for c, s in suffix.items()}
    report["abbr_mined_accepted"] = {"name": abbr_name_m, "address": abbr_addr_m}
    report["phrases_mined_accepted"] = {c: {" ".join(w): s for w, s in m.items()}
                                        for c, m in phrase_m.items()}
    report["mining_pairs_by_country"] = dict(n_pairs_c)
    log("config written")

    # 4-5
    df_counts, script_stats, unsupported = transform_all(a.clean_dir, out, splits, cfg, a.workers)
    report["scripts"] = script_stats
    report["unsupported_script_fields"] = unsupported
    report["df_summary"] = write_df_tables(df_counts, out)
    log("DF tables written")

    # 6
    report["sanity"] = sanity(kept, cfg, a.sanity_pairs, a.seed)
    log("sanity done")

    with open(rep_dir / "phase3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    write_summary(report, rep_dir / "phase3_summary.md")
    log(f"done -> {out}, {rep_dir}")


def write_summary(r, path):
    L = ["# Phase 3 summary\n", "## Sanity: true vs random pairs (share of pairs)",
         "```", json.dumps(r["sanity"], indent=1), "```",
         "\n## Legal suffixes discovered beyond the hand list"]
    for c, toks in r["suffixes_discovered"].items():
        L.append(f"- {c}: {toks}")
    L.append("\n## Mined abbreviations accepted (first 40 per country)")
    for kind in ("name", "address"):
        for c, m in r["abbr_mined_accepted"][kind].items():
            L.append(f"- {kind}|{c}: " + ", ".join(f"{s}->{l}" for s, l in list(m.items())[:40]))
    for c, m in r["phrases_mined_accepted"].items():
        L.append(f"- phrase|{c}: " + ", ".join(f"{w}->{s}" for w, s in list(m.items())[:40]))
    L.append("\n## Scripts per source\n```\n" + json.dumps(r["scripts"], indent=1) + "\n```")
    L.append(f"\nFields still containing unsupported scripts: {r['unsupported_script_fields']}")
    L.append("\n## Document frequencies (vocab size, quantiles, heavy tokens)")
    for k, s in r["df_summary"].items():
        L.append(f"- {k}: vocab={s['vocab']:,} p50={s['p50']} p99={s['p99']} max={s['max']:,} "
                 f">1k={s['tokens_df_gt_1k']} >10k={s['tokens_df_gt_10k']} >100k={s['tokens_df_gt_100k']} "
                 f"top: {', '.join(t for t, _ in s['top25'][:10])}")
    L.append(f"\nMining pairs by country: {r['mining_pairs_by_country']}")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
