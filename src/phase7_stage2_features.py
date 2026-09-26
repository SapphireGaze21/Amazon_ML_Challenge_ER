#!/usr/bin/env python3
"""Phase 7 -- expensive Stage-2 features for Stage-1's top-N candidates only.

This deliberately has no country column or country-derived feature.  It reads the compact
lists written by phase6_stage1.py, joins the records in data_repr, and writes an equally
sharded feature table.  Each worker handles a complete Stage-1 shard, so all candidate-list
(``neighbourhood``) features are calculated correctly.

Example:
  python src/phase7_stage2_features.py --split train --workers 8
  python src/phase7_stage2_features.py --split val   --workers 8
  python src/phase7_stage2_features.py --split test  --workers 8
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from blocking import coarse_key
from common import read_table, write_table

T0 = time.time()
REPR_COLS = ["entity_id", "name_missing", "address_missing", "name_script", "address_script",
             "name_roman", "name_core", "name_suffix", "name_sorted_key", "name_alias", "name_concat",
             "name_initials", "name_phon", "address_roman", "address_tokens_norm",
             "address_nums"]
OUT_BASE = ["s1_id", "cand_id", "stage1_score", "stage1_rank", "passes", "priority", "rank"]
MEASURES = ("ratio", "token_sort_ratio", "token_set_ratio", "partial_ratio", "jaro_winkler")
W = {}


def log(s):
    print(f"[{time.time() - T0:7.0f}s] {s}", flush=True)


def shards(d: Path):
    return [str(p)[:-len(ext)] for p in sorted(d.iterdir())
            for ext in (".parquet", ".tsv.gz") if p.name.endswith(ext)]


def text(x):
    return x if isinstance(x, str) else ""


def tok(x):
    return set(text(x).split())


def overlap(a, b):
    """Jaccard and overlap coefficient; zero for empty evidence, never a false match."""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0, 0.0
    n = len(a & b)
    return n / len(a | b), n / min(len(a), len(b))


def idf_overlap(a, b, idf):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    inter = sum(idf.get(t, 1.0) for t in a & b)
    union = sum(idf.get(t, 1.0) for t in a | b)
    return inter / union if union else 0.0


def longest_postal(nums):
    """The longest numeric run of length >=5: ZIP, PIN, and French CP without country rules."""
    x = [n for n in nums if len(n) >= 5]
    return max(x, key=lambda n: (len(n), n)) if x else None


def number_features(a, b):
    a, b = tok(a), tok(b)
    pa, pb = longest_postal(a), longest_postal(b)
    # Do not call a postal code a house number.  Remaining digit runs are house/unit evidence.
    ha, hb = a - ({pa} if pa else set()), b - ({pb} if pb else set())
    return {
        "house_num_agree": float(bool(ha & hb)),
        "house_num_conflict": float(bool(ha and hb and not (ha & hb))),
        "postal_present_both": float(pa is not None and pb is not None),
        "postal_agree": float(pa is not None and pa == pb),
        "postal_conflict": float(pa is not None and pb is not None and pa != pb),
        "number_agree": float(bool(a & b)),
        "number_conflict": float(bool(a and b and not (a & b))),
    }


def similarities(prefix, a, b):
    a, b = text(a), text(b)
    # Empty text is absence of evidence, not a perfect string match.
    if not a or not b:
        return {f"{prefix}_{m}": 0.0 for m in MEASURES}
    return {f"{prefix}_ratio": fuzz.ratio(a, b) / 100.0,
            f"{prefix}_token_sort_ratio": fuzz.token_sort_ratio(a, b) / 100.0,
            f"{prefix}_token_set_ratio": fuzz.token_set_ratio(a, b) / 100.0,
            f"{prefix}_partial_ratio": fuzz.partial_ratio(a, b) / 100.0,
            # rapidfuzz.distance similarities are already normalized to [0, 1].
            f"{prefix}_jaro_winkler": JaroWinkler.similarity(a, b)}


def name_info(core, idf):
    return float(sum(idf.get(t, 1.0) for t in tok(core)))


def pair_features(x, y, idf_name, idf_addr):
    r = {}
    for p, a, b in (("name_core", x.name_core, y.name_core),
                    ("name_roman", x.name_roman, y.name_roman),
                    ("name_concat", x.name_concat, y.name_concat),
                    ("address_norm", x.address_tokens_norm, y.address_tokens_norm),
                    ("address_roman", x.address_roman, y.address_roman)):
        r.update(similarities(p, a, b))
    for p, a, b in (("name_token", tok(x.name_core), tok(y.name_core)),
                    ("phon", tok(x.name_phon), tok(y.name_phon)),
                    ("coarse_phon", {coarse_key(t) for t in tok(x.name_phon)},
                     {coarse_key(t) for t in tok(y.name_phon)}),
                    ("address_token", tok(x.address_tokens_norm), tok(y.address_tokens_norm))):
        j, c = overlap(a, b)
        r[f"{p}_jaccard"], r[f"{p}_overlap"] = j, c
    r["address_idf_overlap"] = idf_overlap(tok(x.address_tokens_norm), tok(y.address_tokens_norm), idf_addr)
    r["name_idf_overlap"] = idf_overlap(tok(x.name_core), tok(y.name_core), idf_name)
    sx, sy = text(x.name_sorted_key), text(y.name_sorted_key)
    ax, ay = text(x.name_alias), text(y.name_alias)
    ix, iy = text(x.name_initials), text(y.name_initials)
    r["alias_match"] = float(bool((ax and ax == sy) or (ay and ay == sx) or (ax and ay and ax == ay)))
    r["initials_match"] = float(bool(ix and iy and ix == iy))
    r["legal_suffix_agree"] = float(bool(tok(x.name_suffix) & tok(y.name_suffix)))
    r.update(number_features(x.address_nums, y.address_nums))
    r["address_missing_either"] = float(bool(x.address_missing or y.address_missing))
    r["cross_script"] = float(bool(text(x.name_script) and text(y.name_script) and
                                    text(x.name_script) != text(y.name_script)))
    r["name_info_s1"] = name_info(x.name_core, idf_name)
    r["name_info_cand"] = name_info(y.name_core, idf_name)
    r["name_info_min"] = min(r["name_info_s1"], r["name_info_cand"])
    return r


def load_source_index(repr_paths):
    """One compact, read-only index shared copy-on-write by Linux fork workers.

    A Python ``dict(entity_id -> dict)`` costs many times the raw dataframe size.  Keeping
    columns in pandas blocks is substantially smaller, which matters for the 16-vCPU / 128-GB
    SageMaker instances this stage is intended for.  It is constructed once in the parent;
    children only read it.
    """
    frames = [read_table(Path(p), columns=REPR_COLS) for p in repr_paths]
    source = pd.concat(frames, ignore_index=True).set_index("entity_id", drop=False)
    if source.index.has_duplicates:
        dup = source.index[source.index.duplicated()][0]
        raise ValueError(f"entity_id must be unique across sources; duplicate {dup!r}")
    return source


def init_worker(state):
    W.clear(); W.update(state)
    # With Linux/fork this is inherited copy-on-write.  Spawn platforms retain the compatible
    # fallback of constructing it once per worker, rather than once per candidate shard.
    if "source" not in W:
        W["source"] = load_source_index(state["repr_paths"])


def build_shard(base):
    pairs = read_table(Path(base))
    source = W["source"]
    rows = []
    for sid, g in pairs.groupby("s1_id", sort=False):
        if sid not in source.index:
            continue
        x = source.loc[sid]
        present = g.cand_id.isin(source.index)
        gg = g.loc[present]
        if gg.empty:
            continue
        y_rows = source.loc[gg.cand_id.to_list()]
        # itertuples is materially faster than iterrows for the millions of top-N pairs here.
        cand = list(zip(gg.itertuples(index=False), y_rows.itertuples(index=False)))
        if not cand:
            continue
        best = min(cand, key=lambda q: q[0].stage1_rank)[1]
        # Exact duplicate candidate records make a score less decisive; use both nonempty fields.
        copies = Counter((text(y.name_roman), text(y.address_roman)) for _, y in cand
                         if text(y.name_roman) and text(y.address_roman))
        best_score = max(float(z.stage1_score) for z, _ in cand)
        for z, y in cand:
            out = {k: getattr(z, k) for k in OUT_BASE if hasattr(z, k)}
            out.update(pair_features(x, y, W["idf_name"], W["idf_addr"]))
            out["stage1_score_gap"] = best_score - float(z.stage1_score)
            out["stage1_rank_gap"] = float(z.stage1_rank)  # best rank is always zero
            out["to_s1_best_name_ratio"] = fuzz.ratio(text(y.name_core), text(best.name_core)) / 100.0 \
                if text(y.name_core) and text(best.name_core) else 0.0
            out["to_s1_best_addr_ratio"] = fuzz.ratio(text(y.address_tokens_norm), text(best.address_tokens_norm)) / 100.0 \
                if text(y.address_tokens_norm) and text(best.address_tokens_norm) else 0.0
            out["identical_copy_present"] = float(copies[(text(y.name_roman), text(y.address_roman))] > 1)
            rows.append(out)
    return pd.DataFrame(rows)


def token_idf(paths, field):
    df, n = Counter(), 0
    for p in paths:
        d = read_table(Path(p), columns=[field])
        n += len(d)
        for v in d[field]:
            df.update(tok(v))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--stage1-dir", default="data_stage1")
    ap.add_argument("--repr-dir", default="data_repr")
    ap.add_argument("--out-dir", default="data_stage2")
    ap.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)),
                    help="12 is a safe default on ml.r5.4xlarge; use 16 after checking RAM")
    ap.add_argument("--start-method", choices=("auto", "fork", "spawn"), default="auto",
                    help="auto=fork on Linux (shared read-only repr index), spawn elsewhere")
    a = ap.parse_args()
    inp, out = Path(a.stage1_dir) / a.split, Path(a.out_dir) / a.split
    out.mkdir(parents=True, exist_ok=True)
    split = a.split
    repr_paths = [str(Path(a.repr_dir) / f"{split}_source1"),
                  str(Path(a.repr_dir) / f"{split}_source2"), str(Path(a.repr_dir) / f"{split}_source3")]
    # IDF is derived from the candidate pool only (S2+S3), never from country.
    pool_paths = repr_paths[1:]
    log("building global (not country-specific) IDF tables")
    state = {"repr_paths": repr_paths, "idf_name": token_idf(pool_paths, "name_core"),
             "idf_addr": token_idf(pool_paths, "address_tokens_norm")}
    if a.start_method == "fork" or (a.start_method == "auto" and sys.platform.startswith("linux")):
        log("loading shared representation index once (Linux fork / copy-on-write)")
        state["source"] = load_source_index(repr_paths)
    todo = []
    for b in shards(inp):
        target = out / Path(b).name
        if not (target.with_suffix(".parquet").exists() or target.with_suffix(".tsv.gz").exists()):
            todo.append(b)
    log(f"{len(todo)} Stage-1 shards to featurize with {a.workers} workers")
    method = ("fork" if a.start_method == "auto" and sys.platform.startswith("linux")
              else "spawn" if a.start_method == "auto" else a.start_method)
    log(f"multiprocessing start method: {method}")
    ctx = mp.get_context(method)
    with ctx.Pool(a.workers, initializer=init_worker, initargs=(state,)) as pool:
        for base, df in zip(todo, pool.imap(build_shard, todo)):
            write_table(df, out / Path(base).name)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
