#!/usr/bin/env python3
"""
One-time conversion of the six source TSVs to Parquet (needs pyarrow).
Loading 5M-row TSVs takes minutes; Parquet takes seconds, and every later phase reloads them.
Values are kept exactly as raw strings - no cleaning happens here.

Usage:  python src/prepare_parquet.py --data-dir dataset --out-dir data_parquet
"""
import argparse
import time
from pathlib import Path

from common import load_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="data_parquet")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for k in (1, 2, 3):
            t = time.time()
            df = load_source(Path(args.data_dir), split, k)  # TSV path, parse-checked
            df.to_parquet(out / f"{split}_source{k}.parquet", index=False)
            print(f"{split}_source{k}: {len(df):,} rows -> parquet in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
