"""
Shared helpers: loading sources (Parquet if prepared, else TSV) and the light,
profiling-grade text normalisation used in Phase 1.

The normalisation here is deliberately minimal. The real preprocessing is Phase 2 and
will be designed from what Phase 1 finds.
"""

import bisect
import re
import unicodedata
from pathlib import Path

import pandas as pd

from phase0_setup_split import EXPECTED_COLS, read_tsv  # noqa: F401  (re-exported)

SOURCES = ("source1", "source2", "source3")


# ----------------------------------------------------------------------------- loading

def load_source(data_dir: Path, split: str, k: int, parquet_dir: Path | None = None) -> pd.DataFrame:
    """Load {split}_source{k}. Uses Parquet from prepare_parquet.py when present (much faster)."""
    name = f"{split}_source{k}"
    if parquet_dir is not None:
        pq = Path(parquet_dir) / f"{name}.parquet"
        if pq.exists():
            return pd.read_parquet(pq)
    df, info = read_tsv(Path(data_dir) / split / f"{name}.tsv", EXPECTED_COLS)
    if "warning" in info:
        raise RuntimeError(f"{name} did not parse cleanly: {info}")
    df["entity_id"] = df["entity_id"].map(str.strip)
    return df


# ----------------------------------------------------------------------------- normalisation

_LATIN_MAX = 0x024F  # accents are stripped only from Latin letters (é -> e)


def fold(s: str) -> str:
    """
    NFKC + casefold + strip accents from LATIN letters only.
    Indic vowel signs are combining marks too; stripping them blindly would destroy
    Hindi/Tamil words, so combining marks are only dropped after a Latin base letter.
    """
    s = unicodedata.normalize("NFKC", s).casefold()
    if s.isascii():
        return s
    out, last_latin = [], False
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch):
            if last_latin:
                continue
        else:
            last_latin = ord(ch) <= _LATIN_MAX
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# \w misses Indic vowel signs (they are marks, not alphanumerics), so the Indic blocks
# U+0900-U+0DFF are added explicitly; everything else (punctuation, symbols) splits tokens.
_TOKEN_RE = re.compile(r"[\w\u0900-\u0DFF]+")
_DIGITS_RE = re.compile(r"\d+")


def tokens(s: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(s) if t != "_"]


def numeric_tokens(s: str) -> list[str]:
    """Digit runs, converted to ASCII digits (Devanagari ०-९ etc. included)."""
    out = []
    for d in _DIGITS_RE.findall(s):
        try:
            out.append(str(int(d)) if len(d) < 30 else d)
        except ValueError:
            out.append(d)
    return out


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = f" {' '.join(s.split())} "
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


# ----------------------------------------------------------------------------- script detection

_SCRIPT_RANGES = sorted([
    (0x0041, 0x024F, "latin"), (0x1E00, 0x1EFF, "latin"),
    (0x0370, 0x03FF, "greek"), (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"), (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"), (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "kana"), (0x4E00, 0x9FFF, "cjk"), (0xAC00, 0xD7AF, "hangul"),
])
_STARTS = [r[0] for r in _SCRIPT_RANGES]


def _char_script(cp: int) -> str | None:
    i = bisect.bisect_right(_STARTS, cp) - 1
    if i >= 0 and cp <= _SCRIPT_RANGES[i][1]:
        return _SCRIPT_RANGES[i][2]
    return None


def script_of(s: str) -> str:
    """Dominant script of the letters in s; 'mixed' if a second script has >= 20% of letters."""
    if not s:
        return "none"
    if s.isascii():
        return "latin" if any(c.isalpha() for c in s) else "none"
    counts: dict[str, int] = {}
    for ch in s:
        sc = _char_script(ord(ch))
        if sc is None:
            if ch.isalpha():
                sc = "other"
            else:
                continue
        counts[sc] = counts.get(sc, 0) + 1
    if not counts:
        return "none"
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    total = sum(counts.values())
    if len(ranked) > 1 and ranked[1][1] / total >= 0.2:
        return "mixed"
    return ranked[0][0]


# ----------------------------------------------------------------------------- processed tables

def nn(v):
    """Text value or None. pandas 3 stores missing text as NaN, pandas 2 as None; this makes
    both look the same to pure-Python code."""
    return v if isinstance(v, str) else None


def write_table(df: pd.DataFrame, path_no_ext: Path) -> Path:
    """Parquet if pyarrow is installed, else gzipped TSV (fallback)."""
    path_no_ext = Path(path_no_ext)
    try:
        import pyarrow  # noqa: F401
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    except ImportError:
        p = path_no_ext.with_suffix(".tsv.gz")
        df.to_csv(p, sep="\t", index=False)
    return p


def read_table(path_no_ext: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a table written by write_table. Empty strings from the TSV fallback become None
    and *_missing columns become bool, so both formats load identically."""
    path_no_ext = Path(path_no_ext)
    pq = path_no_ext.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq, columns=columns)
    df = pd.read_csv(path_no_ext.with_suffix(".tsv.gz"), sep="\t", dtype=str,
                     keep_default_na=False, usecols=columns)
    for c in df.columns:
        if c.endswith("_missing"):
            df[c] = df[c] == "True"
        else:
            df[c] = df[c].astype(object).where(df[c] != "", None)
    return df
