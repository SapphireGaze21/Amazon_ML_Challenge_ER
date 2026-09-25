"""
Phase 2 cleaning functions (pure, no pandas) - used by phase2_clean.py and by every later
phase, so train, validation and test text is cleaned by exactly the same code.

For each name / address, two cleaned variants are produced:

  *_orig : NFKC + casefold + whitespace collapsed. Accents, punctuation and native digits
           are KEPT. Used for features (e.g. accent-sensitive similarity).
  *_fold : the matching/blocking form. On top of *_orig:
             & -> " and "; dotted abbreviations joined (l.l.c. -> llc, p.v.t. -> pvt);
             apostrophes deleted (mcdonald's -> mcdonalds); invisible format characters
             deleted (zero-width joiners, soft hyphens); accents stripped from LATIN letters
             only (é -> e, œ -> oe) - Indic vowel signs are untouched; all digits -> ASCII
             (१२ -> 12); every other punctuation / symbol -> space; whitespace collapsed.

For addresses, numeric compounds such as "12-3-456", "12/3A", "4 / 7" are also captured
BEFORE punctuation is removed (normalised to "12-3-456", "12-3a", "4-7"), because splitting
them into separate small numbers loses their specificity.

A field is missing when it is empty, a placeholder ("null", "n/a", "na", "-", "unknown", ...)
or becomes empty after cleaning (e.g. only punctuation). Missing fields are None.
"""

import re
import sys
import unicodedata

# ----------------------------------------------------------------------------- tables
# Built once at import (~1 s). str.translate with these runs at C speed.

_APOSTROPHES = "'\u2019\u2018\u02bc\u0060\u00b4\u2032"
_LATIN_COMBINING = [(0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF),
                    (0x20D0, 0x20FF), (0xFE20, 0xFE2F)]
# Latin letters that NFD does not decompose
_SPECIAL_LATIN = {"œ": "oe", "æ": "ae", "ø": "o", "ł": "l", "đ": "d", "ð": "d",
                  "þ": "th", "ı": "i", "ŧ": "t", "ħ": "h"}


def _build_tables():
    delete = {ord(c): None for c in _APOSTROPHES}
    to_space, to_ascii_digit = {}, {}
    for cp in range(0x30000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat == "Cf":                       # zero-width joiner/non-joiner, BOM, bidi marks
            delete[cp] = None
        elif cat[0] in "PS" or cat in ("Zs", "Zl", "Zp", "Cc"):
            to_space[cp] = " "
        elif cat == "Nd" and not ch.isascii():
            to_ascii_digit[cp] = str(unicodedata.digit(ch))
    for c in _APOSTROPHES:                   # deletion wins over "punctuation -> space"
        to_space.pop(ord(c), None)
    strip_marks = {cp: None for lo, hi in _LATIN_COMBINING for cp in range(lo, hi + 1)}
    strip_marks.update({ord(k): v for k, v in _SPECIAL_LATIN.items()})
    return delete, to_space, to_ascii_digit, strip_marks


_DELETE, _TO_SPACE, _TO_ASCII_DIGIT, _STRIP_LATIN_MARKS = _build_tables()

# single letters separated by dots: l.l.c. / u.s.a / p.v.t.  (not "st." or "no.")
_DOTTED_ABBR = re.compile(r"(?<!\w)[^\W\d_](?:\.[^\W\d_])+\.?(?!\w)")
# numeric compounds: 12-3-456, 12/3a, 4 / 7, 1-2-3/4
_NUM_COMPOUND = re.compile(r"(?<![\w])\d+[a-z]?(?:\s*[-/]\s*\d+[a-z]?)+(?![\w])")

# placeholders, compared AFTER folding ("N/A" folds to "n a", "N.A." to "na")
PLACEHOLDERS_FOLDED = {
    "", "null", "none", "nan", "na", "n a", "nil", "unknown", "not available",
    "not applicable", "no address", "address not available", "not known", "tbd",
}


# ----------------------------------------------------------------------------- core

def orig_form(s: str) -> str:
    """NFKC + casefold + collapsed whitespace. Punctuation, accents, native digits kept."""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _pre_fold(s: str) -> str:
    """Steps shared by fold_form and numeric-compound extraction."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.translate(_DELETE)                              # apostrophes, format chars
    if not s.isascii():
        s = unicodedata.normalize("NFD", s).translate(_STRIP_LATIN_MARKS)
        s = unicodedata.normalize("NFC", s).translate(_TO_ASCII_DIGIT)
    return s


def _finish_fold(s: str) -> str:
    s = s.replace("&", " and ")
    s = _DOTTED_ABBR.sub(lambda m: m.group(0).replace(".", ""), s)
    return " ".join(s.translate(_TO_SPACE).split())


def fold_form(s: str) -> str:
    return _finish_fold(_pre_fold(s))


def numeric_compounds(pre_folded: str) -> list[str]:
    out = []
    for m in _NUM_COMPOUND.finditer(pre_folded):
        c = re.sub(r"\s+", "", m.group(0)).replace("/", "-")
        if c not in out:
            out.append(c)
    return out


# ----------------------------------------------------------------------------- per field

def clean_name(raw: str):
    """-> (orig, fold, missing)"""
    if raw is None:
        return None, None, True
    f = fold_form(raw)
    if f in PLACEHOLDERS_FOLDED:
        return None, None, True
    return orig_form(raw), f, False


def clean_address(raw: str):
    """-> (orig, fold, numeric_compounds_space_joined_or_None, missing)"""
    if raw is None:
        return None, None, None, True
    pre = _pre_fold(raw)
    f = _finish_fold(pre)
    if f in PLACEHOLDERS_FOLDED:
        return None, None, None, True
    comps = numeric_compounds(pre)
    return orig_form(raw), f, (" ".join(comps) if comps else None), False


def clean_country(raw: str):
    """-> (country_norm, missing). Trim + casefold only; open set, no value mapping."""
    if raw is None:
        return None, True
    c = raw.strip().casefold()
    if " ".join(c.split()) in PLACEHOLDERS_FOLDED:
        return None, True
    return c, False


# chunk workers for multiprocessing (must be module-level to be picklable)
def clean_name_chunk(chunk):
    return [clean_name(x) for x in chunk]


def clean_address_chunk(chunk):
    return [clean_address(x) for x in chunk]


if __name__ == "__main__":  # quick manual check: python src/clean.py "Some text"
    for arg in sys.argv[1:]:
        print(repr(arg), "->", clean_name(arg), clean_address(arg))
