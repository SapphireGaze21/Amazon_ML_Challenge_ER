"""
Phase 3 representation functions (pure Python; used by phase3_represent.py and later phases).

Per record, from the Phase 2 *_fold columns:
  name_script, address_script      dominant script of the folded text (latin/devanagari/tamil/...)
  name_roman, address_roman        Indic runs romanized (romanize.py); identical to *_fold for Latin
  name_tokens                      tokens of name_roman
  name_core                        distinctive name tokens: connectors (and/et/the), alias markers
                                   (dba/fka/aka), leading honorifics (mr/dr/smt/shri/sri) and legal
                                   words (anywhere) removed; digit-for-letter typos repaired
                                   (5ecure -> secure, l1c -> llc); abbreviations expanded
  name_suffix                      removed legal words (feature)       name_prefix: removed honorifics
  name_sorted_key                  sorted unique core tokens -> handles word-order transpositions
  name_alias                       sorted core of the part after dba/fka/aka, else None
  name_concat                      core tokens glued together (matches squashed names like
                                   "earnosethroat com"); name_initials: first letters of core ("cc")
  name_phon                        phonetic keys of core tokens (custom key, see phonetic_key)
  address_tokens                   tokens of address_roman (glued "no238" split, single letters
                                   "s v road" merged to "sv road")
  address_tokens_norm              same tokens normalised: "null"/"na" dropped, leading zeros
                                   stripped, multi-word states -> initials (north carolina -> nc),
                                   abbreviations/aliases expanded (rd -> road, dilli -> delhi)
  address_nums                     generic numbers (digit runs), unique, in order
All multi-valued fields are space-joined strings (tokens never contain spaces); None if missing.
Character n-grams are NOT stored: they are derived on the fly from name_core / address_roman
when the TF-IDF stage needs them (storing 3/4/5-grams for 24M records would be enormous).

The legal-suffix sets and abbreviation maps are mined from data by phase3_represent.py and
passed in through init_config(); the hand lists below are the fallback for any country
(including ones never seen in training).
"""

import re
from collections import Counter, defaultdict

from common import numeric_tokens, script_of, tokens
from romanize import romanize

# ----------------------------------------------------------------------------- hand lists
# Legal forms, multi-language. Removed wherever they occur (Phase 2 examples showed legal words
# in the middle of names: "LLC Vision Partners", "Atlantic Inc Pimco", "Consultancy Limited Service").
LEGAL_SUFFIX_HAND = {
    # English / US
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "lp", "lllp",
    "ltd", "limited", "plc", "pllc", "pc", "pty",
    # India
    "pvt", "private", "opc", "pvtltd",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "sca", "scop",
    "eirl", "sem", "cie",
    # website-style names ("earnosethroat com") - Phase 1: "com" is a top last token everywhere
    "com",
}
# Native-script legal words seen in Phase 1 (लिमिटेड, లిమిటెడ్ ...). Their romanized forms are
# derived from romanize() itself so the list always matches what the romanizer outputs.
NATIVE_LEGAL = ["लिमिटेड", "प्राइवेट", "एलएलपी", "कंपनी", "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್", "లిమిటెడ్",
                "ప్రైవేట్", "லிமிடெட்", "பிரைவேட்", "লিমিটেড", "প্রাইভেট", "લિમિટેડ", "પ્રાઇવેટ",
                "ലിമിറ്റഡ്", "പ്രൈവറ്റ്", "ਲਿਮਿਟੇਡ", "ਪ੍ਰਾਈਵੇਟ", "ଲିମିଟେଡ", "ପ୍ରାଇଭେଟ"]
LEGAL_SUFFIX_HAND |= {t for w in NATIVE_LEGAL for t in romanize(w).split() if len(t) >= 3}

# Phase 1: legal words were often swapped for honorifics ("ltd -> smt/mr/shri/sri/dr").
# Removed only at the START of a name.
HONORIFICS = {"mr", "mrs", "ms", "dr", "smt", "shri", "sri", "shree", "sree", "messrs", "mme",
              "mlle", "kumari", "km"}
# Phase 3 report: "of" was a top-5 US core token, "de"/"du" top French ones
CONNECTORS = {"and", "et", "the", "und", "of", "de", "du", "des", "la", "le", "les"}
ALIAS_MARKERS = {"dba", "fka", "aka", "formerly"}

ABBR_ADDRESS_HAND = {
    "rd": "road", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "hwy": "highway", "ln": "lane", "ct": "court", "pl": "place",
    "sq": "square", "pkwy": "parkway", "cir": "circle", "apt": "apartment",
    "bldg": "building", "nr": "near", "opp": "opposite",
    # French (test-only country; "27 r jean bart" vs "27 rue jean bart" in Phase 2 top values)
    "r": "rue", "fbg": "faubourg", "imp": "impasse", "rte": "route", "chem": "chemin",
    # ordinals (Phase 1: 2nd -> second, 4th -> fourth ...)
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    # place-name aliases the romanizer cannot fix (दिल्ली -> dilli, ਪੰਜਾਬ -> panjab ...)
    "dilli": "delhi", "panjab": "punjab", "hariyana": "haryana", "orisha": "odisha",
    "orissa": "odisha", "gujrat": "gujarat", "keralam": "kerala", "bengaluru": "bangalore",
    "bengalooru": "bangalore", "bombay": "mumbai", "calcutta": "kolkata", "kalkata": "kolkata",
    "kalakatta": "kolkata", "madras": "chennai", "pondicherry": "puducherry",
    "trivandrum": "thiruvananthapuram", "tiruvanantapuram": "thiruvananthapuram",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "mysore": "mysuru",
    "mangalore": "mangaluru", "banaras": "varanasi", "benares": "varanasi",
}
ABBR_NAME_HAND = {
    "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service",
    "mgmt": "management", "assoc": "associates", "bros": "brothers", "natl": "national",
    "grp": "group", "engg": "engineering", "hldgs": "holdings",
}
ADDRESS_DROP = {"null", "na", "none", "nan", "nil"}  # Phase 1: "city -> null" substitutions

# ----------------------------------------------------------------------------- config
_EMPTY = {"suffix": {}, "abbr_name": {}, "abbr_address": {}, "phrase_address": {}}
_CFG = dict(_EMPTY)


def init_config(cfg: dict | None):
    """cfg = {"suffix": {country: set}, "abbr_name": {country: dict},
              "abbr_address": {country: dict}, "phrase_address": {country: {(w1, w2): initials}}}.
    Also used as the multiprocessing Pool initializer."""
    global _CFG
    _CFG = {**_EMPTY, **(cfg or {})}


def _suffixes(country):
    return _CFG["suffix"].get(country, LEGAL_SUFFIX_HAND)


def _phrases(country):
    return _CFG["phrase_address"].get(country, {})


def _abbr(kind, country):
    hand = ABBR_NAME_HAND if kind == "name" else ABBR_ADDRESS_HAND
    return _CFG[f"abbr_{kind}"].get(country, hand)


# ----------------------------------------------------------------------------- phonetic key
_PHON_SUBS = [("tch", "C"), ("chh", "C"), ("sch", "s"), ("ch", "C"), ("sh", "s"), ("ph", "f"),
              ("gh", "g"), ("kh", "k"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("jh", "j"),
              ("ck", "k"), ("wh", "v"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s")]
_VOWELS = set("aeiouy")


def phonetic_key(tok: str) -> str | None:
    """
    Custom phonetic key aimed at the variation seen in Indian and English business names:
    aspirates (bh/dh/kh -> b/d/k), sh/s, ph/f, w/v, z/s, soft/hard c, doubled letters and
    all vowels after the first letter. Examples:
      sharma, sarma -> srm      krishna, kirushna (Tamil romanized) -> krsn
      enterprises, enterpires -> antrprs      payne, paine -> pn
    Double Metaphone was considered; it is tuned to English/European names and does not
    merge the Indian transliteration variants above. Returns None for tokens that are too
    short or not purely alphabetic ASCII.
    """
    if len(tok) < 3 or not tok.isascii() or not tok.isalpha():
        return None
    t = tok
    for a, b in _PHON_SUBS:
        t = t.replace(a, b)
    out = []
    for i, ch in enumerate(t):
        if ch == "c":
            nxt = t[i + 1] if i + 1 < len(t) else ""
            ch = "s" if nxt in ("e", "i", "y") else "k"
        out.append(ch)
    t = "".join(out).lower()           # the 'C' placeholder (ch) becomes 'c'
    first = "a" if t[0] in _VOWELS else t[0]
    rest = [c for c in t[1:] if c not in _VOWELS and c != "h"]
    key = [first]
    for c in rest:
        if c != key[-1]:
            key.append(c)
    return "".join(key) if len(key) >= 2 else None


# ----------------------------------------------------------------------------- token helpers

def merge_single_letters(toks: list[str]) -> list[str]:
    """Join runs of 2+ single letters: "p c" -> "pc", "l l c" -> "llc", "s v road" -> "sv road".
    Phase 2's dotted-abbreviation rule only caught "p.c."; spaced "P. C." left single letters
    (Phase 1: "c" was a top-5 last name token in US S1)."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


_GLUE = re.compile(r"(?<=[a-z])(?=[0-9])")


def split_glued(toks: list[str]) -> list[str]:
    """Split letter->digit joins in addresses: "no238" -> "no 238", "apartment1st" -> ... """
    out = []
    for t in toks:
        out.extend(p for p in _GLUE.split(t) if p)
    return out


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t",
                       "8": "b", "9": "g"})
_ORDINAL = re.compile(r"^[0-9]+(st|nd|rd|th)$")


def deleet(tok: str) -> str:
    """Phase 1 found digit-for-letter typos in names: ass0ciates, 5ecure, 6roup, denta1, l1c.
    Repaired only in mostly-alphabetic tokens (keeps 3m, b3, 1st as they are)."""
    if tok.isalpha() or tok.isdigit() or _ORDINAL.match(tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    if letters >= 2 and letters > digits:
        return tok.translate(_LEET)
    return tok


def name_token_list(name_fold) -> list[str]:
    """Romanized, merged name tokens (no removal). Shared by mining and build_repr."""
    if not isinstance(name_fold, str):
        return []
    return merge_single_letters(tokens(romanize(name_fold)))


def address_token_list(address_fold) -> list[str]:
    if not isinstance(address_fold, str):
        return []
    return merge_single_letters(split_glued(tokens(romanize(address_fold))))


def name_mining_tokens(name_fold) -> list[str]:
    return [t for t in name_token_list(name_fold) if t not in CONNECTORS and t not in ALIAS_MARKERS]


# ----------------------------------------------------------------------------- core name

def split_core(toks: list[str], suffixes: set):
    """-> (core, removed_legal, removed_honorifics). Connectors dropped anywhere, honorifics
    only at the start, legal words anywhere. Never returns an empty core."""
    body = [t for t in toks if t not in CONNECTORS and t not in ALIAS_MARKERS] or list(toks)
    i, prefix = 0, []
    while i < len(body) - 1 and body[i] in HONORIFICS:
        prefix.append(body[i])
        i += 1
    rest = body[i:]
    core = [t for t in rest if t not in suffixes]
    removed = [t for t in rest if t in suffixes]
    if not core:
        return rest, [], prefix
    return core, removed, prefix


def _dedup(seq):
    return list(dict.fromkeys(seq))


def _core_pipeline(toks, country):
    toks = [deleet(t) for t in toks]
    core, removed, prefix = split_core(toks, _suffixes(country))
    amap = _abbr("name", country)
    return [amap.get(t, t) for t in core], removed, prefix


def normalize_address_tokens(toks: list[str], country) -> list[str]:
    toks = [t for t in toks if t not in ADDRESS_DROP]
    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in toks]
    phrases = _phrases(country)
    if phrases:
        out, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and (toks[i], toks[i + 1]) in phrases:
                out.append(phrases[(toks[i], toks[i + 1])])
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
    amap = _abbr("address", country)
    return [amap.get(t, t) for t in toks]


# ----------------------------------------------------------------------------- per record

def build_repr(name_fold, address_fold, country):
    """-> tuple in the order of REPR_COLUMNS. Any non-string input counts as missing."""
    if not isinstance(country, str):
        country = None
    if not isinstance(name_fold, str):
        n = (None,) * 11
    else:
        toks = name_token_list(name_fold)
        alias_toks = None
        for k, t in enumerate(toks):
            if t in ALIAS_MARKERS:
                main, alias_toks = toks[:k], toks[k + 1:]
                break
        else:
            main = toks
        core_main, removed, prefix = _core_pipeline(main or toks, country)
        core = core_main
        alias = None
        if alias_toks:
            core_alias, rem2, _ = _core_pipeline(alias_toks, country)
            removed = removed + rem2
            core = _dedup(core_main + core_alias)
            alias = " ".join(sorted(set(core_alias))) or None
        phon = _dedup(k for k in (phonetic_key(t) for t in core) if k)
        initials = "".join(t[0] for t in core if t[:1].isalpha()) if len(core) >= 2 else None
        n = (script_of(name_fold), romanize(name_fold), " ".join(toks), " ".join(core),
             " ".join(removed) or None, " ".join(prefix) or None,
             " ".join(sorted(set(core_main))), alias, "".join(core_main) or None,
             initials, " ".join(phon) or None)
    if not isinstance(address_fold, str):
        a = (None,) * 5
    else:
        a_roman = romanize(address_fold)
        atoks = address_token_list(address_fold)
        nums = _dedup(numeric_tokens(a_roman))
        a = (script_of(address_fold), a_roman, " ".join(atoks),
             " ".join(normalize_address_tokens(atoks, country)), " ".join(nums) or None)
    return n + a


REPR_COLUMNS = ["name_script", "name_roman", "name_tokens", "name_core", "name_suffix",
                "name_prefix", "name_sorted_key", "name_alias", "name_concat", "name_initials",
                "name_phon",
                "address_script", "address_roman", "address_tokens", "address_tokens_norm",
                "address_nums"]


def build_repr_chunk(chunk):
    return [build_repr(*r) for r in chunk]


# ----------------------------------------------------------------------------- mining helpers

def merge_maps(hand: dict, mined: dict) -> tuple[dict, list]:
    """Merge hand + mined abbreviation maps safely.
    - a mined entry that reverses a hand entry is dropped (Phase 3 report: mined
      kerala->keralam vs hand keralam->kerala made the two spellings swap instead of merge)
    - chains a->b->c are collapsed to a->c; anything still cyclic is dropped.
    Returns (map, dropped_entries)."""
    m, dropped = dict(hand), []
    for s, l in mined.items():
        if s == l:
            continue
        if m.get(l) == s:
            dropped.append((s, l, "reverses hand entry"))
            continue
        m[s] = l
    out = {}
    for k, v in m.items():
        seen, cur = {k}, v
        while cur in m and cur not in seen:
            seen.add(cur)
            cur = m[cur]
        if cur in seen:
            dropped.append((k, v, "cycle"))
            continue
        out[k] = cur
    return out, dropped


def suffix_like(tok: str, hand: set) -> bool:
    """Could tok plausibly be a legal form? Abbreviation length (pa, od, ei, pra, li) or the same
    phonetic key as a known legal word of 4+ letters (limtid ~ limited, elelpi ~ elaelpi).
    Phase 3 showed that frequency/droppability alone also admits descriptors (bakery, church)."""
    if len(tok) <= 3:
        return True
    k = phonetic_key(tok)
    return k is not None and k in {phonetic_key(h) for h in hand if len(h) >= 4}

def is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


class SubstitutionMiner:
    """Counts single-token substitutions in true pairs and keeps abbreviation-like ones:
    shorter token, same first letter, a subsequence of the longer one (rd -> road)."""

    def __init__(self):
        self.pairs = defaultdict(Counter)    # country -> Counter[(short, long)]
        self.phrases = defaultdict(Counter)  # country -> Counter[((w1, w2), initials)]

    def add_phrase(self, country, a_list, b_list):
        """Two adjacent words on one side vs their initials on the other:
        "north carolina" <-> "nc", "tamil nadu" <-> "tn", "west bengal" <-> "wb"."""
        a, b = set(a_list), set(b_list)
        x, y = a - b, b - a
        if len(x) == 2 and len(y) == 1:
            seq, pair, short = a_list, x, next(iter(y))
        elif len(y) == 2 and len(x) == 1:
            seq, pair, short = b_list, y, next(iter(x))
        else:
            return
        for i in range(len(seq) - 1):
            if {seq[i], seq[i + 1]} == pair and seq[i] != seq[i + 1]:
                w = (seq[i], seq[i + 1])
                if short == w[0][:1] + w[1][:1]:
                    self.phrases[country][(w, short)] += 1
                return

    def phrase_result(self, min_count: int):
        maps, rows = {}, []
        for c, cnt in self.phrases.items():
            m = {}
            for (w, s), n in cnt.items():
                ok = n >= min_count
                rows.append((c, " ".join(w), s, n, ok))
                if ok:
                    m[w] = s
            maps[c] = m
        return maps, rows

    def add(self, country, a_toks, b_toks):
        a, b = set(a_toks), set(b_toks)
        x, y = a - b, b - a
        if len(x) != 1 or len(y) != 1:
            return
        s, l = sorted((next(iter(x)), next(iter(y))), key=len)
        if (len(s) < len(l) and s[0] == l[0] and not s.isdigit() and not l.isdigit()
                and is_subsequence(s, l)):
            self.pairs[country][(s, l)] += 1

    def result(self, min_count: int, min_share: float):
        maps, rows = {}, []
        for c, cnt in self.pairs.items():
            by_short = defaultdict(list)
            for (s, l), n in cnt.items():
                by_short[s].append((n, l))
            m = {}
            for s, lst in by_short.items():
                lst.sort(reverse=True)
                total = sum(n for n, _ in lst)
                n, l = lst[0]
                ok = n >= min_count and n / total >= min_share
                rows.append((c, s, l, n, total, round(n / total, 3), ok))
                if ok:
                    m[s] = l
            # avoid chains (a -> b while b -> c): drop mappings whose target is itself a key
            maps[c] = {s: l for s, l in m.items() if l not in m}
        return maps, rows
