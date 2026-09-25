"""
Offline romanizer for the nine major Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati,
Oriya, Tamil, Telugu, Kannada, Malayalam). No external library, no network, no license issue.

Why custom: these nine Unicode blocks share one layout (inherited from ISCII) - the letter
"ka" sits at offset 0x15 in every block - so one table covers all of them. The output is
tuned to look like everyday Indian English spellings rather than scholarly transliteration:
  - long and short vowels collapse (आ/अ -> a, ई/इ -> i), because "Balaji" is written for बालाजी
  - retroflex and dental letters collapse (ट/त -> t, ड/द -> d), same reason
  - श/ष -> sh, व -> v, anusvara -> n
  - schwa deletion for Hindi-like scripts: the inherent "a" is dropped at the end of a word
    (राम -> ram, not rama) and in the common V-C-a-C-V middle position (कमला -> kamla).
    Dravidian scripts and Oriya write vowels explicitly, so no deletion is applied there.

It is approximate by design: its job is to make the Latin and native-script versions of the
same name share characters and phonetic keys, not to be linguistically perfect.
Characters from any other script are left unchanged.
"""

# block start -> (script, apply schwa deletion)
_BLOCKS = {
    0x0900: ("devanagari", True), 0x0980: ("bengali", True), 0x0A00: ("gurmukhi", True),
    0x0A80: ("gujarati", True), 0x0B00: ("oriya", False), 0x0B80: ("tamil", False),
    0x0C00: ("telugu", False), 0x0C80: ("kannada", False), 0x0D00: ("malayalam", False),
}

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # precomposed nukta letters (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ and Bengali ড় ঢ় য়)
    0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
# consonants that never carry a vowel: Bengali khanda ta, Malayalam chillu letters
_DEAD_CONS = {0x4E: "t", 0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}
_INDEP_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "ri", 0x61: "li",
}
_MATRAS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x62: "li", 0x63: "li",
}
_CODA = {0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}  # candrabindu, anusvara, visarga, tippi
_VIRAMA, _NUKTA = 0x4D, 0x3C
_VOWEL_CARRIERS = {0x72, 0x73}  # Gurmukhi iri / ura
_NUKTA_MAP = {"j": "z", "ph": "f", "d": "r", "dh": "rh"}
_IGNORE = {0x3D, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x71, 0x64, 0x65}  # avagraha, accents,
# length marks (e.g. Tamil au mark), Gurmukhi addak, dandas (dandas are punctuation anyway)


def _block_of(cp: int):
    base = cp & ~0x7F
    return (base, _BLOCKS[base]) if base in _BLOCKS else (None, None)


class _Unit:
    __slots__ = ("cons", "vowel", "coda", "off")

    def __init__(self, cons, vowel, off=None):
        self.cons = cons      # "" for an independent vowel
        self.vowel = vowel    # None = inherent 'a'; "" = no vowel (virama); else explicit
        self.coda = ""
        self.off = off        # block offset of the consonant


_FINAL_ANUSVARA_M = {"malayalam", "telugu", "kannada"}


def _romanize_word(cps: list[int], schwa: bool, script: str = "") -> str:
    units: list[_Unit] = []
    for cp in cps:
        off = cp & 0x7F
        if off in _CONS:
            u = _Unit(_CONS[off], None, off)
            # Malayalam റ്റ (ṟ + virama + ṟ) is pronounced "tt"
            if (script == "malayalam" and off == 0x31 and units and units[-1].off == 0x31
                    and units[-1].vowel == ""):
                units[-1].cons, u.cons = "t", "t"
            units.append(u)
        elif off in _DEAD_CONS:
            units.append(_Unit(_DEAD_CONS[off], ""))
        elif off in _INDEP_VOWELS:
            units.append(_Unit("", _INDEP_VOWELS[off]))
        elif off in _VOWEL_CARRIERS:
            units.append(_Unit("", "a"))
        elif off in _MATRAS:
            if units and units[-1].cons and units[-1].vowel is None:
                units[-1].vowel = _MATRAS[off]
            elif units and not units[-1].cons:   # matra on a vowel carrier
                units[-1].vowel = _MATRAS[off]
            else:
                units.append(_Unit("", _MATRAS[off]))
        elif off == _VIRAMA:
            if units and units[-1].cons:
                units[-1].vowel = ""
        elif off == _NUKTA:
            if units and units[-1].cons in _NUKTA_MAP:
                units[-1].cons = _NUKTA_MAP[units[-1].cons]
        elif off in _CODA:
            if units:
                units[-1].coda += "N" if off in (0x01, 0x02, 0x70) else _CODA[off]
            else:
                units.append(_Unit("", _CODA[off]))
        elif off == 0x50:  # om
            units.append(_Unit("", "om"))
        # anything else (_IGNORE or unassigned) is skipped

    if schwa and len(units) > 1:
        n = len(units)
        # medial: V C(a) C V  ->  delete the 'a'   (कमला -> kamla, कलकाता -> kalkata)
        for i in range(1, n - 1):
            u, prev = units[i], units[i - 1]
            if not (u.cons and u.vowel is None and not u.coda and prev.vowel != "" and not prev.coda):
                continue
            nxt = units[i + 1]              # only before a SINGLE consonant + vowel
            if nxt.cons and nxt.vowel not in (None, ""):   # (राजस्थान keeps: rajasthan)
                u.vowel = ""
        # final: राम -> ram (but a single-letter word keeps its vowel)
        last, prev = units[-1], units[-2]
        after_cluster = bool(prev.cons) and prev.vowel == "" and last.cons in ("r", "y", "v", "n", "l", "m")
        if last.cons and last.vowel is None and not last.coda and not after_cluster:
            last.vowel = ""

    # nasal sign: "m" before p/b/m and word-finally in Malayalam/Telugu/Kannada, else "n"
    for i, u in enumerate(units):
        if "N" in u.coda:
            nxt = units[i + 1].cons if i + 1 < len(units) else None
            m = (nxt is not None and nxt[:1] in ("p", "b", "m")) or \
                (nxt is None and script in _FINAL_ANUSVARA_M)
            u.coda = u.coda.replace("N", "m" if m else "n")

    out = []
    for u in units:
        out.append(u.cons)
        out.append("a" if u.vowel is None else u.vowel)
        out.append(u.coda)
    return "".join(out)


def romanize(s: str | None) -> str | None:
    """Romanize Indic-script runs inside s; everything else is kept as-is."""
    if not isinstance(s, str):
        return None
    if s.isascii():
        return s
    out, run, run_block = [], [], None
    for ch in s:
        cp = ord(ch)
        base, info = _block_of(cp)
        if info is not None:
            if run and base != run_block:
                out.append(_romanize_word(run, _BLOCKS[run_block][1], _BLOCKS[run_block][0]))
                run = []
            run.append(cp)
            run_block = base
        else:
            if run:
                out.append(_romanize_word(run, _BLOCKS[run_block][1], _BLOCKS[run_block][0]))
                run, run_block = [], None
            out.append(ch)
    if run:
        out.append(_romanize_word(run, _BLOCKS[run_block][1], _BLOCKS[run_block][0]))
    return " ".join("".join(out).split())


def has_unsupported_script(s: str | None) -> bool:
    """True if s still contains non-Latin letters after romanization (e.g. Arabic, CJK)."""
    if not isinstance(s, str) or s.isascii():
        return False
    return any(ch.isalpha() and not ch.isascii() for ch in s)
