"""
Unit tests for src/romanize.py and src/represent.py.
Run:  python tests/test_represent.py   (or: pytest tests/)
Cases marked "P1"/"P2" come straight from patterns found in the Phase 1 / Phase 2 reports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from represent import (LEGAL_SUFFIX_HAND, REPR_COLUMNS, SubstitutionMiner,  # noqa: E402
                       build_repr, deleet, init_config, merge_maps, merge_single_letters,
                       phonetic_key, split_core, split_glued, suffix_like)
from romanize import romanize  # noqa: E402

ROMAN = [
    ("शर्मा", "sharma"), ("कृष्णा", "krishna"), ("बालाजी", "balaji"), ("राम", "ram"),
    ("कमला", "kamla"), ("जैन", "jain"), ("गुप्ता", "gupta"), ("संजय", "sanjay"),
    ("कंपनी", "kampani"), ("अंबानी", "ambani"), ("फ़ोन", "fon"), ("लक्ष्मी", "lakshmi"),
    ("प्राइवेट लिमिटेड", "praivet limited"),
    ("राजस्थान", "rajasthan"), ("महाराष्ट्र", "maharashtra"), ("मित्र", "mitra"),  # P1 states
    ("दोस्त", "dost"), ("ट्रेडर्स", "tredars"),
    ("சென்னை", "chennai"), ("ஸ்ரீ", "sri"), ("கிருஷ்ணா", "kirushna"),        # Tamil
    ("కృష్ణ", "krishna"), ("తెలంగాణ", "telangana"),                           # Telugu
    ("ಬೆಂಗಳೂರು", "bengaluru"), ("ಕರ್ನಾಟಕ", "karnataka"),                        # Kannada
    ("തിരുവനന്തപുരം", "tiruvanantapuram"),                                     # Malayalam
    ("কলকাতা", "kalkata"),                                                    # Bengali
    ("ਸਿੰਘ", "singh"), ("ਗੁਰਪ੍ਰੀਤ", "guraprit"),                                # Gurmukhi
    ("પટેલ", "patel"),                                                        # Gujarati
    ("sharma शर्मा traders", "sharma sharma traders"), ("plain ascii", "plain ascii"),
]

PHON = [
    ("sharma", "srm"), ("sarma", "srm"), ("krishna", "krsn"), ("kirushna", "krsn"),
    ("enterprises", "antrprs"), ("enterpires", "antrprs"), ("payne", "pn"), ("paine", "pn"),
    ("balaji", "blj"), ("baalaji", "blj"), ("wadhwa", "vdv"), ("vadhva", "vdv"),
    ("chennai", "cn"), ("gupta", "gpt"), ("guptha", "gpt"), ("ab", None), ("12b", None),
    ("cie", None),                                   # 1-letter key is useless
]

CORE = [  # tokens -> (core, removed_legal, removed_honorific)
    (["acme", "corp"], (["acme"], ["corp"], [])),
    (["sharma", "traders", "pvt", "ltd"], (["sharma", "traders"], ["pvt", "ltd"], [])),
    (["smith", "and", "sons", "llc"], (["smith", "sons"], ["llc"], [])),
    (["the", "acme", "company"], (["acme"], ["company"], [])),
    (["llc", "vision", "partners", "of", "jamaica"],                              # P2
     (["vision", "partners", "jamaica"], ["llc"], [])),                        # P3: "of" dropped
    (["seven", "consultancy", "limited", "service"],                              # P2
     (["seven", "consultancy", "service"], ["limited"], [])),
    (["shri", "balaji", "traders", "limited"], (["balaji", "traders"], ["limited"], ["shri"])),  # P1
    (["dr", "mehta", "clinic"], (["mehta", "clinic"], [], ["dr"])),
    (["limited"], (["limited"], [], [])),                                         # never empty
    (["boulangerie", "dupont", "sarl"], (["boulangerie", "dupont"], ["sarl"], [])),
]

MISC = [
    (merge_single_letters, ["smith", "p", "c"], ["smith", "pc"]),                 # P1 "c"
    (merge_single_letters, ["l", "l", "c"], ["llc"]),
    (merge_single_letters, ["12", "n", "main"], ["12", "n", "main"]),
    (split_glued, ["no238", "apartment1st", "401a", "94th"], ["no", "238", "apartment", "1st", "401a", "94th"]),
    (deleet, "ass0ciates", "associates"), (deleet, "5ecure", "secure"),           # P1 typos
    (deleet, "6roup", "group"), (deleet, "l1c", "llc"), (deleet, "denta1", "dental"),
    (deleet, "1st", "1st"), (deleet, "3m", "3m"), (deleet, "b3", "b3"), (deleet, "2024", "2024"),
]


def run():
    fails, total = 0, 0

    def check(label, got, exp):
        nonlocal fails, total
        total += 1
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    for s, exp in ROMAN:
        check(f"romanize({s!r})", romanize(s), exp)
    for s, exp in PHON:
        check(f"phonetic_key({s!r})", phonetic_key(s), exp)
    for toks, exp in CORE:
        check(f"split_core({toks})", split_core(toks, LEGAL_SUFFIX_HAND), exp)
    for fn, arg, exp in MISC:
        check(f"{fn.__name__}({arg!r})", fn(arg), exp)
    # P3: discovery must keep abbreviations / legal variants and reject descriptors
    for t in ("pa", "ei", "pra", "limtid", "elelpi"):
        check(f"suffix_like({t})", suffix_like(t, LEGAL_SUFFIX_HAND), True)
    for t in ("bakery", "church", "school", "trust", "jean", "sport", "shop", "samiti"):
        check(f"not suffix_like({t})", suffix_like(t, LEGAL_SUFFIX_HAND), False)
    mm, dropped = merge_maps({"keralam": "kerala"}, {"kerala": "keralam", "mh": "maharashtra"})
    check("merge_maps no reversal", mm, {"keralam": "kerala", "mh": "maharashtra"})
    check("merge_maps chain", merge_maps({}, {"a": "b", "b": "c"})[0], {"a": "c", "b": "c"})
    check("french articles", split_core(["comite", "des", "fetes", "de", "lille"], LEGAL_SUFFIX_HAND)[0],
          ["comite", "fetes", "lille"])
    for native in ("elaelpi", "limitet", "praivet"):                              # P1 native legal
        check(f"legal list has {native}", native in LEGAL_SUFFIX_HAND, True)

    init_config(None)  # hand lists only

    def rep(n, a, c):
        return dict(zip(REPR_COLUMNS, build_repr(n, a, c)))

    r = rep("शर्मा ट्रेडर्स प्राइवेट लिमिटेड", "प्लॉट 12 mg rd 500034 दिल्ली", "india")
    check("repr name_script", r["name_script"], "devanagari")
    check("repr name_core", r["name_core"], "sharma tredars")
    check("repr name_suffix", r["name_suffix"], "praivet limited")
    check("repr name_phon", r["name_phon"], "srm trdrs")
    check("repr address_tokens_norm", r["address_tokens_norm"], "plot 12 mg road 500034 delhi")
    check("repr address_nums", r["address_nums"], "12 500034")
    r = rep("ciramira fka riva ltd", "00606 94th ct null", "us")                  # P2
    check("alias", (r["name_sorted_key"], r["name_alias"]), ("ciramira", "riva"))
    check("address zeros/null", r["address_tokens_norm"], "606 94th court")
    r = rep("earnosethroat com", None, "us")                                      # P2
    check("com removed", (r["name_core"], r["name_concat"]), ("earnosethroat", "earnosethroat"))
    r = rep("lille club sarl", "27 r jean bart lille", "france")                  # P2
    check("initials", r["name_initials"], "lc")
    check("french r", r["address_tokens_norm"], "27 rue jean bart lille")
    check("repr missing", build_repr(None, None, "us"), (None,) * len(REPR_COLUMNS))
    check("repr NaN is missing", build_repr(float("nan"), None, "us"), (None,) * len(REPR_COLUMNS))

    m = SubstitutionMiner()
    for _ in range(30):
        m.add("us", ["12", "main", "st"], ["12", "main", "street"])
        m.add_phrase("us", ["12", "main", "st", "raleigh", "north", "carolina"],
                     ["12", "main", "st", "raleigh", "nc"])
    m.add("us", ["enterpires"], ["enterprises"])                  # typo: same length, ignored
    maps, _ = m.result(min_count=20, min_share=0.6)
    check("miner", maps, {"us": {"st": "street"}})
    pmaps, _ = m.phrase_result(min_count=20)
    check("phrase miner", pmaps, {"us": {("north", "carolina"): "nc"}})
    init_config({"phrase_address": {"us": {("north", "carolina"): "nc"}}})
    check("phrase applied", rep("x", "raleigh north carolina", "us")["address_tokens_norm"], "raleigh nc")

    print(f"{total - fails}/{total} passed")
    return fails


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
