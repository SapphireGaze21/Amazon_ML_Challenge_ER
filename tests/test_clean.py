"""
Unit tests for src/clean.py.  Run:  python tests/test_clean.py   (or: pytest tests/)
Each case is (input, expected). Add a case whenever you find a surprising record.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clean import clean_address, clean_country, clean_name, fold_form, orig_form  # noqa: E402

FOLD = [
    ("Payne Enterpires, LLC.", "payne enterpires llc"),
    ("P.V.T. Ltd.", "pvt ltd"),
    ("L.L.C", "llc"),
    ("St. John's Bakery", "st johns bakery"),           # 'st.' is not a dotted abbreviation
    ("McDonald’s", "mcdonalds"),                         # curly apostrophe deleted
    ("Smith & Sons", "smith and sons"),
    ("Smith&Sons", "smith and sons"),
    ("Café Dréxkor", "cafe drexkor"),
    ("Cœur de Lion SARL", "coeur de lion sarl"),
    ("STRASSE Straße", "strasse strasse"),
    ("Hewlett-Packard", "hewlett packard"),
    ("  lots   of\tspace  ", "lots of space"),
    ("#12, 3rd Cross", "12 3rd cross"),
    ("A+ Traders (P) Ltd", "a traders p ltd"),
    ("ＡＢＣ　Ｃｏｒｐ", "abc corp"),                      # full-width -> NFKC
    ("शर्मा ट्रेडर्स", "शर्मा ट्रेडर्स"),                          # Hindi vowel signs kept
    ("கிருஷ்ணா டெக்ஸ்டைல்ஸ்", "கிருஷ்ணா டெக்ஸ்டைல்ஸ்"),          # Tamil kept
    ("प्लॉट १२।", "प्लॉट 12"),                              # Devanagari digits + danda
    ("क्\u200dष", "क्ष"),                                    # zero-width joiner removed
]

ORIG = [
    ("Café  Dréxkor,  LLC.", "café dréxkor, llc."),
    ("प्लॉट १२", "प्लॉट १२"),
]

NAME_MISSING = ["", "   ", "null", "NULL", "N/A", "n.a.", "NA", "-", "--", "?", "...", "Unknown",
                "Not Available"]
NAME_PRESENT = ["Nana Traders", "A", "NAN Corp", "7-Eleven"]

ADDRESS_COMPOUNDS = [
    ("Plot 12-3-456, Banjara Hills", "12-3-456"),
    ("12/3A MG Road", "12-3a"),
    ("Flat 4 / 7, Block 2-B", "4-7"),                    # "2-B": letter after hyphen, not digits
    ("१२/३ गांधी नगर", "12-3"),
    ("1600 Pennsylvania Ave", None),
    ("Suite 100, 200-300 Main", "200-300"),
]


def run():
    fails = 0

    def check(label, got, exp):
        nonlocal fails
        if got != exp:
            fails += 1
            print(f"FAIL {label}: expected {exp!r}, got {got!r}")

    for s, exp in FOLD:
        check(f"fold({s!r})", fold_form(s), exp)
    for s, exp in ORIG:
        check(f"orig({s!r})", orig_form(s), exp)
    for s in NAME_MISSING:
        check(f"missing({s!r})", clean_name(s), (None, None, True))
    for s in NAME_PRESENT:
        check(f"present({s!r})", clean_name(s)[2], False)
    for s, exp in ADDRESS_COMPOUNDS:
        check(f"compound({s!r})", clean_address(s)[2], exp)
    check("country", clean_country("  France "), ("france", False))
    check("country missing", clean_country("null"), (None, True))
    check("address missing", clean_address("NULL"), (None, None, None, True))

    total = len(FOLD) + len(ORIG) + len(NAME_MISSING) + len(NAME_PRESENT) + len(ADDRESS_COMPOUNDS) + 3
    print(f"{total - fails}/{total} passed")
    return fails


def test_all():  # pytest entry point
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
