"""Unit tests for Phase 7 Stage-2 feature primitives."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase7_stage2_features as S  # noqa: E402


def test_numbers_distinguish_house_and_postal():
    f = S.number_features("12 560001", "12 560002")
    assert f["house_num_agree"] == 1.0
    assert f["house_num_conflict"] == 0.0
    assert f["postal_agree"] == 0.0
    assert f["postal_conflict"] == 1.0


def test_generic_postal_is_longest_five_digit_number():
    assert S.longest_postal({"75001", "750010", "12"}) == "750010"


def test_empty_strings_are_no_evidence_not_perfect_fuzzy_match():
    f = S.similarities("name_core", "", "")
    assert all(v == 0.0 for v in f.values())


def test_idf_overlap_rewards_shared_rare_token():
    idf = {"common": 1.0, "rare": 8.0, "other": 1.0}
    assert S.idf_overlap({"common", "rare"}, {"rare"}, idf) > \
           S.idf_overlap({"common", "rare"}, {"common"}, idf)
