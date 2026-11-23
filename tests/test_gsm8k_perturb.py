from fractions import Fraction

import pytest

from perturb_gsm8k import Skip, build_program, check_record, perturb_item, question_numerals, try_perturb

JANET = {
    "id": 0,
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
                "muffins for her friends every day with four. She sells the remainder at the farmers' "
                "market daily for $2 per fresh duck egg. How much in dollars does she make every day?",
    "solution": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
                "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18",
}


def item(q, sol, i=1):
    return {"id": i, "question": q, "solution": sol}


def test_janet_answer_recomputed():
    rec = perturb_item(JANET, seed=0)
    new = rec["number_mapping"]["16"]
    assert list(rec["number_mapping"]) == ["16"]  # three/four (words) and $2 (constant) fixed
    assert int(rec["perturbed_answer"]) == (new - 3 - 4) * 2
    assert f"lay {new} eggs" in rec["perturbed_question"]
    assert rec["perturbed_answer"] != rec["original_answer"]
    assert 8 <= new <= 32
    assert check_record(rec) == []


@pytest.mark.parametrize("seed", range(20))
def test_constraints_hold_across_seeds(seed):
    it = item("Tom has 12,000 marbles and 37 bags. He buys 15 more bags. How many bags?",
              "He has 37 + 15 = <<37+15=52>>52 bags.\n#### 52")
    rec = perturb_item(it, seed)
    m = rec["number_mapping"]
    assert set(m) == {"37", "15"}
    assert int(rec["perturbed_answer"]) == m["37"] + m["15"]
    for k, v in m.items():
        assert v != int(k) and int(k) / 2 <= v <= 2 * int(k)
    assert len(set(m.values())) == len(m)
    assert "12,000" in rec["perturbed_question"]  # unused number untouched
    assert check_record(rec) == []


@pytest.mark.parametrize("seed", range(20))
def test_integer_division_stays_integer(seed):
    # 24 is a common constant -> fixed; 4 and 9 are perturbed; 24/new4 must stay an integer.
    it = item("A baker makes 24 rolls and packs them 4 per box. Each box sells for 9 dollars. Revenue?",
              "Boxes: 24/4=<<24/4=6>>6. Revenue: 6*9=<<6*9=54>>54.\n#### 54")
    rec = perturb_item(it, seed)
    m = rec["number_mapping"]
    assert set(m) == {"4", "9"}
    boxes = Fraction(24, m["4"])
    assert boxes.denominator == 1
    assert Fraction(rec["perturbed_answer"]) == boxes * m["9"] != 54


def test_ambiguous_literal_skipped():
    it = item("Sam has 10 apples, gives away 5, then buys 5 more. How many?",
              "10-5=<<10-5=5>>5. 5+5=<<5+5=10>>10.\n#### 10")
    rec, skip = try_perturb(it, 0)
    assert rec is None and skip.reason == "ambiguous_literal"


def test_unexplained_literal_skipped():
    it = item("Ann reads 13 pages a day for 9 days. How many pages?",
              "13*9=<<13*9=117>>117 then 117+8=<<117+8=125>>125\n#### 125")
    rec, skip = try_perturb(it, 0)
    assert rec is None and skip.reason == "unexplained_literal"


def test_final_must_match_last_step():
    it = item("Ann reads 13 pages a day for 9 days. How many?", "13*9=<<13*9=117>>117\n#### 118")
    with pytest.raises(Skip) as e:
        build_program(it["question"], it["solution"])
    assert e.value.reason == "final_not_last_step"


def test_no_perturbable_numbers():
    it = item("She has three cats and two dogs. How many pets?", "3+2=<<3+2=5>>5\n#### 5")
    assert try_perturb(it, 0)[1].reason == "no_perturbable_numbers"


def test_percent_and_decimal_numerals_are_fixed():
    nums = question_numerals("A 20% tip on $3.50 at 7:30 for the 3rd time, 45 total")
    cand = {t: c for _, _, t, _, c in nums}
    assert cand == {"20": False, "3.50": False, "7": False, "30": False, "3": False, "45": True}


def test_checker_detects_tampering():
    rec = perturb_item(JANET, 0)
    bad = dict(rec, perturbed_answer=str(int(rec["perturbed_answer"]) + 1))
    assert check_record(bad)
    bad = dict(rec, perturbed_question=rec["perturbed_question"].replace("Janet", "Jane"))
    assert check_record(bad)
