import ast

import pytest

from execution import humaneval_program, run_program
from perturb_humaneval import Skip, perturb_item, rename_item

PROMPT = '''import math
from typing import List

SCALE = 3


def helper(values: List[int], factor: int = 2) -> List[int]:
    """Multiply each of values by factor."""
    out = [v * factor for v in values]  # keep comment
    return out


def solve(numbers: List[int], k: int) -> int:
    """ Return the sum of the k largest numbers, times SCALE.
    >>> solve([1, 2, 3], 2)
    15
    """
'''
SOLUTION = '''    ordered = sorted(numbers, reverse=True)
    top = helper(ordered[:k], factor=SCALE)
    total = sum(top)
    return int(math.floor(total))
'''
TEST = '''def check(candidate):
    assert candidate([1, 2, 3], 2) == 15
    assert candidate(numbers=[5, 1], k=1) == 15
    assert candidate([4, 4, 4], k=3) == 36
'''
ITEM = {"id": 7, "task_id": "Synthetic/0", "prompt": PROMPT, "canonical_solution": SOLUTION,
        "test": TEST, "entry_point": "solve"}


def names_in(src):
    return {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)} | {
        n.arg for n in ast.walk(ast.parse(src)) if isinstance(n, ast.arg)}


def test_renames_params_and_locals_only():
    rec = rename_item(ITEM, seed=0)
    m = rec["rename_map"]
    assert set(m) == {"values", "factor", "out", "v", "numbers", "k", "ordered", "top", "total"}
    new = rec["perturbed_prompt"] + rec["perturbed_canonical_solution"]
    kept = names_in(new)
    for protected in ("math", "List", "SCALE", "helper", "sorted", "sum", "int"):
        assert protected in kept
    assert not (set(m) & kept)
    assert "def solve(" in rec["perturbed_prompt"] and "def helper(" in rec["perturbed_prompt"]


def test_formatting_and_comments_preserved():
    rec = rename_item(ITEM, seed=0)
    assert "# keep comment" in rec["perturbed_prompt"]
    assert rec["perturbed_prompt"].count("\n") == PROMPT.count("\n")
    assert '""" Return the sum of the k largest numbers' in rec["perturbed_prompt"]  # docstring kept


def test_keyword_arguments_adapted():
    rec = rename_item(ITEM, seed=0)
    m = rec["rename_map"]
    assert f"factor=SCALE" not in rec["perturbed_canonical_solution"]
    assert f"{m['factor']}=SCALE" in rec["perturbed_canonical_solution"]
    assert f"candidate({m['numbers']}=[5, 1], {m['k']}=1)" in rec["perturbed_test"]


def _behaviour(prompt, sol, entry, args):
    g = {}
    exec(prompt + sol, g)
    return [g[entry](*a) for a in args]


def test_behaviour_preserved_in_process():
    rec = rename_item(ITEM, seed=0)
    args = [([1, 2, 3], 2), ([9, 8, 7, 1], 3), ([0], 1), ([5, 5], 0)]
    assert _behaviour(PROMPT, SOLUTION, "solve", args) == _behaviour(
        rec["perturbed_prompt"], rec["perturbed_canonical_solution"], "solve", args)


def test_behaviour_preserved_under_tests_in_subprocess():
    rec = perturb_item(ITEM, seed=0, timeout=20)
    r = run_program(humaneval_program(rec["perturbed_prompt"], rec["perturbed_canonical_solution"],
                                      rec["perturbed_test"], "solve"), timeout=20)
    assert r["status"] == "passed", r["stderr"]
    # original keyword test would now fail against the renamed function
    r = run_program(humaneval_program(rec["perturbed_prompt"], rec["perturbed_canonical_solution"],
                                      TEST, "solve"), timeout=20)
    assert r["status"] == "failed"


def test_deterministic_and_seed_dependent():
    assert rename_item(ITEM, 0)["rename_map"] == rename_item(ITEM, 0)["rename_map"]
    assert any(rename_item(ITEM, 0)["rename_map"] != rename_item(ITEM, s)["rename_map"] for s in range(1, 5))


def test_kwargs_unpacking_in_tests_is_skipped():
    item = dict(ITEM, test="def check(candidate):\n    assert candidate(**{'numbers': [1], 'k': 1}) == 3\n")
    with pytest.raises(Skip) as e:
        rename_item(item, 0)
    assert e.value.reason == "kwargs_unpacking_to_renamed_function"


def test_dynamic_access_is_skipped():
    item = dict(ITEM, canonical_solution="    return eval('k')\n")
    with pytest.raises(Skip):
        rename_item(item, 0)


def test_docstring_reword_option():
    rec = rename_item(ITEM, 0, reword_docstring=True)
    m = rec["rename_map"]
    assert rec["docstring_reworded"]
    assert f"the {m['k']} largest {m['numbers']}" in rec["perturbed_prompt"]


def test_timeout_is_enforced():
    r = run_program("while True:\n    pass\n", timeout=2)
    assert r["status"] == "timeout"


def test_network_blocked():
    r = run_program("import socket\nsocket.create_connection(('example.com', 80))\n", timeout=20)
    assert r["status"] == "failed" and "network access is disabled" in r["stderr"]
