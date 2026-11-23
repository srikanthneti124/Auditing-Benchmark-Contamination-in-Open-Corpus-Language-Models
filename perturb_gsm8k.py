"""GSM8K number perturbation with programmatic answer recomputation.

The reference solution's <<expr=result>> calculator annotations are turned
into a small re-executable program. Every numeric literal in an annotation is
resolved to exactly one of
  Q[v]  a number v that appears in the question (perturbed),
  S[j]  the result of an earlier annotation j,
  F(c)  a constant that is not perturbed (number words such as "three",
        non-integer / ordinal / percentage / time numerals in the question,
        and common unit constants such as 7, 12, 60, 100).
Items are skipped (with a recorded reason) if any literal is unexplained or
matches more than one of these sources, if the annotations do not
re-evaluate to their stated results, or if the final step does not equal the
#### answer.

Perturbation: each perturbable question value v is replaced by a different
integer in [ceil(v/2), 2v] (one consistent new value per original value, all
new values distinct). An assignment is accepted only if every step that was an
integer stays an integer, non-negative steps stay non-negative (positive stay
positive), and the final answer changes. Up to --max-tries assignments are
drawn per item from an RNG seeded by (seed, id).

Usage:
  python perturb_gsm8k.py --limit 1000 --seed 0 --out perturbed_gsm8k.jsonl
  python perturb_gsm8k.py --validate --out perturbed_gsm8k.jsonl --n-validate 100 --seed 0

Writes <out> (one JSON record per perturbed item) and <out stem>_skipped.csv
(id, reason, detail).
"""
import argparse
import ast
import csv
import json
import math
import operator
import os
import random
import re
import sys
from collections import Counter
from fractions import Fraction

ANNOT = re.compile(r"<<(.*?)>>")
Q_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
COMMON_CONSTANTS = {Fraction(x) for x in (0, 1, 2, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000)} | {
    Fraction(1, 2), Fraction(1, 4), Fraction(1, 10), Fraction(1, 100)}
_W = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "million": 1000000, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "dozen": 12, "pair": 2, "couple": 2,
    "twice": 2, "double": 2, "triple": 3, "quadruple": 4,
}
WORD_VALUES = {w: {Fraction(v)} for w, v in _W.items()}
WORD_VALUES["half"] = {Fraction(2), Fraction(1, 2)}
WORD_VALUES["third"] |= {Fraction(1, 3)}
WORD_VALUES["quarter"] = {Fraction(4), Fraction(1, 4)}
WORD_RE = re.compile(r"\b(" + "|".join(sorted(WORD_VALUES, key=len, reverse=True)) + r")s?\b", re.I)


class Skip(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------- question
def question_numerals(question):
    """[(start, end, text, value: Fraction, perturbable_candidate: bool)]."""
    out = []
    for m in Q_NUM.finditer(question):
        s, e = m.span()
        intpart, frac = m.group(1), m.group(2)
        value = Fraction(intpart.replace(",", "") + (frac or ""))
        nxt, prv = question[e:e + 1], question[s - 1:s] if s else ""
        fixed = bool(frac) or nxt.isalpha() or nxt in "%:/" or prv in ":/"
        out.append((s, e, m.group(0), value, not fixed))
    return out


def word_values(question):
    vals = set()
    for m in WORD_RE.finditer(question):
        vals |= WORD_VALUES[m.group(1).lower()]
    return vals


# ---------------------------------------------------------------- program
_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
_FUNCS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
          "//": operator.floordiv, "%": operator.mod}


def _tree(node, src):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return ("lit", Fraction(ast.get_source_segment(src, node)))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return ("bin", _OPS[type(node.op)], _tree(node.left, src), _tree(node.right, src))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return ("neg", _tree(node.operand, src))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _tree(node.operand, src)
    raise Skip("unparseable_expression", src)


def evaluate(t, Q, S):
    kind = t[0]
    if kind in ("lit", "c"):
        return t[1]
    if kind == "q":
        return Q[t[1]]
    if kind == "s":
        return S[t[1]]
    if kind == "neg":
        return -evaluate(t[1], Q, S)
    op, a, b = t[1], evaluate(t[2], Q, S), evaluate(t[3], Q, S)
    if op == "**":
        if b.denominator != 1 or abs(b) > 10 or abs(a) > 10 ** 6:
            raise Skip("unsupported_power")
        return a ** int(b)
    return _FUNCS[op](a, b)


def template(t):
    kind = t[0]
    if kind == "q":
        return f"Q['{fmt(t[1])}']"
    if kind == "s":
        return f"S[{t[1]}]"
    if kind in ("c", "lit"):
        return f"F('{t[1]}')"
    if kind == "neg":
        return f"(-{template(t[1])})"
    return f"({template(t[2])} {t[1]} {template(t[3])})"


def fmt(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else str(float(v))


def _literals(t):
    if t[0] == "lit":
        yield t[1]
    elif t[0] == "neg":
        yield from _literals(t[1])
    elif t[0] == "bin":
        yield from _literals(t[2])
        yield from _literals(t[3])


def parse_steps(solution):
    body, sep, final = solution.rpartition("####")
    if not sep:
        raise Skip("no_final_answer")
    try:
        final = Fraction(final.strip().replace(",", ""))
    except ValueError:
        raise Skip("unparseable_final_answer", final.strip())
    steps = []
    for m in ANNOT.finditer(body):
        expr, eq, res = m.group(1).rpartition("=")
        if not eq:
            raise Skip("annotation_without_equals", m.group(1))
        expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr).strip()
        res = res.strip().replace(",", "")
        try:
            tree = _tree(ast.parse(expr, mode="eval").body, expr)
            stated = Fraction(res)
        except (SyntaxError, ValueError, ZeroDivisionError):
            raise Skip("unparseable_expression", m.group(1))
        steps.append((tree, stated))
    if not steps:
        raise Skip("no_annotations")
    return steps, final


def resolve(t, P, F, results):
    """Replace literals by q/s/c refs; raise Skip if unexplained/ambiguous."""
    if t[0] == "lit":
        x = t[1]
        cands = [("q", x)] if x in P else []
        cands += [("s", j) for j, r in enumerate(results) if r == x]
        if x in F:
            cands.append(("c", x))
        if not cands:
            raise Skip("unexplained_literal", fmt(x))
        if len(cands) > 1:
            raise Skip("ambiguous_literal", fmt(x))
        return cands[0]
    if t[0] == "neg":
        return ("neg", resolve(t[1], P, F, results))
    return ("bin", t[1], resolve(t[2], P, F, results), resolve(t[3], P, F, results))


def build_program(question, solution):
    """Returns (program steps, perturbable values P, numerals, final value)."""
    steps, final = parse_steps(solution)
    nums = question_numerals(question)
    fixed = word_values(question) | COMMON_CONSTANTS | {v for *_, v, cand in nums if not cand}
    used = {x for tree, _ in steps for x in _literals(tree)}
    P = {v for *_, v, cand in nums if cand and v not in fixed and v in used and v > 0}
    if not P:
        raise Skip("no_perturbable_numbers")
    F = fixed | {v for *_, v, _ in nums if v not in P}
    program, results = [], []
    for tree, stated in steps:
        r = resolve(tree, P, F, results)
        try:
            val = evaluate(r, {v: v for v in P}, results)
        except ZeroDivisionError:
            raise Skip("division_by_zero")
        if abs(val - stated) > Fraction(1, 10 ** 6) * max(1, abs(stated)):
            raise Skip("annotation_mismatch", f"{template(r)} = {fmt(val)} != {fmt(stated)}")
        program.append(r)
        results.append(stated)
    if results[-1] != final:
        raise Skip("final_not_last_step", f"last={fmt(results[-1])} final={fmt(final)}")
    if final.denominator != 1:
        raise Skip("non_integer_answer")
    return program, P, nums, final


def run_program(program, Q):
    S = []
    for t in program:
        S.append(evaluate(t, Q, S))
    return S


def perturb_item(item, seed=0, max_tries=200):
    """Returns a perturbation record or raises Skip."""
    program, P, nums, final = build_program(item["question"], item["solution"])
    orig = run_program(program, {v: v for v in P})
    others = {v for *_, v, _ in nums if v not in P}
    rng = random.Random(f"gsm8k:{seed}:{item['id']}")
    pvals = sorted(P)
    for _ in range(max_tries):
        new, taken = {}, set(others)
        for v in pvals:
            lo, hi = max(1, math.ceil(v / 2)), int(2 * v)
            opts = [x for x in range(lo, hi + 1) if x != v and Fraction(x) not in taken]
            if not opts:
                break
            new[v] = Fraction(rng.choice(opts))
            taken.add(new[v])
        if len(new) != len(pvals):
            continue
        try:
            res = run_program(program, new)
        except (ZeroDivisionError, Skip):
            continue
        ok = all(
            (o.denominator != 1 or r.denominator == 1) and (o < 0 or r >= 0) and (o <= 0 or r > 0)
            for o, r in zip(orig, res)
        )
        if ok and res[-1] != orig[-1]:
            return {
                "id": item["id"],
                "original_question": item["question"],
                "perturbed_question": rewrite_question(item["question"], nums, new),
                "original_answer": fmt(final),
                "perturbed_answer": fmt(res[-1]),
                "number_mapping": {fmt(v): int(n) for v, n in new.items()},
                "program": [template(t) for t in program],
            }
    raise Skip("no_valid_assignment", f"tries={max_tries}")


def try_perturb(item, seed=0, max_tries=200):
    try:
        return perturb_item(item, seed, max_tries), None
    except Skip as s:
        return None, s


def rewrite_question(question, nums, new):
    out, last = [], 0
    for s, e, text, v, cand in nums:
        if cand and v in new:
            n = int(new[v])
            out += [question[last:s], f"{n:,}" if "," in text else str(n)]
            last = e
    return "".join(out) + question[last:]


# ---------------------------------------------------------------- checker
def check_record(rec):
    """Independent re-check: eval() the stored program over Fractions, and
    confirm the question text differs only at mapped numerals."""
    errors = []
    mapping = {k: Fraction(v) for k, v in rec["number_mapping"].items()}
    for label, Q, gold in (("original", {k: Fraction(k) for k in mapping}, rec["original_answer"]),
                           ("perturbed", mapping, rec["perturbed_answer"])):
        S = []
        try:
            for expr in rec["program"]:
                S.append(eval(expr, {"__builtins__": {}}, {"Q": Q, "S": S, "F": Fraction}))
            if S[-1] != Fraction(gold):
                errors.append(f"{label}: program gives {S[-1]}, record says {gold}")
        except Exception as e:  # noqa: BLE001 - report any checker failure
            errors.append(f"{label}: {type(e).__name__}: {e}")
    a, b = question_numerals(rec["original_question"]), question_numerals(rec["perturbed_question"])
    if len(a) != len(b):
        errors.append("numeral count differs between original and perturbed question")
    else:
        for (_, _, ta, va, ca), (_, _, tb, vb, _) in zip(a, b):
            want = mapping.get(fmt(va), va) if ca else va
            if vb != want:
                errors.append(f"numeral {ta!r} -> {tb!r}, expected {fmt(want)}")
    if re.sub(Q_NUM, "#", rec["original_question"]) != re.sub(Q_NUM, "#", rec["perturbed_question"]):
        errors.append("non-numeral text changed")
    return errors


def validate(path, n, seed):
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found; run without --validate first")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if not recs:
        sys.exit(f"error: {path} is empty")
    sample = random.Random(seed).sample(recs, min(n, len(recs)))
    bad = [(r["id"], e) for r in sample if (e := check_record(r))]
    print(f"validated {len(sample)} random items: {len(sample) - len(bad)} agree "
          f"({(len(sample) - len(bad)) / len(sample):.1%})")
    for i, e in bad:
        print(f"  id={i}: {'; '.join(e)}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--out", default="perturbed_gsm8k.jsonl")
    ap.add_argument("--validate", action="store_true", help="re-check a random sample of <out>")
    ap.add_argument("--n-validate", type=int, default=100)
    a = ap.parse_args()
    if a.validate:
        sys.exit(validate(a.out, a.n_validate, a.seed))

    from common import load_benchmark

    from manifest import write_manifest

    write_manifest("perturb_gsm8k.py", a, datasets=["openai/gsm8k"])
    items = load_benchmark("gsm8k", a.limit, a.seed)
    recs, skips = [], []
    for it in items:
        rec, skip = try_perturb(it, a.seed, a.max_tries)
        if rec:
            recs.append(rec)
        else:
            skips.append({"id": it["id"], "reason": skip.reason, "detail": skip.detail})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    skip_path = os.path.splitext(a.out)[0] + "_skipped.csv"
    with open(skip_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "reason", "detail"])
        w.writeheader()
        w.writerows(skips)
    print(f"perturbed {len(recs)}/{len(items)} items ({len(recs) / len(items):.1%}) -> {a.out}")
    print(f"skipped {len(skips)} -> {skip_path}")
    for reason, c in Counter(s["reason"] for s in skips).most_common():
        print(f"  {reason:28s} {c}")


if __name__ == "__main__":
    main()
