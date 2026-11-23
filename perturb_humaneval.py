"""HumanEval perturbation by alpha-renaming (ast-based, formatting-preserving).

Renamed: function parameters and local variables (including lambda and
comprehension variables) of every function in prompt + canonical solution.
Never renamed: builtins, imported names, module-level names (so the
entry-point and helper function names are kept), names declared
global/nonlocal, `except ... as` names, match-pattern names, `_`, dunders,
anything inside a class body. The renaming is one global injective map
old -> new onto names that do not occur anywhere in prompt, solution or
tests, which preserves Python's binding structure. Edits are applied at the
exact source positions reported by `ast`, so comments, quotes and layout are
unchanged.

Keyword arguments: in prompt/solution and in the tests, keywords in calls to
functions defined in the prompt (or to `candidate`) are renamed with the
callee's parameters. Items are skipped when this cannot be guaranteed:
`**kwargs` passed to such a call, keyword calls to unknown local callables,
introspection of `candidate` in the tests, eval/exec/locals()/globals()/vars().

--reword-docstring additionally replaces whole-word mentions of a function's
renamed parameters inside its docstring. No free-form paraphrasing is done.

Each perturbed item is kept only if (1) the original canonical solution passes
the original tests, (2) the renamed canonical solution passes the (keyword-
adapted) tests, run in a subprocess, and (3) the renamed AST equals the
original AST with the rename map applied.

Usage:
  python perturb_humaneval.py --seed 0 --out perturbed_humaneval.jsonl
  python perturb_humaneval.py --seed 0 --reword-docstring --timeout 10
"""
import argparse
import ast
import builtins
import json
import keyword
import random
import re
import sys
from collections import Counter

from execution import humaneval_program, run_program

POOL = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett",
        "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
        "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_DYNAMIC = {"eval", "exec", "locals", "globals", "vars", "__import__"}


class Skip(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _params(fn):
    a = fn.args
    return [x for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x is not None]


def _scan(tree):
    """Collect candidate local names, never-rename names and function params."""
    module_names, protected, local_nodes = set(), set(), []

    def visit(node, in_func, in_class):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                protected.add((al.asname or al.name).split(".")[0])
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            protected.update(node.names)
        if isinstance(node, ast.ExceptHandler) and node.name:
            protected.add(node.name)
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            protected.add(node.name)
        if isinstance(node, ast.MatchMapping) and node.rest:
            protected.add(node.rest)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            (module_names if not in_func and not in_class else protected).add(node.name)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and not in_func:
            module_names.add(node.id)
        if isinstance(node, ast.Name) and node.id in _DYNAMIC:
            raise Skip("dynamic_name_access", node.id)
        if in_func and not in_class:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                local_nodes.append(node.id)
            if isinstance(node, ast.arg):
                local_nodes.append(node.arg)
        for child in ast.iter_child_nodes(node):
            visit(child, in_func or isinstance(node, _FUNCS), in_class or isinstance(node, ast.ClassDef))

    visit(tree, False, False)
    never = protected | module_names | set(dir(builtins)) | set(keyword.kwlist)
    order = []
    for n in local_nodes:
        if n not in never and n not in order and n != "_" and not n.startswith("__"):
            order.append(n)
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, []).append([p.arg for p in _params(node)])
    return order, never, funcs


def _line_offsets(src_bytes):
    offs, pos = [0], 0
    for line in src_bytes.split(b"\n")[:-1]:
        pos += len(line) + 1
        offs.append(pos)
    return offs


class _Editor:
    def __init__(self, src):
        self.b = src.encode("utf-8")
        self.offs = _line_offsets(self.b)
        self.edits = {}

    def rename(self, lineno, col, old, new):
        start = self.offs[lineno - 1] + col
        if self.b[start:start + len(old.encode())] != old.encode():
            raise Skip("position_mismatch", old)
        self.edits[start] = (start + len(old.encode()), new.encode())

    def replace_span(self, start, end, new_bytes):
        self.edits[start] = (end, new_bytes)

    def apply(self):
        out, last = [], 0
        for start in sorted(self.edits):
            end, new = self.edits[start]
            if start < last:
                raise Skip("overlapping_edits")
            out += [self.b[last:start], new]
            last = end
        return (b"".join(out) + self.b[last:]).decode("utf-8")


def _rename_calls(tree, ed, mapping, funcs, never):
    """Rename keywords in calls to functions defined in the source."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else None
        params = funcs.get(name) if name else None
        kws = [k for k in node.keywords if k.arg in mapping or k.arg is None]
        if not kws:
            continue
        if params is None:
            if name and name not in never:  # call to an unknown local callable
                if any(k.arg in mapping for k in kws):
                    raise Skip("keyword_call_to_unknown_callable", name)
            continue
        if len(params) > 1:
            raise Skip("ambiguous_function_name", name)
        for k in kws:
            if k.arg is None:
                if any(p in mapping for p in params[0]):
                    raise Skip("kwargs_unpacking_to_renamed_function", name)
            elif k.arg in params[0]:
                ed.rename(k.lineno, k.col_offset, k.arg, mapping[k.arg])


def _docstring_spans(tree, src):
    """[(byte_start, byte_end, function params)] for function docstrings."""
    b = src.encode("utf-8")
    offs = _line_offsets(b)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                c = first.value
                out.append((offs[c.lineno - 1] + c.col_offset, offs[c.end_lineno - 1] + c.end_col_offset,
                            [p.arg for p in _params(node)]))
    return out


class _Apply(ast.NodeTransformer):
    """Apply the rename map to an AST (for the structural equivalence check)."""

    def __init__(self, mapping, funcs, names=True):
        self.m, self.funcs, self.names = mapping, funcs, names

    def visit_Name(self, n):
        if self.names:
            n.id = self.m.get(n.id, n.id)
        return n

    def visit_arg(self, n):
        if self.names:
            n.arg = self.m.get(n.arg, n.arg)
        self.generic_visit(n)
        return n

    def visit_Call(self, n):
        self.generic_visit(n)
        name = n.func.id if isinstance(n.func, ast.Name) else None
        if name is not None:
            params = self.funcs.get(name)
            if params:
                for k in n.keywords:
                    if k.arg in params[0]:
                        k.arg = self.m.get(k.arg, k.arg)
        return n


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                first.value.value = ""
    return tree


def rename_item(item, seed=0, reword_docstring=False):
    """Pure-source perturbation (no execution). Returns a record or raises Skip."""
    prompt, sol, test, entry = item["prompt"], item["canonical_solution"], item["test"], item["entry_point"]
    if not prompt.endswith("\n"):
        raise Skip("prompt_not_line_aligned")
    full = prompt + sol
    try:
        tree, ttree = ast.parse(full), ast.parse(test)
    except SyntaxError as e:
        raise Skip("syntax_error", str(e))
    order, never, funcs = _scan(tree)
    if not order:
        raise Skip("nothing_to_rename")
    if entry not in funcs:
        raise Skip("entry_point_not_found", entry)
    for node in ast.walk(ttree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "candidate":
            raise Skip("test_introspects_candidate", node.attr)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(a.name.split(".")[0] == "inspect" for a in node.names):
            raise Skip("test_uses_inspect")

    taken = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", full + "\n" + test)) | never
    rng = random.Random(f"humaneval:{seed}:{item['id']}")
    pool = POOL[:]
    rng.shuffle(pool)
    fresh = [p for p in pool if p not in taken]
    k = 2
    while len(fresh) < len(order):
        fresh += [f"{p}{k}" for p in pool if f"{p}{k}" not in taken]
        k += 1
    mapping = dict(zip(order, fresh))

    ed = _Editor(full)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            ed.rename(node.lineno, node.col_offset, node.id, mapping[node.id])
        elif isinstance(node, ast.arg) and node.arg in mapping:
            ed.rename(node.lineno, node.col_offset, node.arg, mapping[node.arg])
    _rename_calls(tree, ed, mapping, funcs, never)
    reworded = False
    if reword_docstring:
        for start, end, params in _docstring_spans(tree, full):
            if start >= len(prompt.encode("utf-8")):
                continue
            seg = ed.b[start:end].decode("utf-8")
            new = seg
            for p in params:
                if p in mapping:
                    new = re.sub(rf"\b{re.escape(p)}\b", mapping[p], new)
            if new != seg:
                ed.replace_span(start, end, new.encode("utf-8"))
                reworded = True
    new_full = ed.apply()

    test_funcs = dict(funcs)
    test_funcs["candidate"] = funcs[entry]
    ted = _Editor(test)
    _rename_calls(ttree, ted, mapping, test_funcs, set(dir(builtins)) | {"check"})
    new_test = ted.apply()

    n_prompt_lines = prompt.count("\n")
    lines = new_full.split("\n")
    new_prompt = "\n".join(lines[:n_prompt_lines]) + "\n"
    new_sol = "\n".join(lines[n_prompt_lines:])

    want = _Apply(mapping, funcs).visit(ast.parse(full))
    got = ast.parse(new_full)
    if reword_docstring:
        want, got = _strip_docstrings(want), _strip_docstrings(got)
    if ast.dump(want) != ast.dump(got):
        raise Skip("ast_mismatch")
    if ast.dump(_Apply(mapping, test_funcs, names=False).visit(ast.parse(test))) != ast.dump(ast.parse(new_test)):
        raise Skip("test_ast_mismatch")
    if new_prompt == prompt:
        raise Skip("prompt_unchanged")
    return {
        "id": item["id"],
        "task_id": item["task_id"],
        "entry_point": entry,
        "original_prompt": prompt,
        "perturbed_prompt": new_prompt,
        "perturbed_canonical_solution": new_sol,
        "perturbed_test": new_test,
        "rename_map": mapping,
        "docstring_reworded": reworded,
    }


def perturb_item(item, seed=0, reword_docstring=False, timeout=10.0):
    rec = rename_item(item, seed, reword_docstring)
    orig = run_program(humaneval_program(item["prompt"], item["canonical_solution"], item["test"], item["entry_point"]), timeout)
    if orig["status"] != "passed":
        raise Skip("original_solution_fails", orig["status"])
    new = run_program(humaneval_program(rec["perturbed_prompt"], rec["perturbed_canonical_solution"],
                                        rec["perturbed_test"], item["entry_point"]), timeout)
    if new["status"] != "passed":
        raise Skip("renamed_solution_fails", new["status"] + " " + new["stderr"][-300:])
    return rec


def try_perturb(item, seed=0, reword_docstring=False, timeout=10.0):
    try:
        return perturb_item(item, seed, reword_docstring, timeout), None
    except Skip as s:
        return None, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="default: all 164 items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reword-docstring", action="store_true")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per test execution")
    ap.add_argument("--out", default="perturbed_humaneval.jsonl")
    a = ap.parse_args()

    from common import load_benchmark

    from manifest import write_manifest

    write_manifest("perturb_humaneval.py", a, datasets=["openai/openai_humaneval"])
    items = load_benchmark("humaneval", a.limit, a.seed)
    recs, skips = [], Counter()
    for it in items:
        rec, skip = try_perturb(it, a.seed, a.reword_docstring, a.timeout)
        if rec:
            recs.append(rec)
        else:
            skips[skip.reason] += 1
            print(f"  skip {it['task_id']}: {skip.reason} {skip.detail[:120]}", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"survived {len(recs)}/{len(items)} items ({len(recs) / len(items):.1%}) -> {a.out}")
    for reason, c in skips.most_common():
        print(f"  {reason:36s} {c}")


if __name__ == "__main__":
    main()
