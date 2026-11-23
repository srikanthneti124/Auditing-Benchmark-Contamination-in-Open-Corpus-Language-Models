"""Structural check of the LaTeX tables written by report.py.

pdflatex is not installed on this machine, so compilation is UNVERIFIED. This
checks what can be checked without it:
  1. every \\begin{env} has a matching \\end{env}, properly nested;
  2. every tabular row (a line ending in \\\\) has as many cells as the column
     specification has columns;
  3. no unescaped special character (% # _ ^ ~ &, or an unbalanced $ or brace)
     in table cells or the caption. Escapes produced by report.tex_escape and
     $...$ math are allowed.

Usage:
  python texcheck.py paper_outputs/*.tex      (exit 1 if any problem)
"""
import glob
import re
import sys

ENV = re.compile(r"\\(begin|end)\{([^}]*)\}")
ESCAPES = [r"\textbackslash{}", r"\textasciitilde{}", r"\^{}", r"\&", r"\%", r"\$", r"\#", r"\_", r"\{", r"\}"]
SPECIAL = set("%#_^~&")


def split_cells(row):
    return re.split(r"(?<!\\)&", row)


def text_problems(s, where):
    """Unescaped special characters in one cell / caption."""
    for e in ESCAPES:
        s = s.replace(e, "")
    if s.count("$") % 2:
        return [f"{where}: unbalanced $ in {s!r}"]
    s = re.sub(r"\$[^$]*\$", "", s)
    out = [f"{where}: unescaped {c!r} in {s!r}" for c in sorted(SPECIAL) if c in s]
    if s.count("{") != s.count("}"):
        out.append(f"{where}: unbalanced braces in {s!r}")
    return out


def n_columns(spec):
    spec = re.sub(r"[pmb]\{[^}]*\}", "l", spec)
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    return len(re.findall(r"[lcr]", spec))


def check_tex(text):
    problems, stack = [], []
    for m in ENV.finditer(text):
        kind, name = m.groups()
        if kind == "begin":
            stack.append(name)
        elif not stack:
            problems.append(f"\\end{{{name}}} without \\begin")
        elif stack[-1] != name:
            problems.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
    if stack:
        problems.append(f"unclosed environment(s): {stack}")
    cap = re.search(r"\\caption\{(.*)\}\s*$", text, re.M)
    if cap:
        problems += text_problems(cap.group(1), "caption")
    for m in re.finditer(r"\\begin\{tabular\}\{((?:[^{}]|\{[^}]*\})*)\}(.*?)\\end\{tabular\}", text, re.S):
        ncol = n_columns(m.group(1))
        for i, ln in enumerate(m.group(2).splitlines()):
            ln = ln.strip()
            if not ln.endswith("\\\\"):
                continue
            cells = split_cells(ln[:-2])
            if len(cells) != ncol:
                problems.append(f"row {i}: {len(cells)} cells, column spec has {ncol}: {ln[:80]!r}")
            for j, c in enumerate(cells):
                problems += text_problems(c.strip(), f"row {i} cell {j}")
    return problems


def main(paths):
    files = [f for p in paths for f in (glob.glob(p) or [p])]
    bad = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            probs = check_tex(fh.read())
        print(f"{'OK  ' if not probs else 'FAIL'} {f}")
        for p in probs:
            print(f"     {p}")
        bad += bool(probs)
    print(f"{len(files) - bad}/{len(files)} files pass the structural check (compilation UNVERIFIED: no pdflatex)")
    return 1 if bad or not files else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
