"""Run a Python program (HumanEval completion + tests) in a separate
subprocess with a hard wall-clock timeout, inside a fresh temp directory.

THIS IS NOT A SECURITY SANDBOX. The child runs as your user with full file
system access. Network is disabled only by monkeypatching the `socket`
module in the child, which code can trivially bypass (e.g. via `_socket`,
ctypes, or spawning another process). On timeout only the direct child is
killed (TerminateProcess on Windows); grandchildren it spawned survive.
For untrusted model output run the whole pipeline inside Docker or WSL with
networking disabled (e.g. `docker run --network none ...`).

Windows-compatible: uses subprocess.run(timeout=...), no signals, no fork,
no `resource` module.

Self-check usage:
  python execution.py            # runs a passing, a failing and a looping program
"""
import argparse
import os
import subprocess
import sys
import tempfile

# Executed in the child before the program. Blocks the common network entry
# points, then exec's the program so `from __future__` imports still work.
RUNNER = r'''
import socket as _socket
def _no_network(*args, **kwargs):
    raise OSError("network access is disabled for benchmark code execution")
for _name in ("connect", "connect_ex", "bind", "sendto"):
    setattr(_socket.socket, _name, _no_network)
_socket.create_connection = _no_network
_socket.getaddrinfo = _no_network
del _socket, _name
import sys
_path = sys.argv[1]
with open(_path, encoding="utf-8") as _f:
    _src = _f.read()
exec(compile(_src, _path, "exec"), {"__name__": "__main__", "__file__": _path})
'''

# Minimal environment: Python on Windows needs SYSTEMROOT to initialise.
_KEEP_ENV = ("SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATH")


def humaneval_program(prompt, completion, test, entry_point):
    return prompt + completion + "\n\n" + test + "\n\n" + f"check({entry_point})\n"


def run_program(source, timeout=10.0):
    """Returns {"status": passed|failed|timeout, "returncode", "stderr"}."""
    with tempfile.TemporaryDirectory(prefix="ceval_", ignore_cleanup_errors=True) as tmp:
        prog = os.path.join(tmp, "program.py")
        runner = os.path.join(tmp, "runner.py")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(source)
        with open(runner, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        env = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
        env.update(TEMP=tmp, TMP=tmp, TMPDIR=tmp, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-B", runner, prog],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "returncode": None, "stderr": ""}
    return {
        "status": "passed" if p.returncode == 0 else "failed",
        "returncode": p.returncode,
        "stderr": p.stderr[-2000:],
    }


def main():
    ap = argparse.ArgumentParser(description="Self-check for the subprocess executor.")
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args()
    cases = {
        "pass": "assert 1 + 1 == 2\n",
        "fail": "assert 1 + 1 == 3\n",
        "network": "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=5)\n",
        "loop": "while True:\n    pass\n",
    }
    for name, src in cases.items():
        r = run_program(src, timeout=min(a.timeout, 3.0) if name == "loop" else a.timeout)
        print(f"{name:8s} -> {r['status']}")


if __name__ == "__main__":
    main()
