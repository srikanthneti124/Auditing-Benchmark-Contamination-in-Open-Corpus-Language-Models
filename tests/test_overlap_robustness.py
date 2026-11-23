"""Overlap robustness: completeness accounting, refusal to write incomplete overlap.csv,
--allow-partial + "complete" column, analyze.py exclusion, HumanEval long-query fallback,
run_overlap.bat restart loop, preflight_api.py. All HTTP is mocked or a local fake server."""
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading

import pandas as pd
import pytest

import analyze
import overlap
import preflight_api
from test_overlap_parallel import ITEMS, FakeAPI, Resp, fake_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.abspath(ROOT)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(overlap, "load_benchmark", lambda bm, limit=None, seed=0: ITEMS[bm])
    monkeypatch.setattr(overlap.time, "sleep", lambda s: None)
    assert os.path.abspath(os.getcwd()) != PROJECT
    return tmp_path


def run(monkeypatch, session, *args):
    monkeypatch.setattr(overlap.requests, "Session", session)
    monkeypatch.setattr(sys, "argv", ["overlap.py", "--out", "out.csv", *args])
    overlap.main()


def ntok(q):
    return len(re.findall(r"\S+", q))


class RejectLong(FakeAPI):
    """HTTP 400 for queries with more than `max_tokens` whitespace tokens (a long-query rejection)."""

    def __init__(self, max_tokens, **kw):
        super().__init__(**kw)
        self.max_tokens = max_tokens

    def session_factory(self):
        api = self
        base = super().session_factory()

        class Session(base):
            def post(self, url, json=None, timeout=None):
                if ntok(json["query"]) > api.max_tokens:
                    with api.lock:
                        api.calls.append(json["query"])
                    return Resp(400, {"error": "query too long"})
                return super().post(url, json=json, timeout=timeout)

        return Session


ARGS = ["--benchmarks", "gsm8k", "humaneval", "--corpora", "pile", "dolma", "--workers", "2"]


def test_accounting_counts_done_failed_remaining(env, monkeypatch, capsys):
    bad = overlap.windows(ITEMS["gsm8k"][0]["text"], 13, 15, 10)[0]
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, FakeAPI(always_fail={bad}).session_factory(), *ARGS)
    assert e.value.code == overlap.EXIT_INCOMPLETE
    out = capsys.readouterr().out
    assert "completeness" in out and "NOT written" in out and "gsm8k/pile: 1 remaining, 1 failed" in out
    assert not os.path.exists("out.csv")
    # --status: same numbers from the cache alone, and no HTTP at all
    def no_http(*a, **k):
        raise AssertionError("--status must not query the API")
    monkeypatch.setattr(overlap.requests, "Session", no_http)
    monkeypatch.setattr(sys, "argv", ["overlap.py", *ARGS, "--status"])
    overlap.main()
    out = capsys.readouterr().out
    expected = {bm: len({w for it in ITEMS[bm] for w in (
        overlap.code_windows(it["text"], 32, 15, 16) if bm == "humaneval" else overlap.windows(it["text"], 13, 15, 10))})
        for bm in ITEMS}
    row = [ln.split() for ln in out.splitlines() if ln.split()[:2] == ["gsm8k", "pile"]][0]
    assert row[2:6] == [str(expected["gsm8k"]), str(expected["gsm8k"] - 1), "1", "1"]
    he = [ln.split() for ln in out.splitlines() if ln.split()[:2] == ["humaneval", "dolma"]][0]
    assert he[2:6] == [str(expected["humaneval"]), str(expected["humaneval"]), "0", "0"]


def test_accounting_function_directly():
    cache = {"i1|a": 1, "i1|b": 0, "i1|short": 2}
    plan = {("bm", "pile"): ["a", "b", "c", "d", "long"]}
    with pytest.MonkeyPatch.context() as m:
        m.setitem(overlap.INDEX, "pile", "i1")
        acct = overlap.accounting(plan, cache, {"i1|long": "short"}, {"i1|c"}).iloc[0]
    assert (acct.expected, acct.done, acct.failed, acct.remaining, acct.fallbacks) == (5, 3, 1, 2, 1)


def test_complete_run_writes_csv_without_extra_column(env, monkeypatch):
    run(monkeypatch, FakeAPI().session_factory(), *ARGS)
    out = pd.read_csv("out.csv")
    assert list(out.columns) == ["corpus", "benchmark", "id", "n_words", "n_windows", "overlap", "max_count"]
    assert not os.path.exists(overlap.NOTICE_FILE)


def test_allow_partial_appends_complete_column_and_notice(env, monkeypatch):
    bad = overlap.windows(ITEMS["gsm8k"][0]["text"], 13, 15, 10)[0]
    run(monkeypatch, FakeAPI(always_fail={bad}).session_factory(), *ARGS, "--allow-partial")
    out = pd.read_csv("out.csv")
    assert list(out.columns)[-1] == "complete" and list(out.columns)[:-1] == [
        "corpus", "benchmark", "id", "n_words", "n_windows", "overlap", "max_count"]
    flags = out.groupby(["benchmark", "corpus"]).complete.unique().map(lambda v: list(v))
    # FakeAPI fails that window text on both indices, so both gsm8k pairs are incomplete
    assert flags[("gsm8k", "pile")] == [False] and flags[("gsm8k", "dolma")] == [False]
    assert flags[("humaneval", "pile")] == [True] and flags[("humaneval", "dolma")] == [True]
    notice = open(overlap.NOTICE_FILE, encoding="utf-8").read()
    assert "gsm8k" in notice and "humaneval" not in notice


def test_analyze_excludes_incomplete_pairs(capsys):
    ov = pd.DataFrame({"corpus": ["pile", "pile", "dolma"], "benchmark": ["gsm8k", "mmlu", "gsm8k"], "id": [1, 1, 1],
                       "overlap": [0.0, 0.5, 1.0], "complete": [False, True, True]})
    kept = analyze.exclude_incomplete(ov)
    assert list(zip(kept.benchmark, kept.corpus)) == [("mmlu", "pile"), ("gsm8k", "dolma")]
    assert "gsm8k/pile (1 items" in capsys.readouterr().out
    assert len(analyze.exclude_incomplete(ov, include=True)) == 3
    as_text = ov.assign(complete=ov.complete.map({True: "True", False: "False"}))  # as read back from CSV
    assert len(analyze.exclude_incomplete(as_text)) == 2
    no_col = ov.drop(columns="complete")
    assert analyze.exclude_incomplete(no_col) is no_col  # missing column means complete


def test_humaneval_long_query_falls_back_to_shorter_windows(env, monkeypatch, capsys):
    api = RejectLong(max_tokens=20)
    run(monkeypatch, api.session_factory(), "--benchmarks", "humaneval", "--corpora", "pile")
    out = pd.read_csv("out.csv")
    fb = json.load(open(overlap.FALLBACK_FILE, encoding="utf-8"))
    long_ws = overlap.code_windows(ITEMS["humaneval"][0]["text"], 32, 15, 16)
    assert len(fb) == len(set(long_ws)) and all(ntok(v) == 16 for v in fb.values())
    for w in long_ws:  # fallback = the first half of the window, as a source slice
        assert fb[f"v4_piletrain_llama|{w}"] == overlap.shorten(w, 16) and w.startswith(overlap.shorten(w, 16))
    row = out[out.id == 0].iloc[0]
    assert row.n_windows == len(long_ws)  # every window counted, through its fallback
    short_counts = [fake_count("v4_piletrain_llama", overlap.shorten(w, 16)) for w in long_ws]
    assert row.overlap == pytest.approx(sum(c > 0 for c in short_counts) / len(short_counts))
    log = open(overlap.FALLBACK_LOG, encoding="utf-8").read().splitlines()
    assert len(log) == len(set(long_ws)) and all("32 -> 16 tokens" in ln for ln in log)
    summ = pd.read_csv(overlap.FALLBACK_SUMMARY).iloc[0]
    assert summ.fallbacks == len(set(long_ws)) and summ.fallback_rate == pytest.approx(len(set(long_ws)) / summ.expected)
    assert "fallback rate" in capsys.readouterr().out
    # a rerun reuses the fallback map: nothing is queried again
    api2 = RejectLong(max_tokens=20)
    run(monkeypatch, api2.session_factory(), "--benchmarks", "humaneval", "--corpora", "pile")
    assert api2.calls == []
    pd.testing.assert_frame_equal(pd.read_csv("out.csv"), out)


def test_fallback_that_is_still_rejected_leaves_pair_incomplete(env, monkeypatch):
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, RejectLong(max_tokens=10).session_factory(), "--benchmarks", "humaneval", "--corpora", "pile")
    assert e.value.code == overlap.EXIT_INCOMPLETE
    assert not os.path.exists(overlap.FALLBACK_FILE)
    log = open(overlap.FAILED_LOG, encoding="utf-8").read()
    assert "rejected, not retried" in log


def test_short_rejected_window_falls_back_to_min_tokens(env, monkeypatch):
    # a 29-token item gives one whole-text window; half (14) is below --code-min-tokens (16): use 16
    text = "def f(a):\n    " + " ".join(f"x{i} = a + {i}" for i in range(5)) + "\n    return a\n"
    items = {"humaneval": [{"id": 0, "text": text}]}
    monkeypatch.setattr(overlap, "load_benchmark", lambda bm, limit=None, seed=0: items[bm])
    n = len(re.findall(r"\S+", text))
    assert 16 < n < 32
    run(monkeypatch, RejectLong(max_tokens=20).session_factory(), "--benchmarks", "humaneval", "--corpora", "pile")
    fb = json.load(open(overlap.FALLBACK_FILE, encoding="utf-8"))
    (orig, short), = fb.items()
    assert ntok(short) == 16 and orig == f"v4_piletrain_llama|{text.strip()}"
    assert f"{n} -> 16 tokens" in open(overlap.FALLBACK_LOG, encoding="utf-8").read()
    assert pd.read_csv("out.csv").n_windows.iloc[0] == 1


def test_rejected_query_is_not_retried(env, monkeypatch):
    api = RejectLong(max_tokens=20)
    run(monkeypatch, api.session_factory(), "--benchmarks", "humaneval", "--corpora", "pile")
    w = overlap.code_windows(ITEMS["humaneval"][0]["text"], 32, 15, 16)[0]
    assert api.calls.count(w) == 1


# ---------------------------------------------------------------- run_overlap.bat restart loop
FAKE_OVERLAP = r'''import os, sys
n = int(open("attempts.txt").read()) if os.path.exists("attempts.txt") else 0
open("attempts.txt", "w").write(str(n + 1))
open("argv.txt", "a").write(" ".join(sys.argv[1:]) + "\n")
print(f"fake overlap attempt {n + 1}")
sys.exit(0 if n + 1 >= int(os.environ["FAKE_SUCCEED_AT"]) else 3)
'''


def run_bat(tmp_path, succeed_at, max_restarts):
    shutil.copy(os.path.join(ROOT, "run_overlap.bat"), tmp_path / "run_overlap.bat")
    (tmp_path / "overlap.py").write_text(FAKE_OVERLAP)
    (tmp_path / "overlap_run.log").write_text("previous log line\n")
    env = dict(os.environ, FAKE_SUCCEED_AT=str(succeed_at), OVERLAP_MAX_RESTARTS=str(max_restarts),
               OVERLAP_RESTART_SLEEP="0")
    p = subprocess.run(["cmd", "/c", str(tmp_path / "run_overlap.bat")], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    return p, (tmp_path / "overlap_run.log").read_text()


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe")
def test_restart_loop_reruns_until_success(tmp_path):
    p, log = run_bat(tmp_path, succeed_at=3, max_restarts=20)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "attempts.txt").read_text() == "3"
    assert log.startswith("previous log line")  # appended, never truncated
    assert [ln for ln in log.splitlines() if ln.startswith("fake overlap attempt")] == [
        "fake overlap attempt 1", "fake overlap attempt 2", "fake overlap attempt 3"]
    assert "attempt 2 exited with code 3" in log and "attempt 3 exited with code 0" in log
    argv = (tmp_path / "argv.txt").read_text().splitlines()
    assert len(argv) == 3 and all("--workers 2" in a and "humaneval" in a for a in argv)


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe")
def test_restart_loop_gives_up_after_max_restarts(tmp_path):
    p, log = run_bat(tmp_path, succeed_at=99, max_restarts=2)
    assert p.returncode == 1
    assert (tmp_path / "attempts.txt").read_text() == "3"  # first attempt + 2 restarts
    assert "still failing after 2 restarts" in p.stdout


def test_bat_header_explains_how_to_stop():
    head = open(os.path.join(ROOT, "run_overlap.bat"), encoding="utf-8").read()
    assert "HOW TO STOP" in head and "Ctrl+C" in head and "--workers 2" in head and ">> overlap_run.log" in head


# ---------------------------------------------------------------- preflight_api.py with a local fake server
class FakeInfinigram(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        status, out = 200, {}
        if self.mode == "forbidden":
            status, out = 403, {"message": "Forbidden"}
        elif body["query_type"] == "count":
            out = {"count": 0 if self.mode == "zero" else 57}
        elif body["query_type"] == "search_docs":
            out = {"documents": [] if self.mode == "nodocs" else [{"spans": [["x", None]]}]}
        data = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    srv = http.server.HTTPServer(("127.0.0.1", 0), FakeInfinigram)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


@pytest.mark.parametrize("mode,code", [("ok", 0), ("zero", 1), ("forbidden", 1), ("nodocs", 1)])
def test_preflight_api(server, capsys, monkeypatch, mode, code):
    monkeypatch.setattr(FakeInfinigram, "mode", mode)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    assert preflight_api.main(["--api-url", url, "--timeout", "10"]) == code
    out = capsys.readouterr().out
    assert out.count("[count]") == 2 and "[search_docs]" in out and "latency=" in out and "status=" in out
    assert ("PREFLIGHT OK" in out) == (code == 0)


def test_preflight_unreachable_server_fails(capsys):
    assert preflight_api.main(["--api-url", "http://127.0.0.1:9/", "--timeout", "2"]) == 1
    assert "PREFLIGHT FAILED" in capsys.readouterr().out
