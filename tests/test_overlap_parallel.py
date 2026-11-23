"""Mocked-API tests for the parallel overlap.py. No network access."""
import hashlib
import json
import os
import sys
import threading

import pandas as pd
import pytest

import overlap

TEXT = [
    "Janet has 16 ducks that lay eggs every single day and she sells the remainder at the market for two dollars",
    "A train leaves the station at noon traveling west at sixty miles per hour toward a distant city on the coast",
    "Short question with exactly ten words in it right here",
    "too short to window",
    "Tom buys 3 apples and 5 pears and then gives half of the fruit to his younger sister who eats two",
]
CODE = [
    'def add(a, b):\n    """Return the sum of a and b.\n    >>> add(1, 2)\n    3\n    """\n'
    "    result = a + b\n    if result > 100:\n        return 100\n    return result\n" * 2,
    'def tiny(x):\n    """Tiny."""\n',
]
ITEMS = {
    "gsm8k": [{"id": i * 7, "text": t} for i, t in enumerate(TEXT)],
    "humaneval": [{"id": i, "text": t} for i, t in enumerate(CODE)],
}


def fake_count(index, q):
    h = int(hashlib.md5(f"{index}|{q}".encode()).hexdigest(), 16)
    return 0 if h % 3 else h % 50 + 1


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeAPI:
    """Stands in for requests.Session. `fail` maps query -> list of statuses to return first;
    `always_fail` queries get 403 forever."""

    def __init__(self, fail=None, always_fail=()):
        self.fail = {k: list(v) for k, v in (fail or {}).items()}
        self.always_fail = set(always_fail)
        self.calls = []
        self.sessions = set()
        self.lock = threading.Lock()

    def session_factory(self):
        api = self

        class Session:
            def __init__(self):
                with api.lock:
                    api.sessions.add((id(self), threading.get_ident()))

            def post(self, url, json=None, timeout=None):
                q, index = json["query"], json["index"]
                with api.lock:
                    api.calls.append(q)
                    if q in api.always_fail:
                        return Resp(403, {"message": "Forbidden"})
                    if api.fail.get(q):
                        return Resp(api.fail[q].pop(0), {"message": "error"})
                return Resp(200, {"count": fake_count(index, q)})

        return Session


def old_sequential_rows(benchmarks, corpora, cache, n=13, k=15, min_words=10,
                        code_benchmarks=("humaneval",), code_n=32, code_min_tokens=16):
    """The original sequential loop, with counts read from `cache` (None if absent)."""
    rows = []
    for bm in benchmarks:
        for corpus in corpora:
            for it in ITEMS[bm]:
                if bm in code_benchmarks:
                    ws = overlap.code_windows(it["text"], code_n, k, code_min_tokens)
                else:
                    ws = overlap.windows(it["text"], n, k, min_words)
                counts = [cache.get(f"{overlap.INDEX[corpus]}|{w}") for w in ws]
                counts = [c for c in counts if c is not None]
                rows.append({
                    "corpus": corpus, "benchmark": bm, "id": it["id"],
                    "n_words": len(it["text"].split()), "n_windows": len(counts),
                    "overlap": (sum(c > 0 for c in counts) / len(counts)) if counts else float("nan"),
                    "max_count": max(counts) if counts else float("nan"),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(overlap, "load_benchmark", lambda bm, limit=None, seed=0: ITEMS[bm])
    monkeypatch.setattr(overlap.time, "sleep", lambda s: None)
    return tmp_path


def run(monkeypatch, api, *args):
    _guard()
    monkeypatch.setattr(overlap.requests, "Session", api.session_factory())
    monkeypatch.setattr(sys, "argv", ["overlap.py", "--out", "out.csv", *args])
    overlap.main()
    return pd.read_csv("out.csv")


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _guard():
    # Never touch the real cache in the project folder.
    assert os.path.abspath(os.getcwd()) != PROJECT_DIR, "test is running in the project folder"


def write_cache(d):
    _guard()
    with open(overlap.CACHE_FILE, "w") as f:
        json.dump(d, f)


def read_cache():
    _guard()
    with open(overlap.CACHE_FILE) as f:
        return json.load(f)


ARGS = ["--benchmarks", "gsm8k", "humaneval", "--corpora", "pile", "dolma", "--k", "15", "--workers", "4"]


def test_a_rows_identical_to_old_sequential_logic(env, monkeypatch):
    api = FakeAPI()
    new = run(monkeypatch, api, *ARGS)
    old = old_sequential_rows(["gsm8k", "humaneval"], ["pile", "dolma"], read_cache())
    pd.testing.assert_frame_equal(new, old, check_dtype=False)
    assert new.n_windows.max() == 15
    assert len(api.sessions) > 1  # more than one thread-local session was used


def test_a_rows_identical_with_partially_prefilled_cache(env, monkeypatch):
    run(monkeypatch, FakeAPI(), *ARGS)
    full = read_cache()
    half = dict(list(full.items())[::2])
    write_cache(half)
    api = FakeAPI()
    new = run(monkeypatch, api, *ARGS)
    assert len(api.calls) == len(full) - len(half)  # only uncached windows were queried
    pd.testing.assert_frame_equal(new, old_sequential_rows(["gsm8k", "humaneval"], ["pile", "dolma"], full),
                                  check_dtype=False)


def test_b_cache_never_shrinks(env, monkeypatch):
    seed = {"v4_piletrain_llama|unrelated window": 5, "other_index|x": 0}
    write_cache(seed)
    run(monkeypatch, FakeAPI(), *ARGS)
    after = read_cache()
    assert all(after[k] == v for k, v in seed.items())
    assert len(after) > len(seed)
    before = len(after)
    run(monkeypatch, FakeAPI(), *ARGS)
    assert len(read_cache()) == before


def test_c_cache_parses_after_simulated_interrupt(env, monkeypatch):
    write_cache({"v4_piletrain_llama|pre-existing": 1})
    real_wait, calls = overlap.wait, {"n": 0}

    def interrupting_wait(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_wait(*a, **k)

    monkeypatch.setattr(overlap, "wait", interrupting_wait)
    with pytest.raises(KeyboardInterrupt):
        run(monkeypatch, FakeAPI(), *ARGS)
    c = read_cache()  # parses
    assert c["v4_piletrain_llama|pre-existing"] == 1
    assert not os.path.exists(overlap.CACHE_FILE + ".tmp")


def test_c_interrupted_save_leaves_previous_file_intact(env, monkeypatch):
    write_cache({"v4_piletrain_llama|pre-existing": 1})
    counter = overlap.Counter()
    counter.cache["v4_piletrain_llama|new"] = 2

    def broken_dump(obj, f):
        f.write('{"truncated": ')
        raise KeyboardInterrupt

    with monkeypatch.context() as m:  # scoped: must not undo the chdir from the env fixture
        m.setattr(overlap.json, "dump", broken_dump)
        with pytest.raises(KeyboardInterrupt):
            counter.save()
    assert read_cache() == {"v4_piletrain_llama|pre-existing": 1}


def test_d_failed_queries_logged_and_excluded_not_zero(env, monkeypatch):
    item = ITEMS["gsm8k"][0]
    ws = overlap.windows(item["text"], 13, 15, 10)
    bad = ws[0]
    api = FakeAPI(always_fail={bad})
    # --allow-partial: a permanently failing window leaves gsm8k/pile incomplete, and since the
    # completeness rule overlap.csv is only written for an incomplete pair when this flag is given.
    out = run(monkeypatch, api, "--benchmarks", "gsm8k", "--corpora", "pile", "--k", "15", "--allow-partial")
    row = out[out.id == item["id"]].iloc[0]
    assert row.n_windows == len(ws) - 1
    assert f"v4_piletrain_llama|{bad}" not in read_cache()
    assert api.calls.count(bad) == overlap.RETRIES
    log = open(overlap.FAILED_LOG, encoding="utf-8").read().splitlines()
    assert len(log) == 1
    assert "v4_piletrain_llama" in log[0] and "HTTP 403" in log[0] and json.dumps(bad) in log[0]
    good = [fake_count("v4_piletrain_llama", w) for w in ws[1:]]
    assert row.overlap == pytest.approx(sum(c > 0 for c in good) / len(good))


@pytest.mark.parametrize("status", [403, 429, 500, 503])
def test_d_retryable_status_then_success_is_cached(env, monkeypatch, status):
    w = overlap.windows(ITEMS["gsm8k"][0]["text"], 13, 15, 10)[0]
    api = FakeAPI(fail={w: [status, status]})
    run(monkeypatch, api, "--benchmarks", "gsm8k", "--corpora", "pile")
    assert read_cache()[f"v4_piletrain_llama|{w}"] == fake_count("v4_piletrain_llama", w)
    assert api.calls.count(w) == 3
    assert not os.path.exists(overlap.FAILED_LOG)


def test_d_network_errors_are_retried(env, monkeypatch):
    api = FakeAPI()
    Session = api.session_factory()
    state = {"n": 0}

    class Flaky(Session):
        def post(self, *a, **k):
            with api.lock:
                state["n"] += 1
                first = state["n"] == 1
            if first:
                raise ConnectionError("simulated reset")
            return super().post(*a, **k)

    monkeypatch.setattr(overlap.requests, "Session", Flaky)
    monkeypatch.setattr(sys, "argv", ["overlap.py", "--out", "out.csv", "--benchmarks", "gsm8k", "--corpora", "pile"])
    overlap.main()
    assert not os.path.exists(overlap.FAILED_LOG)
    assert pd.read_csv("out.csv").n_windows.sum() == sum(
        len(overlap.windows(it["text"], 13, 15, 10)) for it in ITEMS["gsm8k"])


def test_e_k1_and_humaneval_windowing(env, monkeypatch):
    api = FakeAPI()
    out = run(monkeypatch, api, "--benchmarks", "gsm8k", "humaneval", "--corpora", "pile", "--k", "1")
    windowed = out[out.n_windows > 0]
    assert (windowed.n_windows == 1).all()
    code_queries = [q for q in api.calls if "def " in q]
    assert code_queries and all("\n" in q for q in code_queries)  # newlines/indentation preserved
    assert code_queries[0].startswith("def add(a, b):\n    ")
    he = out[out.benchmark == "humaneval"].set_index("id")
    assert he.loc[1].n_windows == 0 and pd.isna(he.loc[1].overlap)  # below --code-min-tokens


def test_schema_unchanged(env, monkeypatch):
    out = run(monkeypatch, FakeAPI(), "--benchmarks", "gsm8k", "--corpora", "pile")
    assert list(out.columns) == ["corpus", "benchmark", "id", "n_words", "n_windows", "overlap", "max_count"]


def test_cache_key_format(env, monkeypatch):
    run(monkeypatch, FakeAPI(), "--benchmarks", "gsm8k", "--corpora", "pile", "dolma")
    keys = read_cache()
    assert all(k.split("|", 1)[0] in overlap.INDEX.values() for k in keys)
    assert overlap.CACHE_FILE == "overlap_cache.json"
