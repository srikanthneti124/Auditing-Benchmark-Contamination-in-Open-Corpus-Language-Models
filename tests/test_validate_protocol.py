"""Validation protocol (validate_sample.py): overlap > 0 only, one snippet per item by one rule,
blind sheet + key, resumable snippet cache, dropped items counted. All HTTP is mocked."""
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

import overlap
import validate_sample as vs

N = 60
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega").split()


def item_text(i):
    return " ".join(f"{WORDS[(i + j) % len(WORDS)]}{i}" for j in range(24))


def fake_items(bm, *a, **k):
    return [{"id": i, "text": item_text(i)} for i in range(N)]


def ov_value(i):
    return [0.0, 0.2, 0.8][i % 3]


def build(tmp_path):
    """overlap.csv and a cache where item i's highest-count window is window (i % n_windows)."""
    rows, cache = [], {}
    for corpus, index in overlap.INDEX.items():
        for i in range(N):
            ws = overlap.windows(item_text(i), 13, 15, 10)
            o = ov_value(i)
            for j, w in enumerate(ws):
                cache[f"{index}|{w}"] = 0 if o == 0 else (100 + i if j == i % len(ws) else j % 2)
            rows.append({"corpus": corpus, "benchmark": "gsm8k", "id": i, "n_words": 24, "n_windows": len(ws),
                         "overlap": o, "max_count": max(cache[f"{index}|{w}"] for w in ws)})
    pd.DataFrame(rows).to_csv(tmp_path / "overlap.csv", index=False)
    (tmp_path / "overlap_cache.json").write_text(json.dumps(cache))
    return cache


class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p


class FakeSearch:
    def __init__(self, fail=(), no_docs=(), interrupt_after=None):
        self.fail, self.no_docs, self.interrupt_after = set(fail), set(no_docs), interrupt_after
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "timeout": timeout, **json})
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt
        q = json["query"]
        if q in self.fail:
            return Resp(500, {"message": "server error"})
        if q in self.no_docs:
            return Resp(200, {"documents": []})
        return Resp(200, {"documents": [{"spans": [["before ", None], [q, "0"], [" after", None]]}]})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vs, "load_benchmark", fake_items)
    monkeypatch.setattr(vs.time, "sleep", lambda s: None)
    return tmp_path


def export(monkeypatch, api, *extra, n=30):
    monkeypatch.setattr(vs.requests, "post", api.post)
    monkeypatch.setattr(sys, "argv", ["validate_sample.py", "export", "--n", str(n), "--retries", "3", *extra])
    vs.main()
    return vs.read_sheet("validation_sample.csv"), pd.read_csv("validation_key.csv")


def expected_window(i, corpus, cache):
    ws = overlap.windows(item_text(i), 13, 15, 10)
    index = overlap.INDEX[corpus]
    counts = [cache[f"{index}|{w}"] for w in ws]
    return ws[int(np.argmax(counts))]


def test_sheet_has_no_revealing_column_and_shows_rubric(env, monkeypatch):
    build(env)
    sheet, key = export(monkeypatch, FakeSearch())
    assert list(sheet.columns) == vs.SHEET_COLS
    leaky = [c for c in sheet.columns if re.search(r"stratum|overlap|count|max|window|^id$|flag", c)]
    assert leaky == []
    head = open("validation_sample.csv", encoding="utf-8-sig").read().splitlines()[:3]
    assert head == vs.RUBRIC
    assert "near-verbatim" in head[0] and "generic or boilerplate" in head[0] and "unsure" in head[0]
    assert (sheet.corpus_snippet.str.len() > 0).all()  # never blank
    assert set(key.stratum) == {"partial", "contaminated"}  # overlap > 0 only, no clean items
    assert (key.overlap > 0).all()
    assert list(key.sample_id) == list(sheet.sample_id)
    assert (key.stratum != key.stratum.shift()).sum() > 5  # shuffled, not in blocks
    assert (key.stratum == "partial").sum() == (key.stratum == "contaminated").sum() == 15  # equal quota


def test_identical_retrieval_rule_for_every_item(env, monkeypatch):
    cache = build(env)
    api = FakeSearch()
    sheet, key = export(monkeypatch, api)
    assert len(api.calls) == len(key)  # one search_docs call per item
    for _, r in key.iterrows():
        assert r.query_window == expected_window(int(r.id), r.corpus, cache)  # the highest-count window
        assert r.query_window_count == cache[f"{overlap.INDEX[r.corpus]}|{r.query_window}"]
    assert sorted(c["query"] for c in api.calls) == sorted(key.query_window)
    assert {(c["query_type"], c["maxnum"], c["max_disp_len"], c["timeout"], c["url"]) for c in api.calls} == {
        ("search_docs", 1, 80, 60.0, overlap.API)}
    joined = sheet.merge(key, on="sample_id")
    assert all(f"[[{w}]]" in s for w, s in zip(joined.query_window, joined.corpus_snippet))


def test_fallback_window_is_used_when_original_was_replaced(env, monkeypatch):
    cache = build(env)
    index = overlap.INDEX["pile"]
    w0 = overlap.windows(item_text(1), 13, 15, 10)[0]
    short = " ".join(w0.split()[:6])
    del cache[f"{index}|{w0}"]
    cache[f"{index}|{short}"] = 10 ** 6
    (env / "overlap_cache.json").write_text(json.dumps(cache))
    (env / "overlap_fallback.json").write_text(json.dumps({f"{index}|{w0}": short}))
    assert vs.highest_count_window(overlap.windows(item_text(1), 13, 15, 10), index, cache,
                                   {f"{index}|{w0}": short}) == (short, 10 ** 6)


@pytest.mark.parametrize("k,n", [(0, 10), (5, 10), (10, 10), (1, 1), (81, 263), (3, 200), (199, 200)])
def test_wilson_matches_reference(k, n):
    lo, hi = vs.wilson(k, n)
    rlo, rhi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(rlo, abs=1e-6) and hi == pytest.approx(rhi, abs=1e-6)


def test_wilson_known_values():
    assert vs.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert vs.wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert all(np.isnan(vs.wilson(0, 0)))


def test_resume_after_interrupt_without_duplicates(env, monkeypatch):
    build(env)
    with pytest.raises(KeyboardInterrupt):
        export(monkeypatch, FakeSearch(interrupt_after=7))
    saved = json.load(open("validation_snippets.json", encoding="utf-8"))
    assert len(saved) == 7  # progress kept by the finally-save
    api = FakeSearch()
    sheet, key = export(monkeypatch, api)
    assert len(api.calls) == len(key) - 7  # only uncached snippets were requested
    assert not key.duplicated(["corpus", "benchmark", "id"]).any() and not sheet.sample_id.duplicated().any()
    cache = json.load(open("validation_snippets.json", encoding="utf-8"))
    assert len(cache) == len(key)
    api2 = FakeSearch()
    export(monkeypatch, api2, "--overwrite")
    assert api2.calls == []  # fully cached: no HTTP at all


def test_failed_retrievals_dropped_and_counted(env, monkeypatch, capsys):
    cache = build(env)
    probe = FakeSearch()
    _, key0 = export(monkeypatch, probe, "--corpora", "pile")  # one corpus: each query window is one item
    targets = list(key0.query_window[:4])
    (env / "validation_snippets.json").unlink()
    api = FakeSearch(fail=targets[:3], no_docs=targets[3:4])
    sheet, key = export(monkeypatch, api, "--overwrite", "--corpora", "pile")
    assert len(key) == len(key0) - 4
    assert not set(targets) & set(key.query_window)
    assert (sheet.corpus_snippet.str.len() > 0).all()
    summ = json.load(open("validation_key_summary.json", encoding="utf-8"))
    assert summ["dropped_by_reason"] == {"request_failed": 3, "no_documents": 1}
    assert sum(summ["dropped"].values()) == 4
    assert sum(summ["kept"].values()) == len(key)
    log = open("validation_snippets_failed.log", encoding="utf-8").read().splitlines()
    assert len(log) == 3 and all("HTTP 500" in ln for ln in log)
    assert sum(c["query"] == targets[0] for c in api.calls) == 3  # --retries 3
    assert "dropped 4 sampled items" in capsys.readouterr().out
    # transient failures are not cached, so a later run retries them
    snip = json.load(open("validation_snippets.json", encoding="utf-8"))
    assert not any(k.endswith("|" + t) for k in snip for t in targets[:3])


def test_export_refuses_to_overwrite_annotations(env, monkeypatch):
    build(env)
    export(monkeypatch, FakeSearch())
    with pytest.raises(SystemExit, match="already exist"):
        export(monkeypatch, FakeSearch())


def fill_and_score(monkeypatch, sheet, key, labels, capsys):
    sheet = sheet.copy()
    sheet["true_contamination"] = labels
    sheet["stratum"] = "partial"  # a stray column in the sheet must be ignored
    vs.write_sheet("filled.csv", sheet)
    monkeypatch.setattr(sys, "argv", ["validate_sample.py", "score", "--filled", "filled.csv",
                                      "--key", "validation_key.csv"])
    vs.main()
    return pd.read_csv("validation_precision.csv").set_index(["subset", "stratum"]), capsys.readouterr().out


def test_score_precision_unsure_drops_and_scope(env, monkeypatch, capsys):
    build(env)
    probe = FakeSearch()
    _, key0 = export(monkeypatch, probe, "--corpora", "pile")
    targets = list(key0.query_window[:2])
    (env / "validation_snippets.json").unlink()
    sheet, key = export(monkeypatch, FakeSearch(fail=targets), "--overwrite", "--corpora", "pile")
    capsys.readouterr()
    labels = np.where(key.stratum == "contaminated", "yes", "no").astype(object)
    labels[0] = "unsure"
    res, out = fill_and_score(monkeypatch, sheet, key, labels, capsys)
    c = res.loc[("all", "contaminated")]
    p = res.loc[("all", "partial")]
    n_c = int((key.stratum == "contaminated").sum()) - int(key.stratum.iloc[0] == "contaminated")
    assert c.precision == 1.0 and c.judged == n_c
    assert p.precision == 0.0
    allf = res.loc[("all", "all_flagged")]
    assert allf.unsure_excluded == 1 and allf.dropped_before_annotation == 2
    assert (allf.wilson_lo, allf.wilson_hi) == pytest.approx(vs.wilson(int(allf.yes), int(allf.judged)))
    assert "Clean-item recall is out of scope" in out and "paraphrased contamination" in out


def test_score_requires_matching_key(env, monkeypatch):
    build(env)
    sheet, key = export(monkeypatch, FakeSearch())
    vs.write_sheet("filled.csv", sheet)
    monkeypatch.setattr(sys, "argv", ["validate_sample.py", "score", "--filled", "filled.csv"])
    with pytest.raises(SystemExit):
        vs.main()  # --key is required
    key.iloc[1:].to_csv("short_key.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["validate_sample.py", "score", "--filled", "filled.csv", "--key", "short_key.csv"])
    with pytest.raises(SystemExit, match="sample_ids differ"):
        vs.main()
