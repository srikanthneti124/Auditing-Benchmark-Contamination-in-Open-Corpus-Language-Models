"""check_storage.py, run_gpu_stages.bat storage gating, --cache-dir, and lock-tolerant atomic saves.
No downloads, no network (Hub metadata mocked or offline), never the project's cache files."""
import json
import os
import shutil
import subprocess
import sys

import pytest

import atomicio
import check_storage
import evaluate
import gpumem
import itemloglik
import overlap
import validate_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GB = 1e9


def write_models(path, entries):
    path.write_text(json.dumps(entries))
    return str(path)


def fake_cache(tmp_path, repo, files):
    d = tmp_path / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
    d.mkdir(parents=True)
    for name, size in files.items():
        with open(d / name, "wb") as f:
            f.truncate(size)
    return str(tmp_path / "hub")


MODELS = {
    "org/small": {"corpus": "pile", "role": "primary", "dtype": "float32", "download_gb": 1.0},
    "org/big": {"corpus": "pile", "role": "primary", "dtype": "float32", "download_gb": 5.0},
    "org/mid": {"corpus": "pile", "role": "sensitivity", "dtype": "float16", "download_gb": 2.0},
    "org/ref": {"corpus": None, "role": "reference", "dtype": "float32", "download_gb": 1.5},
}


# ---------------------------------------------------------------- check_storage
def test_cumulative_pass_fail_with_margin(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECK_STORAGE_FREE_GB", "7.0")  # budget = 7 - 3 = 4 GB
    m = write_models(tmp_path / "m.json", MODELS)
    p = check_storage.plan(m, str(tmp_path / "hub"), 3.0, offline=True)
    v = {r["model"]: (r["verdict"], r["need"]) for r in p["rows"]}
    assert [r["model"] for r in p["rows"]] == ["org/small", "org/big", "org/mid", "org/ref"]  # reference last
    assert v["org/small"] == ("PASS", 1.0 * GB)
    assert v["org/big"][0] == "FAIL"  # 5 > 4 - 1
    assert v["org/mid"] == ("PASS", 2.0 * GB)  # big's size is not reserved because it was skipped
    assert v["org/ref"][0] == "FAIL"  # 1.5 > 4 - 1 - 2
    assert all(r["source"] == "models.json" for r in p["rows"])


def test_cached_bytes_are_subtracted_and_unknown_size_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECK_STORAGE_FREE_GB", "3.5")  # budget 0.5 GB
    hub = fake_cache(tmp_path, "org/small", {"model.safetensors": 2_000_000})
    m = write_models(tmp_path / "m.json", {
        "org/small": {"corpus": "pile", "role": "primary", "dtype": "float32", "download_gb": 0.002},
        "org/nosize": {"corpus": "pile", "role": "primary", "dtype": "float32"}})
    rows = {r["model"]: r for r in check_storage.plan(m, hub, 3.0, offline=True)["rows"]}
    assert rows["org/small"]["need"] == 0 and rows["org/small"]["verdict"] == "PASS"
    assert rows["org/nosize"]["verdict"] == "FAIL" and "unknown" in rows["org/nosize"]["why"]


def test_hub_metadata_used_when_reachable(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECK_STORAGE_FREE_GB", "10")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub = fake_cache(tmp_path, "org/small", {"model-00001.safetensors": 1000})
    sizes = {"org/small": {"model-00001.safetensors": 1000, "model-00002.safetensors": 3 * 10 ** 9}}
    monkeypatch.setattr(check_storage, "hub_weight_bytes", lambda repo, timeout=15: sizes.get(repo))
    m = write_models(tmp_path / "m.json", {"org/small": {"corpus": "pile", "role": "primary", "dtype": "float32",
                                                         "download_gb": 99}})
    r = check_storage.plan(m, hub, 3.0)["rows"][0]
    assert r["source"] == "hub" and r["need"] == 3 * 10 ** 9 and r["verdict"] == "PASS"  # only the missing shard


def test_plan_file_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHECK_STORAGE_FREE_GB", "7.0")
    m = write_models(tmp_path / "m.json", MODELS)
    assert check_storage.main(["--models", m, "--cache-dir", str(tmp_path / "hub"), "--offline",
                               "--plan", str(tmp_path / "plan.txt")]) == 0
    lines = (tmp_path / "plan.txt").read_text().splitlines()
    assert lines == ["org/small float32 256 8 yes PASS", "org/big float32 256 8 yes FAIL",
                     "org/mid float16 256 8 yes PASS", "org/ref float32 256 8 yes FAIL"]
    out = capsys.readouterr().out
    assert "free on its drive: 7.00 GB; safety margin 3.0 GB" in out and "2 PASS, 2 FAIL" in out


def test_never_downloads(tmp_path, monkeypatch):
    import huggingface_hub
    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, name, lambda *a, **k: pytest.fail("check_storage must not download"))
    monkeypatch.setenv("CHECK_STORAGE_FREE_GB", "7.0")
    check_storage.main(["--models", write_models(tmp_path / "m.json", MODELS), "--offline",
                        "--cache-dir", str(tmp_path / "hub")])


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe")
@pytest.mark.parametrize("free_gb,skipped", [("1000", []), ("7.0", ["org/big", "org/ref"])])
def test_gpu_stages_skip_models_without_disk_space(tmp_path, free_gb, skipped):
    for f in ("run_gpu_stages.bat", "modelcfg.py", "check_storage.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    write_models(tmp_path / "models.json", MODELS)
    env = dict(os.environ, HF_HUB_OFFLINE="1", CHECK_STORAGE_FREE_GB=free_gb, TEMP=str(tmp_path))
    p = subprocess.run(["cmd", "/c", str(tmp_path / "run_gpu_stages.bat"), "--dry-run"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    for m in MODELS:
        cmds = [ln for ln in p.stdout.splitlines() if ln.startswith(">") and m in ln]
        if m in skipped:
            assert not cmds and f"SKIPPED {m}: not enough disk space" in p.stdout
        else:
            assert len(cmds) == 2  # evaluate + itemloglik
    assert (tmp_path / "run_gpu_stages_storage_plan.txt").exists() and not (tmp_path / "storage_plan.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe")
def test_gpu_stages_stop_when_storage_check_fails(tmp_path):
    for f in ("run_gpu_stages.bat", "modelcfg.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    (tmp_path / "check_storage.py").write_text("import sys; print('boom'); sys.exit(2)\n")
    write_models(tmp_path / "models.json", MODELS)
    p = subprocess.run(["cmd", "/c", str(tmp_path / "run_gpu_stages.bat"), "--dry-run"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120, env=dict(os.environ, TEMP=str(tmp_path)))
    assert p.returncode == 1 and "check_storage.py failed" in p.stdout and "evaluate" not in p.stdout


# ---------------------------------------------------------------- --cache-dir
class Recorder:
    def __init__(self):
        self.calls = []

    def factory(self, kind):
        rec = self

        class Fake:
            @staticmethod
            def from_pretrained(name, **kw):
                rec.calls.append((kind, name, kw.get("cache_dir", "MISSING")))

                class M:
                    def to(self, dev):
                        return self

                    def eval(self):
                        return self
                return M()
        return Fake


@pytest.mark.parametrize("cache_dir", [None, "D:/somewhere/hf"])
def test_load_model_passes_cache_dir(monkeypatch, cache_dir):
    r = Recorder()
    monkeypatch.setattr(evaluate, "AutoTokenizer", r.factory("tok"))
    monkeypatch.setattr(evaluate, "AutoModelForCausalLM", r.factory("model"))
    kw = {} if cache_dir is None else {"cache_dir": cache_dir}
    evaluate.load_model("org/m", device="cpu", dtype="float32", **kw)
    assert r.calls == [("tok", "org/m", cache_dir), ("model", "org/m", cache_dir)]  # None = HF default


@pytest.mark.parametrize("module", [evaluate, itemloglik])
def test_cli_cache_dir_reaches_loader_and_weights_check(module, tmp_path, monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_check(*a, **k):
        seen["check"] = k.get("cache_dir", "MISSING")

    def fake_load(*a, **k):
        seen["load"] = k.get("cache_dir", "MISSING")
        raise Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "check_weights_fit", fake_check)
    monkeypatch.setattr(module, "load_model", fake_load)
    for argv, want in (([], None), (["--cache-dir", "X:/hf"], "X:/hf")):
        monkeypatch.setattr(sys, "argv", ["s.py", "--model", "EleutherAI/pythia-70m", "--device", "cpu", *argv])
        with pytest.raises(Stop):
            module.main()
        assert seen == {"check": want, "load": want}


def test_weights_estimate_passes_cache_dir(monkeypatch):
    got = {}

    class Cfg:
        @staticmethod
        def from_pretrained(name, **kw):
            got.update(kw)
            raise RuntimeError("stop")

    import transformers
    monkeypatch.setattr(transformers, "AutoConfig", Cfg)
    with pytest.raises(RuntimeError):
        gpumem.estimate_weight_bytes("org/m", None, cache_dir="C:/x")
    assert got["cache_dir"] == "C:/x"


# ---------------------------------------------------------------- lock-tolerant saves
def locked_replace(n_fail):
    real, state = os.replace, {"n": 0}

    def fake(src, dst):
        state["n"] += 1
        if n_fail is None or state["n"] <= n_fail:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(src, dst)
    return fake, state


def test_replace_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(atomicio.time, "sleep", lambda s: None)
    fake, state = locked_replace(3)
    monkeypatch.setattr(atomicio.os, "replace", fake)
    assert atomicio.write_json_atomic("x.json", {"a": 1}) is True
    assert state["n"] == 4 and json.load(open("x.json")) == {"a": 1} and not os.path.exists("x.json.tmp")
    assert not os.path.exists(atomicio.WARN_LOG)


def test_replace_gives_up_keeps_tmp_and_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(atomicio.time, "sleep", lambda s: None)
    (tmp_path / "x.json").write_text(json.dumps({"old": 1}))
    fake, state = locked_replace(None)
    monkeypatch.setattr(atomicio.os, "replace", fake)
    assert atomicio.write_json_atomic("x.json", {"old": 1, "new": 2}) is False
    assert state["n"] == atomicio.RETRIES
    assert json.load(open("x.json")) == {"old": 1}  # old file intact
    assert json.load(open("x.json.tmp")) == {"old": 1, "new": 2}  # newest data kept
    log = open(atomicio.WARN_LOG, encoding="utf-8").read()
    assert "could not replace x.json after 10 attempts" in log and "x.json.tmp" in log
    assert "WARNING" in capsys.readouterr().err


def test_overlap_cache_save_on_locked_file_loses_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert os.path.abspath(os.getcwd()) != os.path.abspath(ROOT)
    monkeypatch.setattr(atomicio.time, "sleep", lambda s: None)
    (tmp_path / overlap.CACHE_FILE).write_text(json.dumps({"i|a": 1}))
    c = overlap.Counter()
    c.cache["i|b"] = 7
    fake, _ = locked_replace(None)
    with monkeypatch.context() as m:
        m.setattr(atomicio.os, "replace", fake)
        c.save()  # does not raise
    assert json.load(open(overlap.CACHE_FILE)) == {"i|a": 1}
    assert json.load(open(overlap.CACHE_FILE + ".tmp")) == {"i|a": 1, "i|b": 7}
    assert "could not replace overlap_cache.json" in open(atomicio.WARN_LOG, encoding="utf-8").read()
    c2 = overlap.Counter()  # next run merges the kept .tmp
    assert c2.cache == {"i|a": 1, "i|b": 7}
    c2.save()  # lock gone: normal save, .tmp consumed
    assert json.load(open(overlap.CACHE_FILE)) == {"i|a": 1, "i|b": 7} and not os.path.exists(overlap.CACHE_FILE + ".tmp")


def test_unreadable_tmp_is_ignored_with_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / overlap.CACHE_FILE).write_text(json.dumps({"i|a": 1}))
    (tmp_path / (overlap.CACHE_FILE + ".tmp")).write_text('{"truncated": ')
    assert overlap.Counter().cache == {"i|a": 1}
    assert "ignoring unreadable" in open(atomicio.WARN_LOG, encoding="utf-8").read()
    assert (tmp_path / (overlap.CACHE_FILE + ".tmp")).exists()  # never deleted


def test_fallback_map_and_snippet_cache_use_the_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(atomicio.time, "sleep", lambda s: None)
    fake, _ = locked_replace(None)
    with monkeypatch.context() as m:
        m.setattr(atomicio.os, "replace", fake)
        assert overlap.save_json_atomic("fb.json", {"k": "short"}) is False
        f = validate_sample.SnippetFetcher("http://x", "snip.json", "failed.log", 80, 1.0, 1)
        f.cache["i|80|q"] = {"status": "ok", "snippet": "s"}
        f.save()
    assert overlap.load_json("fb.json") == {"k": "short"}  # recovered from fb.json.tmp
    f2 = validate_sample.SnippetFetcher("http://x", "snip.json", "failed.log", 80, 1.0, 1)
    assert f2.cache == {"i|80|q": {"status": "ok", "snippet": "s"}}


def test_no_bare_os_replace_left():
    for f in ("overlap.py", "validate_sample.py", "check_overlap_health.py", "analyze.py", "report.py",
              "evaluate.py", "itemloglik.py", "check_perturbed_sample.py"):
        assert "os.replace(" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), f
