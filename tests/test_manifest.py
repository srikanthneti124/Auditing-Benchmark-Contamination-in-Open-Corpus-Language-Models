"""run_manifest.jsonl: fields, input hashes, and that every data-producing script writes it."""
import hashlib
import json
import os
import re
import subprocess
import sys

import manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SCRIPTS = ["overlap.py", "perturb_gsm8k.py", "perturb_humaneval.py", "evaluate.py", "itemloglik.py", "analyze.py",
                "report.py", "validate_sample.py", "check_perturbed_sample.py"]


def test_record_fields_and_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.csv").write_bytes(b"a,b\n1,2\n")
    args = {"seed": 3, "limit": 20, "out": "x.csv", "flag": True, "obj": object()}
    rec = manifest.write_manifest("s.py", args, inputs=["in.csv", "missing.csv", None], model="m", dtype="float32",
                                  revision="step1", device="cpu")
    line = json.loads((tmp_path / "run_manifest.jsonl").read_text().splitlines()[-1])
    assert line == json.loads(json.dumps(rec, default=str))
    for k in ("timestamp", "script", "argv", "args", "seed", "limit", "versions", "device", "dtype", "model",
              "revision", "inputs"):
        assert k in line
    assert (line["seed"], line["limit"], line["dtype"], line["revision"]) == (3, 20, "float32", "step1")
    assert "obj" not in line["args"] and line["args"]["flag"] is True
    assert line["inputs"] == {"in.csv": hashlib.sha256(b"a,b\n1,2\n").hexdigest(), "missing.csv": None}
    assert set(line["versions"]) >= {"python", "torch", "transformers", "datasets"}
    assert line["versions"]["torch"] and line["versions"]["transformers"]


def test_appends_and_env_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest.write_manifest("a.py")
    manifest.write_manifest("b.py")
    assert [json.loads(x)["script"] for x in (tmp_path / "run_manifest.jsonl").read_text().splitlines()] == ["a.py", "b.py"]
    monkeypatch.setenv("RUN_MANIFEST", str(tmp_path / "other.jsonl"))
    manifest.write_manifest("c.py")
    assert json.loads((tmp_path / "other.jsonl").read_text())["script"] == "c.py"


def test_every_data_script_writes_the_manifest():
    for f in DATA_SCRIPTS:
        src = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert re.search(r"write_manifest\(\s*\"" + re.escape(f), src), f


def test_perturb_script_writes_manifest_line(tmp_path):
    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "perturb_gsm8k.py"), "--limit", "3", "--seed", "1",
                        "--out", "p.jsonl"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    rec = json.loads((tmp_path / "run_manifest.jsonl").read_text().splitlines()[-1])
    assert rec["script"] == "perturb_gsm8k.py" and rec["seed"] == 1 and rec["limit"] == 3
    assert rec["argv"] == ["--limit", "3", "--seed", "1", "--out", "p.jsonl"]
