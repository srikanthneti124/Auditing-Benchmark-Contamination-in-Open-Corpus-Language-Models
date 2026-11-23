"""--skip-existing and batched log-likelihood, using the locally cached
EleutherAI/pythia-70m on CPU with the Hugging Face hub set to offline."""
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "EleutherAI/pythia-70m"


def run(script, *args, cwd):
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, script), *args], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return p.stdout


def test_evaluate_skip_existing_never_duplicates(tmp_path):
    common = ["--model", MODEL, "--benchmarks", "mmlu", "--limit", "3", "--device", "cpu", "--out", "r.csv"]
    run("evaluate.py", *common, cwd=tmp_path)
    first = pd.read_csv(tmp_path / "r.csv")
    out = run("evaluate.py", *common, "--skip-existing", cwd=tmp_path)
    assert "nothing new to evaluate" in out
    assert pd.read_csv(tmp_path / "r.csv").equals(first)
    # remove one row and resume: exactly that row comes back
    first.iloc[1:].to_csv(tmp_path / "r.csv", index=False)
    run("evaluate.py", *common, "--skip-existing", cwd=tmp_path)
    again = pd.read_csv(tmp_path / "r.csv")
    assert not again.duplicated(["model", "benchmark", "id", "variant"]).any()
    assert len(again) == len(first)
    key = ["benchmark", "id", "variant"]
    assert again.sort_values(key).reset_index(drop=True).equals(first.sort_values(key).reset_index(drop=True))


def test_evaluate_default_still_appends(tmp_path):
    common = ["--model", MODEL, "--benchmarks", "mmlu", "--limit", "2", "--device", "cpu", "--out", "r.csv"]
    run("evaluate.py", *common, cwd=tmp_path)
    n = len(pd.read_csv(tmp_path / "r.csv"))
    run("evaluate.py", *common, cwd=tmp_path)  # without the flag: unchanged append behaviour
    assert len(pd.read_csv(tmp_path / "r.csv")) == 2 * n


def test_itemloglik_skip_existing(tmp_path):
    common = ["--model", MODEL, "--benchmarks", "mmlu", "--limit", "3", "--device", "cpu", "--out", "ll.csv"]
    run("itemloglik.py", *common, cwd=tmp_path)
    first = pd.read_csv(tmp_path / "ll.csv")
    assert "nothing new to score" in run("itemloglik.py", *common, "--skip-existing", cwd=tmp_path)
    assert pd.read_csv(tmp_path / "ll.csv").equals(first)


def test_batched_logprobs_match_unbatched():
    sys.path.insert(0, ROOT)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import itemloglik
    from evaluate import load_model

    tok, model = load_model(MODEL, device="cpu")
    texts = ["Short.", "A somewhat longer sentence about ducks and eggs at the market.",
             "def f(x):\n    return x + 1\n", "x", "Another text " * 30]
    batched, bs = itemloglik.batch_logprobs(model, tok, texts, 2048, batch_size=3)
    assert bs == 3
    for t, b in zip(texts, batched):
        single = itemloglik.token_logprobs(model, tok, t, 2048)
        assert len(single) == len(b)
        # CPU fp32 matmul rounding depends on batch shape: an UNPADDED batch of 5 different texts
        # already differs from single-sequence scoring by up to 2.6e-3 nats, so this bound tests
        # for padding leakage (which would shift values by whole nats), not bitwise equality.
        assert b == pytest.approx(single, abs=5e-3)
    one, _ = itemloglik.batch_logprobs(model, tok, texts, 2048, batch_size=1)
    assert one == [itemloglik.token_logprobs(model, tok, t, 2048) for t in texts]  # exact at batch size 1
