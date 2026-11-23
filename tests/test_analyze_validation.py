import json

import pandas as pd
import pytest

import analyze


def results(models=("EleutherAI/pythia-70m", "allenai/OLMo-1B-hf"), ids=range(5)):
    return pd.DataFrame([{"model": m, "benchmark": "mmlu", "id": i, "variant": v, "correct": i % 2}
                         for m in models for i in ids for v in ("raw", "perturbed")])


def overlap(corpora=("pile", "dolma"), ids=range(5)):
    return pd.DataFrame([{"corpus": c, "benchmark": "mmlu", "id": i, "n_words": 20, "n_windows": 8,
                          "overlap": i / 5, "max_count": i} for c in corpora for i in ids])


@pytest.fixture(autouse=True)
def clean_map():
    analyze.MODEL_CORPUS.clear()
    yield
    analyze.MODEL_CORPUS.clear()


def test_valid_inputs_pass():
    analyze.validate_inputs(results(), overlap())


def test_duplicate_rows_fail():
    r = results()
    r = pd.concat([r, r.iloc[[0]]])
    with pytest.raises(SystemExit, match="share a \\(model, benchmark, id, variant\\) key"):
        analyze.validate_inputs(r, overlap())


def test_id_sets_differing_across_models_fail():
    r = results()
    r = r[~((r.model == "allenai/OLMo-1B-hf") & (r.id == 3))]
    with pytest.raises(SystemExit, match="id sets differ across models for mmlu/raw.*missing \\[3\\]"):
        analyze.validate_inputs(r, overlap())


def test_missing_corpus_fails():
    with pytest.raises(SystemExit, match="lacks corpus \\['dolma'\\] needed by \\['allenai/OLMo-1B-hf'\\]"):
        analyze.validate_inputs(results(), overlap(corpora=("pile",)))


def test_reference_model_needs_no_corpus():
    r = results(models=("EleutherAI/pythia-70m", "Qwen/Qwen2.5-1.5B"))
    analyze.validate_inputs(r, overlap(corpora=("pile",)), ref_model="Qwen/Qwen2.5-1.5B")


def test_stratum_counts_flag_small_strata():
    raw = pd.DataFrame({"model": ["m"] * 40, "benchmark": ["b"] * 40,
                        "stratum": ["clean"] * 35 + ["contaminated"] * 5})
    n = analyze.stratum_counts(raw).iloc[0]
    assert (n.clean, n.partial, n.contaminated) == (35, 0, 5)
    assert n.flag == "partial n=0<30, contaminated n=5<30"


def test_corpus_map_overrides_and_falls_back(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"allenai/OLMo-1B-0724-hf": "dolma", "my-org/custom": "pile",
                             "EleutherAI/pythia-70m": "dolma", "reference": "Qwen/Qwen2.5-1.5B"}))
    analyze.MODEL_CORPUS.update(analyze.load_corpus_map(p))
    assert analyze.corpus_of("my-org/custom") == "pile"
    assert analyze.corpus_of("EleutherAI/pythia-70m") == "dolma"  # explicit map wins
    assert analyze.corpus_of("EleutherAI/pythia-70m@step1000") == "dolma"  # revision suffix
    assert analyze.corpus_of("EleutherAI/pythia-1b") == "pile"  # fallback
    assert analyze.corpus_of("Qwen/Qwen2.5-1.5B") is None  # "reference" is not a model entry
    assert analyze.corpus_of("reference") is None


def test_corpus_map_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"m": 3}))
    with pytest.raises(SystemExit):
        analyze.load_corpus_map(p)


def test_no_map_keeps_original_rule():
    assert analyze.corpus_of("EleutherAI/pythia-1.4b") == "pile"
    assert analyze.corpus_of("allenai/OLMo-7B-hf") == "dolma"
    assert analyze.corpus_of("Qwen/Qwen2.5-1.5B") is None
