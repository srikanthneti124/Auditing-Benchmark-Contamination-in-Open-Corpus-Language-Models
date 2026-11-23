import numpy as np
import pandas as pd

from analyze import assign_stratum


def test_boundaries_default_threshold():
    ov = [0.0, 1e-9, 0.2, 0.4999, 0.5, 0.75, 1.0]
    assert list(assign_stratum(ov, 0.5)) == [
        "clean", "partial", "partial", "partial", "contaminated", "contaminated", "contaminated"]


def test_custom_threshold():
    assert list(assign_stratum([0, 0.3, 0.31], 0.3)) == ["clean", "contaminated", "contaminated"]


def test_threshold_one_means_only_full_overlap_is_contaminated():
    assert list(assign_stratum([0.0, 0.99, 1.0], 1.0)) == ["clean", "partial", "contaminated"]


def test_accepts_pandas_series():
    s = pd.Series([0.0, 0.1, 0.9])
    assert list(assign_stratum(s, 0.5)) == ["clean", "partial", "contaminated"]


def test_nan_is_not_clean():
    # analyze.py drops NaN overlap rows before assigning strata; NaN must never count as clean.
    assert assign_stratum([np.nan], 0.5)[0] != "clean"


def test_matches_original_inline_expression():
    rng = np.random.default_rng(0)
    ov = np.where(rng.random(1000) < 0.3, 0.0, rng.random(1000))
    orig = np.where(ov == 0, "clean", np.where(ov >= 0.5, "contaminated", "partial"))
    assert (assign_stratum(ov, 0.5) == orig).all()
