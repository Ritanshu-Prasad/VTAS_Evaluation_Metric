import numpy as np
import pytest

from harness.stats import bootstrap_ci, mean, paired_bootstrap, ratio_of_sums


def test_constant_data_has_degenerate_interval():
    ci = bootstrap_ci(np.full(50, 0.3))
    assert ci.point == ci.low == ci.high == pytest.approx(0.3)
    assert ci.level == 0.95


def test_interval_contains_point_and_is_seeded():
    x = np.random.default_rng(1).random(200)
    a, b = bootstrap_ci(x, seed=7), bootstrap_ci(x, seed=7)
    assert a == b
    assert a.low < a.point < a.high
    # 95% CI of a mean of 200 U(0,1) values: half-width ~ 1.96 * 0.289 / sqrt(200) = 0.04
    assert 0.025 < (a.high - a.low) / 2 < 0.055


def test_ratio_of_sums():
    x = np.array([[1, 4], [0, 0], [2, 2]])
    assert ratio_of_sums(x) == pytest.approx(3 / 6)
    assert ratio_of_sums(np.zeros((3, 2))) == 0.0
    ci = bootstrap_ci(x, ratio_of_sums, n_resamples=100)
    assert ci.point == pytest.approx(0.5)


def test_paired_identical_systems():
    x = np.random.default_rng(2).random(100)
    r = paired_bootstrap(x, x)
    assert r.diff.point == r.diff.low == r.diff.high == 0.0
    assert r.p_value == 1.0


def test_paired_detects_clear_difference():
    rng = np.random.default_rng(3)
    a = rng.random(300)
    r = paired_bootstrap(a + 0.1, a + rng.normal(0, 0.01, 300))
    assert r.diff.point == pytest.approx(0.1, abs=0.01)
    assert r.diff.low > 0
    assert r.p_value < 0.01
    assert r.as_dict()["diff"]["point"] == r.diff.point


def test_paired_requires_same_rows():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        bootstrap_ci([], mean)
