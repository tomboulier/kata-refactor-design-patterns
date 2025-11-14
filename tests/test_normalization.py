# test_normalization.py

import numpy as np
import pytest
from strategy.normalization import normalize


def test_minmax_normalization():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        normalize(x, "minmax"),
        np.array([0.0, 0.5, 1.0])
    )


def test_minmax_constant():
    x = np.array([5.0, 5.0, 5.0])
    assert np.allclose(
        normalize(x, "minmax"),
        np.zeros_like(x)
    )


def test_zscore_normalization():
    x = np.array([1.0, 2.0, 3.0])
    expected = (x - np.mean(x)) / np.std(x)
    assert np.allclose(normalize(x, "zscore"), expected)


def test_zscore_constant():
    x = np.array([4.0, 4.0, 4.0])
    assert np.allclose(
        normalize(x, "zscore"),
        np.zeros_like(x)
    )


def test_robust_normalization():
    x = np.array([1.0, 2.0, 100.0])
    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    expected = (x - median) / iqr
    assert np.allclose(normalize(x, "robust"), expected)


def test_robust_constant():
    x = np.array([7.0, 7.0, 7.0])
    assert np.allclose(
        normalize(x, "robust"),
        np.zeros_like(x)
    )


def test_unknown_method():
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        normalize(x, "unknown")
