# tests/test_image_pipeline.py

import numpy as np
import pytest
import pipeline.image_pipeline_dsl as pipeline_dsl
from pipeline.image_pipeline import process_image


def test_doctests():
    # Exécute tous les doctests du module pipeline_dsl
    failed, total = pytest.doctest_module(pipeline_dsl)
    assert failed == 0, f"{failed} doctest(s) failed out of {total}"



def test_normalization():
    img = np.array([[0, 50], [100, 150]], dtype=np.uint8)
    out = process_image(img, normalize=True, crop=False, to_float32=False)
    assert out.min() == 0
    assert out.max() == 1


def test_crop():
    img = np.arange(16).reshape(4, 4)
    out = process_image(img, normalize=False, crop=True, to_float32=False)
    # Should crop to a 4x4 central region for square matrices (no-op)
    assert out.shape == (4, 4)


def test_crop_rectangular():
    img = np.zeros((10, 4))
    out = process_image(img, normalize=False, crop=True, to_float32=False)
    # Should crop to 4x4
    assert out.shape == (4, 4)


def test_to_float32():
    img = np.ones((3, 3), dtype=np.uint8)
    out = process_image(img, normalize=False, crop=False, to_float32=True)
    assert out.dtype == np.float32


def test_pipeline_all_enabled():
    img = np.arange(16).reshape(4, 4)
    out = process_image(img, normalize=True, crop=True, to_float32=True)
    assert out.dtype == np.float32
    assert out.min() == 0
    assert out.max() == 1
