import os
import pytest
import numpy as np
import sys

sys.path.append('../../')

# Random seed to ensure that tests are repeatable
RANDOM_SEED = 23
np.random.seed(RANDOM_SEED)

def test_norm_disc():
    from super_tomo_py.data_handeling.tools import normalise_discritise_data

    img = np.random.randint(0, 255, size=(100, 100, 1))
    mask = np.zeros((100, 100, 1))
    sq_lower = np.random.randint(0, 50)
    sq_upper = sq_lower + np.random.randint(0, 50)
    mask[sq_lower:sq_upper, sq_lower:sq_upper, 0] = 12

    img_norm, mask_norm = normalise_discritise_data(img, mask)

    assert img_norm.shape == img.shape
    assert mask_norm.shape == mask.shape
    assert np.max(img_norm) <= 1.
    assert np.min(img_norm) >= 0.
    assert np.max(mask_norm) <= 1.
    assert np.min(mask_norm) >= 0.

    img = np.random.randint(0, 255, size=(100, 100, 1))
    mask = np.zeros((100, 100, 1))
    sq_lower = np.random.randint(0, 50)
    sq_upper = sq_lower + np.random.randint(0, 50)
    mask[sq_lower:sq_upper, sq_lower:sq_upper, 0] = 12
    sq_lower = np.random.randint(0, 50)
    sq_upper = sq_lower + np.random.randint(0, 50)
    mask[sq_lower:sq_upper, sq_lower:sq_upper, 0] = 26

    img_norm, mask_norm = normalise_discritise_data(img, mask, flag_multi_class=True)

    assert img_norm.shape == img.shape
    assert mask_norm.shape == mask.shape
    assert np.max(img_norm) <= 1.
    assert np.min(img_norm) >= 0.
    assert np.max(mask_norm) <= len(np.unique(mask))
    assert np.min(mask_norm) >= 0.
