"""Test processings."""

import numpy as np

from .process import s16scale, s16min, s16max, unit_to_s16pcm


def test_constants():
    """Test constant values."""

    assert s16scale == 2**(16-1)
    assert s16min == -1 * s16scale
    assert s16max == +1 * s16scale - 1


def test_unit_to_s16pcm():
    """Test `unit_to_s16pcm` function."""

    i_too_sml, gt_too_sml = np.array([-1.1], dtype=np.float32), np.array([-32768], dtype=np.int16)
    i_min,     gt_min     = np.array([-1. ], dtype=np.float32), np.array([-32768], dtype=np.int16)
    i_sml,     gt_sml     = np.array([-0.5], dtype=np.float32), np.array([-16384], dtype=np.int16)
    i_zero,    gt_zero    = np.array([ 0. ], dtype=np.float32), np.array([     0], dtype=np.int16)
    i_big,     gt_big     = np.array([+0.5], dtype=np.float32), np.array([+16384], dtype=np.int16)
    i_max,     gt_max     = np.array([+1. ], dtype=np.float32), np.array([+32767], dtype=np.int16)
    i_too_big, gt_too_big = np.array([+1.1], dtype=np.float32), np.array([+32767], dtype=np.int16)

    # Type
    assert unit_to_s16pcm(i_too_sml).dtype == np.int16

    # Value
    assert unit_to_s16pcm(i_too_sml) == gt_too_sml
    assert unit_to_s16pcm(i_min)     == gt_min
    assert unit_to_s16pcm(i_sml)     == gt_sml
    assert unit_to_s16pcm(i_zero)    == gt_zero
    assert unit_to_s16pcm(i_big)     == gt_big
    assert unit_to_s16pcm(i_max)     == gt_max
    assert unit_to_s16pcm(i_too_big) == gt_too_big

    # Clipping
    assert unit_to_s16pcm(i_too_sml)[0] == -32768
    assert unit_to_s16pcm(i_too_big)[0] == +32767
