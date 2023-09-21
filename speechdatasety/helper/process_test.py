"""Test processings."""

import numpy as np
import torch

from .process import s16scale, s16min, s16max, unit_to_s16pcm, match_length, clip_segment


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


def test_match_length_no_min():
    """Test `match_length` without minimal length constraint."""

    ipt: list[tuple[torch.Tensor, int]] = [
        #             |        unit#0         |        unit#1         |  tail 
        (torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, ]), 1),
        (torch.tensor([ 0,      2,      4,      6,      8,     10,     12,     ]), 2),
        (torch.tensor([ 0,          3,          6,          9,         12,     ]), 3),
    ]
    gt = [
         torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,]),
         torch.tensor([ 0,      2,      4,      6,      8,     10,    ]),
         torch.tensor([ 0,          3,          6,          9,        ]),
    ]

    opt = match_length(ipt, 1)

    for gt_i, opt_i in zip(gt, opt):
        assert torch.equal(gt_i, opt_i)


def test_match_length_min_length():
    """Test `match_length` without minimal length constraint."""

    ipt: list[tuple[torch.Tensor, int]] = [
        #             |        unit#0         |  tail 
        (torch.tensor([ 0,  1,  2,  3,  4,  5, 12, 13, ]), 1),
        (torch.tensor([ 0,      2,      4,     12,     ]), 2),
        (torch.tensor([ 0,          3,         12,     ]), 3),
    ]
    gt = [
        #             |        unit#0         |       unit#0 x2       |
         torch.tensor([ 0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  4,  5,]),
         torch.tensor([ 0,      2,      4,      0,      2,      4,    ]),
         torch.tensor([ 0,          3,          0,          3,        ]),
    ]


    opt = match_length(ipt, 7)

    for gt_i, opt_i in zip(gt, opt):
        assert torch.equal(gt_i, opt_i)


def test_clip_segment():
    """Test `clip_segment`."""

    ipt: list[tuple[torch.Tensor, int]] = [
        #             |        unit#0         |        unit#1         |  tail 
        (torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, ]), 1),
        (torch.tensor([ 0,      2,      4,      6,      8,     10,     12,     ]), 2),
        (torch.tensor([ 0,          3,          6,          9,         12,     ]), 3),
    ]
    gt = [
         torch.tensor([                         6,  7,  8,  9, 10, 11,]),
         torch.tensor([                         6,      8,     10,    ]),
         torch.tensor([                         6,          9,        ]),
    ]

    opt = clip_segment(ipt, 6, 6)

    for gt_i, opt_i in zip(gt, opt):
        assert torch.equal(gt_i, opt_i)
