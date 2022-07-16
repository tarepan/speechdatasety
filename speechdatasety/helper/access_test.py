"""Test data access utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .access import generate_saver_loader, load_np, save_np
from ..interface.speechcorpusy import ItemId     # pyright: ignore [reportMissingTypeStubs]


def test_save_load(tmp_path: Path):
    """Test `save_np` and `load_np`."""

    ary = np.array([1, 2, 3,], dtype=np.int64)
    save_np(tmp_path, ary)

    # Simulate file closing & reopening

    ary_reloaded = load_np(tmp_path)

    assert np.array_equal(ary, ary_reloaded), f"{ary} != {ary_reloaded}"

def test_generate_saver_loader(tmp_path: Path):
    """Test `generate_saver_loader`."""

    subtype, speaker, uttr = "default", "spk_a", "num0"
    item_id = ItemId(subtype, speaker, uttr)

    THoge = NDArray[np.float32]                       # pylint: disable=invalid-name
    THoge_: THoge = np.array([1.,], dtype=np.float32) # pylint: disable=invalid-name
    TFuga = NDArray[np.float32]                       # pylint: disable=invalid-name
    TFuga_: TFuga = np.array([1.,], dtype=np.float32) # pylint: disable=invalid-name
    THogeFuga = Tuple[THoge, TFuga]                   # pylint: disable=invalid-name
    THogeFuga_: THogeFuga = (THoge_, TFuga_)          # pylint: disable=invalid-name

    hoge_fuga: THogeFuga = (np.array([1.,], dtype=np.float32), np.array([1., 2., 4.,], dtype=np.float32))

    name_hoge, name_fuga = "hoge", "fuga"
    save, load = generate_saver_loader(THogeFuga_, ["hoge", "fuga"], tmp_path)

    save(item_id, hoge_fuga)
    hoge_fuga_reloaded = load(item_id)

    # File existance
    assert (tmp_path / speaker / f"{name_hoge}s" / f"{uttr}.{name_hoge}.pt.npy").exists(), f"{name_hoge} npy not exists."
    assert (tmp_path / speaker / f"{name_fuga}s" / f"{uttr}.{name_fuga}.pt.npy").exists(), f"{name_fuga} npy not exists."

    # Value euqality
    assert np.array_equal(hoge_fuga[0], hoge_fuga_reloaded[0]), f"{hoge_fuga[0]} != {hoge_fuga_reloaded[0]}"
    assert np.array_equal(hoge_fuga[1], hoge_fuga_reloaded[1]), f"{hoge_fuga[1]} != {hoge_fuga_reloaded[1]}"
