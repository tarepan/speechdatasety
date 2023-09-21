"""Test data access utilities."""

from pathlib import Path
from typing import Tuple

import torch

from .access import generate_saver_loader, load_pt, save_pt
from ..interface.speechcorpusy import ItemId     # pyright: ignore [reportMissingTypeStubs]


def test_save_load(tmp_path: Path):
    """Test `save_pt` and `load_pt`."""

    path = tmp_path / "aa"
    ipt = torch.tensor([1, 2, 3,])
    save_pt(path, ipt)

    # Simulate file closing & reopening

    ipt_reloaded = load_pt(path)

    assert torch.equal(ipt, ipt_reloaded), f"{ipt} != {ipt_reloaded}"


def test_generate_saver_loader(tmp_path: Path):
    """Test `generate_saver_loader`."""

    corpus, subtype, speaker, uttr = "c1", "default", "spk_a", "num0"
    item_id = ItemId(corpus, subtype, speaker, uttr)

    THoge = torch.Tensor                     # pylint: disable=invalid-name
    THoge_: THoge = torch.Tensor()           # pylint: disable=invalid-name
    TFuga = torch.Tensor                     # pylint: disable=invalid-name
    TFuga_: TFuga = torch.Tensor()           # pylint: disable=invalid-name
    THogeFuga = Tuple[THoge, TFuga]          # pylint: disable=invalid-name
    THogeFuga_: THogeFuga = (THoge_, TFuga_) # pylint: disable=invalid-name

    hoge_fuga: THogeFuga = ( torch.tensor([1.,]), torch.tensor([1., 2., 4.,]), )

    name_hoge, name_fuga = "hoge", "fuga"
    save, load = generate_saver_loader(THogeFuga_, ["hoge", "fuga"], tmp_path)

    save(item_id, hoge_fuga)
    hoge_fuga_reloaded = load(item_id)

    # File existance
    assert (tmp_path / corpus / speaker / f"{name_hoge}s" / f"{uttr}.{name_hoge}.pt").exists(), f"{name_hoge} not exists."
    assert (tmp_path / corpus / speaker / f"{name_fuga}s" / f"{uttr}.{name_fuga}.pt").exists(), f"{name_fuga} not exists."

    # Value euqality
    assert torch.equal(hoge_fuga[0], hoge_fuga_reloaded[0]), f"{hoge_fuga[0]} != {hoge_fuga_reloaded[0]}"
    assert torch.equal(hoge_fuga[1], hoge_fuga_reloaded[1]), f"{hoge_fuga[1]} != {hoge_fuga_reloaded[1]}"
