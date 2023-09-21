"""Data access helpers."""

from pathlib import Path
from typing import Callable, List, Tuple, Any, TypeVar, Union

import torch

from .adress import generate_path_getter     # pyright: ignore [reportMissingTypeStubs]
from ..interface.speechcorpusy import ItemId # pyright: ignore [reportMissingTypeStubs]


def save_pt(path: Path, obj: Any) -> None:
    """Save item to the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path) # pyright: ignore[reportUnknownMemberType]


def load_pt(path: Path) -> Any:
    """Load item in the path."""
    return torch.load(path) # pyright: ignore[reportUnknownMemberType]


NTuple = Union[
    Tuple[Any],
    Tuple[Any, Any],
    Tuple[Any, Any, Any],
    Tuple[Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any],
    Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any],
]

I = TypeVar("I", bound=NTuple)
def generate_saver_loader(type_obj: I, names: List[str], root: Path) -> Tuple[Callable[[ItemId, I], None], Callable[[ItemId], I]]:
    """Generate multiple item save/load utility.
    
    Args:
        type_obj - Example save/load target item for typing
        names    - Names of elements in the item
        root     - Dataset root adress
    Returns:
        _save    - Save utility
        _load    - Load utility
    """

    get_path_funcs = [generate_path_getter(name, root) for name in names]

    def _save(item_id: ItemId, items: I) -> None:
        for i, item in enumerate(items):
            save_pt(get_path_funcs[i](item_id), item)

    def _load(item_id: ItemId) -> I:
        return tuple(load_pt(get_path(item_id)) for get_path in get_path_funcs) #type:ignore ; Hacking

    return _save, _load
