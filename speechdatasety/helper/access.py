"""Data access helpers."""

from pathlib import Path
from typing import Callable, List, Tuple, Any, TypeVar, Union

import numpy as np
from numpy.typing import NDArray, ArrayLike

from .adress import generate_path_getter # pyright: ignore [reportMissingTypeStubs]
from ..interface.speechcorpusy import ItemId     # pyright: ignore [reportMissingTypeStubs]


def save_np(path: Path, array: ArrayLike) -> None:
    """Save numpy array in the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array) # pyright: ignore [reportUnknownMemberType]

def load_np(path: Path) -> NDArray[Any]:
    """Load numpy array in the path."""
    return np.load(f"{str(path)}.npy") # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType] ; because of numpy


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
        names - Names of elements in the item
        root - Dataset root adress
    Returns:
        save_nps - Save utility
        load_nps - Load utility
    """

    get_path_funcs = [generate_path_getter(name, root) for name in names]

    def save_nps(item_id: ItemId, items: I) -> None:
        for i, item in enumerate(items):
            save_np(get_path_funcs[i](item_id), item)

    def load_nps(item_id: ItemId) -> I:
        return tuple(load_np(get_path(item_id)) for get_path in get_path_funcs) #type:ignore ; Hacking

    return save_nps, load_nps
