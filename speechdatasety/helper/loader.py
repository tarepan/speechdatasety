"""DataLoader generator"""

from typing import Optional, TypeVar
from dataclasses import dataclass
from os import cpu_count

from omegaconf import MISSING
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

@dataclass
class ConfLoader:
    """Configuration of train/val/test data loaders.

    Args:
        batch_size_train: Number of datum in a training   batch
        batch_size_val:   Number of datum in a validation batch
        batch_size_test:  Number of datum in a test       batch
        num_workers: Number of data loader worker
        pin_memory: Whether to use data loader pin_memory
        drop_last: Whether to drop last (chipping) batch
    """
    batch_size_train: int = MISSING
    batch_size_val: int = MISSING
    batch_size_test: int = MISSING
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    drop_last: bool = False

class LoaderGenerator:
    """Generator of PyTorch data loaders.
    """
    def __init__(self, conf: ConfLoader):
        self._conf = conf

        if conf.num_workers is None:
            n_cpu = cpu_count()
            conf.num_workers = n_cpu if n_cpu is not None else 0
        self._num_workers = conf.num_workers
        self._pin_memory = conf.pin_memory if conf.pin_memory is not None else True
        self._drop_last = conf.drop_last

    def train(self, dataset: Dataset[S]) -> DataLoader[S]:
        """Generate training dataloader."""

        has_col = hasattr(dataset, "collate_fn")
        return DataLoader(
            dataset,
            batch_size=self._conf.batch_size_train,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=dataset.collate_fn if has_col else None,
            drop_last=self._drop_last,
        )

    def val(self, dataset: Dataset[T]) -> DataLoader[T]:
        """Generate validation dataloader."""

        has_col = hasattr(dataset, "collate_fn")
        return DataLoader(
            dataset,
            batch_size=self._conf.batch_size_val,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=dataset.collate_fn if has_col else None,
            drop_last=self._drop_last,
        )

    def test(self, dataset: Dataset[U]) -> DataLoader[U]:
        """Generate test dataloader."""

        has_col = hasattr(dataset, "collate_fn")
        return DataLoader(
            dataset,
            batch_size=self._conf.batch_size_test,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=dataset.collate_fn if has_col else None,
            drop_last=self._drop_last,
        )


def generate_loader(dataset: Dataset[T], conf: ConfLoader, mode: str) -> DataLoader[T]:
    """Generate the PyTorch data loader from the dataset and configs.
    Args:
        dataset - Dataset which will be wrapped by the loader
        conf - The configurations
        mode :: 'train'|'val'|'test' - Loader mode
    """

    assert mode in ("train", "val", "test"), f"`mode` should be either train/val/test, but {mode}"

    # The number of workers
    num_workers = conf.num_workers
    if num_workers is None:
        n_cpu = cpu_count()
        num_workers = n_cpu if n_cpu is not None else 0

    # Whether to pin memory
    pin_memory = conf.pin_memory if conf.pin_memory is not None else True

    # Wether use dataset-specific collate_fn or default one
    collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None

    # Mode switching
    batch_size = conf.batch_size_train if mode == "train" else conf.batch_size_val if mode == "val" else conf.batch_size_test
    shuffle    = True                  if mode == "train" else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=conf.drop_last,
    )
