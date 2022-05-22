"""DataLoader generator"""

from typing import Optional
from dataclasses import dataclass
from os import cpu_count

from omegaconf import MISSING
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


@dataclass
class ConfLoader:
    """Configuration of train/val/test data loaders.

    Args:
        batch_size_train: Number of datum in a training   batch
        batch_size_val:   Number of datum in a validation batch
        batch_size_test:  Number of datum in a test       batch
        num_workers: Number of data loader worker
        pin_memory: Whether to use data loader pin_memory
    """
    batch_size_train: int = MISSING
    batch_size_val: int = MISSING
    batch_size_test: int = MISSING
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None

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

    def train(self, dataset: Dataset) -> DataLoader:
        """Generate training dataloader."""

        has_col = hasattr(dataset, "collate_fn")
        return DataLoader(
            dataset,
            batch_size=self._conf.batch_size_train,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=dataset.collate_fn if has_col else None,
        )

    def val(self, dataset: Dataset) -> DataLoader:
        """Generate validation dataloader."""

        has_col = hasattr(dataset, "collate_fn")
        return DataLoader(
            dataset,
            batch_size=self._conf.batch_size_val,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=dataset.collate_fn if has_col else None,
        )

    def test(self, dataset: Dataset) -> DataLoader:
        """Generate test dataloader."""

        has_col = hasattr(dataset, "collate_fn")
        return DataLoader(
            dataset,
            batch_size=self._conf.batch_size_test,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=dataset.collate_fn if has_col else None,
        )
